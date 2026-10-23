"""Paper-informed batch fingerprint updating for an unchanged AP vocabulary.

This module owns the temporary reference+batch graph, the two update MLPs,
the cosine-softplus GNN batch update, the Eq. (10) AE update, and a separate immutable
fingerprint result. Update-coordinate truth is deliberately never read.
"""

from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from ae import AE, EncoderFeatureArtifact, load_ae_checkpoint
from graph_build import GraphBuildConfig, GraphState, build_graph_state
from rssi_pairs import (
    NegativeSamplePair,
    RSSI_SIMILARITY_SPACE,
    mine_rssi_pairs,
    normalized_rssi_cosine,
)
from gufu_artifacts import (
    atomic_torch_save,
    atomic_write_json,
    canonical_json_digest,
    load_torch_state,
    read_json_text,
    sha256_file,
)
from gufu_data import (
    NormalizationMode,
    PreparedDataset,
    ae_input_observation_mask,
    ae_value_to_raw_dbm,
    raw_dbm_row_to_ae,
)
from lp import UpdatedGraphSnapshot
from sage_ve import (
    EdgeFeatureSAGE,
    GNNFeatureArtifact,
    GraphTensorInput,
    LoadedGNNCheckpoint,
    cosine_softplus_pair_losses,
    load_gnn_checkpoint,
)
from training_progress import DEFAULT_LOG_EVERY_K_EPOCHS, log_training_epoch, validate_log_interval


UPDATE_CHECKPOINT_FORMAT = "gufu_fingerprint_update_checkpoint_v4"
UPDATE_ARTIFACT_FORMAT = "gufu_updated_fingerprints_v1"
GRAPH_DIM = 34
ENCODER_DIM = 32
COORD_DIM = 2


class FingerprintUpdateConfig(BaseModel):
    """Training and numerical choices for one independent batch update."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    hidden_dim: int = Field(default=64, ge=1)
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)
    virtual_edge_sigma: float = Field(default=0.9, ge=-1.0, le=1.0)
    negative_weight: float = Field(default=1.0, ge=0.0)
    gnn_negative_objective: Literal[
        "positive_negative_cosine_softplus_v1"
    ] = "positive_negative_cosine_softplus_v1"
    contrastive_temperature: float = Field(default=0.1, gt=0.0)
    adjacency_policy: Literal["row_normalized_cross_virtual_v1"] = (
        "row_normalized_cross_virtual_v1"
    )
    location_scaling: Literal["bootstrap_label_axis_minmax_v1"] = (
        "bootstrap_label_axis_minmax_v1"
    )
    gnn_epochs: int = Field(default=1, ge=1)
    updater_epochs: int = Field(default=100, ge=1)
    ae_epochs: int = Field(default=5, ge=1)
    gnn_learning_rate: float = Field(default=0.01, gt=0.0)
    updater_learning_rate: float = Field(default=0.001, gt=0.0)
    ae_learning_rate: float = Field(default=0.001, gt=0.0)
    gnn_weight_decay: float = Field(default=0.0, ge=0.0)
    ae_weight_decay: float = Field(default=0.0, ge=0.0)
    seed: int = 0
    optimizer: Literal["Adam"] = "Adam"
    loss_reduction: Literal["raw_frobenius_norms"] = "raw_frobenius_norms"
    gnn_initialization: Literal["weights_only_from_selected_checkpoint"] = (
        "weights_only_from_selected_checkpoint"
    )
    resumable: Literal[False] = False

    @model_validator(mode="after")
    def finite_values(self):
        for key in (
            "alpha",
            "virtual_edge_sigma",
            "gnn_learning_rate",
            "updater_learning_rate",
            "ae_learning_rate",
            "gnn_weight_decay",
            "ae_weight_decay",
            "negative_weight",
            "contrastive_temperature",
        ):
            if not math.isfinite(getattr(self, key)):
                raise ValueError(f"{key} must be finite")
        return self


class UpdateLocationScaler(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    policy: Literal["bootstrap_label_axis_minmax_v1"] = (
        "bootstrap_label_axis_minmax_v1"
    )

    @model_validator(mode="after")
    def finite_ordered(self):
        values = (self.x_min, self.x_max, self.y_min, self.y_max)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("coordinate scaling values must be finite")
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("coordinate scaler maxima must not be below minima")
        return self

    def encode(self, coordinates: Tensor) -> Tensor:
        spans = coordinates.new_tensor(
            (self.x_max - self.x_min, self.y_max - self.y_min)
        )
        minima = coordinates.new_tensor((self.x_min, self.y_min))
        safe_spans = torch.where(spans == 0, torch.ones_like(spans), spans)
        return (coordinates - minima) / safe_spans

    def decode(self, coordinates: Tensor) -> Tensor:
        spans = coordinates.new_tensor(
            (self.x_max - self.x_min, self.y_max - self.y_min)
        )
        minima = coordinates.new_tensor((self.x_min, self.y_min))
        return minima + coordinates * spans


class OnlineRadioEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    edge_id: str
    sample_node_id: str
    ap_node_id: str
    scalar_weight: float = Field(ge=0.0)
    feature: tuple[float, ...]
    source: Literal[
        "retained_gnn_edge", "new_batch_endpoint_mean", "retained_endpoint_mean"
    ]

    @model_validator(mode="after")
    def valid_edge(self):
        if self.edge_id != f"radio:{self.sample_node_id}|{self.ap_node_id}":
            raise ValueError("online radio edge ID must be endpoint-stable")
        if not self.feature or not all(
            math.isfinite(value) for value in self.feature
        ):
            raise ValueError("online radio edge vector must be finite and nonempty")
        if not math.isfinite(self.scalar_weight):
            raise ValueError("online radio scalar weight must be finite")
        return self


class OnlineVirtualEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    edge_id: str
    left_sample_id: str
    right_sample_id: str
    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    feature: tuple[float, ...]

    @model_validator(mode="after")
    def canonical_pair(self):
        if self.left_sample_id >= self.right_sample_id:
            raise ValueError("online virtual edge endpoints must be in lexical order")
        if self.edge_id != f"virtual:{self.left_sample_id}|{self.right_sample_id}":
            raise ValueError("online virtual edge ID must be endpoint-stable")
        if not self.feature or not all(
            math.isfinite(value) for value in self.feature
        ):
            raise ValueError("online virtual edge vector must be finite and nonempty")
        if not math.isfinite(self.cosine_similarity):
            raise ValueError("online virtual cosine must be finite")
        return self


class OnlineUpdateGraph(BaseModel):
    """Combined temporary graph; it never becomes the persistent reference DB."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_online_update_graph_v4"] = "gufu_online_update_graph_v4"
    graph_id: str = Field(pattern=r"^onlinegraph:[0-9a-f]{64}$")
    reference_graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    current_reference_snapshot_id: str | None = None
    update_graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    dataset_id: str
    reference_batch_id: str
    update_batch_id: str
    normalization: NormalizationMode
    graph_build_config: GraphBuildConfig
    encoder_feature_dim: int = Field(default=ENCODER_DIM, ge=1)
    node_feature_dim: int = Field(default=GRAPH_DIM, ge=3)
    bssid_vocabulary: tuple[str, ...]
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    node_ids: tuple[str, ...]
    node_id_to_row: tuple[tuple[str, int], ...]
    node_types: tuple[Literal["sample", "ap"], ...]
    sample_node_ids: tuple[str, ...]
    reference_sample_ids: tuple[str, ...]
    update_sample_ids: tuple[str, ...]
    ap_node_ids: tuple[str, ...]
    node_feature_sources: tuple[
        Literal[
            "current_reference_gnn_state",
            "new_encoder_feature_plus_seeded_pseudo_xy",
            "current_reference_ap_gnn_state",
            "new_or_reactivated_ap_graph_mean",
        ], ...
    ]
    node_features: tuple[tuple[float, ...], ...]
    # Measured normalized RSSI rows and masks in sample_node_ids order.
    sample_signal_features: tuple[tuple[float, ...], ...]
    sample_signal_masks: tuple[tuple[bool, ...], ...]
    labeled_reference_ids: tuple[str, ...]
    labeled_reference_coordinates: tuple[tuple[float, float], ...]
    reference_label_mask_by_sample: tuple[bool, ...]
    radio_edges: tuple[OnlineRadioEdge, ...]
    virtual_edges: tuple[OnlineVirtualEdge, ...]
    negative_pairs: tuple[NegativeSamplePair, ...]
    temporary_sample_ids_removed_after_update: Literal[True] = True
    temporary_virtual_edges_removed_after_update: Literal[True] = True
    similarity_space: Literal[
        "normalized_rssi_cosine_min_overlap_v1"
    ] = RSSI_SIMILARITY_SPACE
    virtual_edge_rule: Literal["strict_cosine_greater_than_sigma"] = (
        "strict_cosine_greater_than_sigma"
    )
    virtual_edge_sigma: float = Field(ge=-1.0, le=1.0)
    negative_similarity_threshold: float = Field(gt=-1.0, le=1.0)
    minimum_shared_aps: int = Field(ge=1)
    max_negative_pairs_per_sample: int = Field(ge=1)

    @model_validator(mode="after")
    def valid_graph(self):
        if self.graph_build_config.spatial_normalization_enabled:
            metadata = self.graph_build_config.spatial_transform_metadata
            if metadata is None or self.graph_build_config.spatial_transform_sha256 is None:
                raise ValueError("online graph must record its fitted spatial transform")
            if metadata.site_id != self.dataset_id:
                raise ValueError("online graph spatial transform belongs to another site")
        if self.negative_similarity_threshold >= self.virtual_edge_sigma:
            raise ValueError("online negative similarity threshold must be below positive threshold")
        if self.node_ids != self.sample_node_ids + self.ap_node_ids:
            raise ValueError("online node rows must be samples followed by APs")
        if self.sample_node_ids != self.reference_sample_ids + self.update_sample_ids:
            raise ValueError("online sample rows must preserve reference then batch order")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("online graph node IDs must be unique")
        if self.node_id_to_row != tuple(
            (node_id, row) for row, node_id in enumerate(self.node_ids)
        ):
            raise ValueError("online graph ID-to-row mapping is misaligned")
        if self.node_types != ("sample",) * len(self.sample_node_ids) + (
            "ap",
        ) * len(self.ap_node_ids):
            raise ValueError("online graph node types are misaligned")
        if len(self.node_features) != len(self.node_ids) or len(
            self.node_feature_sources
        ) != len(self.node_ids):
            raise ValueError("online graph node metadata must align")
        if len(self.sample_signal_features) != len(self.sample_node_ids) or len(
            self.sample_signal_masks
        ) != len(self.sample_node_ids) or any(
            len(row) != len(self.bssid_vocabulary)
            or any(not math.isfinite(value) for value in row)
            for row in self.sample_signal_features
        ):
            raise ValueError(
                "online normalized RSSI features must align to sample IDs and AP width"
            )
        if any(
            len(row) != len(self.bssid_vocabulary)
            for row in self.sample_signal_masks
        ):
            raise ValueError("online RSSI masks must align to sample IDs and AP width")
        if any(
            value != 0.0
            for feature_row, mask_row in zip(self.sample_signal_features, self.sample_signal_masks)
            for value, observed in zip(feature_row, mask_row)
            if not observed
        ):
            raise ValueError("unobserved online RSSI columns must be zero")
        if self.node_feature_dim != self.encoder_feature_dim + COORD_DIM:
            raise ValueError("online graph feature width must be encoder width plus 2-D coordinates")
        if any(len(row) != self.node_feature_dim for row in self.node_features) or any(
            not math.isfinite(value) for row in self.node_features for value in row
        ):
            raise ValueError("online graph node matrix must match its declared feature width")
        if len(self.reference_label_mask_by_sample) != len(self.reference_sample_ids):
            raise ValueError("reference label mask must align with reference samples")
        if len(self.labeled_reference_ids) != len(
            self.labeled_reference_coordinates
        ):
            raise ValueError("reference location targets must align with labeled IDs")
        if not set(self.labeled_reference_ids) <= set(self.reference_sample_ids):
            raise ValueError("update training labels must come from reference samples")
        if self.bssid_vocabulary and canonical_json_digest(
            self.bssid_vocabulary
        ) != self.vocabulary_sha256:
            raise ValueError("online graph vocabulary digest is inconsistent")
        row_by_id = dict(self.node_id_to_row)
        if any(edge.sample_node_id not in self.sample_node_ids or edge.ap_node_id not in self.ap_node_ids for edge in self.radio_edges):
            raise ValueError("online radio edge endpoints are invalid")
        if any(len(edge.feature) != self.node_feature_dim for edge in self.radio_edges):
            raise ValueError("online radio edge feature width must match graph nodes")
        radio_pairs = [(edge.sample_node_id, edge.ap_node_id) for edge in self.radio_edges]
        if len(set(radio_pairs)) != len(radio_pairs):
            raise ValueError("online radio graph contains duplicate endpoint pairs")
        if self.radio_edges != tuple(
            sorted(self.radio_edges, key=lambda edge: (edge.sample_node_id, edge.ap_node_id))
        ):
            raise ValueError("online radio edges must have deterministic endpoint order")
        virtual_pairs = []
        signal_row = {sample_id: row for row, sample_id in enumerate(self.sample_node_ids)}
        for edge in self.virtual_edges:
            if len(edge.feature) != self.node_feature_dim:
                raise ValueError("online virtual edge feature width must match graph nodes")
            if edge.left_sample_id not in self.sample_node_ids or edge.right_sample_id not in self.sample_node_ids:
                raise ValueError("online virtual edge endpoints must be sample nodes")
            if edge.left_sample_id == edge.right_sample_id:
                raise ValueError("online virtual graph cannot contain self-neighbors")
            left = self.sample_signal_features[signal_row[edge.left_sample_id]]
            right = self.sample_signal_features[signal_row[edge.right_sample_id]]
            left_mask = self.sample_signal_masks[signal_row[edge.left_sample_id]]
            right_mask = self.sample_signal_masks[signal_row[edge.right_sample_id]]
            cosine = normalized_rssi_cosine(
                left,
                right,
                left_mask,
                right_mask,
                minimum_shared_aps=self.minimum_shared_aps,
            )
            if cosine is None or not math.isclose(cosine, edge.cosine_similarity, abs_tol=1e-12):
                raise ValueError("online virtual similarity does not match normalized RSSI")
            if not cosine > self.virtual_edge_sigma:
                raise ValueError("online virtual edge must pass the strict sigma threshold")
            virtual_pairs.append((edge.left_sample_id, edge.right_sample_id))
        if len(set(virtual_pairs)) != len(virtual_pairs):
            raise ValueError("online graph contains duplicate virtual relations")
        if self.virtual_edges != tuple(
            sorted(self.virtual_edges, key=lambda edge: (edge.left_sample_id, edge.right_sample_id))
        ):
            raise ValueError("online virtual edges must have deterministic order")
        expected_virtual, expected_negative = mine_rssi_pairs(
            self.sample_node_ids,
            self.sample_signal_features,
            self.sample_signal_masks,
            virtual_edge_sigma=self.virtual_edge_sigma,
            negative_similarity_threshold=self.negative_similarity_threshold,
            minimum_shared_aps=self.minimum_shared_aps,
            max_negative_pairs_per_sample=self.max_negative_pairs_per_sample,
        )
        actual_virtual = tuple(
            (edge.left_sample_id, edge.right_sample_id, edge.cosine_similarity)
            for edge in self.virtual_edges
        )
        if len(expected_virtual) != len(actual_virtual) or any(
            expected[:2] != actual[:2]
            or not math.isclose(expected[2], actual[2], abs_tol=1e-12)
            for expected, actual in zip(expected_virtual, actual_virtual)
        ):
            raise ValueError("online virtual edges do not match the RSSI mining policy")
        if expected_negative != self.negative_pairs:
            raise ValueError("online negative pairs do not match the RSSI mining policy")
        payload = _online_graph_payload(self)
        if self.graph_id != f"onlinegraph:{canonical_json_digest(payload)}":
            raise ValueError("online graph ID does not match its source state")
        return self


class UpdateLossValues(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    location_supervised: float
    feature_supervised: float
    location_consistency: float
    feature_consistency: float
    total: float

    @model_validator(mode="after")
    def finite_losses(self):
        if any(
            not math.isfinite(getattr(self, field))
            for field in self.model_fields
        ):
            raise ValueError("update loss values must be finite scalars")
        return self


class UpdateEpochLoss(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    epoch: int = Field(ge=1)
    location_supervised: float
    feature_supervised: float
    location_consistency: float
    feature_consistency: float
    total: float

    @model_validator(mode="after")
    def finite_values(self):
        if any(
            not math.isfinite(getattr(self, field))
            for field in (
                "location_supervised",
                "feature_supervised",
                "location_consistency",
                "feature_consistency",
                "total",
            )
        ):
            raise ValueError("epoch losses must be finite scalar values")
        return self


class FingerprintUpdateManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_fingerprint_update_checkpoint_v4"] = UPDATE_CHECKPOINT_FORMAT
    run_id: str
    config: FingerprintUpdateConfig
    dataset_id: str
    reference_batch_id: str
    update_batch_id: str
    reference_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    update_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    update_graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    online_graph_id: str = Field(pattern=r"^onlinegraph:[0-9a-f]{64}$")
    current_reference_snapshot_id: str | None = None
    reference_encoder_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    update_encoder_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_gnn_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ae_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_gnn_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bssid_vocabulary: tuple[str, ...]
    normalization: NormalizationMode
    encoder_feature_dim: int = Field(default=ENCODER_DIM, ge=1)
    graph_feature_dim: int = Field(default=GRAPH_DIM, ge=3)
    feature_target_space: Literal["z_encoder"] = "z_encoder"
    graph_input_space: Literal[
        "z_online_gnn_input_34d", "z_online_gnn_input_v1"
    ] = "z_online_gnn_input_v1"
    location_target_space: Literal["bootstrap_axis_minmax_2d"] = "bootstrap_axis_minmax_2d"
    decoder_input_space: Literal["z_update_mlp_32d", "z_update_mlp_v1"] = "z_update_mlp_v1"
    reference_node_ids: tuple[str, ...]
    update_node_ids: tuple[str, ...]
    labeled_reference_ids: tuple[str, ...]
    adjacency_shape: tuple[int, int]
    adjacency_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    location_scaler: UpdateLocationScaler
    loss_history: tuple[UpdateEpochLoss, ...]
    gnn_loss_history: tuple[float, ...]
    ae_loss_history: tuple[float, ...]
    gnn_model_config: dict[str, object]
    updater_initialization: Literal[
        "seeded_first_batch", "weights_only_previous_batch"
    ] = "seeded_first_batch"
    source_update_checkpoint_sha256: str | None = None
    source_checkpoint_ids_preserved: Literal[True] = True
    update_coordinates_used_as_training_labels: Literal[False] = False
    temporary_nodes_and_virtual_edges_retired: Literal[True] = True
    optimizer_states_saved: Literal[True] = True
    resumable: Literal[False] = False
    training_device: str
    torch_version: str
    created_at_utc: str

    @model_validator(mode="after")
    def validate_manifest(self):
        if self.graph_feature_dim != self.encoder_feature_dim + COORD_DIM:
            raise ValueError("update checkpoint graph and encoder feature dimensions disagree")
        if self.adjacency_shape != (
            len(self.update_node_ids),
            len(self.labeled_reference_ids),
        ):
            raise ValueError("adjacency dimensions do not match stable scan IDs")
        if len(self.loss_history) != self.config.updater_epochs:
            raise ValueError("updater loss history must contain one row per epoch")
        if len(self.gnn_loss_history) != self.config.gnn_epochs:
            raise ValueError("GNN loss history must contain one scalar per epoch")
        if len(self.ae_loss_history) != self.config.ae_epochs:
            raise ValueError("AE loss history must contain one scalar per epoch")
        if any(not math.isfinite(value) for values in (self.gnn_loss_history, self.ae_loss_history) for value in values):
            raise ValueError("model loss histories must be finite")
        if canonical_json_digest(self.bssid_vocabulary) != self.vocabulary_sha256:
            raise ValueError("update checkpoint vocabulary digest is inconsistent")
        if self.updater_initialization == "weights_only_previous_batch":
            if self.source_update_checkpoint_sha256 is None:
                raise ValueError("continued updater weights must identify their previous checkpoint")
        elif self.source_update_checkpoint_sha256 is not None:
            raise ValueError("first-batch updater initialization cannot name a previous checkpoint")
        return self


class FingerprintUpdateRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    scan_id: str
    original_location: tuple[float, float] | None
    raw_rssi_dbm: tuple[float, ...]
    observation_mask: tuple[bool, ...]
    updated_rssi_dbm: tuple[float, ...]
    updated_value_provenance: tuple[Literal["decoder_prediction", "retained_original"], ...]

    @model_validator(mode="after")
    def aligned_values(self):
        width = len(self.raw_rssi_dbm)
        if not (
            len(self.observation_mask)
            == len(self.updated_rssi_dbm)
            == len(self.updated_value_provenance)
            == width
        ):
            raise ValueError("fingerprint values, masks and provenance must align")
        if any(
            not math.isfinite(value)
            for values in (self.raw_rssi_dbm, self.updated_rssi_dbm)
            for value in values
        ):
            raise ValueError("fingerprint RSS values must be finite")
        if self.original_location is not None and not all(
            math.isfinite(value) for value in self.original_location
        ):
            raise ValueError("original reference location must be finite")
        return self


class PredictedUpdateLocation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    scan_id: str
    x: float
    y: float
    source: Literal["location_mlp_from_final_z_gnn"] = "location_mlp_from_final_z_gnn"

    @model_validator(mode="after")
    def finite_xy(self):
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("predicted update coordinates must be finite")
        return self


class UpdatedFingerprintArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_updated_fingerprints_v1"] = UPDATE_ARTIFACT_FORMAT
    run_id: str
    dataset_id: str
    reference_batch_id: str
    update_batch_id: str
    reference_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    update_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    update_graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    online_graph_id: str = Field(pattern=r"^onlinegraph:[0-9a-f]{64}$")
    source_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ae_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_gnn_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bssid_vocabulary: tuple[str, ...]
    normalization: NormalizationMode
    latent_feature_dim: int = Field(default=ENCODER_DIM, ge=1)
    output_unit: Literal["dbm"] = "dbm"
    value_origin: Literal["updated_rows_from_decoder_retained_unlabeled_rows_original"] = (
        "updated_rows_from_decoder_retained_unlabeled_rows_original"
    )
    reference_scan_ids: tuple[str, ...]
    update_scan_ids: tuple[str, ...]
    rows: tuple[FingerprintUpdateRow, ...]
    predicted_update_locations: tuple[PredictedUpdateLocation, ...]
    updated_latent_reference_ids: tuple[str, ...]
    updated_latent_reference_features: tuple[tuple[float, ...], ...]
    decoder_input_space: Literal["z_update_mlp_32d", "z_update_mlp_v1"] = "z_update_mlp_v1"
    temporary_update_scan_ids_discarded: Literal[True] = True
    temporary_virtual_edges_discarded: Literal[True] = True
    created_at_utc: str

    @model_validator(mode="after")
    def aligned_rows(self):
        if self.reference_scan_ids != tuple(row.scan_id for row in self.rows):
            raise ValueError("updated fingerprint rows must retain original reference order")
        if len(set(self.reference_scan_ids)) != len(self.reference_scan_ids):
            raise ValueError("updated fingerprint artifact contains duplicate reference IDs")
        if any(len(row.updated_rssi_dbm) != len(self.bssid_vocabulary) for row in self.rows):
            raise ValueError("updated RSS width does not match ordered AP vocabulary")
        if len(self.predicted_update_locations) == 0:
            raise ValueError("every update scan needs an exported predicted location")
        if self.update_scan_ids != tuple(
            row.scan_id for row in self.predicted_update_locations
        ):
            raise ValueError("predicted locations must align with update scan IDs")
        if len(set(row.scan_id for row in self.predicted_update_locations)) != len(
            self.predicted_update_locations
        ):
            raise ValueError("predicted update scan IDs must be unique")
        if len(self.updated_latent_reference_ids) != len(
            self.updated_latent_reference_features
        ) or len(set(self.updated_latent_reference_ids)) != len(
            self.updated_latent_reference_ids
        ):
            raise ValueError("updated reference latent features must align to stable IDs")
        if any(len(row) != self.latent_feature_dim for row in self.updated_latent_reference_features):
            raise ValueError("updated reference latent feature width must match latent_feature_dim")
        if any(
            not math.isfinite(value)
            for row in self.updated_latent_reference_features
            for value in row
        ):
            raise ValueError("updated reference latent features must be finite")
        if canonical_json_digest(self.bssid_vocabulary) != self.vocabulary_sha256:
            raise ValueError("updated fingerprint vocabulary digest is inconsistent")
        return self


@dataclass(frozen=True)
class FingerprintUpdateContext:
    reference_prepared: PreparedDataset
    reference_graph: GraphState
    reference_encoder_features: EncoderFeatureArtifact
    reference_gnn_features: GNNFeatureArtifact
    update_prepared: PreparedDataset
    update_encoder_features: EncoderFeatureArtifact
    ae_checkpoint_path: Path
    gnn_checkpoint_path: Path
    reference_snapshot: UpdatedGraphSnapshot | None
    online_graph: OnlineUpdateGraph
    tensors: GraphTensorInput
    labeled_reference_ids: tuple[str, ...]
    reference_coordinates: Tensor
    update_encoder_targets: Tensor
    update_ae_inputs: Tensor
    reference_ae_inputs: Tensor
    adjacency: Tensor
    location_scaler: UpdateLocationScaler
    source_ae_checkpoint_sha256: str
    source_gnn_checkpoint_sha256: str
    reference_prepared_sha256: str
    update_prepared_sha256: str
    reference_encoder_artifact_sha256: str
    update_encoder_artifact_sha256: str
    reference_gnn_artifact_sha256: str


@dataclass
class FingerprintUpdaterModels:
    location_model: nn.Module
    feature_model: nn.Module
    gnn_model: EdgeFeatureSAGE
    ae_model: AE
    manifest: FingerprintUpdateManifest
    checkpoint_path: Path
    checkpoint_sha256: str
    optimizer_states: dict


@dataclass(frozen=True)
class FingerprintUpdateTrainingResult:
    models: FingerprintUpdaterModels
    artifact: UpdatedFingerprintArtifact
    artifact_path: Path


class LocationPredictor(nn.Module):
    """MLP 1: final GNN sample feature to scaled 2-D location."""

    def __init__(self, hidden_dim: int, graph_feature_dim: int = GRAPH_DIM):
        super().__init__()
        if graph_feature_dim < 1:
            raise ValueError("graph_feature_dim must be positive")
        self.graph_feature_dim = graph_feature_dim
        self.layers = nn.Sequential(
            nn.Linear(graph_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, COORD_DIM),
        )

    def forward(self, graph_features: Tensor) -> Tensor:
        if graph_features.ndim != 2 or graph_features.shape[1] != self.graph_feature_dim:
            raise ValueError("location MLP input width differs from graph_feature_dim")
        return self.layers(graph_features)


class LocationToFeatureUpdater(nn.Module):
    """MLP 2: normalized 2-D location to the configured AE latent width."""

    def __init__(self, hidden_dim: int, output_feature_dim: int = ENCODER_DIM):
        super().__init__()
        if output_feature_dim < 1:
            raise ValueError("output_feature_dim must be positive")
        self.output_feature_dim = output_feature_dim
        self.layers = nn.Sequential(
            nn.Linear(COORD_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_feature_dim),
        )

    def forward(self, scaled_coordinates: Tensor) -> Tensor:
        if scaled_coordinates.ndim != 2 or scaled_coordinates.shape[1] != COORD_DIM:
            raise ValueError("feature updater expects [samples,2] scaled coordinates")
        return self.layers(scaled_coordinates)


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float | None:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    similarity = sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
    return max(-1.0, min(1.0, similarity))


def _mean(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    return tuple((a + b) / 2.0 for a, b in zip(left, right))


def _online_graph_payload(graph: OnlineUpdateGraph | dict) -> dict:
    data = graph if isinstance(graph, dict) else graph.model_dump(mode="json")
    excluded = {"graph_id", "format"}
    return {key: value for key, value in data.items() if key not in excluded}


def _snapshot_row(snapshot: UpdatedGraphSnapshot, node_id: str) -> tuple[float, ...]:
    try:
        row = dict(snapshot.node_id_to_row)[node_id]
    except KeyError as exc:
        raise ValueError(f"current reference snapshot is missing node {node_id}") from exc
    return snapshot.node_features[row]


def _make_online_graph(
    reference_graph: GraphState,
    reference_gnn: GNNFeatureArtifact,
    update_graph: GraphState,
    *,
    snapshot: UpdatedGraphSnapshot | None,
    sigma: float,
) -> OnlineUpdateGraph:
    if reference_graph.batch_role != "bootstrap_reference":
        raise ValueError("fingerprint updating requires a bootstrap reference graph")
    if update_graph.batch_role != "update_batch":
        raise ValueError("the incoming graph must have update_batch role")
    if reference_graph.dataset_id != update_graph.dataset_id:
        raise ValueError("reference and update dataset IDs differ")
    if reference_graph.normalization != update_graph.normalization:
        raise ValueError("reference and update normalization differ")
    if reference_graph.build_config != update_graph.build_config:
        raise ValueError("reference and update graph construction configs differ")
    if reference_graph.bssid_vocabulary != update_graph.bssid_vocabulary:
        raise ValueError("no-vocabulary-change update requires identical ordered BSSID vocabulary")
    if reference_graph.build_config.virtual_edge_strategy != "paper_cosine_threshold":
        raise ValueError("online update currently requires paper cosine-threshold virtual edges")
    if reference_gnn.graph_id != reference_graph.graph_id:
        raise ValueError("reference GNN feature artifact does not match the reference graph")
    if reference_gnn.node_ids != reference_graph.node_ids:
        raise ValueError("reference GNN rows are not aligned to the reference graph")
    if reference_gnn.node_feature_dim != reference_graph.node_feature_dim:
        raise ValueError("reference GNN node width differs from graph state")
    if reference_graph.encoder_feature_dim != update_graph.encoder_feature_dim:
        raise ValueError("reference and update encoder latent widths differ")
    if reference_graph.node_feature_dim != update_graph.node_feature_dim:
        raise ValueError("reference and update graph feature widths differ")
    if snapshot is not None:
        snapshot_checks = (
            (snapshot.parent_graph_id == reference_graph.graph_id, "parent graph"),
            (snapshot.dataset_id == reference_graph.dataset_id, "dataset"),
            (snapshot.normalization == reference_graph.normalization, "normalization"),
            (snapshot.bssid_vocabulary == reference_graph.bssid_vocabulary, "vocabulary"),
            (snapshot.sample_node_ids == reference_graph.sample_node_ids, "reference samples"),
            (snapshot.gnn_checkpoint_sha256 == reference_gnn.checkpoint_sha256, "GNN checkpoint"),
        )
        for matches, name in snapshot_checks:
            if not matches:
                raise ValueError(f"current reference snapshot has incompatible {name}")

    ref_rows = reference_graph.id_to_row
    ref_gnn_rows = dict(reference_gnn.node_id_to_row)
    if snapshot is None:
        reference_features = {
            node_id: tuple(reference_gnn.node_feature_matrix[ref_gnn_rows[node_id]])
            for node_id in reference_graph.node_ids
        }
        current_ap_ids = reference_graph.ap_node_ids
        current_radio: list[OnlineRadioEdge] = []
        gnn_edge_rows = {
            edge_id: tuple(row)
            for edge_id, row in zip(
                reference_gnn.radio_edge_ids,
                reference_gnn.radio_edge_feature_matrix,
            )
        }
        for edge in reference_graph.radio_edges:
            feature = gnn_edge_rows.get(edge.edge_id)
            source: Literal["retained_gnn_edge", "new_batch_endpoint_mean", "retained_endpoint_mean"]
            if feature is None:
                feature = _mean(
                    reference_features[edge.sample_node_id],
                    reference_features[edge.ap_node_id],
                )
                source = "retained_endpoint_mean"
            else:
                source = "retained_gnn_edge"
            current_radio.append(
                OnlineRadioEdge(
                    edge_id=edge.edge_id,
                    sample_node_id=edge.sample_node_id,
                    ap_node_id=edge.ap_node_id,
                    scalar_weight=edge.scalar_weight,
                    feature=feature,
                    source=source,
                )
            )
        current_snapshot_id = None
    else:
        snapshot_rows = dict(snapshot.node_id_to_row)
        reference_features = {
            node_id: tuple(snapshot.node_features[snapshot_rows[node_id]])
            for node_id in snapshot.node_ids
        }
        current_ap_ids = snapshot.active_ap_node_ids
        current_radio = [
            OnlineRadioEdge(
                edge_id=edge.edge_id,
                sample_node_id=edge.sample_node_id,
                ap_node_id=edge.ap_node_id,
                scalar_weight=edge.scalar_weight,
                feature=edge.edge_feature,
                source="retained_gnn_edge",
            )
            for edge in snapshot.radio_edges
        ]
        current_snapshot_id = snapshot.graph_id

    update_rows = update_graph.id_to_row
    update_features = {
        node_id: tuple(update_graph.node_features[update_rows[node_id]])
        for node_id in update_graph.node_ids
    }
    signal_feature_by_id = {
        scan_id: reference_graph.sample_rssi_features[row]
        for row, scan_id in enumerate(reference_graph.sample_node_ids)
    }
    signal_feature_by_id.update(
        {
            scan_id: update_graph.sample_rssi_features[row]
            for row, scan_id in enumerate(update_graph.sample_node_ids)
        }
    )
    signal_mask_by_id = {
        scan_id: reference_graph.sample_rssi_observation_masks[row]
        for row, scan_id in enumerate(reference_graph.sample_node_ids)
    }
    signal_mask_by_id.update(
        {
            scan_id: update_graph.sample_rssi_observation_masks[row]
            for row, scan_id in enumerate(update_graph.sample_node_ids)
        }
    )
    ap_vocab_order = reference_graph.ap_ids
    active_ap_ids = tuple(
        ap_id
        for ap_id in ap_vocab_order
        if ap_id in set(current_ap_ids) | set(update_graph.ap_node_ids)
    )
    sample_ids = reference_graph.sample_node_ids + update_graph.sample_node_ids
    node_ids = sample_ids + active_ap_ids
    feature_by_id: dict[str, tuple[float, ...]] = {}
    source_by_id: dict[str, str] = {}
    for scan_id in reference_graph.sample_node_ids:
        feature_by_id[scan_id] = reference_features[scan_id]
        source_by_id[scan_id] = "current_reference_gnn_state"
    for scan_id in update_graph.sample_node_ids:
        feature_by_id[scan_id] = update_features[scan_id]
        source_by_id[scan_id] = "new_encoder_feature_plus_seeded_pseudo_xy"
    for ap_id in active_ap_ids:
        if ap_id in reference_features:
            feature_by_id[ap_id] = reference_features[ap_id]
            source_by_id[ap_id] = "current_reference_ap_gnn_state"
        else:
            feature_by_id[ap_id] = update_features[ap_id]
            source_by_id[ap_id] = "new_or_reactivated_ap_graph_mean"

    current_pairs = {(edge.sample_node_id, edge.ap_node_id) for edge in current_radio}
    update_radio: list[OnlineRadioEdge] = []
    for edge in update_graph.radio_edges:
        pair = edge.sample_node_id, edge.ap_node_id
        if pair in current_pairs:
            raise ValueError(f"duplicate sample/AP identity across batches: {pair}")
        update_radio.append(
            OnlineRadioEdge(
                edge_id=edge.edge_id,
                sample_node_id=edge.sample_node_id,
                ap_node_id=edge.ap_node_id,
                scalar_weight=edge.scalar_weight,
                feature=_mean(feature_by_id[edge.sample_node_id], feature_by_id[edge.ap_node_id]),
                source="new_batch_endpoint_mean",
            )
        )
    radio_edges = tuple(
        sorted(current_radio + update_radio, key=lambda edge: (edge.sample_node_id, edge.ap_node_id))
    )

    ordered_signal_features = tuple(signal_feature_by_id[scan_id] for scan_id in sample_ids)
    ordered_signal_masks = tuple(signal_mask_by_id[scan_id] for scan_id in sample_ids)
    virtual_pairs, negative_pairs = mine_rssi_pairs(
        sample_ids,
        ordered_signal_features,
        ordered_signal_masks,
        virtual_edge_sigma=sigma,
        negative_similarity_threshold=reference_graph.build_config.negative_similarity_threshold,
        minimum_shared_aps=reference_graph.build_config.minimum_shared_aps,
        max_negative_pairs_per_sample=reference_graph.build_config.max_negative_pairs_per_sample,
    )
    virtual: list[OnlineVirtualEdge] = []
    for a, b, similarity in virtual_pairs:
        virtual.append(
            OnlineVirtualEdge(
                edge_id=f"virtual:{a}|{b}",
                left_sample_id=a,
                right_sample_id=b,
                cosine_similarity=similarity,
                feature=_mean(feature_by_id[a], feature_by_id[b]),
            )
        )
    virtual_edges = tuple(sorted(virtual, key=lambda edge: (edge.left_sample_id, edge.right_sample_id)))

    label_by_id = {
        scan_id: reference_graph.reference_coordinate_targets[ref_rows[scan_id]]
        for scan_id in reference_graph.sample_node_ids
    }
    labeled_ids = tuple(scan_id for scan_id in reference_graph.sample_node_ids if label_by_id[scan_id] is not None)
    labeled_coords = tuple(label_by_id[scan_id] for scan_id in labeled_ids)
    # Pydantic narrows optional values at runtime only; the mask below guarantees
    # every coordinate target in this tuple is present.
    coordinates = tuple(tuple(value for value in xy) for xy in labeled_coords if xy is not None)
    graph_fields = dict(
        reference_graph_id=reference_graph.graph_id,
        current_reference_snapshot_id=current_snapshot_id,
        update_graph_id=update_graph.graph_id,
        dataset_id=reference_graph.dataset_id,
        reference_batch_id=reference_graph.batch_id,
        update_batch_id=update_graph.batch_id,
        normalization=reference_graph.normalization,
        graph_build_config=reference_graph.build_config,
        encoder_feature_dim=reference_graph.encoder_feature_dim,
        node_feature_dim=reference_graph.node_feature_dim,
        bssid_vocabulary=reference_graph.bssid_vocabulary,
        vocabulary_sha256=reference_graph.vocabulary_sha256,
        node_ids=node_ids,
        node_id_to_row=tuple((node_id, row) for row, node_id in enumerate(node_ids)),
        node_types=("sample",) * len(sample_ids) + ("ap",) * len(active_ap_ids),
        sample_node_ids=sample_ids,
        reference_sample_ids=reference_graph.sample_node_ids,
        update_sample_ids=update_graph.sample_node_ids,
        ap_node_ids=active_ap_ids,
        node_feature_sources=tuple(source_by_id[node_id] for node_id in node_ids),
        node_features=tuple(feature_by_id[node_id] for node_id in node_ids),
        sample_signal_features=tuple(
            signal_feature_by_id[scan_id] for scan_id in sample_ids
        ),
        sample_signal_masks=tuple(signal_mask_by_id[scan_id] for scan_id in sample_ids),
        similarity_space=RSSI_SIMILARITY_SPACE,
        labeled_reference_ids=labeled_ids,
        labeled_reference_coordinates=coordinates,
        reference_label_mask_by_sample=tuple(label_by_id[node_id] is not None for node_id in reference_graph.sample_node_ids),
        radio_edges=radio_edges,
        virtual_edges=virtual_edges,
        negative_pairs=negative_pairs,
        virtual_edge_sigma=sigma,
        negative_similarity_threshold=reference_graph.build_config.negative_similarity_threshold,
        minimum_shared_aps=reference_graph.build_config.minimum_shared_aps,
        max_negative_pairs_per_sample=reference_graph.build_config.max_negative_pairs_per_sample,
    )
    provisional = OnlineUpdateGraph.model_construct(
        graph_id="onlinegraph:" + "0" * 64,
        **graph_fields,
    )
    graph_id = f"onlinegraph:{canonical_json_digest(_online_graph_payload(provisional))}"
    return OnlineUpdateGraph(graph_id=graph_id, **graph_fields)


def _online_graph_tensors(graph: OnlineUpdateGraph) -> GraphTensorInput:
    row_by_id = dict(graph.node_id_to_row)
    edges = tuple(graph.radio_edges) + tuple(graph.virtual_edges)
    left_ids = tuple(edge.sample_node_id for edge in graph.radio_edges) + tuple(
        edge.left_sample_id for edge in graph.virtual_edges
    )
    right_ids = tuple(edge.ap_node_id for edge in graph.radio_edges) + tuple(
        edge.right_sample_id for edge in graph.virtual_edges
    )
    edge_ids = tuple(edge.edge_id for edge in edges)
    edge_features = tuple(edge.feature for edge in edges)
    return GraphTensorInput(
        node_features=torch.tensor(graph.node_features, dtype=torch.float32).reshape((-1, graph.node_feature_dim)),
        edge_features=torch.tensor(edge_features, dtype=torch.float32).reshape((-1, graph.node_feature_dim)),
        edge_left_rows=torch.tensor([row_by_id[node_id] for node_id in left_ids], dtype=torch.long),
        edge_right_rows=torch.tensor([row_by_id[node_id] for node_id in right_ids], dtype=torch.long),
        radio_edge_count=len(graph.radio_edges),
        node_ids=graph.node_ids,
        edge_ids=edge_ids,
        radio_edge_ids=tuple(edge.edge_id for edge in graph.radio_edges),
        virtual_edge_ids=tuple(edge.edge_id for edge in graph.virtual_edges),
        negative_left_rows=torch.tensor(
            [row_by_id[pair.left_sample_id] for pair in graph.negative_pairs],
            dtype=torch.long,
        ),
        negative_right_rows=torch.tensor(
            [row_by_id[pair.right_sample_id] for pair in graph.negative_pairs],
            dtype=torch.long,
        ),
        negative_pair_strengths=torch.tensor(
            [pair.negative_strength for pair in graph.negative_pairs],
            dtype=torch.float32,
        ),
    )


def _location_scaler(coordinates: Tensor) -> UpdateLocationScaler:
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or coordinates.shape[0] < 1:
        raise ValueError("at least one labeled bootstrap coordinate is required")
    return UpdateLocationScaler(
        x_min=float(coordinates[:, 0].min()),
        x_max=float(coordinates[:, 0].max()),
        y_min=float(coordinates[:, 1].min()),
        y_max=float(coordinates[:, 1].max()),
    )


def cross_virtual_adjacency(
    graph: OnlineUpdateGraph,
    *,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Return row-normalized binary A with rows=new scans, cols=labeled refs."""
    label_col = {scan_id: col for col, scan_id in enumerate(graph.labeled_reference_ids)}
    update_row = {scan_id: row for row, scan_id in enumerate(graph.update_sample_ids)}
    adjacency = torch.zeros(
        (len(update_row), len(label_col)), dtype=dtype
    )
    for edge in graph.virtual_edges:
        if edge.left_sample_id in update_row and edge.right_sample_id in label_col:
            adjacency[update_row[edge.left_sample_id], label_col[edge.right_sample_id]] = 1.0
        elif edge.right_sample_id in update_row and edge.left_sample_id in label_col:
            adjacency[update_row[edge.right_sample_id], label_col[edge.left_sample_id]] = 1.0
    degree = adjacency.sum(dim=1, keepdim=True)
    return torch.where(degree > 0, adjacency / degree.clamp_min(1.0), adjacency)


def _coordinates_for_ids(prepared: PreparedDataset, ids: tuple[str, ...]) -> dict[str, tuple[float, float]]:
    table = {row.scan_id: (row.x, row.y) for row in prepared.reference_positions}
    if set(table) != {scan.scan_id for scan in prepared.scans if scan.coordinate_role == "bootstrap_train"}:
        raise ValueError("bootstrap position table and scan coordinate roles do not agree")
    return {scan_id: table[scan_id] for scan_id in ids if scan_id in table}


def prepare_fingerprint_update(
    reference_prepared: PreparedDataset,
    reference_graph: GraphState,
    reference_encoder_features: EncoderFeatureArtifact,
    reference_gnn_features: GNNFeatureArtifact,
    update_prepared: PreparedDataset,
    update_encoder_features: EncoderFeatureArtifact,
    ae_checkpoint_path: str | Path,
    gnn_checkpoint_path: str | Path,
    *,
    reference_snapshot: UpdatedGraphSnapshot | None = None,
    config: FingerprintUpdateConfig | None = None,
) -> FingerprintUpdateContext:
    """Validate all state, construct the transient combined graph and targets."""
    selected = config or FingerprintUpdateConfig()
    if reference_prepared.config.batch_role != "bootstrap_reference":
        raise ValueError("reference data must have bootstrap_reference role")
    if update_prepared.config.batch_role != "update_batch":
        raise ValueError("incoming data must have update_batch role")
    if reference_prepared.config.dataset_id != update_prepared.config.dataset_id:
        raise ValueError("reference and update dataset IDs differ")
    if reference_graph.source_prepared_sha256 != canonical_json_digest(reference_prepared):
        raise ValueError("reference graph does not match reference prepared data")
    if reference_graph.dataset_id != reference_prepared.config.dataset_id:
        raise ValueError("reference graph dataset identity does not match")
    if reference_prepared.bssid_vocabulary != update_prepared.bssid_vocabulary:
        raise ValueError("no-vocabulary-change update requires exactly the same ordered BSSID vocabulary")
    if reference_prepared.config.normalization_mode != update_prepared.config.normalization_mode:
        raise ValueError("reference and update RSS normalization differ")
    if reference_graph.bssid_vocabulary != reference_prepared.bssid_vocabulary:
        raise ValueError("reference graph AP vocabulary does not match prepared data")
    if selected.virtual_edge_sigma != reference_graph.build_config.virtual_edge_sigma:
        raise ValueError("online virtual-edge threshold must match the reference graph config")

    ref_ids = tuple(scan.scan_id for scan in reference_prepared.scans)
    update_ids = tuple(scan.scan_id for scan in update_prepared.scans)
    if reference_graph.sample_node_ids != ref_ids:
        raise ValueError("reference graph sample IDs/order do not match original prepared scans")
    ref_prepared_hash = canonical_json_digest(reference_prepared)
    update_prepared_hash = canonical_json_digest(update_prepared)
    for artifact, prepared, ids, label in (
        (reference_encoder_features, reference_prepared, ref_ids, "reference encoder"),
        (update_encoder_features, update_prepared, update_ids, "update encoder"),
    ):
        if artifact.dataset_id != prepared.config.dataset_id or artifact.batch_id != prepared.config.batch_id:
            raise ValueError(f"{label} artifact dataset/batch identity is incompatible")
        if artifact.scan_ids != ids or len(artifact.feature_matrix) != len(ids):
            raise ValueError(f"{label} artifact scan IDs/order do not match prepared data")
        if artifact.source_prepared_sha256 != canonical_json_digest(prepared):
            raise ValueError(f"{label} artifact was encoded from different prepared data")
        if artifact.bssid_vocabulary != reference_prepared.bssid_vocabulary:
            raise ValueError(f"{label} artifact vocabulary order is incompatible")
        if artifact.normalization != reference_prepared.config.normalization_mode:
            raise ValueError(f"{label} artifact normalization is incompatible")
        if artifact.feature_dim != reference_graph.encoder_feature_dim:
            raise ValueError(f"{label} artifact latent width is incompatible with graph state")

    ae_path = Path(ae_checkpoint_path)
    gnn_path = Path(gnn_checkpoint_path)
    ae_checkpoint = load_ae_checkpoint(ae_path, reference_prepared)
    load_ae_checkpoint(ae_path, update_prepared)
    if reference_encoder_features.checkpoint_sha256 != ae_checkpoint.checkpoint_sha256:
        raise ValueError("reference encoder features do not use the selected AE checkpoint")
    if update_encoder_features.checkpoint_sha256 != ae_checkpoint.checkpoint_sha256:
        raise ValueError("update encoder features do not use the selected AE checkpoint")
    if ae_checkpoint.manifest.model.latent_dim != reference_graph.encoder_feature_dim:
        raise ValueError("AE checkpoint latent width does not match graph feature provenance")
    if ae_checkpoint.manifest.training_config.dtype != "float32":
        raise ValueError("fingerprint updating currently requires the selected AE's recorded float32 dtype")
    if reference_encoder_features.dtype != "float32" or update_encoder_features.dtype != "float32":
        raise ValueError("fingerprint updating requires float32 encoder feature artifacts")
    if reference_gnn_features.encoder_checkpoint_sha256 != ae_checkpoint.checkpoint_sha256:
        raise ValueError("reference GNN features do not use the selected AE checkpoint")
    gnn_checkpoint = load_gnn_checkpoint(gnn_path, reference_graph)
    if reference_gnn_features.checkpoint_sha256 != gnn_checkpoint.checkpoint_sha256:
        raise ValueError("reference GNN artifact does not use the selected GNN checkpoint")
    if reference_gnn_features.vocabulary_sha256 != reference_graph.vocabulary_sha256:
        raise ValueError("reference GNN artifact vocabulary is incompatible")
    if reference_gnn_features.normalization != update_prepared.config.normalization_mode:
        raise ValueError("reference GNN and update normalization differ")

    update_graph = build_graph_state(
        update_prepared,
        update_encoder_features,
        reference_graph.build_config,
    )
    online_graph = _make_online_graph(
        reference_graph,
        reference_gnn_features,
        update_graph,
        snapshot=reference_snapshot,
        sigma=selected.virtual_edge_sigma,
    )
    tensors = _online_graph_tensors(online_graph)
    if not online_graph.radio_edges and not online_graph.virtual_edges:
        raise ValueError("online GNN cosine objective requires at least one positive pair")

    coords_by_id = _coordinates_for_ids(reference_prepared, ref_ids)
    labeled_ids = tuple(
        scan_id for scan_id in reference_graph.sample_node_ids if scan_id in coords_by_id
    )
    if labeled_ids != online_graph.labeled_reference_ids:
        raise ValueError("bootstrap training labels disagree with graph label masks")
    coords = torch.tensor([coords_by_id[scan_id] for scan_id in labeled_ids], dtype=torch.float32)
    scaler = _location_scaler(coords)
    encoder_ref_by_id = {
        scan_id: reference_encoder_features.feature_matrix[row]
        for row, scan_id in enumerate(reference_encoder_features.scan_ids)
    }
    encoder_update_by_id = {
        scan_id: update_encoder_features.feature_matrix[row]
        for row, scan_id in enumerate(update_encoder_features.scan_ids)
    }
    reference_values_by_id = {
        scan.scan_id: raw_dbm_row_to_ae(
            scan.rss_dbm, ae_input_observation_mask(scan), reference_prepared.config.normalization_mode
        )
        for scan in reference_prepared.scans
    }
    update_values_by_id = {
        scan.scan_id: raw_dbm_row_to_ae(
            scan.rss_dbm, scan.observation_mask, update_prepared.config.normalization_mode
        )
        for scan in update_prepared.scans
    }
    ref_scan_by_id = {scan.scan_id: scan for scan in reference_prepared.scans}
    update_scan_by_id = {scan.scan_id: scan for scan in update_prepared.scans}
    if set(encoder_ref_by_id) != set(ref_ids) or set(encoder_update_by_id) != set(update_ids):
        raise ValueError("encoder feature rows are incomplete")
    adjacency = cross_virtual_adjacency(online_graph)
    if adjacency.shape != (len(update_ids), len(labeled_ids)):
        raise AssertionError("cross-virtual adjacency lost its stable-ID dimensions")
    return FingerprintUpdateContext(
        reference_prepared=reference_prepared,
        reference_graph=reference_graph,
        reference_encoder_features=reference_encoder_features,
        reference_gnn_features=reference_gnn_features,
        update_prepared=update_prepared,
        update_encoder_features=update_encoder_features,
        ae_checkpoint_path=ae_path,
        gnn_checkpoint_path=gnn_path,
        reference_snapshot=reference_snapshot,
        online_graph=online_graph,
        tensors=tensors,
        labeled_reference_ids=labeled_ids,
        reference_coordinates=coords,
        update_encoder_targets=torch.tensor(
            [encoder_update_by_id[scan_id] for scan_id in update_ids], dtype=torch.float32
        ),
        update_ae_inputs=torch.tensor(
            [update_values_by_id[scan_id] for scan_id in update_ids], dtype=torch.float32
        ).unsqueeze(1),
        reference_ae_inputs=torch.tensor(
            [reference_values_by_id[scan_id] for scan_id in labeled_ids], dtype=torch.float32
        ).unsqueeze(1),
        adjacency=adjacency,
        location_scaler=scaler,
        source_ae_checkpoint_sha256=ae_checkpoint.checkpoint_sha256,
        source_gnn_checkpoint_sha256=gnn_checkpoint.checkpoint_sha256,
        reference_prepared_sha256=ref_prepared_hash,
        update_prepared_sha256=update_prepared_hash,
        reference_encoder_artifact_sha256=canonical_json_digest(reference_encoder_features),
        update_encoder_artifact_sha256=canonical_json_digest(update_encoder_features),
        reference_gnn_artifact_sha256=canonical_json_digest(reference_gnn_features),
    )


def compute_update_losses(
    location_model: nn.Module,
    feature_model: nn.Module,
    reference_gnn_features: Tensor,
    update_gnn_features: Tensor,
    reference_coordinates_scaled: Tensor,
    update_encoder_targets: Tensor,
    adjacency_update_reference: Tensor,
    *,
    alpha: float,
) -> tuple[dict[str, Tensor], Tensor, Tensor]:
    """Compute Eqs. (5)–(9) under row_normalized_cross_virtual_v1."""
    if not 0.0 <= alpha <= 1.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and in [0,1]")
    k = update_gnn_features.shape[0]
    n = reference_coordinates_scaled.shape[0]
    if adjacency_update_reference.shape != (k, n):
        raise ValueError("A must have shape [num_update_scans,num_labeled_refs]")
    graph_dim = reference_gnn_features.shape[1] if reference_gnn_features.ndim == 2 else 0
    encoder_dim = update_encoder_targets.shape[1] if update_encoder_targets.ndim == 2 else 0
    if graph_dim < 1 or reference_gnn_features.shape != (n, graph_dim):
        raise ValueError("labeled reference GNN features must be [N_ref,graph_feature_dim]")
    if update_gnn_features.shape != (k, graph_dim):
        raise ValueError("update GNN features must align to reference graph feature width")
    if encoder_dim < 1 or update_encoder_targets.shape != (k, encoder_dim):
        raise ValueError("fixed update encoder targets must be [K,encoder_feature_dim]")
    if getattr(location_model, "graph_feature_dim", graph_dim) != graph_dim:
        raise ValueError("location predictor width is incompatible with graph features")
    if getattr(feature_model, "output_feature_dim", encoder_dim) != encoder_dim:
        raise ValueError("feature updater width is incompatible with encoder targets")
    if reference_coordinates_scaled.shape != (n, COORD_DIM):
        raise ValueError("scaled bootstrap coordinates must be [N_ref,2]")

    predicted_reference_xy = location_model(reference_gnn_features)
    predicted_update_xy = location_model(update_gnn_features)
    predicted_reference_z = feature_model(reference_coordinates_scaled)
    predicted_update_z = feature_model(predicted_update_xy)
    location_supervised = torch.linalg.vector_norm(
        predicted_reference_xy - reference_coordinates_scaled
    )
    feature_supervised = torch.linalg.vector_norm(
        predicted_update_z - update_encoder_targets
    )
    nonempty = adjacency_update_reference.sum(dim=1) > 0
    if bool(nonempty.any()):
        location_consistency = torch.linalg.vector_norm(
            adjacency_update_reference[nonempty] @ reference_coordinates_scaled
            - predicted_update_xy[nonempty]
        )
        feature_consistency = torch.linalg.vector_norm(
            adjacency_update_reference[nonempty] @ predicted_reference_z
            - update_encoder_targets[nonempty]
        )
    else:
        # Preserve a differentiable zero tied to the intended prediction paths.
        location_consistency = predicted_update_xy.sum() * 0.0
        feature_consistency = predicted_reference_z.sum() * 0.0
    total = alpha * (location_supervised + feature_supervised) + (1.0 - alpha) * (
        location_consistency + feature_consistency
    )
    losses = {
        "location_supervised": location_supervised,
        "feature_supervised": feature_supervised,
        "location_consistency": location_consistency,
        "feature_consistency": feature_consistency,
        "total": total,
    }
    if any(not torch.isfinite(value).item() for value in losses.values()):
        raise FloatingPointError("update loss contains a non-finite value")
    return losses, predicted_update_xy, predicted_reference_z


def equation4_online_loss(
    node_embeddings: Tensor,
    graph: GraphTensorInput,
    *,
    temperature: float = 0.1,
    negative_weight: float = 1.0,
    epsilon: float | None = None,
) -> Tensor:
    """Mean softplus cosine loss over online positive and negative pairs."""
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("contrastive temperature must be finite and positive")
    if not math.isfinite(negative_weight) or negative_weight < 0.0:
        raise ValueError("negative_weight must be finite and nonnegative")
    if node_embeddings.shape != (len(graph.node_ids), graph.node_features.shape[1]):
        raise ValueError("online cosine loss expects [nodes,online_graph.node_feature_dim]")
    if not graph.edge_ids:
        raise ValueError("online cosine loss requires at least one positive pair")
    if (
        graph.negative_left_rows is None
        or graph.negative_right_rows is None
        or graph.negative_left_rows.numel() == 0
    ):
        raise ValueError("online cosine loss requires at least one mined negative pair")
    positive_pairs = torch.stack(
        (graph.edge_left_rows, graph.edge_right_rows)
    ).to(node_embeddings.device)
    negative_pairs = torch.stack(
        (graph.negative_left_rows, graph.negative_right_rows)
    ).to(node_embeddings.device)
    positive_loss, negative_loss = cosine_softplus_pair_losses(
        node_embeddings,
        positive_pairs,
        negative_pairs,
        temperature=temperature,
    )
    return positive_loss + negative_weight * negative_loss


def compute_ae_update_losses(model: AE, rss_input: Tensor, updated_latent_target: Tensor) -> dict[str, Tensor]:
    """Compute the three printed Eq. (10) Frobenius terms and their total."""
    if rss_input.ndim != 3 or rss_input.shape[1:] != (1, model.input_dim):
        raise ValueError("Eq. (10) RSS input must be [N,1,A]")
    if updated_latent_target.shape != (rss_input.shape[0], model.config.latent_dim):
        raise ValueError("Eq. (10) latent target width must match AE checkpoint")
    if updated_latent_target.requires_grad:
        raise ValueError("Eq. (10) feature target must be fixed/detached")
    encoded, reconstruction = model(rss_input)
    encoded = encoded[:, 0, :]
    target_reconstruction = model.D(updated_latent_target.unsqueeze(1))
    reconstruction_term = torch.linalg.vector_norm(reconstruction - rss_input)
    latent_term = torch.linalg.vector_norm(encoded - updated_latent_target)
    decoder_consistency_term = torch.linalg.vector_norm(
        target_reconstruction - reconstruction
    )
    total = reconstruction_term + 0.5 * (latent_term + decoder_consistency_term)
    result = {
        "reconstruction": reconstruction_term,
        "latent_consistency": latent_term,
        "decoder_consistency": decoder_consistency_term,
        "total": total,
    }
    if any(not torch.isfinite(value).item() for value in result.values()):
        raise FloatingPointError("AE Eq. (10) loss contains a non-finite value")
    return result


def _tensor_ids(graph: OnlineUpdateGraph, ids: tuple[str, ...], features: Tensor) -> Tensor:
    row_by_id = dict(graph.node_id_to_row)
    return features[torch.tensor([row_by_id[node_id] for node_id in ids], dtype=torch.long)]


def _coordinate_training_tensors(context: FingerprintUpdateContext) -> tuple[Tensor, Tensor]:
    return context.location_scaler.encode(context.reference_coordinates), context.adjacency


def _seeded_clone_gnn(loaded: LoadedGNNCheckpoint, seed: int) -> EdgeFeatureSAGE:
    cloned = EdgeFeatureSAGE(loaded.manifest.model_config_record)
    cloned.load_state_dict(copy.deepcopy(loaded.model.state_dict()), strict=True)
    cloned._inference_only = False
    cloned._training_started = False
    cloned._training_completed = False
    return cloned


def _train_online_gnn(
    context: FingerprintUpdateContext,
    source_gnn: LoadedGNNCheckpoint,
    config: FingerprintUpdateConfig,
    *,
    device: torch.device,
    log_every_epochs: int = DEFAULT_LOG_EVERY_K_EPOCHS,
) -> tuple[EdgeFeatureSAGE, tuple[float, ...], dict]:
    model = _seeded_clone_gnn(source_gnn, config.seed).to(device=device, dtype=torch.float32)
    graph = context.tensors
    graph = GraphTensorInput(
        node_features=graph.node_features.to(device),
        edge_features=graph.edge_features.to(device),
        edge_left_rows=graph.edge_left_rows.to(device),
        edge_right_rows=graph.edge_right_rows.to(device),
        radio_edge_count=graph.radio_edge_count,
        node_ids=graph.node_ids,
        edge_ids=graph.edge_ids,
        radio_edge_ids=graph.radio_edge_ids,
        virtual_edge_ids=graph.virtual_edge_ids,
        negative_left_rows=(
            graph.negative_left_rows.to(device)
            if graph.negative_left_rows is not None else None
        ),
        negative_right_rows=(
            graph.negative_right_rows.to(device)
            if graph.negative_right_rows is not None else None
        ),
        negative_pair_strengths=(
            graph.negative_pair_strengths.to(device)
            if graph.negative_pair_strengths is not None else None
        ),
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.gnn_learning_rate, weight_decay=config.gnn_weight_decay
    )
    history: list[float] = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed + 101)
        model.train()
        for _epoch in range(config.gnn_epochs):
            optimizer.zero_grad(set_to_none=True)
            output = model(graph)
            loss = equation4_online_loss(
                output.node_features,
                graph,
                temperature=config.contrastive_temperature,
                negative_weight=config.negative_weight,
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError("online GNN cosine loss is non-finite")
            loss.backward()
            grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not grads or not any(torch.count_nonzero(grad).item() for grad in grads):
                raise RuntimeError("online GNN cosine loss produced no parameter gradient")
            if any(not torch.isfinite(grad).all().item() for grad in grads):
                raise FloatingPointError("online GNN gradient is non-finite")
            optimizer.step()
            history.append(float(loss.detach().cpu()))
            log_training_epoch(
                "Batch GNN",
                _epoch + 1,
                config.gnn_epochs,
                {
                    "loss": history[-1],
                    "nodes": len(graph.node_ids),
                    "radio_edges": graph.radio_edge_count,
                    "virtual_edges": len(graph.virtual_edge_ids),
                    "negative_pairs": (
                        0 if graph.negative_left_rows is None
                        else int(graph.negative_left_rows.numel())
                    ),
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                every_k_epochs=log_every_epochs,
            )
    model.eval()
    model._training_started = True
    model._training_completed = True
    # This GNN state is a selected batch model. It is not optimizer-resumable.
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    return model, tuple(history), optimizer.state_dict()


def _train_updaters(
    context: FingerprintUpdateContext,
    gnn_model: EdgeFeatureSAGE,
    config: FingerprintUpdateConfig,
    *,
    device: torch.device,
    initial_state: tuple[dict[str, Tensor], dict[str, Tensor]] | None = None,
    log_every_epochs: int = DEFAULT_LOG_EVERY_K_EPOCHS,
) -> tuple[LocationPredictor, LocationToFeatureUpdater, tuple[UpdateEpochLoss, ...], dict, Tensor]:
    online_tensors = context.tensors
    graph = GraphTensorInput(
        node_features=online_tensors.node_features.to(device),
        edge_features=online_tensors.edge_features.to(device),
        edge_left_rows=online_tensors.edge_left_rows.to(device),
        edge_right_rows=online_tensors.edge_right_rows.to(device),
        radio_edge_count=online_tensors.radio_edge_count,
        node_ids=online_tensors.node_ids,
        edge_ids=online_tensors.edge_ids,
        radio_edge_ids=online_tensors.radio_edge_ids,
        virtual_edge_ids=online_tensors.virtual_edge_ids,
    )
    with torch.no_grad():
        gnn_output = gnn_model(graph).node_features.detach()
    reference_features = _tensor_ids(
        context.online_graph,
        context.labeled_reference_ids,
        gnn_output,
    )
    update_features = _tensor_ids(
        context.online_graph,
        context.online_graph.update_sample_ids,
        gnn_output,
    )
    y_ref, adjacency = _coordinate_training_tensors(context)
    y_ref = y_ref.to(device)
    adjacency = adjacency.to(device)
    z_update = context.update_encoder_targets.to(device).detach()
    location_model = LocationPredictor(
        config.hidden_dim, context.online_graph.node_feature_dim
    ).to(device)
    feature_model = LocationToFeatureUpdater(
        config.hidden_dim, context.online_graph.encoder_feature_dim
    ).to(device)
    if initial_state is not None:
        location_model.load_state_dict(initial_state[0], strict=True)
        feature_model.load_state_dict(initial_state[1], strict=True)
    optimizer = torch.optim.Adam(
        list(location_model.parameters()) + list(feature_model.parameters()),
        lr=config.updater_learning_rate,
    )
    history: list[UpdateEpochLoss] = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed + 202)
        for epoch in range(1, config.updater_epochs + 1):
            location_model.train()
            feature_model.train()
            optimizer.zero_grad(set_to_none=True)
            losses, _y_update, _z_reference = compute_update_losses(
                location_model,
                feature_model,
                reference_features,
                update_features,
                y_ref,
                z_update,
                adjacency,
                alpha=config.alpha,
            )
            losses["total"].backward()
            model_grads = [
                parameter.grad
                for model in (location_model, feature_model)
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if len(model_grads) == 0 or any(
                not torch.isfinite(grad).all().item() for grad in model_grads
            ):
                raise FloatingPointError("updater MLP gradients are missing or non-finite")
            if not any(torch.count_nonzero(grad).item() for grad in model_grads):
                raise RuntimeError("updater MLP loss produced no gradient signal")
            optimizer.step()
            scalar_losses = {
                key: float(value.detach().cpu()) for key, value in losses.items()
            }
            history.append(UpdateEpochLoss(epoch=epoch, **scalar_losses))
            log_training_epoch(
                "Location/feature updaters",
                epoch,
                config.updater_epochs,
                {
                    **scalar_losses,
                    "labeled_refs": len(context.labeled_reference_ids),
                    "update_scans": len(context.online_graph.update_sample_ids),
                },
                every_k_epochs=log_every_epochs,
            )
    location_model.eval()
    feature_model.eval()
    with torch.no_grad():
        reference_z = feature_model(y_ref).detach().clone()
    return location_model, feature_model, tuple(history), optimizer.state_dict(), reference_z


def _train_update_ae(
    context: FingerprintUpdateContext,
    feature_targets: Tensor,
    config: FingerprintUpdateConfig,
    *,
    device: torch.device,
    log_every_epochs: int = DEFAULT_LOG_EVERY_K_EPOCHS,
) -> tuple[AE, tuple[float, ...], dict]:
    loaded = load_ae_checkpoint(context.ae_checkpoint_path, context.reference_prepared, device=device)
    model = copy.deepcopy(loaded.model).to(device=device, dtype=torch.float32)
    target_z = feature_targets.detach().to(device=device, dtype=torch.float32)
    x = context.reference_ae_inputs.to(device=device, dtype=torch.float32)
    if x.ndim != 3 or x.shape[1] != 1 or x.shape[2] != len(context.online_graph.bssid_vocabulary):
        raise ValueError("Eq. (10) reference RSS input must have shape [N_labeled,1,A]")
    if target_z.shape != (x.shape[0], model.config.latent_dim):
        raise ValueError("Eq. (10) feature targets must match the loaded AE latent width")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.ae_learning_rate,
        weight_decay=config.ae_weight_decay,
    )
    history: list[float] = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed + 303)
        for _epoch in range(config.ae_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = compute_ae_update_losses(model, x, target_z)
            loss = losses["total"]
            loss.backward()
            grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not grads or any(not torch.isfinite(grad).all().item() for grad in grads):
                raise FloatingPointError("AE Eq. (10) gradients are missing or non-finite")
            if not any(torch.count_nonzero(grad).item() for grad in grads):
                raise RuntimeError("AE Eq. (10) produced no parameter gradient")
            optimizer.step()
            history.append(float(loss.detach().cpu()))
            log_training_epoch(
                "Batch AE Eq. (10)",
                _epoch + 1,
                config.ae_epochs,
                {
                    **{
                        key: float(value.detach().cpu())
                        for key, value in losses.items()
                    },
                    "reference_scans": x.shape[0],
                    "aps": x.shape[2],
                },
                every_k_epochs=log_every_epochs,
            )
    model.eval()
    return model, tuple(history), optimizer.state_dict()


def _build_manifest(
    context: FingerprintUpdateContext,
    config: FingerprintUpdateConfig,
    location_scaler: UpdateLocationScaler,
    losses: tuple[UpdateEpochLoss, ...],
    gnn_history: tuple[float, ...],
    ae_history: tuple[float, ...],
    gnn_model: EdgeFeatureSAGE,
    *,
    device: torch.device,
    source_update_checkpoint_sha256: str | None,
) -> FingerprintUpdateManifest:
    adjacency_bytes = context.adjacency.cpu().contiguous().numpy().tobytes()
    source_gnn = load_gnn_checkpoint(context.gnn_checkpoint_path, context.reference_graph)
    return FingerprintUpdateManifest(
        run_id=uuid4().hex,
        config=config,
        dataset_id=context.reference_prepared.config.dataset_id,
        reference_batch_id=context.reference_prepared.config.batch_id,
        update_batch_id=context.update_prepared.config.batch_id,
        reference_prepared_sha256=context.reference_prepared_sha256,
        update_prepared_sha256=context.update_prepared_sha256,
        reference_graph_id=context.reference_graph.graph_id,
        update_graph_id=context.online_graph.update_graph_id,
        online_graph_id=context.online_graph.graph_id,
        current_reference_snapshot_id=context.online_graph.current_reference_snapshot_id,
        reference_encoder_artifact_sha256=context.reference_encoder_artifact_sha256,
        update_encoder_artifact_sha256=context.update_encoder_artifact_sha256,
        reference_gnn_artifact_sha256=context.reference_gnn_artifact_sha256,
        source_ae_checkpoint_sha256=context.source_ae_checkpoint_sha256,
        source_gnn_checkpoint_sha256=context.source_gnn_checkpoint_sha256,
        vocabulary_sha256=context.reference_graph.vocabulary_sha256,
        bssid_vocabulary=context.reference_graph.bssid_vocabulary,
        normalization=context.reference_graph.normalization,
        encoder_feature_dim=context.online_graph.encoder_feature_dim,
        graph_feature_dim=context.online_graph.node_feature_dim,
        reference_node_ids=context.reference_graph.sample_node_ids,
        update_node_ids=context.online_graph.update_sample_ids,
        labeled_reference_ids=context.labeled_reference_ids,
        adjacency_shape=tuple(context.adjacency.shape),
        adjacency_sha256=hashlib.sha256(adjacency_bytes).hexdigest(),
        location_scaler=location_scaler,
        loss_history=losses,
        gnn_loss_history=gnn_history,
        ae_loss_history=ae_history,
        gnn_model_config=source_gnn.manifest.model_config_record.model_dump(mode="json"),
        updater_initialization=(
            "weights_only_previous_batch"
            if source_update_checkpoint_sha256 is not None
            else "seeded_first_batch"
        ),
        source_update_checkpoint_sha256=source_update_checkpoint_sha256,
        training_device=str(device),
        torch_version=torch.__version__,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def _checkpoint_payload(
    manifest: FingerprintUpdateManifest,
    location_model: LocationPredictor,
    feature_model: LocationToFeatureUpdater,
    gnn_model: EdgeFeatureSAGE,
    ae_model: AE,
    optimizer_states: dict,
) -> dict:
    def cpu_state(module: nn.Module) -> dict[str, Tensor]:
        return {
            key: value.detach().cpu().contiguous().clone()
            for key, value in module.state_dict().items()
        }

    return {
        "format": UPDATE_CHECKPOINT_FORMAT,
        "manifest_json": manifest.model_dump_json(),
        "location_model_state": cpu_state(location_model),
        "feature_model_state": cpu_state(feature_model),
        "gnn_model_state": cpu_state(gnn_model),
        "ae_encoder_state": cpu_state(ae_model.E),
        "ae_decoder_state": cpu_state(ae_model.D),
        "optimizer_states": optimizer_states,
        "resumable": False,
    }


def _predict_artifact(
    context: FingerprintUpdateContext,
    models: FingerprintUpdaterModels,
) -> UpdatedFingerprintArtifact:
    graph = context.tensors
    device = next(models.gnn_model.parameters()).device
    graph = GraphTensorInput(
        node_features=graph.node_features.to(device),
        edge_features=graph.edge_features.to(device),
        edge_left_rows=graph.edge_left_rows.to(device),
        edge_right_rows=graph.edge_right_rows.to(device),
        radio_edge_count=graph.radio_edge_count,
        node_ids=graph.node_ids,
        edge_ids=graph.edge_ids,
        radio_edge_ids=graph.radio_edge_ids,
        virtual_edge_ids=graph.virtual_edge_ids,
    )
    models.gnn_model.eval()
    models.location_model.eval()
    models.feature_model.eval()
    models.ae_model.eval()
    with torch.no_grad():
        online_gnn = models.gnn_model(graph).node_features
        row_by_id = dict(context.online_graph.node_id_to_row)
        update_rows = torch.tensor(
            [row_by_id[scan_id] for scan_id in context.online_graph.update_sample_ids],
            dtype=torch.long,
            device=device,
        )
        predicted_scaled_xy = models.location_model(
            online_gnn.index_select(0, update_rows)
        )
        predicted_xy = context.location_scaler.decode(predicted_scaled_xy)
        reference_xy = context.reference_coordinates.to(device)
        scaled_xy = context.location_scaler.encode(reference_xy)
        update_z = models.feature_model(scaled_xy)
        decoded = models.ae_model.D(update_z.unsqueeze(1))[:, 0, :]
        normalized = decoded.cpu().tolist()
        latent_rows = update_z.cpu().tolist()
        predicted_rows = predicted_xy.cpu().tolist()

    original_positions = {
        position.scan_id: (position.x, position.y)
        for position in context.reference_prepared.reference_positions
    }
    labeled_id_set = set(context.labeled_reference_ids)
    rows: list[FingerprintUpdateRow] = []
    for scan in context.reference_prepared.scans:
        if scan.scan_id in labeled_id_set:
            decoded_row = normalized[context.labeled_reference_ids.index(scan.scan_id)]
            output_values = tuple(
                ae_value_to_raw_dbm(value, context.reference_prepared.config.normalization_mode)
                for value in decoded_row
            )
            provenance = ("decoder_prediction",) * len(output_values)
        else:
            # A bootstrap row without a known position cannot be fed to MLP 2;
            # retain its source fingerprint, with its original mask visible.
            output_values = tuple(scan.rss_dbm)
            provenance = ("retained_original",) * len(output_values)
        rows.append(
            FingerprintUpdateRow(
                scan_id=scan.scan_id,
                original_location=original_positions.get(scan.scan_id),
                raw_rssi_dbm=scan.rss_dbm,
                observation_mask=scan.observation_mask,
                updated_rssi_dbm=output_values,
                updated_value_provenance=provenance,
            )
        )
    return UpdatedFingerprintArtifact(
        run_id=models.manifest.run_id,
        dataset_id=context.reference_prepared.config.dataset_id,
        reference_batch_id=context.reference_prepared.config.batch_id,
        update_batch_id=context.update_prepared.config.batch_id,
        reference_prepared_sha256=context.reference_prepared_sha256,
        update_prepared_sha256=context.update_prepared_sha256,
        reference_graph_id=context.reference_graph.graph_id,
        update_graph_id=context.online_graph.update_graph_id,
        online_graph_id=context.online_graph.graph_id,
        source_checkpoint_sha256=models.checkpoint_sha256,
        source_ae_checkpoint_sha256=context.source_ae_checkpoint_sha256,
        source_gnn_checkpoint_sha256=context.source_gnn_checkpoint_sha256,
        vocabulary_sha256=context.reference_graph.vocabulary_sha256,
        bssid_vocabulary=context.reference_graph.bssid_vocabulary,
        normalization=context.reference_prepared.config.normalization_mode,
        latent_feature_dim=context.online_graph.encoder_feature_dim,
        reference_scan_ids=tuple(scan.scan_id for scan in context.reference_prepared.scans),
        update_scan_ids=context.online_graph.update_sample_ids,
        rows=tuple(rows),
        predicted_update_locations=tuple(
            PredictedUpdateLocation(scan_id=scan_id, x=xy[0], y=xy[1])
            for scan_id, xy in zip(context.online_graph.update_sample_ids, predicted_rows)
        ),
        updated_latent_reference_ids=context.labeled_reference_ids,
        updated_latent_reference_features=tuple(
            tuple(float(value) for value in row) for row in latent_rows
        ),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def train_fingerprint_update(
    context: FingerprintUpdateContext,
    checkpoint_path: str | Path,
    artifact_path: str | Path,
    config: FingerprintUpdateConfig | None = None,
    *,
    device: str | torch.device = "cpu",
    initial_updater_checkpoint_path: str | Path | None = None,
    log_every_epochs: int = DEFAULT_LOG_EVERY_K_EPOCHS,
) -> FingerprintUpdateTrainingResult:
    """Train both MLPs and batch-update GNN+AE; save distinct artifacts."""
    selected = config or FingerprintUpdateConfig()
    log_every_epochs = validate_log_interval(log_every_epochs)
    checkpoint_target, artifact_target = Path(checkpoint_path), Path(artifact_path)
    if checkpoint_target.exists():
        raise FileExistsError(f"refusing to overwrite update checkpoint: {checkpoint_target}")
    if artifact_target.exists():
        raise FileExistsError(f"refusing to overwrite fingerprint result: {artifact_target}")
    checkpoint_target.parent.mkdir(parents=True, exist_ok=True)
    artifact_target.parent.mkdir(parents=True, exist_ok=True)
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"requested CUDA device is unavailable: {selected_device}")
    initial_state = None
    source_update_sha256 = None
    if initial_updater_checkpoint_path is not None:
        source_update_sha256 = sha256_file(initial_updater_checkpoint_path)
        prior_payload = load_torch_state(initial_updater_checkpoint_path, map_location="cpu")
        if prior_payload.get("format") != UPDATE_CHECKPOINT_FORMAT:
            raise ValueError("previous updater initialization checkpoint has an unsupported format")
        try:
            prior_manifest = FingerprintUpdateManifest.model_validate_json(
                prior_payload["manifest_json"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"previous updater initialization manifest is invalid: {exc}") from exc
        if prior_manifest.dataset_id != context.reference_prepared.config.dataset_id:
            raise ValueError("previous updater checkpoint belongs to a different dataset")
        if prior_manifest.reference_batch_id != context.reference_prepared.config.batch_id:
            raise ValueError("previous updater checkpoint belongs to a different reference batch")
        if prior_manifest.normalization != context.reference_prepared.config.normalization_mode:
            raise ValueError("previous updater checkpoint uses a different normalization")
        if prior_manifest.config.hidden_dim != selected.hidden_dim:
            raise ValueError("previous updater hidden width differs from this batch configuration")
        if not prior_manifest.bssid_vocabulary:
            raise ValueError("previous updater checkpoint has no vocabulary provenance")
        for name in ("location_model_state", "feature_model_state"):
            if name not in prior_payload:
                raise ValueError(f"previous updater checkpoint is missing {name}")
        initial_state = (
            prior_payload["location_model_state"],
            prior_payload["feature_model_state"],
        )
    source_gnn = load_gnn_checkpoint(context.gnn_checkpoint_path, context.reference_graph, device=selected_device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(selected.seed)
        gnn_model, gnn_history, gnn_optimizer_state = _train_online_gnn(
            context,
            source_gnn,
            selected,
            device=selected_device,
            log_every_epochs=log_every_epochs,
        )
        location_model, feature_model, updater_history, updater_optimizer_state, z_target = _train_updaters(
            context,
            gnn_model,
            selected,
            device=selected_device,
            initial_state=initial_state,
            log_every_epochs=log_every_epochs,
        )
        ae_model, ae_history, ae_optimizer_state = _train_update_ae(
            context,
            z_target,
            selected,
            device=selected_device,
            log_every_epochs=log_every_epochs,
        )
    manifest = _build_manifest(
        context,
        selected,
        context.location_scaler,
        updater_history,
        gnn_history,
        ae_history,
        gnn_model,
        device=selected_device,
        source_update_checkpoint_sha256=source_update_sha256,
    )
    optimizer_states = {
        "online_gnn_adam": gnn_optimizer_state,
        "updater_adam": updater_optimizer_state,
        "updated_ae_adam": ae_optimizer_state,
        "torch_cpu_rng_state": torch.get_rng_state().cpu().clone(),
    }
    payload = _checkpoint_payload(
        manifest, location_model, feature_model, gnn_model, ae_model, optimizer_states
    )
    atomic_torch_save(payload, checkpoint_target)
    checkpoint_sha = sha256_file(checkpoint_target)
    loaded_models = load_fingerprint_update_checkpoint(checkpoint_target, context, device=selected_device)
    artifact = _predict_artifact(context, loaded_models)
    atomic_write_json(artifact, artifact_target)
    return FingerprintUpdateTrainingResult(loaded_models, artifact, artifact_target)


def load_fingerprint_update_checkpoint(
    checkpoint_path: str | Path,
    context: FingerprintUpdateContext,
    *,
    device: str | torch.device = "cpu",
) -> FingerprintUpdaterModels:
    """Load all selected online model weights after validating source identity."""
    path = Path(checkpoint_path)
    payload = load_torch_state(path, map_location="cpu")
    if payload.get("format") != UPDATE_CHECKPOINT_FORMAT:
        raise ValueError("unsupported fingerprint update checkpoint format")
    try:
        manifest = FingerprintUpdateManifest.model_validate_json(payload["manifest_json"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid fingerprint update manifest: {exc}") from exc
    checks = (
        (manifest.reference_prepared_sha256 == context.reference_prepared_sha256, "reference prepared data"),
        (manifest.update_prepared_sha256 == context.update_prepared_sha256, "update prepared data"),
        (manifest.reference_graph_id == context.reference_graph.graph_id, "reference graph"),
        (manifest.update_graph_id == context.online_graph.update_graph_id, "update graph"),
        (manifest.online_graph_id == context.online_graph.graph_id, "online graph"),
        (manifest.source_ae_checkpoint_sha256 == context.source_ae_checkpoint_sha256, "AE checkpoint"),
        (manifest.source_gnn_checkpoint_sha256 == context.source_gnn_checkpoint_sha256, "GNN checkpoint"),
        (manifest.vocabulary_sha256 == context.reference_graph.vocabulary_sha256, "vocabulary"),
        (manifest.adjacency_sha256 == hashlib.sha256(context.adjacency.cpu().contiguous().numpy().tobytes()).hexdigest(), "adjacency"),
    )
    for compatible, label in checks:
        if not compatible:
            raise ValueError(f"fingerprint update checkpoint is incompatible with {label}")
    if payload.get("resumable") is not False or manifest.resumable:
        raise ValueError("this update artifact supports inference, not optimizer continuation")
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"requested CUDA device is unavailable: {selected_device}")
    if (
        manifest.encoder_feature_dim != context.online_graph.encoder_feature_dim
        or manifest.graph_feature_dim != context.online_graph.node_feature_dim
    ):
        raise ValueError("fingerprint update checkpoint is incompatible with feature dimensions")
    location_model = LocationPredictor(
        manifest.config.hidden_dim, manifest.graph_feature_dim
    )
    feature_model = LocationToFeatureUpdater(
        manifest.config.hidden_dim, manifest.encoder_feature_dim
    )
    gnn_config = load_gnn_checkpoint(
        context.gnn_checkpoint_path, context.reference_graph
    ).manifest.model_config_record
    gnn_model = EdgeFeatureSAGE(gnn_config)
    ae_model = load_ae_checkpoint(
        context.ae_checkpoint_path,
        context.reference_prepared,
        device=selected_device,
    ).model
    for key in (
        "location_model_state",
        "feature_model_state",
        "gnn_model_state",
        "ae_encoder_state",
        "ae_decoder_state",
    ):
        if key not in payload:
            raise ValueError(f"fingerprint update checkpoint is missing {key}")
    location_model.load_state_dict(payload["location_model_state"], strict=True)
    feature_model.load_state_dict(payload["feature_model_state"], strict=True)
    gnn_model.load_state_dict(payload["gnn_model_state"], strict=True)
    ae_model.E.load_state_dict(payload["ae_encoder_state"], strict=True)
    ae_model.D.load_state_dict(payload["ae_decoder_state"], strict=True)
    location_model.to(selected_device).eval()
    feature_model.to(selected_device).eval()
    gnn_model.to(selected_device).eval()
    ae_model.to(selected_device).eval()
    gnn_model._training_completed = True
    gnn_model._inference_only = True
    optimizer_states = payload.get("optimizer_states")
    if not isinstance(optimizer_states, dict) or not all(
        key in optimizer_states for key in ("online_gnn_adam", "updater_adam", "updated_ae_adam")
    ):
        raise ValueError("update checkpoint must retain all three optimizer states")
    return FingerprintUpdaterModels(
        location_model=location_model,
        feature_model=feature_model,
        gnn_model=gnn_model,
        ae_model=ae_model,
        manifest=manifest,
        checkpoint_path=path,
        checkpoint_sha256=sha256_file(path),
        optimizer_states=optimizer_states,
    )


def predict_updated_fingerprints(
    context: FingerprintUpdateContext,
    models: FingerprintUpdaterModels,
    artifact_path: str | Path | None = None,
) -> UpdatedFingerprintArtifact:
    """Run inference from a loaded update checkpoint; never retrain implicitly."""
    artifact = _predict_artifact(context, models)
    if artifact_path is not None:
        atomic_write_json(artifact, artifact_path)
    return artifact


def load_updated_fingerprints(path: str | Path) -> UpdatedFingerprintArtifact:
    return UpdatedFingerprintArtifact.model_validate_json(read_json_text(path))


__all__ = [
    "FingerprintUpdateConfig",
    "FingerprintUpdateContext",
    "FingerprintUpdateManifest",
    "FingerprintUpdateTrainingResult",
    "LocationPredictor",
    "LocationToFeatureUpdater",
    "OnlineUpdateGraph",
    "UpdateLocationScaler",
    "UpdatedFingerprintArtifact",
    "PredictedUpdateLocation",
    "compute_ae_update_losses",
    "compute_update_losses",
    "cross_virtual_adjacency",
    "load_fingerprint_update_checkpoint",
    "load_updated_fingerprints",
    "predict_updated_fingerprints",
    "prepare_fingerprint_update",
    "train_fingerprint_update",
]
