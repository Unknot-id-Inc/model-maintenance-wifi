"""Identity-preserving AP and batch vocabulary transitions for GUFU.

This module owns the AP-set comparison, old-vocabulary projection, transient
new-AP graph extension, AP edge decisions, AE parameter-slice migration, and
versioned reference snapshot. It does not turn an absent reading into an AP
deletion and does not accept fabricated or row-position-aligned features.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from ae import (
    AE,
    AEVocabularyMigrationRecord,
    EncoderFeatureArtifact,
    LoadedAECheckpoint,
    load_ae_checkpoint,
    migrate_ae_checkpoint_vocabulary,
)
from fingerprint_update import (
    FingerprintUpdateConfig,
    FingerprintUpdateContext,
    FingerprintUpdaterModels,
    UpdatedFingerprintArtifact,
    compute_ae_update_losses,
)
from graph_build import GraphBuildConfig, GraphState
from gufu_artifacts import (
    atomic_torch_save,
    atomic_write_json,
    canonical_json_digest,
    load_torch_state,
    sha256_file,
)
from gufu_data import (
    PreparedDataset,
    PreparedScan,
    PreprocessingConfig,
    PreparationStats,
    ReferencePosition,
    EvaluationPosition,
    ap_id_for_bssid,
    ae_value_to_raw_dbm,
    canonicalize_bssid,
    raw_dbm_row_to_ae,
)
from lp import (
    CandidateEdgeScore,
    LinkEdgeChange,
    LinkPredictionConfig,
    UpdatedGraphSnapshot,
    score_candidate_pairs_from_features,
)
from sage_ve import GNNFeatureArtifact, GraphTensorInput
from training_progress import DEFAULT_LOG_EVERY_K_EPOCHS, log_training_epoch, validate_log_interval


TRANSITION_STATE_FORMAT = "gufu_ap_transition_state_v1"
TRANSITION_RESULT_FORMAT = "gufu_ap_transition_result_v1"
TRANSITION_MODEL_FORMAT = "gufu_ap_transition_models_v1"
FEATURE_DIM = 34
LATENT_DIM = 32


class NoSharedAPError(ValueError):
    """The incoming batch has no AP that the current encoder can consume."""


class TransitionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    link_prediction: LinkPredictionConfig = LinkPredictionConfig()
    ae_finetune_epochs: int = Field(default=5, ge=1)
    ae_finetune_seed: int = 1701
    reinitialize_seed: int = 1702
    ae_finetune_learning_rate: float = Field(default=0.001, gt=0.0)
    empty_shared_ap_policy: Literal["error"] = "error"
    vocabulary_order_policy: Literal[
        "preserve_active_then_append_new_and_returning_lexical_v1"
    ] = "preserve_active_then_append_new_and_returning_lexical_v1"
    cache_policy: Literal["invalidate_encoder_and_gnn_after_transition_v1"] = (
        "invalidate_encoder_and_gnn_after_transition_v1"
    )

    @model_validator(mode="after")
    def finite_learning_rate(self):
        if not math.isfinite(self.ae_finetune_learning_rate):
            raise ValueError("AE fine-tuning learning rate must be finite")
        return self


class APRegistryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    bssid: str
    status: Literal["active", "tombstoned", "observed_unlinked"]
    first_seen_batch_id: str
    last_seen_batch_id: str
    removed_by_batch_id: str | None = None

    @model_validator(mode="after")
    def canonical_and_status(self):
        if canonicalize_bssid(self.bssid) != self.bssid:
            raise ValueError("AP registry must use full canonical BSSID identities")
        if (self.status == "tombstoned") != (self.removed_by_batch_id is not None):
            raise ValueError("only tombstoned APs carry a forgetting batch ID")
        return self


class BatchLineageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    batch_id: str
    scan_ids: tuple[str, ...]
    prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_state_id: str | None = None

    @model_validator(mode="after")
    def unique_scans(self):
        if len(set(self.scan_ids)) != len(self.scan_ids):
            raise ValueError("batch lineage contains duplicate scan IDs")
        return self


class TransitionRadioEdge(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    edge_id: str
    sample_node_id: str
    bssid: str
    scalar_weight: float = Field(ge=0.0)
    edge_feature: tuple[float, ...]
    source: Literal[
        "retained_gnn_edge",
        "new_ap_endpoint_mean",
        "retained_endpoint_mean",
    ]

    @model_validator(mode="after")
    def valid_identity_and_feature(self):
        if canonicalize_bssid(self.bssid) != self.bssid:
            raise ValueError("radio edge must carry a full canonical BSSID")
        if self.edge_id != f"radio:{self.sample_node_id}|{ap_id_for_bssid(self.bssid)}":
            raise ValueError("transition edge ID must be endpoint stable")
        if not math.isfinite(self.scalar_weight):
            raise ValueError("transition radio weight must be finite")
        if not self.edge_feature or any(
            not math.isfinite(value) for value in self.edge_feature
        ):
            raise ValueError("transition radio edge feature must be finite and nonempty")
        return self


class TransitionReferenceLocation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    scan_id: str
    xy: tuple[float, float] | None

    @model_validator(mode="after")
    def finite_xy(self):
        if self.xy is not None and any(not math.isfinite(value) for value in self.xy):
            raise ValueError("reference coordinate must be finite")
        return self


class VocabularyTransitionState(BaseModel):
    """Immutable current reference values, AP registry, graph rows, and lineage."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_ap_transition_state_v1"] = TRANSITION_STATE_FORMAT
    state_id: str = Field(pattern=r"^transition:[0-9a-f]{64}$")
    parent_state_id: str | None = None
    dataset_id: str
    reference_batch_id: str
    source_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preprocessing_config: PreprocessingConfig
    graph_build_config: GraphBuildConfig
    normalization: Literal["paper_ae", "legacy_raw_div_120"]
    encoder_feature_dim: int = Field(default=LATENT_DIM, ge=1)
    graph_feature_dim: int = Field(default=FEATURE_DIM, ge=3)
    vocabulary_revision: int = Field(ge=0)
    bssid_vocabulary: tuple[str, ...]
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ap_registry: tuple[APRegistryRecord, ...]
    reference_scan_ids: tuple[str, ...]
    reference_locations: tuple[TransitionReferenceLocation, ...]
    reference_rssi_dbm: tuple[tuple[float, ...], ...]
    reference_observation_mask: tuple[tuple[bool, ...], ...]
    reference_value_provenance: tuple[
        tuple[Literal["measured", "decoder_prediction", "edge_prediction", "missing"], ...], ...
    ]
    graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    reference_graph_id: str = Field(pattern=r"^(graph|lpgraph):[0-9a-f]{64}$")
    gnn_feature_graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    node_ids: tuple[str, ...]
    node_types: tuple[Literal["sample", "ap"], ...]
    graph_ap_node_ids: tuple[str, ...]
    graph_input_node_features: tuple[tuple[float, ...], ...]
    graph_node_features: tuple[tuple[float, ...], ...]
    radio_edges: tuple[TransitionRadioEdge, ...]
    ae_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gnn_state_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_feature_artifact_sha256: str | None = None
    gnn_feature_artifact_sha256: str | None = None
    encoder_cache_status: Literal["valid", "invalidated_requires_regeneration"]
    gnn_cache_status: Literal["valid", "invalidated_requires_regeneration"]
    batch_history: tuple[BatchLineageRecord, ...] = ()
    created_at_utc: str

    @model_validator(mode="after")
    def validate_state(self):
        if canonical_json_digest(self.bssid_vocabulary) != self.vocabulary_sha256:
            raise ValueError("transition vocabulary digest is inconsistent")
        if self.preprocessing_config.dataset_id != self.dataset_id or self.preprocessing_config.batch_role != "bootstrap_reference":
            raise ValueError("transition preprocessing provenance must describe the reference dataset")
        if self.preprocessing_config.normalization_mode != self.normalization:
            raise ValueError("transition normalization differs from preprocessing metadata")
        if len(set(self.bssid_vocabulary)) != len(self.bssid_vocabulary) or any(
            canonicalize_bssid(value) != value for value in self.bssid_vocabulary
        ):
            raise ValueError("transition vocabulary must contain unique full BSSIDs")
        scan_count = len(self.reference_scan_ids)
        width = len(self.bssid_vocabulary)
        if self.graph_feature_dim != self.encoder_feature_dim + 2:
            raise ValueError("transition graph width must equal encoder width plus 2-D coordinates")
        if len(set(self.reference_scan_ids)) != scan_count:
            raise ValueError("reference scan IDs must be unique")
        if tuple(item.scan_id for item in self.reference_locations) != self.reference_scan_ids:
            raise ValueError("reference locations must align to original reference scan IDs")
        for matrix in (
            self.reference_rssi_dbm,
            self.reference_observation_mask,
            self.reference_value_provenance,
        ):
            if len(matrix) != scan_count or any(len(row) != width for row in matrix):
                raise ValueError("reference RSS, masks, provenance and vocabulary dimensions differ")
        if any(not math.isfinite(value) for row in self.reference_rssi_dbm for value in row):
            raise ValueError("transition RSS values must be finite")
        for row, mask, provenance in zip(
            self.reference_rssi_dbm,
            self.reference_observation_mask,
            self.reference_value_provenance,
        ):
            for value, measured, origin in zip(row, mask, provenance):
                if origin == "measured" and not measured:
                    raise ValueError("measured provenance requires an observed reading")
                if origin == "missing" and (measured or value != -120.0):
                    raise ValueError("missing transition values require -120 dBm and mask=False")
        registry_ids = tuple(record.bssid for record in self.ap_registry)
        if len(set(registry_ids)) != len(registry_ids):
            raise ValueError("AP registry identities must be unique")
        if registry_ids != tuple(sorted(registry_ids)):
            raise ValueError("AP registry uses deterministic full-BSSID order")
        if not set(self.bssid_vocabulary) <= set(registry_ids):
            raise ValueError("model vocabulary identities must exist in the AP registry")
        if self.node_ids != self.reference_scan_ids + self.graph_ap_node_ids:
            raise ValueError("persistent graph rows must be reference samples followed by AP nodes")
        if self.node_types != ("sample",) * scan_count + ("ap",) * len(self.graph_ap_node_ids):
            raise ValueError("persistent graph node types must match ID order")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("persistent graph node IDs must be unique")
        if len(self.graph_input_node_features) != len(self.node_ids) or any(
            len(row) != self.graph_feature_dim for row in self.graph_input_node_features
        ):
            raise ValueError("persistent graph input features must match graph_feature_dim")
        if len(self.graph_node_features) != len(self.node_ids) or any(
            len(row) != self.graph_feature_dim for row in self.graph_node_features
        ):
            raise ValueError("persistent graph node features must match graph_feature_dim")
        if any(not math.isfinite(value) for matrix in (self.graph_input_node_features, self.graph_node_features) for row in matrix for value in row):
            raise ValueError("persistent graph node features must be finite")
        ap_set = set(self.graph_ap_node_ids)
        row_ids = set(self.reference_scan_ids)
        edge_ids: set[str] = set()
        edge_pairs: set[tuple[str, str]] = set()
        for edge in self.radio_edges:
            if len(edge.edge_feature) != self.graph_feature_dim:
                raise ValueError("transition radio edge width must match graph_feature_dim")
            if edge.edge_id in edge_ids or (edge.sample_node_id, edge.bssid) in edge_pairs:
                raise ValueError("persistent radio edge identities must be unique")
            edge_ids.add(edge.edge_id)
            edge_pairs.add((edge.sample_node_id, edge.bssid))
            if edge.sample_node_id not in row_ids or ap_id_for_bssid(edge.bssid) not in ap_set:
                raise ValueError("persistent radio edge endpoint is not in the graph")
        all_batch_ids = tuple(record.batch_id for record in self.batch_history)
        if len(set(all_batch_ids)) != len(all_batch_ids):
            raise ValueError("transition batch history cannot repeat a batch ID")
        all_scans = tuple(scan for record in self.batch_history for scan in record.scan_ids)
        if len(set(all_scans)) != len(all_scans) or set(all_scans) & set(self.reference_scan_ids):
            raise ValueError("batch scan IDs must be globally unique from reference scans")
        return self


class OnlineNodeFeatures(BaseModel):
    """Final current-GNN output extended with Section 6 new-AP initial rows."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_online_ap_features_v1"] = "gufu_online_ap_features_v1"
    online_graph_id: str = Field(pattern=r"^onlinegraph:[0-9a-f]{64}$")
    source_state_id: str = Field(pattern=r"^transition:[0-9a-f]{64}$")
    input_batch_id: str
    source_model_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bssid_vocabulary: tuple[str, ...]
    feature_dim: int = Field(default=FEATURE_DIM, ge=1)
    node_ids: tuple[str, ...]
    node_types: tuple[Literal["sample", "ap"], ...]
    node_id_to_row: tuple[tuple[str, int], ...]
    node_feature_matrix: tuple[tuple[float, ...], ...]
    temporary_scan_ids: tuple[str, ...]
    temporary_ap_node_ids: tuple[str, ...]
    temporary_node_status: Literal["temporary_until_transition_complete"] = (
        "temporary_until_transition_complete"
    )
    radio_edge_ids: tuple[str, ...]
    radio_edge_feature_matrix: tuple[tuple[float, ...], ...]
    new_ap_bssids: tuple[str, ...]
    ap_initialization_policy: Literal["rss_plus_120_weighted_neighbor_gnn_mean_v1"] = (
        "rss_plus_120_weighted_neighbor_gnn_mean_v1"
    )

    @model_validator(mode="after")
    def aligned_nodes(self):
        if self.node_id_to_row != tuple((node_id, row) for row, node_id in enumerate(self.node_ids)):
            raise ValueError("online node-feature row map must exactly match its ID order")
        if len(set(self.node_ids)) != len(self.node_ids) or len(self.node_types) != len(self.node_ids):
            raise ValueError("online node IDs/types must be unique and aligned")
        if len(self.node_feature_matrix) != len(self.node_ids) or any(
            len(row) != self.feature_dim for row in self.node_feature_matrix
        ):
            raise ValueError("online node features must match feature_dim")
        if any(not math.isfinite(value) for row in self.node_feature_matrix for value in row):
            raise ValueError("online node features must be finite")
        if len(set(self.radio_edge_ids)) != len(self.radio_edge_ids) or len(self.radio_edge_ids) != len(self.radio_edge_feature_matrix):
            raise ValueError("online physical edge features must align to unique edge IDs")
        if any(len(row) != self.feature_dim for row in self.radio_edge_feature_matrix):
            raise ValueError("online physical edge feature rows must match feature_dim")
        if tuple(canonicalize_bssid(value) for value in self.new_ap_bssids) != self.new_ap_bssids:
            raise ValueError("new AP feature IDs must use full canonical BSSIDs")
        if not set(self.temporary_scan_ids + self.temporary_ap_node_ids) <= set(self.node_ids):
            raise ValueError("temporary sample/AP node identities are missing from the online graph")
        if any(not node_id.startswith("ap:") for node_id in self.temporary_ap_node_ids):
            raise ValueError("temporary AP nodes must use stable full-BSSID AP IDs")
        return self


class APGraphModification(BaseModel):
    """Reference-edge delta after current shared-AP and new-AP inference."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_ap_graph_delta_v1"] = "gufu_ap_graph_delta_v1"
    online_graph_id: str = Field(pattern=r"^onlinegraph:[0-9a-f]{64}$")
    source_state_id: str = Field(pattern=r"^transition:[0-9a-f]{64}$")
    input_batch_id: str
    source_model_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: LinkPredictionConfig
    candidate_scores: tuple[CandidateEdgeScore, ...]
    added_reference_edges: tuple[LinkEdgeChange, ...]
    removed_reference_edges: tuple[LinkEdgeChange, ...]
    removed_ap_bssids: tuple[str, ...]
    final_reference_edges: tuple[TransitionRadioEdge, ...]
    converged: Literal[True] = True

    @model_validator(mode="after")
    def valid_delta(self):
        if tuple(sorted(set(self.removed_ap_bssids))) != self.removed_ap_bssids:
            raise ValueError("removed AP BSSIDs must be unique and sorted")
        if any(canonicalize_bssid(value) != value for value in self.removed_ap_bssids):
            raise ValueError("removed AP identities must be full canonical BSSIDs")
        return self


class APTransitionSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    batch_id: str
    shared_bssids: tuple[str, ...]
    newly_observed_bssids: tuple[str, ...]
    returning_bssids: tuple[str, ...]
    missing_from_batch_bssids: tuple[str, ...]
    potentially_outdated_bssids: tuple[str, ...]
    removed_by_forgetting_bssids: tuple[str, ...]
    added_to_reference_bssids: tuple[str, ...]
    vocabulary_before: tuple[str, ...]
    vocabulary_after: tuple[str, ...]
    incoming_scan_count: int = Field(ge=1)
    preserved_new_ap_reading_count: int = Field(ge=0)
    source_state_id: str
    next_state_id: str


class APVocabularyPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    source_state_id: str
    batch_id: str
    shared_bssids: tuple[str, ...]
    newly_observed_bssids: tuple[str, ...]
    returning_bssids: tuple[str, ...]
    missing_from_batch_bssids: tuple[str, ...]
    potentially_outdated_bssids: tuple[str, ...]
    removed_by_forgetting_bssids: tuple[str, ...]
    added_to_reference_bssids: tuple[str, ...]
    vocabulary_before: tuple[str, ...]
    vocabulary_after: tuple[str, ...]


class TransitionFingerprintRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    scan_id: str
    original_location: tuple[float, float] | None
    updated_rssi_dbm: tuple[float, ...]
    observation_mask: tuple[bool, ...]
    value_provenance: tuple[
        Literal["measured", "decoder_prediction", "edge_prediction", "missing"], ...
    ]

    @model_validator(mode="after")
    def aligned(self):
        if not (len(self.updated_rssi_dbm) == len(self.observation_mask) == len(self.value_provenance)):
            raise ValueError("transition output values/masks/provenance must align")
        if any(not math.isfinite(value) for value in self.updated_rssi_dbm):
            raise ValueError("transition output RSS values must be finite")
        return self


class TransitionFingerprintArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_transition_fingerprints_v1"] = "gufu_transition_fingerprints_v1"
    state_id: str = Field(pattern=r"^transition:[0-9a-f]{64}$")
    source_state_id: str = Field(pattern=r"^transition:[0-9a-f]{64}$")
    input_batch_id: str
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bssid_vocabulary: tuple[str, ...]
    normalization: Literal["paper_ae", "legacy_raw_div_120"]
    output_unit: Literal["dbm"] = "dbm"
    rows: tuple[TransitionFingerprintRow, ...]
    temporary_update_scan_ids_discarded: Literal[True] = True
    temporary_virtual_edges_discarded: Literal[True] = True
    ae_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gnn_model_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at_utc: str

    @model_validator(mode="after")
    def aligned_rows(self):
        if canonical_json_digest(self.bssid_vocabulary) != self.vocabulary_sha256:
            raise ValueError("transition fingerprint vocabulary hash is inconsistent")
        if len({row.scan_id for row in self.rows}) != len(self.rows):
            raise ValueError("transition fingerprints contain duplicate reference IDs")
        if any(len(row.updated_rssi_dbm) != len(self.bssid_vocabulary) for row in self.rows):
            raise ValueError("transition RSS output width does not match AP vocabulary")
        return self


class TransitionEncoderFeatureArtifact(BaseModel):
    """Fresh final-AE feature rows after one vocabulary transition."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_transition_encoder_features_v1"] = "gufu_transition_encoder_features_v1"
    source_state_id: str
    input_batch_id: str
    dataset_id: str
    scan_ids: tuple[str, ...]
    feature_space: Literal["z_encoder"] = "z_encoder"
    feature_dim: int = Field(default=LATENT_DIM, ge=1)
    feature_matrix: tuple[tuple[float, ...], ...]
    bssid_vocabulary: tuple[str, ...]
    vocabulary_sha256: str
    normalization: Literal["paper_ae", "legacy_raw_div_120"]
    checkpoint_sha256: str

    @model_validator(mode="after")
    def aligned(self):
        if len(set(self.scan_ids)) != len(self.scan_ids) or len(self.scan_ids) != len(self.feature_matrix):
            raise ValueError("transition encoder rows must align to unique scan IDs")
        if any(len(row) != self.feature_dim for row in self.feature_matrix):
            raise ValueError("transition encoder rows must match feature_dim")
        if canonical_json_digest(self.bssid_vocabulary) != self.vocabulary_sha256:
            raise ValueError("transition encoder vocabulary digest is inconsistent")
        if any(not math.isfinite(value) for row in self.feature_matrix for value in row):
            raise ValueError("transition encoder features must be finite")
        return self


class TransitionGNNFeatureArtifact(BaseModel):
    """Fresh post-edge-change GNN outputs for the transitioned reference graph."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_transition_gnn_features_v1"] = "gufu_transition_gnn_features_v1"
    source_state_id: str
    input_batch_id: str
    graph_id: str
    feature_dim: int = Field(default=FEATURE_DIM, ge=1)
    node_ids: tuple[str, ...]
    node_types: tuple[Literal["sample", "ap"], ...]
    node_feature_matrix: tuple[tuple[float, ...], ...]
    radio_edge_ids: tuple[str, ...]
    radio_edge_feature_matrix: tuple[tuple[float, ...], ...]
    bssid_vocabulary: tuple[str, ...]
    vocabulary_sha256: str
    model_checkpoint_sha256: str
    encoder_feature_artifact_sha256: str

    @model_validator(mode="after")
    def aligned(self):
        if len(set(self.node_ids)) != len(self.node_ids) or len(self.node_types) != len(self.node_ids):
            raise ValueError("transition GNN node IDs/types must be unique and aligned")
        if len(self.node_feature_matrix) != len(self.node_ids) or any(len(row) != self.feature_dim for row in self.node_feature_matrix):
            raise ValueError("transition GNN node features must match feature_dim")
        if len(set(self.radio_edge_ids)) != len(self.radio_edge_ids) or len(self.radio_edge_ids) != len(self.radio_edge_feature_matrix):
            raise ValueError("transition GNN radio edge rows must be unique and aligned")
        if any(len(row) != self.feature_dim for row in self.radio_edge_feature_matrix):
            raise ValueError("transition GNN edge features must match feature_dim")
        if canonical_json_digest(self.bssid_vocabulary) != self.vocabulary_sha256:
            raise ValueError("transition GNN vocabulary digest is inconsistent")
        if any(not math.isfinite(value) for matrix in (self.node_feature_matrix, self.radio_edge_feature_matrix) for row in matrix for value in row):
            raise ValueError("transition GNN features must be finite")
        return self


class BatchTransitionArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_ap_transition_result_v1"] = TRANSITION_RESULT_FORMAT
    transition_id: str
    source_state_id: str
    state_id: str
    batch_id: str
    prepared_batch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    raw_batch_artifact: str
    raw_batch_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    ae_checkpoint_artifact: str
    ae_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    updater_checkpoint_artifact: str
    updater_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fingerprint_artifact: str
    fingerprint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    state_artifact: str
    summary: APTransitionSummary
    online_feature_artifact: str
    graph_modification_artifact: str
    encoder_feature_artifact: str
    gnn_feature_artifact: str
    config: TransitionConfig
    created_at_utc: str


@dataclass(frozen=True)
class APChangeInferenceResult:
    node_features: OnlineNodeFeatures
    modification: APGraphModification


@dataclass(frozen=True)
class BatchTransitionResult:
    state: VocabularyTransitionState
    fingerprints: TransitionFingerprintArtifact
    summary: APTransitionSummary
    manifest: BatchTransitionArtifact
    output_directory: Path
    migrated_ae_checkpoint: LoadedAECheckpoint


def observed_bssids(prepared: PreparedDataset) -> tuple[str, ...]:
    """Return identities with at least one true observation-mask entry."""
    present_columns = {
        column
        for scan in prepared.scans
        for column, observed in enumerate(scan.observation_mask)
        if observed
    }
    return tuple(
        bssid for column, bssid in enumerate(prepared.bssid_vocabulary)
        if column in present_columns
    )


def _transition_state_payload(state: VocabularyTransitionState | dict) -> dict:
    values = state if isinstance(state, dict) else state.model_dump(mode="json")
    return _jsonable({key: value for key, value in values.items() if key not in {"state_id", "format"}})


def _jsonable(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _pythonable(value):
    if isinstance(value, BaseModel):
        return _pythonable(value.model_dump(mode="python"))
    if isinstance(value, dict):
        return {str(key): _pythonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_pythonable(item) for item in value)
    if isinstance(value, list):
        return [_pythonable(item) for item in value]
    return value


def _transition_state_id(values: dict) -> str:
    return f"transition:{canonical_json_digest(_transition_state_payload(values))}"


def _finalize_transition_state(values: dict) -> VocabularyTransitionState:
    """Hash a fully schema-normalized payload so save/reload is byte-stable."""
    normalized = _pythonable(values)
    normalized["graph_id"] = _graph_state_id(normalized)
    normalized["state_id"] = "transition:" + ("0" * 64)
    state = VocabularyTransitionState.model_validate(normalized)
    normalized_state = state.model_dump(mode="json")
    graph_id = _graph_state_id(normalized_state)
    if graph_id != state.graph_id:
        raise ValueError("normalized transition graph fields changed graph identity")
    state_id = _transition_state_id(normalized_state)
    return state.model_copy(update={"state_id": state_id})


def _graph_state_id(values: dict) -> str:
    graph_keys = (
        "dataset_id",
        "reference_scan_ids",
        "bssid_vocabulary",
        "encoder_feature_dim",
        "graph_feature_dim",
        "node_ids",
        "node_types",
        "graph_input_node_features",
        "graph_node_features",
        "radio_edges",
        "reference_graph_id",
    )
    return f"graph:{canonical_json_digest(_jsonable({key: values[key] for key in graph_keys}))}"


def project_prepared_batch_to_vocabulary(
    batch: PreparedDataset,
    target_bssid_vocabulary: tuple[str, ...],
    *,
    zero_shared_ap_policy: Literal["error"] = "error",
) -> PreparedDataset:
    """Make an old-encoder view while leaving the full incoming batch untouched."""
    if batch.config.batch_role != "update_batch":
        raise ValueError("shared-AP projection requires an update_batch PreparedDataset")
    target = tuple(canonicalize_bssid(value) for value in target_bssid_vocabulary)
    if not target or len(set(target)) != len(target):
        raise ValueError("the existing encoder vocabulary must be non-empty and unique")
    shared = set(observed_bssids(batch)) & set(target)
    if not shared:
        if zero_shared_ap_policy == "error":
            raise NoSharedAPError(
                "incoming batch has zero APs in common with the current encoder vocabulary; "
                "the old encoder cannot process this batch"
            )
        raise ValueError(f"unsupported zero-shared-AP policy: {zero_shared_ap_policy}")

    source_column = {bssid: column for column, bssid in enumerate(batch.bssid_vocabulary)}
    projected_scans: list[PreparedScan] = []
    observed_count = 0
    for scan in batch.scans:
        rss: list[float] = []
        mask: list[bool] = []
        for bssid in target:
            column = source_column.get(bssid)
            if column is None:
                rss.append(-120.0)
                mask.append(False)
            else:
                value, observed = scan.rss_dbm[column], scan.observation_mask[column]
                rss.append(value if observed else -120.0)
                mask.append(observed)
                observed_count += int(observed)
        projected_scans.append(
            scan.model_copy(update={"rss_dbm": tuple(rss), "observation_mask": tuple(mask)})
        )
    config = batch.config.model_copy(update={"base_bssid_vocabulary": target})
    projected = PreparedDataset.model_validate(
        {
            **batch.model_dump(mode="python"),
            "config": config,
            "bssid_vocabulary": target,
            "ap_ids": tuple(ap_id_for_bssid(bssid) for bssid in target),
            "scans": tuple(projected_scans),
            "stats": batch.stats.model_copy(update={"observed_readings": observed_count}),
        }
    )
    if tuple(scan.scan_id for scan in projected.scans) != tuple(
        scan.scan_id for scan in batch.scans
    ):
        raise AssertionError("old-vocabulary projection must preserve global scan IDs and order")
    return projected


def classify_and_plan_ap_vocabulary(
    state: VocabularyTransitionState,
    incoming: PreparedDataset,
    modification: APGraphModification,
) -> APVocabularyPlan:
    """Compare full BSSID sets and derive ordered active-model vocabulary."""
    _validate_state_identity(state)
    if incoming.config.dataset_id != state.dataset_id or incoming.config.batch_role != "update_batch":
        raise ValueError("AP vocabulary planning requires a matching update batch")
    if modification.source_state_id != state.state_id or modification.input_batch_id != incoming.config.batch_id:
        raise ValueError("AP graph delta was produced for a different state or batch")
    if modification.source_vocabulary_sha256 != state.vocabulary_sha256:
        raise ValueError("AP graph delta was produced for a different vocabulary")
    incoming_set = set(observed_bssids(incoming))
    current = state.bssid_vocabulary
    registry = {record.bssid: record for record in state.ap_registry}
    removed = tuple(sorted(set(modification.removed_ap_bssids)))
    still_active = tuple(bssid for bssid in current if bssid not in set(removed))
    admitted = {edge.ap_node_id.removeprefix("ap:") for edge in modification.added_reference_edges}
    add_to_vocab = tuple(sorted((incoming_set - set(still_active)) & admitted))
    target = still_active + tuple(bssid for bssid in add_to_vocab if bssid not in set(still_active))
    if not target:
        raise ValueError("AP forgetting would leave an empty AE vocabulary")
    if len(set(target)) != len(target):
        raise ValueError("AP vocabulary plan produced duplicate BSSID columns")
    return APVocabularyPlan(
        source_state_id=state.state_id,
        batch_id=incoming.config.batch_id,
        shared_bssids=tuple(bssid for bssid in current if bssid in incoming_set),
        newly_observed_bssids=tuple(sorted(incoming_set - set(current))),
        returning_bssids=tuple(sorted(
            bssid for bssid in incoming_set
            if bssid in registry and registry[bssid].status == "tombstoned"
        )),
        missing_from_batch_bssids=tuple(bssid for bssid in current if bssid not in incoming_set),
        potentially_outdated_bssids=removed,
        removed_by_forgetting_bssids=removed,
        added_to_reference_bssids=tuple(sorted(admitted & incoming_set)),
        vocabulary_before=current,
        vocabulary_after=target,
    )


def remap_rssi_columns_by_bssid(
    values: tuple[tuple[float, ...], ...],
    masks: tuple[tuple[bool, ...], ...],
    provenance: tuple[tuple[str, ...], ...],
    source_vocabulary: tuple[str, ...],
    target_vocabulary: tuple[str, ...],
) -> tuple[tuple[tuple[float, ...], ...], tuple[tuple[bool, ...], ...], tuple[tuple[str, ...], ...]]:
    """Remap RSS values/masks/origins by identity, never by column position."""
    if not source_vocabulary or not target_vocabulary:
        raise ValueError("RSS vocabulary remap requires non-empty source and target vocabularies")
    if len(set(source_vocabulary)) != len(source_vocabulary) or len(set(target_vocabulary)) != len(target_vocabulary):
        raise ValueError("RSS vocabulary remap requires unique BSSID identities")
    if not (len(values) == len(masks) == len(provenance)):
        raise ValueError("RSS values, masks, and provenance row counts differ")
    if any(len(row) != len(source_vocabulary) for matrix in (values, masks, provenance) for row in matrix):
        raise ValueError("RSS source matrix width does not match its vocabulary")
    source_columns = {bssid: column for column, bssid in enumerate(source_vocabulary)}
    out_values, out_masks, out_provenance = [], [], []
    for value_row, mask_row, origin_row in zip(values, masks, provenance):
        target_values, target_masks, target_origins = [], [], []
        for bssid in target_vocabulary:
            column = source_columns.get(bssid)
            if column is None:
                target_values.append(-120.0)
                target_masks.append(False)
                target_origins.append("missing")
            else:
                target_values.append(value_row[column])
                target_masks.append(mask_row[column])
                target_origins.append(origin_row[column])
        out_values.append(tuple(target_values))
        out_masks.append(tuple(target_masks))
        out_provenance.append(tuple(target_origins))
    return tuple(out_values), tuple(out_masks), tuple(out_provenance)


def remap_node_rows_by_id(
    source_node_ids: tuple[str, ...],
    source_node_features: tuple[tuple[float, ...], ...],
    target_node_ids: tuple[str, ...],
) -> tuple[tuple[float, ...], ...]:
    """Join node rows by stable ID and fail if a target row has no learned source."""
    if len(source_node_ids) != len(source_node_features) or len(set(source_node_ids)) != len(source_node_ids):
        raise ValueError("source node IDs/features must be uniquely aligned")
    if len(set(target_node_ids)) != len(target_node_ids):
        raise ValueError("target node IDs must be unique")
    source_rows = dict(zip(source_node_ids, source_node_features))
    missing = tuple(node_id for node_id in target_node_ids if node_id not in source_rows)
    if missing:
        raise ValueError(f"no learned graph feature row for target node IDs: {missing}")
    return tuple(source_rows[node_id] for node_id in target_node_ids)


def bootstrap_transition_state(
    reference_prepared: PreparedDataset,
    reference_graph: GraphState,
    reference_gnn_features: GNNFeatureArtifact,
    ae_checkpoint_path: str | Path,
    *,
    reference_snapshot: UpdatedGraphSnapshot | None = None,
    parent_state: VocabularyTransitionState | None = None,
) -> VocabularyTransitionState:
    """Create a versioned transition base from selected bootstrap artifacts."""
    if reference_prepared.config.batch_role != "bootstrap_reference":
        raise ValueError("transition base must be bootstrap_reference data")
    if reference_graph.source_prepared_sha256 != canonical_json_digest(reference_prepared):
        raise ValueError("transition graph does not match its prepared reference data")
    if reference_gnn_features.graph_id != reference_graph.graph_id:
        raise ValueError("transition base requires graph-aligned learned GNN features")
    if reference_gnn_features.bssid_vocabulary != reference_prepared.bssid_vocabulary:
        raise ValueError("bootstrap GNN features use a different AP vocabulary")
    ae = load_ae_checkpoint(ae_checkpoint_path, reference_prepared)
    if reference_gnn_features.encoder_checkpoint_sha256 != ae.checkpoint_sha256:
        raise ValueError("bootstrap GNN features do not use the selected AE checkpoint")

    if reference_snapshot is not None:
        checks = (
            (reference_snapshot.parent_graph_id == reference_graph.graph_id, "parent graph"),
            (reference_snapshot.dataset_id == reference_graph.dataset_id, "dataset"),
            (reference_snapshot.bssid_vocabulary == reference_graph.bssid_vocabulary, "vocabulary"),
            (reference_snapshot.sample_node_ids == reference_graph.sample_node_ids, "reference scan IDs"),
            (reference_snapshot.gnn_checkpoint_sha256 == reference_gnn_features.checkpoint_sha256, "GNN checkpoint"),
        )
        for compatible, name in checks:
            if not compatible:
                raise ValueError(f"bootstrap graph snapshot has incompatible {name}")
        node_ids = reference_snapshot.node_ids
        node_types = reference_snapshot.node_types
        graph_ap_ids = reference_snapshot.active_ap_node_ids
        node_features = reference_snapshot.node_features
        base_input_rows = dict(zip(reference_graph.node_ids, reference_graph.node_features))
        graph_input_features = tuple(base_input_rows[node_id] for node_id in node_ids)
        radio_edges = tuple(
            TransitionRadioEdge(
                edge_id=edge.edge_id,
                sample_node_id=edge.sample_node_id,
                bssid=edge.ap_node_id.removeprefix("ap:"),
                scalar_weight=edge.scalar_weight,
                edge_feature=edge.edge_feature,
                source="retained_gnn_edge",
            )
            for edge in reference_snapshot.radio_edges
        )
        removed_by_id = set(reference_snapshot.removed_ap_node_ids)
        reference_graph_id = reference_snapshot.graph_id
    else:
        node_ids = reference_graph.node_ids
        node_types = reference_graph.node_types
        graph_ap_ids = reference_graph.ap_node_ids
        feature_rows = dict(
            zip(reference_gnn_features.node_ids, reference_gnn_features.node_feature_matrix)
        )
        node_features = tuple(feature_rows[node_id] for node_id in node_ids)
        base_input_rows = dict(zip(reference_graph.node_ids, reference_graph.node_features))
        graph_input_features = tuple(base_input_rows[node_id] for node_id in node_ids)
        edge_rows = dict(
            zip(reference_gnn_features.radio_edge_ids, reference_gnn_features.radio_edge_feature_matrix)
        )
        radio_edges = tuple(
            TransitionRadioEdge(
                edge_id=edge.edge_id,
                sample_node_id=edge.sample_node_id,
                bssid=edge.ap_node_id.removeprefix("ap:"),
                scalar_weight=edge.scalar_weight,
                edge_feature=edge_rows[edge.edge_id],
                source="retained_gnn_edge",
            )
            for edge in reference_graph.radio_edges
        )
        removed_by_id = set()
        reference_graph_id = reference_graph.graph_id

    scan_ids = tuple(scan.scan_id for scan in reference_prepared.scans)
    if tuple(node_ids[: len(scan_ids)]) != scan_ids:
        raise ValueError("bootstrap graph rows must begin with original reference scan IDs")
    positions = {row.scan_id: (row.x, row.y) for row in reference_prepared.reference_positions}
    location_rows = tuple(
        TransitionReferenceLocation(scan_id=scan_id, xy=positions.get(scan_id))
        for scan_id in scan_ids
    )
    provenance = tuple(
        tuple("measured" if observed else "missing" for observed in scan.observation_mask)
        for scan in reference_prepared.scans
    )
    registry = tuple(
        sorted(
            (
                APRegistryRecord(
                    bssid=bssid,
                    status="tombstoned" if ap_id_for_bssid(bssid) in removed_by_id else "active",
                    first_seen_batch_id=reference_prepared.config.batch_id,
                    last_seen_batch_id=reference_prepared.config.batch_id,
                    removed_by_batch_id=reference_prepared.config.batch_id
                    if ap_id_for_bssid(bssid) in removed_by_id
                    else None,
                )
                for bssid in reference_prepared.bssid_vocabulary
            ),
            key=lambda record: record.bssid,
        )
    )
    if parent_state is not None:
        _validate_state_identity(parent_state)
        if (
            parent_state.dataset_id != reference_prepared.config.dataset_id
            or parent_state.reference_batch_id != reference_prepared.config.batch_id
            or parent_state.reference_scan_ids != scan_ids
            or parent_state.bssid_vocabulary != reference_prepared.bssid_vocabulary
            or parent_state.normalization != reference_prepared.config.normalization_mode
            or parent_state.reference_observation_mask
            != tuple(scan.observation_mask for scan in reference_prepared.scans)
            or parent_state.reference_rssi_dbm
            != tuple(scan.rss_dbm for scan in reference_prepared.scans)
            or parent_state.reference_value_provenance
            != tuple(
                scan.value_provenance
                if scan.value_provenance is not None
                else tuple("measured" if observed else "missing" for observed in scan.observation_mask)
                for scan in reference_prepared.scans
            )
            or parent_state.reference_locations != location_rows
        ):
            raise ValueError("parent transition state is incompatible with the current reference dataset")
        registry = parent_state.ap_registry
    values = {
        "parent_state_id": parent_state.state_id if parent_state is not None else None,
        "dataset_id": reference_prepared.config.dataset_id,
        "reference_batch_id": reference_prepared.config.batch_id,
        "source_prepared_sha256": (
            parent_state.source_prepared_sha256
            if parent_state is not None
            else canonical_json_digest(reference_prepared)
        ),
        "preprocessing_config": (
            parent_state.preprocessing_config
            if parent_state is not None
            else reference_prepared.config
        ),
        "graph_build_config": reference_graph.build_config,
        "normalization": reference_prepared.config.normalization_mode,
        "encoder_feature_dim": reference_graph.encoder_feature_dim,
        "graph_feature_dim": reference_graph.node_feature_dim,
        "vocabulary_revision": parent_state.vocabulary_revision if parent_state is not None else 0,
        "bssid_vocabulary": reference_prepared.bssid_vocabulary,
        "vocabulary_sha256": canonical_json_digest(reference_prepared.bssid_vocabulary),
        "ap_registry": registry,
        "reference_scan_ids": scan_ids,
        "reference_locations": parent_state.reference_locations if parent_state is not None else location_rows,
        "reference_rssi_dbm": tuple(scan.rss_dbm for scan in reference_prepared.scans),
        "reference_observation_mask": tuple(scan.observation_mask for scan in reference_prepared.scans),
        "reference_value_provenance": (
            parent_state.reference_value_provenance if parent_state is not None else provenance
        ),
        "reference_graph_id": reference_graph_id,
        "gnn_feature_graph_id": reference_gnn_features.graph_id,
        "node_ids": node_ids,
        "node_types": node_types,
        "graph_ap_node_ids": graph_ap_ids,
        "graph_input_node_features": graph_input_features,
        "graph_node_features": node_features,
        "radio_edges": radio_edges,
        "ae_checkpoint_sha256": ae.checkpoint_sha256,
        "gnn_state_sha256": reference_gnn_features.checkpoint_sha256,
        "encoder_feature_artifact_sha256": reference_graph.encoder_feature_artifact_sha256,
        "gnn_feature_artifact_sha256": canonical_json_digest(reference_gnn_features),
        "encoder_cache_status": "valid",
        "gnn_cache_status": "valid",
        "batch_history": parent_state.batch_history if parent_state is not None else (),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return _finalize_transition_state(values)


def _validate_state_identity(state: VocabularyTransitionState) -> None:
    values = state.model_dump(mode="json")
    if state.state_id != _transition_state_id(values):
        raise ValueError("transition state ID does not match its contents")
    if state.graph_id != _graph_state_id(values):
        raise ValueError("transition graph ID does not match its contents")


def infer_ap_graph_changes(
    state: VocabularyTransitionState,
    context: FingerprintUpdateContext,
    models: FingerprintUpdaterModels,
    full_incoming_batch: PreparedDataset,
    config: TransitionConfig | None = None,
) -> APChangeInferenceResult:
    """Infer incoming AP edges after old-vocabulary GNN inference.

    ``context.update_prepared`` must be the identity-preserving projection of
    ``full_incoming_batch`` into the current model vocabulary. New AP raw
    readings are then used only to initialize their graph node and temporary
    measured edges. No update coordinate is read here.
    """
    selected = config or TransitionConfig()
    _validate_state_identity(state)
    if full_incoming_batch.config.batch_role != "update_batch":
        raise ValueError("AP transition input must be an update_batch PreparedDataset")
    if full_incoming_batch.config.dataset_id != state.dataset_id:
        raise ValueError("incoming AP batch belongs to a different dataset")
    if context.reference_prepared.config.dataset_id != state.dataset_id:
        raise ValueError("fingerprint-update context belongs to a different dataset")
    if context.reference_prepared.config.batch_role != "bootstrap_reference":
        raise ValueError("this transition implementation requires the current prepared reference")
    if context.reference_prepared.config.batch_id != state.reference_batch_id:
        raise ValueError("fingerprint-update reference batch does not match transition state")
    if tuple(scan.scan_id for scan in context.reference_prepared.scans) != state.reference_scan_ids:
        raise ValueError("fingerprint-update reference scan IDs do not match transition state")
    if context.online_graph.bssid_vocabulary != state.bssid_vocabulary:
        raise ValueError("old-vocabulary GNN inference does not match transition state")
    if context.update_prepared.config.batch_id != full_incoming_batch.config.batch_id:
        raise ValueError("projected and full incoming batches have different batch IDs")
    if tuple(scan.scan_id for scan in context.update_prepared.scans) != tuple(
        scan.scan_id for scan in full_incoming_batch.scans
    ):
        raise ValueError("projected batch changed stable scan IDs or ordering")
    if context.update_prepared.bssid_vocabulary != state.bssid_vocabulary:
        raise ValueError("incoming batch must be projected into the current AE vocabulary first")
    expected_projection = project_prepared_batch_to_vocabulary(
        full_incoming_batch, state.bssid_vocabulary
    )
    if canonical_json_digest(expected_projection) != canonical_json_digest(context.update_prepared):
        raise ValueError("old-vocabulary encoder input is not the exact identity-based projection of the full batch")
    if tuple(
        value for scan in context.update_prepared.scans for value in scan.source.source_record_sha256
    ) != tuple(
        value for scan in full_incoming_batch.scans for value in scan.source.source_record_sha256
    ):
        raise ValueError("projected and full incoming records have different source provenance")
    if models.checkpoint_sha256 == "" or models.manifest.online_graph_id != context.online_graph.graph_id:
        raise ValueError("selected updater models do not match the transient inference graph")
    if models.manifest.bssid_vocabulary != state.bssid_vocabulary:
        raise ValueError("selected updater checkpoint uses an incompatible AP vocabulary")
    if context.source_ae_checkpoint_sha256 != state.ae_checkpoint_sha256:
        raise ValueError("selected updater context was prepared from a different AE checkpoint")
    if context.source_gnn_checkpoint_sha256 != state.gnn_state_sha256:
        raise ValueError("selected updater context was prepared from a different GNN checkpoint")
    if context.online_graph.reference_graph_id != state.reference_graph_id and not (
        context.reference_snapshot is not None
        and context.reference_snapshot.graph_id == state.reference_graph_id
    ):
        raise ValueError("transient graph does not descend from the current graph snapshot")
    if state.encoder_cache_status != "valid" or state.gnn_cache_status != "valid":
        raise ValueError(
            "transition state learned caches were invalidated; regenerate compatible encoder/GNN artifacts first"
        )
    if models.gnn_model._inference_only is not True:
        raise ValueError("AP inference requires selected, trained GNN weights")

    # Section 6 order: evaluate the shared-AP transient graph before creating
    # the new AP rows. That keeps the old encoder input width unchanged.
    parameter = next(models.gnn_model.parameters())
    graph = context.tensors
    graph = GraphTensorInput(
        node_features=graph.node_features.to(device=parameter.device, dtype=parameter.dtype),
        edge_features=graph.edge_features.to(device=parameter.device, dtype=parameter.dtype),
        edge_left_rows=graph.edge_left_rows.to(device=parameter.device),
        edge_right_rows=graph.edge_right_rows.to(device=parameter.device),
        radio_edge_count=graph.radio_edge_count,
        node_ids=graph.node_ids,
        edge_ids=graph.edge_ids,
        radio_edge_ids=graph.radio_edge_ids,
        virtual_edge_ids=graph.virtual_edge_ids,
    )
    models.gnn_model.eval()
    with torch.inference_mode():
        learned = models.gnn_model(graph)
    base_node_rows = {
        node_id: tuple(float(value) for value in learned.node_features[row].detach().cpu().tolist())
        for row, node_id in enumerate(context.online_graph.node_ids)
    }
    base_edge_rows = {
        edge_id: tuple(float(value) for value in learned.radio_edge_features[row].detach().cpu().tolist())
        for row, edge_id in enumerate(edge.edge_id for edge in context.online_graph.radio_edges)
    }
    if any(len(row) != state.graph_feature_dim for row in base_node_rows.values()):
        raise ValueError("selected GNN output did not preserve graph_feature_dim")

    record_by_bssid = {record.bssid: record for record in state.ap_registry}
    active_graph_bssids = {node_id.removeprefix("ap:") for node_id in state.graph_ap_node_ids}
    new_bssids = tuple(
        sorted(
            bssid
            for bssid in observed_bssids(full_incoming_batch)
            if bssid not in active_graph_bssids
        )
    )
    raw_column = {bssid: column for column, bssid in enumerate(full_incoming_batch.bssid_vocabulary)}
    update_row = dict(context.online_graph.node_id_to_row)
    reference_ids = set(state.reference_scan_ids)
    sample_ids = context.online_graph.sample_node_ids
    ap_ids = tuple(context.online_graph.ap_node_ids) + tuple(ap_id_for_bssid(bssid) for bssid in new_bssids)
    new_ap_features: dict[str, tuple[float, ...]] = {}
    temporary_new_ap_edges: list[tuple[str, str, float]] = []
    temporary_edge_features: dict[str, tuple[float, ...]] = {}
    for bssid in new_bssids:
        column = raw_column[bssid]
        weighted: list[tuple[tuple[float, ...], float]] = []
        for scan in full_incoming_batch.scans:
            if not scan.observation_mask[column]:
                continue
            scalar_weight = scan.rss_dbm[column] + 120.0
            if scalar_weight < 0.0:
                raise ValueError("incoming AP RSS is outside the RSS+120 graph-weight range")
            node_feature = base_node_rows.get(scan.scan_id)
            if node_feature is None:
                raise ValueError(f"incoming sample {scan.scan_id} is absent from final GNN output")
            weighted.append((node_feature, scalar_weight))
            ap_node_id = ap_id_for_bssid(bssid)
            temporary_new_ap_edges.append((scan.scan_id, ap_node_id, scalar_weight))
        total_weight = sum(weight for _feature, weight in weighted)
        if not weighted or total_weight <= 0.0:
            raise ValueError(
                f"new AP {bssid} has no positive RSS+120 sample-neighbor weight for Section 4.2 initialization"
            )
        new_ap_features[ap_id_for_bssid(bssid)] = tuple(
            sum(feature[dimension] * weight for feature, weight in weighted) / total_weight
            for dimension in range(state.graph_feature_dim)
        )
    for sample_id, ap_node_id, _weight in temporary_new_ap_edges:
        sample_feature = base_node_rows[sample_id]
        ap_feature = new_ap_features[ap_node_id]
        temporary_edge_features[f"radio:{sample_id}|{ap_node_id}"] = tuple(
            (left + right) / 2.0 for left, right in zip(sample_feature, ap_feature)
        )

    all_node_ids = tuple(context.online_graph.node_ids) + tuple(
        ap_id_for_bssid(bssid) for bssid in new_bssids
    )
    all_node_features = tuple(base_node_rows[node_id] for node_id in context.online_graph.node_ids) + tuple(
        new_ap_features[ap_id_for_bssid(bssid)] for bssid in new_bssids
    )
    radio_rows: list[tuple[str, str, float]] = [
        (edge.sample_node_id, edge.ap_node_id, edge.scalar_weight)
        for edge in context.online_graph.radio_edges
    ] + temporary_new_ap_edges
    existing_candidates = tuple(
        (edge.sample_node_id, ap_id_for_bssid(edge.bssid)) for edge in state.radio_edges
    )
    virtual_neighbors: dict[str, set[str]] = {}
    virtual_rows: list[tuple[str, str, float]] = []
    for edge in context.online_graph.virtual_edges:
        virtual_rows.append((edge.left_sample_id, edge.right_sample_id, edge.cosine_similarity))
        virtual_neighbors.setdefault(edge.left_sample_id, set()).add(edge.right_sample_id)
        virtual_neighbors.setdefault(edge.right_sample_id, set()).add(edge.left_sample_id)
    candidate_pairs: set[tuple[str, str]] = set(existing_candidates)
    for update_scan in full_incoming_batch.scans:
        for bssid in new_bssids:
            column = raw_column[bssid]
            if not update_scan.observation_mask[column]:
                continue
            ap_node_id = ap_id_for_bssid(bssid)
            for neighbor in virtual_neighbors.get(update_scan.scan_id, ()):
                if neighbor in reference_ids:
                    candidate_pairs.add((neighbor, ap_node_id))

    # The edge vector for a newly measured temporary edge is the named
    # Algorithm-1 endpoint-mean initializer; its scalar remains RSS+120.
    for sample_id, ap_id, weight in temporary_new_ap_edges:
        if weight < 0.0:
            raise ValueError("temporary measured AP edge has negative scalar weight")
    combined_edges = tuple(
        (sample_id, ap_id, weight) for sample_id, ap_id, weight in radio_rows
    )
    prediction = score_candidate_pairs_from_features(
        node_ids=all_node_ids,
        node_feature_matrix=all_node_features,
        sample_node_ids=sample_ids,
        ap_node_ids=ap_ids,
        radio_edges=combined_edges,
        virtual_edges=tuple(virtual_rows),
        candidate_pairs=tuple(sorted(candidate_pairs)),
        config=selected.link_prediction,
    )
    if prediction.outcome != "converged":
        raise RuntimeError(
            "AP graph prediction reached the link-iteration limit; no transition delta was applied"
        )
    score_by_pair = {
        (score.sample_node_id, score.ap_node_id): score for score in prediction.scores
    }
    old_edges = {
        (edge.sample_node_id, ap_id_for_bssid(edge.bssid)): edge for edge in state.radio_edges
    }
    final_edge_map: dict[tuple[str, str], TransitionRadioEdge] = dict(old_edges)
    removed: list[LinkEdgeChange] = []
    added: list[LinkEdgeChange] = []
    for pair, old_edge in old_edges.items():
        score = score_by_pair[pair]
        if score.predicted_weight < selected.link_prediction.link_threshold:
            final_edge_map.pop(pair)
            removed.append(
                LinkEdgeChange(
                    edge_id=old_edge.edge_id,
                    sample_node_id=pair[0],
                    ap_node_id=pair[1],
                    old_weight=old_edge.scalar_weight,
                    new_weight=None,
                    predicted_weight=score.predicted_weight,
                )
            )
    for pair, score in score_by_pair.items():
        if score.existed_before or score.predicted_weight < selected.link_prediction.link_threshold:
            continue
        sample_id, ap_id = pair
        bssid = ap_id.removeprefix("ap:")
        if bssid not in new_bssids:
            continue
        feature_rows = dict(zip(all_node_ids, all_node_features))
        endpoint_mean = tuple(
            (left + right) / 2.0
            for left, right in zip(feature_rows[sample_id], feature_rows[ap_id])
        )
        edge = TransitionRadioEdge(
            edge_id=f"radio:{sample_id}|{ap_id}",
            sample_node_id=sample_id,
            bssid=bssid,
            scalar_weight=score.predicted_weight,
            edge_feature=endpoint_mean,
            source="new_ap_endpoint_mean",
        )
        final_edge_map[pair] = edge
        added.append(
            LinkEdgeChange(
                edge_id=edge.edge_id,
                sample_node_id=sample_id,
                ap_node_id=ap_id,
                old_weight=None,
                new_weight=score.predicted_weight,
                predicted_weight=score.predicted_weight,
            )
        )
    existing_graph_bssids = {
        node_id.removeprefix("ap:") for node_id in state.graph_ap_node_ids
    }
    edges_by_ap: dict[str, int] = {bssid: 0 for bssid in existing_graph_bssids}
    for edge in final_edge_map.values():
        edges_by_ap[edge.bssid] = edges_by_ap.get(edge.bssid, 0) + 1
    removed_ap_bssids = tuple(
        sorted(bssid for bssid, count in edges_by_ap.items() if count == 0)
    )
    final_edges = tuple(sorted(final_edge_map.values(), key=lambda edge: (edge.sample_node_id, edge.bssid)))
    online_values = {
        "online_graph_id": context.online_graph.graph_id,
        "source_state_id": state.state_id,
        "input_batch_id": full_incoming_batch.config.batch_id,
        "source_model_checkpoint_sha256": models.checkpoint_sha256,
        "bssid_vocabulary": state.bssid_vocabulary,
        "feature_dim": state.graph_feature_dim,
        "node_ids": all_node_ids,
        "node_types": tuple(context.online_graph.node_types) + ("ap",) * len(new_bssids),
        "node_id_to_row": tuple((node_id, row) for row, node_id in enumerate(all_node_ids)),
        "node_feature_matrix": all_node_features,
        "temporary_scan_ids": context.online_graph.update_sample_ids,
        "temporary_ap_node_ids": tuple(ap_id_for_bssid(bssid) for bssid in new_bssids),
        "radio_edge_ids": tuple(edge.edge_id for edge in context.online_graph.radio_edges)
        + tuple(sorted(temporary_edge_features)),
        "radio_edge_feature_matrix": tuple(base_edge_rows[edge.edge_id] for edge in context.online_graph.radio_edges)
        + tuple(temporary_edge_features[key] for key in sorted(temporary_edge_features)),
        "new_ap_bssids": new_bssids,
    }
    node_artifact = OnlineNodeFeatures.model_validate(online_values)
    modification = APGraphModification.model_validate(
        {
            "online_graph_id": context.online_graph.graph_id,
            "source_state_id": state.state_id,
            "input_batch_id": full_incoming_batch.config.batch_id,
            "source_model_checkpoint_sha256": models.checkpoint_sha256,
            "source_vocabulary_sha256": state.vocabulary_sha256,
            "config": selected.link_prediction,
            "candidate_scores": prediction.scores,
            "added_reference_edges": tuple(sorted(added, key=lambda item: (item.sample_node_id, item.ap_node_id))),
            "removed_reference_edges": tuple(sorted(removed, key=lambda item: (item.sample_node_id, item.ap_node_id))),
            "removed_ap_bssids": removed_ap_bssids,
            "final_reference_edges": final_edges,
        }
    )
    return APChangeInferenceResult(node_artifact, modification)


def validate_transition_artifacts(
    state: VocabularyTransitionState,
    *,
    encoder_features: EncoderFeatureArtifact | TransitionEncoderFeatureArtifact | None = None,
    gnn_features: GNNFeatureArtifact | TransitionGNNFeatureArtifact | None = None,
) -> None:
    """Reject feature artifacts whose vocabulary, checkpoint, graph, or rows are stale."""
    _validate_state_identity(state)
    if state.encoder_cache_status != "valid" or state.gnn_cache_status != "valid":
        raise ValueError("transition invalidated encoder/GNN feature caches; regenerate them before reuse")
    if isinstance(encoder_features, TransitionEncoderFeatureArtifact):
        checks = (
            (encoder_features.dataset_id == state.dataset_id, "dataset"),
            (encoder_features.scan_ids == state.reference_scan_ids, "reference scan IDs/order"),
            (encoder_features.bssid_vocabulary == state.bssid_vocabulary, "vocabulary"),
            (encoder_features.vocabulary_sha256 == state.vocabulary_sha256, "vocabulary digest"),
            (encoder_features.normalization == state.normalization, "normalization"),
            (encoder_features.checkpoint_sha256 == state.ae_checkpoint_sha256, "AE checkpoint"),
            (encoder_features.feature_dim == state.encoder_feature_dim, "latent feature width"),
            (canonical_json_digest(encoder_features) == state.encoder_feature_artifact_sha256, "artifact digest"),
        )
        for matches, name in checks:
            if not matches:
                raise ValueError(f"transition encoder feature artifact has incompatible {name}")
    elif encoder_features is not None:
        checks = (
            (encoder_features.dataset_id == state.dataset_id, "dataset"),
            (encoder_features.scan_ids == state.reference_scan_ids, "reference scan IDs/order"),
            (encoder_features.bssid_vocabulary == state.bssid_vocabulary, "vocabulary"),
            (encoder_features.vocabulary_sha256 == state.vocabulary_sha256, "vocabulary digest"),
            (encoder_features.normalization == state.normalization, "normalization"),
            (encoder_features.checkpoint_sha256 == state.ae_checkpoint_sha256, "AE checkpoint"),
            (encoder_features.feature_dim == state.encoder_feature_dim, "latent feature width"),
        )
        for matches, name in checks:
            if not matches:
                raise ValueError(f"encoder feature artifact has incompatible {name}")
    if isinstance(gnn_features, TransitionGNNFeatureArtifact):
        checks = (
            (gnn_features.graph_id == state.gnn_feature_graph_id, "graph ID"),
            (gnn_features.node_ids == state.node_ids, "node IDs/order"),
            (gnn_features.node_types == state.node_types, "node types/order"),
            (gnn_features.radio_edge_ids == tuple(edge.edge_id for edge in state.radio_edges), "radio edge IDs/order"),
            (gnn_features.bssid_vocabulary == state.bssid_vocabulary, "vocabulary"),
            (gnn_features.vocabulary_sha256 == state.vocabulary_sha256, "vocabulary digest"),
            (gnn_features.model_checkpoint_sha256 == state.gnn_state_sha256, "GNN checkpoint"),
            (gnn_features.feature_dim == state.graph_feature_dim, "graph feature width"),
            (canonical_json_digest(gnn_features) == state.gnn_feature_artifact_sha256, "artifact digest"),
        )
        for matches, name in checks:
            if not matches:
                raise ValueError(f"transition GNN feature artifact has incompatible {name}")
    elif gnn_features is not None:
        checks = (
            (gnn_features.graph_id == state.graph_id, "graph ID"),
            (gnn_features.node_ids == state.node_ids, "node IDs/order"),
            (gnn_features.node_types == state.node_types, "node types/order"),
            (gnn_features.radio_edge_ids == tuple(edge.edge_id for edge in state.radio_edges), "radio edge IDs/order"),
            (gnn_features.bssid_vocabulary == state.bssid_vocabulary, "vocabulary"),
            (gnn_features.vocabulary_sha256 == state.vocabulary_sha256, "vocabulary digest"),
            (gnn_features.checkpoint_sha256 == state.gnn_state_sha256, "GNN checkpoint"),
            (gnn_features.node_feature_dim == state.graph_feature_dim, "graph feature width"),
            (canonical_json_digest(gnn_features) == state.gnn_feature_artifact_sha256, "artifact digest"),
        )
        for matches, name in checks:
            if not matches:
                raise ValueError(f"GNN feature artifact has incompatible {name}")


def load_transition_state(path: str | Path) -> VocabularyTransitionState:
    from gufu_artifacts import read_json_text

    state = VocabularyTransitionState.model_validate_json(read_json_text(path))
    _validate_state_identity(state)
    return state


def save_transition_state(
    state: VocabularyTransitionState,
    path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    _validate_state_identity(state)
    return atomic_write_json(state, path, overwrite=overwrite)


def _transition_ids(values: dict) -> dict:
    values = dict(values)
    values["graph_id"] = _graph_state_id(values)
    values["state_id"] = _transition_state_id(values)
    return values


def _raw_matrix_with_new_ap_edges(
    state: VocabularyTransitionState,
    fingerprints: UpdatedFingerprintArtifact,
    modification: APGraphModification,
    target_vocabulary: tuple[str, ...],
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[bool, ...], ...],
    tuple[tuple[str, ...], ...],
]:
    state_row_by_id = {scan_id: row for row, scan_id in enumerate(state.reference_scan_ids)}
    output_row_by_id = {row.scan_id: row for row in fingerprints.rows}
    edge_weight = {
        (edge.sample_node_id, edge.bssid): edge.scalar_weight
        for edge in modification.final_reference_edges
    }
    old_column = {bssid: column for column, bssid in enumerate(fingerprints.bssid_vocabulary)}
    new_column = {bssid: column for column, bssid in enumerate(target_vocabulary)}
    raw_rows: list[tuple[float, ...]] = []
    mask_rows: list[tuple[bool, ...]] = []
    provenance_rows: list[tuple[str, ...]] = []
    for scan_id in state.reference_scan_ids:
        old_row = output_row_by_id[scan_id]
        state_index = state_row_by_id[scan_id]
        raw: list[float] = []
        mask: list[bool] = []
        provenance: list[str] = []
        for bssid in target_vocabulary:
            if bssid in old_column:
                source_column = old_column[bssid]
                value = old_row.updated_rssi_dbm[source_column]
                is_observed = old_row.observation_mask[source_column]
                old_origin = old_row.updated_value_provenance[source_column]
                origin = (
                    "measured"
                    if old_origin == "retained_original" and is_observed
                    else "missing"
                    if old_origin == "retained_original" and not is_observed
                    else "decoder_prediction"
                )
            else:
                weight = edge_weight.get((scan_id, bssid))
                if weight is None:
                    value, is_observed, origin = -120.0, False, "missing"
                else:
                    value, is_observed, origin = weight - 120.0, False, "edge_prediction"
            # If a predicted edge has been explicitly removed in the same
            # transition, do not leave it as a reference RSS value.
            if origin == "edge_prediction" and (scan_id, bssid) not in edge_weight:
                value, is_observed, origin = -120.0, False, "missing"
            raw.append(float(value))
            mask.append(bool(is_observed))
            provenance.append(origin)
        raw_rows.append(tuple(raw))
        mask_rows.append(tuple(mask))
        provenance_rows.append(tuple(provenance))
    return tuple(raw_rows), tuple(mask_rows), tuple(provenance_rows)


def apply_batch_transition(
    state: VocabularyTransitionState,
    full_incoming_batch: PreparedDataset,
    fingerprints: UpdatedFingerprintArtifact,
    inference: APChangeInferenceResult,
    source_ae_checkpoint_path: str | Path,
    output_directory: str | Path,
    config: TransitionConfig | None = None,
    *,
    updater_models: FingerprintUpdaterModels,
    log_every_epochs: int = DEFAULT_LOG_EVERY_K_EPOCHS,
) -> BatchTransitionResult:
    """Build and publish one immutable batch/AP/vocabulary transition.

    All files are written in a newly-created staging directory. The parent
    state, its artifacts, and the destination remain untouched if any
    validation, migration, fine-tuning, serialization, or reload check fails.
    """
    selected = config or TransitionConfig()
    log_every_epochs = validate_log_interval(log_every_epochs)
    _validate_state_identity(state)
    batch = full_incoming_batch
    modification = inference.modification
    learned = inference.node_features
    if modification.config != selected.link_prediction:
        raise ValueError("transition configuration does not match the AP edge-inference policy")
    if batch.config.batch_role != "update_batch" or batch.config.dataset_id != state.dataset_id:
        raise ValueError("incoming batch must be an update batch for the transition dataset")
    if batch.config.batch_id == state.reference_batch_id or any(
        row.batch_id == batch.config.batch_id for row in state.batch_history
    ):
        raise ValueError("batch ID was already used; a new batch must have new globally unique scan IDs")
    input_scan_ids = tuple(scan.scan_id for scan in batch.scans)
    if not input_scan_ids:
        raise ValueError("cannot insert an empty update batch")
    if set(input_scan_ids) & (set(state.reference_scan_ids) | {
        scan_id for row in state.batch_history for scan_id in row.scan_ids
    }):
        raise ValueError("incoming scan IDs collide with existing reference or batch lineage")
    if fingerprints.dataset_id != state.dataset_id or fingerprints.reference_batch_id != state.reference_batch_id:
        raise ValueError("fingerprint updater output belongs to a different reference dataset")
    if fingerprints.reference_scan_ids != state.reference_scan_ids:
        raise ValueError("fingerprint updater output changed original reference scan identity/order")
    if fingerprints.bssid_vocabulary != state.bssid_vocabulary or fingerprints.vocabulary_sha256 != state.vocabulary_sha256:
        raise ValueError("fingerprint updater output must use the current old vocabulary")
    if fingerprints.update_batch_id != batch.config.batch_id:
        raise ValueError("fingerprint updater output belongs to a different incoming batch")
    if modification.source_state_id != state.state_id or learned.source_state_id != state.state_id:
        raise ValueError("AP inference artifacts were created from a different parent transition state")
    if modification.input_batch_id != batch.config.batch_id or learned.input_batch_id != batch.config.batch_id:
        raise ValueError("AP inference artifacts refer to a different incoming batch")
    if modification.online_graph_id != learned.online_graph_id:
        raise ValueError("AP node features and edge changes come from different online graphs")
    if modification.source_model_checkpoint_sha256 != updater_models.checkpoint_sha256 or learned.source_model_checkpoint_sha256 != updater_models.checkpoint_sha256:
        raise ValueError("AP inference artifacts do not match the selected learned updater checkpoint")
    if fingerprints.source_checkpoint_sha256 != updater_models.checkpoint_sha256:
        raise ValueError("fingerprint updater output and AP inference use different checkpoints")
    if not updater_models.checkpoint_path.is_file() or sha256_file(updater_models.checkpoint_path) != updater_models.checkpoint_sha256:
        raise ValueError("selected updater checkpoint file is missing or its recorded digest changed")
    if learned.bssid_vocabulary != state.bssid_vocabulary:
        raise ValueError("shared-AP GNN feature artifact must bind to the old model vocabulary")

    registry_by_bssid = {record.bssid: record for record in state.ap_registry}
    incoming_set = set(observed_bssids(batch))
    plan = classify_and_plan_ap_vocabulary(state, batch, modification)
    shared = plan.shared_bssids
    missing = plan.missing_from_batch_bssids
    newly_observed = plan.newly_observed_bssids
    truly_new = tuple(sorted(incoming_set - set(registry_by_bssid)))
    returning = plan.returning_bssids
    # Returning/new APs are admitted only when LP creates at least one
    # persistent reference edge. Their raw update readings remain in the
    # saved incoming PreparedDataset even if they remain unlinked.
    removed = set(plan.removed_by_forgetting_bssids)
    new_active = plan.added_to_reference_bssids
    target_vocab = plan.vocabulary_after
    if set(edge.bssid for edge in modification.final_reference_edges) - set(target_vocab):
        raise ValueError("final graph contains an AP omitted from the target model vocabulary")

    # Map the update result back to original scan identities; update scans are
    # never appended to this reference table.
    raw_values, masks, origin_values = _raw_matrix_with_new_ap_edges(
        state, fingerprints, modification, target_vocab
    )
    raw = list(raw_values)
    value_origins = list(origin_values)
    source_ae = load_ae_checkpoint(source_ae_checkpoint_path)
    if source_ae.manifest.bssid_vocabulary != state.bssid_vocabulary:
        raise ValueError("source AE checkpoint does not match the pre-transition vocabulary")
    if source_ae.manifest.normalization != state.normalization:
        raise ValueError("source AE checkpoint uses a different normalization")
    if updater_models.ae_model.input_dim != len(state.bssid_vocabulary):
        raise ValueError("selected updated AE width does not match the old vocabulary")

    target_path = Path(output_directory)
    if target_path.exists():
        raise FileExistsError(f"refusing to overwrite transition output directory: {target_path}")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target_path.name}.staging-", dir=target_path.parent))
    try:
        prepared_path = staging / "prepared_batch.json"
        atomic_write_json(batch, prepared_path)
        raw_batch_sha = sha256_file(prepared_path)
        updater_checkpoint_path = staging / "updater_models.pt"
        shutil.copy2(updater_models.checkpoint_path, updater_checkpoint_path)
        updater_checkpoint_sha = sha256_file(updater_checkpoint_path)
        if updater_checkpoint_sha != updater_models.checkpoint_sha256:
            raise ValueError("staged updater checkpoint failed its source digest check")
        fingerprints_path = staging / "updated_fingerprints.json"
        atomic_write_json(fingerprints, fingerprints_path)
        online_features_path = staging / "online_node_features.json"
        atomic_write_json(learned, online_features_path)
        delta_path = staging / "ap_graph_changes.json"
        atomic_write_json(modification, delta_path)

        reactivated = tuple(bssid for bssid in target_vocab if bssid in set(returning))
        migrated_path = staging / "ae_transition.pt"
        migrated = migrate_ae_checkpoint_vocabulary(
            source_ae_checkpoint_path,
            target_vocab,
            migrated_path,
            seed=selected.reinitialize_seed,
            reactivated_bssids=reactivated,
            source_model=updater_models.ae_model,
            source_optimizer_state=updater_models.optimizer_states.get("updated_ae_adam"),
            source_model_identity=updater_models.checkpoint_sha256,
        )

        # Construct decoder training inputs from the transitioned RSS matrix.
        # A graph-inferred value is usable input while its observation mask
        # remains false; the mask records measured truth, provenance records
        # the inferred value. Missing values remain the normalization sentinel.
        label_rows = tuple(
            row for row, location in enumerate(state.reference_locations) if location.xy is not None
        )
        latent_by_id = dict(zip(
            fingerprints.updated_latent_reference_ids,
            fingerprints.updated_latent_reference_features,
        ))
        labeled_ids = tuple(state.reference_scan_ids[row] for row in label_rows)
        if not label_rows or any(scan_id not in latent_by_id for scan_id in labeled_ids):
            raise ValueError("Eq. (10) migration fine-tuning requires an updated latent target for every labeled reference")
        if any(
            len(latent_by_id[scan_id]) != state.encoder_feature_dim
            for scan_id in labeled_ids
        ):
            raise ValueError("updated latent targets do not match the transition encoder width")
        ae_rows = []
        for row in label_rows:
            ae_rows.append(
                raw_dbm_row_to_ae(
                    raw[row],
                    tuple(origin != "missing" for origin in value_origins[row]),
                    state.normalization,
                )
            )
        dtype = next(migrated.model.parameters()).dtype
        device = next(migrated.model.parameters()).device
        x = torch.tensor(ae_rows, dtype=dtype, device=device).unsqueeze(1)
        z_target = torch.tensor(
            [latent_by_id[scan_id] for scan_id in labeled_ids], dtype=dtype, device=device
        ).detach()
        model = migrated.model
        optimizer = torch.optim.Adam(model.parameters(), lr=selected.ae_finetune_learning_rate)
        optimizer.load_state_dict(migrated.optimizer_state)
        for group in optimizer.param_groups:
            group["lr"] = selected.ae_finetune_learning_rate
        loss_history: list[float] = []
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(selected.ae_finetune_seed)
            for _epoch in range(selected.ae_finetune_epochs):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                losses = compute_ae_update_losses(model, x, z_target)
                loss = losses["total"]
                loss.backward()
                gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
                if not gradients or any(not torch.isfinite(gradient).all().item() for gradient in gradients):
                    raise FloatingPointError("vocabulary-transition Eq. (10) gradients are missing or non-finite")
                if not any(torch.count_nonzero(gradient).item() for gradient in gradients):
                    raise RuntimeError("vocabulary-transition Eq. (10) produced no parameter gradient")
                optimizer.step()
                loss_history.append(float(loss.detach().cpu()))
                log_training_epoch(
                    "Vocabulary-transition AE Eq. (10)",
                    _epoch + 1,
                    selected.ae_finetune_epochs,
                    {
                        **{
                            key: float(value.detach().cpu())
                            for key, value in losses.items()
                        },
                        "reference_scans": x.shape[0],
                        "aps": x.shape[2],
                        "learning_rate": optimizer.param_groups[0]["lr"],
                    },
                    every_k_epochs=log_every_epochs,
                )
        model.eval()
        migration = migrated.manifest.vocabulary_migration.model_copy(
            update={
                "transition_batch_id": batch.config.batch_id,
                "finetune_epochs": len(loss_history),
                "finetune_seed": selected.ae_finetune_seed,
                "finetune_loss_history": tuple(loss_history),
            }
        )
        manifest = migrated.manifest.model_copy(update={"vocabulary_migration": migration})
        payload = load_torch_state(migrated_path)
        payload.update(
            {
                "encoder_state_dict": model.E.state_dict(),
                "decoder_state_dict": model.D.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "manifest": manifest.model_dump(mode="python"),
            }
        )
        atomic_torch_save(payload, migrated_path, overwrite=True)
        migrated = load_ae_checkpoint(migrated_path)

        edge_values = {edge.edge_id: edge for edge in modification.final_reference_edges}
        persisted_edges = tuple(
            edge_values[edge_id]
            for edge_id in sorted(edge_values)
            if edge_values[edge_id].bssid in target_vocab
        )
        graph_ap_ids = tuple(
            ap_id_for_bssid(bssid)
            for bssid in target_vocab
            if any(edge.bssid == bssid for edge in persisted_edges)
        )
        node_ids = state.reference_scan_ids + graph_ap_ids

        # Decode all labeled reference rows from the final transitioned model.
        model.eval()
        with torch.inference_mode():
            z_all = torch.tensor(
                [latent_by_id[scan_id] for scan_id in labeled_ids], dtype=dtype, device=device
            )
            reconstructed = model.D(z_all.unsqueeze(1))[:, 0, :].detach().cpu().tolist()
        for index, row in enumerate(label_rows):
            normalized = reconstructed[index]
            raw_row = list(raw[row])
            origin_row = list(value_origins[row])
            for column, normalized_value in enumerate(normalized):
                raw_row[column] = ae_value_to_raw_dbm(float(normalized_value), state.normalization)
                origin_row[column] = "decoder_prediction"
            raw[row] = tuple(raw_row)
            value_origins[row] = tuple(origin_row)

        # After edge changes and AE adaptation, encode every retained reference
        # row again, rebuild graph-input AP means from the final radio edge set,
        # and infer the final GNN node/edge features. The update samples and
        # virtual edges have now reached their last consumer and are omitted.
        old_input_rows = dict(zip(state.node_ids, state.graph_input_node_features))
        effective_masks = tuple(
            tuple(origin != "missing" for origin in origins) for origins in value_origins
        )
        encoder_inputs = torch.tensor(
            [
                raw_dbm_row_to_ae(row, mask, state.normalization)
                for row, mask in zip(raw, effective_masks)
            ],
            dtype=dtype,
            device=device,
        ).unsqueeze(1)
        model.eval()
        with torch.inference_mode():
            encoder_output = model.E(encoder_inputs)[:, 0, :]
        encoder_rows = tuple(
            tuple(float(value) for value in row)
            for row in encoder_output.detach().cpu().tolist()
        )
        sample_graph_inputs = {
            scan_id: encoder_rows[row] + tuple(old_input_rows[scan_id][state.encoder_feature_dim:])
            for row, scan_id in enumerate(state.reference_scan_ids)
        }
        ap_bssids = tuple(dict.fromkeys(edge.bssid for edge in persisted_edges))
        graph_ap_ids = tuple(ap_id_for_bssid(bssid) for bssid in target_vocab if bssid in set(ap_bssids))
        weighted_ap_inputs: dict[str, tuple[float, ...]] = {}
        for bssid in ap_bssids:
            incident = tuple(edge for edge in persisted_edges if edge.bssid == bssid)
            total_weight = sum(edge.scalar_weight for edge in incident)
            if total_weight <= 0.0:
                raise ValueError(
                    f"post-transition AP {bssid} has no positive incident RSS+120 weight"
                )
            weighted_ap_inputs[ap_id_for_bssid(bssid)] = tuple(
                sum(edge.scalar_weight * sample_graph_inputs[edge.sample_node_id][column] for edge in incident)
                / total_weight
                for column in range(state.graph_feature_dim)
            )
        node_ids = state.reference_scan_ids + graph_ap_ids
        graph_input_features = tuple(sample_graph_inputs[scan_id] for scan_id in state.reference_scan_ids) + tuple(
            weighted_ap_inputs[ap_id] for ap_id in graph_ap_ids
        )
        row_by_id = {node_id: row for row, node_id in enumerate(node_ids)}
        graph_radio_edges = tuple(
            edge.model_copy(
                update={
                    "edge_feature": tuple(
                        (left + right) / 2.0
                        for left, right in zip(
                            graph_input_features[row_by_id[edge.sample_node_id]],
                            graph_input_features[row_by_id[ap_id_for_bssid(edge.bssid)]],
                        )
                    ),
                    "source": "retained_endpoint_mean",
                }
            )
            for edge in persisted_edges
        )
        edge_left = tuple(row_by_id[edge.sample_node_id] for edge in graph_radio_edges)
        edge_right = tuple(row_by_id[ap_id_for_bssid(edge.bssid)] for edge in graph_radio_edges)
        gnn_parameter = next(updater_models.gnn_model.parameters())
        final_graph_tensors = GraphTensorInput(
            node_features=torch.tensor(graph_input_features, dtype=gnn_parameter.dtype, device=gnn_parameter.device),
            edge_features=torch.tensor(
                [edge.edge_feature for edge in graph_radio_edges],
                dtype=gnn_parameter.dtype,
                device=gnn_parameter.device,
            ).reshape((-1, state.graph_feature_dim)),
            edge_left_rows=torch.tensor(edge_left, dtype=torch.long, device=gnn_parameter.device),
            edge_right_rows=torch.tensor(edge_right, dtype=torch.long, device=gnn_parameter.device),
            radio_edge_count=len(graph_radio_edges),
            node_ids=node_ids,
            edge_ids=tuple(edge.edge_id for edge in graph_radio_edges),
            radio_edge_ids=tuple(edge.edge_id for edge in graph_radio_edges),
            virtual_edge_ids=(),
        )
        updater_models.gnn_model.eval()
        with torch.inference_mode():
            final_graph_output = updater_models.gnn_model(final_graph_tensors)
        node_features = tuple(
            tuple(float(value) for value in row)
            for row in final_graph_output.node_features.detach().cpu().tolist()
        )
        persisted_edges = tuple(
            edge.model_copy(
                update={
                    "edge_feature": tuple(float(value) for value in feature),
                    "source": "retained_gnn_edge",
                }
            )
            for edge, feature in zip(
                graph_radio_edges,
                final_graph_output.radio_edge_features.detach().cpu().tolist(),
            )
        )
        encoder_artifact = TransitionEncoderFeatureArtifact(
            source_state_id=state.state_id,
            input_batch_id=batch.config.batch_id,
            dataset_id=state.dataset_id,
            scan_ids=state.reference_scan_ids,
            feature_dim=updater_models.ae_model.config.latent_dim,
            feature_matrix=encoder_rows,
            bssid_vocabulary=target_vocab,
            vocabulary_sha256=canonical_json_digest(target_vocab),
            normalization=state.normalization,
            checkpoint_sha256=sha256_file(migrated_path),
        )
        encoder_artifact_sha = canonical_json_digest(encoder_artifact)

        registry_map = dict(registry_by_bssid)
        active_target = set(target_vocab)
        for bssid in sorted(set(batch.bssid_vocabulary) | set(registry_map)):
            existing = registry_map.get(bssid)
            if bssid in active_target:
                registry_map[bssid] = APRegistryRecord(
                    bssid=bssid,
                    status="active",
                    first_seen_batch_id=existing.first_seen_batch_id if existing else batch.config.batch_id,
                    last_seen_batch_id=batch.config.batch_id if bssid in incoming_set else (existing.last_seen_batch_id if existing else batch.config.batch_id),
                )
            elif bssid in removed:
                registry_map[bssid] = APRegistryRecord(
                    bssid=bssid,
                    status="tombstoned",
                    first_seen_batch_id=existing.first_seen_batch_id if existing else batch.config.batch_id,
                    last_seen_batch_id=batch.config.batch_id if bssid in incoming_set else (existing.last_seen_batch_id if existing else batch.config.batch_id),
                    removed_by_batch_id=batch.config.batch_id,
                )
            elif bssid in incoming_set and existing and existing.status == "tombstoned":
                registry_map[bssid] = APRegistryRecord(
                    bssid=bssid,
                    status="tombstoned",
                    first_seen_batch_id=existing.first_seen_batch_id,
                    last_seen_batch_id=batch.config.batch_id,
                    removed_by_batch_id=existing.removed_by_batch_id,
                )
            elif bssid in truly_new or (bssid in incoming_set and existing and existing.status == "observed_unlinked"):
                registry_map[bssid] = APRegistryRecord(
                    bssid=bssid,
                    status="observed_unlinked",
                    first_seen_batch_id=existing.first_seen_batch_id if existing else batch.config.batch_id,
                    last_seen_batch_id=batch.config.batch_id,
                )

        history = state.batch_history + (
            BatchLineageRecord(
                batch_id=batch.config.batch_id,
                scan_ids=input_scan_ids,
                prepared_sha256=canonical_json_digest(batch),
                source_state_id=state.state_id,
            ),
        )
        state_values = {
            "parent_state_id": state.state_id,
            "dataset_id": state.dataset_id,
            "reference_batch_id": state.reference_batch_id,
            "source_prepared_sha256": state.source_prepared_sha256,
            "preprocessing_config": state.preprocessing_config.model_copy(
                update={"base_bssid_vocabulary": target_vocab}
            ),
            "graph_build_config": state.graph_build_config,
            "normalization": state.normalization,
            "encoder_feature_dim": state.encoder_feature_dim,
            "graph_feature_dim": state.graph_feature_dim,
            "vocabulary_revision": state.vocabulary_revision + int(target_vocab != state.bssid_vocabulary),
            "bssid_vocabulary": target_vocab,
            "vocabulary_sha256": canonical_json_digest(target_vocab),
            "ap_registry": tuple(sorted(registry_map.values(), key=lambda row: row.bssid)),
            "reference_scan_ids": state.reference_scan_ids,
            "reference_locations": state.reference_locations,
            "reference_rssi_dbm": tuple(raw),
            "reference_observation_mask": tuple(masks),
            "reference_value_provenance": tuple(value_origins),
            "reference_graph_id": state.graph_id,
            "node_ids": node_ids,
            "node_types": ("sample",) * len(state.reference_scan_ids) + ("ap",) * len(graph_ap_ids),
            "graph_ap_node_ids": graph_ap_ids,
            "graph_input_node_features": graph_input_features,
            "graph_node_features": node_features,
            "radio_edges": persisted_edges,
            "ae_checkpoint_sha256": sha256_file(migrated_path),
            "gnn_state_sha256": updater_models.checkpoint_sha256,
            "encoder_feature_artifact_sha256": encoder_artifact_sha,
            "gnn_feature_artifact_sha256": None,
            "encoder_cache_status": "valid",
            "gnn_cache_status": "valid",
            "batch_history": history,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        state_values["graph_id"] = _graph_state_id(state_values)
        state_values["gnn_feature_graph_id"] = state_values["graph_id"]
        gnn_artifact = TransitionGNNFeatureArtifact(
            source_state_id=state.state_id,
            input_batch_id=batch.config.batch_id,
            graph_id=state_values["graph_id"],
            feature_dim=state.graph_feature_dim,
            node_ids=node_ids,
            node_types=state_values["node_types"],
            node_feature_matrix=node_features,
            radio_edge_ids=tuple(edge.edge_id for edge in persisted_edges),
            radio_edge_feature_matrix=tuple(edge.edge_feature for edge in persisted_edges),
            bssid_vocabulary=target_vocab,
            vocabulary_sha256=canonical_json_digest(target_vocab),
            model_checkpoint_sha256=updater_models.checkpoint_sha256,
            encoder_feature_artifact_sha256=encoder_artifact_sha,
        )
        state_values["gnn_feature_artifact_sha256"] = canonical_json_digest(gnn_artifact)
        next_state = _finalize_transition_state(state_values)

        output_rows = tuple(
            TransitionFingerprintRow(
                scan_id=scan_id,
                original_location=location.xy,
                updated_rssi_dbm=tuple(raw[row]),
                observation_mask=tuple(masks[row]),
                value_provenance=tuple(value_origins[row]),
            )
            for row, (scan_id, location) in enumerate(zip(state.reference_scan_ids, state.reference_locations))
        )
        fingerprint_artifact = TransitionFingerprintArtifact(
            state_id=next_state.state_id,
            source_state_id=state.state_id,
            input_batch_id=batch.config.batch_id,
            vocabulary_sha256=next_state.vocabulary_sha256,
            bssid_vocabulary=target_vocab,
            normalization=state.normalization,
            rows=output_rows,
            ae_checkpoint_sha256=sha256_file(migrated_path),
            gnn_model_checkpoint_sha256=updater_models.checkpoint_sha256,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        fingerprint_output_path = staging / "transition_fingerprints.json"
        atomic_write_json(fingerprint_artifact, fingerprint_output_path)
        atomic_write_json(encoder_artifact, staging / "encoder_features.json")
        atomic_write_json(gnn_artifact, staging / "gnn_features.json")
        state_path = staging / "transition_state.json"
        atomic_write_json(next_state, state_path)
        summary = APTransitionSummary(
            batch_id=batch.config.batch_id,
            shared_bssids=shared,
            newly_observed_bssids=newly_observed,
            returning_bssids=returning,
            missing_from_batch_bssids=missing,
            potentially_outdated_bssids=tuple(sorted(removed)),
            removed_by_forgetting_bssids=tuple(sorted(removed)),
            added_to_reference_bssids=tuple(sorted(new_active)),
            vocabulary_before=state.bssid_vocabulary,
            vocabulary_after=target_vocab,
            incoming_scan_count=len(input_scan_ids),
            preserved_new_ap_reading_count=sum(
                scan.observation_mask[column]
                for bssid in truly_new
                for column in (batch.bssid_vocabulary.index(bssid),)
                for scan in batch.scans
            ),
            source_state_id=state.state_id,
            next_state_id=next_state.state_id,
        )
        manifest = BatchTransitionArtifact(
            transition_id=uuid4().hex,
            source_state_id=state.state_id,
            state_id=next_state.state_id,
            batch_id=batch.config.batch_id,
            prepared_batch_sha256=canonical_json_digest(batch),
            raw_batch_artifact="prepared_batch.json",
            raw_batch_sha256=raw_batch_sha,
            ae_checkpoint_artifact="ae_transition.pt",
            ae_checkpoint_sha256=sha256_file(migrated_path),
            updater_checkpoint_artifact="updater_models.pt",
            updater_checkpoint_sha256=updater_checkpoint_sha,
            fingerprint_artifact="transition_fingerprints.json",
            fingerprint_sha256=sha256_file(fingerprint_output_path),
            state_artifact="transition_state.json",
            summary=summary,
            online_feature_artifact="online_node_features.json",
            graph_modification_artifact="ap_graph_changes.json",
            encoder_feature_artifact="encoder_features.json",
            gnn_feature_artifact="gnn_features.json",
            config=selected,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        atomic_write_json(manifest, staging / "transition_manifest.json")
        # Reload all core outputs before publication. This catches invalid
        # metadata and serialization errors while the parent remains intact.
        reloaded_state = load_transition_state(state_path)
        reloaded_ae = load_ae_checkpoint(migrated_path)
        loaded_fingerprints = TransitionFingerprintArtifact.model_validate_json(
            (staging / "transition_fingerprints.json").read_text(encoding="utf-8")
        )
        if reloaded_state.state_id != next_state.state_id or loaded_fingerprints.state_id != next_state.state_id:
            raise ValueError("staged transition outputs have mismatched state lineage")
        if target_path.exists():
            raise FileExistsError(f"refusing to overwrite transition output directory: {target_path}")
        os.replace(staging, target_path)
        final_ae = LoadedAECheckpoint(
            model=reloaded_ae.model,
            manifest=reloaded_ae.manifest,
            checkpoint_sha256=reloaded_ae.checkpoint_sha256,
            path=target_path / "ae_transition.pt",
            optimizer_state=reloaded_ae.optimizer_state,
            random_state=reloaded_ae.random_state,
            ap_parameter_archive=reloaded_ae.ap_parameter_archive,
        )
        return BatchTransitionResult(
            state=reloaded_state,
            fingerprints=loaded_fingerprints,
            summary=summary,
            manifest=manifest,
            output_directory=target_path,
            migrated_ae_checkpoint=final_ae,
        )
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise

