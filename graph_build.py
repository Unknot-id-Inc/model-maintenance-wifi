"""Deterministic, provenance-bound graph construction for GUFU Sec. 4.2.

This module creates the graph input state only. It does not train a GNN or
silently synthesize encoder features. Paper virtual-edge selection uses strict
cosine-thresholding; a top-k mode exists only as an explicitly named alternative.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Literal, Mapping

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ae import EncoderFeatureArtifact
from gufu_artifacts import atomic_write_json, canonical_json_digest, read_json_text
from gufu_data import (
    PreparedDataset,
    PreparedScan,
    canonicalize_bssid,
    rssi_dbm_to_graph_weight,
)


GRAPH_BUILDER_VERSION = "gufu_graph_builder_v1"
GRAPH_FEATURE_SPACE = "z_encoder_concat_location_input"


class GraphConstructionError(ValueError):
    """The prepared data cannot form a graph under the declared policy."""


class GraphBuildConfig(BaseModel):
    """Named graph initialization policies and their provenance."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    virtual_edge_strategy: Literal[
        "paper_cosine_threshold", "top_k_alternative"
    ] = "paper_cosine_threshold"
    virtual_edge_sigma: float = Field(default=0.95, ge=-1.0, le=1.0)
    top_k: int | None = Field(default=None, ge=1)
    similarity_feature_space: Literal[
        "z_graph_input_augmented"
    ] = "z_graph_input_augmented"
    radio_weight_policy: Literal["observed_rssi_dbm_plus_120"] = (
        "observed_rssi_dbm_plus_120"
    )
    virtual_edge_scalar_weight_policy: Literal["unspecified_none"] = (
        "unspecified_none"
    )
    virtual_edge_symmetry_policy: Literal[
        "one_undirected_pair_expanded_by_consumer"
    ] = "one_undirected_pair_expanded_by_consumer"
    virtual_edge_lifetime: Literal["gnn_aggregation_then_remove"] = (
        "gnn_aggregation_then_remove"
    )
    pseudo_location_policy: Literal["sha256_uniform_configured_bounds_v1"] = (
        "sha256_uniform_configured_bounds_v1"
    )
    pseudo_location_seed: int = 0
    pseudo_location_bounds: tuple[float, float, float, float] | None = None
    zero_norm_similarity_policy: Literal["omit_virtual_pair"] = "omit_virtual_pair"
    zero_weight_ap_policy: Literal["error"] = "error"
    isolated_vocabulary_ap_policy: Literal["omit_and_record"] = "omit_and_record"

    @model_validator(mode="after")
    def validate_strategy_options(self):
        if self.virtual_edge_strategy == "paper_cosine_threshold" and self.top_k is not None:
            raise ValueError("top_k is only valid with virtual_edge_strategy='top_k_alternative'")
        if self.virtual_edge_strategy == "top_k_alternative" and self.top_k is None:
            raise ValueError("top_k_alternative requires an explicit positive top_k")
        if self.pseudo_location_bounds is not None:
            if not all(math.isfinite(value) for value in self.pseudo_location_bounds):
                raise ValueError("pseudo-location bounds must be finite")
            min_x, max_x, min_y, max_y = self.pseudo_location_bounds
            if min_x > max_x or min_y > max_y:
                raise ValueError("pseudo-location bounds must be ordered min/max pairs")
        return self


class SampleLocationInitialization(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    scan_id: str
    xy: tuple[float, float]
    source: Literal["bootstrap_reference_label", "seeded_pseudo_location"]

    @model_validator(mode="after")
    def finite_coordinates(self):
        if not all(math.isfinite(value) for value in self.xy):
            raise ValueError("sample graph location inputs must be finite")
        return self


class RadioGraphEdge(BaseModel):
    """One undirected physical sample/AP edge; vector and scalar are separate."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    edge_id: str
    sample_node_id: str
    ap_node_id: str
    rssi_dbm: float
    scalar_weight: float = Field(ge=0.0)
    edge_feature: tuple[float, ...]

    @model_validator(mode="after")
    def validate_radio_edge(self):
        if not math.isfinite(self.rssi_dbm) or not math.isfinite(self.scalar_weight):
            raise ValueError("radio edge RSS and weight must be finite")
        if not math.isclose(self.scalar_weight, self.rssi_dbm + 120.0, abs_tol=1e-9):
            raise ValueError("radio weight must be observed RSS dBm plus 120")
        if any(not math.isfinite(value) for value in self.edge_feature):
            raise ValueError("radio edge feature must be finite")
        return self


class VirtualGraphEdge(BaseModel):
    """One canonical undirected, temporary sample/sample virtual edge."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    edge_id: str
    sample_node_id_left: str
    sample_node_id_right: str
    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    scalar_weight: Literal[None] = None
    edge_feature: tuple[float, ...]
    temporary: Literal[True] = True

    @model_validator(mode="after")
    def validate_virtual_edge(self):
        if self.sample_node_id_left >= self.sample_node_id_right:
            raise ValueError("virtual edge endpoints must be distinct and canonically ordered")
        if not math.isfinite(self.cosine_similarity):
            raise ValueError("virtual cosine similarity must be finite")
        if any(not math.isfinite(value) for value in self.edge_feature):
            raise ValueError("virtual edge feature must be finite")
        return self


class GraphState(BaseModel):
    """Serializable graph inputs with stable row maps and source provenance.

    ``node_features`` is a single ordinary matrix shaped
    ``[num_nodes, encoder_feature_dim + 2]``:
    sample rows contain encoder features concatenated with location inputs, and
    AP rows contain the physical-RSS-weighted average of neighboring samples.
    The two goodness/fairness vectors do not exist until link prediction.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    builder_version: Literal["gufu_graph_builder_v1"] = GRAPH_BUILDER_VERSION
    graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    graph_feature_space: Literal[
        "z_encoder_concat_location_input"
    ] = GRAPH_FEATURE_SPACE
    dataset_id: str
    batch_id: str
    batch_role: Literal["bootstrap_reference", "update_batch"]
    source_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_feature_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalization: Literal["paper_ae", "legacy_raw_div_120"]
    bssid_vocabulary: tuple[str, ...]
    ap_ids: tuple[str, ...]
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    build_config: GraphBuildConfig
    encoder_feature_dim: int = Field(ge=1)
    coordinate_feature_dim: Literal[2] = 2
    node_feature_dim: int = Field(ge=3)
    node_ids: tuple[str, ...]
    node_id_to_row: tuple[tuple[str, int], ...]
    node_types: tuple[Literal["sample", "ap"], ...]
    sample_node_ids: tuple[str, ...]
    ap_node_ids: tuple[str, ...]
    omitted_unobserved_ap_ids: tuple[str, ...]
    node_features: tuple[tuple[float, ...], ...]
    reference_sample_mask: tuple[bool, ...]
    new_sample_mask: tuple[bool, ...]
    reference_label_mask: tuple[bool, ...]
    reference_coordinate_targets: tuple[tuple[float, float] | None, ...]
    sample_location_initializations: tuple[SampleLocationInitialization, ...]
    radio_edges: tuple[RadioGraphEdge, ...]
    virtual_edges: tuple[VirtualGraphEdge, ...]
    virtual_edges_are_temporary: Literal[True] = True

    @model_validator(mode="after")
    def validate_graph_contract(self):
        node_count = len(self.node_ids)
        if len(set(self.node_ids)) != node_count:
            raise ValueError("graph node IDs must be unique")
        if self.node_ids != self.sample_node_ids + self.ap_node_ids:
            raise ValueError("graph rows must be samples followed by AP nodes")
        if self.node_id_to_row != tuple((node_id, row) for row, node_id in enumerate(self.node_ids)):
            raise ValueError("node ID-to-row mapping must exactly follow node order")
        if len(self.node_types) != node_count or len(self.node_features) != node_count:
            raise ValueError("node types/features must align with node IDs")
        if self.node_feature_dim != self.encoder_feature_dim + self.coordinate_feature_dim:
            raise ValueError("graph feature width must equal encoder width plus coordinate width")
        if any(len(row) != self.node_feature_dim for row in self.node_features):
            raise ValueError("node feature matrix must match its declared feature width")
        if any(not math.isfinite(value) for row in self.node_features for value in row):
            raise ValueError("node features must be finite")
        if len(self.reference_sample_mask) != node_count or len(self.new_sample_mask) != node_count:
            raise ValueError("sample role masks must align with all node rows")
        if len(self.reference_label_mask) != node_count or len(self.reference_coordinate_targets) != node_count:
            raise ValueError("coordinate label masks/targets must align with all node rows")
        if len(self.ap_ids) != len(self.bssid_vocabulary):
            raise ValueError("AP IDs and full BSSID vocabulary must align")
        if canonical_json_digest(self.bssid_vocabulary) != self.vocabulary_sha256:
            raise ValueError("graph vocabulary hash is inconsistent")
        if len(set(self.ap_ids)) != len(self.ap_ids):
            raise ValueError("graph source AP IDs must be unique")
        if tuple(canonicalize_bssid(bssid) for bssid in self.bssid_vocabulary) != self.bssid_vocabulary:
            raise ValueError("graph vocabulary must contain full canonical BSSIDs")
        if self.ap_ids != tuple(f"ap:{bssid}" for bssid in self.bssid_vocabulary):
            raise ValueError("graph AP IDs must exactly preserve ordered full BSSID identities")
        if len(set(self.ap_node_ids)) != len(self.ap_node_ids):
            raise ValueError("graph AP node IDs must be unique")
        if len(set(self.omitted_unobserved_ap_ids)) != len(self.omitted_unobserved_ap_ids):
            raise ValueError("omitted unobserved AP IDs must be unique")
        if any(ap_id not in self.ap_ids for ap_id in self.ap_node_ids):
            raise ValueError("graph contains an AP node outside the source vocabulary")
        if set(self.ap_node_ids) & set(self.omitted_unobserved_ap_ids):
            raise ValueError("observed and omitted AP node sets overlap")
        if set(self.ap_node_ids) | set(self.omitted_unobserved_ap_ids) != set(self.ap_ids):
            raise ValueError("every vocabulary AP must be included or marked unobserved")
        expected_ap_node_ids = tuple(
            ap_id for ap_id in self.ap_ids if ap_id not in set(self.omitted_unobserved_ap_ids)
        )
        expected_omitted_ap_ids = tuple(
            ap_id for ap_id in self.ap_ids if ap_id not in set(self.ap_node_ids)
        )
        if self.ap_node_ids != expected_ap_node_ids:
            raise ValueError("AP node rows must preserve ordered vocabulary identity")
        if self.omitted_unobserved_ap_ids != expected_omitted_ap_ids:
            raise ValueError("omitted AP IDs must preserve ordered vocabulary identity")
        if self.node_types != ("sample",) * len(self.sample_node_ids) + ("ap",) * len(self.ap_node_ids):
            raise ValueError("node types must match sample/AP node ordering")

        if len(self.sample_location_initializations) != len(self.sample_node_ids):
            raise ValueError("every sample node needs one location initialization")
        if tuple(item.scan_id for item in self.sample_location_initializations) != self.sample_node_ids:
            raise ValueError("sample location initializations must align by scan ID")

        row_by_id = dict(self.node_id_to_row)
        for row, node_id in enumerate(self.node_ids):
            is_sample = row < len(self.sample_node_ids)
            ref = self.reference_sample_mask[row]
            new = self.new_sample_mask[row]
            labeled = self.reference_label_mask[row]
            target = self.reference_coordinate_targets[row]
            if is_sample:
                if ref == new:
                    raise ValueError("each sample must be exactly reference or new")
                if ref != (self.batch_role == "bootstrap_reference"):
                    raise ValueError("reference mask disagrees with graph batch role")
                if new != (self.batch_role == "update_batch"):
                    raise ValueError("new-sample mask disagrees with graph batch role")
                if labeled and (not ref or target is None):
                    raise ValueError("only a labeled reference sample may have a label target")
                if (target is not None) != labeled:
                    raise ValueError("coordinate target presence must match the label mask")
                if target is not None and not all(math.isfinite(value) for value in target):
                    raise ValueError("coordinate targets must be finite")
                initialization = self.sample_location_initializations[row]
                if initialization.source == "bootstrap_reference_label":
                    if not ref or not labeled or target != initialization.xy:
                        raise ValueError("labeled graph input must retain the bootstrap reference label")
                else:
                    if labeled:
                        raise ValueError("pseudo-location input cannot replace an available reference label")
                    bounds = self.build_config.pseudo_location_bounds
                    if bounds is None:
                        raise ValueError("pseudo-location input requires recorded map bounds")
                    min_x, max_x, min_y, max_y = bounds
                    x, y = initialization.xy
                    if (min_x == max_x and x != min_x) or (
                        min_x < max_x and not min_x <= x < max_x
                    ):
                        raise ValueError("pseudo-location x is outside recorded map bounds")
                    if (min_y == max_y and y != min_y) or (
                        min_y < max_y and not min_y <= y < max_y
                    ):
                        raise ValueError("pseudo-location y is outside recorded map bounds")
                if self.node_features[row][self.encoder_feature_dim:] != initialization.xy:
                    raise ValueError("sample graph feature must end with its declared location input")
            else:
                if ref or new or labeled or target is not None:
                    raise ValueError("AP nodes cannot carry sample coordinate masks/labels")

        if canonical_json_digest(self.bssid_vocabulary) != self.vocabulary_sha256:
            raise ValueError("graph vocabulary digest mismatch")
        expected_graph_id = _graph_id(
            source_prepared_sha256=self.source_prepared_sha256,
            encoder_feature_artifact_sha256=self.encoder_feature_artifact_sha256,
            build_config=self.build_config,
            node_ids=self.node_ids,
        )
        if self.graph_id != expected_graph_id:
            raise ValueError("graph ID does not match its inputs/configuration/node order")

        edge_ids: set[str] = set()
        radio_keys: set[tuple[str, str]] = set()
        for edge in self.radio_edges:
            if edge.edge_id in edge_ids:
                raise ValueError("graph edge IDs must be unique")
            edge_ids.add(edge.edge_id)
            if edge.sample_node_id not in self.sample_node_ids or edge.ap_node_id not in self.ap_node_ids:
                raise ValueError("physical radio edges must connect sample to AP nodes")
            key = edge.sample_node_id, edge.ap_node_id
            if key in radio_keys:
                raise ValueError("duplicate physical sample/AP edge")
            radio_keys.add(key)
            expected_edge_feature = _mean_features(
                self.node_features[row_by_id[edge.sample_node_id]],
                self.node_features[row_by_id[edge.ap_node_id]],
            )
            if not _features_close(edge.edge_feature, expected_edge_feature):
                raise ValueError("physical edge feature must be the endpoint mean")

        virtual_pairs: set[tuple[str, str]] = set()
        for edge in self.virtual_edges:
            if edge.edge_id in edge_ids:
                raise ValueError("graph edge IDs must be unique")
            edge_ids.add(edge.edge_id)
            if edge.sample_node_id_left not in self.sample_node_ids or edge.sample_node_id_right not in self.sample_node_ids:
                raise ValueError("virtual edges must connect two sample nodes")
            pair = (edge.sample_node_id_left, edge.sample_node_id_right)
            if pair in virtual_pairs:
                raise ValueError("duplicate virtual sample edge")
            virtual_pairs.add(pair)
            similarity = _cosine_similarity(
                self.node_features[row_by_id[pair[0]]],
                self.node_features[row_by_id[pair[1]]],
            )
            if similarity is None or not math.isclose(similarity, edge.cosine_similarity, abs_tol=1e-12):
                raise ValueError("virtual edge similarity does not match augmented node features")
            if self.build_config.virtual_edge_strategy == "paper_cosine_threshold":
                if not similarity > self.build_config.virtual_edge_sigma:
                    raise ValueError("paper virtual edges require cosine similarity strictly above sigma")
            expected_edge_feature = _mean_features(
                self.node_features[row_by_id[pair[0]]],
                self.node_features[row_by_id[pair[1]]],
            )
            if not _features_close(edge.edge_feature, expected_edge_feature):
                raise ValueError("virtual edge feature must be the endpoint mean")

        sample_features_by_id = {
            sample_id: self.node_features[row_by_id[sample_id]]
            for sample_id in self.sample_node_ids
        }
        expected_virtual_edges = _similarity_pairs(
            self.sample_node_ids, sample_features_by_id, self.build_config
        )
        actual_virtual_edges = tuple(
            (edge.sample_node_id_left, edge.sample_node_id_right, edge.cosine_similarity)
            for edge in self.virtual_edges
        )
        if len(expected_virtual_edges) != len(actual_virtual_edges) or any(
            expected_item[:2] != actual_item[:2]
            or not math.isclose(expected_item[2], actual_item[2], abs_tol=1e-12)
            for expected_item, actual_item in zip(expected_virtual_edges, actual_virtual_edges)
        ):
            raise ValueError("virtual edge table does not match configured similarity policy")

        if self.radio_edges != tuple(
            sorted(self.radio_edges, key=lambda edge: (edge.sample_node_id, edge.ap_node_id))
        ):
            raise ValueError("physical edge order must be deterministic by endpoint IDs")
        if self.virtual_edges != tuple(
            sorted(self.virtual_edges, key=lambda edge: (edge.sample_node_id_left, edge.sample_node_id_right))
        ):
            raise ValueError("virtual edge order must be deterministic by endpoint IDs")

        for ap_id in self.ap_node_ids:
            ap_edges = [edge for edge in self.radio_edges if edge.ap_node_id == ap_id]
            total_weight = sum(edge.scalar_weight for edge in ap_edges)
            if not ap_edges or total_weight <= 0:
                raise ValueError("AP feature initialization requires positive incident RSS weight")
            expected_ap_feature = tuple(
                sum(
                    edge.scalar_weight * self.node_features[row_by_id[edge.sample_node_id]][column]
                    for edge in ap_edges
                ) / total_weight
                for column in range(self.node_feature_dim)
            )
            actual = self.node_features[row_by_id[ap_id]]
            if not _features_close(actual, expected_ap_feature):
                raise ValueError("AP graph features must be the incident-RSS-weighted sample mean")
        return self

    @property
    def id_to_row(self) -> dict[str, int]:
        """Return a fresh ID-to-row mapping; serialized state remains immutable."""
        return dict(self.node_id_to_row)

    @property
    def node_count(self) -> int:
        return len(self.node_ids)

    @property
    def physical_edge_count(self) -> int:
        return len(self.radio_edges)

    @property
    def virtual_edge_count(self) -> int:
        return len(self.virtual_edges)

    def node_feature_tensor(self) -> torch.Tensor:
        """Return ordinary ``[num_nodes, node_feature_dim]`` features."""
        return torch.tensor(self.node_features, dtype=torch.float32)

    def edge_index_tensor(
        self, edge_type: Literal["radio", "virtual"], *, expand_undirected: bool = True
    ) -> torch.Tensor:
        """Return a 2×E row-index tensor, optionally expanding reverse arcs."""
        row_by_id = self.id_to_row
        if edge_type == "radio":
            pairs = [(row_by_id[e.sample_node_id], row_by_id[e.ap_node_id]) for e in self.radio_edges]
        elif edge_type == "virtual":
            pairs = [
                (row_by_id[e.sample_node_id_left], row_by_id[e.sample_node_id_right])
                for e in self.virtual_edges
            ]
        else:
            raise ValueError(f"unsupported graph edge type: {edge_type}")
        if expand_undirected:
            pairs = [arc for source, target in pairs for arc in ((source, target), (target, source))]
        if not pairs:
            return torch.empty((2, 0), dtype=torch.long)
        return torch.tensor(pairs, dtype=torch.long).T.contiguous()


def _mean_features(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    return tuple((a + b) / 2.0 for a, b in zip(left, right))


def _features_close(left: tuple[float, ...], right: tuple[float, ...]) -> bool:
    return len(left) == len(right) and all(
        math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
        for a, b in zip(left, right)
    )


def _cosine_similarity(
    left: tuple[float, ...], right: tuple[float, ...]
) -> float | None:
    left_norm_sq = sum(value * value for value in left)
    right_norm_sq = sum(value * value for value in right)
    if left_norm_sq == 0.0 or right_norm_sq == 0.0:
        return None
    dot = sum(a * b for a, b in zip(left, right))
    similarity = dot / math.sqrt(left_norm_sq * right_norm_sq)
    return max(-1.0, min(1.0, similarity))


def _pseudo_location(
    scan: PreparedScan,
    seed: int,
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    """Stable hash-derived uniform point inside configured map bounds."""
    # Source row/path identify the record without using its content hash, whose
    # raw line also contains the evaluation-only coordinates for update batches.
    payload = "\0".join(
        (
            "gufu-pseudo-location-v1",
            str(seed),
            scan.dataset_id,
            scan.batch_id,
            scan.source.source_file,
            str(scan.source.source_row),
        )
    ).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    x_bits = int.from_bytes(digest[0:8], "big") >> 11
    y_bits = int.from_bytes(digest[8:16], "big") >> 11
    scale = float(1 << 53)
    unit_x, unit_y = x_bits / scale, y_bits / scale
    min_x, max_x, min_y, max_y = bounds
    return min_x + unit_x * (max_x - min_x), min_y + unit_y * (max_y - min_y)


def _resolve_pseudo_location_config(
    prepared: PreparedDataset,
    config: GraphBuildConfig,
    reference_positions: dict[str, tuple[float, float]],
) -> GraphBuildConfig:
    """Resolve bounds only from bootstrap labels or explicit trusted config."""
    if config.pseudo_location_bounds is not None:
        return config
    if prepared.config.batch_role == "update_batch":
        raise GraphConstructionError(
            "update batches require pseudo_location_bounds from trusted map/bootstrap configuration; "
            "evaluation coordinates are not used to infer bounds"
        )
    if not reference_positions:
        raise GraphConstructionError(
            "bootstrap graph needs at least one reference label to infer pseudo-location bounds"
        )
    xs = tuple(x for x, _y in reference_positions.values())
    ys = tuple(y for _x, y in reference_positions.values())
    config_data = config.model_dump(mode="json")
    config_data["pseudo_location_bounds"] = (min(xs), max(xs), min(ys), max(ys))
    return GraphBuildConfig.model_validate(config_data)


def _graph_id(
    *,
    source_prepared_sha256: str,
    encoder_feature_artifact_sha256: str,
    build_config: GraphBuildConfig,
    node_ids: tuple[str, ...],
) -> str:
    identity = {
        "builder_version": GRAPH_BUILDER_VERSION,
        "source_prepared_sha256": source_prepared_sha256,
        "encoder_feature_artifact_sha256": encoder_feature_artifact_sha256,
        "build_config": build_config.model_dump(mode="json"),
        "node_ids": node_ids,
    }
    return "graph:" + canonical_json_digest(identity)


def _validate_inputs(
    prepared: PreparedDataset, features: EncoderFeatureArtifact
) -> None:
    if not prepared.scans:
        raise GraphConstructionError("graph construction requires at least one retained scan")
    if features.dataset_id != prepared.config.dataset_id:
        raise GraphConstructionError("encoder feature dataset_id does not match prepared data")
    if features.batch_id != prepared.config.batch_id:
        raise GraphConstructionError("encoder feature batch_id does not match prepared data")
    if features.source_prepared_sha256 != canonical_json_digest(prepared):
        raise GraphConstructionError("encoder features were produced from a different prepared artifact")
    if features.bssid_vocabulary != prepared.bssid_vocabulary:
        raise GraphConstructionError("encoder feature vocabulary order does not match prepared data")
    if features.vocabulary_sha256 != canonical_json_digest(prepared.bssid_vocabulary):
        raise GraphConstructionError("encoder feature vocabulary hash does not match prepared data")
    if features.normalization != prepared.config.normalization_mode:
        raise GraphConstructionError("encoder feature normalization does not match prepared data")
    prepared_scan_ids = tuple(scan.scan_id for scan in prepared.scans)
    if len(features.scan_ids) != len(prepared_scan_ids) or set(features.scan_ids) != set(prepared_scan_ids):
        raise GraphConstructionError("encoder feature scan IDs must exactly cover prepared scans")
    if len(features.feature_matrix) != len(features.scan_ids):
        raise GraphConstructionError("encoder feature rows do not match its scan IDs")
    if features.feature_dim < 1:
        raise GraphConstructionError("encoder feature width must be positive")


def _similarity_pairs(
    sample_node_ids: tuple[str, ...],
    features_by_scan_id: dict[str, tuple[float, ...]],
    config: GraphBuildConfig,
) -> tuple[tuple[str, str, float], ...]:
    similarities: dict[tuple[str, str], float] = {}
    neighbors_by_id: dict[str, list[tuple[str, float]]] = {
        sample_id: [] for sample_id in sample_node_ids
    }
    for index, left_id in enumerate(sample_node_ids):
        for right_id in sample_node_ids[index + 1 :]:
            similarity = _cosine_similarity(features_by_scan_id[left_id], features_by_scan_id[right_id])
            if similarity is not None:
                pair = tuple(sorted((left_id, right_id)))
                similarities[pair] = similarity
                neighbors_by_id[left_id].append((right_id, similarity))
                neighbors_by_id[right_id].append((left_id, similarity))

    if config.virtual_edge_strategy == "paper_cosine_threshold":
        chosen = {
            pair for pair, similarity in similarities.items()
            if similarity > config.virtual_edge_sigma
        }
    else:
        # Named alternative: directed KNN selections are merged as one undirected
        # edge if either endpoint selects the other. Ties break by stable scan ID.
        neighbor_pairs: set[tuple[str, str]] = set()
        for sample_id in sorted(sample_node_ids):
            candidates = neighbors_by_id[sample_id]
            candidates.sort(key=lambda item: (-item[1], item[0]))
            for neighbor_id, _similarity in candidates[: config.top_k]:
                neighbor_pairs.add(tuple(sorted((sample_id, neighbor_id))))
        chosen = neighbor_pairs

    return tuple(
        (left_id, right_id, similarities[(left_id, right_id)])
        for left_id, right_id in sorted(chosen)
    )


def build_graph_state(
    prepared: PreparedDataset,
    encoder_features: EncoderFeatureArtifact,
    config: GraphBuildConfig | None = None,
    *,
    radio_edge_weights: Mapping[tuple[str, str], float] | None = None,
) -> GraphState:
    """Build a stable sample/AP graph from matching prepared and AE artifacts.

    Bootstrap samples append their true bootstrap XY label. Update-batch samples
    never read ``evaluation_positions``: they receive a stable hash-seeded
    pseudo-location within explicit trusted bounds. Bootstrap bounds are derived
    from bootstrap labels only when the caller did not provide them.
    """
    selected_config = config or GraphBuildConfig()
    _validate_inputs(prepared, encoder_features)
    feature_by_scan_id = {
        scan_id: tuple(row)
        for scan_id, row in zip(encoder_features.scan_ids, encoder_features.feature_matrix)
    }
    scan_ids = tuple(scan.scan_id for scan in prepared.scans)
    reference_positions = (
        {position.scan_id: (position.x, position.y) for position in prepared.training_reference_positions()}
        if prepared.config.batch_role == "bootstrap_reference"
        else {}
    )
    selected_config = _resolve_pseudo_location_config(
        prepared, selected_config, reference_positions
    )

    sample_initializations: list[SampleLocationInitialization] = []
    coordinate_target_by_scan_id: dict[str, tuple[float, float] | None] = {}
    for scan in prepared.scans:
        if scan.scan_id in reference_positions:
            xy = reference_positions[scan.scan_id]
            source: Literal["bootstrap_reference_label", "seeded_pseudo_location"] = "bootstrap_reference_label"
            coordinate_target_by_scan_id[scan.scan_id] = xy
        else:
            bounds = selected_config.pseudo_location_bounds
            if bounds is None:
                raise GraphConstructionError(
                    "unlabeled sample initialization requires resolved pseudo-location bounds"
                )
            xy = _pseudo_location(scan, selected_config.pseudo_location_seed, bounds)
            source = "seeded_pseudo_location"
            coordinate_target_by_scan_id[scan.scan_id] = None
        sample_initializations.append(
            SampleLocationInitialization(scan_id=scan.scan_id, xy=xy, source=source)
        )

    sample_feature_by_id = {
        initialization.scan_id: feature_by_scan_id[initialization.scan_id] + initialization.xy
        for initialization in sample_initializations
    }
    scan_by_id = {scan.scan_id: scan for scan in prepared.scans}
    bssid_to_column = {bssid: column for column, bssid in enumerate(prepared.bssid_vocabulary)}
    ap_weight_by_scan: dict[tuple[str, str], tuple[float, float]] = {}
    if radio_edge_weights is None:
        for scan in prepared.scans:
            for column, bssid in enumerate(prepared.bssid_vocabulary):
                if not scan.observation_mask[column]:
                    continue
                weight = rssi_dbm_to_graph_weight(scan.rss_dbm[column], True)
                if weight is None or weight < 0.0:
                    raise GraphConstructionError(
                        f"RSS+120 must be nonnegative for graph edge {scan.scan_id} -> ap:{bssid}"
                    )
                ap_weight_by_scan[(scan.scan_id, f"ap:{bssid}")] = (weight, scan.rss_dbm[column])
    else:
        for (scan_id, bssid), raw_weight in radio_edge_weights.items():
            if scan_id not in scan_by_id:
                raise GraphConstructionError(f"radio edge references unknown sample ID: {scan_id}")
            if bssid not in bssid_to_column:
                raise GraphConstructionError(f"radio edge references BSSID outside vocabulary: {bssid}")
            if not math.isfinite(raw_weight) or raw_weight < 0.0:
                raise GraphConstructionError("radio edge scalar weights must be finite RSS+120 values")
            ap_id = f"ap:{bssid}"
            pair = (scan_id, ap_id)
            if pair in ap_weight_by_scan:
                raise GraphConstructionError(f"duplicate explicit radio edge: {pair}")
            ap_weight_by_scan[pair] = (float(raw_weight), float(raw_weight) - 120.0)

    observed_ap_node_ids = tuple(
        ap_id for ap_id in prepared.ap_ids
        if any(edge_ap_id == ap_id for _scan_id, edge_ap_id in ap_weight_by_scan)
    )
    observed_ap_set = set(observed_ap_node_ids)
    omitted_ap_ids = tuple(ap_id for ap_id in prepared.ap_ids if ap_id not in observed_ap_set)

    ap_feature_by_id: dict[str, tuple[float, ...]] = {}
    for ap_node_id in observed_ap_node_ids:
        weighted_rows: list[tuple[float, tuple[float, ...], str, float]] = []
        for (scan_id, edge_ap_id), (weight, rssi_dbm) in ap_weight_by_scan.items():
            if edge_ap_id == ap_node_id:
                weighted_rows.append((weight, sample_feature_by_id[scan_id], scan_id, rssi_dbm))
        total_weight = sum(item[0] for item in weighted_rows)
        if total_weight <= 0.0:
            raise GraphConstructionError(
                f"AP {ap_node_id} has observed edges but total RSS+120 weight is zero; "
                "the paper's weighted-mean initialization is undefined"
            )
        ap_feature_by_id[ap_node_id] = tuple(
            sum(weight * feature[column] for weight, feature, _scan_id, _rssi in weighted_rows)
            / total_weight
            for column in range(len(weighted_rows[0][1]))
        )
        for weight, _feature, scan_id, rssi_dbm in weighted_rows:
            ap_weight_by_scan[(scan_id, ap_node_id)] = (weight, rssi_dbm)

    node_ids = scan_ids + observed_ap_node_ids
    node_types: tuple[Literal["sample", "ap"], ...] = (
        ("sample",) * len(scan_ids) + ("ap",) * len(observed_ap_node_ids)
    )
    node_features = tuple(
        sample_feature_by_id[scan_id] for scan_id in scan_ids
    ) + tuple(ap_feature_by_id[ap_id] for ap_id in observed_ap_node_ids)
    id_to_row = {node_id: row for row, node_id in enumerate(node_ids)}

    radio_edges = []
    for (scan_id, ap_node_id), (weight, rssi_dbm) in sorted(ap_weight_by_scan.items()):
        radio_edges.append(
            RadioGraphEdge(
                edge_id=f"radio:{scan_id}|{ap_node_id}",
                sample_node_id=scan_id,
                ap_node_id=ap_node_id,
                rssi_dbm=rssi_dbm,
                scalar_weight=weight,
                edge_feature=_mean_features(
                    node_features[id_to_row[scan_id]], node_features[id_to_row[ap_node_id]]
                ),
            )
        )

    virtual_edges = []
    for left_id, right_id, similarity in _similarity_pairs(
        scan_ids, sample_feature_by_id, selected_config
    ):
        virtual_edges.append(
            VirtualGraphEdge(
                edge_id=f"virtual:{left_id}|{right_id}",
                sample_node_id_left=left_id,
                sample_node_id_right=right_id,
                cosine_similarity=similarity,
                scalar_weight=None,
                edge_feature=_mean_features(
                    node_features[id_to_row[left_id]], node_features[id_to_row[right_id]]
                ),
            )
        )

    reference_sample_mask = tuple(
        index < len(scan_ids) and prepared.config.batch_role == "bootstrap_reference"
        for index in range(len(node_ids))
    )
    new_sample_mask = tuple(
        index < len(scan_ids) and prepared.config.batch_role == "update_batch"
        for index in range(len(node_ids))
    )
    reference_label_mask = tuple(
        node_id in coordinate_target_by_scan_id and coordinate_target_by_scan_id[node_id] is not None
        for node_id in node_ids
    )
    coordinate_targets = tuple(
        coordinate_target_by_scan_id.get(node_id) for node_id in node_ids
    )

    source_prepared_digest = canonical_json_digest(prepared)
    encoder_artifact_digest = canonical_json_digest(encoder_features)
    graph_id = _graph_id(
        source_prepared_sha256=source_prepared_digest,
        encoder_feature_artifact_sha256=encoder_artifact_digest,
        build_config=selected_config,
        node_ids=node_ids,
    )
    return GraphState(
        graph_id=graph_id,
        dataset_id=prepared.config.dataset_id,
        batch_id=prepared.config.batch_id,
        batch_role=prepared.config.batch_role,
        source_prepared_sha256=source_prepared_digest,
        encoder_feature_artifact_sha256=encoder_artifact_digest,
        encoder_checkpoint_sha256=encoder_features.checkpoint_sha256,
        normalization=prepared.config.normalization_mode,
        bssid_vocabulary=prepared.bssid_vocabulary,
        ap_ids=prepared.ap_ids,
        vocabulary_sha256=encoder_features.vocabulary_sha256,
        build_config=selected_config,
        node_ids=node_ids,
        node_id_to_row=tuple((node_id, row) for node_id, row in id_to_row.items()),
        node_types=node_types,
        encoder_feature_dim=encoder_features.feature_dim,
        coordinate_feature_dim=2,
        node_feature_dim=encoder_features.feature_dim + 2,
        sample_node_ids=scan_ids,
        ap_node_ids=observed_ap_node_ids,
        omitted_unobserved_ap_ids=omitted_ap_ids,
        node_features=node_features,
        reference_sample_mask=reference_sample_mask,
        new_sample_mask=new_sample_mask,
        reference_label_mask=reference_label_mask,
        reference_coordinate_targets=coordinate_targets,
        sample_location_initializations=tuple(sample_initializations),
        radio_edges=tuple(radio_edges),
        virtual_edges=tuple(virtual_edges),
    )


def save_graph_state(
    graph_state: GraphState, path: str | Path, *, overwrite: bool = False
) -> Path:
    """Atomically persist a graph JSON artifact without accidental overwrite."""
    return atomic_write_json(graph_state, path, overwrite=overwrite)


def load_graph_state(path: str | Path) -> GraphState:
    """Load and validate a graph state, including row/edge identity contracts."""
    return GraphState.model_validate_json(read_json_text(path))
