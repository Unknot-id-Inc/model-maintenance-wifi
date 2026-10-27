"""Deterministic, checkpoint-feature-based link prediction for GUFU Sec. 6.

This implements the documented ``radio_normalized_plus_virtual_cosine_v1``
interpretation recorded in ``docs/gufu_repair_plan.md``. Virtual-edge scalar
weights and several Algorithm 2 choices are underspecified by the paper, so
this is a named inference, not an author-verified reproduction.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from graph_build import (
    GraphBuildConfig,
    GraphState,
    RadioGraphEdge,
    SampleLocationInitialization,
)
from gufu_artifacts import atomic_write_json, canonical_json_digest, read_json_text
from sage_ve import GNNFeatureArtifact


# Backwards-compatible defaults for older serialized artifacts. Runtime
# dimensions are taken from the validated graph/checkpoint artifacts.
LP_FEATURE_DIM = 34
LP_RESULT_FORMAT = "gufu_link_prediction_v1"
LP_GRAPH_FORMAT = "gufu_lp_graph_snapshot_v1"


class LinkPredictionError(ValueError):
    """Link prediction inputs or the bounded iteration failed validation."""


class LinkPredictionConfig(BaseModel):
    """Named iteration, scalar-weight, and decision policies."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    max_iterations: int = Field(default=2000, ge=1)
    convergence_epsilon: float = Field(default=1e-3, ge=0.0)
    link_threshold: float = Field(default=0.1, ge=0.0)
    threshold_source: Literal["implementation_detail_epsilon_as_raw_weight"] = (
        "implementation_detail_epsilon_as_raw_weight"
    )
    threshold_units: Literal["rssi_plus_120_scalar_weight"] = (
        "rssi_plus_120_scalar_weight"
    )
    iteration_weight_policy: Literal["radio_normalized_plus_virtual_cosine_v1"] = (
        "radio_normalized_plus_virtual_cosine_v1"
    )
    predicted_radio_weight_policy: Literal[
        "saturate_at_observed_max_v1"
    ] = "saturate_at_observed_max_v1"
    isolated_node_policy: Literal["keep_initial_g_f"] = "keep_initial_g_f"
    zero_radio_weight_policy: Literal["normalized_zero_and_zero_score_scale"] = (
        "normalized_zero_and_zero_score_scale"
    )
    retained_edge_policy: Literal["preserve_existing_weight"] = (
        "preserve_existing_weight"
    )
    symmetry_policy: Literal["one_undirected_pair_shared_weight"] = (
        "one_undirected_pair_shared_weight"
    )

    @model_validator(mode="after")
    def finite_parameters(self):
        if not math.isfinite(self.convergence_epsilon) or not math.isfinite(
            self.link_threshold
        ):
            raise ValueError("link prediction tolerances and thresholds must be finite")
        return self


class CandidateEdgeScore(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    edge_id: str
    sample_node_id: str
    ap_node_id: str
    predicted_weight: float
    existed_before: bool

    @model_validator(mode="after")
    def finite_score(self):
        if self.edge_id != f"radio:{self.sample_node_id}|{self.ap_node_id}":
            raise ValueError("candidate edge ID must preserve stable sample/AP identity")
        if not math.isfinite(self.predicted_weight):
            raise ValueError("candidate edge scores must be finite")
        return self


class FeatureTablePrediction(BaseModel):
    """Scores explicit candidate pairs over an ID-keyed learned feature table."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    outcome: Literal["converged", "failed"]
    iterations: int = Field(ge=0)
    goodness_residual: float = Field(ge=0.0)
    fairness_residual: float = Field(ge=0.0)
    radio_weight_max: float = Field(ge=0.0)
    scores: tuple[CandidateEdgeScore, ...]

    @model_validator(mode="after")
    def finite_result(self):
        values = (self.goodness_residual, self.fairness_residual, self.radio_weight_max)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("feature-table prediction state must be finite")
        if self.outcome == "failed" and self.scores:
            raise ValueError("failed feature-table prediction cannot expose edge scores")
        if any(
            score.predicted_weight > self.radio_weight_max + 1e-9
            for score in self.scores
        ):
            raise ValueError("predicted graph weights exceed the observed radio scale")
        return self


class LinkEdgeChange(BaseModel):
    """A single canonical undirected AP/sample edge delta entry."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    edge_id: str
    sample_node_id: str
    ap_node_id: str
    old_weight: float | None
    new_weight: float | None
    predicted_weight: float

    @model_validator(mode="after")
    def valid_edge_change(self):
        if self.edge_id != f"radio:{self.sample_node_id}|{self.ap_node_id}":
            raise ValueError("edge delta ID must preserve its stable sample/AP identity")
        if self.old_weight is None and self.new_weight is None:
            raise ValueError("edge delta must add, remove, or change a stored weight")
        values = [self.predicted_weight]
        if self.old_weight is not None:
            values.append(self.old_weight)
        if self.new_weight is not None:
            values.append(self.new_weight)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("edge delta weights must be finite")
        if self.old_weight is not None and self.old_weight < 0.0:
            raise ValueError("old physical edge weights cannot be negative")
        if self.new_weight is not None and self.new_weight < 0.0:
            raise ValueError("new physical edge weights must be nonnegative")
        return self


class UpdatedRadioEdge(BaseModel):
    """Radio edge in the post-LP learned-feature graph snapshot."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    edge_id: str
    sample_node_id: str
    ap_node_id: str
    rssi_dbm: float
    scalar_weight: float = Field(ge=0.0)
    edge_feature: tuple[float, ...]
    edge_feature_source: Literal[
        "gnn_learned_physical_edge", "gnn_endpoint_mean_initialization"
    ]

    @model_validator(mode="after")
    def valid_updated_edge(self):
        if self.edge_id != f"radio:{self.sample_node_id}|{self.ap_node_id}":
            raise ValueError("updated edge ID does not match endpoint IDs")
        if not math.isfinite(self.rssi_dbm) or not math.isfinite(self.scalar_weight):
            raise ValueError("updated edge RSS and scalar weight must be finite")
        if not math.isclose(self.scalar_weight, self.rssi_dbm + 120.0, abs_tol=1e-9):
            raise ValueError("updated edge must preserve the RSS+120 scalar convention")
        if not self.edge_feature or any(
            not math.isfinite(value) for value in self.edge_feature
        ):
            raise ValueError("updated learned edge feature must be a finite vector")
        return self


class UpdatedGraphSnapshot(BaseModel):
    """Complete immutable LP result graph; virtual edges have expired."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_lp_graph_snapshot_v1"] = LP_GRAPH_FORMAT
    graph_id: str = Field(pattern=r"^lpgraph:[0-9a-f]{64}$")
    parent_graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    gnn_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_id: str
    batch_id: str
    batch_role: Literal["bootstrap_reference", "update_batch"]
    normalization: Literal["paper_ae", "legacy_raw_div_120"]
    source_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_feature_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bssid_vocabulary: tuple[str, ...]
    ap_ids: tuple[str, ...]
    build_config: GraphBuildConfig
    feature_space: Literal["z_gnn_node"] = "z_gnn_node"
    node_feature_dim: int = Field(default=LP_FEATURE_DIM, ge=1)
    node_ids: tuple[str, ...]
    node_id_to_row: tuple[tuple[str, int], ...]
    node_types: tuple[Literal["sample", "ap"], ...]
    sample_node_ids: tuple[str, ...]
    sample_location_initializations: tuple[SampleLocationInitialization, ...]
    reference_sample_mask: tuple[bool, ...]
    new_sample_mask: tuple[bool, ...]
    reference_label_mask: tuple[bool, ...]
    reference_coordinate_targets: tuple[tuple[float, float] | None, ...]
    active_ap_node_ids: tuple[str, ...]
    omitted_unobserved_ap_ids: tuple[str, ...]
    removed_ap_node_ids: tuple[str, ...]
    node_features: tuple[tuple[float, ...], ...]
    radio_edges: tuple[UpdatedRadioEdge, ...]
    virtual_edges_removed_after_prediction: Literal[True] = True
    symmetry_policy: Literal["one_undirected_pair_shared_weight"] = (
        "one_undirected_pair_shared_weight"
    )

    @model_validator(mode="after")
    def valid_updated_graph(self):
        if len(self.ap_ids) != len(self.bssid_vocabulary):
            raise ValueError("updated graph AP IDs must align with its full vocabulary")
        if self.ap_ids != tuple(f"ap:{bssid}" for bssid in self.bssid_vocabulary):
            raise ValueError("updated graph must preserve full BSSID identities")
        if canonical_json_digest(self.bssid_vocabulary) != self.vocabulary_sha256:
            raise ValueError("updated graph vocabulary digest is inconsistent")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("updated graph node IDs must be unique")
        if self.node_ids != self.sample_node_ids + self.active_ap_node_ids:
            raise ValueError("updated graph rows must be samples then active APs")
        if self.node_id_to_row != tuple(
            (node_id, row) for row, node_id in enumerate(self.node_ids)
        ):
            raise ValueError("updated graph row mapping must align with node IDs")
        if self.node_types != ("sample",) * len(self.sample_node_ids) + (
            "ap",
        ) * len(self.active_ap_node_ids):
            raise ValueError("updated graph node types must align with row order")
        node_count = len(self.node_ids)
        if any(
            len(values) != node_count
            for values in (
                self.reference_sample_mask,
                self.new_sample_mask,
                self.reference_label_mask,
                self.reference_coordinate_targets,
            )
        ):
            raise ValueError("updated graph role and label metadata must align with nodes")
        if len(self.sample_location_initializations) != len(self.sample_node_ids):
            raise ValueError("updated graph must retain every sample location initialization")
        if tuple(item.scan_id for item in self.sample_location_initializations) != self.sample_node_ids:
            raise ValueError("sample location initializations must preserve scan identity")
        for row, node_id in enumerate(self.node_ids):
            is_sample = row < len(self.sample_node_ids)
            reference = self.reference_sample_mask[row]
            new_sample = self.new_sample_mask[row]
            label = self.reference_label_mask[row]
            target = self.reference_coordinate_targets[row]
            if is_sample:
                if reference == new_sample:
                    raise ValueError("each updated sample must remain reference or new")
                if reference != (self.batch_role == "bootstrap_reference"):
                    raise ValueError("updated graph reference mask disagrees with batch role")
                if (target is not None) != label or (label and (not reference or target is None)):
                    raise ValueError("updated graph labels must preserve bootstrap-only targets")
                if target is not None and any(not math.isfinite(value) for value in target):
                    raise ValueError("updated graph coordinate targets must be finite")
            elif reference or new_sample or label or target is not None:
                raise ValueError("AP rows cannot carry sample labels or roles")
        if len(self.node_features) != len(self.node_ids) or any(
            len(row) != self.node_feature_dim for row in self.node_features
        ):
            raise ValueError("updated graph node feature width disagrees with node_feature_dim")
        if any(not math.isfinite(value) for row in self.node_features for value in row):
            raise ValueError("updated graph features must be finite")
        all_ap_sets = (
            set(self.active_ap_node_ids),
            set(self.omitted_unobserved_ap_ids),
            set(self.removed_ap_node_ids),
        )
        if any(all_ap_sets[left] & all_ap_sets[right] for left in range(3) for right in range(left + 1, 3)):
            raise ValueError("active, omitted, and removed AP node sets must not overlap")
        if set.union(*all_ap_sets) != set(self.ap_ids):
            raise ValueError("updated graph must retain active, omitted, and removed AP identities")
        if self.active_ap_node_ids != tuple(
            ap_id for ap_id in self.ap_ids if ap_id in set(self.active_ap_node_ids)
        ):
            raise ValueError("active AP rows must preserve original vocabulary order")
        if self.omitted_unobserved_ap_ids != tuple(
            ap_id for ap_id in self.ap_ids if ap_id in set(self.omitted_unobserved_ap_ids)
        ):
            raise ValueError("omitted AP identities must preserve original vocabulary order")
        if self.removed_ap_node_ids != tuple(
            ap_id for ap_id in self.ap_ids if ap_id in set(self.removed_ap_node_ids)
        ):
            raise ValueError("removed AP identities must preserve original vocabulary order")

        node_set = set(self.node_ids)
        seen_edge_ids: set[str] = set()
        seen_pairs: set[tuple[str, str]] = set()
        for edge in self.radio_edges:
            if len(edge.edge_feature) != self.node_feature_dim:
                raise ValueError("updated radio edge feature width must match node_feature_dim")
            if edge.edge_id in seen_edge_ids:
                raise ValueError("updated physical edge IDs must be unique")
            seen_edge_ids.add(edge.edge_id)
            pair = edge.sample_node_id, edge.ap_node_id
            if pair in seen_pairs:
                raise ValueError("updated physical pairs must be stored once")
            seen_pairs.add(pair)
            if edge.sample_node_id not in self.sample_node_ids or edge.ap_node_id not in node_set:
                raise ValueError("updated radio edges must connect active sample/AP nodes")
        if self.radio_edges != tuple(
            sorted(self.radio_edges, key=lambda edge: (edge.sample_node_id, edge.ap_node_id))
        ):
            raise ValueError("updated radio edges must use deterministic endpoint order")
        identity = _updated_graph_digest(self)
        if self.graph_id != f"lpgraph:{identity}":
            raise ValueError("updated graph ID does not match its source, features, and edges")
        return self

    def directed_radio_edges(self) -> tuple[tuple[str, str, float], ...]:
        """Expand undirected radio pairs into two arcs sharing the same weight."""
        return tuple(
            arc
            for edge in self.radio_edges
            for arc in (
                (edge.sample_node_id, edge.ap_node_id, edge.scalar_weight),
                (edge.ap_node_id, edge.sample_node_id, edge.scalar_weight),
            )
        )


class EdgeDelta(BaseModel):
    """ID-aligned link scores, convergence result, and explicit graph delta."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_link_prediction_v1"] = LP_RESULT_FORMAT
    parent_graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    dataset_id: str
    batch_id: str
    gnn_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: LinkPredictionConfig
    outcome: Literal["converged", "no_change", "failed"]
    termination: Literal["residual_tolerance", "max_iterations"]
    iterations: int = Field(ge=0)
    goodness_residual: float = Field(ge=0.0)
    fairness_residual: float = Field(ge=0.0)
    radio_weight_max: float = Field(ge=0.0)
    threshold_value: float = Field(ge=0.0)
    threshold_units: Literal["rssi_plus_120_scalar_weight"]
    node_feature_dim: int = Field(default=LP_FEATURE_DIM, ge=1)
    node_ids: tuple[str, ...]
    sample_node_ids: tuple[str, ...]
    goodness_features: tuple[tuple[float, ...], ...]
    fairness_features: tuple[tuple[float, ...], ...]
    candidate_scores: tuple[CandidateEdgeScore, ...]
    added_edges: tuple[LinkEdgeChange, ...]
    removed_edges: tuple[LinkEdgeChange, ...]
    changed_edges: tuple[LinkEdgeChange, ...]
    ap_removals: tuple[str, ...]
    updated_graph: UpdatedGraphSnapshot | None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def valid_result(self):
        if not math.isfinite(self.goodness_residual) or not math.isfinite(
            self.fairness_residual
        ):
            raise ValueError("link prediction residuals must be finite")
        if not math.isfinite(self.radio_weight_max) or not math.isfinite(
            self.threshold_value
        ):
            raise ValueError("link prediction weight scale and threshold must be finite")
        if self.threshold_value != self.config.link_threshold or self.threshold_units != self.config.threshold_units:
            raise ValueError("recorded edge threshold must match the explicit config and units")
        if len(self.goodness_features) != len(self.node_ids) or len(
            self.fairness_features
        ) != len(self.node_ids):
            raise ValueError("G/F feature rows must align with result node IDs")
        if self.node_ids[: len(self.sample_node_ids)] != self.sample_node_ids:
            raise ValueError("result sample IDs must retain source node order")
        if any(len(row) != self.node_feature_dim for row in self.goodness_features) or any(
            len(row) != self.node_feature_dim for row in self.fairness_features
        ):
            raise ValueError("G/F feature matrices must match node_feature_dim")
        if any(
            not math.isfinite(value)
            for matrix in (self.goodness_features, self.fairness_features)
            for row in matrix
            for value in row
        ):
            raise ValueError("G/F features must be finite")
        score_keys = tuple((row.sample_node_id, row.ap_node_id) for row in self.candidate_scores)
        if len(set(score_keys)) != len(score_keys):
            raise ValueError("candidate score pairs must be unique")
        for changes in (self.added_edges, self.removed_edges, self.changed_edges):
            keys = tuple((row.sample_node_id, row.ap_node_id) for row in changes)
            if len(set(keys)) != len(keys):
                raise ValueError("each edge delta category must contain unique pairs")
        all_changes = [
            (row.sample_node_id, row.ap_node_id)
            for rows in (self.added_edges, self.removed_edges, self.changed_edges)
            for row in rows
        ]
        if len(set(all_changes)) != len(all_changes):
            raise ValueError("an edge pair cannot occur in multiple delta categories")
        scores_by_pair = {
            (score.sample_node_id, score.ap_node_id): score
            for score in self.candidate_scores
        }
        if any(
            score.predicted_weight > self.radio_weight_max + 1e-9
            for score in self.candidate_scores
        ):
            raise ValueError("predicted graph weights exceed the observed radio scale")
        for change in self.added_edges:
            if change.old_weight is not None or change.new_weight is None:
                raise ValueError("added edge must have only a new weight")
            score = scores_by_pair.get((change.sample_node_id, change.ap_node_id))
            if (
                score is None
                or score.existed_before
                or score.predicted_weight < self.threshold_value
                or not math.isclose(score.predicted_weight, change.new_weight, abs_tol=1e-9)
            ):
                raise ValueError("added edge disagrees with its supported score/threshold")
        for change in self.removed_edges:
            if change.old_weight is None or change.new_weight is not None:
                raise ValueError("removed edge must have only its old weight")
            score = scores_by_pair.get((change.sample_node_id, change.ap_node_id))
            if (
                score is None
                or not score.existed_before
                or score.predicted_weight >= self.threshold_value
            ):
                raise ValueError("removed edge disagrees with its existing-edge score/threshold")
        if self.changed_edges:
            raise ValueError(
                "the selected preserve_existing_weight policy does not emit changed edges"
            )
        if self.outcome == "failed":
            if self.termination != "max_iterations" or self.updated_graph is not None:
                raise ValueError("failed iteration must not expose an applicable graph delta")
            if self.added_edges or self.removed_edges or self.changed_edges or self.ap_removals:
                raise ValueError("failed iteration must not expose partial graph changes")
            if not self.failure_reason:
                raise ValueError("failed iteration needs an explicit failure reason")
        else:
            if self.termination != "residual_tolerance" or self.updated_graph is None:
                raise ValueError("completed link prediction must include its updated graph")
            if self.failure_reason is not None:
                raise ValueError("successful result cannot include a failure reason")
            if self.outcome == "no_change" and (
                self.added_edges or self.removed_edges or self.changed_edges or self.ap_removals
            ):
                raise ValueError("no_change outcome cannot contain edge or AP changes")
            if self.outcome == "converged" and not (
                self.added_edges or self.removed_edges or self.changed_edges or self.ap_removals
            ):
                raise ValueError("converged outcome must contain an applied change")
            if self.updated_graph.parent_graph_id != self.parent_graph_id:
                raise ValueError("updated graph parent ID does not match result")
            if self.updated_graph.node_feature_dim != self.node_feature_dim:
                raise ValueError("updated graph feature width does not match link result")
            if self.updated_graph.gnn_checkpoint_sha256 != self.gnn_checkpoint_sha256:
                raise ValueError("updated graph checkpoint does not match result")
            if (
                self.updated_graph.dataset_id != self.dataset_id
                or self.updated_graph.batch_id != self.batch_id
                or self.updated_graph.vocabulary_sha256 != self.vocabulary_sha256
            ):
                raise ValueError("updated graph data/vocabulary provenance does not match result")
            if self.updated_graph.sample_node_ids != self.sample_node_ids:
                raise ValueError("updated graph must preserve all sample IDs in source order")
            if self.ap_removals != self.updated_graph.removed_ap_node_ids:
                raise ValueError("AP-removal delta must match updated graph tombstones")
            updated_edges = {
                (edge.sample_node_id, edge.ap_node_id): edge
                for edge in self.updated_graph.radio_edges
            }
            for change in self.added_edges + self.changed_edges:
                edge = updated_edges.get((change.sample_node_id, change.ap_node_id))
                if edge is None or not math.isclose(
                    edge.scalar_weight, change.new_weight, abs_tol=1e-9
                ):
                    raise ValueError("added/changed edge is missing from updated graph")
            if any(
                (change.sample_node_id, change.ap_node_id) in updated_edges
                for change in self.removed_edges
            ):
                raise ValueError("removed edge remains in updated graph")
        return self


def _updated_graph_digest(graph: UpdatedGraphSnapshot | dict) -> str:
    values = graph if isinstance(graph, dict) else graph.model_dump(mode="json")

    def jsonable(value):
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, dict):
            return {key: jsonable(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [jsonable(item) for item in value]
        return value

    identity_fields = (
        "parent_graph_id",
        "gnn_checkpoint_sha256",
        "dataset_id",
        "batch_id",
        "batch_role",
        "normalization",
        "source_prepared_sha256",
        "encoder_feature_artifact_sha256",
        "encoder_checkpoint_sha256",
        "vocabulary_sha256",
        "bssid_vocabulary",
        "ap_ids",
        "build_config",
        "node_feature_dim",
        "node_ids",
        "node_types",
        "node_features",
        "sample_location_initializations",
        "reference_sample_mask",
        "new_sample_mask",
        "reference_label_mask",
        "reference_coordinate_targets",
        "omitted_unobserved_ap_ids",
        "removed_ap_node_ids",
        "radio_edges",
    )
    return canonical_json_digest(
        {field: jsonable(values[field]) for field in identity_fields}
    )


def _validate_inputs(graph: GraphState, learned: GNNFeatureArtifact) -> None:
    if not isinstance(graph, GraphState):
        raise TypeError("predict_edges requires a typed GraphState")
    if not isinstance(learned, GNNFeatureArtifact):
        raise TypeError("predict_edges requires a loaded GNNFeatureArtifact; random features are forbidden")
    pairs = (
        (learned.graph_id, graph.graph_id, "graph ID"),
        (learned.dataset_id, graph.dataset_id, "dataset identity"),
        (learned.batch_id, graph.batch_id, "batch identity"),
        (learned.source_prepared_sha256, graph.source_prepared_sha256, "prepared-data identity"),
        (learned.encoder_feature_artifact_sha256, graph.encoder_feature_artifact_sha256, "encoder artifact"),
        (learned.encoder_checkpoint_sha256, graph.encoder_checkpoint_sha256, "encoder checkpoint"),
        (learned.normalization, graph.normalization, "normalization"),
        (learned.vocabulary_sha256, graph.vocabulary_sha256, "vocabulary hash"),
        (learned.bssid_vocabulary, graph.bssid_vocabulary, "ordered vocabulary"),
        (learned.graph_build_config, graph.build_config, "graph construction config"),
        (learned.node_ids, graph.node_ids, "node ID order"),
        (learned.node_types, graph.node_types, "node types"),
        (learned.node_id_to_row, graph.node_id_to_row, "node row mapping"),
        (
            learned.radio_edge_ids,
            tuple(edge.edge_id for edge in graph.radio_edges),
            "physical edge identity/order",
        ),
        (
            learned.virtual_edge_ids,
            tuple(edge.edge_id for edge in graph.virtual_edges),
            "temporary virtual edge identity/order",
        ),
    )
    for actual, expected, name in pairs:
        if actual != expected:
            raise LinkPredictionError(f"learned GNN state has incompatible {name}")
    if learned.feature_space != "z_gnn_node" or learned.node_feature_dim != graph.node_feature_dim:
        raise LinkPredictionError("link prediction requires graph-width learned z_gnn_node features")
    if len(learned.node_feature_matrix) != len(graph.node_ids):
        raise LinkPredictionError("learned node features do not cover every graph node")
    if len(learned.radio_edge_feature_matrix) != len(graph.radio_edges):
        raise LinkPredictionError("learned physical-edge vectors do not align with graph edges")
    if graph.node_ids != graph.sample_node_ids + graph.ap_node_ids:
        raise LinkPredictionError("graph has an invalid sample/AP node ordering")


def _neighbors_and_normalized_weights(
    graph: GraphState, radio_weight_max: float
) -> dict[str, tuple[tuple[str, float], ...]]:
    neighbors: dict[str, list[tuple[str, float]]] = {node_id: [] for node_id in graph.node_ids}
    for edge in graph.radio_edges:
        weight = edge.scalar_weight / radio_weight_max if radio_weight_max > 0.0 else 0.0
        neighbors[edge.sample_node_id].append((edge.ap_node_id, weight))
        neighbors[edge.ap_node_id].append((edge.sample_node_id, weight))
    # The paper creates virtual edges from cosine-thresholded sample similarity
    # but does not assign them scalar weights. Keep GraphState.scalar_weight=None
    # and use that observed cosine as this named LP-only normalized weight.
    for edge in graph.virtual_edges:
        weight = edge.cosine_similarity
        neighbors[edge.sample_node_id_left].append((edge.sample_node_id_right, weight))
        neighbors[edge.sample_node_id_right].append((edge.sample_node_id_left, weight))
    return {
        node_id: tuple(sorted(rows, key=lambda item: item[0]))
        for node_id, rows in neighbors.items()
    }


def _update_goodness_fairness(
    graph: GraphState,
    learned: GNNFeatureArtifact,
    config: LinkPredictionConfig,
) -> tuple[
    tuple[tuple[float, ...], ...],
    tuple[tuple[float, ...], ...],
    int,
    float,
    float,
    bool,
]:
    node_ids = graph.node_ids
    row_by_id = {node_id: row for row, node_id in enumerate(node_ids)}
    radio_weight_max = max((edge.scalar_weight for edge in graph.radio_edges), default=0.0)
    neighbors = _neighbors_and_normalized_weights(graph, radio_weight_max)

    # Separate mutable buffers preserve the independent G(0)=F(0)=Z contract.
    goodness = [list(row) for row in learned.node_feature_matrix]
    fairness = [list(row) for row in learned.node_feature_matrix]
    g_residual = 0.0
    f_residual = 0.0
    iterations = 0
    converged = not any(neighbors.values())

    while not converged and iterations < config.max_iterations:
        next_goodness = [row[:] for row in goodness]
        next_fairness = [row[:] for row in fairness]

        for node_id in node_ids:
            adjacent = neighbors[node_id]
            if not adjacent:
                # Degree-zero equations are undefined; the recorded policy is
                # to preserve both independent initial vectors for this node.
                continue
            row = row_by_id[node_id]
            degree = len(adjacent)
            next_goodness[row] = [
                sum(fairness[row_by_id[neighbor_id]][column] * weight
                    for neighbor_id, weight in adjacent)
                / degree
                for column in range(graph.node_feature_dim)
            ]

        for node_id in node_ids:
            adjacent = neighbors[node_id]
            if not adjacent:
                continue
            row = row_by_id[node_id]
            degree = len(adjacent)
            next_fairness[row] = [
                1.0
                - sum(
                    abs(weight - next_goodness[row_by_id[neighbor_id]][column])
                    for neighbor_id, weight in adjacent
                )
                / (2.0 * degree)
                for column in range(graph.node_feature_dim)
            ]

        g_residual = sum(
            math.sqrt(sum((new - old) ** 2 for new, old in zip(g_row, old_row)))
            for g_row, old_row in zip(next_goodness, goodness)
        )
        f_residual = sum(
            math.sqrt(sum((new - old) ** 2 for new, old in zip(f_row, old_row)))
            for f_row, old_row in zip(next_fairness, fairness)
        )
        if not math.isfinite(g_residual) or not math.isfinite(f_residual):
            raise LinkPredictionError("goodness/fairness iteration produced a non-finite residual")
        if any(
            not math.isfinite(value)
            for matrix in (next_goodness, next_fairness)
            for row in matrix
            for value in row
        ):
            raise LinkPredictionError("goodness/fairness iteration produced a non-finite feature")
        goodness, fairness = next_goodness, next_fairness
        iterations += 1
        # Algorithm 2 continues only while BOTH residual sums exceed epsilon.
        converged = (
            g_residual <= config.convergence_epsilon
            or f_residual <= config.convergence_epsilon
        )

    return (
        tuple(tuple(row) for row in goodness),
        tuple(tuple(row) for row in fairness),
        iterations,
        g_residual,
        f_residual,
        converged,
    )


def _candidate_pairs(graph: GraphState) -> tuple[tuple[str, str], ...]:
    existing = {(edge.sample_node_id, edge.ap_node_id) for edge in graph.radio_edges}
    ap_by_sample: dict[str, set[str]] = {sample_id: set() for sample_id in graph.sample_node_ids}
    for edge in graph.radio_edges:
        ap_by_sample[edge.sample_node_id].add(edge.ap_node_id)

    # Missing candidates: AP --(radio)--> sample u --(virtual)--> sample v.
    supported_missing: set[tuple[str, str]] = set()
    for virtual in graph.virtual_edges:
        for ap_id in ap_by_sample[virtual.sample_node_id_left]:
            pair = virtual.sample_node_id_right, ap_id
            if pair not in existing:
                supported_missing.add(pair)
        for ap_id in ap_by_sample[virtual.sample_node_id_right]:
            pair = virtual.sample_node_id_left, ap_id
            if pair not in existing:
                supported_missing.add(pair)

    # Existing edges are evaluated separately so the Section 6.2 forgetting
    # rule remains reachable despite Algorithm 2's contradictory non-edge gate.
    return tuple(sorted(existing | supported_missing))


def _predicted_weight(
    sample_id: str,
    ap_id: str,
    row_by_id: dict[str, int],
    goodness: tuple[tuple[float, ...], ...],
    fairness: tuple[tuple[float, ...], ...],
    radio_weight_max: float,
) -> float:
    sample_row, ap_row = row_by_id[sample_id], row_by_id[ap_id]
    left = sum(
        goodness[sample_row][index] * fairness[ap_row][index]
        for index in range(len(goodness[sample_row]))
    )
    right = sum(
        fairness[sample_row][index] * goodness[ap_row][index]
        for index in range(len(goodness[sample_row]))
    )
    return _bounded_predicted_radio_weight(left, right, radio_weight_max)


def _bounded_predicted_radio_weight(
    left_score: float, right_score: float, radio_weight_max: float
) -> float:
    """Convert the symmetrized G/F score to a bounded graph edge weight.

    The raw dot-product score is unbounded, while downstream graph and
    propagation scales are tied to observed radio weights. Preserve scores
    within that scale and saturate positive outliers at its maximum. This is a
    graph stability bound; the resulting weight is not converted into an RSS
    measurement or supplied to the AE. Negative scores remain below any
    configured nonnegative edge threshold and are never stored as edges.
    """
    if not all(
        math.isfinite(value)
        for value in (left_score, right_score, radio_weight_max)
    ):
        raise LinkPredictionError("predicted radio score inputs must be finite")
    if radio_weight_max < 0.0:
        raise LinkPredictionError("maximum observed radio weight cannot be negative")
    raw_weight = radio_weight_max * (left_score + right_score) / 2.0
    if not math.isfinite(raw_weight):
        raise LinkPredictionError("predicted radio weight is non-finite")
    return min(raw_weight, radio_weight_max)


def score_candidate_pairs_from_features(
    *,
    node_ids: tuple[str, ...],
    node_feature_matrix: tuple[tuple[float, ...], ...],
    sample_node_ids: tuple[str, ...],
    ap_node_ids: tuple[str, ...],
    radio_edges: tuple[tuple[str, str, float], ...],
    virtual_edges: tuple[tuple[str, str, float], ...],
    candidate_pairs: tuple[tuple[str, str], ...],
    config: LinkPredictionConfig | None = None,
) -> FeatureTablePrediction:
    """Apply the milestone 5 Eq. (11)–(13) policy to a transient feature table.

    This is used when Section 6 adds AP nodes after shared-AP inference: the
    transient graph is keyed by stable IDs but is not yet a persistent
    ``GraphState``/``GNNFeatureArtifact`` pair. Feature rows must still be real
    learned outputs or the documented AP weighted-mean initialization.
    """
    selected = config or LinkPredictionConfig()
    if not node_ids or len(set(node_ids)) != len(node_ids):
        raise LinkPredictionError("transient feature graph needs unique node IDs")
    feature_dim = len(node_feature_matrix[0]) if node_feature_matrix else 0
    if len(node_feature_matrix) != len(node_ids) or feature_dim < 1 or any(
        len(row) != feature_dim for row in node_feature_matrix
    ):
        raise LinkPredictionError("transient learned feature matrix must have a consistent positive width")
    if any(not math.isfinite(value) for row in node_feature_matrix for value in row):
        raise LinkPredictionError("transient learned feature matrix contains non-finite values")
    if len(set(sample_node_ids)) != len(sample_node_ids) or len(set(ap_node_ids)) != len(ap_node_ids):
        raise LinkPredictionError("transient sample/AP IDs must be unique")
    if set(sample_node_ids) & set(ap_node_ids) or set(sample_node_ids) | set(ap_node_ids) != set(node_ids):
        raise LinkPredictionError("transient node IDs must partition into samples and APs")
    sample_set, ap_set = set(sample_node_ids), set(ap_node_ids)
    row_by_id = {node_id: row for row, node_id in enumerate(node_ids)}

    normalized_radio: dict[tuple[str, str], float] = {}
    max_weight = 0.0
    for sample_id, ap_id, weight in radio_edges:
        if sample_id not in sample_set or ap_id not in ap_set:
            raise LinkPredictionError("transient radio edges must connect a sample to an AP")
        if not math.isfinite(weight) or weight < 0.0:
            raise LinkPredictionError("transient radio weights must be finite and nonnegative")
        key = sample_id, ap_id
        if key in normalized_radio:
            raise LinkPredictionError("transient radio pairs must be unique")
        max_weight = max(max_weight, weight)
        normalized_radio[key] = weight

    neighbors: dict[str, list[tuple[str, float]]] = {node_id: [] for node_id in node_ids}
    for (sample_id, ap_id), weight in normalized_radio.items():
        normalized = weight / max_weight if max_weight > 0.0 else 0.0
        neighbors[sample_id].append((ap_id, normalized))
        neighbors[ap_id].append((sample_id, normalized))
    virtual_seen: set[tuple[str, str]] = set()
    for left_id, right_id, cosine in virtual_edges:
        pair = tuple(sorted((left_id, right_id)))
        if left_id not in sample_set or right_id not in sample_set or left_id == right_id:
            raise LinkPredictionError("transient virtual edges must join distinct sample IDs")
        if pair in virtual_seen:
            raise LinkPredictionError("transient virtual edges must be unique")
        if not math.isfinite(cosine) or not -1.0 <= cosine <= 1.0:
            raise LinkPredictionError("transient virtual cosine must be finite in [-1,1]")
        virtual_seen.add(pair)
        neighbors[left_id].append((right_id, cosine))
        neighbors[right_id].append((left_id, cosine))

    pairs = tuple(sorted(candidate_pairs))
    if len(set(pairs)) != len(pairs):
        raise LinkPredictionError("transient candidate pairs must be unique")
    for sample_id, ap_id in pairs:
        if sample_id not in sample_set or ap_id not in ap_set:
            raise LinkPredictionError("transient candidates must be sample/AP IDs")

    goodness = [list(row) for row in node_feature_matrix]
    fairness = [list(row) for row in node_feature_matrix]
    iterations = 0
    g_residual = f_residual = 0.0
    converged = not any(neighbors.values())
    while not converged and iterations < selected.max_iterations:
        next_g = [row[:] for row in goodness]
        next_f = [row[:] for row in fairness]
        for node_id in node_ids:
            adjacent = neighbors[node_id]
            if not adjacent:
                continue
            row = row_by_id[node_id]
            degree = len(adjacent)
            next_g[row] = [
                sum(fairness[row_by_id[neighbor]][dim] * weight for neighbor, weight in adjacent)
                / degree
                for dim in range(feature_dim)
            ]
        for node_id in node_ids:
            adjacent = neighbors[node_id]
            if not adjacent:
                continue
            row = row_by_id[node_id]
            degree = len(adjacent)
            next_f[row] = [
                1.0
                - sum(abs(weight - next_g[row_by_id[neighbor]][dim]) for neighbor, weight in adjacent)
                / (2.0 * degree)
                for dim in range(feature_dim)
            ]
        g_residual = sum(
            math.sqrt(sum((new - old) ** 2 for new, old in zip(new_row, old_row)))
            for new_row, old_row in zip(next_g, goodness)
        )
        f_residual = sum(
            math.sqrt(sum((new - old) ** 2 for new, old in zip(new_row, old_row)))
            for new_row, old_row in zip(next_f, fairness)
        )
        if not math.isfinite(g_residual) or not math.isfinite(f_residual):
            raise LinkPredictionError("transient goodness/fairness residual is non-finite")
        goodness, fairness = next_g, next_f
        iterations += 1
        converged = (
            g_residual <= selected.convergence_epsilon
            or f_residual <= selected.convergence_epsilon
        )

    if not converged:
        return FeatureTablePrediction(
            outcome="failed",
            iterations=iterations,
            goodness_residual=g_residual,
            fairness_residual=f_residual,
            radio_weight_max=max_weight,
            scores=(),
        )
    scores = tuple(
        CandidateEdgeScore(
            edge_id=f"radio:{sample_id}|{ap_id}",
            sample_node_id=sample_id,
            ap_node_id=ap_id,
            predicted_weight=_predicted_weight(
                sample_id,
                ap_id,
                row_by_id,
                tuple(tuple(row) for row in goodness),
                tuple(tuple(row) for row in fairness),
                max_weight,
            ),
            existed_before=(sample_id, ap_id) in normalized_radio,
        )
        for sample_id, ap_id in pairs
    )
    return FeatureTablePrediction(
        outcome="converged",
        iterations=iterations,
        goodness_residual=g_residual,
        fairness_residual=f_residual,
        radio_weight_max=max_weight,
        scores=scores,
    )


def _make_updated_graph(
    graph: GraphState,
    learned: GNNFeatureArtifact,
    delta_fields: dict,
) -> UpdatedGraphSnapshot:
    if delta_fields["outcome"] == "failed":
        raise LinkPredictionError("cannot apply a failed link prediction result")

    changes_by_pair = {
        (change.sample_node_id, change.ap_node_id): change
        for name in ("added_edges", "removed_edges", "changed_edges")
        for change in delta_fields[name]
    }
    original_by_pair = {
        (edge.sample_node_id, edge.ap_node_id): edge for edge in graph.radio_edges
    }
    for pair, change in changes_by_pair.items():
        if pair[0] not in graph.sample_node_ids or pair[1] not in graph.ap_node_ids:
            raise LinkPredictionError(f"edge delta references unknown endpoint IDs: {pair}")
        exists = pair in original_by_pair
        if change.old_weight is None and exists:
            raise LinkPredictionError(f"edge delta attempts to add existing pair: {pair}")
        if change.old_weight is not None:
            if not exists or not math.isclose(
                original_by_pair[pair].scalar_weight, change.old_weight, abs_tol=1e-9
            ):
                raise LinkPredictionError(f"edge delta old weight does not match source graph: {pair}")

    result_weight_by_pair = {
        pair: edge.scalar_weight for pair, edge in original_by_pair.items()
    }
    for pair, change in changes_by_pair.items():
        if change.new_weight is None:
            result_weight_by_pair.pop(pair, None)
        else:
            result_weight_by_pair[pair] = change.new_weight

    orphan_aps = tuple(
        ap_id
        for ap_id in graph.ap_node_ids
        if not any(pair[1] == ap_id for pair in result_weight_by_pair)
    )
    expected_ap_removals = tuple(
        ap_id for ap_id in graph.ap_node_ids if ap_id in set(orphan_aps)
    )
    if tuple(delta_fields["ap_removals"]) != expected_ap_removals:
        raise LinkPredictionError("AP-removal list does not match the resulting radio-edge state")

    active_aps = tuple(ap_id for ap_id in graph.ap_node_ids if ap_id not in set(orphan_aps))
    node_ids = graph.sample_node_ids + active_aps
    original_rows = {node_id: row for row, node_id in enumerate(learned.node_ids)}
    node_features = tuple(
        tuple(learned.node_feature_matrix[original_rows[node_id]]) for node_id in node_ids
    )
    graph_rows = graph.id_to_row
    retained_rows = tuple(graph_rows[node_id] for node_id in node_ids)
    reference_sample_mask = tuple(graph.reference_sample_mask[row] for row in retained_rows)
    new_sample_mask = tuple(graph.new_sample_mask[row] for row in retained_rows)
    reference_label_mask = tuple(graph.reference_label_mask[row] for row in retained_rows)
    reference_coordinate_targets = tuple(
        graph.reference_coordinate_targets[row] for row in retained_rows
    )
    row_by_id = {node_id: row for row, node_id in enumerate(node_ids)}
    learned_radio_vectors = {
        edge_id: tuple(row)
        for edge_id, row in zip(
            learned.radio_edge_ids, learned.radio_edge_feature_matrix
        )
    }
    edges: list[UpdatedRadioEdge] = []
    for pair, weight in sorted(result_weight_by_pair.items()):
        sample_id, ap_id = pair
        source_edge = original_by_pair.get(pair)
        if source_edge is None:
            edge_feature = tuple(
                (left + right) / 2.0
                for left, right in zip(
                    node_features[row_by_id[sample_id]], node_features[row_by_id[ap_id]]
                )
            )
            edge_feature_source = "gnn_endpoint_mean_initialization"
        else:
            edge_feature = learned_radio_vectors[source_edge.edge_id]
            edge_feature_source = "gnn_learned_physical_edge"
        edges.append(
            UpdatedRadioEdge(
                edge_id=f"radio:{sample_id}|{ap_id}",
                sample_node_id=sample_id,
                ap_node_id=ap_id,
                rssi_dbm=weight - 120.0,
                scalar_weight=weight,
                edge_feature=edge_feature,
                edge_feature_source=edge_feature_source,
            )
        )

    payload = {
        "parent_graph_id": graph.graph_id,
        "gnn_checkpoint_sha256": learned.checkpoint_sha256,
        "dataset_id": graph.dataset_id,
        "batch_id": graph.batch_id,
        "batch_role": graph.batch_role,
        "normalization": graph.normalization,
        "source_prepared_sha256": graph.source_prepared_sha256,
        "encoder_feature_artifact_sha256": graph.encoder_feature_artifact_sha256,
        "encoder_checkpoint_sha256": graph.encoder_checkpoint_sha256,
        "vocabulary_sha256": graph.vocabulary_sha256,
        "bssid_vocabulary": graph.bssid_vocabulary,
        "ap_ids": graph.ap_ids,
        "build_config": graph.build_config,
        "node_feature_dim": graph.node_feature_dim,
        "node_ids": node_ids,
        "node_types": ("sample",) * len(graph.sample_node_ids)
        + ("ap",) * len(active_aps),
        "node_features": node_features,
        "sample_location_initializations": graph.sample_location_initializations,
        "reference_sample_mask": reference_sample_mask,
        "new_sample_mask": new_sample_mask,
        "reference_label_mask": reference_label_mask,
        "reference_coordinate_targets": reference_coordinate_targets,
        "omitted_unobserved_ap_ids": graph.omitted_unobserved_ap_ids,
        "removed_ap_node_ids": orphan_aps,
        "radio_edges": tuple(edges),
    }
    graph_id = f"lpgraph:{_updated_graph_digest(payload)}"
    return UpdatedGraphSnapshot(
        graph_id=graph_id,
        node_id_to_row=tuple((node_id, row) for row, node_id in enumerate(node_ids)),
        sample_node_ids=graph.sample_node_ids,
        active_ap_node_ids=active_aps,
        **payload,
    )


def apply_edge_delta(
    graph: GraphState, learned: GNNFeatureArtifact, delta: EdgeDelta
) -> UpdatedGraphSnapshot:
    """Apply a completed delta into a new immutable graph snapshot.

    The source ``GraphState`` and ``GNNFeatureArtifact`` are never mutated.
    Failed iterations cannot be applied. The returned snapshot stores one
    canonical undirected radio pair and drops temporary virtual edges.
    """
    _validate_inputs(graph, learned)
    if delta.parent_graph_id != graph.graph_id:
        raise LinkPredictionError("edge delta belongs to a different source graph")
    if delta.gnn_checkpoint_sha256 != learned.checkpoint_sha256:
        raise LinkPredictionError("edge delta belongs to a different GNN checkpoint")
    if delta.outcome == "failed":
        raise LinkPredictionError("cannot apply a failed link prediction iteration")
    values = {
        "outcome": delta.outcome,
        "added_edges": delta.added_edges,
        "removed_edges": delta.removed_edges,
        "changed_edges": delta.changed_edges,
        "ap_removals": delta.ap_removals,
    }
    computed = _make_updated_graph(graph, learned, values)
    if delta.updated_graph is not None and delta.updated_graph != computed:
        raise LinkPredictionError("embedded updated graph does not match applying this delta")
    return computed


def predict_edges(
    graph: GraphState,
    learned_node_features: GNNFeatureArtifact,
    config: LinkPredictionConfig | None = None,
) -> EdgeDelta:
    """Predict supported AP/sample links from compatible learned GNN state."""
    _validate_inputs(graph, learned_node_features)
    selected = config or LinkPredictionConfig()
    try:
        (
            goodness,
            fairness,
            iterations,
            goodness_residual,
            fairness_residual,
            converged,
        ) = _update_goodness_fairness(graph, learned_node_features, selected)
    except LinkPredictionError as error:
        raise

    radio_weight_max = max(
        (edge.scalar_weight for edge in graph.radio_edges), default=0.0
    )
    shared = {
        "parent_graph_id": graph.graph_id,
        "dataset_id": graph.dataset_id,
        "batch_id": graph.batch_id,
        "gnn_checkpoint_sha256": learned_node_features.checkpoint_sha256,
        "vocabulary_sha256": graph.vocabulary_sha256,
        "config": selected,
        "iterations": iterations,
        "goodness_residual": goodness_residual,
        "fairness_residual": fairness_residual,
        "radio_weight_max": radio_weight_max,
        "threshold_value": selected.link_threshold,
        "threshold_units": selected.threshold_units,
        "node_ids": graph.node_ids,
        "sample_node_ids": graph.sample_node_ids,
        "node_feature_dim": graph.node_feature_dim,
        "goodness_features": goodness,
        "fairness_features": fairness,
    }

    if not converged:
        return EdgeDelta(
            **shared,
            outcome="failed",
            termination="max_iterations",
            candidate_scores=(),
            added_edges=(),
            removed_edges=(),
            changed_edges=(),
            ap_removals=(),
            updated_graph=None,
            failure_reason=(
                f"goodness residual {goodness_residual:.6g} and fairness residual "
                f"{fairness_residual:.6g} both exceed tolerance "
                f"{selected.convergence_epsilon:.6g} after {iterations} iterations"
            ),
        )

    row_by_id = {node_id: row for row, node_id in enumerate(graph.node_ids)}
    existing_by_pair = {
        (edge.sample_node_id, edge.ap_node_id): edge for edge in graph.radio_edges
    }
    scores: list[CandidateEdgeScore] = []
    added: list[LinkEdgeChange] = []
    removed: list[LinkEdgeChange] = []
    for sample_id, ap_id in _candidate_pairs(graph):
        predicted = _predicted_weight(
            sample_id,
            ap_id,
            row_by_id,
            goodness,
            fairness,
            radio_weight_max,
        )
        pair = sample_id, ap_id
        existing = existing_by_pair.get(pair)
        scores.append(
            CandidateEdgeScore(
                edge_id=f"radio:{sample_id}|{ap_id}",
                sample_node_id=sample_id,
                ap_node_id=ap_id,
                predicted_weight=predicted,
                existed_before=existing is not None,
            )
        )
        if existing is None:
            if predicted >= selected.link_threshold:
                added.append(
                    LinkEdgeChange(
                        edge_id=f"radio:{sample_id}|{ap_id}",
                        sample_node_id=sample_id,
                        ap_node_id=ap_id,
                        old_weight=None,
                        new_weight=predicted,
                        predicted_weight=predicted,
                    )
                )
        elif predicted < selected.link_threshold:
            removed.append(
                LinkEdgeChange(
                    edge_id=existing.edge_id,
                    sample_node_id=sample_id,
                    ap_node_id=ap_id,
                    old_weight=existing.scalar_weight,
                    new_weight=None,
                    predicted_weight=predicted,
                )
            )
        # Existing links above threshold keep their observed weight: Eq. 13
        # scores them, but Algorithm 2 specifies neither a retained-edge rewrite
        # nor an edge-weight delta for that branch.

    resulting_pairs = set(existing_by_pair)
    resulting_pairs.difference_update(
        (change.sample_node_id, change.ap_node_id) for change in removed
    )
    resulting_pairs.update(
        (change.sample_node_id, change.ap_node_id) for change in added
    )
    ap_removals = tuple(
        ap_id
        for ap_id in graph.ap_node_ids
        if not any(pair[1] == ap_id for pair in resulting_pairs)
    )
    no_changes = not added and not removed and not ap_removals
    provisional = {
        **shared,
        "outcome": "no_change" if no_changes else "converged",
        "termination": "residual_tolerance",
        "candidate_scores": tuple(scores),
        "added_edges": tuple(added),
        "removed_edges": tuple(removed),
        "changed_edges": (),
        "ap_removals": ap_removals,
        "updated_graph": None,
        "failure_reason": None,
    }
    updated = _make_updated_graph(graph, learned_node_features, provisional)
    return EdgeDelta(**{**provisional, "updated_graph": updated})


def save_edge_delta(
    delta: EdgeDelta, path: str | Path, *, overwrite: bool = False
) -> Path:
    """Atomically save scores, deltas, convergence, and updated graph JSON."""
    if not isinstance(delta, EdgeDelta):
        raise TypeError("save_edge_delta requires an EdgeDelta")
    return atomic_write_json(delta, path, overwrite=overwrite)


def load_edge_delta(path: str | Path) -> EdgeDelta:
    """Load and validate a complete, provenance-bound link prediction result."""
    return EdgeDelta.model_validate_json(read_json_text(path))


def load_gnn_features_for_link_prediction(path: str | Path) -> GNNFeatureArtifact:
    """Explicit strict file adapter; it never creates features or loads random state."""
    return GNNFeatureArtifact.model_validate_json(read_json_text(path))
