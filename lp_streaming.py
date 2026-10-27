"""Memory-bounded, file-backed implementation of the GUFU link-prediction math.

The compact input bundle stores only node embeddings and CSR adjacency needed by
Section 6. Candidate scores and edge deltas are written as JSONL while they are
computed; G/F and the resulting edge table are stored as NumPy arrays. No
candidate-sized Python object collection or expanded ``UpdatedGraphSnapshot``
is built by :func:`predict_edges_streaming`.

The formulas, neighbor order, candidate rule, threshold comparisons, stopping
rule, and edge-weight conventions mirror ``lp.predict_edges``. This changes the
storage/API representation, not the selected link-prediction interpretation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Iterator, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from graph_build import GraphBuildConfig, GraphState, SampleLocationInitialization
from gufu_artifacts import atomic_write_json, canonical_json_digest, read_json_text, sha256_file
from lp import (
    CandidateEdgeScore,
    LinkEdgeChange,
    LinkPredictionConfig,
    _bounded_predicted_radio_weight,
    _validate_inputs,
)
from sage_ve import GNNFeatureArtifact


STREAM_INPUT_FORMAT = "gufu_lp_stream_input_v3"
STREAM_RESULT_FORMAT = "gufu_lp_stream_result_v1"


class StreamArrayArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*\.npy$")
    dtype: Literal["float64", "float32", "int64", "uint8"]
    shape: tuple[int, ...]
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class StreamTextArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_]*\.jsonl$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class StreamingLPInputManifest(BaseModel):
    """Identity and disk arrays sufficient to run LP without GraphState."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_lp_stream_input_v3"] = STREAM_INPUT_FORMAT
    input_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    dataset_id: str
    batch_id: str
    batch_role: Literal["bootstrap_reference", "update_batch"]
    normalization: Literal["paper_ae", "legacy_raw_div_120"]
    source_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_feature_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    gnn_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bssid_vocabulary: tuple[str, ...]
    ap_ids: tuple[str, ...]
    node_ids: tuple[str, ...]
    node_types: tuple[Literal["sample", "ap"], ...]
    sample_node_ids: tuple[str, ...]
    ap_node_ids: tuple[str, ...]
    graph_build_config: GraphBuildConfig
    encoder_feature_dim: int = Field(ge=1)
    sample_location_initializations: tuple[SampleLocationInitialization, ...]
    reference_sample_mask: tuple[bool, ...]
    new_sample_mask: tuple[bool, ...]
    reference_label_mask: tuple[bool, ...]
    reference_coordinate_targets: tuple[tuple[float, float] | None, ...]
    omitted_unobserved_ap_ids: tuple[str, ...]
    node_feature_dim: int = Field(ge=1)
    radio_edge_count: int = Field(ge=0)
    virtual_edge_count: int = Field(ge=0)
    radio_weight_max: float = Field(ge=0.0)
    arrays: tuple[StreamArrayArtifact, ...]

    @model_validator(mode="after")
    def validate_input_manifest(self):
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("streaming LP node IDs must be unique")
        if self.node_ids != self.sample_node_ids + self.ap_node_ids:
            raise ValueError("streaming LP node order must be samples followed by APs")
        if self.node_types != ("sample",) * len(self.sample_node_ids) + (
            "ap",
        ) * len(self.ap_node_ids):
            raise ValueError("streaming LP node types do not match node order")
        if self.node_feature_dim != self.encoder_feature_dim + 2:
            raise ValueError("streaming LP graph width must equal encoder width plus XY")
        if self.graph_build_config.spatial_normalization_enabled:
            metadata = self.graph_build_config.spatial_transform_metadata
            if metadata is None or self.graph_build_config.spatial_transform_sha256 is None:
                raise ValueError("streaming LP input must record its fitted spatial transform")
            if metadata.site_id != self.dataset_id:
                raise ValueError("streaming LP spatial transform belongs to another site")
        if len(self.sample_location_initializations) != len(self.sample_node_ids):
            raise ValueError("streaming LP input must retain sample location provenance")
        for row, initialization in enumerate(self.sample_location_initializations):
            if self.reference_label_mask[row] and initialization.source != "bootstrap_reference_label":
                raise ValueError("available LP input positions need bootstrap label provenance")
        if any(
            len(values) != len(self.node_ids)
            for values in (
                self.reference_sample_mask,
                self.new_sample_mask,
                self.reference_label_mask,
                self.reference_coordinate_targets,
            )
        ):
            raise ValueError("streaming LP input node metadata is misaligned")
        if self.ap_ids != tuple(f"ap:{bssid}" for bssid in self.bssid_vocabulary):
            raise ValueError("streaming LP AP identities do not match ordered vocabulary")
        if canonical_json_digest(self.bssid_vocabulary) != self.vocabulary_sha256:
            raise ValueError("streaming LP vocabulary digest is inconsistent")
        names = tuple(item.name for item in self.arrays)
        if len(set(names)) != len(names) or set(names) != set(_INPUT_ARRAY_NAMES):
            raise ValueError("streaming LP input array index is incomplete or duplicated")
        expected_shapes = {
            "node_features.npy": (len(self.node_ids), self.node_feature_dim),
            "sample_signal_features.npy": (
                len(self.sample_node_ids),
                len(self.bssid_vocabulary),
            ),
            "sample_signal_masks.npy": (
                len(self.sample_node_ids),
                len(self.bssid_vocabulary),
            ),
            "radio_indptr.npy": (len(self.node_ids) + 1,),
            "radio_neighbor_rows.npy": (2 * self.radio_edge_count,),
            "radio_neighbor_weights.npy": (2 * self.radio_edge_count,),
            "radio_edge_indices.npy": (2 * self.radio_edge_count,),
            "virtual_indptr.npy": (len(self.sample_node_ids) + 1,),
            "virtual_neighbor_rows.npy": (2 * self.virtual_edge_count,),
            "virtual_neighbor_weights.npy": (2 * self.virtual_edge_count,),
            "radio_edge_features.npy": (self.radio_edge_count, self.node_feature_dim),
        }
        for item in self.arrays:
            if item.shape != expected_shapes[item.name]:
                raise ValueError(f"streaming LP input shape is invalid for {item.name}")
        if not math.isfinite(self.radio_weight_max):
            raise ValueError("streaming LP radio weight maximum must be finite")
        identity = canonical_json_digest(
            self.model_dump(mode="json", exclude={"input_id"})
        )
        if identity != self.input_id:
            raise ValueError("streaming LP input identity does not match its metadata")
        return self


class StreamingLPResultManifest(BaseModel):
    """Small manifest for a disk-backed, exact link-prediction result."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_lp_stream_result_v1"] = STREAM_RESULT_FORMAT
    result_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    parent_graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    updated_graph_id: str | None = None
    dataset_id: str
    batch_id: str
    gnn_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    config: LinkPredictionConfig
    candidate_policy: Literal[
        "section6_supported_pairs_v1", "explicit_sorted_stable_ids_v1"
    ]
    candidate_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    outcome: Literal["converged", "no_change", "failed"]
    termination: Literal["residual_tolerance", "max_iterations"]
    iterations: int = Field(ge=0)
    goodness_residual: float = Field(ge=0.0)
    fairness_residual: float = Field(ge=0.0)
    radio_weight_max: float = Field(ge=0.0)
    threshold_value: float = Field(ge=0.0)
    threshold_units: Literal["rssi_plus_120_scalar_weight"]
    node_ids: tuple[str, ...]
    sample_node_ids: tuple[str, ...]
    ap_node_ids: tuple[str, ...]
    node_feature_dim: int = Field(ge=1)
    candidate_count: int = Field(ge=0)
    added_count: int = Field(ge=0)
    removed_count: int = Field(ge=0)
    ap_removals: tuple[str, ...]
    active_ap_node_ids: tuple[str, ...]
    updated_edge_count: int = Field(ge=0)
    failure_reason: str | None = None
    arrays: tuple[StreamArrayArtifact, ...]
    text_artifacts: tuple[StreamTextArtifact, ...]

    @model_validator(mode="after")
    def validate_result_manifest(self):
        if self.node_ids != self.sample_node_ids + self.ap_node_ids:
            raise ValueError("streaming LP sample identity/order is inconsistent")
        if self.threshold_value != self.config.link_threshold:
            raise ValueError("streaming LP threshold disagrees with its configuration")
        if self.threshold_units != self.config.threshold_units:
            raise ValueError("streaming LP threshold units disagree with configuration")
        if not all(
            math.isfinite(value)
            for value in (
                self.goodness_residual,
                self.fairness_residual,
                self.radio_weight_max,
            )
        ):
            raise ValueError("streaming LP scalar result values must be finite")
        if self.outcome == "failed":
            if (
                self.termination != "max_iterations"
                or self.updated_edge_count
                or self.updated_graph_id is not None
            ):
                raise ValueError("failed streaming LP result cannot publish an updated graph")
        else:
            if self.termination != "residual_tolerance":
                raise ValueError("completed streaming LP result must satisfy residual stopping rule")
            if self.updated_graph_id is None or not self.updated_graph_id.startswith("lpgraph:"):
                raise ValueError("completed streaming LP result is missing its graph identity")
        if self.added_count + self.removed_count > self.candidate_count:
            raise ValueError("streaming LP changes exceed scored candidate count")
        if len(set(self.ap_removals)) != len(self.ap_removals) or not set(
            self.ap_removals
        ).issubset(self.ap_node_ids):
            raise ValueError("streaming LP AP removals must identify input AP nodes")
        if len(set(self.active_ap_node_ids)) != len(self.active_ap_node_ids) or not set(
            self.active_ap_node_ids
        ).issubset(self.ap_node_ids):
            raise ValueError("streaming LP active APs must identify input AP nodes")
        if set(self.active_ap_node_ids) & set(self.ap_removals):
            raise ValueError("streaming LP APs cannot be active and removed")
        if self.outcome != "failed" and set(self.active_ap_node_ids) | set(
            self.ap_removals
        ) != set(self.ap_node_ids):
            raise ValueError("streaming LP must classify every input AP as active or removed")
        if self.outcome != "failed" and (self.active_ap_node_ids != tuple(
            ap_id for ap_id in self.ap_node_ids if ap_id in set(self.active_ap_node_ids)
        ) or self.ap_removals != tuple(
            ap_id for ap_id in self.ap_node_ids if ap_id in set(self.ap_removals)
        )):
            raise ValueError("streaming LP AP outcomes must preserve input vocabulary order")
        array_names = {item.name for item in self.arrays}
        text_names = {item.name for item in self.text_artifacts}
        expected_arrays = {
            "goodness_features.npy",
            "fairness_features.npy",
            "updated_edge_endpoints.npy",
            "updated_edge_weights.npy",
            "updated_edge_features.npy",
            "updated_edge_feature_sources.npy",
            "active_node_rows.npy",
        }
        expected_text = {
            "candidate_scores.jsonl",
            "added_edges.jsonl",
            "removed_edges.jsonl",
        }
        if array_names != expected_arrays or text_names != expected_text:
            raise ValueError("streaming LP result artifact index is incomplete")
        if self.outcome == "failed" and (self.candidate_count or self.added_count or self.removed_count):
            raise ValueError("failed streaming LP result cannot include candidate outputs")
        arrays_by_name = {item.name: item for item in self.arrays}
        edge_count = 0 if self.outcome == "failed" else self.updated_edge_count
        expected_shapes = {
            "goodness_features.npy": (len(self.node_ids), self.node_feature_dim),
            "fairness_features.npy": (len(self.node_ids), self.node_feature_dim),
            "updated_edge_endpoints.npy": (edge_count, 2),
            "updated_edge_weights.npy": (edge_count,),
            "updated_edge_features.npy": (edge_count, self.node_feature_dim),
            "updated_edge_feature_sources.npy": (edge_count,),
            "active_node_rows.npy": (
                0
                if self.outcome == "failed"
                else len(self.sample_node_ids) + len(self.active_ap_node_ids),
            ),
        }
        for name, shape in expected_shapes.items():
            if arrays_by_name[name].shape != shape:
                raise ValueError(f"streaming LP result array has an invalid shape: {name}")
        identity_payload = {
            "input_id": self.input_id,
            "parent_graph_id": self.parent_graph_id,
            "updated_graph_id": self.updated_graph_id,
            "gnn_checkpoint_sha256": self.gnn_checkpoint_sha256,
            "config": self.config.model_dump(mode="json"),
            "candidate_policy": self.candidate_policy,
            "candidate_set_sha256": self.candidate_set_sha256,
            "outcome": self.outcome,
            "termination": self.termination,
            "iterations": self.iterations,
            "goodness_residual": self.goodness_residual,
            "fairness_residual": self.fairness_residual,
            "candidate_count": self.candidate_count,
            "added_count": self.added_count,
            "removed_count": self.removed_count,
            "ap_removals": self.ap_removals,
            "arrays": [item.model_dump(mode="json") for item in self.arrays],
            "text_artifacts": [
                item.model_dump(mode="json") for item in self.text_artifacts
            ],
        }
        if canonical_json_digest(identity_payload) != self.result_id:
            raise ValueError("streaming LP result identity does not match its metadata")
        return self


@dataclass(frozen=True)
class StreamingLPResult:
    """Handle for compact disk artifacts; records are exposed as iterators."""

    directory: Path
    manifest: StreamingLPResultManifest

    def iter_candidate_scores(self) -> Iterator[CandidateEdgeScore]:
        yield from _iter_jsonl(self.directory / "candidate_scores.jsonl", CandidateEdgeScore)

    def iter_added_edges(self) -> Iterator[LinkEdgeChange]:
        yield from _iter_jsonl(self.directory / "added_edges.jsonl", LinkEdgeChange)

    def iter_removed_edges(self) -> Iterator[LinkEdgeChange]:
        yield from _iter_jsonl(self.directory / "removed_edges.jsonl", LinkEdgeChange)


@dataclass(frozen=True)
class _MappedInput:
    manifest: StreamingLPInputManifest
    node_features: np.ndarray
    sample_signal_features: np.ndarray
    sample_signal_masks: np.ndarray
    radio_indptr: np.ndarray
    radio_neighbor_rows: np.ndarray
    radio_neighbor_weights: np.ndarray
    radio_edge_indices: np.ndarray
    virtual_indptr: np.ndarray
    virtual_neighbor_rows: np.ndarray
    virtual_neighbor_weights: np.ndarray
    radio_edge_features: np.ndarray


_INPUT_ARRAY_NAMES = (
    "node_features.npy",
    "sample_signal_features.npy",
    "sample_signal_masks.npy",
    "radio_indptr.npy",
    "radio_neighbor_rows.npy",
    "radio_neighbor_weights.npy",
    "radio_edge_indices.npy",
    "virtual_indptr.npy",
    "virtual_neighbor_rows.npy",
    "virtual_neighbor_weights.npy",
    "radio_edge_features.npy",
)


def _open_npy(path: Path, dtype: np.dtype, shape: tuple[int, ...]):
    if math.prod(shape) == 0:
        array = np.empty(shape, dtype=dtype)
        np.save(path, array, allow_pickle=False)
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _finish_npy(array: np.ndarray) -> None:
    if isinstance(array, np.memmap):
        array.flush()


def _read_npy_shape_header(path: Path):
    with path.open("rb") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            return np.lib.format.read_array_header_1_0(stream)
        if version in {(2, 0), (3, 0)}:
            return np.lib.format.read_array_header_2_0(stream)
        raise ValueError(f"unsupported NumPy array format {version} in {path.name}")


def _read_npy(path: Path, artifact: StreamArrayArtifact, *, verify_hash: bool):
    if verify_hash and sha256_file(path) != artifact.sha256:
        raise ValueError(f"streaming LP array digest mismatch: {artifact.name}")
    array = np.load(
        path,
        mmap_mode="r" if math.prod(artifact.shape) else None,
        allow_pickle=False,
    )
    if tuple(array.shape) != artifact.shape or array.dtype.name != artifact.dtype:
        raise ValueError(f"streaming LP array shape/dtype mismatch: {artifact.name}")
    return array


def _load_npy_mapped(path: Path):
    """Load an array read-only, using mmap whenever NumPy permits it."""
    shape, _fortran_order, _dtype = _read_npy_shape_header(path)
    return np.load(
        path,
        mmap_mode="r" if math.prod(shape) else None,
        allow_pickle=False,
    )


def _array_artifact(path: Path, name: str) -> StreamArrayArtifact:
    shape, _fortran_order, _dtype = _read_npy_shape_header(path)
    array = np.load(
        path,
        mmap_mode="r" if math.prod(shape) else None,
        allow_pickle=False,
    )
    return StreamArrayArtifact(
        name=name,
        dtype=array.dtype.name,
        shape=tuple(int(value) for value in array.shape),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _text_artifact(path: Path, name: str) -> StreamTextArtifact:
    return StreamTextArtifact(
        name=name, sha256=sha256_file(path), size_bytes=path.stat().st_size
    )


def _array_map(artifacts: tuple[StreamArrayArtifact, ...], directory: Path, *, verify_hash: bool):
    return {
        item.name: _read_npy(directory / item.name, item, verify_hash=verify_hash)
        for item in artifacts
    }


def _prefix(counts: np.ndarray) -> np.ndarray:
    result = np.empty((len(counts) + 1,), dtype=np.int64)
    result[0] = 0
    np.cumsum(counts, dtype=np.int64, out=result[1:])
    return result


def _sort_csr_rows(
    indptr: np.ndarray,
    neighbor_rows: np.ndarray,
    weights: np.ndarray,
    node_ids: tuple[str, ...],
    edge_indices: np.ndarray | None = None,
) -> None:
    rank = np.empty((len(node_ids),), dtype=np.int64)
    for position, row in enumerate(sorted(range(len(node_ids)), key=node_ids.__getitem__)):
        rank[row] = position
    for row in range(len(indptr) - 1):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        if stop - start < 2:
            continue
        order = np.argsort(rank[neighbor_rows[start:stop]], kind="stable")
        old_neighbors = neighbor_rows[start:stop].copy()
        old_weights = weights[start:stop].copy()
        neighbor_rows[start:stop] = old_neighbors[order]
        weights[start:stop] = old_weights[order]
        if edge_indices is not None:
            old_indices = edge_indices[start:stop].copy()
            edge_indices[start:stop] = old_indices[order]


def _create_staging_directory(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite streaming LP artifact: {target}")
    return Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))


def _publish_directory(staging: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite streaming LP artifact: {target}")
    os.replace(staging, target)


def _write_completion(directory: Path, manifest_name: str, identity: str) -> None:
    atomic_write_json(
        {
            "format": "gufu_streaming_artifact_complete_v1",
            "identity": identity,
            "manifest_sha256": sha256_file(directory / manifest_name),
        },
        directory / "COMPLETED.json",
    )


def save_streaming_lp_input(
    graph: GraphState,
    learned: GNNFeatureArtifact,
    directory: str | Path,
) -> StreamingLPInputManifest:
    """Export a compact, provenance-bound LP input bundle.

    This one-time conversion validates the full GraphState/GNN artifact, then
    writes the node embeddings and only the scalar adjacency data used by LP.
    GraphSAGE edge vectors are kept in a mapped side array for exact output
    reconstruction, not copied into the Python graph representation.
    """
    _validate_inputs(graph, learned)
    target = Path(directory)
    staging = _create_staging_directory(target)
    node_ids = graph.node_ids
    row_by_id = dict(graph.node_id_to_row)
    node_count = len(node_ids)
    sample_count = len(graph.sample_node_ids)
    feature_dim = graph.node_feature_dim
    radio_count = len(graph.radio_edges)
    virtual_count = len(graph.virtual_edges)

    node_features = _open_npy(
        staging / "node_features.npy", np.dtype("float64"), (node_count, feature_dim)
    )
    for row, values in enumerate(learned.node_feature_matrix):
        node_features[row, :] = values
    _finish_npy(node_features)
    sample_signal_features = _open_npy(
        staging / "sample_signal_features.npy",
        np.dtype("float64"),
        (sample_count, len(graph.bssid_vocabulary)),
    )
    sample_signal_masks = _open_npy(
        staging / "sample_signal_masks.npy",
        np.dtype("uint8"),
        (sample_count, len(graph.bssid_vocabulary)),
    )
    for row in range(sample_count):
        sample_signal_features[row, :] = graph.sample_rssi_features[row]
        sample_signal_masks[row, :] = graph.sample_rssi_observation_masks[row]
    _finish_npy(sample_signal_features)
    _finish_npy(sample_signal_masks)

    radio_counts = np.zeros((node_count,), dtype=np.int64)
    for edge in graph.radio_edges:
        left, right = row_by_id[edge.sample_node_id], row_by_id[edge.ap_node_id]
        radio_counts[left] += 1
        radio_counts[right] += 1
    radio_indptr = _prefix(radio_counts)
    np.save(staging / "radio_indptr.npy", radio_indptr, allow_pickle=False)
    radio_size = 2 * radio_count
    radio_neighbor_rows = _open_npy(
        staging / "radio_neighbor_rows.npy", np.dtype("int64"), (radio_size,)
    )
    radio_neighbor_weights = _open_npy(
        staging / "radio_neighbor_weights.npy", np.dtype("float64"), (radio_size,)
    )
    radio_edge_indices = _open_npy(
        staging / "radio_edge_indices.npy", np.dtype("int64"), (radio_size,)
    )
    radio_cursor = radio_indptr[:-1].copy()
    radio_edge_features = _open_npy(
        staging / "radio_edge_features.npy",
        np.dtype("float64"),
        (radio_count, feature_dim),
    )
    for edge_index, (edge, feature_row) in enumerate(
        zip(graph.radio_edges, learned.radio_edge_feature_matrix)
    ):
        left, right = row_by_id[edge.sample_node_id], row_by_id[edge.ap_node_id]
        for source, neighbor in ((left, right), (right, left)):
            cursor = int(radio_cursor[source])
            radio_neighbor_rows[cursor] = neighbor
            radio_neighbor_weights[cursor] = edge.scalar_weight
            radio_edge_indices[cursor] = edge_index
            radio_cursor[source] += 1
        radio_edge_features[edge_index, :] = feature_row
    for array in (radio_neighbor_rows, radio_neighbor_weights, radio_edge_indices):
        _finish_npy(array)
    _finish_npy(radio_edge_features)
    _sort_csr_rows(
        radio_indptr,
        radio_neighbor_rows,
        radio_neighbor_weights,
        node_ids,
        radio_edge_indices,
    )
    for array in (radio_neighbor_rows, radio_neighbor_weights, radio_edge_indices):
        _finish_npy(array)

    virtual_counts = np.zeros((sample_count,), dtype=np.int64)
    for edge in graph.virtual_edges:
        left, right = row_by_id[edge.sample_node_id_left], row_by_id[edge.sample_node_id_right]
        virtual_counts[left] += 1
        virtual_counts[right] += 1
    virtual_indptr = _prefix(virtual_counts)
    np.save(staging / "virtual_indptr.npy", virtual_indptr, allow_pickle=False)
    virtual_size = 2 * virtual_count
    virtual_neighbor_rows = _open_npy(
        staging / "virtual_neighbor_rows.npy", np.dtype("int64"), (virtual_size,)
    )
    virtual_neighbor_weights = _open_npy(
        staging / "virtual_neighbor_weights.npy", np.dtype("float64"), (virtual_size,)
    )
    virtual_cursor = virtual_indptr[:-1].copy()
    for edge in graph.virtual_edges:
        left, right = row_by_id[edge.sample_node_id_left], row_by_id[edge.sample_node_id_right]
        for source, neighbor in ((left, right), (right, left)):
            cursor = int(virtual_cursor[source])
            virtual_neighbor_rows[cursor] = neighbor
            virtual_neighbor_weights[cursor] = edge.cosine_similarity
            virtual_cursor[source] += 1
    _finish_npy(virtual_neighbor_rows)
    _finish_npy(virtual_neighbor_weights)
    _sort_csr_rows(
        virtual_indptr,
        virtual_neighbor_rows,
        virtual_neighbor_weights,
        node_ids,
    )
    for array in (virtual_neighbor_rows, virtual_neighbor_weights):
        _finish_npy(array)

    arrays = tuple(
        _array_artifact(staging / name, name) for name in _INPUT_ARRAY_NAMES
    )
    manifest_values = {
        "graph_id": graph.graph_id,
        "dataset_id": graph.dataset_id,
        "batch_id": graph.batch_id,
        "batch_role": graph.batch_role,
        "normalization": graph.normalization,
        "source_prepared_sha256": graph.source_prepared_sha256,
        "encoder_feature_artifact_sha256": graph.encoder_feature_artifact_sha256,
        "encoder_checkpoint_sha256": graph.encoder_checkpoint_sha256,
        "gnn_checkpoint_sha256": learned.checkpoint_sha256,
        "vocabulary_sha256": graph.vocabulary_sha256,
        "bssid_vocabulary": graph.bssid_vocabulary,
        "ap_ids": graph.ap_ids,
        "node_ids": graph.node_ids,
        "node_types": graph.node_types,
        "sample_node_ids": graph.sample_node_ids,
        "ap_node_ids": graph.ap_node_ids,
        "graph_build_config": graph.build_config,
        "sample_location_initializations": graph.sample_location_initializations,
        "reference_sample_mask": graph.reference_sample_mask,
        "new_sample_mask": graph.new_sample_mask,
        "reference_label_mask": graph.reference_label_mask,
        "reference_coordinate_targets": graph.reference_coordinate_targets,
        "omitted_unobserved_ap_ids": graph.omitted_unobserved_ap_ids,
        "node_feature_dim": feature_dim,
        "encoder_feature_dim": graph.encoder_feature_dim,
        "radio_edge_count": radio_count,
        "virtual_edge_count": virtual_count,
        "radio_weight_max": max(
            (edge.scalar_weight for edge in graph.radio_edges), default=0.0
        ),
        "arrays": arrays,
    }
    identity_values = {
        **manifest_values,
        "graph_build_config": graph.build_config.model_dump(mode="json"),
        "sample_location_initializations": [
            item.model_dump(mode="json")
            for item in graph.sample_location_initializations
        ],
        "arrays": [item.model_dump(mode="json") for item in arrays],
    }
    input_id = canonical_json_digest({"format": STREAM_INPUT_FORMAT, **identity_values})
    manifest = StreamingLPInputManifest(input_id=input_id, **manifest_values)
    atomic_write_json(manifest, staging / "manifest.json")
    _write_completion(staging, "manifest.json", input_id)
    _publish_directory(staging, target)
    return manifest


def load_streaming_lp_input(
    directory: str | Path, *, verify_hashes: bool = True
) -> tuple[StreamingLPInputManifest, _MappedInput]:
    root = Path(directory)
    marker_path = root / "COMPLETED.json"
    manifest_path = root / "manifest.json"
    if not marker_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"streaming LP input is incomplete: {root}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    manifest = StreamingLPInputManifest.model_validate_json(read_json_text(manifest_path))
    if marker.get("identity") != manifest.input_id or marker.get(
        "manifest_sha256"
    ) != sha256_file(manifest_path):
        raise ValueError("streaming LP input completion marker is inconsistent")
    arrays = _array_map(manifest.arrays, root, verify_hash=verify_hashes)
    mapped = _MappedInput(
        manifest=manifest,
        node_features=arrays["node_features.npy"],
        sample_signal_features=arrays["sample_signal_features.npy"],
        sample_signal_masks=arrays["sample_signal_masks.npy"],
        radio_indptr=arrays["radio_indptr.npy"],
        radio_neighbor_rows=arrays["radio_neighbor_rows.npy"],
        radio_neighbor_weights=arrays["radio_neighbor_weights.npy"],
        radio_edge_indices=arrays["radio_edge_indices.npy"],
        virtual_indptr=arrays["virtual_indptr.npy"],
        virtual_neighbor_rows=arrays["virtual_neighbor_rows.npy"],
        virtual_neighbor_weights=arrays["virtual_neighbor_weights.npy"],
        radio_edge_features=arrays["radio_edge_features.npy"],
    )
    return manifest, mapped


def validate_streaming_lp_input(
    graph: GraphState,
    learned: GNNFeatureArtifact,
    directory: str | Path,
    *,
    verify_hashes: bool = True,
) -> StreamingLPInputManifest:
    """Check an existing disk bundle against the current graph and learned rows.

    Row-wise comparisons avoid constructing another full feature matrix while
    still rejecting a same-ID/checkpoint bundle whose feature values changed.
    """
    _validate_inputs(graph, learned)
    manifest, mapped = load_streaming_lp_input(
        directory, verify_hashes=verify_hashes
    )
    if (
        manifest.graph_id != graph.graph_id
        or manifest.gnn_checkpoint_sha256 != learned.checkpoint_sha256
        or manifest.vocabulary_sha256 != graph.vocabulary_sha256
        or manifest.node_ids != graph.node_ids
        or manifest.radio_edge_count != len(graph.radio_edges)
        or manifest.virtual_edge_count != len(graph.virtual_edges)
    ):
        raise ValueError("streaming LP input bundle belongs to different graph lineage")
    for row, values in enumerate(learned.node_feature_matrix):
        if not np.array_equal(mapped.node_features[row], np.asarray(values, dtype=np.float64)):
            raise ValueError(f"streaming LP node feature row changed for {graph.node_ids[row]}")
    for row in range(len(graph.sample_node_ids)):
        expected = np.asarray(graph.sample_rssi_features[row], dtype=np.float64)
        if not np.array_equal(mapped.sample_signal_features[row], expected):
            raise ValueError(
                f"streaming LP normalized RSSI row changed for {graph.sample_node_ids[row]}"
            )
        expected_mask = np.asarray(graph.sample_rssi_observation_masks[row], dtype=np.uint8)
        if not np.array_equal(mapped.sample_signal_masks[row], expected_mask):
            raise ValueError(
                f"streaming LP RSSI mask row changed for {graph.sample_node_ids[row]}"
            )
    for row, values in enumerate(learned.radio_edge_feature_matrix):
        if not np.array_equal(
            mapped.radio_edge_features[row], np.asarray(values, dtype=np.float64)
        ):
            raise ValueError(
                f"streaming LP physical-edge feature row changed for "
                f"{graph.radio_edges[row].edge_id}"
            )
    return manifest


def _neighbors(mapped: _MappedInput, row: int) -> Iterator[tuple[int, float]]:
    manifest = mapped.manifest
    r_start, r_stop = map(int, mapped.radio_indptr[row : row + 2])
    if row < len(manifest.sample_node_ids):
        v_start, v_stop = map(int, mapped.virtual_indptr[row : row + 2])
    else:
        v_start = v_stop = 0
    r_cursor, v_cursor = r_start, v_start
    radio_max = manifest.radio_weight_max
    while r_cursor < r_stop or v_cursor < v_stop:
        if r_cursor >= r_stop:
            neighbor = int(mapped.virtual_neighbor_rows[v_cursor])
            weight = float(mapped.virtual_neighbor_weights[v_cursor])
            v_cursor += 1
        elif v_cursor >= v_stop:
            neighbor = int(mapped.radio_neighbor_rows[r_cursor])
            raw_weight = float(mapped.radio_neighbor_weights[r_cursor])
            weight = raw_weight / radio_max if radio_max > 0.0 else 0.0
            r_cursor += 1
        else:
            radio_neighbor = int(mapped.radio_neighbor_rows[r_cursor])
            virtual_neighbor = int(mapped.virtual_neighbor_rows[v_cursor])
            if manifest.node_ids[radio_neighbor] < manifest.node_ids[virtual_neighbor]:
                neighbor = radio_neighbor
                raw_weight = float(mapped.radio_neighbor_weights[r_cursor])
                weight = raw_weight / radio_max if radio_max > 0.0 else 0.0
                r_cursor += 1
            else:
                neighbor = virtual_neighbor
                weight = float(mapped.virtual_neighbor_weights[v_cursor])
                v_cursor += 1
        yield neighbor, weight


def _initial_feature_files(directory: Path, node_features: np.ndarray):
    shape = tuple(int(value) for value in node_features.shape)
    goodness = _open_npy(directory / "goodness_a.npy", np.dtype("float64"), shape)
    fairness = _open_npy(directory / "fairness_a.npy", np.dtype("float64"), shape)
    for row in range(shape[0]):
        goodness[row, :] = node_features[row, :]
        fairness[row, :] = node_features[row, :]
    _finish_npy(goodness)
    _finish_npy(fairness)
    goodness_next = _open_npy(directory / "goodness_b.npy", np.dtype("float64"), shape)
    fairness_next = _open_npy(directory / "fairness_b.npy", np.dtype("float64"), shape)
    _finish_npy(goodness_next)
    _finish_npy(fairness_next)
    return goodness, fairness, goodness_next, fairness_next


def _finite_array_row(row: np.ndarray) -> bool:
    return all(math.isfinite(float(value)) for value in row)


def _report_progress(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        callback(f"[link prediction] {message}")


def _run_goodness_fairness(
    mapped: _MappedInput,
    config: LinkPredictionConfig,
    work_directory: Path,
    *,
    progress_callback: Callable[[str], None] | None = None,
) -> tuple[Path, Path, int, float, float, bool]:
    goodness, fairness, goodness_next, fairness_next = _initial_feature_files(
        work_directory, mapped.node_features
    )
    node_count, feature_dim = mapped.node_features.shape
    no_edges = not (
        mapped.radio_neighbor_rows.size or mapped.virtual_neighbor_rows.size
    )
    iterations = 0
    g_residual = f_residual = 0.0
    converged = no_edges
    started_at = monotonic()
    last_progress_at = started_at
    progress_iteration_step = max(1, math.ceil(config.max_iterations / 10))
    _report_progress(
        progress_callback,
        f"goodness/fairness solver started: nodes={node_count:,}, "
        f"features={feature_dim}, max_iterations={config.max_iterations:,}, "
        f"tolerance={config.convergence_epsilon:g}",
    )
    while not converged and iterations < config.max_iterations:
        for row in range(node_count):
            start = int(mapped.radio_indptr[row])
            stop = int(mapped.radio_indptr[row + 1])
            if row < len(mapped.manifest.sample_node_ids):
                start_virtual = int(mapped.virtual_indptr[row])
                stop_virtual = int(mapped.virtual_indptr[row + 1])
            else:
                start_virtual = stop_virtual = 0
            degree = stop - start + stop_virtual - start_virtual
            if degree == 0:
                goodness_next[row, :] = goodness[row, :]
                continue
            # Cache only this row's sparse neighborhood. Memory therefore
            # scales with the maximum node degree, not the total edge set.
            adjacent = tuple(_neighbors(mapped, row))
            for column in range(feature_dim):
                total = sum(
                    float(fairness[neighbor, column]) * weight
                    for neighbor, weight in adjacent
                )
                goodness_next[row, column] = total / degree
            if not _finite_array_row(goodness_next[row]):
                raise ValueError("goodness/fairness iteration produced a non-finite feature")

        for row in range(node_count):
            start = int(mapped.radio_indptr[row])
            stop = int(mapped.radio_indptr[row + 1])
            if row < len(mapped.manifest.sample_node_ids):
                start_virtual = int(mapped.virtual_indptr[row])
                stop_virtual = int(mapped.virtual_indptr[row + 1])
            else:
                start_virtual = stop_virtual = 0
            degree = stop - start + stop_virtual - start_virtual
            if degree == 0:
                fairness_next[row, :] = fairness[row, :]
                continue
            adjacent = tuple(_neighbors(mapped, row))
            for column in range(feature_dim):
                total = sum(
                    abs(weight - float(goodness_next[neighbor, column]))
                    for neighbor, weight in adjacent
                )
                fairness_next[row, column] = 1.0 - total / (2.0 * degree)
            if not _finite_array_row(fairness_next[row]):
                raise ValueError("goodness/fairness iteration produced a non-finite feature")

        g_residual = 0.0
        f_residual = 0.0
        for row in range(node_count):
            g_residual += math.sqrt(
                sum(
                    (float(goodness_next[row, column]) - float(goodness[row, column])) ** 2
                    for column in range(feature_dim)
                )
            )
            f_residual += math.sqrt(
                sum(
                    (float(fairness_next[row, column]) - float(fairness[row, column])) ** 2
                    for column in range(feature_dim)
                )
            )
        if not math.isfinite(g_residual) or not math.isfinite(f_residual):
            raise ValueError("goodness/fairness iteration produced a non-finite residual")
        goodness.flush()
        fairness.flush()
        goodness_next.flush()
        fairness_next.flush()
        goodness, goodness_next = goodness_next, goodness
        fairness, fairness_next = fairness_next, fairness
        iterations += 1
        converged = (
            g_residual <= config.convergence_epsilon
            or f_residual <= config.convergence_epsilon
        )
        now = monotonic()
        if (
            iterations % progress_iteration_step == 0
            or now - last_progress_at >= 10.0
            or converged
            or iterations == config.max_iterations
        ):
            _report_progress(
                progress_callback,
                f"goodness/fairness iteration {iterations:,}/"
                f"{config.max_iterations:,}; residuals="
                f"({g_residual:.6g}, {f_residual:.6g}); "
                f"elapsed={now - started_at:.1f}s"
                + ("; converged" if converged else ""),
            )
            last_progress_at = now
    if no_edges:
        _report_progress(
            progress_callback,
            "goodness/fairness solver skipped because the graph has no edges",
        )
    return (
        Path(goodness.filename),
        Path(fairness.filename),
        iterations,
        g_residual,
        f_residual,
        converged,
    )


def _iter_candidates(mapped: _MappedInput) -> Iterator[tuple[int, int, float | None]]:
    manifest = mapped.manifest
    sample_rows_by_id = sorted(
        range(len(manifest.sample_node_ids)),
        key=lambda row: manifest.node_ids[row],
    )
    for sample_row in sample_rows_by_id:
        existing: dict[int, float] = {}
        radio_start, radio_stop = map(int, mapped.radio_indptr[sample_row : sample_row + 2])
        for index in range(radio_start, radio_stop):
            ap_row = int(mapped.radio_neighbor_rows[index])
            if ap_row >= len(manifest.sample_node_ids):
                existing[ap_row] = float(mapped.radio_neighbor_weights[index])
        candidates = set(existing)
        virtual_start, virtual_stop = map(
            int, mapped.virtual_indptr[sample_row : sample_row + 2]
        )
        for virtual_index in range(virtual_start, virtual_stop):
            neighbor_sample = int(mapped.virtual_neighbor_rows[virtual_index])
            radio_start, radio_stop = map(
                int, mapped.radio_indptr[neighbor_sample : neighbor_sample + 2]
            )
            for index in range(radio_start, radio_stop):
                ap_row = int(mapped.radio_neighbor_rows[index])
                if ap_row >= len(manifest.sample_node_ids):
                    candidates.add(ap_row)
        for ap_row in sorted(candidates, key=manifest.node_ids.__getitem__):
            yield sample_row, ap_row, existing.get(ap_row)


def _iter_explicit_candidates(
    mapped: _MappedInput,
    candidate_pairs: IterableABC[tuple[str, str]],
) -> Iterator[tuple[int, int, float | None]]:
    """Resolve a sorted stream of stable sample/AP IDs to row candidates.

    The caller owns candidate generation and may keep it on disk (for example,
    a SQLite ordered cursor). Pairs must arrive strictly lexicographically
    sorted, which enforces uniqueness without retaining a candidate-sized set.
    Existing scalar weights are recovered from the mapped radio CSR.
    """
    manifest = mapped.manifest
    sample_rows = {node_id: row for row, node_id in enumerate(manifest.sample_node_ids)}
    ap_rows = {
        node_id: row
        for row, node_id in enumerate(manifest.node_ids)
        if row >= len(manifest.sample_node_ids)
    }
    previous: tuple[str, str] | None = None
    for pair in candidate_pairs:
        if (
            not isinstance(pair, tuple)
            or len(pair) != 2
            or not all(isinstance(value, str) for value in pair)
        ):
            raise ValueError("explicit LP candidates must be (sample_id, ap_id) string pairs")
        sample_id, ap_id = pair
        if previous is not None and pair <= previous:
            raise ValueError(
                "explicit LP candidates must be strictly sorted and unique by stable ID"
            )
        previous = pair
        if sample_id not in sample_rows or ap_id not in ap_rows:
            raise ValueError(
                f"explicit LP candidate must identify a sample/AP node pair: {pair!r}"
            )
        sample_row, ap_row = sample_rows[sample_id], ap_rows[ap_id]
        start, stop = map(int, mapped.radio_indptr[sample_row : sample_row + 2])
        old_weight = None
        for index in range(start, stop):
            if int(mapped.radio_neighbor_rows[index]) == ap_row:
                old_weight = float(mapped.radio_neighbor_weights[index])
                break
        yield sample_row, ap_row, old_weight


def _iter_jsonl(path: Path, model_type):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield model_type.model_validate_json(line)


def _write_jsonl_record(stream, model: BaseModel) -> None:
    stream.write(model.model_dump_json() + "\n")


def _load_npy_memmap(path: Path, dtype: np.dtype, shape: tuple[int, ...]):
    if math.prod(shape) == 0:
        array = np.empty(shape, dtype=dtype)
        np.save(path, array, allow_pickle=False)
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _write_updated_edge_arrays(
    mapped: _MappedInput,
    directory: Path,
    added_path: Path,
    removed_path: Path,
    *,
    added_count: int,
    removed_count: int,
    goodness_available: bool,
) -> tuple[int, tuple[str, ...], tuple[str, ...]]:
    manifest = mapped.manifest
    if not goodness_available:
        empty_shapes = {
            "updated_edge_endpoints.npy": (0, 2),
            "updated_edge_weights.npy": (0,),
            "updated_edge_features.npy": (0, manifest.node_feature_dim),
            "updated_edge_feature_sources.npy": (0,),
            "active_node_rows.npy": (0,),
        }
        for name, shape in empty_shapes.items():
            array = np.empty(shape, dtype=np.uint8 if "sources" in name else np.float64)
            if "endpoints" in name or "rows" in name:
                array = np.empty(shape, dtype=np.int64)
            np.save(directory / name, array, allow_pickle=False)
        return 0, (), ()

    expected_count = manifest.radio_edge_count + added_count - removed_count
    endpoints = _load_npy_memmap(
        directory / "updated_edge_endpoints.npy", np.dtype("int64"), (expected_count, 2)
    )
    weights = _load_npy_memmap(
        directory / "updated_edge_weights.npy", np.dtype("float64"), (expected_count,)
    )
    features = _load_npy_memmap(
        directory / "updated_edge_features.npy",
        np.dtype("float64"),
        (expected_count, manifest.node_feature_dim),
    )
    sources = _load_npy_memmap(
        directory / "updated_edge_feature_sources.npy",
        np.dtype("uint8"),
        (expected_count,),
    )
    node_row = {node_id: row for row, node_id in enumerate(manifest.node_ids)}
    ap_degree = {
        ap_row: int(mapped.radio_indptr[ap_row + 1] - mapped.radio_indptr[ap_row])
        for ap_row in range(len(manifest.sample_node_ids), len(manifest.node_ids))
    }

    added_records = _iter_jsonl(added_path, LinkEdgeChange)
    removed_records = _iter_jsonl(removed_path, LinkEdgeChange)
    next_added = next(added_records, None)
    next_removed = next(removed_records, None)
    output_row = 0
    sample_rows_by_id = sorted(
        range(len(manifest.sample_node_ids)),
        key=lambda row: manifest.node_ids[row],
    )
    for sample_row in sample_rows_by_id:
        sample_id = manifest.node_ids[sample_row]
        old_edges: dict[int, tuple[float, int]] = {}
        start, stop = map(int, mapped.radio_indptr[sample_row : sample_row + 2])
        for index in range(start, stop):
            ap_row = int(mapped.radio_neighbor_rows[index])
            if ap_row >= len(manifest.sample_node_ids):
                old_edges[ap_row] = (
                    float(mapped.radio_neighbor_weights[index]),
                    int(mapped.radio_edge_indices[index]),
                )
        removed_for_sample: set[int] = set()
        while next_removed is not None and next_removed.sample_node_id == sample_id:
            removed_for_sample.add(node_row[next_removed.ap_node_id])
            next_removed = next(removed_records, None)
        added_for_sample: dict[int, float] = {}
        while next_added is not None and next_added.sample_node_id == sample_id:
            added_for_sample[node_row[next_added.ap_node_id]] = float(next_added.new_weight)
            next_added = next(added_records, None)

        result_ap_rows = (set(old_edges) - removed_for_sample) | set(added_for_sample)
        for ap_row in sorted(result_ap_rows, key=manifest.node_ids.__getitem__):
            endpoints[output_row, 0] = sample_row
            endpoints[output_row, 1] = ap_row
            if ap_row in added_for_sample:
                weight = added_for_sample[ap_row]
                sources[output_row] = 1
                for column in range(manifest.node_feature_dim):
                    features[output_row, column] = (
                        float(mapped.node_features[sample_row, column])
                        + float(mapped.node_features[ap_row, column])
                    ) / 2.0
            else:
                weight, edge_index = old_edges[ap_row]
                sources[output_row] = 0
                features[output_row, :] = mapped.radio_edge_features[edge_index, :]
            weights[output_row] = weight
            output_row += 1
            if ap_row in added_for_sample:
                ap_degree[ap_row] += 1
        for ap_row in removed_for_sample:
            ap_degree[ap_row] -= 1

    if output_row != expected_count:
        raise ValueError(
            f"streaming updated-edge count mismatch: wrote {output_row}, expected {expected_count}"
        )
    for array in (endpoints, weights, features, sources):
        if isinstance(array, np.memmap):
            array.flush()
    if next_added is not None or next_removed is not None:
        raise ValueError("streaming delta records are not sorted by stable endpoint IDs")
    ap_removals = tuple(
        ap_id
        for ap_id in manifest.ap_node_ids
        if ap_degree[node_row[ap_id]] == 0
    )
    active_aps = tuple(ap_id for ap_id in manifest.ap_node_ids if ap_id not in set(ap_removals))
    active_rows = np.empty(
        (len(manifest.sample_node_ids) + len(active_aps),), dtype=np.int64
    )
    active_rows[: len(manifest.sample_node_ids)] = np.arange(
        len(manifest.sample_node_ids), dtype=np.int64
    )
    active_rows[len(manifest.sample_node_ids) :] = [node_row[ap_id] for ap_id in active_aps]
    np.save(directory / "active_node_rows.npy", active_rows, allow_pickle=False)
    return output_row, ap_removals, active_aps


def _copy_final_features(source: Path, target: Path) -> None:
    source_array = np.load(source, mmap_mode="r", allow_pickle=False)
    target_array = _load_npy_memmap(
        target,
        np.dtype("float64"),
        tuple(int(value) for value in source_array.shape),
    )
    for row in range(source_array.shape[0]):
        target_array[row, :] = source_array[row, :]
    if isinstance(target_array, np.memmap):
        target_array.flush()


def _canonical_json_chunks(value):
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        yield b"{"
        for index, key in enumerate(sorted(value)):
            if index:
                yield b","
            yield json.dumps(key, ensure_ascii=False).encode("utf-8")
            yield b":"
            yield from _canonical_json_chunks(value[key])
        yield b"}"
    elif isinstance(value, (tuple, list)) or (
        isinstance(value, IterableABC) and not isinstance(value, (str, bytes))
    ):
        yield b"["
        for index, item in enumerate(value):
            if index:
                yield b","
            yield from _canonical_json_chunks(item)
        yield b"]"
    else:
        yield json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


def _stream_updated_graph_id(
    mapped: _MappedInput,
    directory: Path,
    active_aps: tuple[str, ...],
    ap_removals: tuple[str, ...],
) -> str:
    manifest = mapped.manifest
    endpoints = _load_npy_mapped(directory / "updated_edge_endpoints.npy")
    weights = _load_npy_mapped(directory / "updated_edge_weights.npy")
    features = _load_npy_mapped(directory / "updated_edge_features.npy")
    sources = _load_npy_mapped(directory / "updated_edge_feature_sources.npy")
    active_rows = _load_npy_mapped(directory / "active_node_rows.npy")
    node_ids = manifest.sample_node_ids + active_aps

    def active_features():
        for original_row in active_rows:
            yield tuple(float(value) for value in mapped.node_features[int(original_row)])

    def active_metadata(values):
        for original_row in active_rows:
            yield values[int(original_row)]

    def edge_records():
        for index in range(len(endpoints)):
            sample_row, ap_row = (int(value) for value in endpoints[index])
            sample_id = manifest.node_ids[sample_row]
            ap_id = manifest.node_ids[ap_row]
            scalar_weight = float(weights[index])
            yield {
                "edge_id": f"radio:{sample_id}|{ap_id}",
                "sample_node_id": sample_id,
                "ap_node_id": ap_id,
                "rssi_dbm": scalar_weight - 120.0,
                "scalar_weight": scalar_weight,
                "edge_feature": tuple(float(value) for value in features[index]),
                "edge_feature_source": (
                    "gnn_endpoint_mean_initialization"
                    if int(sources[index]) == 1
                    else "gnn_learned_physical_edge"
                ),
            }

    identity_fields = {
        "parent_graph_id": manifest.graph_id,
        "gnn_checkpoint_sha256": manifest.gnn_checkpoint_sha256,
        "dataset_id": manifest.dataset_id,
        "batch_id": manifest.batch_id,
        "batch_role": manifest.batch_role,
        "normalization": manifest.normalization,
        "source_prepared_sha256": manifest.source_prepared_sha256,
        "encoder_feature_artifact_sha256": manifest.encoder_feature_artifact_sha256,
        "encoder_checkpoint_sha256": manifest.encoder_checkpoint_sha256,
        "vocabulary_sha256": manifest.vocabulary_sha256,
        "bssid_vocabulary": manifest.bssid_vocabulary,
        "ap_ids": manifest.ap_ids,
        "build_config": manifest.graph_build_config.model_dump(mode="json"),
        "node_feature_dim": manifest.node_feature_dim,
        "node_ids": node_ids,
        "node_types": ("sample",) * len(manifest.sample_node_ids)
        + ("ap",) * len(active_aps),
        "node_features": active_features(),
        "sample_location_initializations": [
            item.model_dump(mode="json")
            for item in manifest.sample_location_initializations
        ],
        "reference_sample_mask": active_metadata(manifest.reference_sample_mask),
        "new_sample_mask": active_metadata(manifest.new_sample_mask),
        "reference_label_mask": active_metadata(manifest.reference_label_mask),
        "reference_coordinate_targets": active_metadata(
            manifest.reference_coordinate_targets
        ),
        "omitted_unobserved_ap_ids": manifest.omitted_unobserved_ap_ids,
        "removed_ap_node_ids": ap_removals,
        "radio_edges": edge_records(),
    }
    digest = hashlib.sha256()
    for chunk in _canonical_json_chunks(identity_fields):
        digest.update(chunk)
    return f"lpgraph:{digest.hexdigest()}"


def _result_array_artifacts(directory: Path) -> tuple[StreamArrayArtifact, ...]:
    names = (
        "goodness_features.npy",
        "fairness_features.npy",
        "updated_edge_endpoints.npy",
        "updated_edge_weights.npy",
        "updated_edge_features.npy",
        "updated_edge_feature_sources.npy",
        "active_node_rows.npy",
    )
    return tuple(_array_artifact(directory / name, name) for name in names)


def predict_edges_streaming(
    input_directory: str | Path,
    output_directory: str | Path,
    config: LinkPredictionConfig | None = None,
    *,
    candidate_pairs: IterableABC[tuple[str, str]] | None = None,
    verify_input_hashes: bool = True,
    progress_callback: Callable[[str], None] | None = None,
) -> StreamingLPResult:
    """Run the same LP equations using memory-mapped compact graph state.

    The result preserves all candidate-score and change records on disk. It
    does not construct a Python ``EdgeDelta`` or a typed expanded graph. The
    updated radio graph is available as endpoint, weight, and edge-feature NPY
    arrays plus the stable IDs in the result/input manifests.
    """
    selected = config or LinkPredictionConfig()
    target = Path(output_directory)
    started_at = monotonic()
    staging = _create_staging_directory(target)
    try:
        _report_progress(progress_callback, "validating and mapping LP input artifacts")
        input_manifest, mapped = load_streaming_lp_input(
            input_directory, verify_hashes=verify_input_hashes
        )
        _report_progress(
            progress_callback,
            f"input ready: samples={len(input_manifest.sample_node_ids):,}, "
            f"APs={len(input_manifest.ap_node_ids):,}, "
            f"nodes={len(input_manifest.node_ids):,}, "
            f"radio edges={input_manifest.radio_edge_count:,}, "
            f"virtual edges={input_manifest.virtual_edge_count:,}",
        )
        if input_manifest.radio_weight_max < 0.0:
            raise ValueError("streaming LP input has an invalid radio weight maximum")

        _report_progress(progress_callback, "starting goodness/fairness propagation")
        goodness_path, fairness_path, iterations, g_residual, f_residual, converged = (
            _run_goodness_fairness(
                mapped,
                selected,
                staging,
                progress_callback=progress_callback,
            )
        )
        candidate_path = staging / "candidate_scores.jsonl"
        added_path = staging / "added_edges.jsonl"
        removed_path = staging / "removed_edges.jsonl"
        candidate_count = added_count = removed_count = 0
        ap_removals: tuple[str, ...] = ()
        active_aps: tuple[str, ...] = ()
        updated_edge_count = 0
        updated_graph_id = None
        failure_reason = None
        candidate_set_hasher = hashlib.sha256()
        outcome: Literal["converged", "no_change", "failed"]
        termination: Literal["residual_tolerance", "max_iterations"]

        if not converged:
            outcome = "failed"
            termination = "max_iterations"
            failure_reason = (
                f"goodness residual {g_residual:.6g} and fairness residual "
                f"{f_residual:.6g} both exceed tolerance "
                f"{selected.convergence_epsilon:.6g} after {iterations} iterations"
            )
            candidate_path.write_text("", encoding="utf-8")
            added_path.write_text("", encoding="utf-8")
            removed_path.write_text("", encoding="utf-8")
        else:
            outcome = "converged"
            termination = "residual_tolerance"
            goodness = np.load(goodness_path, mmap_mode="r", allow_pickle=False)
            fairness = np.load(fairness_path, mmap_mode="r", allow_pickle=False)
            scoring_started_at = monotonic()
            last_scoring_progress_at = scoring_started_at
            last_scoring_progress_count = 0
            _report_progress(
                progress_callback,
                "starting candidate-edge scoring and delta serialization",
            )
            with (
                candidate_path.open("w", encoding="utf-8", newline="\n") as candidate_file,
                added_path.open("w", encoding="utf-8", newline="\n") as added_file,
                removed_path.open("w", encoding="utf-8", newline="\n") as removed_file,
            ):
                candidates = (
                    _iter_candidates(mapped)
                    if candidate_pairs is None
                    else _iter_explicit_candidates(mapped, candidate_pairs)
                )
                for sample_row, ap_row, old_weight in candidates:
                    sample_id = input_manifest.node_ids[sample_row]
                    ap_id = input_manifest.node_ids[ap_row]
                    candidate_set_hasher.update(
                        sample_id.encode("utf-8")
                        + b"\0"
                        + ap_id.encode("utf-8")
                        + b"\n"
                    )
                    sample_g = goodness[sample_row]
                    sample_f = fairness[sample_row]
                    ap_g = goodness[ap_row]
                    ap_f = fairness[ap_row]
                    left = sum(
                        float(sample_g[column]) * float(ap_f[column])
                        for column in range(input_manifest.node_feature_dim)
                    )
                    right = sum(
                        float(sample_f[column]) * float(ap_g[column])
                        for column in range(input_manifest.node_feature_dim)
                    )
                    predicted = _bounded_predicted_radio_weight(
                        left, right, input_manifest.radio_weight_max
                    )
                    score = CandidateEdgeScore(
                        edge_id=f"radio:{sample_id}|{ap_id}",
                        sample_node_id=sample_id,
                        ap_node_id=ap_id,
                        predicted_weight=predicted,
                        existed_before=old_weight is not None,
                    )
                    _write_jsonl_record(candidate_file, score)
                    candidate_count += 1
                    if old_weight is None and predicted >= selected.link_threshold:
                        change = LinkEdgeChange(
                            edge_id=score.edge_id,
                            sample_node_id=sample_id,
                            ap_node_id=ap_id,
                            old_weight=None,
                            new_weight=predicted,
                            predicted_weight=predicted,
                        )
                        _write_jsonl_record(added_file, change)
                        added_count += 1
                    elif old_weight is not None and predicted < selected.link_threshold:
                        change = LinkEdgeChange(
                            edge_id=score.edge_id,
                            sample_node_id=sample_id,
                            ap_node_id=ap_id,
                            old_weight=old_weight,
                            new_weight=None,
                            predicted_weight=predicted,
                        )
                        _write_jsonl_record(removed_file, change)
                        removed_count += 1
                    if progress_callback is not None and (
                        candidate_count == 1 or candidate_count % 1024 == 0
                    ):
                        now = monotonic()
                        if (
                            candidate_count == 1
                            or candidate_count - last_scoring_progress_count >= 5000
                            or now - last_scoring_progress_at >= 10.0
                        ):
                            _report_progress(
                                progress_callback,
                                f"scored {candidate_count:,} candidate edges; "
                                f"added={added_count:,}, removed={removed_count:,}; "
                                f"elapsed={now - scoring_started_at:.1f}s",
                            )
                            last_scoring_progress_at = now
                            last_scoring_progress_count = candidate_count
            _report_progress(
                progress_callback,
                f"candidate scoring complete: {candidate_count:,} scored, "
                f"added={added_count:,}, removed={removed_count:,}; "
                f"elapsed={monotonic() - scoring_started_at:.1f}s",
            )
            _report_progress(
                progress_callback,
                "writing updated graph arrays and final LP artifacts",
            )
            updated_edge_count, ap_removals, active_aps = _write_updated_edge_arrays(
                mapped,
                staging,
                added_path,
                removed_path,
                added_count=added_count,
                removed_count=removed_count,
                goodness_available=True,
            )
            if not added_count and not removed_count and not ap_removals:
                outcome = "no_change"
            updated_graph_id = _stream_updated_graph_id(
                mapped, staging, active_aps, ap_removals
            )

        _copy_final_features(goodness_path, staging / "goodness_features.npy")
        _copy_final_features(fairness_path, staging / "fairness_features.npy")
        if not converged:
            _write_updated_edge_arrays(
                mapped,
                staging,
                added_path,
                removed_path,
                added_count=0,
                removed_count=0,
                goodness_available=False,
            )

        # Only the final independent G/F buffers belong in the published
        # artifact. The two ping-pong work pairs have served their purpose.
        for scratch_name in (
            "goodness_a.npy",
            "goodness_b.npy",
            "fairness_a.npy",
            "fairness_b.npy",
        ):
            (staging / scratch_name).unlink(missing_ok=True)

        arrays = _result_array_artifacts(staging)
        text_artifacts = tuple(
            _text_artifact(staging / name, name)
            for name in (
                "candidate_scores.jsonl",
                "added_edges.jsonl",
                "removed_edges.jsonl",
            )
        )
        summary = {
            "input_id": input_manifest.input_id,
            "parent_graph_id": input_manifest.graph_id,
            "updated_graph_id": updated_graph_id,
            "gnn_checkpoint_sha256": input_manifest.gnn_checkpoint_sha256,
            "config": selected.model_dump(mode="json"),
            "candidate_policy": (
                "section6_supported_pairs_v1"
                if candidate_pairs is None
                else "explicit_sorted_stable_ids_v1"
            ),
            "candidate_set_sha256": candidate_set_hasher.hexdigest(),
            "outcome": outcome,
            "termination": termination,
            "iterations": iterations,
            "goodness_residual": g_residual,
            "fairness_residual": f_residual,
            "candidate_count": candidate_count,
            "added_count": added_count,
            "removed_count": removed_count,
            "ap_removals": ap_removals,
            "arrays": [item.model_dump(mode="json") for item in arrays],
            "text_artifacts": [
                item.model_dump(mode="json") for item in text_artifacts
            ],
        }
        result_id = canonical_json_digest(summary)
        result_manifest = StreamingLPResultManifest(
            result_id=result_id,
            input_id=input_manifest.input_id,
            parent_graph_id=input_manifest.graph_id,
            updated_graph_id=updated_graph_id,
            dataset_id=input_manifest.dataset_id,
            batch_id=input_manifest.batch_id,
            gnn_checkpoint_sha256=input_manifest.gnn_checkpoint_sha256,
            vocabulary_sha256=input_manifest.vocabulary_sha256,
            config=selected,
            candidate_policy=(
                "section6_supported_pairs_v1"
                if candidate_pairs is None
                else "explicit_sorted_stable_ids_v1"
            ),
            candidate_set_sha256=candidate_set_hasher.hexdigest(),
            outcome=outcome,
            termination=termination,
            iterations=iterations,
            goodness_residual=g_residual,
            fairness_residual=f_residual,
            radio_weight_max=input_manifest.radio_weight_max,
            threshold_value=selected.link_threshold,
            threshold_units=selected.threshold_units,
            node_ids=input_manifest.node_ids,
            sample_node_ids=input_manifest.sample_node_ids,
            ap_node_ids=input_manifest.ap_node_ids,
            node_feature_dim=input_manifest.node_feature_dim,
            candidate_count=candidate_count,
            added_count=added_count,
            removed_count=removed_count,
            ap_removals=ap_removals,
            active_ap_node_ids=active_aps,
            updated_edge_count=updated_edge_count,
            failure_reason=failure_reason,
            arrays=arrays,
            text_artifacts=text_artifacts,
        )
        atomic_write_json(result_manifest, staging / "manifest.json")
        _write_completion(staging, "manifest.json", result_id)
        _publish_directory(staging, target)
        _report_progress(
            progress_callback,
            f"complete: outcome={result_manifest.outcome}, "
            f"iterations={result_manifest.iterations:,}, "
            f"candidates={result_manifest.candidate_count:,}, "
            f"elapsed={monotonic() - started_at:.1f}s; output={target}",
        )
        return StreamingLPResult(target, result_manifest)
    except Exception:
        # Failed staging directories have no completion marker and cannot be
        # loaded as results. Keep them for diagnosis; never recursively delete.
        raise


def load_streaming_lp_result(
    directory: str | Path,
    *,
    input_directory: str | Path | None = None,
    verify_hashes: bool = True,
) -> StreamingLPResult:
    root = Path(directory)
    marker_path = root / "COMPLETED.json"
    manifest_path = root / "manifest.json"
    if not marker_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"streaming LP result is incomplete: {root}")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    manifest = StreamingLPResultManifest.model_validate_json(read_json_text(manifest_path))
    if marker.get("identity") != manifest.result_id or marker.get(
        "manifest_sha256"
    ) != sha256_file(manifest_path):
        raise ValueError("streaming LP result completion marker is inconsistent")
    for artifact in manifest.arrays:
        path = root / artifact.name
        if not path.is_file() or (verify_hashes and sha256_file(path) != artifact.sha256):
            raise ValueError(f"streaming LP result array is missing or corrupt: {artifact.name}")
        _read_npy(path, artifact, verify_hash=False)
    for artifact in manifest.text_artifacts:
        path = root / artifact.name
        if not path.is_file() or (verify_hashes and sha256_file(path) != artifact.sha256):
            raise ValueError(f"streaming LP result table is missing or corrupt: {artifact.name}")
    expected_records = {
        "candidate_scores.jsonl": manifest.candidate_count,
        "added_edges.jsonl": manifest.added_count,
        "removed_edges.jsonl": manifest.removed_count,
    }
    for name, expected_count in expected_records.items():
        with (root / name).open("r", encoding="utf-8") as stream:
            observed_count = sum(1 for line in stream if line.strip())
        if observed_count != expected_count:
            raise ValueError(f"streaming LP result record count mismatch: {name}")
    if input_directory is not None:
        input_manifest, _mapped = load_streaming_lp_input(
            input_directory, verify_hashes=verify_hashes
        )
        if (
            input_manifest.input_id != manifest.input_id
            or input_manifest.graph_id != manifest.parent_graph_id
            or input_manifest.gnn_checkpoint_sha256 != manifest.gnn_checkpoint_sha256
            or input_manifest.vocabulary_sha256 != manifest.vocabulary_sha256
        ):
            raise ValueError("streaming LP result does not match the supplied input lineage")
    return StreamingLPResult(root, manifest)


__all__ = [
    "StreamingLPInputManifest",
    "StreamingLPResult",
    "StreamingLPResultManifest",
    "load_streaming_lp_input",
    "load_streaming_lp_result",
    "predict_edges_streaming",
    "save_streaming_lp_input",
    "validate_streaming_lp_input",
]
