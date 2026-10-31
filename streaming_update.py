"""Disk-backed online graph construction for the batch updater.

This module is the first downstream consumer of the streamed link-prediction
arrays. It builds the same temporary online graph as the eager updater without
constructing candidate-sized Python edge tuples. Virtual edges and bounded
negative pairs use normalized RSSI rows; coordinates remain in GNN node and
edge vectors. Pair enumeration is still quadratic in sample count.

The returned numeric arrays are read-only NumPy memmaps. Stable IDs and small
lineage/configuration metadata stay in the JSON manifest.
"""

from __future__ import annotations

import math
import json
import multiprocessing
import os
import shutil
import sqlite3
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from time import monotonic
from typing import Callable, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ae import EncoderFeatureArtifact
from graph_build import (
    GraphBuildConfig,
    _graph_id,
    _pseudo_location,
    resolve_spatial_transform,
)
from gufu_artifacts import atomic_write_json, canonical_json_digest, read_json_text, sha256_file
from gufu_data import PreparedDataset, raw_dbm_row_to_ae, rssi_dbm_to_graph_weight
from spatial_normalization import model_position_feature
from rssi_pairs import (
    BoundedNegativeSampler,
    RSSI_SIMILARITY_SPACE,
    normalized_rssi_cosine,
)
from lp_streaming import (
    StreamArrayArtifact,
    StreamingLPInputManifest,
    _array_artifact as _lp_array_artifact,
    _create_staging_directory as _lp_create_staging_directory,
    _finish_npy as _lp_finish_npy,
    _open_npy as _lp_open_npy,
    _prefix as _lp_prefix,
    _publish_directory as _lp_publish_directory,
    _sort_csr_rows as _lp_sort_csr_rows,
    _write_completion as _lp_write_completion,
    load_streaming_lp_input,
    load_streaming_lp_result,
    predict_edges_streaming,
)
from lp import LinkPredictionConfig
from graph_build import SampleLocationInitialization


ONLINE_GRAPH_FORMAT = "gufu_streaming_online_graph_v4"

_VIRTUAL_PAIR_DTYPE = np.dtype(
    [("left_pos", "<i8"), ("right_pos", "<i8"), ("similarity", "<f8")]
)
_RSSI_WORKER_FEATURES: np.ndarray | None = None
_RSSI_WORKER_MASKS: np.ndarray | None = None
_RSSI_WORKER_FEATURE_ROWS: np.ndarray | None = None
_RSSI_WORKER_MINIMUM_SHARED_APS = 0
_RSSI_WORKER_VIRTUAL_SIGMA = 0.0
_RSSI_WORKER_NEGATIVE_THRESHOLD = 0.0
_RSSI_WORKER_NEGATIVE_CAP = 0
_RSSI_WORKER_PROGRESS_QUEUE = None


def _initialize_rssi_pair_worker(
    feature_path: str,
    mask_path: str,
    lexical_feature_rows: np.ndarray,
    minimum_shared_aps: int,
    virtual_edge_sigma: float,
    negative_similarity_threshold: float,
    max_negative_pairs_per_sample: int,
    progress_queue,
) -> None:
    """Open read-only pair-mining inputs once in each worker process."""
    global _RSSI_WORKER_FEATURES, _RSSI_WORKER_MASKS
    global _RSSI_WORKER_FEATURE_ROWS, _RSSI_WORKER_MINIMUM_SHARED_APS
    global _RSSI_WORKER_VIRTUAL_SIGMA, _RSSI_WORKER_NEGATIVE_THRESHOLD
    global _RSSI_WORKER_NEGATIVE_CAP
    global _RSSI_WORKER_PROGRESS_QUEUE
    _RSSI_WORKER_FEATURES = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    _RSSI_WORKER_MASKS = np.load(mask_path, mmap_mode="r", allow_pickle=False)
    _RSSI_WORKER_FEATURE_ROWS = np.asarray(lexical_feature_rows, dtype=np.int64)
    _RSSI_WORKER_MINIMUM_SHARED_APS = int(minimum_shared_aps)
    _RSSI_WORKER_VIRTUAL_SIGMA = float(virtual_edge_sigma)
    _RSSI_WORKER_NEGATIVE_THRESHOLD = float(negative_similarity_threshold)
    _RSSI_WORKER_NEGATIVE_CAP = int(max_negative_pairs_per_sample)
    _RSSI_WORKER_PROGRESS_QUEUE = progress_queue


def _mine_rssi_pair_chunk(
    task: tuple[int, int, int, str],
) -> tuple[int, int, tuple[tuple[int, int, float], ...], int]:
    """Score one contiguous range of left endpoints and spool virtual pairs."""
    if (
        _RSSI_WORKER_FEATURES is None
        or _RSSI_WORKER_MASKS is None
        or _RSSI_WORKER_FEATURE_ROWS is None
    ):
        raise RuntimeError("RSSI pair worker was not initialized")
    task_id, left_start, left_stop, spool_path = task
    sample_count = len(_RSSI_WORKER_FEATURE_ROWS)
    chunk_pair_count = sum(
        sample_count - left_pos - 1 for left_pos in range(left_start, left_stop)
    )
    progress_step = max(1, min(100_000, math.ceil(chunk_pair_count / 4)))
    started_at = last_progress_at = monotonic()
    checked_pairs = virtual_count = 0
    best_by_sample: list[list[tuple[float, int]]] = [
        [] for _ in range(sample_count)
    ]
    spool_buffer: list[tuple[int, int, float]] = []

    def flush_spool(spool_file) -> None:
        if spool_buffer:
            np.asarray(spool_buffer, dtype=_VIRTUAL_PAIR_DTYPE).tofile(spool_file)
            spool_buffer.clear()

    with open(spool_path, "wb") as spool_file:
        for left_pos in range(left_start, left_stop):
            left_row = int(_RSSI_WORKER_FEATURE_ROWS[left_pos])
            left = _RSSI_WORKER_FEATURES[left_row]
            left_mask = _RSSI_WORKER_MASKS[left_row]
            for right_pos in range(left_pos + 1, sample_count):
                right_row = int(_RSSI_WORKER_FEATURE_ROWS[right_pos])
                similarity = normalized_rssi_cosine(
                    left,
                    _RSSI_WORKER_FEATURES[right_row],
                    left_mask,
                    _RSSI_WORKER_MASKS[right_row],
                    minimum_shared_aps=_RSSI_WORKER_MINIMUM_SHARED_APS,
                )
                checked_pairs += 1
                if similarity is None:
                    continue
                if similarity > _RSSI_WORKER_VIRTUAL_SIGMA:
                    spool_buffer.append((left_pos, right_pos, similarity))
                    virtual_count += 1
                    if len(spool_buffer) >= 8192:
                        flush_spool(spool_file)
                elif similarity <= _RSSI_WORKER_NEGATIVE_THRESHOLD:
                    for anchor, other in (
                        (left_pos, right_pos),
                        (right_pos, left_pos),
                    ):
                        candidates = best_by_sample[anchor]
                        candidates.append((float(similarity), other))
                        # lexical sample positions are already sorted by ID,
                        # which is the sampler's deterministic tie-break.
                        candidates.sort(key=lambda item: (-item[0], item[1]))
                        del candidates[_RSSI_WORKER_NEGATIVE_CAP :]

                now = monotonic()
                if (
                    checked_pairs % progress_step == 0
                    or now - last_progress_at >= 10.0
                ):
                    if _RSSI_WORKER_PROGRESS_QUEUE is not None:
                        _RSSI_WORKER_PROGRESS_QUEUE.put(
                            (task_id, checked_pairs)
                        )
                    last_progress_at = now
        flush_spool(spool_file)

    negative_candidates: dict[tuple[int, int], float] = {}
    for anchor, candidates in enumerate(best_by_sample):
        for similarity, other in candidates:
            edge = (min(anchor, other), max(anchor, other))
            negative_candidates[edge] = similarity
    ordered_candidates = tuple(
        (left_pos, right_pos, similarity)
        for (left_pos, right_pos), similarity in sorted(negative_candidates.items())
    )
    return (
        task_id,
        virtual_count,
        ordered_candidates,
        checked_pairs,
    )


def _parallel_mine_rssi_relations(
    *,
    sample_ids: tuple[str, ...],
    lexical_feature_rows: np.ndarray,
    signal_features_path: Path,
    signal_masks_path: Path,
    spool_directory: Path,
    minimum_shared_aps: int,
    virtual_edge_sigma: float,
    negative_similarity_threshold: float,
    max_negative_pairs_per_sample: int,
    num_workers: int,
    progress_callback: Callable[[str], None] | None,
) -> tuple[int, BoundedNegativeSampler]:
    """Mine relations in parallel and spool virtual edges in stable order."""
    sample_count = len(sample_ids)
    pair_count = sample_count * (sample_count - 1) // 2
    sampler = BoundedNegativeSampler(
        sample_ids,
        negative_similarity_threshold=negative_similarity_threshold,
        max_pairs_per_sample=max_negative_pairs_per_sample,
    )
    spool_directory.mkdir(parents=True, exist_ok=True)
    available_left_rows = max(0, sample_count - 1)
    if available_left_rows == 0:
        if progress_callback is not None:
            progress_callback(
                "[pipeline] RSSI pair pass 'mine relations' complete: "
                "0/0 pairs; virtual edges=0; negatives=0; elapsed=0.0s"
            )
        return 0, sampler

    effective_workers = min(num_workers, available_left_rows)
    # Balance ranges by pair count rather than row count: early left rows have
    # many more comparisons than late rows. One range per worker also bounds
    # the size of each worker's local negative-candidate summary.
    def pair_prefix(left_rows: int) -> int:
        return left_rows * (2 * sample_count - left_rows - 1) // 2

    task_count = effective_workers
    boundaries = [0]
    for part in range(1, task_count):
        target_pairs = pair_count * part // task_count
        low = boundaries[-1] + 1
        high = available_left_rows - (task_count - part)
        while low < high:
            middle = (low + high) // 2
            if pair_prefix(middle) < target_pairs:
                low = middle + 1
            else:
                high = middle
        options = [low]
        if low - 1 > boundaries[-1]:
            options.append(low - 1)
        boundary = min(options, key=lambda value: abs(pair_prefix(value) - target_pairs))
        boundaries.append(boundary)
    boundaries.append(available_left_rows)
    task_ranges = tuple(zip(boundaries[:-1], boundaries[1:]))
    progress_step = max(1, min(50_000, math.ceil(pair_count / 20)))
    if progress_callback is not None:
        progress_callback(
            f"[pipeline] RSSI pair pass 'mine relations' started: "
            f"{sample_count:,} samples, {pair_count:,} possible pairs; "
            f"workers={effective_workers}, tasks={task_count:,}"
        )

    # Explicit fork keeps this usable from the repository's notebook-driven
    # Linux workflow. Workers touch only read-only NumPy maps and Python math;
    # platforms without fork use their available spawn implementation.
    start_method = (
        "fork"
        if "fork" in multiprocessing.get_all_start_methods()
        else multiprocessing.get_all_start_methods()[0]
    )
    context = multiprocessing.get_context(start_method)
    progress_queue = context.Queue()
    total_processed = 0
    virtual_count = 0
    completed_tasks = 0
    active_progress: dict[int, int] = {}
    last_reported_pairs = 0
    started_at = last_report_at = monotonic()
    next_task_id = 0
    pending = {}

    def task_values(task_id: int):
        left_start, left_stop = task_ranges[task_id]
        task_pairs = sum(
            sample_count - left_pos - 1
            for left_pos in range(left_start, left_stop)
        )
        spool_path = spool_directory / f"virtual_pairs_{left_start:012d}.bin"
        return left_start, left_stop, task_pairs, spool_path

    def submit_available(executor: ProcessPoolExecutor) -> None:
        nonlocal next_task_id
        while next_task_id < task_count and len(pending) < effective_workers * 2:
            task_id = next_task_id
            next_task_id += 1
            left_start, left_stop, task_pairs, spool_path = task_values(task_id)
            active_progress[task_id] = 0
            future = executor.submit(
                _mine_rssi_pair_chunk,
                (task_id, left_start, left_stop, str(spool_path)),
            )
            pending[future] = (task_id, task_pairs)

    def receive_worker_progress() -> None:
        nonlocal total_processed
        while True:
            try:
                task_id, checked_pairs = progress_queue.get_nowait()
            except Empty:
                return
            previous = active_progress.get(task_id)
            if previous is None or checked_pairs <= previous:
                continue
            total_processed += checked_pairs - previous
            active_progress[task_id] = checked_pairs

    def report_progress(*, force: bool = False) -> None:
        nonlocal last_reported_pairs, last_report_at
        now = monotonic()
        if progress_callback is None:
            return
        if (
            not force
            and total_processed - last_reported_pairs < progress_step
            and now - last_report_at < 10.0
        ):
            return
        progress_callback(
            f"[pipeline] RSSI pair pass 'mine relations': "
            f"{total_processed:,}/{pair_count:,} pairs "
            f"({total_processed / max(1, pair_count):.0%}); "
            f"tasks={completed_tasks:,}/{task_count:,}; "
            f"workers={effective_workers}; elapsed={now - started_at:.1f}s"
        )
        last_reported_pairs = total_processed
        last_report_at = now

    try:
        with ProcessPoolExecutor(
            max_workers=effective_workers,
            mp_context=context,
            initializer=_initialize_rssi_pair_worker,
            initargs=(
                str(signal_features_path),
                str(signal_masks_path),
                lexical_feature_rows,
                minimum_shared_aps,
                virtual_edge_sigma,
                negative_similarity_threshold,
                max_negative_pairs_per_sample,
                progress_queue,
            ),
        ) as executor:
            submit_available(executor)
            while pending:
                receive_worker_progress()
                done, _ = wait(
                    tuple(pending), timeout=1.0, return_when=FIRST_COMPLETED
                )
                receive_worker_progress()
                for future in done:
                    task_id, task_pairs = pending.pop(future)
                    (
                        result_task_id,
                        task_virtual_count,
                        negative_candidates,
                        checked_pairs,
                    ) = future.result()
                    if result_task_id != task_id or checked_pairs != task_pairs:
                        raise RuntimeError("RSSI pair worker returned inconsistent task results")
                    previous = active_progress.pop(task_id, 0)
                    total_processed += checked_pairs - previous
                    virtual_count += task_virtual_count
                    completed_tasks += 1
                    for left_pos, right_pos, similarity in negative_candidates:
                        sampler.add(left_pos, right_pos, similarity)
                    report_progress(force=completed_tasks == task_count)
                submit_available(executor)
                report_progress()
        receive_worker_progress()
        if progress_callback is not None:
            progress_callback(
                f"[pipeline] RSSI pair pass 'mine relations' complete: "
                f"{total_processed:,}/{pair_count:,} pairs; "
                f"virtual edges={virtual_count:,}; "
                f"negative pairs={len(sampler.selected()):,}; "
                f"elapsed={monotonic() - started_at:.1f}s"
            )
    finally:
        progress_queue.close()
        progress_queue.join_thread()

    if total_processed != pair_count:
        raise RuntimeError(
            f"RSSI pair mining checked {total_processed} pairs; expected {pair_count}"
        )
    return virtual_count, sampler


class StreamingOnlineGraphManifest(BaseModel):
    """Identity, roles, labels, and array index for one transient update graph."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_streaming_online_graph_v4"] = ONLINE_GRAPH_FORMAT
    online_graph_id: str = Field(pattern=r"^onlinegraph:[0-9a-f]{64}$")
    dataset_id: str
    reference_batch_id: str
    update_batch_id: str
    reference_graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    update_graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    reference_lp_result_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_gnn_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    update_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    update_encoder_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalization: Literal["paper_ae", "legacy_raw_div_120"]
    bssid_vocabulary: tuple[str, ...]
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_build_config: GraphBuildConfig
    virtual_edge_sigma: float = Field(ge=-1.0, le=1.0)
    similarity_feature_space: Literal[
        "normalized_rssi_cosine_min_overlap_v1"
    ] = (
        RSSI_SIMILARITY_SPACE
    )
    encoder_feature_dim: int = Field(ge=1)
    node_feature_dim: int = Field(ge=3)
    node_ids: tuple[str, ...]
    sample_node_ids: tuple[str, ...]
    reference_sample_ids: tuple[str, ...]
    update_sample_ids: tuple[str, ...]
    ap_node_ids: tuple[str, ...]
    node_types: tuple[Literal["sample", "ap"], ...]
    node_feature_sources: tuple[
        Literal[
            "current_reference_gnn_state",
            "new_encoder_feature_plus_seeded_pseudo_xy",
            "current_reference_ap_gnn_state",
            "new_or_reactivated_ap_graph_mean",
            "post_transition_ae_reencoded_reference_sample",
            "post_transition_reference_ap_weighted_mean",
            "new_encoder_feature_plus_unavailable_position",
            "reference_graph_pre_gnn_features",
        ], ...
    ]
    sample_location_initializations: tuple[SampleLocationInitialization, ...]
    sample_position_available_mask: tuple[bool, ...]
    labeled_reference_ids: tuple[str, ...]
    labeled_reference_coordinates: tuple[tuple[float, float], ...]
    radio_edge_count: int = Field(ge=0)
    virtual_edge_count: int = Field(ge=0)
    negative_pair_count: int = Field(ge=0)
    adjacency_shape: tuple[int, int]
    adjacency_nonzero_count: int = Field(ge=0)
    arrays: tuple[StreamArrayArtifact, ...]

    @model_validator(mode="after")
    def aligned_identity(self):
        if self.virtual_edge_sigma != self.graph_build_config.virtual_edge_sigma:
            raise ValueError("streamed online sigma must match its graph-build config")
        if self.similarity_feature_space != self.graph_build_config.similarity_feature_space:
            raise ValueError("streamed online similarity space must match graph-build config")
        if self.node_ids != self.sample_node_ids + self.ap_node_ids:
            raise ValueError("online node rows must be samples followed by APs")
        if self.sample_node_ids != self.reference_sample_ids + self.update_sample_ids:
            raise ValueError("online samples must preserve reference then update order")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("online graph node IDs must be unique")
        if self.node_types != ("sample",) * len(self.sample_node_ids) + (
            "ap",
        ) * len(self.ap_node_ids):
            raise ValueError("online graph node types do not align")
        if len(self.node_feature_sources) != len(self.node_ids):
            raise ValueError("online feature provenance must align with node IDs")
        if len(self.sample_location_initializations) != len(self.sample_node_ids):
            raise ValueError("online sample location metadata must align with sample IDs")
        if tuple(item.scan_id for item in self.sample_location_initializations) != self.sample_node_ids:
            raise ValueError("online sample locations must follow sample row order")
        if len(self.sample_position_available_mask) != len(self.sample_node_ids):
            raise ValueError("online position availability must align with sample IDs")
        if any(
            available and initialization.source != "bootstrap_reference_label"
            for available, initialization in zip(
                self.sample_position_available_mask, self.sample_location_initializations
            )
        ):
            raise ValueError("available online positions must have bootstrap label provenance")
        available_ids = {
            scan_id
            for scan_id, available in zip(
                self.sample_node_ids, self.sample_position_available_mask
            )
            if available
        }
        if available_ids != set(self.labeled_reference_ids):
            raise ValueError("online position availability must match labeled reference IDs")
        if self.node_feature_dim != self.encoder_feature_dim + 2:
            raise ValueError("online graph width must equal encoder width plus XY")
        if self.graph_build_config.spatial_normalization_enabled:
            metadata = self.graph_build_config.spatial_transform_metadata
            if metadata is None or self.graph_build_config.spatial_transform_sha256 is None:
                raise ValueError("online graph must record its fitted spatial transform")
            if metadata.site_id != self.dataset_id:
                raise ValueError("online graph spatial transform belongs to another site")
        if len(self.labeled_reference_ids) != len(self.labeled_reference_coordinates):
            raise ValueError("bootstrap label IDs and coordinates must align")
        if not set(self.labeled_reference_ids) <= set(self.reference_sample_ids):
            raise ValueError("only reference scans may appear as training labels")
        if self.adjacency_shape != (
            len(self.update_sample_ids), len(self.labeled_reference_ids)
        ):
            raise ValueError("sparse update/reference adjacency dimensions are invalid")
        if canonical_json_digest(self.bssid_vocabulary) != self.vocabulary_sha256:
            raise ValueError("online graph vocabulary digest is inconsistent")
        names = tuple(item.name for item in self.arrays)
        expected = {
            "node_features.npy",
            "sample_signal_features.npy",
            "sample_signal_masks.npy",
            "radio_endpoints.npy",
            "radio_weights.npy",
            "radio_edge_features.npy",
            "radio_edge_sources.npy",
            "virtual_endpoints.npy",
            "virtual_similarities.npy",
            "virtual_edge_features.npy",
            "negative_endpoints.npy",
            "negative_similarities.npy",
            "negative_strengths.npy",
            "update_reference_indptr.npy",
            "update_reference_cols.npy",
            "update_reference_values.npy",
        }
        if len(names) != len(set(names)) or set(names) != expected:
            raise ValueError("online graph array index is incomplete or duplicated")
        shapes = {item.name: item.shape for item in self.arrays}
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
            "radio_endpoints.npy": (self.radio_edge_count, 2),
            "radio_weights.npy": (self.radio_edge_count,),
            "radio_edge_features.npy": (self.radio_edge_count, self.node_feature_dim),
            "radio_edge_sources.npy": (self.radio_edge_count,),
            "virtual_endpoints.npy": (self.virtual_edge_count, 2),
            "virtual_similarities.npy": (self.virtual_edge_count,),
            "virtual_edge_features.npy": (self.virtual_edge_count, self.node_feature_dim),
            "negative_endpoints.npy": (self.negative_pair_count, 2),
            "negative_similarities.npy": (self.negative_pair_count,),
            "negative_strengths.npy": (self.negative_pair_count,),
            "update_reference_indptr.npy": (len(self.update_sample_ids) + 1,),
            "update_reference_cols.npy": (self.adjacency_nonzero_count,),
            "update_reference_values.npy": (self.adjacency_nonzero_count,),
        }
        if shapes != expected_shapes:
            raise ValueError("online graph numeric array shapes disagree with metadata")
        identity = canonical_json_digest(
            self.model_dump(mode="json", exclude={"online_graph_id"})
        )
        if self.online_graph_id != f"onlinegraph:{identity}":
            raise ValueError("online graph identity does not match its manifest")
        return self


@dataclass(frozen=True)
class MappedStreamingOnlineGraph:
    directory: Path
    manifest: StreamingOnlineGraphManifest
    node_features: np.ndarray
    sample_signal_features: np.ndarray
    sample_signal_masks: np.ndarray
    radio_endpoints: np.ndarray
    radio_weights: np.ndarray
    radio_edge_features: np.ndarray
    radio_edge_sources: np.ndarray
    virtual_endpoints: np.ndarray
    virtual_similarities: np.ndarray
    virtual_edge_features: np.ndarray
    negative_endpoints: np.ndarray
    negative_similarities: np.ndarray
    negative_strengths: np.ndarray
    update_reference_indptr: np.ndarray
    update_reference_cols: np.ndarray
    update_reference_values: np.ndarray


@dataclass(frozen=True)
class StreamingAPInferenceState:
    """Disk-backed candidate scores and edge decisions for one AP transition."""

    input_directory: Path
    candidate_index_path: Path
    result: object


def _array_artifact(path: Path, name: str) -> StreamArrayArtifact:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return StreamArrayArtifact(
        name=name,
        dtype=array.dtype.name,
        shape=tuple(int(value) for value in array.shape),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _jsonable(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _open_array(path: Path, dtype: str, shape: tuple[int, ...]) -> np.ndarray:
    if math.prod(shape) == 0:
        array = np.empty(shape, dtype=dtype)
        np.save(path, array, allow_pickle=False)
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _prefix(counts: np.ndarray) -> np.ndarray:
    result = np.empty((len(counts) + 1,), dtype=np.int64)
    result[0] = 0
    np.cumsum(counts, out=result[1:])
    return result


def _cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    # Python accumulation intentionally matches fingerprint_update._cosine's
    # sequential float64 sums, including its zero-norm policy.
    left_values = tuple(float(value) for value in left)
    right_values = tuple(float(value) for value in right)
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    dot = sum(a * b for a, b in zip(left_values, right_values))
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def _complete(staging: Path, manifest: StreamingOnlineGraphManifest) -> None:
    atomic_write_json(
        {
            "format": "gufu_streaming_artifact_complete_v1",
            "identity": manifest.online_graph_id,
            "manifest_sha256": sha256_file(staging / "manifest.json"),
        },
        staging / "COMPLETED.json",
    )


def build_streaming_online_graph(
    reference_lp_input_directory: str | Path,
    reference_lp_result_directory: str | Path,
    update_prepared: PreparedDataset,
    update_encoder_features: EncoderFeatureArtifact,
    output_directory: str | Path,
    *,
    virtual_edge_sigma: float,
    spatial_transform_path_override: str | Path | None = None,
    reference_node_ids_override: tuple[str, ...] | None = None,
    reference_node_features_override: np.ndarray | None = None,
    num_workers: int = 4,
    progress_callback: Callable[[str], None] | None = None,
) -> StreamingOnlineGraphManifest:
    """Build disk-backed temporary graph state directly from the streamed LP.

    The procedure uses exactly the local updater's initialized feature spaces:
    reference node features come from the selected reference GNN artifact,
    unless matching pre-GNN node features are supplied as an override;
    update sample/AP features come from ``build_graph_state``; newly measured
    update radio-edge vectors are endpoint means; when a pre-GNN reference
    override is supplied, LP radio-edge features are also reset to endpoint
    means so every edge matches the GNN's training input space. Virtual similarities use
    normalized RSSI rows and strict ``cosine > sigma``. Bounded negative pairs
    use the same RSSI rows with a separate conservative threshold. The full
    ``z_encoder || XY`` vectors remain in node and edge features.
    RSSI pair mining uses up to ``num_workers`` processes (default four).
    Accepted virtual-pair records are spooled under the staging directory and
    replayed in stable order while the output arrays are filled.
    ``spatial_transform_path_override`` lets an evaluator relocate the saved
    transform after an output directory has been renamed; its contents must
    still match the identity recorded in the LP input manifest.
    Update evaluation coordinates are not read for labels or initialization.
    """
    if not math.isfinite(virtual_edge_sigma) or not -1.0 <= virtual_edge_sigma <= 1.0:
        raise ValueError("virtual_edge_sigma must be finite and in [-1, 1]")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 1:
        raise ValueError("num_workers must be a positive integer")
    input_manifest, mapped = load_streaming_lp_input(reference_lp_input_directory)
    if (reference_node_ids_override is None) != (reference_node_features_override is None):
        raise ValueError("reference node feature override requires both node IDs and features")
    if reference_node_features_override is not None:
        reference_node_features_override = np.asarray(
            reference_node_features_override, dtype=np.float64
        )
        if reference_node_ids_override != input_manifest.node_ids:
            raise ValueError("reference feature override IDs/order differ from the LP input")
        if reference_node_features_override.shape != (
            len(input_manifest.node_ids), input_manifest.node_feature_dim
        ):
            raise ValueError("reference feature override shape differs from the LP input")
        if not np.isfinite(reference_node_features_override).all():
            raise ValueError("reference feature override contains non-finite values")
    lp_result = load_streaming_lp_result(
        reference_lp_result_directory,
        input_directory=reference_lp_input_directory,
    )
    result = lp_result.manifest
    if result.outcome == "failed" or result.updated_graph_id is None:
        raise ValueError("cannot build an online graph from a failed LP result")
    if input_manifest.batch_role != "bootstrap_reference":
        raise ValueError("streaming online graph requires a bootstrap LP input")
    if update_prepared.config.batch_role != "update_batch":
        raise ValueError("online graph input must have update_batch role")
    if update_prepared.config.dataset_id != input_manifest.dataset_id:
        raise ValueError("reference and update dataset IDs differ")
    if update_prepared.config.normalization_mode != input_manifest.normalization:
        raise ValueError("reference and update normalization differs")
    if update_prepared.bssid_vocabulary != input_manifest.bssid_vocabulary:
        raise ValueError("streaming online graph currently requires unchanged vocabulary")
    if update_encoder_features.dataset_id != update_prepared.config.dataset_id or update_encoder_features.batch_id != update_prepared.config.batch_id:
        raise ValueError("update encoder artifact belongs to different dataset/batch")
    if update_encoder_features.source_prepared_sha256 != canonical_json_digest(update_prepared):
        raise ValueError("update encoder artifact was produced from different prepared data")
    if update_encoder_features.scan_ids != tuple(scan.scan_id for scan in update_prepared.scans):
        raise ValueError("update encoder rows must exactly follow prepared scan order")
    if update_encoder_features.bssid_vocabulary != input_manifest.bssid_vocabulary:
        raise ValueError("update encoder vocabulary order differs from reference")
    if update_encoder_features.normalization != input_manifest.normalization:
        raise ValueError("update encoder normalization differs from reference")
    if update_encoder_features.checkpoint_sha256 != input_manifest.encoder_checkpoint_sha256:
        raise ValueError("update encoder does not use the reference AE checkpoint")
    if update_encoder_features.feature_dim != input_manifest.encoder_feature_dim:
        raise ValueError("update encoder feature width differs from reference graph")
    if update_encoder_features.feature_dim + 2 != input_manifest.node_feature_dim:
        raise ValueError("update graph feature width differs from reference graph")
    if input_manifest.graph_build_config.virtual_edge_strategy != "paper_cosine_threshold":
        raise ValueError("streaming updater requires paper cosine-threshold graph configuration")
    if virtual_edge_sigma != input_manifest.graph_build_config.virtual_edge_sigma:
        raise ValueError("online virtual sigma must match the recorded graph configuration")
    graph_build_config_for_resolution = input_manifest.graph_build_config
    if spatial_transform_path_override is not None:
        graph_build_config_for_resolution = graph_build_config_for_resolution.model_copy(
            update={"spatial_transform_path": str(spatial_transform_path_override)}
        )
    resolved_build_config, spatial_transform = resolve_spatial_transform(
        graph_build_config_for_resolution, input_manifest.dataset_id
    )
    if (
        resolved_build_config.model_copy(
            update={
                "spatial_transform_path": input_manifest.graph_build_config.spatial_transform_path
            }
        )
        != input_manifest.graph_build_config
    ):
        raise ValueError("online graph spatial configuration was not fully resolved at bootstrap")

    reference_sample_ids = input_manifest.sample_node_ids
    update_sample_ids = tuple(scan.scan_id for scan in update_prepared.scans)
    sample_ids = reference_sample_ids + update_sample_ids
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("new batch scan IDs collide with reference IDs")

    # Construct only the per-scan update features and AP accumulators here.
    # Calling build_graph_state would enumerate and retain every update-batch
    # virtual pair before this disk-backed path gets a chance to stream them.
    bounds = input_manifest.graph_build_config.pseudo_location_bounds
    if bounds is None:
        raise ValueError("streaming update graph requires recorded bootstrap map bounds")
    update_encoder_rows = {
        scan_id: row for row, scan_id in enumerate(update_encoder_features.scan_ids)
    }
    update_scan_by_id = {scan.scan_id: scan for scan in update_prepared.scans}
    update_pseudo_locations = {
        scan_id: _pseudo_location(
            update_scan_by_id[scan_id],
            input_manifest.graph_build_config.pseudo_location_seed,
            bounds,
        )
        for scan_id in update_sample_ids
    }
    update_ap_ids = tuple(
        ap_id
        for column, ap_id in enumerate(update_prepared.ap_ids)
        if any(scan.observation_mask[column] for scan in update_prepared.scans)
    )
    update_node_ids = update_sample_ids + update_ap_ids
    update_graph_id = _graph_id(
        source_prepared_sha256=canonical_json_digest(update_prepared),
        encoder_feature_artifact_sha256=canonical_json_digest(update_encoder_features),
        build_config=input_manifest.graph_build_config,
        node_ids=update_node_ids,
    )

    input_rows = {node_id: row for row, node_id in enumerate(input_manifest.node_ids)}
    result_arrays = {
        item.name: np.load(
            Path(reference_lp_result_directory) / item.name,
            mmap_mode="r" if math.prod(item.shape) else None,
            allow_pickle=False,
        )
        for item in result.arrays
    }
    updated_endpoints = result_arrays["updated_edge_endpoints.npy"]
    updated_weights = result_arrays["updated_edge_weights.npy"]
    updated_edge_features = result_arrays["updated_edge_features.npy"]
    updated_edge_sources = result_arrays["updated_edge_feature_sources.npy"]
    active_old_ap_ids = set(result.active_ap_node_ids)
    update_ap_id_set = set(update_ap_ids)
    ap_ids = tuple(
        ap_id
        for ap_id in input_manifest.ap_ids
        if ap_id in active_old_ap_ids or ap_id in update_ap_id_set
    )
    node_ids = sample_ids + ap_ids
    new_rows = {node_id: row for row, node_id in enumerate(node_ids)}
    feature_dim = input_manifest.node_feature_dim

    target = Path(output_directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite streaming online graph: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    database_path = staging / "radio_sort.sqlite3"
    try:
        node_features = _open_array(
            staging / "node_features.npy", "float64", (len(node_ids), feature_dim)
        )
        sample_signal_features = _open_array(
            staging / "sample_signal_features.npy",
            "float64",
            (len(sample_ids), len(input_manifest.bssid_vocabulary)),
        )
        sample_signal_masks = _open_array(
            staging / "sample_signal_masks.npy",
            "uint8",
            (len(sample_ids), len(input_manifest.bssid_vocabulary)),
        )
        node_sources: list[str] = []
        for row, node_id in enumerate(reference_sample_ids):
            source_row = input_rows[node_id]
            node_features[row] = (
                mapped.node_features[source_row]
                if reference_node_features_override is None
                else reference_node_features_override[source_row]
            )
            sample_signal_features[row] = mapped.sample_signal_features[row]
            sample_signal_masks[row] = mapped.sample_signal_masks[row]
            node_sources.append(
                "current_reference_gnn_state"
                if reference_node_features_override is None
                else "reference_graph_pre_gnn_features"
            )
        update_rssi_rows = update_prepared.ae_matrix()
        update_rssi_masks = update_prepared.mask_matrix()
        for node_id in update_sample_ids:
            update_row = update_encoder_rows[node_id]
            signal_mask = update_rssi_masks[update_row]
            signal_row = tuple(
                float(value) if observed else 0.0
                for value, observed in zip(update_rssi_rows[update_row], signal_mask)
            )
            sample_signal_features[new_rows[node_id]] = signal_row
            sample_signal_masks[new_rows[node_id]] = signal_mask
            encoder_row = update_encoder_features.feature_matrix[update_row]
            position_raw = update_pseudo_locations[node_id]
            if spatial_transform is None:
                position_feature = position_raw
                source = "new_encoder_feature_plus_seeded_pseudo_xy"
            else:
                # Update truth is unavailable to the GNN. Keep the seeded raw
                # initialization in metadata, but feed a finite zero block.
                position_feature = model_position_feature(
                    spatial_transform,
                    position_raw,
                    available=False,
                    position_weight=input_manifest.graph_build_config.position_weight,
                )
                position_feature = tuple(float(value) for value in position_feature)
                source = "new_encoder_feature_plus_unavailable_position"
            node_features[new_rows[node_id]] = tuple(encoder_row) + tuple(position_feature)
            node_sources.append(source)
        update_ap_row = {ap_id: row for row, ap_id in enumerate(update_ap_ids)}
        update_ap_weight_sums = np.zeros((len(update_ap_ids),), dtype=np.float64)
        update_ap_feature_sums = np.zeros(
            (len(update_ap_ids), feature_dim), dtype=np.float64
        )
        for scan in update_prepared.scans:
            sample_feature = node_features[new_rows[scan.scan_id]]
            for column, ap_id in enumerate(update_prepared.ap_ids):
                if not scan.observation_mask[column]:
                    continue
                weight = rssi_dbm_to_graph_weight(scan.rss_dbm[column], True)
                if weight is None or weight < 0.0:
                    raise ValueError(
                        f"update RSS+120 weight is invalid for {scan.scan_id} -> {ap_id}"
                    )
                row = update_ap_row[ap_id]
                update_ap_weight_sums[row] += weight
                update_ap_feature_sums[row] += weight * sample_feature
        for ap_id in ap_ids:
            old_row = input_rows.get(ap_id)
            if ap_id in active_old_ap_ids and old_row is not None:
                node_features[new_rows[ap_id]] = (
                    mapped.node_features[old_row]
                    if reference_node_features_override is None
                    else reference_node_features_override[old_row]
                )
                node_sources.append(
                    "current_reference_ap_gnn_state"
                    if reference_node_features_override is None
                    else "reference_graph_pre_gnn_features"
                )
            else:
                if ap_id not in update_ap_row:
                    raise ValueError(f"new AP {ap_id} has no update-graph initialization")
                update_row = update_ap_row[ap_id]
                denominator = float(update_ap_weight_sums[update_row])
                if denominator <= 0.0:
                    raise ValueError(
                        f"new AP {ap_id} has zero incident RSS+120 weight; "
                        "initialization is undefined"
                    )
                node_features[new_rows[ap_id]] = (
                    update_ap_feature_sums[update_row] / denominator
                )
                node_sources.append("new_or_reactivated_ap_graph_mean")
        if isinstance(node_features, np.memmap):
            node_features.flush()
        if isinstance(sample_signal_features, np.memmap):
            sample_signal_features.flush()
        if isinstance(sample_signal_masks, np.memmap):
            sample_signal_masks.flush()

        # SQLite gives a bounded-RAM stable external sort and duplicate-pair
        # rejection while preserving the exact lexical edge order expected by
        # the existing GNN and AP-updater implementations.
        connection = sqlite3.connect(database_path)
        connection.execute(
            "CREATE TABLE radio (sample_id TEXT NOT NULL, ap_id TEXT NOT NULL, "
            "weight REAL NOT NULL, feature BLOB NOT NULL, source INTEGER NOT NULL, "
            "PRIMARY KEY(sample_id, ap_id)) WITHOUT ROWID"
        )
        for index, (old_left, old_right) in enumerate(updated_endpoints):
            left_id = input_manifest.node_ids[int(old_left)]
            right_id = input_manifest.node_ids[int(old_right)]
            if left_id.startswith("ap:"):
                sample_id, ap_id = right_id, left_id
            else:
                sample_id, ap_id = left_id, right_id
            if sample_id not in new_rows or ap_id not in new_rows or ap_id not in active_old_ap_ids:
                raise ValueError("streamed LP result contains an invalid/inactive radio endpoint")
            if reference_node_features_override is None:
                feature = np.asarray(updated_edge_features[index], dtype=np.float64)
            else:
                feature = (
                    np.asarray(node_features[new_rows[left_id]], dtype=np.float64)
                    + np.asarray(node_features[new_rows[right_id]], dtype=np.float64)
                ) / 2.0
            connection.execute(
                "INSERT INTO radio VALUES (?, ?, ?, ?, ?)",
                (
                    sample_id,
                    ap_id,
                    float(updated_weights[index]),
                    feature.tobytes(order="C"),
                    int(updated_edge_sources[index]),
                ),
            )
        for scan in update_prepared.scans:
            for column, bssid in enumerate(update_prepared.bssid_vocabulary):
                if not scan.observation_mask[column]:
                    continue
                weight = rssi_dbm_to_graph_weight(scan.rss_dbm[column], True)
                if weight is None or weight < 0.0:
                    raise ValueError(
                        f"update RSS+120 weight is invalid for {scan.scan_id} -> {bssid}"
                    )
                ap_id = f"ap:{bssid}"
            # Existing APs use the reference GNN feature row, while new APs
            # use the update graph's weighted-mean initialization. Initialize
            # this edge from the actual mixed online endpoint rows.
                left = np.asarray(node_features[new_rows[scan.scan_id]], dtype=np.float64)
                right = np.asarray(node_features[new_rows[ap_id]], dtype=np.float64)
                feature = (left + right) / 2.0
                connection.execute(
                    "INSERT INTO radio VALUES (?, ?, ?, ?, ?)",
                    (
                        scan.scan_id,
                        ap_id,
                        float(weight),
                        feature.tobytes(order="C"),
                        2,
                    ),
                )
        connection.commit()
        radio_count = int(connection.execute("SELECT COUNT(*) FROM radio").fetchone()[0])

        radio_endpoints = _open_array(staging / "radio_endpoints.npy", "int64", (radio_count, 2))
        radio_weights = _open_array(staging / "radio_weights.npy", "float64", (radio_count,))
        radio_features = _open_array(
            staging / "radio_edge_features.npy", "float64", (radio_count, feature_dim)
        )
        radio_sources = _open_array(staging / "radio_edge_sources.npy", "uint8", (radio_count,))
        for row, record in enumerate(
            connection.execute(
                "SELECT sample_id, ap_id, weight, feature, source FROM radio "
                "ORDER BY sample_id, ap_id"
            )
        ):
            sample_id, ap_id, weight, feature_bytes, source = record
            radio_endpoints[row] = (new_rows[sample_id], new_rows[ap_id])
            radio_weights[row] = weight
            radio_features[row] = np.frombuffer(feature_bytes, dtype=np.float64)
            radio_sources[row] = source
        for array in (radio_endpoints, radio_weights, radio_features, radio_sources):
            if isinstance(array, np.memmap):
                array.flush()
        connection.close()
        database_path.unlink(missing_ok=True)

        labeled_pairs = tuple(
            (scan_id, coordinate)
            for scan_id, is_labeled, coordinate in zip(
                input_manifest.sample_node_ids,
                input_manifest.reference_label_mask[: len(input_manifest.sample_node_ids)],
                input_manifest.reference_coordinate_targets[: len(input_manifest.sample_node_ids)],
            )
            if is_labeled and coordinate is not None
        )
        labeled_ids = tuple(item[0] for item in labeled_pairs)
        labeled_coordinates = tuple(item[1] for item in labeled_pairs)
        label_col = {scan_id: col for col, scan_id in enumerate(labeled_ids)}
        update_row = {scan_id: row for row, scan_id in enumerate(update_sample_ids)}
        lexical_samples = tuple(sorted(sample_ids))
        sample_row = {scan_id: row for row, scan_id in enumerate(sample_ids)}
        lexical_feature_rows = np.asarray(
            [sample_row[sample_id] for sample_id in lexical_samples], dtype=np.int64
        )
        virtual_spool_directory = staging / "virtual_pair_spool"
        virtual_count, pair_sampler = _parallel_mine_rssi_relations(
            sample_ids=lexical_samples,
            lexical_feature_rows=lexical_feature_rows,
            signal_features_path=staging / "sample_signal_features.npy",
            signal_masks_path=staging / "sample_signal_masks.npy",
            spool_directory=virtual_spool_directory,
            minimum_shared_aps=input_manifest.graph_build_config.minimum_shared_aps,
            virtual_edge_sigma=virtual_edge_sigma,
            negative_similarity_threshold=input_manifest.graph_build_config.negative_similarity_threshold,
            max_negative_pairs_per_sample=input_manifest.graph_build_config.max_negative_pairs_per_sample,
            num_workers=num_workers,
            progress_callback=progress_callback,
        )
        negative_pairs = pair_sampler.selected()
        adjacency_counts = np.zeros((len(update_sample_ids),), dtype=np.int64)

        virtual_endpoints = _open_array(
            staging / "virtual_endpoints.npy", "int64", (virtual_count, 2)
        )
        virtual_similarities = _open_array(
            staging / "virtual_similarities.npy", "float64", (virtual_count,)
        )
        virtual_features = _open_array(
            staging / "virtual_edge_features.npy", "float64", (virtual_count, feature_dim)
        )
        negative_endpoints = _open_array(
            staging / "negative_endpoints.npy", "int64", (len(negative_pairs), 2)
        )
        negative_similarities = _open_array(
            staging / "negative_similarities.npy", "float64", (len(negative_pairs),)
        )
        negative_strengths = _open_array(
            staging / "negative_strengths.npy", "float64", (len(negative_pairs),)
        )
        for index, pair in enumerate(negative_pairs):
            negative_endpoints[index] = (
                sample_row[pair.left_sample_id],
                sample_row[pair.right_sample_id],
            )
            negative_similarities[index] = pair.signal_cosine_similarity
            negative_strengths[index] = pair.negative_strength
        adjacency_db = sqlite3.connect(staging / "adjacency_sort.sqlite3")
        adjacency_db.execute(
            "CREATE TABLE adjacency (update_row INTEGER NOT NULL, ref_col INTEGER NOT NULL, "
            "PRIMARY KEY(update_row, ref_col)) WITHOUT ROWID"
        )
        if progress_callback is not None:
            progress_callback(
                f"[pipeline] RSSI pair pass 'write virtual edges' started from "
                f"{virtual_count:,} spooled relations"
            )
        virtual_write_started = last_virtual_write_progress = monotonic()
        virtual_write_step = max(1, math.ceil(virtual_count / 20))
        virtual_row = 0
        spool_paths = sorted(virtual_spool_directory.glob("virtual_pairs_*.bin"))
        for spool_path in spool_paths:
            spool_size = spool_path.stat().st_size
            if spool_size % _VIRTUAL_PAIR_DTYPE.itemsize:
                raise ValueError(f"temporary virtual-pair spool is truncated: {spool_path}")
            with spool_path.open("rb") as spool_file:
                while True:
                    spooled_records = np.fromfile(
                        spool_file, dtype=_VIRTUAL_PAIR_DTYPE, count=8192
                    )
                    if not len(spooled_records):
                        break
                    for record in spooled_records:
                        left_id = lexical_samples[int(record["left_pos"])]
                        right_id = lexical_samples[int(record["right_pos"])]
                        similarity = float(record["similarity"])
                        left_row, right_row = new_rows[left_id], new_rows[right_id]
                        virtual_endpoints[virtual_row] = (left_row, right_row)
                        virtual_similarities[virtual_row] = similarity
                        virtual_features[virtual_row] = (
                            node_features[left_row] + node_features[right_row]
                        ) / 2.0
                        if left_id in update_row and right_id in label_col:
                            adjacency_counts[update_row[left_id]] += 1
                            adjacency_db.execute(
                                "INSERT INTO adjacency VALUES (?, ?)",
                                (update_row[left_id], label_col[right_id]),
                            )
                        elif right_id in update_row and left_id in label_col:
                            adjacency_counts[update_row[right_id]] += 1
                            adjacency_db.execute(
                                "INSERT INTO adjacency VALUES (?, ?)",
                                (update_row[right_id], label_col[left_id]),
                            )
                        virtual_row += 1
                        now = monotonic()
                        if progress_callback is not None and (
                            virtual_row == 1
                            or virtual_row % virtual_write_step == 0
                            or now - last_virtual_write_progress >= 10.0
                        ):
                            progress_callback(
                                f"[pipeline] RSSI pair pass 'write virtual edges': "
                                f"{virtual_row:,}/{virtual_count:,} relations written "
                                f"({virtual_row / max(1, virtual_count):.0%}); "
                                f"elapsed={now - virtual_write_started:.1f}s"
                            )
                            last_virtual_write_progress = now
        if virtual_row != virtual_count:
            raise ValueError(
                f"temporary virtual-pair spool contains {virtual_row} records; "
                f"expected {virtual_count}"
            )
        if progress_callback is not None:
            progress_callback(
                f"[pipeline] RSSI pair pass 'write virtual edges' complete: "
                f"{virtual_row:,}/{virtual_count:,} relations; "
                f"elapsed={monotonic() - virtual_write_started:.1f}s"
            )
        shutil.rmtree(virtual_spool_directory)
        adjacency_indptr = _prefix(adjacency_counts)
        np.save(staging / "update_reference_indptr.npy", adjacency_indptr, allow_pickle=False)
        adjacency_cols = _open_array(
            staging / "update_reference_cols.npy", "int64", (int(adjacency_counts.sum()),)
        )
        adjacency_values = _open_array(
            staging / "update_reference_values.npy", "float64", (int(adjacency_counts.sum()),)
        )
        adjacency_cursor = adjacency_indptr[:-1].copy()
        adjacency_db.commit()
        for update_index, ref_col in adjacency_db.execute(
            "SELECT update_row, ref_col FROM adjacency ORDER BY update_row, ref_col"
        ):
            cursor = int(adjacency_cursor[update_index])
            adjacency_cols[cursor] = ref_col
            adjacency_values[cursor] = 1.0 / int(adjacency_counts[update_index])
            adjacency_cursor[update_index] += 1
        adjacency_db.close()
        (staging / "adjacency_sort.sqlite3").unlink(missing_ok=True)
        for array in (
            virtual_endpoints,
            virtual_similarities,
            virtual_features,
            negative_endpoints,
            negative_similarities,
            negative_strengths,
            adjacency_cols,
            adjacency_values,
        ):
            if isinstance(array, np.memmap):
                array.flush()

        array_names = (
            "node_features.npy",
            "sample_signal_features.npy",
            "sample_signal_masks.npy",
            "radio_endpoints.npy",
            "radio_weights.npy",
            "radio_edge_features.npy",
            "radio_edge_sources.npy",
            "virtual_endpoints.npy",
            "virtual_similarities.npy",
            "virtual_edge_features.npy",
            "negative_endpoints.npy",
            "negative_similarities.npy",
            "negative_strengths.npy",
            "update_reference_indptr.npy",
            "update_reference_cols.npy",
            "update_reference_values.npy",
        )
        arrays = tuple(_array_artifact(staging / name, name) for name in array_names)
        sample_initializations = tuple(input_manifest.sample_location_initializations) + tuple(
            SampleLocationInitialization(
                scan_id=scan_id,
                xy=update_pseudo_locations[scan_id],
                source="seeded_pseudo_location",
            )
            for scan_id in update_sample_ids
        )
        sample_position_available = tuple(
            bool(input_manifest.reference_label_mask[row])
            for row in range(len(reference_sample_ids))
        ) + (False,) * len(update_sample_ids)
        manifest_fields = dict(
            dataset_id=input_manifest.dataset_id,
            reference_batch_id=input_manifest.batch_id,
            update_batch_id=update_prepared.config.batch_id,
            reference_graph_id=input_manifest.graph_id,
            update_graph_id=update_graph_id,
            reference_lp_result_id=result.result_id,
            reference_gnn_checkpoint_sha256=input_manifest.gnn_checkpoint_sha256,
            encoder_checkpoint_sha256=input_manifest.encoder_checkpoint_sha256,
            reference_prepared_sha256=input_manifest.source_prepared_sha256,
            update_prepared_sha256=canonical_json_digest(update_prepared),
            update_encoder_artifact_sha256=canonical_json_digest(update_encoder_features),
            normalization=input_manifest.normalization,
            bssid_vocabulary=input_manifest.bssid_vocabulary,
            vocabulary_sha256=input_manifest.vocabulary_sha256,
            graph_build_config=input_manifest.graph_build_config,
            virtual_edge_sigma=virtual_edge_sigma,
            similarity_feature_space=RSSI_SIMILARITY_SPACE,
            encoder_feature_dim=update_encoder_features.feature_dim,
            node_feature_dim=feature_dim,
            node_ids=node_ids,
            sample_node_ids=sample_ids,
            reference_sample_ids=reference_sample_ids,
            update_sample_ids=update_sample_ids,
            ap_node_ids=ap_ids,
            node_types=("sample",) * len(sample_ids) + ("ap",) * len(ap_ids),
            node_feature_sources=tuple(node_sources),
            sample_location_initializations=sample_initializations,
            sample_position_available_mask=sample_position_available,
            labeled_reference_ids=labeled_ids,
            labeled_reference_coordinates=labeled_coordinates,
            radio_edge_count=radio_count,
            virtual_edge_count=virtual_count,
            negative_pair_count=len(negative_pairs),
            adjacency_shape=(len(update_sample_ids), len(labeled_ids)),
            adjacency_nonzero_count=int(adjacency_counts.sum()),
            arrays=arrays,
        )
        identity = canonical_json_digest(
            _jsonable({"format": ONLINE_GRAPH_FORMAT, **manifest_fields})
        )
        manifest = StreamingOnlineGraphManifest(
            online_graph_id=f"onlinegraph:{identity}", **manifest_fields
        )
        atomic_write_json(manifest, staging / "manifest.json")
        _complete(staging, manifest)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite streaming online graph: {target}")
        os.replace(staging, target)
        return manifest
    except Exception:
        # Keep incomplete staging artifacts for diagnosis. Without a
        # completion marker they cannot be consumed as a valid graph state.
        raise


def load_streaming_online_graph(
    directory: str | Path, *, verify_hashes: bool = True
) -> MappedStreamingOnlineGraph:
    root = Path(directory)
    manifest_path = root / "manifest.json"
    marker_path = root / "COMPLETED.json"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise ValueError(f"streaming online graph is incomplete: {root}")
    marker = json.loads(read_json_text(marker_path))
    manifest = StreamingOnlineGraphManifest.model_validate_json(
        read_json_text(manifest_path)
    )
    if marker.get("identity") != manifest.online_graph_id or marker.get(
        "manifest_sha256"
    ) != sha256_file(manifest_path):
        raise ValueError("streaming online graph completion marker is inconsistent")
    arrays: dict[str, np.ndarray] = {}
    for item in manifest.arrays:
        path = root / item.name
        if not path.is_file() or path.stat().st_size != item.size_bytes:
            raise ValueError(f"streaming online graph array is missing: {item.name}")
        if verify_hashes and sha256_file(path) != item.sha256:
            raise ValueError(f"streaming online graph array hash mismatch: {item.name}")
        array = np.load(
            path,
            mmap_mode="r" if math.prod(item.shape) else None,
            allow_pickle=False,
        )
        if tuple(array.shape) != item.shape or array.dtype.name != item.dtype:
            raise ValueError(f"streaming online graph array shape/dtype mismatch: {item.name}")
        arrays[item.name] = array
    return MappedStreamingOnlineGraph(
        directory=root,
        manifest=manifest,
        node_features=arrays["node_features.npy"],
        sample_signal_features=arrays["sample_signal_features.npy"],
        sample_signal_masks=arrays["sample_signal_masks.npy"],
        radio_endpoints=arrays["radio_endpoints.npy"],
        radio_weights=arrays["radio_weights.npy"],
        radio_edge_features=arrays["radio_edge_features.npy"],
        radio_edge_sources=arrays["radio_edge_sources.npy"],
        virtual_endpoints=arrays["virtual_endpoints.npy"],
        virtual_similarities=arrays["virtual_similarities.npy"],
        virtual_edge_features=arrays["virtual_edge_features.npy"],
        negative_endpoints=arrays["negative_endpoints.npy"],
        negative_similarities=arrays["negative_similarities.npy"],
        negative_strengths=arrays["negative_strengths.npy"],
        update_reference_indptr=arrays["update_reference_indptr.npy"],
        update_reference_cols=arrays["update_reference_cols.npy"],
        update_reference_values=arrays["update_reference_values.npy"],
    )


def save_streaming_online_lp_input(
    bootstrap_lp_input_directory: str | Path,
    online_graph_directory: str | Path,
    *,
    learned_gnn_state,
    output_directory: str | Path,
    reference_prepared: PreparedDataset | None = None,
) -> StreamingLPInputManifest:
    """Export a streamed online graph and its learned GNN state to LP.

    This is the explicit cell-7 handoff: the LP input contains final,
    ID-aligned online GNN node/physical-edge features, not the graph's initial
    random/pseudo feature rows and not a newly initialized model. Only numeric
    tables are mapped on disk. A cleaned reference-only graph additionally
    requires the updated ``reference_prepared`` snapshot and emits a new
    bootstrap-role LP input for the next batch.
    """
    bootstrap, _bootstrap_arrays = load_streaming_lp_input(
        bootstrap_lp_input_directory
    )
    graph = load_streaming_online_graph(online_graph_directory)
    if bootstrap.batch_role != "bootstrap_reference":
        raise ValueError("online LP export currently needs bootstrap reference metadata")
    if graph.manifest.reference_graph_id != bootstrap.graph_id:
        raise ValueError("online graph and bootstrap LP bundle have different graph identities")
    if graph.manifest.reference_sample_ids != bootstrap.sample_node_ids:
        raise ValueError("online graph reference sample IDs/order differ from bootstrap")
    reference_only = not graph.manifest.update_sample_ids
    if reference_only:
        if reference_prepared is None:
            raise ValueError("reference-only LP export requires the updated prepared dataset")
        if (
            tuple(scan.scan_id for scan in reference_prepared.scans)
            != graph.manifest.reference_sample_ids
            or reference_prepared.bssid_vocabulary != graph.manifest.bssid_vocabulary
            or reference_prepared.config.dataset_id != graph.manifest.dataset_id
            or reference_prepared.config.normalization_mode != graph.manifest.normalization
            or canonical_json_digest(reference_prepared)
            != graph.manifest.reference_prepared_sha256
        ):
            raise ValueError("updated prepared dataset differs from transitioned graph state")
    elif graph.manifest.bssid_vocabulary != bootstrap.bssid_vocabulary:
        raise ValueError("temporary online LP export cannot change BSSID vocabulary")
    state_manifest = getattr(learned_gnn_state, "manifest", None)
    node_embeddings = getattr(learned_gnn_state, "node_embeddings", None)
    radio_edge_features = getattr(learned_gnn_state, "radio_edge_features", None)
    gnn_checkpoint_sha256 = getattr(state_manifest, "checkpoint_sha256", None)
    if state_manifest is None or node_embeddings is None or radio_edge_features is None:
        raise ValueError("online LP export requires a loaded streaming GNN state")
    if (
        state_manifest.online_graph_id != graph.manifest.online_graph_id
        or state_manifest.node_ids != graph.manifest.node_ids
        or state_manifest.vocabulary_sha256 != graph.manifest.vocabulary_sha256
        or state_manifest.source_gnn_checkpoint_sha256
        != graph.manifest.reference_gnn_checkpoint_sha256
    ):
        raise ValueError("streaming GNN state does not match online graph/checkpoint lineage")
    if not isinstance(gnn_checkpoint_sha256, str) or len(gnn_checkpoint_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in gnn_checkpoint_sha256
    ):
        raise ValueError("selected streamed GNN checkpoint SHA-256 is malformed")
    if tuple(node_embeddings.shape) != (
        len(graph.manifest.node_ids),
        graph.manifest.node_feature_dim,
    ):
        raise ValueError("streaming GNN node feature array is not aligned to online graph")
    if tuple(radio_edge_features.shape) != (
        graph.manifest.radio_edge_count,
        graph.manifest.node_feature_dim,
    ):
        raise ValueError("streaming GNN radio vectors are not aligned to online graph edges")

    target = Path(output_directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite online LP input bundle: {target}")
    import tempfile

    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent)
    )
    try:
        node_ids = graph.manifest.node_ids
        sample_ids = graph.manifest.sample_node_ids
        ap_ids = graph.manifest.ap_node_ids
        node_row = {node_id: row for row, node_id in enumerate(node_ids)}
        n_nodes, n_samples = len(node_ids), len(sample_ids)
        n_radio, n_virtual = (
            graph.manifest.radio_edge_count,
            graph.manifest.virtual_edge_count,
        )

        node_features = _lp_open_npy(
            staging / "node_features.npy",
            np.dtype("float32"),
            tuple(node_embeddings.shape),
        )
        for start in range(0, n_nodes, 4096):
            node_features[start : start + 4096] = node_embeddings[start : start + 4096]
        _lp_finish_npy(node_features)
        signal_features = _lp_open_npy(
            staging / "sample_signal_features.npy",
            np.dtype("float64"),
            (n_samples, len(graph.manifest.bssid_vocabulary)),
        )
        signal_masks = _lp_open_npy(
            staging / "sample_signal_masks.npy",
            np.dtype("uint8"),
            (n_samples, len(graph.manifest.bssid_vocabulary)),
        )
        if tuple(graph.sample_signal_features.shape) != tuple(signal_features.shape):
            raise ValueError("online graph signal features do not align to sample IDs")
        if tuple(graph.sample_signal_masks.shape) != tuple(signal_masks.shape):
            raise ValueError("online graph RSSI masks do not align to sample IDs")
        for start in range(0, n_samples, 4096):
            signal_features[start : start + 4096] = graph.sample_signal_features[
                start : start + 4096
            ]
            signal_masks[start : start + 4096] = graph.sample_signal_masks[
                start : start + 4096
            ]
        _lp_finish_npy(signal_features)
        _lp_finish_npy(signal_masks)

        radio_degree = np.zeros((n_nodes,), dtype=np.int64)
        for left, right in graph.radio_endpoints:
            radio_degree[int(left)] += 1
            radio_degree[int(right)] += 1
        radio_indptr = _lp_prefix(radio_degree)
        np.save(staging / "radio_indptr.npy", radio_indptr, allow_pickle=False)
        radio_rows = _lp_open_npy(
            staging / "radio_neighbor_rows.npy", np.dtype("int64"), (2 * n_radio,)
        )
        radio_weights = _lp_open_npy(
            staging / "radio_neighbor_weights.npy", np.dtype("float64"), (2 * n_radio,)
        )
        radio_indices = _lp_open_npy(
            staging / "radio_edge_indices.npy", np.dtype("int64"), (2 * n_radio,)
        )
        cursor = radio_indptr[:-1].copy()
        for edge_index, ((left, right), weight) in enumerate(
            zip(graph.radio_endpoints, graph.radio_weights)
        ):
            left, right = int(left), int(right)
            for source, neighbor in ((left, right), (right, left)):
                pos = int(cursor[source])
                radio_rows[pos] = neighbor
                radio_weights[pos] = float(weight)
                radio_indices[pos] = edge_index
                cursor[source] += 1
        _lp_sort_csr_rows(radio_indptr, radio_rows, radio_weights, node_ids, radio_indices)
        for array in (radio_rows, radio_weights, radio_indices):
            _lp_finish_npy(array)

        virtual_degree = np.zeros((n_samples,), dtype=np.int64)
        for left, right in graph.virtual_endpoints:
            virtual_degree[int(left)] += 1
            virtual_degree[int(right)] += 1
        virtual_indptr = _lp_prefix(virtual_degree)
        np.save(staging / "virtual_indptr.npy", virtual_indptr, allow_pickle=False)
        virtual_rows = _lp_open_npy(
            staging / "virtual_neighbor_rows.npy", np.dtype("int64"), (2 * n_virtual,)
        )
        virtual_weights = _lp_open_npy(
            staging / "virtual_neighbor_weights.npy", np.dtype("float64"), (2 * n_virtual,)
        )
        cursor = virtual_indptr[:-1].copy()
        for (left, right), similarity in zip(
            graph.virtual_endpoints, graph.virtual_similarities
        ):
            left, right = int(left), int(right)
            for source, neighbor in ((left, right), (right, left)):
                pos = int(cursor[source])
                virtual_rows[pos] = neighbor
                virtual_weights[pos] = float(similarity)
                cursor[source] += 1
        _lp_sort_csr_rows(virtual_indptr, virtual_rows, virtual_weights, sample_ids)
        for array in (virtual_rows, virtual_weights):
            _lp_finish_npy(array)

        radio_feature_file = _lp_open_npy(
            staging / "radio_edge_features.npy",
            np.dtype("float32"),
            tuple(radio_edge_features.shape),
        )
        for start in range(0, n_radio, 4096):
            radio_feature_file[start : start + 4096] = radio_edge_features[
                start : start + 4096
            ]
        _lp_finish_npy(radio_feature_file)

        bootstrap_positions = {
            scan_id: (initialization, labeled, coordinate)
            for scan_id, initialization, labeled, coordinate in zip(
                bootstrap.sample_node_ids,
                bootstrap.sample_location_initializations,
                bootstrap.reference_label_mask[: len(bootstrap.sample_node_ids)],
                bootstrap.reference_coordinate_targets[: len(bootstrap.sample_node_ids)],
            )
        }
        initializations = []
        label_mask: list[bool] = []
        coordinate_targets: list[tuple[float, float] | None] = []
        reference_mask = []
        update_mask = []
        updated_label_map = (
            {
                item.scan_id: (float(item.x), float(item.y))
                for item in reference_prepared.reference_positions
            }
            if reference_only
            else None
        )
        reference_ids = set(graph.manifest.reference_sample_ids)
        update_ids = set(graph.manifest.update_sample_ids)
        online_initializations = {
            item.scan_id: item for item in graph.manifest.sample_location_initializations
        }
        for scan_id in sample_ids:
            is_reference = scan_id in reference_ids
            is_update = scan_id in update_ids
            if is_reference == is_update:
                raise ValueError(f"online sample role is ambiguous for {scan_id}")
            reference_mask.append(is_reference)
            update_mask.append(is_update)
            if is_reference:
                if reference_only:
                    coordinate = updated_label_map.get(scan_id)
                    original = bootstrap_positions.get(scan_id)
                    if original is None:
                        raise ValueError(f"reference location metadata is missing {scan_id}")
                    original_initialization = original[0]
                    xy = coordinate if coordinate is not None else original_initialization.xy
                    initializations.append(
                        SampleLocationInitialization(
                            scan_id=scan_id,
                            xy=xy,
                            source=(
                                "bootstrap_reference_label"
                                if coordinate is not None
                                else original_initialization.source
                            ),
                        )
                    )
                    label_mask.append(coordinate is not None)
                    coordinate_targets.append(coordinate)
                else:
                    try:
                        initialization, labeled, coordinate = bootstrap_positions[scan_id]
                    except KeyError as exc:
                        raise ValueError(f"current reference metadata is missing {scan_id}") from exc
                    initializations.append(initialization)
                    label_mask.append(bool(labeled))
                    coordinate_targets.append(coordinate)
            else:
                try:
                    initialization = online_initializations[scan_id]
                except KeyError as exc:
                    raise ValueError(f"online location metadata is missing {scan_id}") from exc
                initializations.append(initialization)
                label_mask.append(False)
                coordinate_targets.append(None)
        all_node_reference_mask = tuple(reference_mask + [False] * len(ap_ids))
        all_node_update_mask = tuple(update_mask + [False] * len(ap_ids))
        all_node_label_mask = tuple(label_mask + [False] * len(ap_ids))
        all_coordinate_targets = tuple(coordinate_targets + [None] * len(ap_ids))

        # Construct the strict LP model's graph ID from the streamed source
        # graph and selected checkpoint. It is a new immutable identity for
        # this exact learned online state, not the old bootstrap graph ID.
        graph_id = (
            graph.manifest.update_graph_id
            if reference_only
            else "graph:" + canonical_json_digest(
                {
                    "online_graph_id": graph.manifest.online_graph_id,
                    "gnn_checkpoint_sha256": gnn_checkpoint_sha256,
                    "node_ids": node_ids,
                }
            )
        )
        arrays = tuple(
            _lp_array_artifact(staging / name, name)
            for name in (
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
        )
        fields = dict(
            graph_id=graph_id,
            dataset_id=graph.manifest.dataset_id,
            batch_id=(
                graph.manifest.reference_batch_id
                if reference_only
                else graph.manifest.update_batch_id
            ),
            batch_role="bootstrap_reference" if reference_only else "update_batch",
            normalization=graph.manifest.normalization,
            source_prepared_sha256=(
                canonical_json_digest(reference_prepared)
                if reference_only
                else canonical_json_digest(
                    {
                        "reference": graph.manifest.reference_prepared_sha256,
                        "update": graph.manifest.update_prepared_sha256,
                    }
                )
            ),
            encoder_feature_artifact_sha256=(
                graph.manifest.update_encoder_artifact_sha256
                if reference_only
                else canonical_json_digest(
                    {
                        "reference": bootstrap.encoder_feature_artifact_sha256,
                        "update": graph.manifest.update_encoder_artifact_sha256,
                    }
                )
            ),
            encoder_checkpoint_sha256=graph.manifest.encoder_checkpoint_sha256,
            gnn_checkpoint_sha256=gnn_checkpoint_sha256,
            vocabulary_sha256=graph.manifest.vocabulary_sha256,
            bssid_vocabulary=graph.manifest.bssid_vocabulary,
            ap_ids=tuple(f"ap:{bssid}" for bssid in graph.manifest.bssid_vocabulary),
            node_ids=node_ids,
            node_types=graph.manifest.node_types,
            sample_node_ids=sample_ids,
            ap_node_ids=ap_ids,
            graph_build_config=graph.manifest.graph_build_config,
            encoder_feature_dim=graph.manifest.encoder_feature_dim,
            sample_location_initializations=tuple(initializations),
            reference_sample_mask=all_node_reference_mask,
            new_sample_mask=all_node_update_mask,
            reference_label_mask=all_node_label_mask,
            reference_coordinate_targets=all_coordinate_targets,
            omitted_unobserved_ap_ids=tuple(
                ap_id
                for ap_id in (f"ap:{bssid}" for bssid in graph.manifest.bssid_vocabulary)
                if ap_id not in set(ap_ids)
            ),
            node_feature_dim=graph.manifest.node_feature_dim,
            radio_edge_count=n_radio,
            virtual_edge_count=n_virtual,
            radio_weight_max=max(
                (float(value) for value in graph.radio_weights), default=0.0
            ),
            arrays=arrays,
        )
        provisional = StreamingLPInputManifest.model_construct(
            input_id="0" * 64, **fields
        )
        input_id = canonical_json_digest(
            provisional.model_dump(mode="json", exclude={"input_id"})
        )
        manifest = StreamingLPInputManifest(input_id=input_id, **fields)
        atomic_write_json(manifest, staging / "manifest.json")
        _lp_write_completion(staging, "manifest.json", input_id)
        _lp_publish_directory(staging, target)
        return manifest
    except Exception:
        # An incomplete staging directory has no completion marker and is
        # deliberately left for diagnosis; it cannot be consumed as an LP input.
        raise


def save_streaming_online_lp_input_with_new_aps(
    bootstrap_lp_input_directory: str | Path,
    online_graph_directory: str | Path,
    *,
    learned_gnn_state,
    full_update_prepared: PreparedDataset,
    output_directory: str | Path,
) -> StreamingLPInputManifest:
    """Export the learned online graph plus newly observed AP nodes to LP.

    Update scans are encoded and aggregated using the old vocabulary first.
    This function then extends that learned graph with each BSSID observed in
    the full incoming batch but absent from its active AP rows. New AP vectors
    use the Section 4.2 RSS+120 weighted mean of final online sample vectors;
    their temporary measured edges use endpoint means. IDs and weights are
    persisted directly into mapped LP arrays.

    The extended result is still a *transition-time* LP input. Its sample
    rows include temporary update scans; callers must apply deltas only to
    reference edges and must clean temporary rows before committing a later
    reference snapshot.
    """
    graph = load_streaming_online_graph(online_graph_directory)
    gm = graph.manifest
    if full_update_prepared.config.batch_role != "update_batch":
        raise ValueError("new-AP LP extension requires a full update_batch artifact")
    if full_update_prepared.config.dataset_id != gm.dataset_id:
        raise ValueError("new-AP LP extension dataset identity mismatch")
    if full_update_prepared.config.batch_id != gm.update_batch_id:
        raise ValueError("new-AP LP extension batch identity mismatch")
    if full_update_prepared.config.normalization_mode != gm.normalization:
        raise ValueError("new-AP LP extension normalization mismatch")
    if tuple(scan.scan_id for scan in full_update_prepared.scans) != gm.update_sample_ids:
        raise ValueError("full update scan IDs/order differ from online graph")

    target = Path(output_directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite online LP input bundle: {target}")
    with tempfile.TemporaryDirectory(
        prefix=f".{target.name}.base-", dir=target.parent
    ) as temporary:
        base_dir = Path(temporary) / "base"
        base_manifest = save_streaming_online_lp_input(
            bootstrap_lp_input_directory,
            online_graph_directory,
            learned_gnn_state=learned_gnn_state,
            output_directory=base_dir,
        )
        base_manifest, base = load_streaming_lp_input(base_dir)
        active = {node_id.removeprefix("ap:") for node_id in gm.ap_node_ids}
        observed_columns = {
            bssid
            for column, bssid in enumerate(full_update_prepared.bssid_vocabulary)
            if any(scan.observation_mask[column] for scan in full_update_prepared.scans)
        }
        new_bssids = tuple(sorted(observed_columns - active))
        new_ap_ids = tuple(f"ap:{bssid}" for bssid in new_bssids)
        old_ap_ids = base_manifest.ap_node_ids
        sample_ids = base_manifest.sample_node_ids
        old_node_ids = base_manifest.node_ids
        old_node_count = len(old_node_ids)
        sample_count = len(sample_ids)
        feature_dim = base_manifest.node_feature_dim
        old_radio_count = base_manifest.radio_edge_count
        update_row_by_id = {
            scan_id: row for row, scan_id in enumerate(sample_ids)
        }
        old_ap_set = set(old_ap_ids)
        if any(ap_id in old_ap_set for ap_id in new_ap_ids):
            raise AssertionError("new AP extension selected an already active identity")

        # Accumulate AP initialization only in the incoming-batch-sized table.
        ap_weight_sum = np.zeros((len(new_bssids),), dtype=np.float64)
        ap_feature_sum = np.zeros((len(new_bssids), feature_dim), dtype=np.float64)
        added_edge_count = 0
        raw_column = {
            bssid: column
            for column, bssid in enumerate(full_update_prepared.bssid_vocabulary)
        }
        ap_row_by_bssid = {
            bssid: old_node_count + row for row, bssid in enumerate(new_bssids)
        }
        for scan in full_update_prepared.scans:
            sample_row = update_row_by_id[scan.scan_id]
            for new_ap_index, bssid in enumerate(new_bssids):
                column = raw_column[bssid]
                if not scan.observation_mask[column]:
                    continue
                weight = rssi_dbm_to_graph_weight(scan.rss_dbm[column], True)
                if weight is None or weight < 0.0:
                    raise ValueError(
                        f"new AP {bssid} has invalid RSS+120 graph weight {weight}"
                    )
                ap_weight_sum[new_ap_index] += weight
                ap_feature_sum[new_ap_index] += weight * np.asarray(
                    base.node_features[sample_row], dtype=np.float64
                )
                added_edge_count += 1
        ap_features = np.zeros((len(new_bssids), feature_dim), dtype=np.float64)
        for row, bssid in enumerate(new_bssids):
            if ap_weight_sum[row] <= 0.0:
                raise ValueError(
                    f"new AP {bssid} has zero total RSS+120 initialization weight"
                )
            ap_features[row] = ap_feature_sum[row] / ap_weight_sum[row]

        base_vocabulary_set = set(base_manifest.bssid_vocabulary)
        target_vocabulary = base_manifest.bssid_vocabulary + tuple(
            bssid for bssid in new_bssids if bssid not in base_vocabulary_set
        )
        if len(set(target_vocabulary)) != len(target_vocabulary):
            raise ValueError("extended LP vocabulary contains duplicate BSSID identities")
        target_ap_ids = tuple(f"ap:{bssid}" for bssid in target_vocabulary)
        node_ids = old_node_ids + new_ap_ids
        ap_ids = old_ap_ids + new_ap_ids
        node_count = len(node_ids)
        radio_count = old_radio_count + added_edge_count

        def new_edge_rows():
            for scan in full_update_prepared.scans:
                sample_row = update_row_by_id[scan.scan_id]
                for new_ap_index, bssid in enumerate(new_bssids):
                    column = raw_column[bssid]
                    if not scan.observation_mask[column]:
                        continue
                    weight = rssi_dbm_to_graph_weight(scan.rss_dbm[column], True)
                    if weight is None or weight < 0.0:
                        raise ValueError(
                            f"new AP {bssid} has invalid RSS+120 graph weight {weight}"
                        )
                    yield sample_row, ap_row_by_bssid[bssid], float(weight), new_ap_index

        staging = _lp_create_staging_directory(target)
        try:
            node_features = _lp_open_npy(
                staging / "node_features.npy",
                np.dtype("float64"),
                (node_count, feature_dim),
            )
            node_features[:old_node_count] = base.node_features[:]
            if new_bssids:
                node_features[old_node_count:] = ap_features
            _lp_finish_npy(node_features)

            old_indptr = base.radio_indptr
            old_neighbors = base.radio_neighbor_rows
            old_weights = base.radio_neighbor_weights
            old_edge_indices = base.radio_edge_indices
            degree = np.zeros((node_count,), dtype=np.int64)
            degree[:old_node_count] = np.diff(old_indptr)
            for sample_row, ap_row, _weight, _ap_index in new_edge_rows():
                degree[sample_row] += 1
                degree[ap_row] += 1
            radio_indptr = _lp_prefix(degree)
            np.save(staging / "radio_indptr.npy", radio_indptr, allow_pickle=False)
            directed_count = int(radio_indptr[-1])
            neighbors = _lp_open_npy(
                staging / "radio_neighbor_rows.npy", np.dtype("int64"), (directed_count,)
            )
            weights = _lp_open_npy(
                staging / "radio_neighbor_weights.npy", np.dtype("float64"), (directed_count,)
            )
            edge_indices = _lp_open_npy(
                staging / "radio_edge_indices.npy", np.dtype("int64"), (directed_count,)
            )
            cursor = radio_indptr[:-1].copy()
            for row in range(old_node_count):
                start, stop = map(int, old_indptr[row : row + 2])
                count = stop - start
                offset = int(cursor[row])
                neighbors[offset : offset + count] = old_neighbors[start:stop]
                weights[offset : offset + count] = old_weights[start:stop]
                edge_indices[offset : offset + count] = old_edge_indices[start:stop]
                cursor[row] += count
            for edge_offset, (sample_row, ap_row, weight, _ap_index) in enumerate(new_edge_rows()):
                edge_index = old_radio_count + edge_offset
                for source, neighbor in ((sample_row, ap_row), (ap_row, sample_row)):
                    row_cursor = int(cursor[source])
                    neighbors[row_cursor] = neighbor
                    weights[row_cursor] = weight
                    edge_indices[row_cursor] = edge_index
                    cursor[source] += 1
            _lp_sort_csr_rows(radio_indptr, neighbors, weights, node_ids, edge_indices)
            for array in (neighbors, weights, edge_indices):
                _lp_finish_npy(array)

            radio_features = _lp_open_npy(
                staging / "radio_edge_features.npy",
                np.dtype("float64"),
                (radio_count, feature_dim),
            )
            radio_features[:old_radio_count] = base.radio_edge_features[:]
            for edge_offset, (sample_row, ap_row, _weight, _ap_index) in enumerate(new_edge_rows()):
                radio_features[old_radio_count + edge_offset] = (
                    np.asarray(node_features[sample_row], dtype=np.float64)
                    + np.asarray(node_features[ap_row], dtype=np.float64)
                ) / 2.0
            _lp_finish_npy(radio_features)

            # Preserve measured normalized RSSI in the expanded AP order. New
            # AP columns are zero/masked for old reference scans and populated
            # from the complete incoming batch for update scans.
            signal_features = _lp_open_npy(
                staging / "sample_signal_features.npy",
                np.dtype("float64"),
                (sample_count, len(target_vocabulary)),
            )
            signal_masks = _lp_open_npy(
                staging / "sample_signal_masks.npy",
                np.dtype("uint8"),
                (sample_count, len(target_vocabulary)),
            )
            old_signal_width = len(base_manifest.bssid_vocabulary)
            signal_features[:, :old_signal_width] = base.sample_signal_features
            signal_masks[:, :old_signal_width] = base.sample_signal_masks
            if len(target_vocabulary) > old_signal_width:
                signal_features[:, old_signal_width:] = 0.0
                signal_masks[:, old_signal_width:] = 0
            full_rows = full_update_prepared.ae_matrix()
            full_masks = full_update_prepared.mask_matrix()
            full_row_by_id = {
                scan.scan_id: row for row, scan in enumerate(full_update_prepared.scans)
            }
            target_col = {bssid: col for col, bssid in enumerate(target_vocabulary)}
            for scan_id in gm.update_sample_ids:
                source_row = full_row_by_id[scan_id]
                target_row = update_row_by_id[scan_id]
                for source_col, bssid in enumerate(full_update_prepared.bssid_vocabulary):
                    col = target_col[bssid]
                    observed = bool(full_masks[source_row][source_col])
                    signal_features[target_row, col] = (
                        float(full_rows[source_row][source_col]) if observed else 0.0
                    )
                    signal_masks[target_row, col] = 1 if observed else 0
            _lp_finish_npy(signal_features)
            _lp_finish_npy(signal_masks)
            for name, array in (
                ("virtual_indptr.npy", base.virtual_indptr),
                ("virtual_neighbor_rows.npy", base.virtual_neighbor_rows),
                ("virtual_neighbor_weights.npy", base.virtual_neighbor_weights),
            ):
                del array
                shutil.copyfile(base_dir / name, staging / name)

            arrays = tuple(
                _lp_array_artifact(staging / name, name)
                for name in (
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
            )
            max_weight = max(
                float(base_manifest.radio_weight_max),
                max((edge[2] for edge in new_edge_rows()), default=0.0),
            )
            graph_id = "graph:" + canonical_json_digest(
                {
                    "online_graph_id": gm.online_graph_id,
                    "gnn_checkpoint_sha256": learned_gnn_state.manifest.checkpoint_sha256,
                    "full_update_prepared_sha256": canonical_json_digest(full_update_prepared),
                    "vocabulary": target_vocabulary,
                    "node_ids": node_ids,
                }
            )
            reference_ids = set(gm.reference_sample_ids)
            update_ids = set(gm.update_sample_ids)
            metadata_by_scan = {
                scan_id: (init, label, coordinate)
                for scan_id, init, label, coordinate in zip(
                    base_manifest.sample_node_ids,
                    base_manifest.sample_location_initializations,
                    base_manifest.reference_label_mask[:sample_count],
                    base_manifest.reference_coordinate_targets[:sample_count],
                )
            }
            initializations = []
            label_mask = []
            coordinate_targets = []
            for scan_id in sample_ids:
                if scan_id not in reference_ids and scan_id not in update_ids:
                    raise ValueError(f"online sample has no role metadata: {scan_id}")
                try:
                    init, label, coordinate = metadata_by_scan[scan_id]
                except KeyError as exc:
                    raise ValueError(f"online sample location metadata is missing {scan_id}") from exc
                initializations.append(init)
                label_mask.append(bool(label))
                coordinate_targets.append(coordinate)
            fields = {
                name: getattr(base_manifest, name)
                for name in StreamingLPInputManifest.model_fields
                if name != "input_id"
            }
            fields.update(
                graph_id=graph_id,
                source_prepared_sha256=canonical_json_digest(
                    {
                        "online_graph": base_manifest.source_prepared_sha256,
                        "full_update": canonical_json_digest(full_update_prepared),
                    }
                ),
                encoder_feature_artifact_sha256=canonical_json_digest(
                    {
                        "old_vocabulary_update_features": base_manifest.encoder_feature_artifact_sha256,
                        "full_update_vocabulary": canonical_json_digest(full_update_prepared),
                    }
                ),
                vocabulary_sha256=canonical_json_digest(target_vocabulary),
                bssid_vocabulary=target_vocabulary,
                ap_ids=target_ap_ids,
                node_ids=node_ids,
                node_types=("sample",) * sample_count + ("ap",) * len(ap_ids),
                ap_node_ids=ap_ids,
                reference_sample_mask=base_manifest.reference_sample_mask + (False,) * len(new_ap_ids),
                new_sample_mask=base_manifest.new_sample_mask + (False,) * len(new_ap_ids),
                reference_label_mask=tuple(label_mask) + (False,) * len(ap_ids),
                reference_coordinate_targets=tuple(coordinate_targets) + (None,) * len(ap_ids),
                omitted_unobserved_ap_ids=tuple(
                    ap_id for ap_id in target_ap_ids if ap_id not in set(ap_ids)
                ),
                radio_edge_count=radio_count,
                radio_weight_max=max_weight,
                arrays=arrays,
            )
            provisional = StreamingLPInputManifest.model_construct(input_id="0" * 64, **fields)
            input_id = canonical_json_digest(
                provisional.model_dump(mode="json", exclude={"input_id"})
            )
            manifest = StreamingLPInputManifest(input_id=input_id, **fields)
            atomic_write_json(manifest, staging / "manifest.json")
            _lp_write_completion(staging, "manifest.json", input_id)
            _lp_publish_directory(staging, target)
            return manifest
        except Exception:
            raise


def iter_streaming_ap_transition_candidates(
    bootstrap_lp_input_directory: str | Path,
    bootstrap_lp_result_directory: str | Path,
    online_graph_directory: str | Path,
    *,
    full_update_prepared: PreparedDataset,
    extended_lp_manifest: StreamingLPInputManifest,
    sqlite_path: str | Path,
    progress_callback: Callable[[str], None] | None = None,
):
    """Yield the AP-transition candidate rule from an ordered disk index.

    Existing bootstrap reference edges are candidates for forgetting.
    Proposed new/reactivated-AP edges are limited to reference samples reached
    by an update sample's existing temporary virtual edge, and only when that
    update scan measured the AP. Candidate pairs are deduplicated and sorted
    by stable IDs in SQLite, so the scoring caller need not materialize them.
    """
    from lp_streaming import load_streaming_lp_input, load_streaming_lp_result

    graph = load_streaming_online_graph(online_graph_directory)
    gm = graph.manifest
    base, _ = load_streaming_lp_input(bootstrap_lp_input_directory)
    result = load_streaming_lp_result(
        bootstrap_lp_result_directory,
        input_directory=bootstrap_lp_input_directory,
    )
    if result.manifest.outcome == "failed":
        raise ValueError("AP transition cannot use a failed bootstrap LP result")
    if gm.reference_graph_id != base.graph_id:
        raise ValueError("AP transition reference graph differs from bootstrap LP input")
    if result.manifest.parent_graph_id != base.graph_id:
        raise ValueError("AP transition bootstrap LP result has different graph lineage")
    if tuple(scan.scan_id for scan in full_update_prepared.scans) != gm.update_sample_ids:
        raise ValueError("AP candidate source scan IDs differ from online graph")
    if extended_lp_manifest.sample_node_ids != gm.sample_node_ids:
        raise ValueError("extended LP sample ordering differs from online graph")
    if extended_lp_manifest.node_ids[: len(gm.sample_node_ids)] != gm.sample_node_ids:
        raise ValueError("extended LP node rows do not preserve online sample rows")

    destination = Path(sqlite_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite AP candidate index: {destination}")
    connection = sqlite3.connect(destination)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE candidates (sample_id TEXT NOT NULL, ap_id TEXT NOT NULL, "
            "PRIMARY KEY(sample_id, ap_id)) WITHOUT ROWID"
        )
        reference_set = set(gm.reference_sample_ids)
        base_endpoints = np.load(
            Path(bootstrap_lp_result_directory) / "updated_edge_endpoints.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        if progress_callback is not None:
            progress_callback(
                "[link prediction] indexing AP-transition candidates from "
                f"{len(base_endpoints):,} reference edges and "
                f"{len(graph.virtual_endpoints):,} temporary virtual edges"
            )
        insert_buffer: list[tuple[str, str]] = []
        endpoint_count = len(base_endpoints)
        endpoint_progress_step = max(1, math.ceil(endpoint_count / 10))
        for endpoint_index, (sample_row, ap_row) in enumerate(base_endpoints, start=1):
            sample_id = base.node_ids[int(sample_row)]
            ap_id = base.node_ids[int(ap_row)]
            if sample_id in reference_set and ap_id.startswith("ap:"):
                insert_buffer.append((sample_id, ap_id))
                if len(insert_buffer) >= 8192:
                    connection.executemany(
                        "INSERT OR IGNORE INTO candidates VALUES (?, ?)", insert_buffer
                    )
                    insert_buffer.clear()
            if progress_callback is not None and (
                endpoint_index == 1
                or endpoint_index % endpoint_progress_step == 0
                or endpoint_index == endpoint_count
            ):
                progress_callback(
                    f"[link prediction] AP candidate index: reference edges "
                    f"{endpoint_index:,}/{endpoint_count:,}"
                )
        if insert_buffer:
            connection.executemany(
                "INSERT OR IGNORE INTO candidates VALUES (?, ?)", insert_buffer
            )
        del insert_buffer

        active_online_ap_ids = set(gm.ap_node_ids)
        new_ap_bssids = tuple(
            ap_id.removeprefix("ap:")
            for ap_id in extended_lp_manifest.ap_node_ids
            if ap_id not in active_online_ap_ids
        )
        if not set(f"ap:{bssid}" for bssid in new_ap_bssids) <= set(
            extended_lp_manifest.ap_node_ids
        ):
            raise AssertionError("new AP candidate identity is absent from extended LP graph")
        raw_columns = {
            bssid: column
            for column, bssid in enumerate(full_update_prepared.bssid_vocabulary)
        }
        update_scan_by_id = {
            scan.scan_id: scan for scan in full_update_prepared.scans
        }
        update_ids = set(gm.update_sample_ids)
        sample_row_to_id = gm.sample_node_ids
        new_candidate_buffer: list[tuple[str, str]] = []
        virtual_count = len(graph.virtual_endpoints)
        virtual_progress_step = max(1, math.ceil(virtual_count / 10))
        for virtual_index, (left, right) in enumerate(graph.virtual_endpoints, start=1):
            left_id = sample_row_to_id[int(left)]
            right_id = sample_row_to_id[int(right)]
            if left_id in update_ids and right_id in reference_set:
                update_id, reference_id = left_id, right_id
            elif right_id in update_ids and left_id in reference_set:
                update_id, reference_id = right_id, left_id
            else:
                continue
            scan = update_scan_by_id[update_id]
            for bssid in new_ap_bssids:
                column = raw_columns.get(bssid)
                if column is not None and scan.observation_mask[column]:
                    new_candidate_buffer.append((reference_id, f"ap:{bssid}"))
                    if len(new_candidate_buffer) >= 8192:
                        connection.executemany(
                            "INSERT OR IGNORE INTO candidates VALUES (?, ?)",
                            new_candidate_buffer,
                        )
                        new_candidate_buffer.clear()
            if progress_callback is not None and (
                virtual_index == 1
                or virtual_index % virtual_progress_step == 0
                or virtual_index == virtual_count
            ):
                progress_callback(
                    f"[link prediction] AP candidate index: temporary virtual edges "
                    f"{virtual_index:,}/{virtual_count:,}"
                )
        if new_candidate_buffer:
            connection.executemany(
                "INSERT OR IGNORE INTO candidates VALUES (?, ?)",
                new_candidate_buffer,
            )
        connection.commit()
        candidate_total = int(
            connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
        )
        if progress_callback is not None:
            progress_callback(
                f"[link prediction] AP candidate index ready: "
                f"{candidate_total:,} unique sample/AP pairs"
            )
        cursor = connection.execute(
            "SELECT sample_id, ap_id FROM candidates ORDER BY sample_id, ap_id"
        )
        for sample_id, ap_id in cursor:
            yield sample_id, ap_id
    finally:
        connection.close()


def infer_streaming_ap_graph_changes(
    bootstrap_lp_input_directory: str | Path,
    bootstrap_lp_result_directory: str | Path,
    online_graph_directory: str | Path,
    *,
    learned_gnn_state,
    full_update_prepared: PreparedDataset,
    input_directory: str | Path,
    candidate_index_path: str | Path,
    result_directory: str | Path,
    config: LinkPredictionConfig,
    progress_callback: Callable[[str], None] | None = None,
) -> StreamingAPInferenceState:
    """Run AP-transition Eq. (11)--(13) over streamed learned online state.

    This composes old-vocabulary GNN output, Section 4.2 new-AP initialization,
    the AP-specific candidate rule, and streamed G/F scoring. It never invokes
    the generic all-sample/all-AP candidate enumeration and never creates a
    Python candidate/delta collection.
    """
    input_path = Path(input_directory)
    candidate_path = Path(candidate_index_path)
    result_path = Path(result_directory)
    if progress_callback is not None:
        progress_callback(
            "[link prediction] preparing AP-transition LP input from the updated graph"
        )
    manifest = save_streaming_online_lp_input_with_new_aps(
        bootstrap_lp_input_directory,
        online_graph_directory,
        learned_gnn_state=learned_gnn_state,
        full_update_prepared=full_update_prepared,
        output_directory=input_path,
    )
    if progress_callback is not None:
        progress_callback(
            f"[link prediction] AP-transition input ready: "
            f"samples={len(manifest.sample_node_ids):,}, "
            f"APs={len(manifest.ap_node_ids):,}, nodes={len(manifest.node_ids):,}, "
            f"radio edges={manifest.radio_edge_count:,}"
        )
    candidates = iter_streaming_ap_transition_candidates(
        bootstrap_lp_input_directory,
        bootstrap_lp_result_directory,
        online_graph_directory,
        full_update_prepared=full_update_prepared,
        extended_lp_manifest=manifest,
        sqlite_path=candidate_path,
        progress_callback=progress_callback,
    )
    result = predict_edges_streaming(
        input_path,
        result_path,
        config,
        candidate_pairs=candidates,
        progress_callback=progress_callback,
    )
    if result.manifest.candidate_policy != "explicit_sorted_stable_ids_v1":
        raise AssertionError("AP transition LP result lost its explicit candidate policy")
    if result.manifest.outcome == "failed":
        raise RuntimeError(
            "streamed AP transition did not converge; no edge delta should be applied"
        )
    return StreamingAPInferenceState(input_path, candidate_path, result)


def build_streaming_reference_graph_after_transition(
    bootstrap_lp_input_directory: str | Path,
    transition_state,
    post_transition_ae_state,
    full_update_prepared: PreparedDataset,
    source_gnn_state,
    output_directory: str | Path,
    *,
    reference_prepared: PreparedDataset | None = None,
    device: str = "cpu",
    chunk_size: int = 256,
) -> StreamingOnlineGraphManifest:
    """Rebuild the persistent reference graph from transitioned RSS arrays.

    The AE is run in chunks over every retained reference ID. Its ``z_encoder``
    rows are joined to the source graph's stable sample IDs, with immutable
    sample XY initialization carried by ID. AP graph rows are reinitialized by
    RSS+120 weighted means over the final radio edge set. Radio edge vectors
    start at endpoint means, and virtual edges use the recorded cosine rule.
    Temporary update nodes/edges are absent from this persisted reference
    graph.
    """
    tm = transition_state.manifest
    pm = post_transition_ae_state.manifest
    if pm.transition_id != tm.transition_id or pm.bssid_vocabulary != tm.vocabulary_after:
        raise ValueError("post-transition AE state differs from transitioned vocabulary/state")
    if pm.reference_scan_ids != tm.reference_scan_ids:
        raise ValueError("post-transition AE changed retained reference IDs/order")
    if full_update_prepared.config.dataset_id != tm.dataset_id or full_update_prepared.config.batch_id != tm.update_batch_id:
        raise ValueError("reference graph update provenance differs from AP transition")
    if full_update_prepared.config.batch_role != "update_batch":
        raise ValueError("post-transition graph requires the original update_batch artifact")
    if reference_prepared is not None:
        if reference_prepared.config.batch_role != "bootstrap_reference":
            raise ValueError("post-transition graph needs a reference-role prepared dataset")
        if reference_prepared.config.dataset_id != tm.dataset_id:
            raise ValueError("post-transition reference dataset identity differs")
        if tuple(scan.scan_id for scan in reference_prepared.scans) != tm.reference_scan_ids:
            raise ValueError("post-transition prepared rows differ from retained reference IDs")
        if reference_prepared.bssid_vocabulary != tm.vocabulary_after:
            raise ValueError("post-transition prepared vocabulary differs from target vocabulary")
        if reference_prepared.config.normalization_mode != tm.normalization:
            raise ValueError("post-transition prepared normalization differs")
    if chunk_size < 1:
        raise ValueError("post-transition graph chunk size must be positive")
    base, _ = load_streaming_lp_input(bootstrap_lp_input_directory)
    if base.sample_node_ids != tm.reference_scan_ids:
        raise ValueError("reference graph base metadata has different scan IDs/order")
    if base.bssid_vocabulary != tm.vocabulary_before:
        raise ValueError("reference graph base metadata uses a different source vocabulary")
    if base.graph_build_config.virtual_edge_strategy != "paper_cosine_threshold":
        raise ValueError("post-transition streaming graph requires the paper cosine rule")
    if source_gnn_state.manifest.node_ids[: len(tm.reference_scan_ids)] != tm.reference_scan_ids:
        raise ValueError("selected source GNN state does not preserve reference sample rows")

    target = Path(output_directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite post-transition reference graph: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    edge_db_path = transition_state.directory / tm.reference_edges.name
    connection = sqlite3.connect(f"file:{edge_db_path}?mode=ro", uri=True)
    try:
        sample_ids = tm.reference_scan_ids
        sample_count = len(sample_ids)
        latent_dim = post_transition_ae_state.model.config.latent_dim
        feature_dim = latent_dim + 2
        if source_gnn_state.manifest.model_config_record.graph_feature_dim != feature_dim:
            raise ValueError("post-transition AE latent width changed the GNN graph width")
        if source_gnn_state.manifest.reference_graph_id != base.graph_id:
            raise ValueError("post-transition GNN weights descend from a different reference graph")
        ap_degree = {
            ap_id: int(count)
            for ap_id, count in connection.execute(
                "SELECT ap_id, COUNT(*) FROM edges GROUP BY ap_id"
            )
        }
        active_bssids = tuple(
            bssid for bssid in tm.vocabulary_after
            if ap_degree.get(f"ap:{bssid}", 0) > 0
        )
        if active_bssids != tm.vocabulary_after:
            raise ValueError("transition vocabulary contains APs with no persistent reference edges")
        ap_ids = tuple(f"ap:{bssid}" for bssid in active_bssids)
        node_ids = sample_ids + ap_ids
        node_row = {node_id: row for row, node_id in enumerate(node_ids)}
        sample_row = {node_id: row for row, node_id in enumerate(sample_ids)}
        if len(node_row) != len(node_ids):
            raise ValueError("post-transition reference node IDs are not unique")

        init_by_id = {
            item.scan_id: item
            for item in base.sample_location_initializations
        }
        if set(init_by_id) != set(sample_ids):
            raise ValueError("bootstrap graph lacks sample location initialization provenance")
        resolved_build_config, spatial_transform = resolve_spatial_transform(
            base.graph_build_config, base.dataset_id
        )
        if resolved_build_config != base.graph_build_config:
            raise ValueError("post-transition graph spatial configuration is unresolved")
        position_available_by_id = {
            scan_id: bool(base.reference_label_mask[row])
            for row, scan_id in enumerate(base.sample_node_ids)
        }
        node_features = _open_array(
            staging / "node_features.npy", "float64", (len(node_ids), feature_dim)
        )
        sample_signal_features = _open_array(
            staging / "sample_signal_features.npy",
            "float64",
            (sample_count, len(tm.vocabulary_after)),
        )
        sample_signal_masks = _open_array(
            staging / "sample_signal_masks.npy",
            "uint8",
            (sample_count, len(tm.vocabulary_after)),
        )
        model = post_transition_ae_state.model.to(device=device, dtype=torch.float32).eval()
        with torch.inference_mode():
            for start in range(0, sample_count, chunk_size):
                stop = min(start + chunk_size, sample_count)
                rows = np.arange(start, stop, dtype=np.int64)
                raw = np.asarray(post_transition_ae_state.updated_rssi_dbm[rows], dtype=np.float64)
                masks = np.asarray(post_transition_ae_state.observation_mask[rows], dtype=np.uint8)
                origins = np.asarray(post_transition_ae_state.value_provenance[rows], dtype=np.uint8)
                inputs = np.empty((len(rows), 1, len(tm.vocabulary_after)), dtype=np.float32)
                for offset in range(len(rows)):
                    usable = (masks[offset] != 0) | (origins[offset] == 1) | (origins[offset] == 2)
                    inputs[offset, 0, :] = raw_dbm_row_to_ae(
                        raw[offset], tuple(bool(value) for value in usable), tm.normalization
                    )
                latent = model.E(torch.tensor(inputs, dtype=torch.float32, device=device))[:, 0, :].cpu().numpy()
                for offset, row in enumerate(rows):
                    scan_id = sample_ids[int(row)]
                    measured_mask = tuple(bool(value) for value in masks[offset])
                    normalized = raw_dbm_row_to_ae(
                        raw[offset], measured_mask, tm.normalization
                    )
                    sample_signal_features[int(row)] = tuple(
                        float(value) if observed else 0.0
                        for value, observed in zip(normalized, measured_mask)
                    )
                    sample_signal_masks[int(row)] = measured_mask
                    node_features[int(row), :latent_dim] = latent[offset]
                    if spatial_transform is None:
                        position_feature = init_by_id[scan_id].xy
                    else:
                        position_raw = init_by_id[scan_id].xy
                        position_feature = model_position_feature(
                            spatial_transform,
                            position_raw,
                            available=position_available_by_id[scan_id],
                            position_weight=base.graph_build_config.position_weight,
                        )
                        position_feature = tuple(float(value) for value in position_feature)
                    node_features[int(row), latent_dim:] = position_feature

        ap_feature_sums = np.zeros((len(ap_ids), feature_dim), dtype=np.float64)
        ap_weight_sums = np.zeros((len(ap_ids),), dtype=np.float64)
        ap_row = {ap_id: row for row, ap_id in enumerate(ap_ids)}
        radio_count = int(connection.execute("SELECT COUNT(*) FROM edges").fetchone()[0])
        for sample_id, ap_id, weight in connection.execute(
            "SELECT sample_id, ap_id, weight FROM edges ORDER BY sample_id, ap_id"
        ):
            row = ap_row[ap_id]
            weight = float(weight)
            if weight < 0.0 or not math.isfinite(weight):
                raise ValueError("persistent RSS+120 edge weight is invalid")
            ap_feature_sums[row] += weight * np.asarray(node_features[sample_row[sample_id]])
            ap_weight_sums[row] += weight
        for row, ap_id in enumerate(ap_ids):
            if ap_weight_sums[row] <= 0.0:
                raise ValueError(f"post-transition AP has no positive radio weight: {ap_id}")
            node_features[sample_count + row] = ap_feature_sums[row] / ap_weight_sums[row]
        if isinstance(node_features, np.memmap):
            node_features.flush()
        if isinstance(sample_signal_features, np.memmap):
            sample_signal_features.flush()
        if isinstance(sample_signal_masks, np.memmap):
            sample_signal_masks.flush()

        radio_endpoints = _open_array(staging / "radio_endpoints.npy", "int64", (radio_count, 2))
        radio_weights = _open_array(staging / "radio_weights.npy", "float64", (radio_count,))
        radio_edge_features = _open_array(
            staging / "radio_edge_features.npy", "float64", (radio_count, feature_dim)
        )
        radio_edge_sources = _open_array(staging / "radio_edge_sources.npy", "uint8", (radio_count,))
        for index, (sample_id, ap_id, weight) in enumerate(
            connection.execute("SELECT sample_id, ap_id, weight FROM edges ORDER BY sample_id, ap_id")
        ):
            left, right = sample_row[sample_id], node_row[ap_id]
            radio_endpoints[index] = (left, right)
            radio_weights[index] = float(weight)
            radio_edge_features[index] = (
                np.asarray(node_features[left]) + np.asarray(node_features[right])
            ) / 2.0
            radio_edge_sources[index] = 1
        for array in (radio_endpoints, radio_weights, radio_edge_features, radio_edge_sources):
            if isinstance(array, np.memmap):
                array.flush()

        virtual_db = sqlite3.connect(staging / "virtual_sort.sqlite3")
        virtual_db.execute("PRAGMA journal_mode=OFF")
        virtual_db.execute("PRAGMA synchronous=OFF")
        virtual_db.execute(
            "CREATE TABLE virtual_edges(left_id TEXT, right_id TEXT, similarity REAL, "
            "PRIMARY KEY(left_id,right_id)) WITHOUT ROWID"
        )
        lexical_rows = tuple(sorted(range(sample_count), key=sample_ids.__getitem__))
        lexical_ids = tuple(sample_ids[row] for row in lexical_rows)
        pair_sampler = BoundedNegativeSampler(
            lexical_ids,
            negative_similarity_threshold=base.graph_build_config.negative_similarity_threshold,
            max_pairs_per_sample=base.graph_build_config.max_negative_pairs_per_sample,
        )
        pending: list[tuple[str, str, float]] = []
        for left_pos, left_row in enumerate(lexical_rows):
            left_id = sample_ids[left_row]
            left = sample_signal_features[left_row]
            for right_pos, right_row in enumerate(lexical_rows[left_pos + 1 :], start=left_pos + 1):
                right_id = sample_ids[right_row]
                similarity = normalized_rssi_cosine(
                    left,
                    sample_signal_features[right_row],
                    sample_signal_masks[left_row],
                    sample_signal_masks[right_row],
                    minimum_shared_aps=base.graph_build_config.minimum_shared_aps,
                )
                if similarity is not None and similarity > base.graph_build_config.virtual_edge_sigma:
                    pending.append((left_id, right_id, similarity))
                    if len(pending) >= 8192:
                        virtual_db.executemany("INSERT INTO virtual_edges VALUES (?, ?, ?)", pending)
                        pending.clear()
                elif similarity is not None and similarity <= base.graph_build_config.negative_similarity_threshold:
                    pair_sampler.add(left_pos, right_pos, similarity)
        if pending:
            virtual_db.executemany("INSERT INTO virtual_edges VALUES (?, ?, ?)", pending)
        pending.clear()
        virtual_db.commit()
        virtual_count = int(virtual_db.execute("SELECT COUNT(*) FROM virtual_edges").fetchone()[0])
        negative_pairs = pair_sampler.selected()
        virtual_endpoints = _open_array(
            staging / "virtual_endpoints.npy", "int64", (virtual_count, 2)
        )
        virtual_similarities = _open_array(
            staging / "virtual_similarities.npy", "float64", (virtual_count,)
        )
        virtual_edge_features = _open_array(
            staging / "virtual_edge_features.npy", "float64", (virtual_count, feature_dim)
        )
        negative_endpoints = _open_array(
            staging / "negative_endpoints.npy", "int64", (len(negative_pairs), 2)
        )
        negative_similarities = _open_array(
            staging / "negative_similarities.npy", "float64", (len(negative_pairs),)
        )
        negative_strengths = _open_array(
            staging / "negative_strengths.npy", "float64", (len(negative_pairs),)
        )
        for index, pair in enumerate(negative_pairs):
            negative_endpoints[index] = (
                sample_row[pair.left_sample_id], sample_row[pair.right_sample_id]
            )
            negative_similarities[index] = pair.signal_cosine_similarity
            negative_strengths[index] = pair.negative_strength
        for index, (left_id, right_id, similarity) in enumerate(
            virtual_db.execute("SELECT left_id, right_id, similarity FROM virtual_edges ORDER BY left_id,right_id")
        ):
            left, right = sample_row[left_id], sample_row[right_id]
            virtual_endpoints[index] = (left, right)
            virtual_similarities[index] = float(similarity)
            virtual_edge_features[index] = (
                np.asarray(node_features[left]) + np.asarray(node_features[right])
            ) / 2.0
        virtual_db.close()
        (staging / "virtual_sort.sqlite3").unlink(missing_ok=True)
        for array in (
            virtual_endpoints,
            virtual_similarities,
            virtual_edge_features,
            negative_endpoints,
            negative_similarities,
            negative_strengths,
        ):
            if isinstance(array, np.memmap):
                array.flush()

        empty_indptr = np.zeros((1,), dtype=np.int64)
        np.save(staging / "update_reference_indptr.npy", empty_indptr, allow_pickle=False)
        np.save(staging / "update_reference_cols.npy", np.empty((0,), dtype=np.int64), allow_pickle=False)
        np.save(staging / "update_reference_values.npy", np.empty((0,), dtype=np.float64), allow_pickle=False)
        names = (
            "node_features.npy",
            "sample_signal_features.npy",
            "sample_signal_masks.npy",
            "radio_endpoints.npy",
            "radio_weights.npy",
            "radio_edge_features.npy",
            "radio_edge_sources.npy",
            "virtual_endpoints.npy",
            "virtual_similarities.npy",
            "virtual_edge_features.npy",
            "negative_endpoints.npy",
            "negative_similarities.npy",
            "negative_strengths.npy",
            "update_reference_indptr.npy",
            "update_reference_cols.npy",
            "update_reference_values.npy",
        )
        arrays = tuple(_array_artifact(staging / name, name) for name in names)
        labels = tuple(
            (scan_id, coordinate)
            for scan_id, coordinate in zip(sample_ids, tm.original_coordinates)
            if coordinate is not None
        )
        label_ids = tuple(item[0] for item in labels)
        label_xy = tuple(item[1] for item in labels)
        if label_ids != pm.labeled_reference_ids:
            raise ValueError("post-transition reference labels differ from Eq. (9) targets")
        reference_state_sha = (
            canonical_json_digest(reference_prepared)
            if reference_prepared is not None
            else canonical_json_digest(
                {
                    "transition_id": tm.transition_id,
                    "post_transition_ae_id": pm.post_update_id,
                    "reference_edges_sha256": tm.reference_edges.sha256,
                }
            )
        )
        update_graph_id = "graph:" + canonical_json_digest(
            {
                "dataset": tm.dataset_id,
                "update_batch": tm.update_batch_id,
                "transition": tm.transition_id,
                "prepared_reference_sha256": reference_state_sha,
                "vocabulary": tm.vocabulary_after,
                "graph_build_config": base.graph_build_config.model_dump(mode="json"),
            }
        )
        fields = dict(
            dataset_id=tm.dataset_id,
            reference_batch_id=tm.update_batch_id,
            update_batch_id=tm.update_batch_id,
            reference_graph_id=source_gnn_state.manifest.reference_graph_id,
            update_graph_id=update_graph_id,
            reference_lp_result_id=tm.ap_lp_result_id,
            reference_gnn_checkpoint_sha256=source_gnn_state.manifest.checkpoint_sha256,
            encoder_checkpoint_sha256=pm.checkpoint_sha256,
            reference_prepared_sha256=reference_state_sha,
            update_prepared_sha256=canonical_json_digest(full_update_prepared),
            update_encoder_artifact_sha256=canonical_json_digest(
                {"post_transition_ae_id": pm.post_update_id, "feature_space": "z_encoder"}
            ),
            normalization=tm.normalization,
            bssid_vocabulary=tm.vocabulary_after,
            vocabulary_sha256=canonical_json_digest(tm.vocabulary_after),
            graph_build_config=base.graph_build_config,
            virtual_edge_sigma=base.graph_build_config.virtual_edge_sigma,
            similarity_feature_space=RSSI_SIMILARITY_SPACE,
            encoder_feature_dim=latent_dim,
            node_feature_dim=feature_dim,
            node_ids=node_ids,
            sample_node_ids=sample_ids,
            reference_sample_ids=sample_ids,
            update_sample_ids=(),
            ap_node_ids=ap_ids,
            node_types=("sample",) * sample_count + ("ap",) * len(ap_ids),
            node_feature_sources=("post_transition_ae_reencoded_reference_sample",) * sample_count
            + ("post_transition_reference_ap_weighted_mean",) * len(ap_ids),
            sample_location_initializations=tuple(init_by_id[scan_id] for scan_id in sample_ids),
            sample_position_available_mask=tuple(
                position_available_by_id[scan_id] for scan_id in sample_ids
            ),
            labeled_reference_ids=label_ids,
            labeled_reference_coordinates=label_xy,
            radio_edge_count=radio_count,
            virtual_edge_count=virtual_count,
            negative_pair_count=len(negative_pairs),
            adjacency_shape=(0, len(label_ids)),
            adjacency_nonzero_count=0,
            arrays=arrays,
        )
        identity = canonical_json_digest(_jsonable({"format": ONLINE_GRAPH_FORMAT, **fields}))
        manifest = StreamingOnlineGraphManifest(
            online_graph_id=f"onlinegraph:{identity}", **fields
        )
        atomic_write_json(manifest, staging / "manifest.json")
        _complete(staging, manifest)
        if target.exists():
            raise FileExistsError(f"refusing to overwrite post-transition reference graph: {target}")
        os.replace(staging, target)
        return manifest
    except Exception:
        # Incomplete staging is deliberately left without a completion marker.
        raise
    finally:
        connection.close()


__all__ = [
    "MappedStreamingOnlineGraph",
    "StreamingAPInferenceState",
    "StreamingOnlineGraphManifest",
    "build_streaming_online_graph",
    "build_streaming_reference_graph_after_transition",
    "load_streaming_online_graph",
    "iter_streaming_ap_transition_candidates",
    "infer_streaming_ap_graph_changes",
    "save_streaming_online_lp_input",
    "save_streaming_online_lp_input_with_new_aps",
]
