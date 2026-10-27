"""Chunked, disk-backed streaming updater and AE fine-tuning.

The online GNN feature table and sparse cross-virtual adjacency remain mapped
on disk. The updater's Frobenius norms are global over each equation, and its
parameter gradients are accumulated over chunks with the exact derivative of
those norms. Streaming Eq. (10) AE fine-tuning and AP-vocabulary migration
are also implemented here as explicit post-update stages.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from ae import (
    AE,
    AECheckpointManifest,
    AECheckpointTrainingConfig,
    AEModelConfig,
    CHECKPOINT_FORMAT,
    DatasetIdentity,
    EncoderFeatureArtifact,
    load_ae_checkpoint,
)
from fingerprint_update import (
    FingerprintUpdateConfig,
    LocationPredictor,
    LocationToFeatureUpdater,
    UpdateEpochLoss,
    UpdateLocationScaler,
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
    PreparedDataset,
    ae_input_observation_mask,
    ae_value_to_raw_dbm,
    raw_dbm_row_to_ae,
)
from lp_streaming import StreamArrayArtifact, _array_artifact
from streaming_gnn import StreamingGNNState
from streaming_update import MappedStreamingOnlineGraph, load_streaming_online_graph
from training_progress import (
    DEFAULT_LOG_EVERY_K_EPOCHS,
    log_training_epoch,
    validate_log_interval,
)


STREAM_UPDATER_FORMAT = "gufu_streaming_updater_v1"
STREAM_AE_UPDATE_FORMAT = "gufu_streaming_ae_update_v1"
STREAM_POST_TRANSITION_AE_FORMAT = "gufu_streaming_post_transition_ae_v2"


class StreamingUpdaterManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_streaming_updater_v1"] = STREAM_UPDATER_FORMAT
    updater_id: str = Field(pattern=r"^streamupdater:[0-9a-f]{64}$")
    online_graph_id: str = Field(pattern=r"^onlinegraph:[0-9a-f]{64}$")
    gnn_artifact_id: str = Field(pattern=r"^streamgnn:[0-9a-f]{64}$")
    dataset_id: str
    reference_batch_id: str
    update_batch_id: str
    reference_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    update_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    update_encoder_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ae_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_gnn_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_updater_id: str | None = Field(
        default=None, pattern=r"^streamupdater:[0-9a-f]{64}$"
    )
    initialization_policy: Literal["fresh", "previous_batch_weights_only"] = "fresh"
    resume_policy: Literal["fresh_adam_each_batch"] = "fresh_adam_each_batch"
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bssid_vocabulary: tuple[str, ...]
    normalization: Literal["paper_ae", "legacy_raw_div_120"]
    reference_ids: tuple[str, ...]
    labeled_reference_ids: tuple[str, ...]
    update_ids: tuple[str, ...]
    location_scaler: UpdateLocationScaler
    graph_feature_dim: int = Field(ge=3)
    encoder_feature_dim: int = Field(ge=1)
    adjacency_policy: Literal["row_normalized_cross_virtual_v1"] = (
        "row_normalized_cross_virtual_v1"
    )
    objective: Literal["streamed_exact_global_frobenius_eq5_eq9_v1"] = (
        "streamed_exact_global_frobenius_eq5_eq9_v1"
    )
    chunk_size: int = Field(ge=1)
    config: FingerprintUpdateConfig
    loss_history: tuple[UpdateEpochLoss, ...]
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arrays: tuple[StreamArrayArtifact, ...]
    created_at_utc: str

    @model_validator(mode="after")
    def aligned(self):
        if self.graph_feature_dim != self.encoder_feature_dim + 2:
            raise ValueError("streamed updater dimensions must satisfy graph=encoder+XY")
        if self.labeled_reference_ids != tuple(
            node_id for node_id in self.reference_ids if node_id in set(self.labeled_reference_ids)
        ):
            raise ValueError("labeled reference IDs must preserve reference ID order")
        if len(set(self.reference_ids + self.update_ids)) != len(self.reference_ids + self.update_ids):
            raise ValueError("streaming updater sample IDs collide")
        if len(self.loss_history) != self.config.updater_epochs:
            raise ValueError("streaming updater loss history has an invalid epoch count")
        expected_shapes = {
            "predicted_update_scaled_locations.npy": (len(self.update_ids), 2),
            "predicted_update_locations.npy": (len(self.update_ids), 2),
            "updated_reference_latents.npy": (
                len(self.labeled_reference_ids), self.encoder_feature_dim
            ),
        }
        if {item.name: item.shape for item in self.arrays} != expected_shapes:
            raise ValueError("streaming updater output arrays are misaligned")
        identity = canonical_json_digest(
            self.model_dump(mode="json", exclude={"updater_id"})
        )
        if self.updater_id != f"streamupdater:{identity}":
            raise ValueError("streaming updater identity does not match its metadata")
        return self


@dataclass(frozen=True)
class StreamingUpdaterState:
    directory: Path
    manifest: StreamingUpdaterManifest
    location_model: LocationPredictor
    feature_model: LocationToFeatureUpdater
    optimizer_state: dict
    predicted_update_locations: np.ndarray
    updated_reference_latents: np.ndarray


class StreamingAEUpdateManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_streaming_ae_update_v1"] = STREAM_AE_UPDATE_FORMAT
    update_id: str = Field(pattern=r"^streamae:[0-9a-f]{64}$")
    updater_id: str = Field(pattern=r"^streamupdater:[0-9a-f]{64}$")
    online_graph_id: str = Field(pattern=r"^onlinegraph:[0-9a-f]{64}$")
    dataset_id: str
    reference_batch_id: str
    update_batch_id: str
    reference_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    update_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ae_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    updater_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bssid_vocabulary: tuple[str, ...]
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalization: Literal["paper_ae", "legacy_raw_div_120"]
    reference_scan_ids: tuple[str, ...]
    labeled_reference_ids: tuple[str, ...]
    original_coordinates: tuple[tuple[float, float] | None, ...]
    update_scan_ids: tuple[str, ...]
    model_config_record: AEModelConfig
    config: FingerprintUpdateConfig
    objective: Literal["eq10_exact_global_frobenius_chunk_dropout_v1"] = (
        "eq10_exact_global_frobenius_chunk_dropout_v1"
    )
    chunk_size: int = Field(ge=1)
    loss_history: tuple[tuple[float, float, float], ...]
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arrays: tuple[StreamArrayArtifact, ...]
    created_at_utc: str

    @model_validator(mode="after")
    def aligned(self):
        if len(self.reference_scan_ids) != len(self.original_coordinates):
            raise ValueError("AE-updated coordinates must align with reference scan IDs")
        if self.labeled_reference_ids != tuple(
            scan_id
            for scan_id in self.reference_scan_ids
            if scan_id in set(self.labeled_reference_ids)
        ):
            raise ValueError("AE-updated labeled IDs must preserve reference order")
        if len(self.loss_history) != self.config.ae_epochs:
            raise ValueError("AE update loss history does not match completed epochs")
        expected = {
            "updated_rssi_dbm.npy": (len(self.reference_scan_ids), len(self.bssid_vocabulary)),
            "source_rssi_dbm.npy": (len(self.reference_scan_ids), len(self.bssid_vocabulary)),
            "observation_mask.npy": (len(self.reference_scan_ids), len(self.bssid_vocabulary)),
            "value_provenance.npy": (len(self.reference_scan_ids), len(self.bssid_vocabulary)),
        }
        if {item.name: item.shape for item in self.arrays} != expected:
            raise ValueError("AE update fingerprint arrays are misaligned")
        if canonical_json_digest(self.bssid_vocabulary) != self.vocabulary_sha256:
            raise ValueError("AE update vocabulary digest is inconsistent")
        identity = canonical_json_digest(
            self.model_dump(mode="json", exclude={"update_id"})
        )
        if self.update_id != f"streamae:{identity}":
            raise ValueError("streaming AE update identity is inconsistent")
        return self


@dataclass(frozen=True)
class StreamingAEUpdateState:
    directory: Path
    manifest: StreamingAEUpdateManifest
    model: AE
    optimizer_state: dict
    updated_rssi_dbm: np.ndarray
    source_rssi_dbm: np.ndarray
    observation_mask: np.ndarray
    value_provenance: np.ndarray


class StreamingPostTransitionAEManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_streaming_post_transition_ae_v2"] = STREAM_POST_TRANSITION_AE_FORMAT
    post_update_id: str = Field(pattern=r"^streampostae:[0-9a-f]{64}$")
    transition_id: str = Field(pattern=r"^streamtransition:[0-9a-f]{64}$")
    source_ae_update_id: str = Field(pattern=r"^streamae:[0-9a-f]{64}$")
    source_updater_id: str = Field(pattern=r"^streamupdater:[0-9a-f]{64}$")
    source_migrated_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dataset_id: str
    reference_batch_id: str
    update_batch_id: str
    update_scan_ids: tuple[str, ...]
    update_signal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    update_observed_readings: int = Field(gt=0)
    reference_scan_ids: tuple[str, ...]
    original_coordinates: tuple[tuple[float, float] | None, ...]
    labeled_reference_ids: tuple[str, ...]
    bssid_vocabulary: tuple[str, ...]
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalization: Literal["paper_ae", "legacy_raw_div_120"]
    objective: Literal["eq10_reference_anchor_plus_masked_update_rss_v1"] = (
        "eq10_reference_anchor_plus_masked_update_rss_v1"
    )
    model_config_record: AEModelConfig
    config: FingerprintUpdateConfig
    loss_history: tuple[tuple[float, float, float], ...]
    arrays: tuple[StreamArrayArtifact, ...]
    created_at_utc: str

    @model_validator(mode="after")
    def aligned(self):
        if not self.update_scan_ids:
            raise ValueError("post-transition AE requires measured update scans")
        if len(set(self.update_scan_ids)) != len(self.update_scan_ids):
            raise ValueError("post-transition AE update scan IDs must be unique")
        if len(self.reference_scan_ids) != len(self.original_coordinates):
            raise ValueError("post-transition AE coordinates must align with reference IDs")
        if self.labeled_reference_ids != tuple(
            scan_id for scan_id in self.reference_scan_ids
            if scan_id in set(self.labeled_reference_ids)
        ):
            raise ValueError("post-transition AE label IDs must preserve reference order")
        if len(self.loss_history) != self.config.ae_epochs:
            raise ValueError("post-transition AE loss history does not match configured epochs")
        if canonical_json_digest(self.bssid_vocabulary) != self.vocabulary_sha256:
            raise ValueError("post-transition AE vocabulary hash mismatch")
        expected = {
            "updated_rssi_dbm.npy": (len(self.reference_scan_ids), len(self.bssid_vocabulary)),
            "observation_mask.npy": (len(self.reference_scan_ids), len(self.bssid_vocabulary)),
            "value_provenance.npy": (len(self.reference_scan_ids), len(self.bssid_vocabulary)),
        }
        if {item.name: item.shape for item in self.arrays} != expected:
            raise ValueError("post-transition AE arrays do not align with IDs/vocabulary")
        identity = canonical_json_digest(
            self.model_dump(mode="json", exclude={"post_update_id"})
        )
        if self.post_update_id != f"streampostae:{identity}":
            raise ValueError("post-transition AE identity differs from metadata")
        return self


@dataclass(frozen=True)
class StreamingPostTransitionAEState:
    directory: Path
    manifest: StreamingPostTransitionAEManifest
    model: AE
    updated_rssi_dbm: np.ndarray
    observation_mask: np.ndarray
    value_provenance: np.ndarray


def _open(path: Path, shape: tuple[int, ...], dtype="float32") -> np.ndarray:
    if math.prod(shape) == 0:
        arr = np.empty(shape, dtype=dtype)
        np.save(path, arr, allow_pickle=False)
        return arr
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _tensor(values, device: torch.device) -> Tensor:
    return torch.tensor(np.asarray(values), dtype=torch.float32, device=device)


def _ae_chunk(
    scans,
    start: int,
    stop: int,
    normalization: str,
) -> np.ndarray:
    return np.asarray(
        [
            raw_dbm_row_to_ae(
                scan.rss_dbm,
                ae_input_observation_mask(scan),
                normalization,
            )
            for scan in scans[start:stop]
        ],
        dtype=np.float32,
    )[:, None, :]


def _seed_ae_pass(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _scaled_xy(coordinates: np.ndarray, scaler: UpdateLocationScaler) -> np.ndarray:
    result = np.empty_like(coordinates, dtype=np.float32)
    spans = np.asarray((scaler.x_max - scaler.x_min, scaler.y_max - scaler.y_min))
    spans[spans == 0.0] = 1.0
    result[:] = (coordinates - np.asarray((scaler.x_min, scaler.y_min))) / spans
    return result


def _adjacency_targets(
    graph: MappedStreamingOnlineGraph,
    scaler: UpdateLocationScaler,
    coordinates: np.ndarray,
    directory: Path,
):
    n_update = len(graph.manifest.update_sample_ids)
    n_label = len(graph.manifest.labeled_reference_ids)
    xy = _open(directory / "cross_reference_xy.npy", (n_update, 2))
    nonempty = np.zeros((n_update,), dtype=np.bool_)
    y_scaled = _scaled_xy(coordinates, scaler)
    indptr, cols, weights = (
        graph.update_reference_indptr,
        graph.update_reference_cols,
        graph.update_reference_values,
    )
    if len(indptr) != n_update + 1 or len(coordinates) != n_label:
        raise ValueError("streamed cross-reference adjacency rows/columns are misaligned")
    for row in range(n_update):
        start, stop = int(indptr[row]), int(indptr[row + 1])
        if start == stop:
            xy[row] = 0.0
            continue
        row_cols = np.asarray(cols[start:stop], dtype=np.int64)
        row_weights = np.asarray(weights[start:stop], dtype=np.float64)
        if np.any(row_cols < 0) or np.any(row_cols >= n_label):
            raise ValueError("cross-reference adjacency column is out of range")
        if not math.isclose(float(row_weights.sum()), 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("nonempty cross-reference adjacency row is not normalized")
        xy[row] = row_weights @ y_scaled[row_cols]
        nonempty[row] = True
    if isinstance(xy, np.memmap):
        xy.flush()
    return xy, nonempty, y_scaled


def _global_losses(
    graph: MappedStreamingOnlineGraph,
    gnn_state: StreamingGNNState,
    location_model: LocationPredictor,
    feature_model: LocationToFeatureUpdater,
    ref_rows: np.ndarray,
    update_rows: np.ndarray,
    y_ref: np.ndarray,
    z_update: np.ndarray,
    cross_xy: np.ndarray,
    nonempty: np.ndarray,
    feature_ref_path: Path,
    update_xy_path: Path,
    *,
    chunk_size: int,
    device: torch.device,
):
    feature_ref = _open(
        feature_ref_path,
        (len(ref_rows), graph.manifest.encoder_feature_dim),
    )
    update_xy = _open(update_xy_path, (len(update_rows), 2))
    sums = np.zeros((4,), dtype=np.float64)
    location_model.eval()
    feature_model.eval()
    with torch.no_grad():
        for start in range(0, len(ref_rows), chunk_size):
            stop = min(start + chunk_size, len(ref_rows))
            x = _tensor(gnn_state.node_embeddings[ref_rows[start:stop]], device)
            y = _tensor(y_ref[start:stop], device)
            z = feature_model(y)
            feature_ref[start:stop] = z.cpu().numpy()
            sums[0] += float((location_model(x) - y).double().square().sum().cpu())
        for start in range(0, len(update_rows), chunk_size):
            stop = min(start + chunk_size, len(update_rows))
            x = _tensor(gnn_state.node_embeddings[update_rows[start:stop]], device)
            xy = location_model(x)
            update_xy[start:stop] = xy.cpu().numpy()
            zhat = feature_model(xy)
            fixed_z = _tensor(z_update[start:stop], device)
            sums[1] += float((zhat - fixed_z).double().square().sum().cpu())
            use = nonempty[start:stop]
            if np.any(use):
                target_xy = _tensor(cross_xy[start:stop], device)
                sums[2] += float((xy[torch.as_tensor(use, device=device)] - target_xy[torch.as_tensor(use, device=device)]).double().square().sum().cpu())
        if isinstance(feature_ref, np.memmap):
            feature_ref.flush()
        if isinstance(update_xy, np.memmap):
            update_xy.flush()
        for row in np.flatnonzero(nonempty):
            start, stop = int(graph.update_reference_indptr[row]), int(graph.update_reference_indptr[row + 1])
            col = np.asarray(graph.update_reference_cols[start:stop], dtype=np.int64)
            weight = np.asarray(graph.update_reference_values[start:stop], dtype=np.float64)
            mixed = weight @ np.asarray(feature_ref[col], dtype=np.float64)
            residual = mixed - z_update[row]
            sums[3] += float(np.dot(residual, residual))
    return np.sqrt(sums), feature_ref, update_xy


def _accumulate_global_gradients(
    graph: MappedStreamingOnlineGraph,
    gnn_state: StreamingGNNState,
    location_model: LocationPredictor,
    feature_model: LocationToFeatureUpdater,
    ref_rows: np.ndarray,
    update_rows: np.ndarray,
    y_ref: np.ndarray,
    z_update: np.ndarray,
    cross_xy: np.ndarray,
    nonempty: np.ndarray,
    norm: np.ndarray,
    feature_ref: np.ndarray,
    update_xy: np.ndarray,
    grad_ref_path: Path,
    *,
    alpha: float,
    chunk_size: int,
    device: torch.device,
):
    location_model.train()
    feature_model.train()
    weight_lp, weight_lu = alpha, alpha
    weight_lcp, weight_lcu = 1.0 - alpha, 1.0 - alpha

    if weight_lp and norm[0] > 0.0:
        for start in range(0, len(ref_rows), chunk_size):
            stop = min(start + chunk_size, len(ref_rows))
            x = _tensor(gnn_state.node_embeddings[ref_rows[start:stop]], device)
            y = _tensor(y_ref[start:stop], device)
            residual = location_model(x) - y
            (weight_lp * residual.square().sum() / (2.0 * float(norm[0]))).backward()

    if weight_lu and norm[1] > 0.0:
        for start in range(0, len(update_rows), chunk_size):
            stop = min(start + chunk_size, len(update_rows))
            x = _tensor(gnn_state.node_embeddings[update_rows[start:stop]], device)
            target = _tensor(z_update[start:stop], device)
            prediction = feature_model(location_model(x))
            residual = prediction - target
            (weight_lu * residual.square().sum() / (2.0 * float(norm[1]))).backward()

    if weight_lcp and norm[2] > 0.0:
        for start in range(0, len(update_rows), chunk_size):
            stop = min(start + chunk_size, len(update_rows))
            rows = np.flatnonzero(nonempty[start:stop]) + start
            if not len(rows):
                continue
            x = _tensor(gnn_state.node_embeddings[update_rows[rows]], device)
            target = _tensor(cross_xy[rows], device)
            residual = location_model(x) - target
            (weight_lcp * residual.square().sum() / (2.0 * float(norm[2]))).backward()

    if weight_lcu and norm[3] > 0.0:
        grad_ref = _open(grad_ref_path, (len(ref_rows), graph.manifest.encoder_feature_dim))
        grad_ref[:] = 0.0
        for row in np.flatnonzero(nonempty):
            start, stop = int(graph.update_reference_indptr[row]), int(graph.update_reference_indptr[row + 1])
            col = np.asarray(graph.update_reference_cols[start:stop], dtype=np.int64)
            edge_weight = np.asarray(graph.update_reference_values[start:stop], dtype=np.float64)
            mixed = edge_weight @ np.asarray(feature_ref[col], dtype=np.float64)
            error = mixed - z_update[row]
            contribution = (
                weight_lcu * error / (2.0 * float(norm[3]))
            )
            np.add.at(grad_ref, col, edge_weight[:, None] * contribution[None, :])
        if isinstance(grad_ref, np.memmap):
            grad_ref.flush()
        for start in range(0, len(ref_rows), chunk_size):
            stop = min(start + chunk_size, len(ref_rows))
            target_gradient = _tensor(grad_ref[start:stop], device)
            if not torch.count_nonzero(target_gradient).item():
                continue
            y = _tensor(y_ref[start:stop], device)
            output = feature_model(y)
            (output * target_gradient).sum().backward()


def load_streaming_updater_warm_start(
    directory: str | Path,
    *,
    graph_feature_dim: int,
    encoder_feature_dim: int,
) -> tuple[str, dict[str, Tensor], dict[str, Tensor]]:
    """Load prior-batch updater weights for a fresh-optimizer continuation.

    Only the model weights cross the batch boundary. The previous graph and
    vocabulary lineage remain attached to that prior result, while dimensions
    are checked before those weights initialize the next batch's training.
    """
    root = Path(directory)
    manifest_path = root / "manifest.json"
    marker_path = root / "COMPLETED.json"
    checkpoint_path = root / "updaters.pt"
    if not manifest_path.is_file() or not marker_path.is_file() or not checkpoint_path.is_file():
        raise ValueError(f"previous-batch updater checkpoint is incomplete: {root}")
    manifest = StreamingUpdaterManifest.model_validate_json(read_json_text(manifest_path))
    marker = json.loads(read_json_text(marker_path))
    if (
        marker.get("identity") != manifest.updater_id
        or marker.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("previous-batch updater completion marker is inconsistent")
    if sha256_file(checkpoint_path) != manifest.checkpoint_sha256:
        raise ValueError("previous-batch updater checkpoint digest mismatch")
    if (
        manifest.graph_feature_dim != graph_feature_dim
        or manifest.encoder_feature_dim != encoder_feature_dim
    ):
        raise ValueError("previous-batch updater dimensions are incompatible with this graph")
    payload = load_torch_state(checkpoint_path, map_location="cpu")
    if payload.get("format") != STREAM_UPDATER_FORMAT:
        raise ValueError("unsupported previous-batch updater checkpoint format")
    location = payload.get("location_model_state")
    feature = payload.get("feature_model_state")
    if not isinstance(location, dict) or not isinstance(feature, dict):
        raise ValueError("previous-batch updater is missing learned model weights")
    return (
        manifest.updater_id,
        {key: value.detach().cpu() for key, value in location.items()},
        {key: value.detach().cpu() for key, value in feature.items()},
    )


def train_streaming_updaters(
    online_graph_directory: str | Path,
    gnn_state: StreamingGNNState,
    reference_prepared: PreparedDataset,
    update_prepared: PreparedDataset,
    update_encoder: EncoderFeatureArtifact,
    output_directory: str | Path,
    config: FingerprintUpdateConfig,
    *,
    source_ae_checkpoint_sha256: str,
    source_updater_id: str | None = None,
    initial_location_state: dict[str, Tensor] | None = None,
    initial_feature_state: dict[str, Tensor] | None = None,
    chunk_size: int = 4096,
    device: str | torch.device = "cpu",
    log_every_epochs: int = DEFAULT_LOG_EVERY_K_EPOCHS,
) -> StreamingUpdaterState:
    """Train Eq. (5)--(9) MLPs without materializing graph tensors or dense A."""
    if chunk_size < 1:
        raise ValueError("streamed updater chunk_size must be positive")
    log_every_epochs = validate_log_interval(log_every_epochs)
    graph = load_streaming_online_graph(online_graph_directory)
    gm = graph.manifest
    sm = gnn_state.manifest
    if sm.online_graph_id != gm.online_graph_id or sm.node_ids != gm.node_ids:
        raise ValueError("streamed updater GNN state does not match online graph identity/order")
    if sm.vocabulary_sha256 != gm.vocabulary_sha256:
        raise ValueError("streamed updater GNN state uses another vocabulary")
    if reference_prepared.config.batch_role != "bootstrap_reference":
        raise ValueError("streamed updater needs a bootstrap reference dataset")
    if update_prepared.config.batch_role != "update_batch":
        raise ValueError("streamed updater input must have update_batch role")
    if reference_prepared.config.dataset_id != gm.dataset_id or update_prepared.config.dataset_id != gm.dataset_id:
        raise ValueError("streamed updater datasets differ from online graph")
    if reference_prepared.config.normalization_mode != gm.normalization or update_prepared.config.normalization_mode != gm.normalization:
        raise ValueError("streamed updater normalization differs from online graph")
    if reference_prepared.bssid_vocabulary != gm.bssid_vocabulary or update_prepared.bssid_vocabulary != gm.bssid_vocabulary:
        raise ValueError("streamed updater currently requires unchanged ordered vocabulary")
    if canonical_json_digest(reference_prepared) != gm.reference_prepared_sha256:
        raise ValueError("online graph was built from different reference prepared data")
    if canonical_json_digest(update_prepared) != gm.update_prepared_sha256:
        raise ValueError("online graph was built from different update prepared data")
    if canonical_json_digest(update_encoder) != gm.update_encoder_artifact_sha256:
        raise ValueError("online graph uses a different update encoder artifact")
    if update_encoder.scan_ids != gm.update_sample_ids or update_encoder.bssid_vocabulary != gm.bssid_vocabulary:
        raise ValueError("update encoder IDs/vocabulary are incompatible")
    if update_encoder.checkpoint_sha256 != gm.encoder_checkpoint_sha256:
        raise ValueError("update encoder checkpoint differs from online graph")
    if source_ae_checkpoint_sha256 != gm.encoder_checkpoint_sha256:
        raise ValueError("source AE checkpoint differs from online graph")
    if (initial_location_state is None) != (initial_feature_state is None):
        raise ValueError("updater warm start requires both learned model states")
    if (source_updater_id is None) != (initial_location_state is None):
        raise ValueError("updater warm-start ID and both model states must be supplied together")
    if update_encoder.feature_dim != gm.encoder_feature_dim:
        raise ValueError("update encoder width differs from online graph")
    if not gm.labeled_reference_ids:
        raise ValueError("streamed updater requires at least one bootstrap training label")
    if len(gm.labeled_reference_ids) != len(gm.labeled_reference_coordinates):
        raise AssertionError("online graph label IDs/coordinates are misaligned")

    ref_positions = {
        row.scan_id: (row.x, row.y) for row in reference_prepared.reference_positions
    }
    if tuple(scan.scan_id for scan in reference_prepared.scans) != gm.reference_sample_ids:
        raise ValueError("streamed updater reference scan identity/order changed")
    if tuple(ref_positions.get(scan_id) for scan_id in gm.labeled_reference_ids) != gm.labeled_reference_coordinates:
        raise ValueError("online graph labels differ from immutable bootstrap coordinates")
    if config.virtual_edge_sigma != gm.virtual_edge_sigma:
        raise ValueError("streamed updater virtual edge sigma differs from graph provenance")

    label_ids = gm.labeled_reference_ids
    update_ids = gm.update_sample_ids
    ref_pos = {node_id: row for row, node_id in enumerate(gm.node_ids)}
    ref_rows = np.asarray([ref_pos[node_id] for node_id in label_ids], dtype=np.int64)
    update_rows = np.asarray([ref_pos[node_id] for node_id in update_ids], dtype=np.int64)
    coordinates = np.asarray(gm.labeled_reference_coordinates, dtype=np.float32)
    scaler = UpdateLocationScaler(
        x_min=float(coordinates[:, 0].min()),
        x_max=float(coordinates[:, 0].max()),
        y_min=float(coordinates[:, 1].min()),
        y_max=float(coordinates[:, 1].max()),
    )
    y_ref = _scaled_xy(coordinates, scaler)
    encoder_rows = {scan_id: row for row, scan_id in enumerate(update_encoder.scan_ids)}
    z_update = np.asarray(
        [update_encoder.feature_matrix[encoder_rows[scan_id]] for scan_id in update_ids],
        dtype=np.float32,
    )

    target = Path(output_directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite streaming updater state: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"requested CUDA device is unavailable: {selected_device}")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        location_model = LocationPredictor(config.hidden_dim, gm.node_feature_dim).to(selected_device)
        feature_model = LocationToFeatureUpdater(config.hidden_dim, gm.encoder_feature_dim).to(selected_device)
        if initial_location_state is not None:
            location_model.load_state_dict(initial_location_state, strict=True)
        if initial_feature_state is not None:
            feature_model.load_state_dict(initial_feature_state, strict=True)
        optimizer = torch.optim.Adam(
            list(location_model.parameters()) + list(feature_model.parameters()),
            lr=config.updater_learning_rate,
        )
        cross_xy, nonempty, _ = _adjacency_targets(graph, scaler, coordinates, staging)
        history: list[UpdateEpochLoss] = []
        for epoch in range(1, config.updater_epochs + 1):
            optimizer.zero_grad(set_to_none=True)
            norms, feature_ref, update_xy = _global_losses(
                graph,
                gnn_state,
                location_model,
                feature_model,
                ref_rows,
                update_rows,
                y_ref,
                z_update,
                cross_xy,
                nonempty,
                staging / "reference_feature_pass.npy",
                staging / "update_location_pass.npy",
                chunk_size=chunk_size,
                device=selected_device,
            )
            _accumulate_global_gradients(
                graph,
                gnn_state,
                location_model,
                feature_model,
                ref_rows,
                update_rows,
                y_ref,
                z_update,
                cross_xy,
                nonempty,
                norms,
                feature_ref,
                update_xy,
                staging / "reference_feature_gradient.npy",
                alpha=config.alpha,
                chunk_size=chunk_size,
                device=selected_device,
            )
            gradients = [parameter.grad for group in optimizer.param_groups for parameter in group["params"] if parameter.grad is not None]
            if not gradients or any(not torch.isfinite(grad).all().item() for grad in gradients):
                raise FloatingPointError("streamed updater gradients are missing or non-finite")
            if not any(torch.count_nonzero(grad).item() for grad in gradients):
                raise RuntimeError("streamed updater objective produced no parameter gradient")
            optimizer.step()
            a, b, c, d = (float(value) for value in norms)
            total = config.alpha * (a + b) + (1.0 - config.alpha) * (c + d)
            epoch_loss = UpdateEpochLoss(
                epoch=epoch,
                location_supervised=a,
                feature_supervised=b,
                location_consistency=c,
                feature_consistency=d,
                total=total,
            )
            history.append(epoch_loss)
            log_training_epoch(
                "Streaming location/feature updaters",
                epoch,
                config.updater_epochs,
                {
                    **epoch_loss.model_dump(exclude={"epoch"}),
                    "labeled_refs": len(label_ids),
                    "update_scans": len(update_ids),
                    "chunk_size": chunk_size,
                },
                every_k_epochs=log_every_epochs,
            )
            # Per-epoch arrays are staging scratch, not completed artifacts.
            for scratch in (
                staging / "reference_feature_pass.npy",
                staging / "update_location_pass.npy",
                staging / "reference_feature_gradient.npy",
            ):
                scratch.unlink(missing_ok=True)

        location_model.eval()
        feature_model.eval()
        predicted_scaled = _open(staging / "predicted_update_scaled_locations.npy", (len(update_ids), 2))
        for start in range(0, len(update_rows), chunk_size):
            stop = min(start + chunk_size, len(update_rows))
            with torch.no_grad():
                pred = location_model(_tensor(gnn_state.node_embeddings[update_rows[start:stop]], selected_device))
            predicted_scaled[start:stop] = pred.cpu().numpy()
        predicted_raw = _open(staging / "predicted_update_locations.npy", (len(update_ids), 2))
        spans = np.asarray((scaler.x_max - scaler.x_min, scaler.y_max - scaler.y_min), dtype=np.float32)
        predicted_raw[:] = np.asarray((scaler.x_min, scaler.y_min), dtype=np.float32) + predicted_scaled[:] * spans
        latent_ref = _open(staging / "updated_reference_latents.npy", (len(label_ids), gm.encoder_feature_dim))
        for start in range(0, len(label_ids), chunk_size):
            stop = min(start + chunk_size, len(label_ids))
            with torch.no_grad():
                latent_ref[start:stop] = feature_model(_tensor(y_ref[start:stop], selected_device)).cpu().numpy()
        for arr in (predicted_scaled, predicted_raw, latent_ref):
            if isinstance(arr, np.memmap):
                arr.flush()

        checkpoint_path = staging / "updaters.pt"
        atomic_torch_save(
            {
                "format": STREAM_UPDATER_FORMAT,
                "location_model_state": {key: value.detach().cpu().clone() for key, value in location_model.state_dict().items()},
                "feature_model_state": {key: value.detach().cpu().clone() for key, value in feature_model.state_dict().items()},
                "optimizer_state": optimizer.state_dict(),
                "torch_cpu_rng_state": torch.get_rng_state().cpu().clone(),
                "resumable": False,
            },
            checkpoint_path,
        )
        arrays = tuple(
            _array_artifact(staging / name, name)
            for name in (
                "predicted_update_scaled_locations.npy",
                "predicted_update_locations.npy",
                "updated_reference_latents.npy",
            )
        )
        manifest_fields = dict(
            online_graph_id=gm.online_graph_id,
            gnn_artifact_id=sm.artifact_id,
            dataset_id=gm.dataset_id,
            reference_batch_id=gm.reference_batch_id,
            update_batch_id=gm.update_batch_id,
            reference_prepared_sha256=gm.reference_prepared_sha256,
            update_prepared_sha256=gm.update_prepared_sha256,
            update_encoder_artifact_sha256=gm.update_encoder_artifact_sha256,
            source_ae_checkpoint_sha256=gm.encoder_checkpoint_sha256,
            source_gnn_checkpoint_sha256=sm.checkpoint_sha256,
            source_updater_id=source_updater_id,
            initialization_policy=(
                "previous_batch_weights_only" if source_updater_id is not None else "fresh"
            ),
            vocabulary_sha256=gm.vocabulary_sha256,
            bssid_vocabulary=gm.bssid_vocabulary,
            normalization=gm.normalization,
            reference_ids=gm.reference_sample_ids,
            labeled_reference_ids=label_ids,
            update_ids=update_ids,
            location_scaler=scaler,
            graph_feature_dim=gm.node_feature_dim,
            encoder_feature_dim=gm.encoder_feature_dim,
            chunk_size=chunk_size,
            config=config,
            loss_history=tuple(history),
            checkpoint_sha256=sha256_file(checkpoint_path),
            arrays=arrays,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        provisional = StreamingUpdaterManifest.model_construct(
            updater_id="streamupdater:" + "0" * 64, **manifest_fields
        )
        updater_id = "streamupdater:" + canonical_json_digest(
            provisional.model_dump(mode="json", exclude={"updater_id"})
        )
        manifest = StreamingUpdaterManifest(updater_id=updater_id, **manifest_fields)
        atomic_write_json(manifest, staging / "manifest.json")
        atomic_write_json(
            {
                "format": "gufu_streaming_artifact_complete_v1",
                "identity": manifest.updater_id,
                "manifest_sha256": sha256_file(staging / "manifest.json"),
            },
            staging / "COMPLETED.json",
        )
        if target.exists():
            raise FileExistsError(f"refusing to overwrite streaming updater state: {target}")
        os.replace(staging, target)
    return load_streaming_updater(target, graph, gnn_state)


def load_streaming_updater(
    directory: str | Path,
    graph: MappedStreamingOnlineGraph,
    gnn_state: StreamingGNNState,
    *,
    verify_hashes: bool = True,
) -> StreamingUpdaterState:
    root = Path(directory)
    manifest_path = root / "manifest.json"
    marker_path = root / "COMPLETED.json"
    checkpoint_path = root / "updaters.pt"
    if not manifest_path.is_file() or not marker_path.is_file() or not checkpoint_path.is_file():
        raise ValueError(f"streaming updater artifact is incomplete: {root}")
    manifest = StreamingUpdaterManifest.model_validate_json(read_json_text(manifest_path))
    marker = json.loads(read_json_text(marker_path))
    if marker.get("identity") != manifest.updater_id or marker.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("streaming updater completion marker does not match manifest")
    gm, sm = graph.manifest, gnn_state.manifest
    if manifest.online_graph_id != gm.online_graph_id or manifest.gnn_artifact_id != sm.artifact_id:
        raise ValueError("streaming updater model state belongs to different graph/GNN lineage")
    if manifest.reference_ids != gm.reference_sample_ids or manifest.update_ids != gm.update_sample_ids:
        raise ValueError("streaming updater ID order differs from graph")
    if manifest.vocabulary_sha256 != gm.vocabulary_sha256 or manifest.source_gnn_checkpoint_sha256 != sm.checkpoint_sha256:
        raise ValueError("streaming updater checkpoint vocabulary/GNN provenance mismatch")
    if verify_hashes and sha256_file(checkpoint_path) != manifest.checkpoint_sha256:
        raise ValueError("streaming updater checkpoint hash mismatch")
    payload = load_torch_state(checkpoint_path, map_location="cpu")
    if payload.get("format") != STREAM_UPDATER_FORMAT:
        raise ValueError("unsupported streaming updater checkpoint format")
    location = LocationPredictor(manifest.config.hidden_dim, manifest.graph_feature_dim)
    feature = LocationToFeatureUpdater(manifest.config.hidden_dim, manifest.encoder_feature_dim)
    location.load_state_dict(payload["location_model_state"], strict=True)
    feature.load_state_dict(payload["feature_model_state"], strict=True)
    device = next(gnn_state.model.parameters()).device
    location.to(device).eval()
    feature.to(device).eval()
    for item in manifest.arrays:
        path = root / item.name
        if not path.is_file() or path.stat().st_size != item.size_bytes:
            raise ValueError(f"streaming updater output array is missing: {item.name}")
        if verify_hashes and sha256_file(path) != item.sha256:
            raise ValueError(f"streaming updater output array hash mismatch: {item.name}")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(value.shape) != item.shape or value.dtype.name != item.dtype:
            raise ValueError(f"streaming updater output shape/dtype mismatch: {item.name}")
    locations = np.load(root / "predicted_update_locations.npy", mmap_mode="r", allow_pickle=False)
    latents = np.load(root / "updated_reference_latents.npy", mmap_mode="r", allow_pickle=False)
    if locations.shape != (len(manifest.update_ids), 2) or latents.shape != (len(manifest.labeled_reference_ids), manifest.encoder_feature_dim):
        raise ValueError("streaming updater output array shape is invalid")
    return StreamingUpdaterState(
        directory=root,
        manifest=manifest,
        location_model=location,
        feature_model=feature,
        optimizer_state=payload["optimizer_state"],
        predicted_update_locations=locations,
        updated_reference_latents=latents,
    )


def train_streaming_ae_update(
    reference_prepared: PreparedDataset,
    update_prepared: PreparedDataset,
    source_ae_checkpoint_path: str | Path,
    updater_state: StreamingUpdaterState,
    output_directory: str | Path,
    config: FingerprintUpdateConfig,
    *,
    chunk_size: int = 256,
    device: str | torch.device = "cpu",
    log_every_epochs: int = DEFAULT_LOG_EVERY_K_EPOCHS,
) -> StreamingAEUpdateState:
    """Stream Eq. (10) fine-tuning and decode reference fingerprints.

    Frobenius reductions are global over all labeled reference rows. Each
    epoch computes global norms in one pass and accumulates the exact norm
    gradients in a second pass. Dropout uses a named deterministic chunk
    sequence so the two passes see identical masks. Hidden update coordinates
    are never accessed.
    """
    if chunk_size < 1:
        raise ValueError("streaming AE chunk_size must be positive")
    log_every_epochs = validate_log_interval(log_every_epochs)
    if reference_prepared.config.batch_role != "bootstrap_reference":
        raise ValueError("streaming AE update requires bootstrap reference data")
    if update_prepared.config.batch_role != "update_batch":
        raise ValueError("streaming AE update requires update_batch provenance")
    rm = updater_state.manifest
    ref_ids = tuple(scan.scan_id for scan in reference_prepared.scans)
    update_ids = tuple(scan.scan_id for scan in update_prepared.scans)
    if ref_ids != rm.reference_ids or update_ids != rm.update_ids:
        raise ValueError("streaming AE update scan IDs/order differ from updater state")
    if canonical_json_digest(reference_prepared) != rm.reference_prepared_sha256:
        raise ValueError("streaming AE update reference data differs from updater lineage")
    if canonical_json_digest(update_prepared) != rm.update_prepared_sha256:
        raise ValueError("streaming AE update batch differs from updater lineage")
    if reference_prepared.config.dataset_id != rm.dataset_id or update_prepared.config.dataset_id != rm.dataset_id:
        raise ValueError("streaming AE update dataset identity differs")
    if reference_prepared.bssid_vocabulary != rm.bssid_vocabulary or update_prepared.bssid_vocabulary != rm.bssid_vocabulary:
        raise ValueError("streaming AE update currently requires unchanged AP vocabulary")
    if reference_prepared.config.normalization_mode != rm.normalization or update_prepared.config.normalization_mode != rm.normalization:
        raise ValueError("streaming AE update normalization differs from updater lineage")
    if updater_state.updated_reference_latents.shape != (
        len(rm.labeled_reference_ids), rm.encoder_feature_dim
    ):
        raise ValueError("streaming AE update latent target dimensions are incompatible")

    source = load_ae_checkpoint(
        source_ae_checkpoint_path, reference_prepared, device="cpu", dtype="float32"
    )
    if source.checkpoint_sha256 != rm.source_ae_checkpoint_sha256:
        raise ValueError("streaming AE update source checkpoint differs from updater")
    if source.manifest.model.latent_dim != rm.encoder_feature_dim:
        raise ValueError("streaming AE latent width differs from updater output")
    label_positions = {
        row.scan_id: (row.x, row.y) for row in reference_prepared.reference_positions
    }
    if set(label_positions) != set(rm.labeled_reference_ids):
        raise ValueError("AE update labels must be exactly the immutable bootstrap labels")
    if tuple(scan_id for scan_id in ref_ids if scan_id in label_positions) != rm.labeled_reference_ids:
        raise ValueError("AE update label IDs are not in reference scan order")
    scan_by_id = {scan.scan_id: scan for scan in reference_prepared.scans}
    labeled_scans = tuple(scan_by_id[scan_id] for scan_id in rm.labeled_reference_ids)
    if any(len(scan.rss_dbm) != len(rm.bssid_vocabulary) for scan in labeled_scans):
        raise ValueError("AE update reference RSS width differs from AP vocabulary")

    target = Path(output_directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite streaming AE update: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"requested CUDA device is unavailable: {selected_device}")
    cuda_devices = (
        [selected_device.index if selected_device.index is not None else torch.cuda.current_device()]
        if selected_device.type == "cuda"
        else []
    )
    model = copy.deepcopy(source.model).to(selected_device, dtype=torch.float32)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.ae_learning_rate, weight_decay=config.ae_weight_decay
    )
    loss_history: list[tuple[float, float, float, float]] = []
    saved_rng_state: dict[str, object] = {}

    with torch.random.fork_rng(devices=cuda_devices):
        for epoch in range(1, config.ae_epochs + 1):
            epoch_seed = config.seed + 303 + epoch
            model.train()
            _seed_ae_pass(epoch_seed, selected_device)
            squared = np.zeros((3,), dtype=np.float64)
            with torch.no_grad():
                for start in range(0, len(labeled_scans), chunk_size):
                    stop = min(start + chunk_size, len(labeled_scans))
                    x = _tensor(
                        _ae_chunk(labeled_scans, start, stop, rm.normalization),
                        selected_device,
                    )
                    z_target = _tensor(
                        updater_state.updated_reference_latents[start:stop],
                        selected_device,
                    ).detach()
                    z, reconstruction = model(x)
                    target_reconstruction = model.D(z_target.unsqueeze(1))
                    squared[0] += float((reconstruction - x).double().square().sum().cpu())
                    squared[1] += float((z[:, 0, :] - z_target).double().square().sum().cpu())
                    squared[2] += float((target_reconstruction - reconstruction).double().square().sum().cpu())
            norms = np.sqrt(squared)

            optimizer.zero_grad(set_to_none=True)
            model.train()
            _seed_ae_pass(epoch_seed, selected_device)
            for start in range(0, len(labeled_scans), chunk_size):
                stop = min(start + chunk_size, len(labeled_scans))
                x = _tensor(
                    _ae_chunk(labeled_scans, start, stop, rm.normalization),
                    selected_device,
                )
                z_target = _tensor(
                    updater_state.updated_reference_latents[start:stop], selected_device
                ).detach()
                z, reconstruction = model(x)
                target_reconstruction = model.D(z_target.unsqueeze(1))
                objective = reconstruction.sum() * 0.0
                if norms[0] > 0.0:
                    residual = reconstruction - x
                    objective = objective + residual.square().sum() / (2.0 * float(norms[0]))
                if norms[1] > 0.0:
                    residual = z[:, 0, :] - z_target
                    objective = objective + 0.5 * residual.square().sum() / (2.0 * float(norms[1]))
                if norms[2] > 0.0:
                    residual = target_reconstruction - reconstruction
                    objective = objective + 0.5 * residual.square().sum() / (2.0 * float(norms[2]))
                objective.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or any(not torch.isfinite(grad).all().item() for grad in gradients):
                raise FloatingPointError("streamed Eq. (10) AE gradients are missing or non-finite")
            if not any(torch.count_nonzero(grad).item() for grad in gradients):
                raise RuntimeError("streamed Eq. (10) objective produced no parameter gradient")
            optimizer.step()
            loss_history.append(tuple(float(value) for value in norms))
            log_training_epoch(
                "Streaming batch AE Eq. (10)",
                epoch,
                config.ae_epochs,
                {
                    "reconstruction_frobenius": float(norms[0]),
                    "latent_frobenius": float(norms[1]),
                    "decoder_frobenius": float(norms[2]),
                    "total": float(norms[0] + 0.5 * (norms[1] + norms[2])),
                    "labeled_reference_scans": len(labeled_scans),
                    "APs": len(rm.bssid_vocabulary),
                    "chunk_size": chunk_size,
                },
                every_k_epochs=log_every_epochs,
            )
        saved_rng_state = {
            "torch_cpu": torch.get_rng_state().cpu().clone(),
            "torch_cuda": torch.cuda.get_rng_state_all() if selected_device.type == "cuda" else None,
        }

    model.eval()
    n_ref, n_ap = len(ref_ids), len(rm.bssid_vocabulary)
    updated = _open(staging / "updated_rssi_dbm.npy", (n_ref, n_ap))
    source_rss = _open(staging / "source_rssi_dbm.npy", (n_ref, n_ap))
    masks = _open(staging / "observation_mask.npy", (n_ref, n_ap), dtype="uint8")
    provenance = _open(staging / "value_provenance.npy", (n_ref, n_ap), dtype="uint8")
    label_row = {scan_id: row for row, scan_id in enumerate(rm.labeled_reference_ids)}
    parameter = next(model.parameters())
    with torch.inference_mode():
        for ref_row, scan_id in enumerate(ref_ids):
            scan = scan_by_id[scan_id]
            source_rss[ref_row] = np.asarray(scan.rss_dbm, dtype=np.float32)
            masks[ref_row] = np.asarray(scan.observation_mask, dtype=np.uint8)
            if scan_id not in label_row:
                updated[ref_row] = source_rss[ref_row]
                provenance[ref_row] = 0
                continue
            latent = _tensor(
                updater_state.updated_reference_latents[label_row[scan_id] : label_row[scan_id] + 1],
                selected_device,
            )
            normalized = model.D(latent.unsqueeze(1))[0, 0].cpu().numpy()
            updated[ref_row] = np.asarray(
                [ae_value_to_raw_dbm(float(value), rm.normalization) for value in normalized],
                dtype=np.float32,
            )
            provenance[ref_row] = 1
    for array in (updated, source_rss, masks, provenance):
        if isinstance(array, np.memmap):
            array.flush()

    source_training = source.manifest.training_config.model_copy(
        update={
            "epochs": config.ae_epochs,
            "batch_size": chunk_size,
            "learning_rate": config.ae_learning_rate,
            "weight_decay": config.ae_weight_decay,
            "seed": config.seed,
            "shuffle": False,
            "optimizer_betas": (0.9, 0.999),
            "optimizer_epsilon": 1.0e-8,
            "amsgrad": False,
            "dtype": "float32",
            "loss": "batch_frobenius_norm",
        }
    )
    target_training_dataset = DatasetIdentity(
        dataset_id=reference_prepared.config.dataset_id,
        batch_id=reference_prepared.config.batch_id,
        batch_role="bootstrap_reference",
        prepared_sha256=canonical_json_digest(reference_prepared),
        scan_ids=ref_ids,
        scan_count=len(ref_ids),
    )
    ae_checkpoint_manifest = AECheckpointManifest(
        model=model.config,
        normalization=rm.normalization,
        bssid_vocabulary=rm.bssid_vocabulary,
        vocabulary_sha256=rm.vocabulary_sha256,
        training_dataset=target_training_dataset,
        training_config=AECheckpointTrainingConfig.model_validate(source_training),
        initialization_mode="streaming_update",
        source_checkpoint_sha256=source.checkpoint_sha256,
        epochs_completed=len(loss_history),
        loss_history=tuple(
            total + 0.5 * (latent + decoder)
            for total, latent, decoder in loss_history
        ),
        loss_history_definition="eq10_total_frobenius_objective_per_epoch",
        optimizer_state_present=True,
        random_state_present=True,
        device_used=str(selected_device),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint_objective="streaming_eq10_update_v1",
        update_loss_components=tuple(loss_history),
        training_resume_compatible=False,
    )
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(config.seed)
    random_state = {
        "torch_cpu": saved_rng_state["torch_cpu"],
        "shuffle_generator": shuffle_generator.get_state().cpu().clone(),
        "torch_cuda": saved_rng_state["torch_cuda"],
    }
    checkpoint_path = staging / "updated_ae.pt"
    atomic_torch_save(
        {
            "format": CHECKPOINT_FORMAT,
            "encoder_state_dict": {key: value.detach().cpu().clone() for key, value in model.E.state_dict().items()},
            "decoder_state_dict": {key: value.detach().cpu().clone() for key, value in model.D.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "random_state": random_state,
            # Preserve identity-keyed parameter slices for APs that may return
            # after the forgetting mechanism removes them from the vocabulary.
            "ap_parameter_archive": source.ap_parameter_archive,
            "manifest": ae_checkpoint_manifest.model_dump(mode="python"),
        },
        checkpoint_path,
    )
    arrays = tuple(
        _array_artifact(staging / name, name)
        for name in (
            "updated_rssi_dbm.npy",
            "source_rssi_dbm.npy",
            "observation_mask.npy",
            "value_provenance.npy",
        )
    )
    coord_map = label_positions
    fields = dict(
        updater_id=rm.updater_id,
        online_graph_id=rm.online_graph_id,
        dataset_id=rm.dataset_id,
        reference_batch_id=rm.reference_batch_id,
        update_batch_id=rm.update_batch_id,
        reference_prepared_sha256=rm.reference_prepared_sha256,
        update_prepared_sha256=rm.update_prepared_sha256,
        source_ae_checkpoint_sha256=source.checkpoint_sha256,
        updater_checkpoint_sha256=sha256_file(updater_state.directory / "updaters.pt"),
        bssid_vocabulary=rm.bssid_vocabulary,
        vocabulary_sha256=rm.vocabulary_sha256,
        normalization=rm.normalization,
        reference_scan_ids=ref_ids,
        labeled_reference_ids=rm.labeled_reference_ids,
        original_coordinates=tuple(coord_map.get(scan_id) for scan_id in ref_ids),
        update_scan_ids=update_ids,
        model_config_record=model.config,
        config=config,
        chunk_size=chunk_size,
        loss_history=tuple(loss_history),
        checkpoint_sha256=sha256_file(checkpoint_path),
        arrays=arrays,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    provisional = StreamingAEUpdateManifest.model_construct(
        update_id="streamae:" + "0" * 64, **fields
    )
    update_id = "streamae:" + canonical_json_digest(
        provisional.model_dump(mode="json", exclude={"update_id"})
    )
    manifest = StreamingAEUpdateManifest(update_id=update_id, **fields)
    atomic_write_json(manifest, staging / "manifest.json")
    atomic_write_json(
        {
            "format": "gufu_streaming_artifact_complete_v1",
            "identity": manifest.update_id,
            "manifest_sha256": sha256_file(staging / "manifest.json"),
        },
        staging / "COMPLETED.json",
    )
    if target.exists():
        raise FileExistsError(f"refusing to overwrite streaming AE update: {target}")
    os.replace(staging, target)
    return load_streaming_ae_update(target, reference_prepared, updater_state)


def load_streaming_ae_update(
    directory: str | Path,
    reference_prepared: PreparedDataset,
    updater_state: StreamingUpdaterState,
    *,
    verify_hashes: bool = True,
) -> StreamingAEUpdateState:
    root = Path(directory)
    manifest_path = root / "manifest.json"
    marker_path = root / "COMPLETED.json"
    checkpoint_path = root / "updated_ae.pt"
    if not manifest_path.is_file() or not marker_path.is_file() or not checkpoint_path.is_file():
        raise ValueError(f"streaming AE update is incomplete: {root}")
    manifest = StreamingAEUpdateManifest.model_validate_json(read_json_text(manifest_path))
    marker = json.loads(read_json_text(marker_path))
    if marker.get("identity") != manifest.update_id or marker.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("streaming AE update completion marker is inconsistent")
    if (
        manifest.updater_id != updater_state.manifest.updater_id
        or manifest.online_graph_id != updater_state.manifest.online_graph_id
        or manifest.reference_scan_ids != tuple(scan.scan_id for scan in reference_prepared.scans)
        or manifest.reference_prepared_sha256 != canonical_json_digest(reference_prepared)
        or manifest.bssid_vocabulary != reference_prepared.bssid_vocabulary
        or manifest.update_scan_ids != updater_state.manifest.update_ids
    ):
        raise ValueError("streaming AE update is incompatible with reference/updater lineage")
    if verify_hashes and sha256_file(checkpoint_path) != manifest.checkpoint_sha256:
        raise ValueError("streaming AE update checkpoint hash mismatch")
    payload = load_torch_state(checkpoint_path, map_location="cpu")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported streaming AE update checkpoint format")
    ae_manifest = AECheckpointManifest.model_validate(payload.get("manifest"))
    if (
        ae_manifest.checkpoint_objective != "streaming_eq10_update_v1"
        or ae_manifest.bssid_vocabulary != manifest.bssid_vocabulary
        or ae_manifest.vocabulary_sha256 != manifest.vocabulary_sha256
        or ae_manifest.normalization != manifest.normalization
        or ae_manifest.model != manifest.model_config_record
        or ae_manifest.training_resume_compatible
    ):
        raise ValueError("standard AE checkpoint metadata differs from streamed Eq. (10) state")
    model = AE(manifest.model_config_record.input_dim, config=manifest.model_config_record)
    model.E.load_state_dict(payload["encoder_state_dict"], strict=True)
    model.D.load_state_dict(payload["decoder_state_dict"], strict=True)
    model.eval()
    arrays = {}
    for item in manifest.arrays:
        path = root / item.name
        if not path.is_file() or path.stat().st_size != item.size_bytes:
            raise ValueError(f"streaming AE fingerprint array is missing: {item.name}")
        if verify_hashes and sha256_file(path) != item.sha256:
            raise ValueError(f"streaming AE fingerprint array digest mismatch: {item.name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != item.shape or array.dtype.name != item.dtype:
            raise ValueError(f"streaming AE fingerprint array shape/dtype mismatch: {item.name}")
        arrays[item.name] = array
    return StreamingAEUpdateState(
        directory=root,
        manifest=manifest,
        model=model,
        optimizer_state=payload["optimizer_state_dict"],
        updated_rssi_dbm=arrays["updated_rssi_dbm.npy"],
        source_rssi_dbm=arrays["source_rssi_dbm.npy"],
        observation_mask=arrays["observation_mask.npy"],
        value_provenance=arrays["value_provenance.npy"],
    )


def _transition_ae_chunk(
    transition_state, rows: np.ndarray, normalization: str
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(transition_state.updated_rssi_dbm[rows], dtype=np.float64)
    masks = np.asarray(transition_state.observation_mask[rows], dtype=np.uint8)
    origins = np.asarray(transition_state.value_provenance[rows], dtype=np.uint8)
    # Eq. (13) edge scores are graph weights, never AE input or targets.
    # Prior decoder estimates can be inputs, but only measured RSS is a target.
    usable = (masks != 0) | (origins == 1)
    matrix = np.empty_like(values, dtype=np.float32)
    for row in range(len(rows)):
        matrix[row] = raw_dbm_row_to_ae(
            values[row],
            tuple(bool(value) for value in usable[row]),
            normalization,
        )
    return matrix[:, None, :], (masks != 0)[:, None, :].astype(np.float32)


def _update_ae_chunk(
    update_prepared: PreparedDataset,
    rows: np.ndarray,
    target_vocabulary: tuple[str, ...],
    normalization: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Map actual update-batch measurements into the expanded AE vocabulary."""
    source_column = {
        bssid: column
        for column, bssid in enumerate(update_prepared.bssid_vocabulary)
    }
    target_sources = tuple(
        (target_column, source_column[bssid])
        for target_column, bssid in enumerate(target_vocabulary)
        if bssid in source_column
    )
    raw = np.full((len(rows), len(target_vocabulary)), -120.0, dtype=np.float64)
    observed = np.zeros((len(rows), len(target_vocabulary)), dtype=np.uint8)
    for output_row, scan_index in enumerate(rows):
        scan = update_prepared.scans[int(scan_index)]
        for target_column, source_idx in target_sources:
            if scan.observation_mask[source_idx]:
                raw[output_row, target_column] = scan.rss_dbm[source_idx]
                observed[output_row, target_column] = 1
    matrix = np.empty_like(raw, dtype=np.float32)
    for row in range(len(rows)):
        matrix[row] = raw_dbm_row_to_ae(
            raw[row], tuple(bool(value) for value in observed[row]), normalization
        )
    return matrix[:, None, :], observed[:, None, :].astype(np.float32)


def fine_tune_streaming_ae_after_ap_transition(
    transition_state,
    source_ae_update: StreamingAEUpdateState,
    updater_state: StreamingUpdaterState,
    migrated_ae_checkpoint_path: str | Path,
    output_directory: str | Path,
    config: FingerprintUpdateConfig,
    *,
    update_prepared: PreparedDataset,
    chunk_size: int = 256,
    device: str | torch.device = "cpu",
    log_every_epochs: int = DEFAULT_LOG_EVERY_K_EPOCHS,
) -> StreamingPostTransitionAEState:
    """Fine-tune the migrated AE from measured update scans and reference anchors.

    AP migration is applied to the trained AE weights before this call. Shared
    slices and returning AP archives therefore remain real learned state. The
    reconstruction term uses measured RSS from the unlabeled update batch,
    with missing cells masked out, plus an anchored reference reconstruction
    term. Eq. (13) scores remain graph weights and are excluded from AE inputs
    and targets. The fitted decoder fills unavailable values for all retained
    reference scans, while immutable labeled-reference latent targets remain
    the Eq. (9) anchor.
    """
    from ae import AECheckpointManifest

    if chunk_size < 1:
        raise ValueError("post-transition AE chunk size must be positive")
    log_every_epochs = validate_log_interval(log_every_epochs)
    tm = transition_state.manifest
    am = source_ae_update.manifest
    um = updater_state.manifest
    if tm.transition_id == "":
        raise ValueError("post-transition AE requires a completed transition artifact")
    if tm.ae_update_id != am.update_id or tm.ae_update_id != source_ae_update.manifest.update_id:
        raise ValueError("AP transition does not descend from the selected Eq. (10) AE result")
    if tm.reference_scan_ids != am.reference_scan_ids or tm.update_scan_ids != am.update_scan_ids:
        raise ValueError("post-transition AE changed scan identities or order")
    if update_prepared.config.batch_role != "update_batch":
        raise ValueError("post-transition AE requires the original unlabeled update batch")
    if update_prepared.config.dataset_id != tm.dataset_id:
        raise ValueError("post-transition AE update dataset identity differs")
    if update_prepared.config.batch_id != tm.update_batch_id:
        raise ValueError("post-transition AE update batch identity differs")
    update_ids = tuple(scan.scan_id for scan in update_prepared.scans)
    if update_ids != tm.update_scan_ids or update_ids != am.update_scan_ids or update_ids != um.update_ids:
        raise ValueError("post-transition AE update scan IDs/order differ from pipeline state")
    if update_prepared.config.normalization_mode != tm.normalization:
        raise ValueError("post-transition AE update normalization differs")
    if tm.vocabulary_before != am.bssid_vocabulary:
        raise ValueError("AP transition source vocabulary differs from Eq. (10) checkpoint")
    if tm.reference_scan_ids != um.reference_ids:
        raise ValueError("post-transition updater ID order differs from its online graph")
    if um.updater_id != am.updater_id or um.online_graph_id != am.online_graph_id:
        raise ValueError("post-transition AE updater model is from a different online graph")
    if tuple(updater_state.updated_reference_latents.shape) != (
        len(um.labeled_reference_ids), um.encoder_feature_dim
    ):
        raise ValueError("post-transition latent targets have an incompatible shape")

    migrated = load_ae_checkpoint(
        migrated_ae_checkpoint_path, device="cpu", dtype="float32"
    )
    migration = migrated.manifest.vocabulary_migration
    if (
        migrated.manifest.checkpoint_objective != "streaming_eq10_update_v1"
        or migration is None
        or migration.source_vocabulary != tm.vocabulary_before
        or migration.target_vocabulary != tm.vocabulary_after
        or migrated.manifest.bssid_vocabulary != tm.vocabulary_after
        or migrated.manifest.normalization != tm.normalization
        or migrated.manifest.source_checkpoint_sha256 != am.checkpoint_sha256
    ):
        raise ValueError("migrated AE checkpoint is incompatible with AP transition lineage")
    if migrated.model.config.latent_dim != um.encoder_feature_dim:
        raise ValueError("migrated AE latent width differs from Eq. (9) targets")
    reference_ids = tm.reference_scan_ids
    label_ids = um.labeled_reference_ids
    label_coords = tuple(
        coordinate
        for coordinate in tm.original_coordinates
        if coordinate is not None
    )
    coordinate_ids = tuple(
        scan_id
        for scan_id, coordinate in zip(reference_ids, tm.original_coordinates)
        if coordinate is not None
    )
    if coordinate_ids != label_ids or len(label_coords) != len(label_ids):
        raise ValueError("post-transition training labels must exactly match bootstrap labels")
    label_rows = np.asarray(
        [row for row, scan_id in enumerate(reference_ids) if scan_id in set(label_ids)],
        dtype=np.int64,
    )
    if tuple(reference_ids[int(row)] for row in label_rows) != label_ids:
        raise ValueError("post-transition latent targets do not preserve reference order")

    update_source_columns = {
        bssid: column
        for column, bssid in enumerate(update_prepared.bssid_vocabulary)
    }
    update_target_sources = tuple(
        (target_column, update_source_columns[bssid])
        for target_column, bssid in enumerate(tm.vocabulary_after)
        if bssid in update_source_columns
    )
    update_observed_readings = sum(
        sum(bool(scan.observation_mask[source_column]) for _, source_column in update_target_sources)
        for scan in update_prepared.scans
    )
    if update_observed_readings == 0:
        raise ValueError(
            "post-transition AE update batch has no measured RSS in the transitioned vocabulary"
        )
    update_signal_sha256 = canonical_json_digest(
        (
            update_prepared.bssid_vocabulary,
            tuple(
                (scan.scan_id, scan.rss_dbm, scan.observation_mask)
                for scan in update_prepared.scans
            ),
        )
    )

    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"requested CUDA device is unavailable: {selected_device}")
    cuda_devices = (
        [selected_device.index if selected_device.index is not None else torch.cuda.current_device()]
        if selected_device.type == "cuda"
        else []
    )
    model = migrated.model.to(selected_device, dtype=torch.float32)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.ae_learning_rate, weight_decay=config.ae_weight_decay
    )
    optimizer.load_state_dict(migrated.optimizer_state)
    for group in optimizer.param_groups:
        group["lr"] = config.ae_learning_rate
        group["weight_decay"] = config.ae_weight_decay

    target = Path(output_directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite post-transition AE state: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    history: list[tuple[float, float, float]] = []
    saved_rng_state: dict[str, object] = {}
    label_target_rows = updater_state.updated_reference_latents
    with torch.random.fork_rng(devices=cuda_devices):
        for epoch in range(1, config.ae_epochs + 1):
            epoch_seed = config.seed + 1303 + epoch
            model.train()
            _seed_ae_pass(epoch_seed, selected_device)
            squared = np.zeros((4,), dtype=np.float64)
            with torch.no_grad():
                for start in range(0, len(label_rows), chunk_size):
                    stop = min(start + chunk_size, len(label_rows))
                    rows = label_rows[start:stop]
                    x_values, reconstruction_mask = _transition_ae_chunk(
                        transition_state, rows, tm.normalization
                    )
                    x = _tensor(x_values, selected_device)
                    loss_mask = _tensor(reconstruction_mask, selected_device)
                    z_target = _tensor(label_target_rows[start:stop], selected_device).detach()
                    z, reconstruction = model(x)
                    target_reconstruction = model.D(z_target.unsqueeze(1))
                    residual = (reconstruction - x) * loss_mask
                    squared[0] += float(residual.double().square().sum().cpu())
                    squared[2] += float((z[:, 0, :] - z_target).double().square().sum().cpu())
                    squared[3] += float((target_reconstruction - reconstruction).double().square().sum().cpu())
                for start in range(0, len(update_prepared.scans), chunk_size):
                    stop = min(start + chunk_size, len(update_prepared.scans))
                    rows = np.arange(start, stop, dtype=np.int64)
                    x_values, reconstruction_mask = _update_ae_chunk(
                        update_prepared, rows, tm.vocabulary_after, tm.normalization
                    )
                    x = _tensor(x_values, selected_device)
                    loss_mask = _tensor(reconstruction_mask, selected_device)
                    _, reconstruction = model(x)
                    residual = (reconstruction - x) * loss_mask
                    squared[1] += float(residual.double().square().sum().cpu())
            reference_reconstruction_norm = float(np.sqrt(squared[0]))
            update_reconstruction_norm = float(np.sqrt(squared[1]))
            norms = np.asarray(
                (
                    0.5 * (reference_reconstruction_norm + update_reconstruction_norm),
                    np.sqrt(squared[2]),
                    np.sqrt(squared[3]),
                ),
                dtype=np.float64,
            )

            optimizer.zero_grad(set_to_none=True)
            model.train()
            _seed_ae_pass(epoch_seed, selected_device)
            for start in range(0, len(label_rows), chunk_size):
                stop = min(start + chunk_size, len(label_rows))
                rows = label_rows[start:stop]
                x_values, reconstruction_mask = _transition_ae_chunk(
                    transition_state, rows, tm.normalization
                )
                x = _tensor(x_values, selected_device)
                loss_mask = _tensor(reconstruction_mask, selected_device)
                z_target = _tensor(label_target_rows[start:stop], selected_device).detach()
                z, reconstruction = model(x)
                target_reconstruction = model.D(z_target.unsqueeze(1))
                objective = reconstruction.sum() * 0.0
                if reference_reconstruction_norm > 0.0:
                    residual = (reconstruction - x) * loss_mask
                    objective = objective + 0.5 * residual.square().sum() / reference_reconstruction_norm
                if norms[1] > 0.0:
                    objective = objective + 0.5 * (z[:, 0, :] - z_target).square().sum() / (2.0 * float(norms[1]))
                if norms[2] > 0.0:
                    objective = objective + 0.5 * (target_reconstruction - reconstruction).square().sum() / (2.0 * float(norms[2]))
                objective.backward()
            for start in range(0, len(update_prepared.scans), chunk_size):
                stop = min(start + chunk_size, len(update_prepared.scans))
                rows = np.arange(start, stop, dtype=np.int64)
                x_values, reconstruction_mask = _update_ae_chunk(
                    update_prepared, rows, tm.vocabulary_after, tm.normalization
                )
                x = _tensor(x_values, selected_device)
                loss_mask = _tensor(reconstruction_mask, selected_device)
                _, reconstruction = model(x)
                if update_reconstruction_norm > 0.0:
                    residual = (reconstruction - x) * loss_mask
                    objective = 0.5 * residual.square().sum() / update_reconstruction_norm
                    objective.backward()
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            if not gradients or any(not torch.isfinite(gradient).all().item() for gradient in gradients):
                raise FloatingPointError("post-transition Eq. (10) gradients are missing or non-finite")
            if not any(torch.count_nonzero(gradient).item() for gradient in gradients):
                raise RuntimeError("post-transition Eq. (10) produced no parameter gradient")
            optimizer.step()
            history.append(tuple(float(value) for value in norms))
            log_training_epoch(
                "Streaming post-transition AE Eq. (10) + measured update RSS",
                epoch,
                config.ae_epochs,
                {
                    "balanced_reconstruction_frobenius": float(norms[0]),
                    "reference_reconstruction_frobenius": reference_reconstruction_norm,
                    "update_reconstruction_frobenius": update_reconstruction_norm,
                    "latent_frobenius": float(norms[1]),
                    "decoder_frobenius": float(norms[2]),
                    "labeled_reference_scans": len(label_rows),
                    "unlabeled_update_scans": len(update_prepared.scans),
                    "update_observed_rssi_targets": update_observed_readings,
                    "APs": len(tm.vocabulary_after),
                    "chunk_size": chunk_size,
                },
                every_k_epochs=log_every_epochs,
            )
        saved_rng_state = {
            "torch_cpu": torch.get_rng_state().cpu().clone(),
            "torch_cuda": torch.cuda.get_rng_state_all() if selected_device.type == "cuda" else None,
        }

    model.eval()
    n_reference, n_ap = len(reference_ids), len(tm.vocabulary_after)
    updated = _open(staging / "updated_rssi_dbm.npy", (n_reference, n_ap))
    masks = _open(staging / "observation_mask.npy", (n_reference, n_ap), dtype="uint8")
    provenance = _open(staging / "value_provenance.npy", (n_reference, n_ap), dtype="uint8")
    updated[:] = transition_state.updated_rssi_dbm[:]
    masks[:] = transition_state.observation_mask[:]
    provenance[:] = transition_state.value_provenance[:]
    label_target_row = {
        int(reference_row): target_row
        for target_row, reference_row in enumerate(label_rows)
    }
    with torch.inference_mode():
        for start in range(0, n_reference, chunk_size):
            stop = min(start + chunk_size, n_reference)
            rows = np.arange(start, stop, dtype=np.int64)
            x_values, _ = _transition_ae_chunk(
                transition_state, rows, tm.normalization
            )
            x = _tensor(x_values, selected_device)
            latent = model.E(x)[:, 0, :]
            for offset, row in enumerate(rows):
                target_row = label_target_row.get(int(row))
                if target_row is not None:
                    latent[offset] = _tensor(
                        label_target_rows[target_row], selected_device
                    )
            normalized = model.D(latent.unsqueeze(1))[:, 0, :].cpu().numpy()
            for offset, row_value in enumerate(rows):
                row = int(row_value)
                decoded = np.asarray(
                    [
                        ae_value_to_raw_dbm(float(value), tm.normalization)
                        for value in normalized[offset]
                    ],
                    dtype=np.float32,
                )
                if row in label_target_row:
                    updated[row] = decoded
                    provenance[row, :] = 1
                else:
                    unavailable = np.asarray(masks[row] == 0)
                    updated[row, unavailable] = decoded[unavailable]
                    provenance[row, unavailable] = 1
    for array in (updated, masks, provenance):
        if isinstance(array, np.memmap):
            array.flush()

    total_history = tuple(a + 0.5 * (b + c) for a, b, c in history)
    training_config = migrated.manifest.training_config.model_copy(
        update={
            "epochs": config.ae_epochs,
            "batch_size": chunk_size,
            "learning_rate": config.ae_learning_rate,
            "weight_decay": config.ae_weight_decay,
            "seed": config.seed,
            "shuffle": False,
            "optimizer_betas": (0.9, 0.999),
            "optimizer_epsilon": 1.0e-8,
            "amsgrad": False,
            "dtype": "float32",
            "loss": "batch_frobenius_norm",
        }
    )
    transition_prepared_identity = canonical_json_digest(
        {
            "transition_id": tm.transition_id,
            "reference_scan_ids": reference_ids,
            "vocabulary": tm.vocabulary_after,
            "rssi_sha256": sha256_file(transition_state.directory / "updated_rssi_dbm.npy"),
        }
    )
    training_dataset = DatasetIdentity(
        dataset_id=tm.dataset_id,
        batch_id=tm.reference_batch_id,
        batch_role="bootstrap_reference",
        prepared_sha256=transition_prepared_identity,
        scan_ids=reference_ids,
        scan_count=len(reference_ids),
    )
    updated_migration = migration.model_copy(
        update={
            "transition_batch_id": tm.update_batch_id,
            "finetune_epochs": len(history),
            "finetune_seed": config.seed + 1303,
            "finetune_loss_history": total_history,
        }
    )
    checkpoint_manifest = migrated.manifest.model_copy(
        update={
            "training_dataset": training_dataset,
            "training_config": AECheckpointTrainingConfig.model_validate(training_config),
            "initialization_mode": "vocabulary_migration",
            "source_checkpoint_sha256": migrated.checkpoint_sha256,
            "vocabulary_migration": updated_migration,
            "epochs_completed": len(history),
            "loss_history": total_history,
            "loss_history_definition": "eq10_total_frobenius_objective_per_epoch",
            "checkpoint_objective": "streaming_eq10_update_v1",
            "update_loss_components": tuple(history),
            "training_resume_compatible": False,
            "device_used": str(selected_device),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(config.seed)
    random_state = {
        "torch_cpu": saved_rng_state["torch_cpu"],
        "shuffle_generator": random_generator.get_state().cpu().clone(),
        "torch_cuda": saved_rng_state["torch_cuda"],
    }
    checkpoint_path = staging / "post_transition_ae.pt"
    payload = load_torch_state(migrated.path, map_location="cpu")
    payload.update(
        {
            "encoder_state_dict": {key: value.detach().cpu().clone() for key, value in model.E.state_dict().items()},
            "decoder_state_dict": {key: value.detach().cpu().clone() for key, value in model.D.state_dict().items()},
            "optimizer_state_dict": optimizer.state_dict(),
            "random_state": random_state,
            "manifest": checkpoint_manifest.model_dump(mode="python"),
        }
    )
    atomic_torch_save(payload, checkpoint_path)
    arrays = tuple(
        _array_artifact(staging / name, name)
        for name in ("updated_rssi_dbm.npy", "observation_mask.npy", "value_provenance.npy")
    )
    fields = dict(
        transition_id=tm.transition_id,
        source_ae_update_id=am.update_id,
        source_updater_id=um.updater_id,
        source_migrated_checkpoint_sha256=migrated.checkpoint_sha256,
        checkpoint_sha256=sha256_file(checkpoint_path),
        dataset_id=tm.dataset_id,
        reference_batch_id=tm.reference_batch_id,
        update_batch_id=tm.update_batch_id,
        update_scan_ids=update_ids,
        update_signal_sha256=update_signal_sha256,
        update_observed_readings=update_observed_readings,
        reference_scan_ids=reference_ids,
        original_coordinates=tm.original_coordinates,
        labeled_reference_ids=label_ids,
        bssid_vocabulary=tm.vocabulary_after,
        vocabulary_sha256=canonical_json_digest(tm.vocabulary_after),
        normalization=tm.normalization,
        model_config_record=model.config,
        config=config,
        loss_history=tuple(history),
        arrays=arrays,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    provisional = StreamingPostTransitionAEManifest.model_construct(
        post_update_id="streampostae:" + "0" * 64, **fields
    )
    post_id = "streampostae:" + canonical_json_digest(
        provisional.model_dump(mode="json", exclude={"post_update_id"})
    )
    manifest = StreamingPostTransitionAEManifest(post_update_id=post_id, **fields)
    atomic_write_json(manifest, staging / "manifest.json")
    atomic_write_json(
        {
            "format": "gufu_streaming_artifact_complete_v1",
            "identity": manifest.post_update_id,
            "manifest_sha256": sha256_file(staging / "manifest.json"),
        },
        staging / "COMPLETED.json",
    )
    if target.exists():
        raise FileExistsError(f"refusing to overwrite post-transition AE state: {target}")
    os.replace(staging, target)
    checked = load_ae_checkpoint(target / "post_transition_ae.pt", device=selected_device)
    if checked.checkpoint_sha256 != manifest.checkpoint_sha256:
        raise ValueError("post-transition AE checkpoint failed its recorded digest")
    return load_streaming_post_transition_ae(target, device=selected_device)


def load_streaming_post_transition_ae(
    directory: str | Path,
    *,
    device: str | torch.device = "cpu",
    verify_hashes: bool = True,
) -> StreamingPostTransitionAEState:
    root = Path(directory)
    manifest_path = root / "manifest.json"
    marker_path = root / "COMPLETED.json"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise ValueError(f"post-transition AE artifact is incomplete: {root}")
    manifest = StreamingPostTransitionAEManifest.model_validate_json(read_json_text(manifest_path))
    marker = json.loads(read_json_text(marker_path))
    if marker.get("identity") != manifest.post_update_id or marker.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("post-transition AE completion marker is inconsistent")
    checkpoint_path = root / "post_transition_ae.pt"
    if not checkpoint_path.is_file() or (verify_hashes and sha256_file(checkpoint_path) != manifest.checkpoint_sha256):
        raise ValueError("post-transition AE checkpoint is missing or corrupt")
    loaded = load_ae_checkpoint(checkpoint_path, device=device, dtype="float32")
    if loaded.manifest.bssid_vocabulary != manifest.bssid_vocabulary or loaded.manifest.model != manifest.model_config_record:
        raise ValueError("post-transition AE checkpoint metadata mismatch")
    arrays = {}
    for item in manifest.arrays:
        path = root / item.name
        if not path.is_file() or path.stat().st_size != item.size_bytes:
            raise ValueError(f"post-transition AE array is missing or truncated: {item.name}")
        if verify_hashes and sha256_file(path) != item.sha256:
            raise ValueError(f"post-transition AE array hash mismatch: {item.name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != item.shape or array.dtype.name != item.dtype:
            raise ValueError(f"post-transition AE array shape/dtype mismatch: {item.name}")
        arrays[item.name] = array
    if np.any(arrays["value_provenance.npy"] == 2):
        raise ValueError(
            "post-transition AE artifact contains graph edge scores as RSS provenance"
        )
    return StreamingPostTransitionAEState(
        directory=root,
        manifest=manifest,
        model=loaded.model,
        updated_rssi_dbm=arrays["updated_rssi_dbm.npy"],
        observation_mask=arrays["observation_mask.npy"],
        value_provenance=arrays["value_provenance.npy"],
    )


__all__ = [
    "StreamingAEUpdateState",
    "StreamingPostTransitionAEManifest",
    "StreamingPostTransitionAEState",
    "StreamingUpdaterManifest",
    "StreamingUpdaterState",
    "fine_tune_streaming_ae_after_ap_transition",
    "load_streaming_updater_warm_start",
    "load_streaming_ae_update",
    "load_streaming_post_transition_ae",
    "load_streaming_updater",
    "train_streaming_ae_update",
    "train_streaming_updaters",
]
