"""Restartable, disk-backed orchestration for ``RUN_MODE='lp_streaming'``.

The module calls only the streamed graph/LP/updater APIs. Importing it is
side-effect free; directory creation and model work happen only in explicit
functions or under the CLI main guard.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ae import (
    LoadedAECheckpoint,
    encode_prepared_dataset,
    load_ae_checkpoint,
    migrate_ae_checkpoint_vocabulary,
)
from batch_transition import TransitionConfig, project_prepared_batch_to_vocabulary
from fingerprint_update import FingerprintUpdateConfig
from graph_build import GraphBuildConfig
from gufu_artifacts import (
    atomic_write_json,
    canonical_json_digest,
    read_json_text,
    sha256_file,
)
from gufu_data import (
    PreprocessingConfig,
    PreparedDataset,
    load_prepared_dataset,
    prepare_dataset_files,
    save_prepared_dataset,
)
from lp import LinkPredictionConfig
from lp_streaming import (
    StreamingLPInputManifest,
    load_streaming_lp_input,
    load_streaming_lp_result,
    predict_edges_streaming,
    save_streaming_lp_input,
)
from sage_ve import (
    GNNTrainingConfig,
    LoadedGNNCheckpoint,
    load_streaming_gnn_checkpoint,
)
from streaming_fingerprint_update import (
    StreamingUpdaterState,
    fine_tune_streaming_ae_after_ap_transition,
    load_streaming_updater_warm_start,
    train_streaming_ae_update,
    train_streaming_updaters,
)
from streaming_gnn import (
    StreamingGNNState,
    load_streaming_gnn_state,
    train_streaming_gnn,
)
from streaming_transition import (
    StreamingReferenceTransitionState,
    apply_streaming_ap_delta_to_reference,
    load_streaming_reference_transition,
    materialize_streaming_reference_prepared,
)
from streaming_update import (
    StreamingOnlineGraphManifest,
    build_streaming_online_graph,
    build_streaming_reference_graph_after_transition,
    infer_streaming_ap_graph_changes,
    load_streaming_online_graph,
    save_streaming_online_lp_input,
)
from training_progress import DEFAULT_LOG_EVERY_K_EPOCHS, validate_log_interval


PIPELINE_FORMAT = "gufu_streaming_pipeline_v1"
SNAPSHOT_FORMAT = "gufu_streaming_batch_snapshot_v1"


class StreamingPipelineConfig(BaseModel):
    """Serializable choices shared by notebook and fresh-process execution."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    dataset_id: str
    preprocessing: PreprocessingConfig
    graph_build: GraphBuildConfig
    gnn_training: GNNTrainingConfig
    link_prediction: LinkPredictionConfig
    fingerprint_update: FingerprintUpdateConfig
    transition: TransitionConfig
    device: str = "cpu"
    seed: int = 0
    gnn_chunk_size: int = Field(default=4096, ge=1)
    updater_chunk_size: int = Field(default=4096, ge=1)
    ae_chunk_size: int = Field(default=256, ge=1)
    encoder_batch_size: int = Field(default=128, ge=1)
    log_every_epochs: int = Field(default=DEFAULT_LOG_EVERY_K_EPOCHS, ge=1)
    rssi_pair_num_workers: int = Field(default=4, ge=1)

    @model_validator(mode="after")
    def compatible_stages(self):
        if self.preprocessing.dataset_id != self.dataset_id:
            raise ValueError("streaming pipeline preprocessing dataset_id differs")
        if self.graph_build.virtual_edge_sigma != self.fingerprint_update.virtual_edge_sigma:
            raise ValueError("graph/updater virtual-edge thresholds must match")
        if self.transition.link_prediction != self.link_prediction:
            raise ValueError("batch and AP-transition LP configurations must match")
        if self.log_every_epochs < 1:
            raise ValueError("log_every_epochs must be positive")
        return self


class SnapshotFile(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class StageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    name: str
    status: Literal["complete"] = "complete"
    artifact: str | None = None


class SnapshotCurrentState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    reference_prepared: str
    reference_value_provenance: str
    ae_checkpoint: str
    reference_graph: str
    reference_gnn: str
    reference_lp_input: str
    latest_updaters: str


class StreamingBatchSnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_streaming_batch_snapshot_v1"] = SNAPSHOT_FORMAT
    snapshot_id: str = Field(pattern=r"^streambatch:[0-9a-f]{64}$")
    sequence_number: int = Field(ge=1)
    parent_snapshot_id: str | None = Field(
        default=None, pattern=r"^streambatch:[0-9a-f]{64}$"
    )
    dataset_id: str
    batch_id: str
    source_input_paths: tuple[str, ...]
    source_input_sha256: tuple[str, ...]
    source_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    update_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_reference_graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    resulting_reference_graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    source_gnn_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resulting_gnn_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_ae_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    resulting_ae_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vocabulary_before: tuple[str, ...]
    vocabulary_after: tuple[str, ...]
    reference_scan_ids: tuple[str, ...]
    update_scan_ids: tuple[str, ...]
    stage_records: tuple[StageRecord, ...]
    metrics: tuple[tuple[str, float], ...]
    pipeline_config: StreamingPipelineConfig
    current_state: SnapshotCurrentState
    artifacts: tuple[SnapshotFile, ...]

    @model_validator(mode="after")
    def valid_identity(self):
        if len(self.source_input_paths) != len(self.source_input_sha256):
            raise ValueError("snapshot source paths and digests must align")
        if len(set(self.reference_scan_ids)) != len(self.reference_scan_ids):
            raise ValueError("snapshot reference IDs must be unique")
        if len(set(self.update_scan_ids)) != len(self.update_scan_ids):
            raise ValueError("snapshot update IDs must be unique")
        if set(self.reference_scan_ids) & set(self.update_scan_ids):
            raise ValueError("temporary update IDs collide with permanent reference IDs")
        if len(set(self.vocabulary_after)) != len(self.vocabulary_after):
            raise ValueError("snapshot vocabulary contains duplicate AP identities")
        expected = "streambatch:" + canonical_json_digest(
            self.model_dump(mode="json", exclude={"snapshot_id"})
        )
        if self.snapshot_id != expected:
            raise ValueError("snapshot identity does not match its state/index manifest")
        names = tuple(item.path for item in self.artifacts)
        if len(names) != len(set(names)):
            raise ValueError("snapshot artifact index contains duplicate paths")
        return self


@dataclass(frozen=True)
class StreamingPipelineResult:
    root: Path
    latest_snapshot_directory: Path | None
    latest_snapshot: StreamingBatchSnapshotManifest | None
    completed_batch_ids: tuple[str, ...]
    skipped_completed_batch_ids: tuple[str, ...]


@dataclass(frozen=True)
class _RuntimeState:
    prepared: PreparedDataset
    ae_checkpoint_path: Path
    gnn_state: LoadedGNNCheckpoint | StreamingGNNState
    gnn_state_graph_directory: Path | None
    lp_input_directory: Path
    snapshot_id: str | None
    sequence_number: int
    updater_directory: Path | None


def _file_sha(path: Path) -> str:
    return sha256_file(path)


def _source_inputs(paths: tuple[Path, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return tuple(str(path.resolve()) for path in paths), tuple(_file_sha(path) for path in paths)


def _natural_key(path: Path):
    return tuple(int(part) if part.isdigit() else part.casefold() for part in re.split(r"(\d+)", path.name))


def select_update_paths(
    *,
    update_paths: tuple[str | Path, ...] | None = None,
    update_directory: str | Path | None = None,
    pattern: str = "fingerprint[1-9]*.txt",
    bootstrap_paths: tuple[str | Path, ...] = (),
) -> tuple[Path, ...]:
    """Resolve explicit paths in caller order or naturally sort a directory."""
    if update_paths is not None and update_directory is not None:
        raise ValueError("choose explicit update_paths or update_directory")
    if update_paths is None:
        if update_directory is None:
            return ()
        root = Path(update_directory)
        paths = tuple(sorted(root.glob(pattern), key=_natural_key))
    else:
        paths = tuple(Path(item) for item in update_paths)
    excluded = {Path(item).resolve() for item in bootstrap_paths}
    result = tuple(path.resolve() for path in paths if path.resolve() not in excluded)
    if len(set(result)) != len(result):
        raise ValueError("streaming update input paths contain duplicates")
    missing = tuple(path for path in result if not path.is_file())
    if missing:
        raise FileNotFoundError(f"streaming update input is missing: {missing[0]}")
    return result


def _copy_tree(source: Path, target: Path) -> Path:
    if target.exists():
        raise FileExistsError(f"refusing to overwrite copied pipeline artifact: {target}")
    shutil.copytree(source, target)
    return target


def _snapshot_manifest(snapshot_directory: Path) -> StreamingBatchSnapshotManifest:
    manifest_path = snapshot_directory / "manifest.json"
    marker_path = snapshot_directory / "COMPLETED.json"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise ValueError(f"streaming batch snapshot is incomplete: {snapshot_directory}")
    manifest = StreamingBatchSnapshotManifest.model_validate_json(read_json_text(manifest_path))
    marker = json.loads(read_json_text(marker_path))
    if (
        marker.get("snapshot_id") != manifest.snapshot_id
        or marker.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("streaming snapshot completion marker disagrees with manifest")
    for item in manifest.artifacts:
        path = snapshot_directory / item.path
        if not path.is_file() or path.stat().st_size != item.size_bytes:
            raise ValueError(f"snapshot artifact is missing or truncated: {item.path}")
        if sha256_file(path) != item.sha256:
            raise ValueError(f"snapshot artifact digest mismatch: {item.path}")
    return manifest


def _recover_latest(root: Path) -> tuple[Path, StreamingBatchSnapshotManifest] | None:
    snapshots = root / "snapshots"
    if not snapshots.exists():
        return None
    complete = []
    for directory in snapshots.iterdir():
        if not directory.is_dir() or directory.name.startswith(".staging-"):
            continue
        if (directory / "COMPLETED.json").is_file():
            complete.append((directory, _snapshot_manifest(directory)))
    if not complete:
        return None
    complete.sort(key=lambda pair: pair[1].sequence_number)
    sequences = [manifest.sequence_number for _directory, manifest in complete]
    if sequences != list(range(1, len(sequences) + 1)):
        raise ValueError("completed streaming snapshots contain a sequence gap or fork")
    for index in range(1, len(complete)):
        if complete[index][1].parent_snapshot_id != complete[index - 1][1].snapshot_id:
            raise ValueError("completed streaming snapshots do not form one parent chain")
    directory, manifest = complete[-1]
    atomic_write_json(
        {
            "format": "gufu_streaming_latest_v1",
            "snapshot_id": manifest.snapshot_id,
            "directory": str(directory.relative_to(root)),
            "manifest_sha256": sha256_file(directory / "manifest.json"),
        },
        root / "LATEST.json",
        overwrite=True,
    )
    return directory, manifest


def _load_runtime_from_snapshot(
    root: Path, directory: Path, manifest: StreamingBatchSnapshotManifest, *, device: str
) -> _RuntimeState:
    current = manifest.current_state
    prepared = load_prepared_dataset(
        directory / current.reference_prepared,
        value_provenance_path=directory / current.reference_value_provenance,
    )
    ae_path = directory / current.ae_checkpoint
    loaded_ae = load_ae_checkpoint(ae_path, prepared, device=device)
    if loaded_ae.checkpoint_sha256 != manifest.resulting_ae_checkpoint_sha256:
        raise ValueError("latest snapshot AE checkpoint identity is inconsistent")
    graph_directory = directory / current.reference_graph
    graph = load_streaming_online_graph(graph_directory)
    gnn_directory = directory / current.reference_gnn
    gnn = load_streaming_gnn_state(gnn_directory, graph_directory, device=device)
    if gnn.manifest.checkpoint_sha256 != manifest.resulting_gnn_checkpoint_sha256:
        raise ValueError("latest snapshot GNN checkpoint identity is inconsistent")
    lp_directory = directory / current.reference_lp_input
    lp_manifest, _ = load_streaming_lp_input(lp_directory)
    if (
        lp_manifest.graph_id != graph.manifest.update_graph_id
        or lp_manifest.gnn_checkpoint_sha256 != gnn.manifest.checkpoint_sha256
        or lp_manifest.source_prepared_sha256 != canonical_json_digest(prepared)
        or lp_manifest.bssid_vocabulary != prepared.bssid_vocabulary
    ):
        raise ValueError("latest LP input is incompatible with reference graph/model/data")
    return _RuntimeState(
        prepared=prepared,
        ae_checkpoint_path=ae_path,
        gnn_state=gnn,
        gnn_state_graph_directory=graph_directory,
        lp_input_directory=lp_directory,
        snapshot_id=manifest.snapshot_id,
        sequence_number=manifest.sequence_number,
        updater_directory=directory / current.latest_updaters,
    )


def _validate_bootstrap(
    prepared: PreparedDataset,
    ae_checkpoint_path: str | Path,
    gnn_checkpoint: LoadedGNNCheckpoint,
    lp_input_directory: str | Path,
    lp_result_directory: str | Path | None,
    config: StreamingPipelineConfig,
) -> tuple[LoadedAECheckpoint, StreamingLPInputManifest, object | None]:
    if prepared.config.dataset_id != config.dataset_id:
        raise ValueError("bootstrap prepared data uses a different dataset ID")
    if prepared.config.batch_role != "bootstrap_reference":
        raise ValueError("streaming pipeline bootstrap must be bootstrap_reference")
    ae = load_ae_checkpoint(ae_checkpoint_path, prepared, device=config.device)
    input_manifest, _ = load_streaming_lp_input(lp_input_directory)
    ids = tuple(scan.scan_id for scan in prepared.scans)
    configured_graph = config.graph_build.model_dump(mode="json")
    input_graph = input_manifest.graph_build_config.model_dump(mode="json")
    configured_bounds = configured_graph.pop("pseudo_location_bounds")
    input_bounds = input_graph.pop("pseudo_location_bounds")
    graph_config_matches = (
        configured_graph == input_graph
        and (configured_bounds is None or configured_bounds == input_bounds)
        and input_bounds is not None
    )
    if (
        input_manifest.batch_role != "bootstrap_reference"
        or input_manifest.sample_node_ids != ids
        or input_manifest.source_prepared_sha256 != canonical_json_digest(prepared)
        or input_manifest.bssid_vocabulary != prepared.bssid_vocabulary
        or input_manifest.encoder_checkpoint_sha256 != ae.checkpoint_sha256
        or input_manifest.gnn_checkpoint_sha256 != gnn_checkpoint.checkpoint_sha256
        or input_manifest.graph_id != gnn_checkpoint.manifest.graph_id
        or not graph_config_matches
    ):
        raise ValueError("bootstrap LP input does not match prepared/AE/GNN/config lineage")
    result = None
    if lp_result_directory is not None:
        result = load_streaming_lp_result(
            lp_result_directory, input_directory=lp_input_directory
        )
        if (
            result.manifest.config != config.link_prediction
            or result.manifest.outcome == "failed"
        ):
            raise ValueError("bootstrap streamed LP result is failed or config-incompatible")
    return ae, input_manifest, result


def _load_existing_completed(
    root: Path,
) -> dict[tuple[str, str], tuple[Path, StreamingBatchSnapshotManifest]]:
    result = {}
    snapshots = root / "snapshots"
    if not snapshots.exists():
        return result
    for directory in snapshots.iterdir():
        if not directory.is_dir() or directory.name.startswith(".staging-"):
            continue
        if not (directory / "COMPLETED.json").is_file():
            continue
        manifest = _snapshot_manifest(directory)
        for source_path, source_sha in zip(
            manifest.source_input_paths, manifest.source_input_sha256
        ):
            key = (source_path, source_sha)
            if key in result:
                raise ValueError("one input file appears in multiple completed batch snapshots")
            result[key] = (directory, manifest)
    return result


def _stage_call(observer, name: str, staging: Path, records: list[StageRecord], artifact=None):
    records.append(StageRecord(name=name, artifact=artifact))
    if observer is not None:
        observer(name, staging)


def _process_batch(
    root: Path,
    state: _RuntimeState,
    input_path: Path,
    config: StreamingPipelineConfig,
    *,
    source_root: str | Path | None,
    initial_lp_result_directory: str | Path | None,
    stage_observer: Callable[[str, Path], None] | None,
    progress_callback: Callable[[str], None] | None,
) -> tuple[Path, StreamingBatchSnapshotManifest]:
    seq = state.sequence_number + 1
    batch_id = input_path.stem
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", batch_id):
        raise ValueError(f"update filename stem is not a valid batch ID: {batch_id}")
    snapshot_root = root / "snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".staging-{seq:04d}-{batch_id}-", dir=snapshot_root)
    )
    if progress_callback is not None:
        progress_callback(
            f"[pipeline] batch {batch_id}: starting snapshot staging at {staging.name}"
        )
    records: list[StageRecord] = []

    # The first snapshot captures the external cell-6 boundary so it can be
    # recovered after its parent notebook session is gone. Later snapshots
    # retain the parent snapshot and reference its immutable LP bundle.
    reference_lp_input = state.lp_input_directory
    if state.snapshot_id is None:
        reference_lp_input = _copy_tree(
            state.lp_input_directory, staging / "starting_reference_lp_input"
        )
    source_ae_checkpoint = staging / "source_ae_checkpoint.pt"
    shutil.copy2(state.ae_checkpoint_path, source_ae_checkpoint)
    source_prepared_path = staging / "reference_prepared_before.json"
    source_provenance_path = staging / "reference_value_provenance_before.json"
    save_prepared_dataset(
        state.prepared,
        source_prepared_path,
        value_provenance_path=(
            source_provenance_path
            if any(scan.value_provenance is not None for scan in state.prepared.scans)
            else None
        ),
    )
    source_gnn_checkpoint = staging / "source_gnn_checkpoint.pt"
    if isinstance(state.gnn_state, StreamingGNNState):
        gnn_source_path = state.gnn_state.directory / "gnn_state.pt"
    else:
        gnn_source_path = state.gnn_state.checkpoint_path
    shutil.copy2(gnn_source_path, source_gnn_checkpoint)

    if initial_lp_result_directory is not None and state.snapshot_id is None:
        if progress_callback is not None:
            progress_callback(
                f"[pipeline] batch {batch_id}: reusing the supplied bootstrap LP result"
            )
        lp_result_dir = _copy_tree(
            Path(initial_lp_result_directory), staging / "base_link_prediction_result"
        )
        base_result = load_streaming_lp_result(
            lp_result_dir, input_directory=reference_lp_input
        )
    else:
        lp_result_dir = staging / "base_link_prediction_result"
        if progress_callback is not None:
            progress_callback(
                f"[pipeline] batch {batch_id}: scoring links for the current reference graph"
            )
        base_result = predict_edges_streaming(
            reference_lp_input,
            lp_result_dir,
            config.link_prediction,
            progress_callback=progress_callback,
        )
    if base_result.manifest.outcome == "failed":
        raise RuntimeError("reference LP did not converge; refusing to train on failed graph state")
    _stage_call(stage_observer, "base_link_prediction", staging, records, "base_link_prediction_result")

    update_preprocessing = config.preprocessing.model_copy(
        update={
            "batch_id": batch_id,
            "batch_role": "update_batch",
            "base_bssid_vocabulary": state.prepared.bssid_vocabulary,
        }
    )
    if progress_callback is not None:
        progress_callback(
            f"[pipeline] batch {batch_id}: preparing incoming scans from {input_path}"
        )
    full_update = prepare_dataset_files(
        [input_path], update_preprocessing, source_root=source_root
    )
    if progress_callback is not None:
        progress_callback(
            f"[pipeline] batch {batch_id}: prepared {len(full_update.scans):,} "
            f"incoming scans with {len(full_update.bssid_vocabulary):,} observed APs"
        )
    reference_ids = {scan.scan_id for scan in state.prepared.scans}
    update_ids = tuple(scan.scan_id for scan in full_update.scans)
    if len(set(update_ids)) != len(update_ids) or reference_ids.intersection(update_ids):
        raise ValueError("incoming scans are duplicate or collide with retained reference IDs")
    full_update_path = staging / "input_prepared_batch.json"
    save_prepared_dataset(full_update, full_update_path)
    shared_update = project_prepared_batch_to_vocabulary(
        full_update, state.prepared.bssid_vocabulary
    )
    shared_update_path = staging / "shared_vocabulary_update.json"
    save_prepared_dataset(shared_update, shared_update_path)
    _stage_call(stage_observer, "update_shared_preprocessing", staging, records, "input_prepared_batch.json")

    update_encoder_path = staging / "update_encoder_features.json"
    if progress_callback is not None:
        progress_callback(
            f"[pipeline] batch {batch_id}: encoding incoming scans with the shared AE"
        )
    update_encoder = encode_prepared_dataset(
        shared_update,
        source_ae_checkpoint,
        update_encoder_path,
        batch_size=config.encoder_batch_size,
        device=config.device,
    )
    if progress_callback is not None:
        progress_callback(
            f"[pipeline] batch {batch_id}: AE encoding complete for "
            f"{len(update_encoder.scan_ids):,} scans"
        )
    _stage_call(stage_observer, "update_shared_ae_encoding", staging, records, "update_encoder_features.json")

    online_graph_path = staging / "online_graph"
    if progress_callback is not None:
        progress_callback(
            f"[pipeline] batch {batch_id}: starting online graph construction "
            f"for {len(state.prepared.scans) + len(shared_update.scans):,} samples"
        )
    online_graph_manifest = build_streaming_online_graph(
        reference_lp_input,
        lp_result_dir,
        shared_update,
        update_encoder,
        online_graph_path,
        virtual_edge_sigma=config.graph_build.virtual_edge_sigma,
        num_workers=config.rssi_pair_num_workers,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(
            f"[pipeline] batch {batch_id}: online graph ready: "
            f"nodes={len(online_graph_manifest.node_ids):,}, "
            f"radio edges={online_graph_manifest.radio_edge_count:,}, "
            f"virtual edges={online_graph_manifest.virtual_edge_count:,}"
        )
    _stage_call(stage_observer, "update_graph_aggregation", staging, records, "online_graph")

    batch_gnn_training = config.gnn_training.model_copy(
        update={"seed": config.gnn_training.seed + seq * 100003}
    )
    online_gnn_path = staging / "online_gnn"
    online_gnn = train_streaming_gnn(
        online_graph_path,
        state.gnn_state,
        online_gnn_path,
        batch_gnn_training,
        chunk_size=config.gnn_chunk_size,
        device=config.device,
        log_every_epochs=config.log_every_epochs,
    )
    _stage_call(stage_observer, "update_gnn_training", staging, records, "online_gnn")

    warm_start = (None, None, None)
    if state.updater_directory is not None:
        warm_start = load_streaming_updater_warm_start(
            state.updater_directory,
            graph_feature_dim=online_graph_manifest.node_feature_dim,
            encoder_feature_dim=online_graph_manifest.encoder_feature_dim,
        )
    updater_path = staging / "updaters"
    updaters = train_streaming_updaters(
        online_graph_path,
        online_gnn,
        state.prepared,
        shared_update,
        update_encoder,
        updater_path,
        config.fingerprint_update,
        source_ae_checkpoint_sha256=load_ae_checkpoint(
            source_ae_checkpoint, state.prepared, device="cpu"
        ).checkpoint_sha256,
        source_updater_id=warm_start[0],
        initial_location_state=warm_start[1],
        initial_feature_state=warm_start[2],
        chunk_size=config.updater_chunk_size,
        device=config.device,
        log_every_epochs=config.log_every_epochs,
    )
    _stage_call(stage_observer, "update_updater_training", staging, records, "updaters")

    ae_update_path = staging / "ae_update"
    ae_update = train_streaming_ae_update(
        state.prepared,
        shared_update,
        source_ae_checkpoint,
        updaters,
        ae_update_path,
        config.fingerprint_update,
        chunk_size=config.ae_chunk_size,
        device=config.device,
        log_every_epochs=config.log_every_epochs,
    )
    _stage_call(stage_observer, "update_ae_update", staging, records, "ae_update")

    if progress_callback is not None:
        progress_callback(
            f"[pipeline] batch {batch_id}: starting AP-transition link prediction"
        )
    ap_inference = infer_streaming_ap_graph_changes(
        reference_lp_input,
        lp_result_dir,
        online_graph_path,
        learned_gnn_state=online_gnn,
        full_update_prepared=full_update,
        input_directory=staging / "ap_transition_lp_input",
        candidate_index_path=staging / "ap_transition_candidates.sqlite3",
        result_directory=staging / "ap_transition_result",
        config=config.transition.link_prediction,
        progress_callback=progress_callback,
    )
    if progress_callback is not None:
        progress_callback(
            f"[pipeline] batch {batch_id}: AP-transition link prediction finished "
            f"({ap_inference.result.manifest.candidate_count:,} candidate edges, "
            f"{ap_inference.result.manifest.iterations:,} propagation iterations)"
        )
    _stage_call(stage_observer, "update_edge_prediction", staging, records, "ap_transition_result")

    transition_path = staging / "reference_transition"
    transition = apply_streaming_ap_delta_to_reference(
        reference_lp_input,
        lp_result_dir,
        ap_inference.input_directory,
        ap_inference.result.directory,
        full_update,
        ae_update,
        transition_path,
    )
    transition = load_streaming_reference_transition(transition_path)
    transitioned_prepared = materialize_streaming_reference_prepared(
        state.prepared, transition
    )
    transitioned_prepared_path = staging / "reference_prepared.json"
    transitioned_provenance_path = staging / "reference_value_provenance.json"
    save_prepared_dataset(
        transitioned_prepared,
        transitioned_prepared_path,
        value_provenance_path=transitioned_provenance_path,
    )
    _stage_call(stage_observer, "update_transition_state", staging, records, "reference_transition")

    migrated_ae_path = ae_update_path / "updated_ae.pt"
    source_updated_ae = load_ae_checkpoint(
        migrated_ae_path, state.prepared, device=config.device
    )
    migrated_ae_path = staging / "migrated_ae.pt"
    archived_ids = set(source_updated_ae.ap_parameter_archive)
    reactivated = tuple(
        bssid
        for bssid in transition.manifest.vocabulary_after
        if bssid not in state.prepared.bssid_vocabulary and bssid in archived_ids
    )
    migrated = migrate_ae_checkpoint_vocabulary(
        source_updated_ae.path,
        transition.manifest.vocabulary_after,
        migrated_ae_path,
        seed=config.transition.reinitialize_seed + seq,
        reactivated_bssids=reactivated,
        source_model=source_updated_ae.model,
        source_optimizer_state=source_updated_ae.optimizer_state,
    )
    if migrated.manifest.vocabulary_migration is None:
        raise AssertionError("AE migration checkpoint omitted its transition record")
    _stage_call(stage_observer, "update_vocabulary_migration", staging, records, str(migrated_ae_path.relative_to(staging)))

    post_transition_ae_path = staging / "post_transition_ae"
    post_transition_ae = fine_tune_streaming_ae_after_ap_transition(
        transition,
        ae_update,
        updaters,
        migrated_ae_path,
        post_transition_ae_path,
        config.fingerprint_update,
        chunk_size=config.ae_chunk_size,
        device=config.device,
        log_every_epochs=config.log_every_epochs,
    )
    _stage_call(stage_observer, "update_ae_realign", staging, records, "post_transition_ae")

    reference_graph_path = staging / "reference_graph"
    reference_graph_manifest = build_streaming_reference_graph_after_transition(
        reference_lp_input,
        transition,
        post_transition_ae,
        full_update,
        online_gnn,
        reference_graph_path,
        reference_prepared=transitioned_prepared,
        device=config.device,
        chunk_size=config.ae_chunk_size,
    )
    _stage_call(stage_observer, "update_reference_graph_refresh", staging, records, "reference_graph")

    final_gnn_training = config.gnn_training.model_copy(
        update={"seed": config.gnn_training.seed + seq * 100003 + 1}
    )
    final_gnn_path = staging / "reference_gnn"
    final_gnn = train_streaming_gnn(
        reference_graph_path,
        online_gnn,
        final_gnn_path,
        final_gnn_training,
        chunk_size=config.gnn_chunk_size,
        device=config.device,
        log_every_epochs=config.log_every_epochs,
    )
    _stage_call(stage_observer, "update_post_transition_gnn", staging, records, "reference_gnn")

    reference_lp_path = staging / "reference_lp_input"
    reference_lp_manifest = save_streaming_online_lp_input(
        reference_lp_input,
        reference_graph_path,
        learned_gnn_state=final_gnn,
        output_directory=reference_lp_path,
        reference_prepared=transitioned_prepared,
    )
    if (
        reference_lp_manifest.batch_role != "bootstrap_reference"
        or reference_lp_manifest.graph_id != reference_graph_manifest.update_graph_id
        or reference_lp_manifest.sample_node_ids
        != tuple(scan.scan_id for scan in transitioned_prepared.scans)
        or reference_lp_manifest.bssid_vocabulary
        != transitioned_prepared.bssid_vocabulary
    ):
        raise ValueError("post-transition LP state failed reference identity validation")
    _stage_call(stage_observer, "update_cleanup", staging, records, "reference_lp_input")

    metrics = (
        ("base_lp_iterations", float(base_result.manifest.iterations)),
        ("base_lp_candidates", float(base_result.manifest.candidate_count)),
        ("ap_lp_iterations", float(ap_inference.result.manifest.iterations)),
        ("ap_lp_candidates", float(ap_inference.result.manifest.candidate_count)),
        ("ap_edges_added", float(ap_inference.result.manifest.added_count)),
        ("ap_edges_removed", float(ap_inference.result.manifest.removed_count)),
        ("ap_removals", float(len(ap_inference.result.manifest.ap_removals))),
        ("reference_scan_count", float(len(transitioned_prepared.scans))),
        ("temporary_update_scan_count", float(len(full_update.scans))),
        ("vocabulary_before_count", float(len(state.prepared.bssid_vocabulary))),
        ("vocabulary_after_count", float(len(transitioned_prepared.bssid_vocabulary))),
        ("gnn_final_loss", float(final_gnn.manifest.loss_history[-1])),
        ("ae_eq10_final_loss", float(ae_update.manifest.loss_history[-1][0] + 0.5 * (ae_update.manifest.loss_history[-1][1] + ae_update.manifest.loss_history[-1][2]))),
    )
    current = SnapshotCurrentState(
        reference_prepared="reference_prepared.json",
        reference_value_provenance="reference_value_provenance.json",
        ae_checkpoint="post_transition_ae/post_transition_ae.pt",
        reference_graph="reference_graph",
        reference_gnn="reference_gnn",
        reference_lp_input="reference_lp_input",
        latest_updaters="updaters",
    )
    source_paths, source_hashes = _source_inputs((input_path,))
    files = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file() or path.name in {"manifest.json", "COMPLETED.json"}:
            continue
        files.append(
            SnapshotFile(
                path=path.relative_to(staging).as_posix(),
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
            )
        )
    fields = dict(
        sequence_number=seq,
        parent_snapshot_id=state.snapshot_id,
        dataset_id=config.dataset_id,
        batch_id=batch_id,
        source_input_paths=source_paths,
        source_input_sha256=source_hashes,
        source_prepared_sha256=canonical_json_digest(state.prepared),
        update_prepared_sha256=canonical_json_digest(full_update),
        source_reference_graph_id=base_result.manifest.parent_graph_id,
        resulting_reference_graph_id=reference_lp_manifest.graph_id,
        source_gnn_checkpoint_sha256=base_result.manifest.gnn_checkpoint_sha256,
        resulting_gnn_checkpoint_sha256=final_gnn.manifest.checkpoint_sha256,
        source_ae_checkpoint_sha256=source_updated_ae.manifest.source_checkpoint_sha256
        or source_updated_ae.checkpoint_sha256,
        resulting_ae_checkpoint_sha256=post_transition_ae.manifest.checkpoint_sha256,
        vocabulary_before=state.prepared.bssid_vocabulary,
        vocabulary_after=transitioned_prepared.bssid_vocabulary,
        reference_scan_ids=tuple(scan.scan_id for scan in transitioned_prepared.scans),
        update_scan_ids=update_ids,
        stage_records=tuple(records),
        metrics=metrics,
        pipeline_config=config,
        current_state=current,
        artifacts=tuple(files),
    )
    provisional = StreamingBatchSnapshotManifest.model_construct(
        snapshot_id="streambatch:" + "0" * 64, **fields
    )
    snapshot_id = "streambatch:" + canonical_json_digest(
        provisional.model_dump(mode="json", exclude={"snapshot_id"})
    )
    manifest = StreamingBatchSnapshotManifest(snapshot_id=snapshot_id, **fields)
    atomic_write_json(manifest, staging / "manifest.json")
    atomic_write_json(
        {
            "format": "gufu_streaming_snapshot_complete_v1",
            "snapshot_id": manifest.snapshot_id,
            "manifest_sha256": sha256_file(staging / "manifest.json"),
        },
        staging / "COMPLETED.json",
    )
    _stage_call(stage_observer, "update_before_commit", staging, records, "COMPLETED.json")
    final_name = f"{seq:04d}-{batch_id}-{manifest.snapshot_id[-12:]}"
    final_directory = snapshot_root / final_name
    if final_directory.exists():
        raise FileExistsError(f"refusing to overwrite completed snapshot: {final_directory}")
    os.replace(staging, final_directory)
    atomic_write_json(
        {
            "format": "gufu_streaming_latest_v1",
            "snapshot_id": manifest.snapshot_id,
            "directory": str(final_directory.relative_to(root)),
            "manifest_sha256": sha256_file(final_directory / "manifest.json"),
        },
        root / "LATEST.json",
        overwrite=True,
    )
    return final_directory, manifest


def run_streaming_sequential(
    output_root: str | Path,
    config: StreamingPipelineConfig,
    *,
    bootstrap_prepared: PreparedDataset | None = None,
    bootstrap_ae_checkpoint_path: str | Path | None = None,
    bootstrap_gnn_checkpoint: LoadedGNNCheckpoint | None = None,
    bootstrap_lp_input_directory: str | Path | None = None,
    bootstrap_lp_result_directory: str | Path | None = None,
    bootstrap_paths: tuple[str | Path, ...] = (),
    update_paths: tuple[str | Path, ...] | None = None,
    update_directory: str | Path | None = None,
    update_pattern: str = "fingerprint[1-9]*.txt",
    source_root: str | Path | None = None,
    stage_observer: Callable[[str, Path], None] | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> StreamingPipelineResult:
    """Run or resume update batches from their preceding completed state.

    A precomputed cell-6 LP result is consumed on the first batch when given.
    Subsequent batches always score the current reference LP input. Each
    completed batch is atomically published as a versioned snapshot. Partial
    ``.staging`` directories never become current state and are retained for
    diagnosis.
    """
    validate_log_interval(config.log_every_epochs)
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    updates = select_update_paths(
        update_paths=update_paths,
        update_directory=update_directory,
        pattern=update_pattern,
        bootstrap_paths=tuple(bootstrap_paths),
    )
    recovered = _recover_latest(root)
    completed = _load_existing_completed(root)
    if progress_callback is not None:
        progress_callback(
            f"[pipeline] streaming run: {len(updates):,} update batches selected; "
            f"{len(completed):,} completed batches found; "
            f"resume={'yes' if recovered is not None else 'no'}; output={root}"
        )
    skipped: list[str] = []
    done_now: list[str] = []

    if recovered is not None:
        directory, manifest = recovered
        if manifest.pipeline_config != config:
            raise ValueError("latest snapshot config differs; choose a fresh output root")
        state = _load_runtime_from_snapshot(root, directory, manifest, device=config.device)
        bootstrap_record = root / "BOOTSTRAP.json"
        if bootstrap_prepared is not None and bootstrap_record.is_file():
            record = json.loads(read_json_text(bootstrap_record))
            if record.get("prepared_sha256") != canonical_json_digest(bootstrap_prepared):
                raise ValueError("resume bootstrap dataset differs from the original run")
        initial_result = None
    else:
        required = (
            bootstrap_prepared,
            bootstrap_ae_checkpoint_path,
            bootstrap_gnn_checkpoint,
            bootstrap_lp_input_directory,
        )
        if any(value is None for value in required):
            raise ValueError("a new streaming run requires all cell-2 through cell-6 bootstrap artifacts")
        ae, lp_manifest, lp_result = _validate_bootstrap(
            bootstrap_prepared,
            bootstrap_ae_checkpoint_path,
            bootstrap_gnn_checkpoint,
            bootstrap_lp_input_directory,
            bootstrap_lp_result_directory,
            config,
        )
        bootstrap_payload = {
            "format": PIPELINE_FORMAT,
            "dataset_id": config.dataset_id,
            "prepared_sha256": canonical_json_digest(bootstrap_prepared),
            "ae_checkpoint_sha256": ae.checkpoint_sha256,
            "gnn_checkpoint_sha256": bootstrap_gnn_checkpoint.checkpoint_sha256,
            "lp_input_id": lp_manifest.input_id,
            "lp_result_id": lp_result.manifest.result_id if lp_result is not None else None,
            "pipeline_config": config.model_dump(mode="json"),
        }
        bootstrap_record = root / "BOOTSTRAP.json"
        if bootstrap_record.exists():
            old = json.loads(read_json_text(bootstrap_record))
            if old != bootstrap_payload:
                raise ValueError("streaming output root already belongs to different bootstrap artifacts")
        else:
            atomic_write_json(bootstrap_payload, bootstrap_record)
        state = _RuntimeState(
            prepared=bootstrap_prepared,
            ae_checkpoint_path=Path(bootstrap_ae_checkpoint_path),
            gnn_state=bootstrap_gnn_checkpoint,
            gnn_state_graph_directory=None,
            lp_input_directory=Path(bootstrap_lp_input_directory),
            snapshot_id=None,
            sequence_number=0,
            updater_directory=None,
        )
        initial_result = bootstrap_lp_result_directory

    for input_path in updates:
        source_path = str(input_path.resolve())
        source_sha = _file_sha(input_path)
        key = (source_path, source_sha)
        batch_id = input_path.stem
        if key in completed:
            existing_directory, manifest = completed[key]
            if manifest.batch_id != batch_id:
                raise ValueError("completed input path now maps to a different batch ID")
            skipped.append(batch_id)
            if progress_callback is not None:
                progress_callback(
                    f"[pipeline] batch {batch_id}: already committed; skipping"
                )
            if manifest.snapshot_id == state.snapshot_id:
                state = _load_runtime_from_snapshot(
                    root, existing_directory, manifest, device=config.device
                )
            continue
        conflicting_path = next(
            (
                manifest
                for (prior_path, _prior_sha), (_directory, manifest) in completed.items()
                if prior_path == source_path
            ),
            None,
        )
        if conflicting_path is not None:
            raise ValueError("an already completed update input changed on disk")
        final_directory, manifest = _process_batch(
            root,
            state,
            input_path,
            config,
            source_root=source_root,
            initial_lp_result_directory=initial_result,
            stage_observer=stage_observer,
            progress_callback=progress_callback,
        )
        initial_result = None
        done_now.append(batch_id)
        state = _load_runtime_from_snapshot(
            root, final_directory, manifest, device=config.device
        )
        completed[key] = (final_directory, manifest)
        if progress_callback is not None:
            progress_callback(
                f"[pipeline] batch {batch_id}: committed snapshot "
                f"{final_directory.name}"
            )

    latest_pair = _recover_latest(root)
    if latest_pair is None:
        if progress_callback is not None:
            progress_callback("[pipeline] run finished with no completed update snapshots")
        return StreamingPipelineResult(root, None, None, tuple(done_now), tuple(skipped))
    directory, manifest = latest_pair
    if progress_callback is not None:
        progress_callback(
            f"[pipeline] run complete: batches completed now={len(done_now):,}, "
            f"already complete={len(skipped):,}; latest snapshot={directory.name}"
        )
    return StreamingPipelineResult(
        root=root,
        latest_snapshot_directory=directory,
        latest_snapshot=manifest,
        completed_batch_ids=tuple(done_now),
        skipped_completed_batch_ids=tuple(skipped),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run/resume the GUFU disk-backed update pipeline from bootstrap stage artifacts."
    )
    parser.add_argument("--config", required=True, help="JSON StreamingPipelineConfig")
    parser.add_argument("--prepared", help="bootstrap PreparedDataset JSON")
    parser.add_argument("--ae-checkpoint", help="bootstrap AE checkpoint")
    parser.add_argument("--gnn-checkpoint", help="bootstrap GNN checkpoint")
    parser.add_argument("--lp-input", help="bootstrap cell-6 streaming LP input directory")
    parser.add_argument("--lp-result", help="bootstrap cell-6 streaming LP result directory")
    parser.add_argument("--update", action="append", dest="updates", help="explicit update path (repeatable)")
    parser.add_argument("--update-directory")
    parser.add_argument("--update-pattern", default="fingerprint[1-9]*.txt")
    parser.add_argument("--bootstrap-path", action="append", default=[])
    parser.add_argument("--source-root")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = StreamingPipelineConfig.model_validate_json(
        Path(args.config).read_text(encoding="utf-8")
    )
    root = Path(args.output)
    if (root / "LATEST.json").is_file():
        result = run_streaming_sequential(
            root,
            config,
            update_paths=tuple(args.updates) if args.updates is not None else None,
            update_directory=args.update_directory,
            update_pattern=args.update_pattern,
            bootstrap_paths=tuple(args.bootstrap_path),
            source_root=args.source_root,
        )
    else:
        required = (
            args.prepared,
            args.ae_checkpoint,
            args.gnn_checkpoint,
            args.lp_input,
        )
        if any(value is None for value in required):
            parser.error(
                "new runs require --prepared --ae-checkpoint --gnn-checkpoint --lp-input"
            )
        prepared = load_prepared_dataset(args.prepared)
        # The cell-6 compact bundle carries enough immutable graph identity to
        # validate the checkpoint; loading GraphState here would defeat the
        # disk-backed handoff immediately after LP.
        gnn = load_streaming_gnn_checkpoint(
            args.gnn_checkpoint, args.lp_input, device=config.device
        )
        result = run_streaming_sequential(
            root,
            config,
            bootstrap_prepared=prepared,
            bootstrap_ae_checkpoint_path=args.ae_checkpoint,
            bootstrap_gnn_checkpoint=gnn,
            bootstrap_lp_input_directory=args.lp_input,
            bootstrap_lp_result_directory=args.lp_result,
            bootstrap_paths=tuple(args.bootstrap_path),
            update_paths=tuple(args.updates) if args.updates is not None else None,
            update_directory=args.update_directory,
            update_pattern=args.update_pattern,
            source_root=args.source_root,
        )
    print(
        f"Streaming pipeline root={result.root}; completed={result.completed_batch_ids}; "
        f"skipped={result.skipped_completed_batch_ids}; "
        f"latest={result.latest_snapshot_directory}"
    )
    return 0


__all__ = [
    "SnapshotCurrentState",
    "StreamingBatchSnapshotManifest",
    "StreamingPipelineConfig",
    "StreamingPipelineResult",
    "main",
    "run_streaming_sequential",
    "select_update_paths",
]


if __name__ == "__main__":
    raise SystemExit(main())
