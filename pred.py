"""Reusable GUFU bootstrap and sequential batch-update orchestrator.

The CLI and ``gufu.py`` call the same APIs. Importing this module has no output
or training side effects; each batch is staged and published as one snapshot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Sequence
from uuid import uuid4

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ae import (
    AETrainingConfig,
    EncoderFeatureArtifact,
    encode_prepared_dataset,
    load_ae_checkpoint,
    load_encoder_features,
    train_autoencoder,
)
from batch_transition import (
    BatchTransitionArtifact,
    BatchTransitionResult,
    TransitionConfig,
    VocabularyTransitionState,
    apply_batch_transition,
    bootstrap_transition_state,
    infer_ap_graph_changes,
    load_transition_state,
    project_prepared_batch_to_vocabulary,
    save_transition_state,
)
from fingerprint_update import (
    FingerprintUpdateConfig,
    FingerprintUpdateManifest,
    FingerprintUpdaterModels,
    load_fingerprint_update_checkpoint,
    load_updated_fingerprints,
    prepare_fingerprint_update,
    train_fingerprint_update,
)
from graph_build import (
    GraphBuildConfig,
    GraphState,
    build_graph_state,
    load_graph_state,
    save_graph_state,
)
from gufu_artifacts import (
    atomic_write_json,
    canonical_json_digest,
    load_torch_state,
    read_json_text,
    sha256_file,
)
from gufu_data import (
    NormalizationMode,
    PreparedDataset,
    PreprocessingConfig,
    ap_id_for_bssid,
    load_prepared_dataset,
    prepare_dataset_files,
    save_prepared_dataset,
)
from lp import (
    LinkPredictionConfig,
    UpdatedGraphSnapshot,
    load_edge_delta,
    predict_edges,
    save_edge_delta,
)
from sage_ve import (
    EdgeFeatureSAGE,
    GNNFeatureArtifact,
    GNNModelConfig,
    GNNTrainingConfig,
    encode_graph_state,
    load_gnn_checkpoint,
    load_gnn_features,
    retarget_gnn_checkpoint,
    train_gnn,
)
from training_progress import DEFAULT_LOG_EVERY_K_EPOCHS, validate_log_interval


PIPELINE_FORMAT = "gufu_completed_pipeline_snapshot_v1"
PIPELINE_MARKER_FORMAT = "gufu_pipeline_completion_marker_v1"
PIPELINE_SNAPSHOT_DIR = "snapshots"


class OrchestrationConfig(BaseModel):
    """All preprocessing and model choices needed to reproduce batch stages."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    dataset_id: str
    input_format: Literal["gufu_db", "gufu_jsonl"] = "gufu_db"
    rssi_threshold_dbm: float | None = None
    prune_randomized_bssids: bool = False
    malformed_record_policy: Literal["drop", "error"] = "drop"
    empty_scan_policy: Literal["keep", "drop"] = "keep"
    duplicate_bssid_policy: Literal["strongest", "last", "error"] = "strongest"
    normalization_mode: NormalizationMode = "paper_ae"
    ae_action: Literal["train", "load"] = "load"
    gnn_action: Literal["train", "load"] = "load"
    ae_training: AETrainingConfig = Field(default_factory=AETrainingConfig)
    graph_build: GraphBuildConfig = Field(default_factory=GraphBuildConfig)
    gnn_model: GNNModelConfig = Field(default_factory=GNNModelConfig)
    gnn_training: GNNTrainingConfig = Field(default_factory=GNNTrainingConfig)
    link_prediction: LinkPredictionConfig = Field(default_factory=LinkPredictionConfig)
    fingerprint_update: FingerprintUpdateConfig = Field(default_factory=FingerprintUpdateConfig)
    transition: TransitionConfig = Field(default_factory=TransitionConfig)
    device: str = "cpu"

    @model_validator(mode="after")
    def validate_choices(self):
        if not self.dataset_id.strip():
            raise ValueError("dataset_id must not be blank")
        if self.rssi_threshold_dbm is not None and not math.isfinite(self.rssi_threshold_dbm):
            raise ValueError("RSSI threshold must be finite")
        selected_device = torch.device(self.device)
        if selected_device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError(f"requested CUDA device is unavailable: {selected_device}")
        if self.link_prediction != self.transition.link_prediction:
            raise ValueError("bootstrap and AP-transition link-prediction configurations must match")
        if self.fingerprint_update.virtual_edge_sigma != self.graph_build.virtual_edge_sigma:
            raise ValueError("fingerprint updater and graph virtual-edge thresholds must match")
        expected_graph_dim = self.ae_training.latent_dim + 2
        if self.gnn_model.graph_feature_dim != expected_graph_dim:
            raise ValueError(
                "GNN graph_feature_dim must equal AE latent_dim + 2; configure all four "
                "GNN node/edge input/output widths to that value"
            )
        return self

    def preprocessing(
        self,
        batch_id: str,
        role: Literal["bootstrap_reference", "update_batch"],
        *,
        vocabulary: tuple[str, ...] = (),
    ) -> PreprocessingConfig:
        return PreprocessingConfig(
            dataset_id=self.dataset_id,
            batch_id=batch_id,
            batch_role=role,
            input_format=self.input_format,
            base_bssid_vocabulary=vocabulary,
            rssi_threshold_dbm=self.rssi_threshold_dbm,
            prune_randomized_bssids=self.prune_randomized_bssids,
            malformed_record_policy=self.malformed_record_policy,
            empty_scan_policy=self.empty_scan_policy,
            duplicate_bssid_policy=self.duplicate_bssid_policy,
            normalization_mode=self.normalization_mode,
        )


class SnapshotArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    key: str
    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def safe_relative_path(self):
        candidate = Path(self.relative_path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("snapshot artifact paths must be relative and contained")
        return self


class StageRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    stage: Literal[
        "preprocessing",
        "ae_encoding",
        "graph_aggregation",
        "gnn_training_or_load",
        "edge_prediction",
        "updater_training",
        "ae_realign_and_reconstruct",
        "post_transition_feature_refresh",
        "temporary_node_cleanup",
    ]
    status: Literal["complete", "no_change"]
    input_graph_id: str | None = None
    output_graph_id: str | None = None
    metrics: tuple[tuple[str, float | int | str], ...] = ()


class PipelineSnapshotManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_completed_pipeline_snapshot_v1"] = PIPELINE_FORMAT
    completion_status: Literal["complete"] = "complete"
    snapshot_id: str
    snapshot_kind: Literal["bootstrap", "batch"]
    sequence_index: int = Field(ge=0)
    dataset_id: str
    batch_id: str
    parent_snapshot_sha256: str | None = None
    parent_transition_state_id: str | None = None
    transition_state_id: str = Field(pattern=r"^transition:[0-9a-f]{64}$")
    bootstrap_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bootstrap_ae_source_sha256: str | None = None
    bootstrap_gnn_source_sha256: str | None = None
    current_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_prepared_sha256: str | None = None
    input_source_files: tuple[str, ...] = ()
    input_paths: tuple[str, ...] = ()
    input_file_sha256: tuple[str, ...] = ()
    bssid_vocabulary: tuple[str, ...]
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_scan_count: int = Field(ge=1)
    update_scan_count: int = Field(ge=0)
    stage_records: tuple[StageRecord, ...]
    configuration_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifacts: tuple[SnapshotArtifact, ...]
    artifact_index: tuple[tuple[str, str], ...]
    bootstrap_source_paths: tuple[str, ...] = ()
    update_coordinates_used_as_training_labels: Literal[False] = False
    temporary_nodes_and_virtual_edges_committed: Literal[False] = False
    resume_boundary: Literal["completed_batch_atomic_weights_only_next_batch"] = (
        "completed_batch_atomic_weights_only_next_batch"
    )
    created_at_utc: str

    @model_validator(mode="after")
    def aligned_manifest(self):
        if canonical_json_digest(self.bssid_vocabulary) != self.vocabulary_sha256:
            raise ValueError("snapshot vocabulary digest is inconsistent")
        if len(self.artifact_index) != len(dict(self.artifact_index)):
            raise ValueError("snapshot artifact keys must be unique")
        if set(dict(self.artifact_index).values()) != {item.relative_path for item in self.artifacts}:
            raise ValueError("snapshot artifact index and checksums disagree")
        if self.snapshot_kind == "bootstrap":
            if self.sequence_index != 0 or self.update_scan_count != 0 or self.parent_snapshot_sha256 is not None:
                raise ValueError("bootstrap snapshot has invalid sequence/parent metadata")
            if self.bootstrap_ae_source_sha256 is not None and not re.fullmatch(
                r"[0-9a-f]{64}", self.bootstrap_ae_source_sha256
            ):
                raise ValueError("bootstrap AE source digest is malformed")
            if self.bootstrap_gnn_source_sha256 is not None and not re.fullmatch(
                r"[0-9a-f]{64}", self.bootstrap_gnn_source_sha256
            ):
                raise ValueError("bootstrap GNN source digest is malformed")
        elif self.sequence_index < 1 or self.parent_snapshot_sha256 is None or self.input_prepared_sha256 is None:
            raise ValueError("batch snapshot must identify its completed parent and input batch")
        elif self.bootstrap_ae_source_sha256 is not None or self.bootstrap_gnn_source_sha256 is not None:
            raise ValueError("only bootstrap snapshots record initial checkpoint sources")
        return self


class CompletionMarker(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_pipeline_completion_marker_v1"] = PIPELINE_MARKER_FORMAT
    snapshot_id: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    completed_at_utc: str


@dataclass(frozen=True)
class PipelineState:
    """All compatible learned/current state needed for the next batch."""

    prepared: PreparedDataset
    transition_state: VocabularyTransitionState
    encoder_features: EncoderFeatureArtifact
    graph_state: GraphState
    gnn_features: GNNFeatureArtifact
    ae_checkpoint_path: Path
    gnn_checkpoint_path: Path
    updater_checkpoint_path: Path | None
    reference_snapshot: UpdatedGraphSnapshot | None
    snapshot_directory: Path
    manifest: PipelineSnapshotManifest
    completion_sha256: str
    configuration: OrchestrationConfig


@dataclass(frozen=True)
class SequentialRunResult:
    state: PipelineState
    completed_batch_ids: tuple[str, ...]
    skipped_completed_batch_ids: tuple[str, ...]


StageObserver = Callable[[str, Path], None]


def natural_path_key(path: str | Path) -> tuple[tuple[int, int | str], ...]:
    """Natural filename key: numeric runs compare as integers."""
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", Path(path).name)
        if part
    )


def discover_update_paths(
    *,
    explicit_paths: Sequence[str | Path] | None = None,
    input_directory: str | Path | None = None,
    pattern: str = "*.txt",
    bootstrap_paths: Sequence[str | Path] = (),
) -> tuple[Path, ...]:
    """Preserve explicit order or naturally sort a directory, excluding bootstrap."""
    if explicit_paths is not None and input_directory is not None:
        raise ValueError("provide explicit update paths or an input directory, not both")
    if explicit_paths is None and input_directory is None:
        return ()
    if explicit_paths is not None:
        paths = tuple(Path(value) for value in explicit_paths)
    else:
        paths = tuple(sorted(Path(input_directory).glob(pattern), key=natural_path_key))
    excluded = {Path(value).resolve() for value in bootstrap_paths}
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in excluded:
            continue
        if resolved in seen:
            raise ValueError(f"duplicate update input path: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        seen.add(resolved)
        result.append(path)
    return tuple(result)


def _provenance_complete(prepared: PreparedDataset) -> PreparedDataset:
    """Make measured/missing provenance explicit for an initial reference."""
    payload = prepared.model_dump(mode="python")
    for row, scan in enumerate(prepared.scans):
        payload["scans"][row]["value_provenance"] = tuple(
            "measured" if observed else "missing" for observed in scan.observation_mask
        )
    return PreparedDataset.model_validate(payload)


def _prepared_after_transition(
    source: PreparedDataset,
    transition: BatchTransitionResult,
) -> PreparedDataset:
    """Apply updated values by original scan ID while retaining source/labels."""
    fingerprints = transition.fingerprints
    state = transition.state
    source_by_id = {scan.scan_id: scan for scan in source.scans}
    if tuple(row.scan_id for row in fingerprints.rows) != tuple(
        scan.scan_id for scan in source.scans
    ):
        raise ValueError("transition changed the immutable reference scan ID/order")
    if fingerprints.bssid_vocabulary != state.bssid_vocabulary:
        raise ValueError("transition fingerprint vocabulary differs from transition state")
    payload = source.model_dump(mode="python")
    payload["config"]["base_bssid_vocabulary"] = state.bssid_vocabulary
    payload["bssid_vocabulary"] = state.bssid_vocabulary
    payload["ap_ids"] = tuple(ap_id_for_bssid(bssid) for bssid in state.bssid_vocabulary)
    target_rows = {row.scan_id: row for row in fingerprints.rows}
    scans = []
    for scan_id in state.reference_scan_ids:
        original = source_by_id[scan_id]
        updated = target_rows[scan_id]
        row = original.model_dump(mode="python")
        row.update(
            rss_dbm=updated.updated_rssi_dbm,
            observation_mask=updated.observation_mask,
            value_provenance=updated.value_provenance,
        )
        scans.append(row)
    payload["scans"] = scans
    payload["stats"]["observed_readings"] = sum(
        sum(row.observation_mask) for row in fingerprints.rows
    )
    result = PreparedDataset.model_validate_json(json.dumps(payload, allow_nan=False))
    if tuple(scan.scan_id for scan in result.scans) != state.reference_scan_ids:
        raise AssertionError("transition reference rows lost stable ID order")
    if result.reference_positions != source.reference_positions:
        raise AssertionError("transition changed immutable bootstrap labels")
    return result


def _save_current_prepared(prepared: PreparedDataset, path: Path, sidecar_path: Path) -> None:
    if any(scan.value_provenance is None for scan in prepared.scans):
        raise ValueError("current reference snapshot requires explicit per-cell RSS provenance")
    save_prepared_dataset(prepared, path, value_provenance_path=sidecar_path)


def _source_file_digests(prepared: PreparedDataset) -> tuple[str, ...]:
    by_source = {scan.source.source_file: scan.source.source_file_sha256 for scan in prepared.scans}
    return tuple(by_source.get(name, "0" * 64) for name in prepared.source_files)


def _artifact_records_and_index(root: Path) -> tuple[tuple[SnapshotArtifact, ...], tuple[tuple[str, str], ...]]:
    records: list[SnapshotArtifact] = []
    index: list[tuple[str, str]] = []
    canonical_keys = {
        "current_prepared.json": "current_prepared",
        "current_value_provenance.json": "current_value_provenance",
        "input_prepared_batch.json": "input_prepared_batch",
        "shared_vocabulary_batch.json": "shared_vocabulary_batch",
        "update_encoder_features.json": "update_encoder_features",
        "updated_fingerprints.json": "updated_fingerprints",
        "transition_state.json": "transition_state",
        "encoder_features.json": "encoder_features",
        "graph_state.json": "graph_state",
        "ae_checkpoint.pt": "ae_checkpoint",
        "gnn_checkpoint.pt": "gnn_checkpoint",
        "updater_checkpoint.pt": "updater_checkpoint",
        "gnn_features.json": "gnn_features",
        "link_prediction.json": "link_prediction",
        "pipeline_config.json": "configuration",
    }
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name in {"snapshot_manifest.json", "COMPLETED.json"}:
            continue
        relative = path.relative_to(root).as_posix()
        key = canonical_keys.get(relative, relative.replace("/", "__").replace(".", "_"))
        if any(item.key == key for item in records):
            key = f"{key}_{len(records)}"
        records.append(SnapshotArtifact(key=key, relative_path=relative, sha256=sha256_file(path)))
        index.append((key, relative))
    return tuple(records), tuple(index)


def _notify(observer: StageObserver | None, stage: str, staging_path: Path) -> None:
    if observer is not None:
        observer(stage, staging_path)


def _publish_snapshot(
    staging: Path,
    destination: Path,
    manifest: PipelineSnapshotManifest,
    *,
    observer: StageObserver | None = None,
) -> tuple[Path, str]:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite completed pipeline snapshot: {destination}")
    records, index = _artifact_records_and_index(staging)
    manifest = manifest.model_copy(update={"artifacts": records, "artifact_index": index})
    manifest_path = staging / "snapshot_manifest.json"
    atomic_write_json(manifest, manifest_path)
    manifest_sha = sha256_file(manifest_path)
    for item in records:
        artifact_path = staging / item.relative_path
        if sha256_file(artifact_path) != item.sha256:
            raise ValueError(f"staged artifact digest changed before commit: {item.relative_path}")
    _notify(observer, "before_completion_marker", staging)
    marker = CompletionMarker(
        snapshot_id=manifest.snapshot_id,
        manifest_sha256=manifest_sha,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    atomic_write_json(marker, staging / "COMPLETED.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite completed pipeline snapshot: {destination}")
    os.replace(staging, destination)
    return destination, sha256_file(destination / "COMPLETED.json")


def _load_pipeline_snapshot(
    directory: str | Path,
    *,
    expected_configuration: OrchestrationConfig | None = None,
) -> PipelineState:
    root = Path(directory)
    marker_path = root / "COMPLETED.json"
    manifest_path = root / "snapshot_manifest.json"
    if not marker_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"pipeline snapshot is incomplete: {root}")
    marker = CompletionMarker.model_validate_json(read_json_text(marker_path))
    manifest_sha = sha256_file(manifest_path)
    if marker.manifest_sha256 != manifest_sha:
        raise ValueError("pipeline completion marker does not match its manifest")
    manifest = PipelineSnapshotManifest.model_validate_json(read_json_text(manifest_path))
    if marker.snapshot_id != manifest.snapshot_id:
        raise ValueError("pipeline marker and manifest identify different snapshots")
    for artifact in manifest.artifacts:
        artifact_path = root / artifact.relative_path
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact.sha256:
            raise ValueError(f"pipeline snapshot artifact is missing or corrupt: {artifact.relative_path}")
    config_path = root / dict(manifest.artifact_index)["configuration"]
    configuration = OrchestrationConfig.model_validate_json(read_json_text(config_path))
    if canonical_json_digest(configuration) != manifest.configuration_sha256:
        raise ValueError("pipeline configuration digest is inconsistent")
    if expected_configuration is not None and canonical_json_digest(expected_configuration) != manifest.configuration_sha256:
        raise ValueError("resume configuration differs from the completed pipeline snapshot")
    paths = dict(manifest.artifact_index)
    prepared_path = root / paths["current_prepared"]
    provenance_path = root / paths["current_value_provenance"]
    prepared = load_prepared_dataset(prepared_path, value_provenance_path=provenance_path)
    transition_state = load_transition_state(root / paths["transition_state"])
    encoder_features = load_encoder_features(root / paths["encoder_features"])
    graph_state = load_graph_state(root / paths["graph_state"])
    ae_path = root / paths["ae_checkpoint"]
    ae_checkpoint = load_ae_checkpoint(ae_path, prepared)
    gnn_path = root / paths["gnn_checkpoint"]
    loaded_gnn = load_gnn_checkpoint(gnn_path, graph_state, device=configuration.device)
    gnn_features = load_gnn_features(root / paths["gnn_features"])
    updater_path = root / paths["updater_checkpoint"] if "updater_checkpoint" in paths else None
    if manifest.snapshot_kind == "batch" and updater_path is None:
        raise ValueError("completed batch snapshot is missing its learned updater checkpoint")
    if updater_path is not None:
        updater_payload = load_torch_state(updater_path, map_location="cpu")
        if updater_payload.get("format") != "gufu_fingerprint_update_checkpoint_v1":
            raise ValueError("completed snapshot updater checkpoint has an unsupported format")
        updater_manifest = FingerprintUpdateManifest.model_validate_json(
            updater_payload["manifest_json"]
        )
        history_batch_ids = {record.batch_id for record in transition_state.batch_history}
        if (
            updater_manifest.dataset_id != prepared.config.dataset_id
            or updater_manifest.update_batch_id != manifest.batch_id
            or manifest.batch_id not in history_batch_ids
            or canonical_json_digest(updater_manifest.bssid_vocabulary)
            != updater_manifest.vocabulary_sha256
        ):
            raise ValueError("completed snapshot updater checkpoint has incompatible batch/lineage identity")
        transition_manifest_path = root / paths.get(
            "transition__transition_manifest_json", "transition/transition_manifest.json"
        )
        if not transition_manifest_path.is_file():
            raise ValueError("completed batch snapshot is missing its AP transition manifest")
        transition_manifest = BatchTransitionArtifact.model_validate_json(
            read_json_text(transition_manifest_path)
        )
        if (
            transition_manifest.batch_id != manifest.batch_id
            or transition_manifest.updater_checkpoint_sha256 != sha256_file(updater_path)
            or transition_state.parent_state_id != transition_manifest.state_id
        ):
            raise ValueError("batch updater, AP transition and current state lineage do not agree")
    link_delta = None
    reference_snapshot = None
    current_reference_graph_id = graph_state.graph_id
    if "link_prediction" in paths:
        link_delta = load_edge_delta(root / paths["link_prediction"])
        if link_delta.outcome == "failed" or link_delta.updated_graph is None:
            raise ValueError("bootstrap snapshot contains a failed link-prediction stage")
        if link_delta.parent_graph_id != graph_state.graph_id:
            raise ValueError("bootstrap link-prediction state belongs to a different graph")
        reference_snapshot = link_delta.updated_graph
        current_reference_graph_id = reference_snapshot.graph_id
    checks = (
        (transition_state.dataset_id == prepared.config.dataset_id, "dataset"),
        (transition_state.reference_scan_ids == tuple(scan.scan_id for scan in prepared.scans), "reference scan IDs"),
        (transition_state.bssid_vocabulary == prepared.bssid_vocabulary, "vocabulary"),
        (transition_state.ae_checkpoint_sha256 == ae_checkpoint.checkpoint_sha256, "AE checkpoint"),
        (transition_state.gnn_state_sha256 == loaded_gnn.checkpoint_sha256, "GNN checkpoint"),
        (transition_state.encoder_feature_artifact_sha256 == canonical_json_digest(encoder_features), "encoder features"),
        (transition_state.gnn_feature_artifact_sha256 == canonical_json_digest(gnn_features), "GNN features"),
        (transition_state.reference_graph_id == current_reference_graph_id, "current graph ID"),
        (transition_state.encoder_cache_status == "valid" and transition_state.gnn_cache_status == "valid", "feature-cache status"),
        (manifest.transition_state_id == transition_state.state_id, "transition state ID"),
        (manifest.current_prepared_sha256 == canonical_json_digest(prepared), "prepared data digest"),
        (manifest.reference_scan_count == len(prepared.scans), "reference sample count"),
        (manifest.bssid_vocabulary == prepared.bssid_vocabulary, "snapshot vocabulary"),
    )
    for valid, label in checks:
        if not valid:
            raise ValueError(f"completed pipeline snapshot has incompatible {label}")
    return PipelineState(
        prepared=prepared,
        transition_state=transition_state,
        encoder_features=encoder_features,
        graph_state=graph_state,
        gnn_features=gnn_features,
        ae_checkpoint_path=ae_path,
        gnn_checkpoint_path=gnn_path,
        updater_checkpoint_path=updater_path,
        reference_snapshot=reference_snapshot,
        snapshot_directory=root,
        manifest=manifest,
        completion_sha256=sha256_file(marker_path),
        configuration=configuration,
    )


def load_completed_snapshot(
    directory: str | Path,
    *,
    expected_configuration: OrchestrationConfig | None = None,
) -> PipelineState:
    """Load only a complete, hash-verified immutable pipeline snapshot."""
    return _load_pipeline_snapshot(directory, expected_configuration=expected_configuration)


def load_latest_completed_snapshot(
    output_root: str | Path,
    *,
    expected_configuration: OrchestrationConfig | None = None,
) -> PipelineState | None:
    """Find the latest completed batch; hidden staging directories are ignored."""
    snapshots = Path(output_root) / PIPELINE_SNAPSHOT_DIR
    candidates = [snapshots / "bootstrap"]
    batch_root = snapshots / "batches"
    if batch_root.is_dir():
        candidates.extend(path for path in batch_root.iterdir() if path.is_dir() and not path.name.startswith("."))
    complete: list[PipelineState] = []
    for candidate in candidates:
        if not (candidate / "COMPLETED.json").is_file():
            continue
        complete.append(
            load_completed_snapshot(candidate, expected_configuration=expected_configuration)
        )
    if not complete:
        return None
    complete.sort(key=lambda state: state.manifest.sequence_index)
    indexes = [state.manifest.sequence_index for state in complete]
    if indexes[0] != 0 or complete[0].manifest.snapshot_kind != "bootstrap":
        raise ValueError("completed pipeline history is missing its bootstrap snapshot")
    if len(indexes) != len(set(indexes)):
        raise ValueError("pipeline has multiple completed snapshots at one sequence index")
    for previous, current in zip(complete, complete[1:]):
        if current.manifest.sequence_index != previous.manifest.sequence_index + 1:
            raise ValueError("completed pipeline snapshot sequence has a gap")
        if current.manifest.parent_snapshot_sha256 != previous.completion_sha256:
            raise ValueError("completed pipeline snapshot does not descend from its previous snapshot")
    return complete[-1]


def bootstrap_pipeline(
    prepared: PreparedDataset,
    output_directory: str | Path,
    configuration: OrchestrationConfig,
    *,
    ae_checkpoint_path: str | Path | None = None,
    gnn_checkpoint_path: str | Path | None = None,
    bootstrap_source_paths: Sequence[str | Path] = (),
    stage_observer: StageObserver | None = None,
    log_every_epochs: int = DEFAULT_LOG_EVERY_K_EPOCHS,
) -> PipelineState:
    """Build the one-time reference model/graph state and publish its snapshot."""
    log_every_epochs = validate_log_interval(log_every_epochs)
    if prepared.config.dataset_id != configuration.dataset_id:
        raise ValueError("bootstrap prepared data and orchestrator dataset IDs differ")
    if prepared.config.batch_role != "bootstrap_reference":
        raise ValueError("bootstrap requires a labeled bootstrap_reference dataset")
    if not prepared.reference_positions:
        raise ValueError("bootstrap requires at least one known reference coordinate")
    if prepared.config.normalization_mode != configuration.normalization_mode:
        raise ValueError("bootstrap normalization differs from orchestration configuration")
    prepared = _provenance_complete(prepared)
    ae_source_sha256 = (
        sha256_file(ae_checkpoint_path)
        if configuration.ae_action == "load" and ae_checkpoint_path is not None
        else None
    )
    gnn_source_sha256 = (
        sha256_file(gnn_checkpoint_path)
        if configuration.gnn_action == "load" and gnn_checkpoint_path is not None
        else None
    )
    destination = Path(output_directory)
    if destination.exists():
        loaded = load_completed_snapshot(destination, expected_configuration=configuration)
        if loaded.manifest.snapshot_kind != "bootstrap" or loaded.manifest.bootstrap_prepared_sha256 != canonical_json_digest(prepared):
            raise ValueError("existing bootstrap snapshot belongs to different reference input")
        if ae_checkpoint_path is not None and loaded.manifest.bootstrap_ae_source_sha256 != ae_source_sha256:
            raise ValueError("existing bootstrap snapshot used a different AE source checkpoint")
        if gnn_checkpoint_path is not None and loaded.manifest.bootstrap_gnn_source_sha256 != gnn_source_sha256:
            raise ValueError("existing bootstrap snapshot used a different GNN source checkpoint")
        return loaded
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        stages: list[StageRecord] = []
        config_path = staging / "pipeline_config.json"
        atomic_write_json(configuration, config_path)
        prepared_path = staging / "current_prepared.json"
        provenance_path = staging / "current_value_provenance.json"
        _save_current_prepared(prepared, prepared_path, provenance_path)
        _notify(stage_observer, "bootstrap_preprocessing", staging)
        stages.append(
            StageRecord(
                stage="preprocessing",
                status="complete",
                metrics=(("reference_scans", len(prepared.scans)), ("labeled_scans", len(prepared.reference_positions))),
            )
        )

        ae_path = staging / "ae_checkpoint.pt"
        if configuration.ae_action == "train":
            train_autoencoder(
                prepared,
                ae_path,
                configuration.ae_training,
                initialization="fresh",
                device=configuration.device,
                log_every_epochs=log_every_epochs,
            )
        else:
            if ae_checkpoint_path is None or not Path(ae_checkpoint_path).is_file():
                raise FileNotFoundError("ae_action='load' requires an existing AE checkpoint path")
            shutil.copy2(ae_checkpoint_path, ae_path)
            load_ae_checkpoint(ae_path, prepared, device=configuration.device)
        ae_checkpoint = load_ae_checkpoint(ae_path, prepared, device=configuration.device)
        encoder_path = staging / "encoder_features.json"
        encoder_features = encode_prepared_dataset(
            prepared,
            ae_path,
            encoder_path,
            batch_size=configuration.ae_training.batch_size,
            device=configuration.device,
        )
        stages.append(
            StageRecord(
                stage="ae_encoding",
                status="complete",
                metrics=(("ae_checkpoint_sha256", ae_checkpoint.checkpoint_sha256), ("encoded_scans", len(encoder_features.scan_ids))),
            )
        )
        _notify(stage_observer, "bootstrap_ae_encoding", staging)

        graph_state = build_graph_state(prepared, encoder_features, configuration.graph_build)
        graph_path = staging / "graph_state.json"
        save_graph_state(graph_state, graph_path)
        stages.append(
            StageRecord(
                stage="graph_aggregation",
                status="complete",
                input_graph_id=None,
                output_graph_id=graph_state.graph_id,
                metrics=(("reference_samples", len(graph_state.sample_node_ids)), ("temporary_virtual_edges", len(graph_state.virtual_edges))),
            )
        )
        _notify(stage_observer, "bootstrap_graph_aggregation", staging)

        gnn_path = staging / "gnn_checkpoint.pt"
        gnn_features_path = staging / "gnn_features.json"
        if configuration.gnn_action == "train":
            train_gnn(
                graph_state,
                EdgeFeatureSAGE(configuration.gnn_model),
                gnn_path,
                gnn_features_path,
                training_config=configuration.gnn_training,
                device=configuration.device,
                log_every_epochs=log_every_epochs,
            )
        else:
            if gnn_checkpoint_path is None or not Path(gnn_checkpoint_path).is_file():
                raise FileNotFoundError("gnn_action='load' requires an existing GNN checkpoint path")
            shutil.copy2(gnn_checkpoint_path, gnn_path)
            load_gnn_checkpoint(gnn_path, graph_state, device=configuration.device)
            encode_graph_state(
                graph_state,
                gnn_path,
                gnn_features_path,
                device=configuration.device,
            )
        loaded_gnn = load_gnn_checkpoint(gnn_path, graph_state, device=configuration.device)
        gnn_features = load_gnn_features(gnn_features_path)
        stages.append(
            StageRecord(
                stage="gnn_training_or_load",
                status="complete",
                input_graph_id=graph_state.graph_id,
                output_graph_id=gnn_features.graph_id,
                metrics=(("gnn_checkpoint_sha256", loaded_gnn.checkpoint_sha256), ("node_rows", len(gnn_features.node_ids))),
            )
        )
        _notify(stage_observer, "bootstrap_gnn", staging)

        edge_delta = predict_edges(graph_state, gnn_features, configuration.link_prediction)
        if edge_delta.outcome == "failed" or edge_delta.updated_graph is None:
            raise RuntimeError(f"bootstrap edge prediction failed: {edge_delta.failure_reason}")
        edge_delta_path = staging / "link_prediction.json"
        save_edge_delta(edge_delta, edge_delta_path)
        stages.append(
            StageRecord(
                stage="edge_prediction",
                status="no_change" if edge_delta.outcome == "no_change" else "complete",
                input_graph_id=graph_state.graph_id,
                output_graph_id=edge_delta.updated_graph.graph_id,
                metrics=(("iterations", edge_delta.iterations), ("added_edges", len(edge_delta.added_edges)), ("removed_edges", len(edge_delta.removed_edges))),
            )
        )
        _notify(stage_observer, "bootstrap_edge_prediction", staging)

        transition_state = bootstrap_transition_state(
            prepared,
            graph_state,
            gnn_features,
            ae_path,
            reference_snapshot=edge_delta.updated_graph,
        )
        transition_state_path = staging / "transition_state.json"
        save_transition_state(transition_state, transition_state_path)
        stages.append(
            StageRecord(
                stage="temporary_node_cleanup",
                status="complete",
                input_graph_id=graph_state.graph_id,
                output_graph_id=transition_state.graph_id,
                metrics=(("reference_samples_retained", len(transition_state.reference_scan_ids)), ("virtual_edges_retired", len(graph_state.virtual_edges))),
            )
        )
        _notify(stage_observer, "bootstrap_transition_state", staging)

        manifest = PipelineSnapshotManifest(
            snapshot_id=uuid4().hex,
            snapshot_kind="bootstrap",
            sequence_index=0,
            dataset_id=prepared.config.dataset_id,
            batch_id=prepared.config.batch_id,
            transition_state_id=transition_state.state_id,
            bootstrap_prepared_sha256=canonical_json_digest(prepared),
            bootstrap_ae_source_sha256=ae_source_sha256,
            bootstrap_gnn_source_sha256=gnn_source_sha256,
            current_prepared_sha256=canonical_json_digest(prepared),
            input_source_files=prepared.source_files,
            input_paths=tuple(
                str(Path(path).resolve()) for path in bootstrap_source_paths
            ),
            input_file_sha256=_source_file_digests(prepared),
            bssid_vocabulary=prepared.bssid_vocabulary,
            vocabulary_sha256=canonical_json_digest(prepared.bssid_vocabulary),
            reference_scan_count=len(prepared.scans),
            update_scan_count=0,
            stage_records=tuple(stages),
            configuration_sha256=canonical_json_digest(configuration),
            artifacts=(),
            artifact_index=(),
            bootstrap_source_paths=tuple(str(Path(path).resolve()) for path in bootstrap_source_paths),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        _notify(stage_observer, "bootstrap_before_commit", staging)
        _publish_snapshot(staging, destination, manifest, observer=stage_observer)
        staging = Path()  # ownership moved to the committed destination
        return load_completed_snapshot(destination, expected_configuration=configuration)
    except Exception:
        if staging != Path() and staging.exists():
            shutil.rmtree(staging)
        raise


def _batch_output_name(sequence_index: int, batch_id: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", batch_id):
        raise ValueError("batch ID is not a safe snapshot path component")
    return f"{sequence_index:05d}-{batch_id}"


def _source_training_config(models: FingerprintUpdaterModels) -> GNNTrainingConfig:
    config = models.manifest.config
    history = models.manifest.gnn_loss_history
    if not history:
        raise ValueError("updated GNN checkpoint has no completed loss history")
    return GNNTrainingConfig(
        epochs=len(history),
        learning_rate=config.gnn_learning_rate,
        weight_decay=config.gnn_weight_decay,
        seed=config.seed,
        objective_epsilon=config.equation4_epsilon,
    )


def update_one_batch(
    state: PipelineState,
    full_incoming_batch: PreparedDataset,
    output_directory: str | Path,
    configuration: OrchestrationConfig | None = None,
    *,
    input_paths: Sequence[str | Path] = (),
    stage_observer: StageObserver | None = None,
    log_every_epochs: int = DEFAULT_LOG_EVERY_K_EPOCHS,
) -> PipelineState:
    """Process one full batch from the prior completed model/graph/fingerprint state."""
    log_every_epochs = validate_log_interval(log_every_epochs)
    selected = configuration or state.configuration
    if canonical_json_digest(selected) != state.manifest.configuration_sha256:
        raise ValueError("batch configuration differs from the previous completed snapshot")
    if full_incoming_batch.config.batch_role != "update_batch":
        raise ValueError("single-batch update requires an update_batch PreparedDataset")
    if full_incoming_batch.config.dataset_id != state.prepared.config.dataset_id:
        raise ValueError("incoming batch and current pipeline state have different dataset IDs")
    if full_incoming_batch.config.normalization_mode != state.prepared.config.normalization_mode:
        raise ValueError("incoming batch normalization differs from current model state")
    if tuple(scan.scan_id for scan in state.prepared.scans) != state.transition_state.reference_scan_ids:
        raise ValueError("current prepared fingerprints do not match transition-state reference IDs")
    if len(state.prepared.scans) != state.manifest.reference_scan_count:
        raise ValueError("current reference sample count differs from its completed snapshot")
    prior_history = {item.batch_id: item.scan_ids for item in state.transition_state.batch_history}
    if full_incoming_batch.config.batch_id in prior_history:
        if prior_history[full_incoming_batch.config.batch_id] == tuple(scan.scan_id for scan in full_incoming_batch.scans):
            raise ValueError(f"batch {full_incoming_batch.config.batch_id!r} is already complete; resume from its snapshot")
        raise ValueError("batch ID was reused with different scan identities")
    if not full_incoming_batch.scans:
        raise ValueError("cannot commit an empty update batch")

    destination = Path(output_directory)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite update snapshot: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        records: list[StageRecord] = []
        input_ids = tuple(scan.scan_id for scan in full_incoming_batch.scans)
        if set(input_ids) & (
            set(state.transition_state.reference_scan_ids)
            | {scan_id for record in state.transition_state.batch_history for scan_id in record.scan_ids}
        ):
            raise ValueError("incoming scan IDs collide with a previous reference or batch")

        full_batch_path = staging / "input_prepared_batch.json"
        save_prepared_dataset(full_incoming_batch, full_batch_path)
        projected = project_prepared_batch_to_vocabulary(
            full_incoming_batch, state.transition_state.bssid_vocabulary
        )
        projected_path = staging / "shared_vocabulary_batch.json"
        save_prepared_dataset(projected, projected_path)
        update_encoder_path = staging / "update_encoder_features.json"
        update_encoder = encode_prepared_dataset(
            projected,
            state.ae_checkpoint_path,
            update_encoder_path,
            batch_size=selected.ae_training.batch_size,
            device=selected.device,
        )
        records.append(
            StageRecord(
                stage="ae_encoding",
                status="complete",
                input_graph_id=None,
                output_graph_id=None,
                metrics=(("update_scans_encoded", len(update_encoder.scan_ids)), ("old_vocabulary_width", len(state.transition_state.bssid_vocabulary))),
            )
        )
        if update_encoder.scan_ids != input_ids:
            raise ValueError("old-vocabulary projection or AE encoder changed incoming scan identities")
        _notify(stage_observer, "update_shared_ae_encoding", staging)

        update_context = prepare_fingerprint_update(
            reference_prepared=state.prepared,
            reference_graph=state.graph_state,
            reference_encoder_features=state.encoder_features,
            reference_gnn_features=state.gnn_features,
            update_prepared=projected,
            update_encoder_features=update_encoder,
            ae_checkpoint_path=state.ae_checkpoint_path,
            gnn_checkpoint_path=state.gnn_checkpoint_path,
            reference_snapshot=state.reference_snapshot,
            config=selected.fingerprint_update,
        )
        if update_context.online_graph.update_sample_ids != input_ids:
            raise ValueError("incoming batch samples did not enter the transient update graph")
        records.append(
            StageRecord(
                stage="graph_aggregation",
                status="complete",
                input_graph_id=state.graph_state.graph_id,
                output_graph_id=update_context.online_graph.graph_id,
                metrics=(
                    ("reference_nodes", len(update_context.online_graph.reference_sample_ids)),
                    ("temporary_update_nodes", len(update_context.online_graph.update_sample_ids)),
                    ("temporary_radio_edges", len(update_context.online_graph.radio_edges)),
                    ("temporary_virtual_edges", len(update_context.online_graph.virtual_edges)),
                ),
            )
        )
        _notify(stage_observer, "update_graph_aggregation", staging)

        updater_checkpoint_path = staging / "updater_checkpoint.pt"
        updated_fingerprints_path = staging / "updated_fingerprints.json"
        train_fingerprint_update(
            update_context,
            updater_checkpoint_path,
            updated_fingerprints_path,
            selected.fingerprint_update,
            device=selected.device,
            initial_updater_checkpoint_path=state.updater_checkpoint_path,
            log_every_epochs=log_every_epochs,
        )
        updater_models = load_fingerprint_update_checkpoint(
            updater_checkpoint_path,
            update_context,
            device=selected.device,
        )
        updated_fingerprints = load_updated_fingerprints(updated_fingerprints_path)
        if updater_models.manifest.update_node_ids != input_ids:
            raise ValueError("updater checkpoint omitted one or more incoming scans")
        if updated_fingerprints.update_scan_ids != input_ids:
            raise ValueError("updated fingerprint result lost incoming scan identity/order")
        records.append(
            StageRecord(
                stage="updater_training",
                status="complete",
                input_graph_id=update_context.online_graph.graph_id,
                output_graph_id=updater_models.manifest.online_graph_id,
                metrics=(
                    ("gnn_epochs", len(updater_models.manifest.gnn_loss_history)),
                    ("updater_epochs", len(updater_models.manifest.loss_history)),
                    ("ae_epochs", len(updater_models.manifest.ae_loss_history)),
                    ("final_updater_loss", updater_models.manifest.loss_history[-1].total),
                    ("update_scans", len(updater_models.manifest.update_node_ids)),
                    ("updater_initialization", updater_models.manifest.updater_initialization),
                    ("source_update_checkpoint_sha256", updater_models.manifest.source_update_checkpoint_sha256 or "none"),
                ),
            )
        )
        _notify(stage_observer, "update_updater_training", staging)

        ap_inference = infer_ap_graph_changes(
            state.transition_state,
            update_context,
            updater_models,
            full_incoming_batch,
            selected.transition,
        )
        modification = ap_inference.modification
        changed_edges = len(modification.added_reference_edges) + len(modification.removed_reference_edges)
        edge_status: Literal["complete", "no_change"] = "no_change" if changed_edges == 0 and not modification.removed_ap_bssids else "complete"
        records.append(
            StageRecord(
                stage="edge_prediction",
                status=edge_status,
                input_graph_id=update_context.online_graph.graph_id,
                output_graph_id=modification.online_graph_id,
                metrics=(
                    ("candidate_pairs_scored", len(modification.candidate_scores)),
                    ("edges_added", len(modification.added_reference_edges)),
                    ("edges_removed", len(modification.removed_reference_edges)),
                    ("aps_forgotten", len(modification.removed_ap_bssids)),
                ),
            )
        )
        _notify(stage_observer, "update_edge_prediction", staging)

        transition_directory = staging / "transition"
        transition_result = apply_batch_transition(
            state.transition_state,
            full_incoming_batch,
            updated_fingerprints,
            ap_inference,
            state.ae_checkpoint_path,
            transition_directory,
            selected.transition,
            updater_models=updater_models,
            log_every_epochs=log_every_epochs,
        )
        if transition_result.state.reference_scan_ids != state.transition_state.reference_scan_ids:
            raise ValueError("AP transition changed the retained reference sample IDs")
        records.append(
            StageRecord(
                stage="ae_realign_and_reconstruct",
                status="complete",
                input_graph_id=update_context.online_graph.graph_id,
                output_graph_id=transition_result.state.graph_id,
                metrics=(
                    ("vocabulary_before", len(state.transition_state.bssid_vocabulary)),
                    ("vocabulary_after", len(transition_result.state.bssid_vocabulary)),
                    ("ae_realign_epochs", transition_result.migrated_ae_checkpoint.manifest.vocabulary_migration.finetune_epochs),
                    ("original_reference_scans", len(transition_result.state.reference_scan_ids)),
                ),
            )
        )
        _notify(stage_observer, "update_ae_realign", staging)

        next_prepared = _prepared_after_transition(state.prepared, transition_result)
        current_prepared_path = staging / "current_prepared.json"
        current_provenance_path = staging / "current_value_provenance.json"
        _save_current_prepared(next_prepared, current_prepared_path, current_provenance_path)
        next_ae_path = staging / "ae_checkpoint.pt"
        shutil.copy2(transition_result.migrated_ae_checkpoint.path, next_ae_path)
        load_ae_checkpoint(next_ae_path, next_prepared, device=selected.device)
        next_encoder_path = staging / "encoder_features.json"
        next_encoder = encode_prepared_dataset(
            next_prepared,
            next_ae_path,
            next_encoder_path,
            batch_size=selected.ae_training.batch_size,
            device=selected.device,
        )

        edge_weights = {
            (edge.sample_node_id, edge.bssid): edge.scalar_weight
            for edge in transition_result.state.radio_edges
        }
        next_graph = build_graph_state(
            next_prepared,
            next_encoder,
            transition_result.state.graph_build_config,
            radio_edge_weights=edge_weights,
        )
        next_graph_path = staging / "graph_state.json"
        save_graph_state(next_graph, next_graph_path)
        next_gnn_path = staging / "gnn_checkpoint.pt"
        gnn_training_config = _source_training_config(updater_models)
        source_gnn_history = tuple(updater_models.manifest.gnn_loss_history)
        retarget_gnn_checkpoint(
            updater_models.gnn_model,
            next_graph,
            gnn_training_config,
            source_gnn_history,
            next_gnn_path,
            source_checkpoint_sha256=updater_models.checkpoint_sha256,
            transition_state_id=transition_result.state.state_id,
        )
        next_gnn_features_path = staging / "gnn_features.json"
        encode_graph_state(
            next_graph,
            next_gnn_path,
            next_gnn_features_path,
            device=selected.device,
        )
        next_gnn_features = load_gnn_features(next_gnn_features_path)
        next_transition_state = bootstrap_transition_state(
            next_prepared,
            next_graph,
            next_gnn_features,
            next_ae_path,
            parent_state=transition_result.state,
        )
        next_transition_path = staging / "transition_state.json"
        save_transition_state(next_transition_state, next_transition_path)
        records.append(
            StageRecord(
                stage="post_transition_feature_refresh",
                status="complete",
                input_graph_id=transition_result.state.graph_id,
                output_graph_id=next_graph.graph_id,
                metrics=(
                    ("encoder_rows", len(next_encoder.scan_ids)),
                    ("gnn_node_rows", len(next_gnn_features.node_ids)),
                    ("gnn_edge_rows", len(next_gnn_features.radio_edge_ids)),
                    ("gnn_retarget_source", updater_models.checkpoint_sha256),
                ),
            )
        )
        _notify(stage_observer, "update_post_transition_features", staging)

        if (
            tuple(next_graph.sample_node_ids) != state.transition_state.reference_scan_ids
            or set(input_ids) & set(next_graph.node_ids)
            or len(next_graph.sample_node_ids) != len(state.prepared.scans)
        ):
            raise ValueError("temporary update nodes survived or reference sample count changed")
        records.append(
            StageRecord(
                stage="temporary_node_cleanup",
                status="complete",
                input_graph_id=update_context.online_graph.graph_id,
                output_graph_id=next_graph.graph_id,
                metrics=(("temporary_update_nodes_retained", 0), ("reference_samples", len(next_graph.sample_node_ids))),
            )
        )
        _notify(stage_observer, "update_cleanup", staging)

        atomic_write_json(selected, staging / "pipeline_config.json")
        manifest = PipelineSnapshotManifest(
            snapshot_id=uuid4().hex,
            snapshot_kind="batch",
            sequence_index=state.manifest.sequence_index + 1,
            dataset_id=state.prepared.config.dataset_id,
            batch_id=full_incoming_batch.config.batch_id,
            parent_snapshot_sha256=state.completion_sha256,
            parent_transition_state_id=state.transition_state.state_id,
            transition_state_id=next_transition_state.state_id,
            bootstrap_prepared_sha256=state.manifest.bootstrap_prepared_sha256,
            current_prepared_sha256=canonical_json_digest(next_prepared),
            input_prepared_sha256=canonical_json_digest(full_incoming_batch),
            input_source_files=full_incoming_batch.source_files,
            input_paths=tuple(
                str(Path(path).resolve()) for path in input_paths
            ) or full_incoming_batch.source_files,
            input_file_sha256=_source_file_digests(full_incoming_batch),
            bssid_vocabulary=next_prepared.bssid_vocabulary,
            vocabulary_sha256=canonical_json_digest(next_prepared.bssid_vocabulary),
            reference_scan_count=len(next_prepared.scans),
            update_scan_count=len(full_incoming_batch.scans),
            stage_records=tuple(records),
            configuration_sha256=canonical_json_digest(selected),
            artifacts=(),
            artifact_index=(),
            bootstrap_source_paths=state.manifest.bootstrap_source_paths,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        _notify(stage_observer, "update_before_commit", staging)
        _publish_snapshot(staging, destination, manifest, observer=stage_observer)
        staging = Path()
        return load_completed_snapshot(destination, expected_configuration=selected)
    except Exception:
        if staging != Path() and staging.exists():
            shutil.rmtree(staging)
        raise


def _validate_preprocessing_config(
    prepared: PreparedDataset,
    configuration: OrchestrationConfig,
    *,
    role: Literal["bootstrap_reference", "update_batch"],
    vocabulary: tuple[str, ...] = (),
) -> None:
    expected = configuration.preprocessing(
        prepared.config.batch_id,
        role,
        vocabulary=vocabulary,
    )
    if prepared.config != expected:
        raise ValueError(
            f"prepared {role} data uses preprocessing settings different from the orchestrator configuration"
        )


def run_sequential(
    output_root: str | Path,
    configuration: OrchestrationConfig,
    *,
    bootstrap_prepared: PreparedDataset | None = None,
    bootstrap_paths: Sequence[str | Path] | None = None,
    bootstrap_batch_id: str | None = None,
    update_paths: Sequence[str | Path] | None = None,
    update_directory: str | Path | None = None,
    update_pattern: str = "*.txt",
    bootstrap_source_paths: Sequence[str | Path] | None = None,
    source_root: str | Path | None = None,
    ae_checkpoint_path: str | Path | None = None,
    gnn_checkpoint_path: str | Path | None = None,
    resume: bool = True,
    stage_observer: StageObserver | None = None,
    log_every_epochs: int = DEFAULT_LOG_EVERY_K_EPOCHS,
) -> SequentialRunResult:
    """Bootstrap once, then process input files in a deterministic sequence.

    A restart validates and loads the newest completed snapshot, then skips
    batches whose stable scan IDs already appear in its lineage. Explicit path
    order is preserved; directory discovery uses natural filename order.
    """
    log_every_epochs = validate_log_interval(log_every_epochs)
    root = Path(output_root)
    selected = configuration
    latest = (
        load_latest_completed_snapshot(root, expected_configuration=selected)
        if resume
        else None
    )
    if not resume and (root / PIPELINE_SNAPSHOT_DIR).exists():
        existing = list((root / PIPELINE_SNAPSHOT_DIR).rglob("COMPLETED.json"))
        if existing:
            raise FileExistsError(
                "resume is disabled but this output root already contains completed snapshots; choose a new output root"
            )

    bootstrap_file_paths = tuple(Path(path) for path in (bootstrap_paths or ()))
    if latest is None:
        if bootstrap_prepared is None:
            if not bootstrap_file_paths:
                raise ValueError(
                    "a fresh pipeline requires bootstrap_prepared or bootstrap_paths"
                )
            selected_batch_id = bootstrap_batch_id or bootstrap_file_paths[0].stem
            if not selected_batch_id:
                raise ValueError("bootstrap batch ID must not be blank")
            bootstrap_prepared = prepare_dataset_files(
                bootstrap_file_paths,
                selected.preprocessing(selected_batch_id, "bootstrap_reference"),
                source_root=source_root,
            )
        _validate_preprocessing_config(
            bootstrap_prepared,
            selected,
            role="bootstrap_reference",
        )
        recorded_bootstrap_paths = (
            tuple(bootstrap_source_paths)
            if bootstrap_source_paths is not None
            else bootstrap_file_paths
        )
        state = bootstrap_pipeline(
            bootstrap_prepared,
            root / PIPELINE_SNAPSHOT_DIR / "bootstrap",
            selected,
            ae_checkpoint_path=ae_checkpoint_path,
            gnn_checkpoint_path=gnn_checkpoint_path,
            bootstrap_source_paths=recorded_bootstrap_paths,
            stage_observer=stage_observer,
            log_every_epochs=log_every_epochs,
        )
    else:
        state = latest
        if bootstrap_prepared is not None:
            supplied_bootstrap = _provenance_complete(bootstrap_prepared)
            _validate_preprocessing_config(
                supplied_bootstrap,
                selected,
                role="bootstrap_reference",
            )
            if canonical_json_digest(supplied_bootstrap) != state.manifest.bootstrap_prepared_sha256:
                raise ValueError("resume bootstrap data differs from the completed pipeline lineage")
        bootstrap_snapshot = load_completed_snapshot(
            root / PIPELINE_SNAPSHOT_DIR / "bootstrap",
            expected_configuration=selected,
        )
        if ae_checkpoint_path is not None:
            if bootstrap_snapshot.manifest.bootstrap_ae_source_sha256 != sha256_file(ae_checkpoint_path):
                raise ValueError("resume AE source checkpoint differs from the bootstrap lineage")
        if gnn_checkpoint_path is not None:
            if bootstrap_snapshot.manifest.bootstrap_gnn_source_sha256 != sha256_file(gnn_checkpoint_path):
                raise ValueError("resume GNN source checkpoint differs from the bootstrap lineage")

    recorded_bootstrap = tuple(state.manifest.bootstrap_source_paths)
    excluded_bootstrap = tuple(bootstrap_file_paths) + recorded_bootstrap
    paths = discover_update_paths(
        explicit_paths=update_paths,
        input_directory=update_directory,
        pattern=update_pattern,
        bootstrap_paths=excluded_bootstrap,
    )
    completed: list[str] = []
    skipped: list[str] = []
    seen_batch_ids: set[str] = set()
    for path in paths:
        batch_id = path.stem
        if batch_id in seen_batch_ids:
            raise ValueError(
                f"multiple update paths resolve to batch ID {batch_id!r}; use distinct filenames"
            )
        seen_batch_ids.add(batch_id)
        full_batch = prepare_dataset_files(
            (path,),
            selected.preprocessing(
                batch_id,
                "update_batch",
                vocabulary=state.prepared.bssid_vocabulary,
            ),
            source_root=source_root,
        )
        prior_batches = {
            record.batch_id: record.scan_ids
            for record in state.transition_state.batch_history
        }
        if batch_id in prior_batches:
            current_ids = tuple(scan.scan_id for scan in full_batch.scans)
            if current_ids != prior_batches[batch_id]:
                raise ValueError(
                    f"completed batch {batch_id!r} now has different stable scan IDs"
                )
            skipped.append(batch_id)
            continue
        _validate_preprocessing_config(
            full_batch,
            selected,
            role="update_batch",
            vocabulary=state.prepared.bssid_vocabulary,
        )
        destination = (
            root
            / PIPELINE_SNAPSHOT_DIR
            / "batches"
            / _batch_output_name(state.manifest.sequence_index + 1, batch_id)
        )
        state = update_one_batch(
            state,
            full_batch,
            destination,
            selected,
            input_paths=(path,),
            stage_observer=stage_observer,
            log_every_epochs=log_every_epochs,
        )
        completed.append(batch_id)
    return SequentialRunResult(
        state=state,
        completed_batch_ids=tuple(completed),
        skipped_completed_batch_ids=tuple(skipped),
    )


def _cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the GUFU sequential batch pipeline")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset-id")
    parser.add_argument("--config-json")
    parser.add_argument("--bootstrap", nargs="+")
    parser.add_argument("--bootstrap-batch-id")
    updates = parser.add_mutually_exclusive_group()
    updates.add_argument("--updates", nargs="+")
    updates.add_argument("--updates-directory")
    parser.add_argument("--update-pattern", default="*.txt")
    parser.add_argument("--source-root")
    parser.add_argument("--ae-checkpoint")
    parser.add_argument("--gnn-checkpoint")
    parser.add_argument("--train-bootstrap-models", action="store_true")
    parser.add_argument(
        "--log-every-epochs",
        type=int,
        default=DEFAULT_LOG_EVERY_K_EPOCHS,
        help="print training metrics every K epochs and on each stage's final epoch",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _cli_parser().parse_args(argv)
    if args.config_json:
        if args.train_bootstrap_models:
            raise ValueError("--train-bootstrap-models cannot override an explicit configuration file")
        configuration = OrchestrationConfig.model_validate_json(
            read_json_text(args.config_json)
        )
        if args.dataset_id is not None and args.dataset_id != configuration.dataset_id:
            raise ValueError("--dataset-id disagrees with the explicit configuration")
    else:
        if not args.dataset_id:
            raise ValueError("provide --dataset-id or --config-json")
        configuration = OrchestrationConfig(
            dataset_id=args.dataset_id,
            ae_action="train" if args.train_bootstrap_models else "load",
            gnn_action="train" if args.train_bootstrap_models else "load",
        )
    result = run_sequential(
        args.output_root,
        configuration,
        bootstrap_paths=args.bootstrap,
        bootstrap_batch_id=args.bootstrap_batch_id,
        update_paths=args.updates,
        update_directory=args.updates_directory,
        update_pattern=args.update_pattern,
        source_root=args.source_root,
        ae_checkpoint_path=args.ae_checkpoint,
        gnn_checkpoint_path=args.gnn_checkpoint,
        resume=not args.no_resume,
        log_every_epochs=args.log_every_epochs,
    )
    print(f"Pipeline snapshot: {result.state.snapshot_directory}")
    print(f"Completed this run: {result.completed_batch_ids}")
    print(f"Already completed: {result.skipped_completed_batch_ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
