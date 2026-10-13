"""Disk-backed application of one streamed AP edge delta to reference RSS.

The module consumes streamed score/delta JSONL and mapped edge arrays. It
retains the original reference scan IDs and coordinates, applies only
reference-edge changes, derives the post-forgetting vocabulary by full BSSID
identity, and writes remapped RSS/mask/provenance arrays plus the resulting
reference radio table. It does not create ``APGraphModification`` or a
candidate-sized Python edge list.
"""

from __future__ import annotations

import json
import hashlib
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from batch_transition import observed_bssids
from gufu_artifacts import (
    atomic_write_json,
    canonical_json_digest,
    read_json_text,
    sha256_file,
)
from gufu_data import PreparedDataset, PreparedScan, ap_id_for_bssid
from lp_streaming import (
    StreamArrayArtifact,
    load_streaming_lp_input,
    load_streaming_lp_result,
)
from streaming_fingerprint_update import StreamingAEUpdateState


STREAMING_TRANSITION_FORMAT = "gufu_streaming_reference_transition_v1"


class StreamingFileArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class StreamingReferenceTransitionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_streaming_reference_transition_v1"] = STREAMING_TRANSITION_FORMAT
    transition_id: str = Field(pattern=r"^streamtransition:[0-9a-f]{64}$")
    parent_graph_id: str
    parent_lp_result_id: str
    ap_lp_result_id: str
    ae_update_id: str
    dataset_id: str
    reference_batch_id: str
    update_batch_id: str
    reference_scan_ids: tuple[str, ...]
    original_coordinates: tuple[tuple[float, float] | None, ...]
    update_scan_ids: tuple[str, ...]
    vocabulary_before: tuple[str, ...]
    vocabulary_after: tuple[str, ...]
    vocabulary_before_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vocabulary_after_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    removed_by_forgetting_bssids: tuple[str, ...]
    admitted_bssids: tuple[str, ...]
    normalization: Literal["paper_ae", "legacy_raw_div_120"]
    threshold_value: float = Field(ge=0.0)
    threshold_units: Literal["rssi_plus_120_scalar_weight"]
    value_provenance_codes: tuple[str, ...] = (
        "measured",
        "decoder_prediction",
        "edge_prediction",
        "missing",
    )
    arrays: tuple[StreamArrayArtifact, ...]
    reference_edges: StreamingFileArtifact
    created_at_utc: str

    @model_validator(mode="after")
    def aligned(self):
        if len(set(self.reference_scan_ids)) != len(self.reference_scan_ids):
            raise ValueError("streamed transition reference IDs must be unique")
        if len(self.reference_scan_ids) != len(self.original_coordinates):
            raise ValueError("streamed transition coordinates must align with reference IDs")
        if len(set(self.update_scan_ids)) != len(self.update_scan_ids):
            raise ValueError("streamed transition update IDs must be unique")
        if not self.vocabulary_after or len(set(self.vocabulary_after)) != len(self.vocabulary_after):
            raise ValueError("streamed transition target vocabulary must be nonempty and unique")
        if canonical_json_digest(self.vocabulary_before) != self.vocabulary_before_sha256:
            raise ValueError("streamed transition source vocabulary digest mismatch")
        if canonical_json_digest(self.vocabulary_after) != self.vocabulary_after_sha256:
            raise ValueError("streamed transition target vocabulary digest mismatch")
        if set(self.vocabulary_after) & set(self.removed_by_forgetting_bssids):
            raise ValueError("forgotten APs cannot remain in target vocabulary")
        shapes = {item.name: item.shape for item in self.arrays}
        expected = {
            "updated_rssi_dbm.npy": (len(self.reference_scan_ids), len(self.vocabulary_after)),
            "observation_mask.npy": (len(self.reference_scan_ids), len(self.vocabulary_after)),
            "value_provenance.npy": (len(self.reference_scan_ids), len(self.vocabulary_after)),
        }
        if shapes != expected:
            raise ValueError("streamed transition arrays do not align with IDs/vocabulary")
        identity = canonical_json_digest(
            self.model_dump(mode="json", exclude={"transition_id"})
        )
        if self.transition_id != f"streamtransition:{identity}":
            raise ValueError("streamed transition ID does not match its manifest")
        return self


@dataclass(frozen=True)
class StreamingReferenceTransitionState:
    directory: Path
    manifest: StreamingReferenceTransitionManifest
    updated_rssi_dbm: np.ndarray
    observation_mask: np.ndarray
    value_provenance: np.ndarray


def materialize_streaming_reference_prepared(
    previous_reference: PreparedDataset,
    transition_state: StreamingReferenceTransitionState,
) -> PreparedDataset:
    """Create the next immutable reference dataset view from streamed RSS.

    Scan IDs, source rows, and bootstrap coordinate labels remain those of the
    original reference scans. RSS, masks, provenance, and the AP vocabulary
    transition by full BSSID identity. Update/evaluation labels are not copied
    into this reference artifact.
    """
    manifest = transition_state.manifest
    previous_ids = tuple(scan.scan_id for scan in previous_reference.scans)
    if previous_ids != manifest.reference_scan_ids:
        raise ValueError("reference prepared scan IDs differ from transition state")
    if previous_reference.config.dataset_id != manifest.dataset_id:
        raise ValueError("reference prepared dataset differs from transition state")
    if previous_reference.bssid_vocabulary != manifest.vocabulary_before:
        raise ValueError("reference prepared vocabulary differs from transition source")
    shape = (len(previous_ids), len(manifest.vocabulary_after))
    if any(
        array.shape != shape
        for array in (
            transition_state.updated_rssi_dbm,
            transition_state.observation_mask,
            transition_state.value_provenance,
        )
    ):
        raise ValueError("transition RSS, masks, and provenance do not align")
    code_to_name = {
        0: "measured",
        1: "decoder_prediction",
        2: "edge_prediction",
        3: "missing",
    }
    scans = []
    for row, old_scan in enumerate(previous_reference.scans):
        scans.append(
            PreparedScan(
                timestamp=old_scan.timestamp,
                scan_id=old_scan.scan_id,
                dataset_id=old_scan.dataset_id,
                batch_id=old_scan.batch_id,
                batch_role=old_scan.batch_role,
                source=old_scan.source,
                coordinate_role=old_scan.coordinate_role,
                has_coordinates=old_scan.has_coordinates,
                rss_dbm=tuple(
                    float(value) for value in transition_state.updated_rssi_dbm[row]
                ),
                observation_mask=tuple(
                    bool(value) for value in transition_state.observation_mask[row]
                ),
                value_provenance=tuple(
                    code_to_name[int(value)]
                    for value in transition_state.value_provenance[row]
                ),
            )
        )
    stats = previous_reference.stats.model_copy(
        update={
            "observed_readings": int(
                np.asarray(transition_state.observation_mask, dtype=np.uint8).sum()
            )
        }
    )
    return PreparedDataset.model_validate(
        {
            "schema_version": previous_reference.schema_version,
            "config": previous_reference.config.model_copy(
                update={"base_bssid_vocabulary": ()}
            ),
            "bssid_vocabulary": manifest.vocabulary_after,
            "ap_ids": tuple(ap_id_for_bssid(value) for value in manifest.vocabulary_after),
            "scans": tuple(scans),
            "reference_positions": previous_reference.reference_positions,
            "evaluation_positions": (),
            "source_files": previous_reference.source_files,
            "dropped_records": previous_reference.dropped_records,
            "stats": stats,
        }
    )


def _open_array(path: Path, dtype, shape: tuple[int, ...]) -> np.ndarray:
    if not np.prod(shape, dtype=np.int64):
        array = np.empty(shape, dtype=dtype)
        np.save(path, array, allow_pickle=False)
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _array_artifact(path: Path, name: str) -> StreamArrayArtifact:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return StreamArrayArtifact(
        name=name,
        dtype=array.dtype.name,
        shape=tuple(int(value) for value in array.shape),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def apply_streaming_ap_delta_to_reference(
    bootstrap_lp_input_directory: str | Path,
    bootstrap_lp_result_directory: str | Path,
    transition_lp_input_directory: str | Path,
    transition_lp_result_directory: str | Path,
    update_prepared: PreparedDataset,
    ae_update: StreamingAEUpdateState,
    output_directory: str | Path,
) -> StreamingReferenceTransitionState:
    """Apply streamed AP removals/additions and remap reference RSS by BSSID.

    ``ae_update`` is the Eq. (10) result before AP topology/vocabulary
    transition. For old APs its decoded values are retained; added reference
    edges write ``predicted RSS+120 - 120`` with false measured masks and
    ``edge_prediction`` provenance. New/returning APs enter vocabulary only
    when the streamed edge rule admits at least one reference edge, matching
    the audited ``classify_and_plan_ap_vocabulary`` policy.
    """
    base, _ = load_streaming_lp_input(bootstrap_lp_input_directory)
    base_lp = load_streaming_lp_result(
        bootstrap_lp_result_directory,
        input_directory=bootstrap_lp_input_directory,
    )
    transition_input, _ = load_streaming_lp_input(transition_lp_input_directory)
    transition_lp = load_streaming_lp_result(
        transition_lp_result_directory,
        input_directory=transition_lp_input_directory,
    )
    am = ae_update.manifest
    if base_lp.manifest.outcome == "failed" or transition_lp.manifest.outcome == "failed":
        raise ValueError("cannot apply an unsuccessful streamed LP result")
    if transition_lp.manifest.candidate_policy != "explicit_sorted_stable_ids_v1":
        raise ValueError("AP transition must use the recorded restricted candidate policy")
    if transition_lp.manifest.parent_graph_id != transition_input.graph_id:
        raise ValueError("AP transition scores do not match their LP input graph")
    if transition_input.dataset_id != am.dataset_id or update_prepared.config.dataset_id != am.dataset_id:
        raise ValueError("streamed transition dataset identity mismatch")
    if update_prepared.config.batch_id != am.update_batch_id:
        raise ValueError("streamed transition update batch identity mismatch")
    if update_prepared.config.batch_role != "update_batch":
        raise ValueError("streamed transition requires update_batch input")
    if tuple(scan.scan_id for scan in update_prepared.scans) != am.update_scan_ids:
        raise ValueError("streamed transition update scan IDs/order mismatch")
    if (
        am.reference_scan_ids != base.sample_node_ids
        or am.bssid_vocabulary != base.bssid_vocabulary
        or am.reference_prepared_sha256 != base.source_prepared_sha256
    ):
        raise ValueError("Eq. (10) arrays do not descend from this bootstrap reference")
    if transition_input.sample_node_ids[: len(am.reference_scan_ids)] != am.reference_scan_ids:
        raise ValueError("AP transition LP input changed reference sample ordering")
    if transition_input.bssid_vocabulary[: len(am.bssid_vocabulary)] != am.bssid_vocabulary:
        raise ValueError("AP transition input did not preserve the old vocabulary prefix")
    if transition_input.encoder_checkpoint_sha256 != am.source_ae_checkpoint_sha256:
        raise ValueError("AP transition and Eq. (10) update use different source AE checkpoints")
    if transition_lp.manifest.gnn_checkpoint_sha256 != transition_input.gnn_checkpoint_sha256:
        raise ValueError("AP transition scores use a different learned GNN checkpoint")
    candidate_digest = hashlib.sha256()
    observed_candidate_count = 0
    for score in transition_lp.iter_candidate_scores():
        candidate_digest.update(
            score.sample_node_id.encode("utf-8")
            + b"\0"
            + score.ap_node_id.encode("utf-8")
            + b"\n"
        )
        observed_candidate_count += 1
    if (
        candidate_digest.hexdigest() != transition_lp.manifest.candidate_set_sha256
        or observed_candidate_count != transition_lp.manifest.candidate_count
    ):
        raise ValueError("AP transition candidate score stream disagrees with its recorded set digest")

    target = Path(output_directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite streaming reference transition: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    db_path = staging / "reference_edges.sqlite3"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute(
            "CREATE TABLE edges (sample_id TEXT NOT NULL, ap_id TEXT NOT NULL, "
            "weight REAL NOT NULL, PRIMARY KEY(sample_id, ap_id)) WITHOUT ROWID"
        )
        reference_ids = am.reference_scan_ids
        reference_set = set(reference_ids)
        base_endpoints = np.load(
            Path(bootstrap_lp_result_directory) / "updated_edge_endpoints.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        base_weights = np.load(
            Path(bootstrap_lp_result_directory) / "updated_edge_weights.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        for edge_index, (sample_row, ap_row) in enumerate(base_endpoints):
            sample_id = base.node_ids[int(sample_row)]
            ap_id = base.node_ids[int(ap_row)]
            if sample_id in reference_set:
                weight = float(base_weights[edge_index])
                connection.execute(
                    "INSERT INTO edges VALUES (?, ?, ?)", (sample_id, ap_id, weight)
                )
        connection.commit()

        old_ap_ids = set(base_lp.manifest.active_ap_node_ids)
        for change in transition_lp.iter_removed_edges():
            if change.sample_node_id not in reference_set:
                raise ValueError("AP transition attempted to remove a temporary-sample edge")
            deleted = connection.execute(
                "DELETE FROM edges WHERE sample_id=? AND ap_id=?",
                (change.sample_node_id, change.ap_node_id),
            ).rowcount
            if deleted != 1:
                raise ValueError("AP transition removal does not match a retained reference edge")
        admitted: set[str] = set()
        for change in transition_lp.iter_added_edges():
            if change.sample_node_id not in reference_set or change.new_weight is None:
                raise ValueError("AP transition attempted to add an invalid reference edge")
            if change.ap_node_id not in transition_input.ap_node_ids:
                raise ValueError("AP transition addition references an absent AP node")
            bssid = change.ap_node_id.removeprefix("ap:")
            admitted.add(bssid)
            connection.execute(
                "INSERT INTO edges VALUES (?, ?, ?)",
                (change.sample_node_id, change.ap_node_id, float(change.new_weight)),
            )
        connection.commit()
        degree_by_ap = {
            ap_id: int(count)
            for ap_id, count in connection.execute(
                "SELECT ap_id, COUNT(*) FROM edges GROUP BY ap_id"
            )
        }
        removed = tuple(
            sorted(
                ap_id.removeprefix("ap:")
                for ap_id in old_ap_ids
                if degree_by_ap.get(ap_id, 0) == 0
            )
        )
        current_vocabulary = am.bssid_vocabulary
        incoming = set(observed_bssids(update_prepared))
        removed_set = set(removed)
        still_active = tuple(bssid for bssid in current_vocabulary if bssid not in removed_set)
        admitted_to_reference = tuple(
            sorted((incoming - set(still_active)) & admitted)
        )
        target_vocabulary = still_active + tuple(
            bssid for bssid in admitted_to_reference if bssid not in set(still_active)
        )
        if not target_vocabulary:
            raise ValueError("AP forgetting would leave an empty AE vocabulary")
        if len(set(target_vocabulary)) != len(target_vocabulary):
            raise ValueError("streamed transition created duplicate target AP identities")

        reference_row = {scan_id: row for row, scan_id in enumerate(reference_ids)}
        old_column = {bssid: col for col, bssid in enumerate(current_vocabulary)}
        target_column = {bssid: col for col, bssid in enumerate(target_vocabulary)}
        updated = _open_array(
            staging / "updated_rssi_dbm.npy", np.float32,
            (len(reference_ids), len(target_vocabulary)),
        )
        masks = _open_array(
            staging / "observation_mask.npy", np.uint8,
            (len(reference_ids), len(target_vocabulary)),
        )
        provenance = _open_array(
            staging / "value_provenance.npy", np.uint8,
            (len(reference_ids), len(target_vocabulary)),
        )
        updated[:] = -120.0
        masks[:] = 0
        provenance[:] = 3
        for target_col, bssid in enumerate(target_vocabulary):
            source_col = old_column.get(bssid)
            if source_col is None:
                continue
            updated[:, target_col] = ae_update.updated_rssi_dbm[:, source_col]
            masks[:, target_col] = ae_update.observation_mask[:, source_col]
            source_codes = ae_update.value_provenance[:, source_col]
            for row in range(len(reference_ids)):
                if int(source_codes[row]) == 1:
                    provenance[row, target_col] = 1
                elif bool(masks[row, target_col]):
                    provenance[row, target_col] = 0
                else:
                    provenance[row, target_col] = 3

        for change in transition_lp.iter_added_edges():
            bssid = change.ap_node_id.removeprefix("ap:")
            target_col = target_column.get(bssid)
            if target_col is None:
                raise ValueError("admitted edge AP is absent from the planned vocabulary")
            row = reference_row[change.sample_node_id]
            updated[row, target_col] = float(change.new_weight) - 120.0
            masks[row, target_col] = 0
            provenance[row, target_col] = 2
        for array in (updated, masks, provenance):
            if isinstance(array, np.memmap):
                array.flush()

        arrays = tuple(
            _array_artifact(staging / name, name)
            for name in (
                "updated_rssi_dbm.npy",
                "observation_mask.npy",
                "value_provenance.npy",
            )
        )
        connection.close()
        edge_artifact = StreamingFileArtifact(
            name="reference_edges.sqlite3",
            sha256=sha256_file(db_path),
            size_bytes=db_path.stat().st_size,
        )
        fields = dict(
            parent_graph_id=base.graph_id,
            parent_lp_result_id=base_lp.manifest.result_id,
            ap_lp_result_id=transition_lp.manifest.result_id,
            ae_update_id=am.update_id,
            dataset_id=am.dataset_id,
            reference_batch_id=am.reference_batch_id,
            update_batch_id=am.update_batch_id,
            reference_scan_ids=reference_ids,
            original_coordinates=am.original_coordinates,
            update_scan_ids=am.update_scan_ids,
            vocabulary_before=current_vocabulary,
            vocabulary_after=target_vocabulary,
            vocabulary_before_sha256=canonical_json_digest(current_vocabulary),
            vocabulary_after_sha256=canonical_json_digest(target_vocabulary),
            removed_by_forgetting_bssids=removed,
            admitted_bssids=admitted_to_reference,
            normalization=am.normalization,
            threshold_value=transition_lp.manifest.threshold_value,
            threshold_units=transition_lp.manifest.threshold_units,
            arrays=arrays,
            reference_edges=edge_artifact,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        provisional = StreamingReferenceTransitionManifest.model_construct(
            transition_id="streamtransition:" + "0" * 64, **fields
        )
        transition_id = "streamtransition:" + canonical_json_digest(
            provisional.model_dump(mode="json", exclude={"transition_id"})
        )
        manifest = StreamingReferenceTransitionManifest(
            transition_id=transition_id, **fields
        )
        atomic_write_json(manifest, staging / "manifest.json")
        atomic_write_json(
            {
                "format": "gufu_streaming_artifact_complete_v1",
                "identity": manifest.transition_id,
                "manifest_sha256": sha256_file(staging / "manifest.json"),
            },
            staging / "COMPLETED.json",
        )
        if target.exists():
            raise FileExistsError(f"refusing to overwrite streaming transition: {target}")
        os.replace(staging, target)
        arrays = {
            item.name: np.load(target / item.name, mmap_mode="r", allow_pickle=False)
            for item in manifest.arrays
        }
        return StreamingReferenceTransitionState(
            directory=target,
            manifest=manifest,
            updated_rssi_dbm=arrays["updated_rssi_dbm.npy"],
            observation_mask=arrays["observation_mask.npy"],
            value_provenance=arrays["value_provenance.npy"],
        )
    except Exception:
        connection.close()
        # Leave diagnostic staging artifacts without a completion marker.
        raise


def load_streaming_reference_transition(
    directory: str | Path,
    *,
    verify_hashes: bool = True,
) -> StreamingReferenceTransitionState:
    root = Path(directory)
    manifest_path = root / "manifest.json"
    marker_path = root / "COMPLETED.json"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise ValueError(f"streaming reference transition is incomplete: {root}")
    manifest = StreamingReferenceTransitionManifest.model_validate_json(
        read_json_text(manifest_path)
    )
    marker = json.loads(read_json_text(marker_path))
    if (
        marker.get("identity") != manifest.transition_id
        or marker.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise ValueError("streaming reference transition completion marker is inconsistent")
    edge_path = root / manifest.reference_edges.name
    if not edge_path.is_file() or edge_path.stat().st_size != manifest.reference_edges.size_bytes:
        raise ValueError("streaming reference edge table is missing or truncated")
    if verify_hashes and sha256_file(edge_path) != manifest.reference_edges.sha256:
        raise ValueError("streaming reference edge table hash mismatch")
    arrays = {}
    for item in manifest.arrays:
        path = root / item.name
        if not path.is_file() or path.stat().st_size != item.size_bytes:
            raise ValueError(f"streaming transition array is missing or truncated: {item.name}")
        if verify_hashes and sha256_file(path) != item.sha256:
            raise ValueError(f"streaming transition array hash mismatch: {item.name}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if tuple(array.shape) != item.shape or array.dtype.name != item.dtype:
            raise ValueError(f"streaming transition array shape/dtype mismatch: {item.name}")
        arrays[item.name] = array
    return StreamingReferenceTransitionState(
        directory=root,
        manifest=manifest,
        updated_rssi_dbm=arrays["updated_rssi_dbm.npy"],
        observation_mask=arrays["observation_mask.npy"],
        value_provenance=arrays["value_provenance.npy"],
    )


__all__ = [
    "StreamingReferenceTransitionManifest",
    "StreamingReferenceTransitionState",
    "apply_streaming_ap_delta_to_reference",
    "load_streaming_reference_transition",
]
