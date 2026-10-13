"""Post-fit metrics with explicit measured/evaluation-only data provenance.

These functions are reporting-only. They do not modify model, graph, or
fingerprint artifacts and are never called by a fitting/update stage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from fingerprint_update import UpdatedFingerprintArtifact
from gufu_data import EvaluationPosition
from gufu_data import PreparedDataset


class ReferenceRSSIMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    metric: Literal["measured_reference_rssi_agreement"] = (
        "measured_reference_rssi_agreement"
    )
    unit: Literal["dBm"] = "dBm"
    measured_cells: int = Field(ge=1)
    rmse_dbm: float = Field(ge=0.0)
    mae_dbm: float = Field(ge=0.0)
    evaluation_scope: Literal["observed_reference_cells_not_held_out"] = (
        "observed_reference_cells_not_held_out"
    )


class LocalizationMetric(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    metric: Literal["held_out_update_localization_rmse"] = (
        "held_out_update_localization_rmse"
    )
    estimator: Literal["location_mlp_from_final_z_gnn"] = (
        "location_mlp_from_final_z_gnn"
    )
    truth_source: Literal["separate_evaluation_only_positions"] = (
        "separate_evaluation_only_positions"
    )
    coordinate_units: Literal["source_coordinate_units"] = "source_coordinate_units"
    paired_scans: int = Field(ge=1)
    rmse: float = Field(ge=0.0)


def evaluate_measured_reference_rssi(
    artifact: UpdatedFingerprintArtifact,
) -> ReferenceRSSIMetric:
    """Compare updated reference cells to measured dBm values under their mask.

    This is an in-sample consistency diagnostic: the reference measurements
    also participate in fitting. Missing cells are excluded by the stored
    observation mask, never inferred from the RSS sentinel.
    """
    squared_errors: list[float] = []
    absolute_errors: list[float] = []
    for row in artifact.rows:
        for measured, updated, observed in zip(
            row.raw_rssi_dbm, row.updated_rssi_dbm, row.observation_mask
        ):
            if observed:
                difference = updated - measured
                squared_errors.append(difference * difference)
                absolute_errors.append(abs(difference))
    if not squared_errors:
        raise ValueError("RSSI metric has no observed reference cells")
    return ReferenceRSSIMetric(
        measured_cells=len(squared_errors),
        rmse_dbm=math.sqrt(sum(squared_errors) / len(squared_errors)),
        mae_dbm=sum(absolute_errors) / len(absolute_errors),
    )


def evaluate_update_localization(
    artifact: UpdatedFingerprintArtifact,
    evaluation_positions: Sequence[EvaluationPosition],
) -> LocalizationMetric:
    """Score the named location-MLP estimate against separately held-out truth.

    Truth is joined by stable scan ID. It is accepted only as a separate
    evaluation argument and is not part of any fitting/update API.
    """
    if not evaluation_positions:
        raise ValueError("localization metric requires evaluation-only positions")
    truth_by_id = {item.scan_id: (item.x, item.y) for item in evaluation_positions}
    if len(truth_by_id) != len(evaluation_positions):
        raise ValueError("evaluation positions contain duplicate scan IDs")
    predicted_by_id = {
        item.scan_id: (item.x, item.y)
        for item in artifact.predicted_update_locations
    }
    if len(predicted_by_id) != len(artifact.predicted_update_locations):
        raise ValueError("predicted locations contain duplicate scan IDs")
    if any(
        item.source != "location_mlp_from_final_z_gnn"
        for item in artifact.predicted_update_locations
    ):
        raise ValueError("localization evaluator received an unknown estimator")
    paired_ids = tuple(
        scan_id for scan_id in artifact.update_scan_ids
        if scan_id in truth_by_id and scan_id in predicted_by_id
    )
    if not paired_ids:
        raise ValueError("no update scan IDs have both a prediction and evaluation truth")
    squared_errors = []
    for scan_id in paired_ids:
        predicted_x, predicted_y = predicted_by_id[scan_id]
        truth_x, truth_y = truth_by_id[scan_id]
        squared_errors.append(
            (predicted_x - truth_x) ** 2 + (predicted_y - truth_y) ** 2
        )
    return LocalizationMetric(
        paired_scans=len(paired_ids),
        rmse=math.sqrt(sum(squared_errors) / len(squared_errors)),
    )


def evaluate_streamed_update_localization(
    update_scan_ids: Sequence[str],
    predicted_locations: Sequence[Sequence[float]],
    evaluation_positions: Sequence[EvaluationPosition],
) -> LocalizationMetric:
    """Evaluate a disk-loaded, ID-aligned location array without model state."""
    ids = tuple(update_scan_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("streamed update IDs contain duplicates")
    if len(predicted_locations) != len(ids):
        raise ValueError("streamed predicted-location rows do not align with scan IDs")
    predicted_by_id = {}
    for scan_id, row in zip(ids, predicted_locations):
        if len(row) != 2:
            raise ValueError("each streamed location prediction must have two coordinates")
        point = (float(row[0]), float(row[1]))
        if not all(math.isfinite(value) for value in point):
            raise ValueError("streamed location prediction is not finite")
        predicted_by_id[scan_id] = point
    if not evaluation_positions:
        raise ValueError("localization metric requires evaluation-only positions")
    truth_by_id = {item.scan_id: (item.x, item.y) for item in evaluation_positions}
    if len(truth_by_id) != len(evaluation_positions):
        raise ValueError("evaluation positions contain duplicate scan IDs")
    paired_ids = tuple(scan_id for scan_id in ids if scan_id in truth_by_id)
    if not paired_ids:
        raise ValueError("no streamed prediction IDs have evaluation-only truth")
    squared = [
        (predicted_by_id[scan_id][0] - truth_by_id[scan_id][0]) ** 2
        + (predicted_by_id[scan_id][1] - truth_by_id[scan_id][1]) ** 2
        for scan_id in paired_ids
    ]
    return LocalizationMetric(
        paired_scans=len(paired_ids),
        rmse=math.sqrt(sum(squared) / len(squared)),
    )


def evaluate_streamed_reference_rssi(
    source_reference: PreparedDataset,
    updated_rssi_dbm: Sequence[Sequence[float]],
    target_bssid_vocabulary: Sequence[str],
) -> ReferenceRSSIMetric:
    """Compare only cells still known to contain measured source RSSI.

    Cells previously replaced by an AE/edge estimate are excluded even when
    their historical observation mask remains true.
    """
    target_vocabulary = tuple(target_bssid_vocabulary)
    if len(updated_rssi_dbm) != len(source_reference.scans):
        raise ValueError("streamed reference RSS rows do not align with source scans")
    if any(len(row) != len(target_vocabulary) for row in updated_rssi_dbm):
        raise ValueError("streamed reference RSS width does not match target vocabulary")
    target_column = {bssid: column for column, bssid in enumerate(target_vocabulary)}
    source_column = {
        bssid: column for column, bssid in enumerate(source_reference.bssid_vocabulary)
    }
    squared = []
    absolute = []
    for row, scan in enumerate(source_reference.scans):
        provenance = scan.value_provenance
        for bssid, old_column in source_column.items():
            new_column = target_column.get(bssid)
            if new_column is None or not scan.observation_mask[old_column]:
                continue
            if provenance is not None and provenance[old_column] != "measured":
                continue
            delta = float(updated_rssi_dbm[row][new_column]) - scan.rss_dbm[old_column]
            if not math.isfinite(delta):
                raise ValueError("streamed RSSI comparison contains a non-finite value")
            squared.append(delta * delta)
            absolute.append(abs(delta))
    if not squared:
        raise ValueError("RSSI metric has no source cells with measured provenance")
    return ReferenceRSSIMetric(
        measured_cells=len(squared),
        rmse_dbm=math.sqrt(sum(squared) / len(squared)),
        mae_dbm=sum(absolute) / len(absolute),
    )
