"""Post-fit metrics with explicit measured/evaluation-only data provenance.

These functions are reporting-only. They do not modify model, graph, or
fingerprint artifacts and are never called by a fitting/update stage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import torch

from fingerprint_update import (
    LocationPredictor,
    UpdatedFingerprintArtifact,
    UpdateLocationScaler,
)
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


def train_reference_location_mlp_predictions(
    reference_embeddings: Sequence[Sequence[float]] | np.ndarray,
    reference_coordinates: Sequence[Sequence[float]] | np.ndarray,
    query_embeddings: Sequence[Sequence[float]] | np.ndarray,
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Fit the same supervised location MLP on reference labels and score queries.

    This is an evaluation-only probe: only the supplied reference embeddings and
    their bootstrap coordinates train the MLP. Query coordinates are never read.
    """
    x_ref = np.asarray(reference_embeddings, dtype=np.float32)
    xy_ref = np.asarray(reference_coordinates, dtype=np.float32)
    x_query = np.asarray(query_embeddings, dtype=np.float32)
    if x_ref.ndim != 2 or x_query.ndim != 2 or x_ref.shape[1] != x_query.shape[1]:
        raise ValueError("reference/query embedding matrices must have matching widths")
    if xy_ref.shape != (len(x_ref), 2) or len(x_ref) < 1 or len(x_query) < 1:
        raise ValueError("reference coordinates and embeddings are not aligned")
    if not np.isfinite(x_ref).all() or not np.isfinite(x_query).all() or not np.isfinite(xy_ref).all():
        raise ValueError("localization MLP inputs contain non-finite values")
    if hidden_dim < 1 or epochs < 1 or learning_rate <= 0.0:
        raise ValueError("localization MLP training configuration is invalid")

    scaler = UpdateLocationScaler(
        x_min=float(xy_ref[:, 0].min()),
        x_max=float(xy_ref[:, 0].max()),
        y_min=float(xy_ref[:, 1].min()),
        y_max=float(xy_ref[:, 1].max()),
    )
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"requested CUDA device is unavailable: {selected_device}")
    x_train = torch.as_tensor(x_ref, dtype=torch.float32, device=selected_device)
    y_raw = torch.as_tensor(xy_ref, dtype=torch.float32, device=selected_device)
    x_test = torch.as_tensor(x_query, dtype=torch.float32, device=selected_device)
    y_train = scaler.encode(y_raw)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = LocationPredictor(hidden_dim, x_ref.shape[1]).to(selected_device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            residual = model(x_train) - y_train
            loss = torch.linalg.vector_norm(residual)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("reference-only localization MLP loss is non-finite")
            loss.backward()
            gradients = [parameter.grad for parameter in model.parameters()]
            if not any(gradient is not None for gradient in gradients):
                raise RuntimeError("reference-only localization MLP received no gradient")
            if not any(torch.count_nonzero(gradient).item() for gradient in gradients):
                break
            optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = scaler.decode(model(x_test))
    result = prediction.detach().cpu().numpy().astype(np.float64, copy=False)
    if not np.isfinite(result).all():
        raise FloatingPointError("localization MLP produced non-finite coordinates")
    return result


def compare_streamed_localization_predictions(
    update_scan_ids: Sequence[str],
    original_predictions: Sequence[Sequence[float]],
    updated_predictions: Sequence[Sequence[float]],
    evaluation_positions: Sequence[EvaluationPosition],
    *,
    original_estimator: str,
    updated_estimator: str,
) -> dict:
    """Create paired aggregate and per-scan localization errors by scan ID."""
    ids = tuple(update_scan_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("localization comparison update IDs contain duplicates")
    if len(original_predictions) != len(ids) or len(updated_predictions) != len(ids):
        raise ValueError("localization comparison predictions do not align with update IDs")
    truth_by_id = {item.scan_id: (float(item.x), float(item.y)) for item in evaluation_positions}
    if len(truth_by_id) != len(evaluation_positions):
        raise ValueError("evaluation positions contain duplicate scan IDs")
    paired_ids = tuple(scan_id for scan_id in ids if scan_id in truth_by_id)
    if not paired_ids:
        raise ValueError("no update scans have evaluation-only truth")
    original_by_id = {}
    updated_by_id = {}
    for scan_id, original, updated in zip(ids, original_predictions, updated_predictions):
        if len(original) != 2 or len(updated) != 2:
            raise ValueError("each localization prediction must contain two coordinates")
        original_point = (float(original[0]), float(original[1]))
        updated_point = (float(updated[0]), float(updated[1]))
        if not all(math.isfinite(value) for value in (*original_point, *updated_point)):
            raise ValueError("localization comparison contains non-finite coordinates")
        original_by_id[scan_id] = original_point
        updated_by_id[scan_id] = updated_point

    rows = []
    original_squared, updated_squared = [], []
    original_radial, updated_radial = [], []
    original_dx, original_dy, updated_dx, updated_dy = [], [], [], []
    for scan_id in paired_ids:
        truth_x, truth_y = truth_by_id[scan_id]
        original_x, original_y = original_by_id[scan_id]
        updated_x, updated_y = updated_by_id[scan_id]
        odx, ody = original_x - truth_x, original_y - truth_y
        udx, udy = updated_x - truth_x, updated_y - truth_y
        original_error = math.hypot(odx, ody)
        updated_error = math.hypot(udx, udy)
        original_squared.append(original_error * original_error)
        updated_squared.append(updated_error * updated_error)
        original_radial.append(original_error)
        updated_radial.append(updated_error)
        original_dx.append(odx)
        original_dy.append(ody)
        updated_dx.append(udx)
        updated_dy.append(udy)
        rows.append(
            {
                "scan_id": scan_id,
                "truth_x": truth_x,
                "truth_y": truth_y,
                "original_prediction_x": original_x,
                "original_prediction_y": original_y,
                "updated_prediction_x": updated_x,
                "updated_prediction_y": updated_y,
                "original_error": original_error,
                "updated_error": updated_error,
                "updated_minus_original_error": updated_error - original_error,
                "prediction_shift": math.hypot(updated_x - original_x, updated_y - original_y),
            }
        )
    original_rmse = math.sqrt(sum(original_squared) / len(original_squared))
    updated_rmse = math.sqrt(sum(updated_squared) / len(updated_squared))
    original_mae = sum(original_radial) / len(original_radial)
    updated_mae = sum(updated_radial) / len(updated_radial)
    return {
        "metric": "paired_held_out_update_localization_comparison",
        "truth_source": "separate_evaluation_only_positions",
        "coordinate_units": "source_coordinate_units",
        "original_estimator": original_estimator,
        "updated_estimator": updated_estimator,
        "paired_scans": len(paired_ids),
        "original_2d_rmse": original_rmse,
        "updated_2d_rmse": updated_rmse,
        "rmse_delta_updated_minus_original": updated_rmse - original_rmse,
        "rmse_change_percent": (
            100.0 * (updated_rmse / original_rmse - 1.0)
            if original_rmse > 0.0
            else None
        ),
        "original_mean_radial_error": original_mae,
        "updated_mean_radial_error": updated_mae,
        "original_median_radial_error": float(np.median(original_radial)),
        "updated_median_radial_error": float(np.median(updated_radial)),
        "original_p90_radial_error": float(np.percentile(original_radial, 90)),
        "updated_p90_radial_error": float(np.percentile(updated_radial, 90)),
        "original_mean_x_bias": sum(original_dx) / len(original_dx),
        "original_mean_y_bias": sum(original_dy) / len(original_dy),
        "updated_mean_x_bias": sum(updated_dx) / len(updated_dx),
        "updated_mean_y_bias": sum(updated_dy) / len(updated_dy),
        "scans_improved": sum(new < old for old, new in zip(original_radial, updated_radial)),
        "scans_worsened": sum(new > old for old, new in zip(original_radial, updated_radial)),
        "scans_unchanged": sum(new == old for old, new in zip(original_radial, updated_radial)),
        "per_scan": rows,
    }


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
