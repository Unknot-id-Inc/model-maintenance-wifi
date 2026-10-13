import pytest

from gufu_data import EvaluationPosition, PreprocessingConfig, prepare_dataset_files
from gufu_evaluation import (
    evaluate_streamed_reference_rssi,
    evaluate_streamed_update_localization,
)


def test_streamed_localization_joins_only_by_scan_id(tmp_path):
    path = tmp_path / "update.txt"
    path.write_text("1,2 001122334455,-60\n3,4 001122334455,-70\n", encoding="utf-8")
    prepared = prepare_dataset_files(
        [path],
        PreprocessingConfig(
            dataset_id="eval",
            batch_id="u1",
            batch_role="update_batch",
        ),
        source_root=tmp_path,
    )
    result = evaluate_streamed_update_localization(
        (prepared.scans[1].scan_id, prepared.scans[0].scan_id),
        ((3.0, 5.0), (2.0, 2.0)),
        prepared.evaluation_positions,
    )
    assert result.paired_scans == 2
    assert result.rmse == pytest.approx(1.0)
    with pytest.raises(ValueError, match="do not align"):
        evaluate_streamed_update_localization(
            (prepared.scans[0].scan_id,), ((0.0, 0.0), (0.0, 0.0)), prepared.evaluation_positions
        )


def test_streamed_reference_rssi_excludes_prior_predictions_and_missing_cells(tmp_path):
    path = tmp_path / "reference.txt"
    path.write_text(
        "0,0 001122334455,-60 001122334466,-70\n"
        "1,1 001122334455,-80\n",
        encoding="utf-8",
    )
    prepared = prepare_dataset_files(
        [path],
        PreprocessingConfig(dataset_id="eval", batch_id="b0", batch_role="bootstrap_reference"),
        source_root=tmp_path,
    )
    metric = evaluate_streamed_reference_rssi(
        prepared,
        ((-58.0, -70.0), (-75.0, -120.0)),
        prepared.bssid_vocabulary,
    )
    assert metric.measured_cells == 3
    assert metric.rmse_dbm == pytest.approx((29.0 / 3.0) ** 0.5)
    assert metric.mae_dbm == pytest.approx(7.0 / 3.0)
