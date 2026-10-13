from __future__ import annotations

import ast
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_SOURCE = ROOT / "gufu.py"


def _cells(source: str) -> tuple[str, ...]:
    parts = re.split(r"(?m)^# %%\s*$", source)
    return tuple(part.strip() for part in parts if part.strip())


def test_gufu_source_has_nine_ordered_parseable_stages():
    source = NOTEBOOK_SOURCE.read_text(encoding="utf-8")
    cells = _cells(source)

    assert len(cells) == 9
    for cell in cells:
        ast.parse(cell)

    expected = (
        "1. Imports, explicit run configuration",
        "2. Prepare/load the authoritative bootstrap dataset",
        "3. Train/load the AE",
        "4. Build/load the graph",
        "5. Train/load the GNN",
        "6. Inspect link modification",
        "7. Demonstrate the updater",
        "8. Run/resume the sequential orchestrator",
        "9. Optional evaluation",
    )
    positions = tuple(source.index(title) for title in expected)
    assert positions == tuple(sorted(positions))
    assert source.count("# %%") == 9


def test_gufu_has_explicit_modes_shared_pipeline_and_safe_file_handling():
    source = NOTEBOOK_SOURCE.read_text(encoding="utf-8")

    assert 'RUN_MODE = "lp_streaming"' in source
    assert 'if RUN_MODE == "full":' in source
    assert 'DATA_PROFILE = "real_dataset"' in source
    assert 'AE_LOSS = "mean_squared_error"' in source
    assert '"batch_frobenius_norm": "frobenius"' in source
    assert "AE_DROPOUT_PROBABILITY = " in source
    assert "AE_DENSE_WIDTH = 256" in source
    assert "AE_LATENT_DIM = 32" in source
    assert "graph_feature_dim=AE_LATENT_DIM + 2" in source
    assert "edge_feature_dim=AE_LATENT_DIM + 2" in source
    assert "if AE_LATENT_DIM != 32" not in source
    assert source.count("dropout_probability=AE_DROPOUT_PROBABILITY") == 2
    assert source.count("dense_width=AE_DENSE_WIDTH") == 2
    assert source.count("latent_dim=AE_LATENT_DIM") == 2
    assert 'loss=AE_LOSS' in source
    assert 'd{AE_DENSE_WIDTH}-z{AE_LATENT_DIM}-drop{' in source
    assert '"cpu_smoke"' in source
    assert 'virtual_edge_sigma=1.0' in source
    assert 'RUN_ID = (' in source
    assert 'RUN_DIR.mkdir(parents=True, exist_ok=True)' in source
    assert 'Path("output") / "pipelines" / DATASET_ID / RUN_ID' in source
    assert 'change RUN_ID for a fresh run' in source
    assert "run_sequential(" in source
    assert "from pred import run_sequential" in source
    assert "RUN_EVALUATION = False" in source
    assert "evaluate_measured_reference_rssi" in source
    assert "evaluate_update_localization" in source
    assert "Measured-reference RSSI" in source
    assert "Held-out localization" in source
    assert "save_prepared_dataset" in source
    assert "shutil.rmtree" not in source
    assert "os.system" not in source
    assert "from lp.py import" not in source
    assert "load_file_comparison" not in source
    assert "cur_record" not in source
    assert "torch.Tensor(cur_record)" not in source
    assert "fingerprint2.txt" not in source
    assert '"lp_streaming"' in source
    assert "predict_edges_streaming(" in source
    assert source.count("run_streaming_sequential(") >= 2
    assert "bootstrap_lp_result_directory=STREAMING_LP_RESULT_DIR" in source
    assert "STREAMING_PIPELINE_CONFIG_PATH" in source
    assert "evaluate_streamed_update_localization" in source
    assert "legacy in-memory" in source
    assert "resume=PIPELINE_RESUME" in source

