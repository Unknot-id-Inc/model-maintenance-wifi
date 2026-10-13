from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from ae import AETrainingConfig, encode_prepared_dataset, train_autoencoder
from batch_transition import TransitionConfig
from fingerprint_update import FingerprintUpdateConfig
from graph_build import GraphBuildConfig, build_graph_state
from gufu_artifacts import canonical_json_digest
from gufu_data import PreprocessingConfig, prepare_dataset_files, save_prepared_dataset
from lp import LinkPredictionConfig
from lp_streaming import load_streaming_lp_input, predict_edges_streaming, save_streaming_lp_input
from sage_ve import (
    EdgeFeatureSAGE,
    GNNModelConfig,
    GNNTrainingConfig,
    load_gnn_checkpoint,
    train_gnn,
)
from streaming_pipeline import StreamingPipelineConfig, run_streaming_sequential


def _bootstrap(tmp_path):
    bootstrap_path = tmp_path / "fingerprint0.txt"
    bootstrap_path.write_text(
        "0,0 001122334455,-50 001122334466,-70\n"
        "10,10 001122334455,-55 001122334466,-65\n"
        "20,20 001122334455,-60 001122334466,-60\n",
        encoding="utf-8",
    )
    prepared = prepare_dataset_files(
        [bootstrap_path],
        PreprocessingConfig(
            dataset_id="stream-test",
            batch_id="fingerprint0",
            batch_role="bootstrap_reference",
            normalization_mode="paper_ae",
        ),
        source_root=tmp_path,
    )
    ae_path = tmp_path / "ae.pt"
    train_autoencoder(
        prepared,
        ae_path,
        AETrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            seed=12,
            shuffle=False,
            dropout_probability=0.0,
            dense_width=16,
            latent_dim=8,
            dtype="float32",
        ),
        log_every_epochs=1,
    )
    ae_features = encode_prepared_dataset(
        prepared, ae_path, tmp_path / "ae-features.json", batch_size=2
    )
    graph_config = GraphBuildConfig(
        virtual_edge_strategy="paper_cosine_threshold",
        virtual_edge_sigma=-1.0,
        pseudo_location_seed=12,
    )
    graph = build_graph_state(prepared, ae_features, graph_config)
    gnn_path = tmp_path / "gnn.pt"
    train_gnn(
        graph,
        EdgeFeatureSAGE(
            GNNModelConfig(
                graph_feature_dim=graph.node_feature_dim,
                edge_feature_dim=graph.node_feature_dim,
                node_output_dim=graph.node_feature_dim,
                edge_output_dim=graph.node_feature_dim,
                num_layers=1,
                dropout_probability=0.0,
                initialization_seed=13,
            )
        ),
        gnn_path,
        tmp_path / "gnn-features.json",
        training_config=GNNTrainingConfig(
            epochs=1, learning_rate=0.001, seed=13, objective_epsilon=1e-8
        ),
        log_every_epochs=1,
    )
    from sage_ve import load_gnn_features

    gnn_features = load_gnn_features(tmp_path / "gnn-features.json")
    lp_input_dir = tmp_path / "bootstrap-lp-input"
    save_streaming_lp_input(graph, gnn_features, lp_input_dir)
    link = LinkPredictionConfig(
        max_iterations=2, convergence_epsilon=1e9, link_threshold=0.0
    )
    lp_result_dir = tmp_path / "bootstrap-lp-result"
    predict_edges_streaming(lp_input_dir, lp_result_dir, link)
    loaded_gnn = load_gnn_checkpoint(gnn_path, graph, device="cpu")
    config = StreamingPipelineConfig(
        dataset_id="stream-test",
        preprocessing=prepared.config,
        graph_build=graph_config,
        gnn_training=GNNTrainingConfig(
            epochs=1, learning_rate=0.001, seed=20, objective_epsilon=1e-8
        ),
        link_prediction=link,
        fingerprint_update=FingerprintUpdateConfig(
            hidden_dim=8,
            alpha=0.5,
            virtual_edge_sigma=graph_config.virtual_edge_sigma,
            updater_epochs=1,
            ae_epochs=1,
            seed=21,
        ),
        transition=TransitionConfig(
            link_prediction=link,
            ae_finetune_epochs=1,
            ae_finetune_seed=22,
            reinitialize_seed=23,
        ),
        device="cpu",
        seed=20,
        gnn_chunk_size=2,
        updater_chunk_size=2,
        ae_chunk_size=2,
        encoder_batch_size=2,
        log_every_epochs=1,
    )
    return bootstrap_path, prepared, ae_path, loaded_gnn, lp_input_dir, lp_result_dir, config


def _updates(tmp_path, first_xy=(88, 99), second_xy=(-7, 34)):
    first = tmp_path / "fingerprint1.txt"
    first.write_text(
        f"{first_xy[0]},{first_xy[1]} 001122334455,-42 001122334466,-78 001122334477,-60\n"
        f"{first_xy[1]},{first_xy[0]} 001122334455,-68 001122334466,-49 001122334477,-80\n",
        encoding="utf-8",
    )
    second = tmp_path / "fingerprint2.txt"
    second.write_text(
        f"{second_xy[0]},{second_xy[1]} 001122334455,-45 001122334466,-74 001122334477,-61\n"
        f"{second_xy[1]},{second_xy[0]} 001122334455,-66 001122334466,-52 001122334477,-79\n",
        encoding="utf-8",
    )
    return first, second


def _run(root, bootstrap, updates):
    bootstrap_path, prepared, ae_path, gnn, lp_input, lp_result, config = bootstrap
    return run_streaming_sequential(
        root,
        config,
        bootstrap_prepared=prepared,
        bootstrap_ae_checkpoint_path=ae_path,
        bootstrap_gnn_checkpoint=gnn,
        bootstrap_lp_input_directory=lp_input,
        bootstrap_lp_result_directory=lp_result,
        bootstrap_paths=(bootstrap_path,),
        update_paths=updates,
        source_root=bootstrap_path.parent,
    )


def test_two_batch_streaming_snapshots_resume_from_completed_state(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    bootstrap = _bootstrap(data)
    first, second = _updates(data)
    third = data / "fingerprint3.txt"
    third.write_text(
        "11,12 001122334455,-46 001122334466,-73 001122334477,-62\n"
        "13,14 001122334455,-65 001122334466,-53 001122334477,-78\n",
        encoding="utf-8",
    )
    root = tmp_path / "resumable"
    root.mkdir()
    unrelated = root / "keep-me.txt"
    unrelated.write_text("user-owned\n", encoding="utf-8")

    first_result = _run(root, bootstrap, (first,))
    assert first_result.completed_batch_ids == ("fingerprint1",)
    first_snapshot = first_result.latest_snapshot
    assert first_snapshot is not None
    assert first_snapshot.reference_scan_ids == tuple(
        scan.scan_id for scan in bootstrap[1].scans
    )
    assert len(first_snapshot.reference_scan_ids) == 3
    assert len(first_snapshot.update_scan_ids) == 2
    assert first_snapshot.vocabulary_after == bootstrap[1].bssid_vocabulary + (
        "001122334477",
    )

    resumed = run_streaming_sequential(
        root,
        bootstrap[6],
        bootstrap_paths=(bootstrap[0],),
        update_paths=(first, second, third),
        source_root=data,
    )
    assert resumed.completed_batch_ids == ("fingerprint2", "fingerprint3")
    assert resumed.skipped_completed_batch_ids == ("fingerprint1",)
    final = resumed.latest_snapshot
    assert final is not None
    manifests = [
        json.loads(path.read_text())
        for path in sorted((root / "snapshots").glob("*/manifest.json"))
    ]
    second_snapshot_id = next(item["snapshot_id"] for item in manifests if item["batch_id"] == "fingerprint2")
    assert final.parent_snapshot_id == second_snapshot_id
    second_manifest = next(item for item in manifests if item["batch_id"] == "fingerprint2")
    second_metrics = dict(second_manifest["metrics"])
    assert second_metrics["ap_edges_added"] == 0.0
    assert second_metrics["ap_edges_removed"] == 0.0
    assert len(final.reference_scan_ids) == 3
    assert len(final.update_scan_ids) == 2
    assert final.sequence_number == 3
    assert final.vocabulary_after == bootstrap[1].bssid_vocabulary + (
        "001122334477",
    )
    assert (resumed.latest_snapshot_directory / "COMPLETED.json").is_file()
    assert unrelated.read_text(encoding="utf-8") == "user-owned\n"

    uninterrupted_root = tmp_path / "uninterrupted"
    uninterrupted = _run(uninterrupted_root, bootstrap, (first, second, third))
    assert uninterrupted.latest_snapshot is not None
    left = resumed.latest_snapshot_directory
    right = uninterrupted.latest_snapshot_directory
    left_prepared = json.loads((left / "reference_prepared.json").read_text())
    right_prepared = json.loads((right / "reference_prepared.json").read_text())
    assert left_prepared["scans"] == right_prepared["scans"]
    left_gnn = torch.load(left / "reference_gnn" / "gnn_state.pt", map_location="cpu", weights_only=True)
    right_gnn = torch.load(right / "reference_gnn" / "gnn_state.pt", map_location="cpu", weights_only=True)
    for name in left_gnn["state_dict"]:
        assert torch.equal(left_gnn["state_dict"][name], right_gnn["state_dict"][name])


def test_streaming_snapshot_failure_keeps_previous_latest_and_retries(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    bootstrap = _bootstrap(data)
    first, _second = _updates(data)
    interrupted_root = tmp_path / "interrupted"

    def interrupt(stage, _path):
        if stage == "update_edge_prediction":
            raise RuntimeError("simulated interruption before snapshot commit")

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _run_with_observer(interrupted_root, bootstrap, (first,), interrupt)
    assert not (interrupted_root / "LATEST.json").exists()
    assert list((interrupted_root / "snapshots").glob(".staging-*"))

    recovered = _run(interrupted_root, bootstrap, (first,))
    assert recovered.latest_snapshot is not None
    assert recovered.latest_snapshot.sequence_number == 1
    assert recovered.completed_batch_ids == ("fingerprint1",)


def test_streaming_snapshot_resumes_in_a_fresh_process(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    bootstrap = _bootstrap(data)
    first, _second = _updates(data)
    root = tmp_path / "fresh-process"
    root.mkdir()
    prepared_path = data / "prepared.json"
    save_prepared_dataset(bootstrap[1], prepared_path)
    config_path = data / "pipeline-config.json"
    config_path.write_text(bootstrap[6].model_dump_json(), encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get(
        "PYTHONPATH", ""
    )
    common = [
        sys.executable,
        "-m",
        "streaming_pipeline",
        "--config",
        str(config_path),
        "--output",
        str(root),
    ]
    new_run = [
        *common,
        "--prepared",
        str(prepared_path),
        "--ae-checkpoint",
        str(bootstrap[2]),
        "--gnn-checkpoint",
        str(bootstrap[3].checkpoint_path),
        "--lp-input",
        str(bootstrap[4]),
        "--lp-result",
        str(bootstrap[5]),
        "--bootstrap-path",
        str(bootstrap[0]),
        "--update",
        str(first),
        "--source-root",
        str(data),
    ]
    script_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        new_run,
        cwd=script_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "completed=('fingerprint1',)" in completed.stdout

    resumed = subprocess.run(
        [*common, "--update", str(first), "--bootstrap-path", str(bootstrap[0]), "--source-root", str(data)],
        cwd=script_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert "skipped=('fingerprint1',)" in resumed.stdout
    assert "latest=" in resumed.stdout


def _run_with_observer(root, bootstrap, updates, observer):
    bootstrap_path, prepared, ae_path, gnn, lp_input, lp_result, config = bootstrap
    return run_streaming_sequential(
        root,
        config,
        bootstrap_prepared=prepared,
        bootstrap_ae_checkpoint_path=ae_path,
        bootstrap_gnn_checkpoint=gnn,
        bootstrap_lp_input_directory=lp_input,
        bootstrap_lp_result_directory=lp_result,
        bootstrap_paths=(bootstrap_path,),
        update_paths=updates,
        source_root=bootstrap_path.parent,
        stage_observer=observer,
    )


def test_hidden_update_coordinates_do_not_change_fit_arrays(tmp_path):
    data_a = tmp_path / "data-a"
    data_b = tmp_path / "data-b"
    data_a.mkdir()
    data_b.mkdir()
    boot_a = _bootstrap(data_a)
    boot_b = _bootstrap(data_b)
    # Source paths are both named the same relative to their source roots, so
    # prepared IDs, input order, and seeded pseudo-locations are identical.
    updates_a = _updates(data_a, (1, 2), (3, 4))
    updates_b = _updates(data_b, (1000, -500), (-900, 700))
    root_a, root_b = tmp_path / "run-a", tmp_path / "run-b"
    result_a = _run(root_a, boot_a, updates_a)
    result_b = _run(root_b, boot_b, updates_b)
    assert result_a.latest_snapshot is not None and result_b.latest_snapshot is not None
    assert result_a.latest_snapshot.update_scan_ids == result_b.latest_snapshot.update_scan_ids
    state_a = result_a.latest_snapshot_directory
    state_b = result_b.latest_snapshot_directory
    arrays_a = np.load(state_a / "post_transition_ae" / "updated_rssi_dbm.npy", mmap_mode="r")
    arrays_b = np.load(state_b / "post_transition_ae" / "updated_rssi_dbm.npy", mmap_mode="r")
    np.testing.assert_array_equal(arrays_a, arrays_b)
    gnn_a = torch.load(state_a / "reference_gnn" / "gnn_state.pt", map_location="cpu", weights_only=True)
    gnn_b = torch.load(state_b / "reference_gnn" / "gnn_state.pt", map_location="cpu", weights_only=True)
    for name in gnn_a["state_dict"]:
        assert torch.equal(gnn_a["state_dict"][name], gnn_b["state_dict"][name])
