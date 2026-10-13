from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from ae import AETrainingConfig
from batch_transition import TransitionConfig
from fingerprint_update import FingerprintUpdateConfig, FingerprintUpdateManifest
from graph_build import GraphBuildConfig
from gufu_data import load_prepared_dataset, prepare_dataset_files
from lp import (
    LinkPredictionConfig,
    LinkPredictionError,
    load_edge_delta,
    predict_edges,
)
from pred import (
    OrchestrationConfig,
    discover_update_paths,
    load_completed_snapshot,
    load_latest_completed_snapshot,
    run_sequential,
)
from gufu_artifacts import canonical_json_digest, load_torch_state, sha256_file
from sage_ve import GNNModelConfig, GNNTrainingConfig


AP_A = "001122334455"
AP_B = "001122334466"
AP_C = "001122334477"


def _write_inputs(root):
    bootstrap = root / "fingerprint0.txt"
    bootstrap.write_text(
        json.dumps({"coordinates": [0, 0], "data": [
            {"bssid": AP_A, "rssi": -50}, {"bssid": AP_B, "rssi": -65}
        ]}) + "\n"
        + json.dumps({"coordinates": [0, 0], "data": [
            {"bssid": AP_A, "rssi": -55}
        ]}) + "\n"
        + json.dumps({"coordinates": [20, 20], "data": [
            {"bssid": AP_A, "rssi": -60}, {"bssid": AP_B, "rssi": -55}
        ]}) + "\n",
        encoding="utf-8",
    )
    update1 = root / "fingerprint1.txt"
    update1.write_text(
        json.dumps({"data": [
            {"bssid": AP_C, "rssi": -45}, {"bssid": AP_A, "rssi": -52}
        ]}) + "\n"
        + json.dumps({"data": [
            {"bssid": AP_A, "rssi": -57}, {"bssid": AP_B, "rssi": -59},
            {"bssid": AP_C, "rssi": -48}
        ]}) + "\n",
        encoding="utf-8",
    )
    update2 = root / "fingerprint2.txt"
    update2.write_text(
        json.dumps({"data": [
            {"bssid": AP_B, "rssi": -63}, {"bssid": AP_A, "rssi": -53}
        ]}) + "\n"
        + json.dumps({"data": [
            {"bssid": AP_A, "rssi": -58}, {"bssid": AP_B, "rssi": -58}
        ]}) + "\n",
        encoding="utf-8",
    )
    update3 = root / "fingerprint3.txt"
    update3.write_text(
        json.dumps({"data": [
            {"bssid": AP_A, "rssi": -54}, {"bssid": AP_B, "rssi": -62}
        ]}) + "\n"
        + json.dumps({"data": [
            {"bssid": AP_B, "rssi": -57}, {"bssid": AP_A, "rssi": -59}
        ]}) + "\n",
        encoding="utf-8",
    )
    return bootstrap, update1, update2, update3


def _configuration(dataset_id="orchestrator-test", *, latent_dim=32):
    lp = LinkPredictionConfig(
        max_iterations=4,
        convergence_epsilon=1.0e9,
        link_threshold=0.0,
    )
    graph = GraphBuildConfig(
        virtual_edge_strategy="paper_cosine_threshold",
        virtual_edge_sigma=-0.999,
        pseudo_location_seed=9,
    )
    return OrchestrationConfig(
        dataset_id=dataset_id,
        input_format="gufu_jsonl",
        ae_action="train",
        gnn_action="train",
        ae_training=AETrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            seed=31,
            shuffle=False,
            latent_dim=latent_dim,
        ),
        graph_build=graph,
        gnn_model=GNNModelConfig(
            graph_feature_dim=latent_dim + 2,
            edge_feature_dim=latent_dim + 2,
            node_output_dim=latent_dim + 2,
            edge_output_dim=latent_dim + 2,
            num_layers=1,
            dropout_probability=0.0,
            initialization_seed=17,
        ),
        gnn_training=GNNTrainingConfig(
            epochs=1,
            learning_rate=0.002,
            seed=23,
            objective_epsilon=1.0e-8,
        ),
        link_prediction=lp,
        fingerprint_update=FingerprintUpdateConfig(
            hidden_dim=8,
            alpha=0.5,
            virtual_edge_sigma=-0.999,
            gnn_epochs=1,
            updater_epochs=1,
            ae_epochs=1,
            gnn_learning_rate=0.001,
            updater_learning_rate=0.01,
            ae_learning_rate=0.0001,
            seed=41,
        ),
        transition=TransitionConfig(
            link_prediction=lp,
            ae_finetune_epochs=1,
            ae_finetune_seed=43,
            reinitialize_seed=47,
        ),
        device="cpu",
    )


def _bootstrap(root, path, configuration):
    return prepare_dataset_files(
        (path,),
        configuration.preprocessing("fingerprint0", "bootstrap_reference"),
        source_root=root,
    )


def _run(root, output, configuration, bootstrap, bootstrap_path, updates, **kwargs):
    return run_sequential(
        output,
        configuration,
        bootstrap_prepared=bootstrap,
        bootstrap_paths=(bootstrap_path,),
        bootstrap_source_paths=(bootstrap_path,),
        update_paths=updates,
        source_root=root,
        **kwargs,
    )


@pytest.fixture(scope="module", autouse=True)
def single_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def test_bootstrap_multi_batch_handoff_no_change_continues_and_resume_matches(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    bootstrap_path, batch1, batch2, batch3 = _write_inputs(inputs)
    config = _configuration()
    bootstrap = _bootstrap(inputs, bootstrap_path, config)

    resumed_root = tmp_path / "resumed"
    first = _run(
        inputs,
        resumed_root,
        config,
        bootstrap,
        bootstrap_path,
        (batch1,),
    )
    assert first.completed_batch_ids == ("fingerprint1",)
    first_batch_state = first.state
    assert len(first_batch_state.transition_state.reference_scan_ids) == 3
    assert first_batch_state.prepared.bssid_vocabulary == (AP_A, AP_B, AP_C)
    assert first_batch_state.updater_checkpoint_path is not None

    unrelated = resumed_root / "keep-me.txt"
    unrelated.write_text("user data survives resume\n", encoding="utf-8")
    config_path = tmp_path / "pipeline-config.json"
    config_path.write_text(config.model_dump_json(), encoding="utf-8")
    repository = Path(__file__).resolve().parents[1]
    subprocess_environment = dict(os.environ)
    subprocess_environment.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        PYTHONDONTWRITEBYTECODE="1",
    )
    process = subprocess.run(
        [
            sys.executable,
            str(repository / "pred.py"),
            "--config-json", str(config_path),
            "--output-root", str(resumed_root),
            "--bootstrap", str(bootstrap_path),
            "--bootstrap-batch-id", "fingerprint0",
            "--updates", str(batch1), str(batch2), str(batch3),
            "--source-root", str(inputs),
        ],
        cwd=repository,
        env=subprocess_environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "Completed this run: ('fingerprint2', 'fingerprint3')" in process.stdout
    assert "Already completed: ('fingerprint1',)" in process.stdout
    assert unrelated.read_text(encoding="utf-8") == "user data survives resume\n"
    state2 = load_completed_snapshot(
        resumed_root / "snapshots" / "batches" / "00002-fingerprint2",
        expected_configuration=config,
    )
    state3 = load_latest_completed_snapshot(
        resumed_root, expected_configuration=config
    )
    assert state3 is not None
    assert state2.manifest.parent_snapshot_sha256 == first_batch_state.completion_sha256
    assert state3.manifest.parent_snapshot_sha256 == state2.completion_sha256
    assert state2.manifest.parent_transition_state_id == first_batch_state.transition_state.state_id
    assert state3.manifest.parent_transition_state_id == state2.transition_state.state_id
    edge_stage2 = next(row for row in state2.manifest.stage_records if row.stage == "edge_prediction")
    assert edge_stage2.status == "no_change"
    update_stage2 = next(row for row in state2.manifest.stage_records if row.stage == "updater_training")
    assert dict(update_stage2.metrics)["updater_initialization"] == "weights_only_previous_batch"
    assert dict(update_stage2.metrics)["source_update_checkpoint_sha256"] == sha256_file(
        first_batch_state.updater_checkpoint_path
    )
    state2_updater_payload = load_torch_state(state2.updater_checkpoint_path)
    state2_updater_manifest = FingerprintUpdateManifest.model_validate_json(
        state2_updater_payload["manifest_json"]
    )
    assert state2_updater_manifest.bssid_vocabulary == first_batch_state.prepared.bssid_vocabulary
    assert state2_updater_manifest.source_gnn_checkpoint_sha256 == sha256_file(
        first_batch_state.gnn_checkpoint_path
    )
    assert next(row for row in state3.manifest.stage_records if row.stage == "updater_training").status == "complete"
    state2_shared_batch = load_prepared_dataset(
        state2.snapshot_directory
        / dict(state2.manifest.artifact_index)["shared_vocabulary_batch"]
    )
    assert state2_shared_batch.bssid_vocabulary == first_batch_state.prepared.bssid_vocabulary

    reference_ids = tuple(scan.scan_id for scan in bootstrap.scans)
    assert bootstrap.reference_positions[0].x == bootstrap.reference_positions[1].x
    assert bootstrap.reference_positions[0].y == bootstrap.reference_positions[1].y
    assert bootstrap.scans[1].observation_mask[bootstrap.bssid_vocabulary.index(AP_B)] is False
    assert bootstrap.evaluation_positions == ()
    for state in (first_batch_state, state2, state3):
        assert state.manifest.reference_scan_count == len(reference_ids)
        assert state.graph_state.sample_node_ids == reference_ids
        assert not set(state.transition_state.batch_history[-1].scan_ids) & set(state.graph_state.node_ids)
        assert state.encoder_features.scan_ids == reference_ids
        assert state.encoder_features.checkpoint_sha256 == sha256_file(
            state.ae_checkpoint_path
        )
        assert state.gnn_features.node_ids == state.graph_state.node_ids
        assert state.gnn_features.checkpoint_sha256 == sha256_file(
            state.gnn_checkpoint_path
        )
    assert len(state3.prepared.scans) == len(bootstrap.scans)
    assert state3.prepared.reference_positions == bootstrap.reference_positions
    for state in (first_batch_state, state2, state3):
        prepared_path = state.snapshot_directory / dict(state.manifest.artifact_index)[
            "input_prepared_batch"
        ]
        incoming = load_prepared_dataset(prepared_path)
        assert incoming.evaluation_positions == ()
        assert all(scan.coordinate_role == "unavailable" for scan in incoming.scans)

    bootstrap_state = load_completed_snapshot(
        resumed_root / "snapshots" / "bootstrap",
        expected_configuration=config,
    )
    assert bootstrap_state.encoder_features.scan_ids == reference_ids
    assert bootstrap_state.encoder_features.checkpoint_sha256 == sha256_file(
        bootstrap_state.ae_checkpoint_path
    )
    assert bootstrap_state.gnn_features.node_ids == bootstrap_state.graph_state.node_ids
    assert bootstrap_state.gnn_features.checkpoint_sha256 == sha256_file(
        bootstrap_state.gnn_checkpoint_path
    )
    graph = bootstrap_state.graph_state
    encoder_by_id = {
        scan_id: row
        for scan_id, row in zip(
            bootstrap_state.encoder_features.scan_ids,
            bootstrap_state.encoder_features.feature_matrix,
        )
    }
    for scan_id in reference_ids:
        graph_row = graph.node_features[graph.id_to_row[scan_id]]
        assert graph_row[: graph.encoder_feature_dim] == pytest.approx(encoder_by_id[scan_id])
    for ap_id in graph.ap_node_ids:
        incident = [edge for edge in graph.radio_edges if edge.ap_node_id == ap_id]
        denominator = sum(edge.scalar_weight for edge in incident)
        expected = tuple(
            sum(
                edge.scalar_weight
                * graph.node_features[graph.id_to_row[edge.sample_node_id]][column]
                for edge in incident
            ) / denominator
            for column in range(graph.node_feature_dim)
        )
        assert graph.node_features[graph.id_to_row[ap_id]] == pytest.approx(expected)
    edge_path = bootstrap_state.snapshot_directory / dict(
        bootstrap_state.manifest.artifact_index
    )["link_prediction"]
    saved_delta = load_edge_delta(edge_path)
    assert saved_delta.gnn_checkpoint_sha256 == bootstrap_state.gnn_features.checkpoint_sha256
    assert saved_delta.gnn_checkpoint_sha256 == sha256_file(
        bootstrap_state.gnn_checkpoint_path
    )
    recomputed_delta = predict_edges(
        graph,
        bootstrap_state.gnn_features,
        config.link_prediction,
    )
    assert recomputed_delta.candidate_scores == saved_delta.candidate_scores
    assert recomputed_delta.goodness_features == saved_delta.goodness_features

    altered_matrix = [list(row) for row in bootstrap_state.gnn_features.node_feature_matrix]
    altered_matrix[graph.id_to_row[reference_ids[0]]][0] += 0.5
    altered_learned = bootstrap_state.gnn_features.model_copy(
        update={"node_feature_matrix": tuple(tuple(row) for row in altered_matrix)}
    )
    altered_delta = predict_edges(graph, altered_learned, config.link_prediction)
    assert altered_delta.goodness_features != saved_delta.goodness_features
    wrong_ids = bootstrap_state.gnn_features.model_copy(
        update={"node_ids": tuple(reversed(bootstrap_state.gnn_features.node_ids))}
    )
    with pytest.raises(LinkPredictionError, match="node ID order"):
        predict_edges(graph, wrong_ids, config.link_prediction)
    wrong_vocabulary = tuple(reversed(graph.bssid_vocabulary))
    wrong_vocab_features = bootstrap_state.gnn_features.model_copy(
        update={
            "bssid_vocabulary": wrong_vocabulary,
            "vocabulary_sha256": canonical_json_digest(wrong_vocabulary),
        }
    )
    with pytest.raises(LinkPredictionError, match="vocabulary"):
        predict_edges(graph, wrong_vocab_features, config.link_prediction)

    uninterrupted_root = tmp_path / "uninterrupted"
    uninterrupted = _run(
        inputs,
        uninterrupted_root,
        config,
        bootstrap,
        bootstrap_path,
        (batch1, batch2, batch3),
    )
    assert uninterrupted.completed_batch_ids == ("fingerprint1", "fingerprint2", "fingerprint3")
    assert tuple(scan.rss_dbm for scan in uninterrupted.state.prepared.scans) == tuple(
        scan.rss_dbm for scan in state3.prepared.scans
    )
    assert uninterrupted.state.prepared.bssid_vocabulary == state3.prepared.bssid_vocabulary
    assert torch.allclose(
        torch.tensor(uninterrupted.state.encoder_features.feature_matrix),
        torch.tensor(state3.encoder_features.feature_matrix),
        rtol=1e-6,
        atol=1e-6,
    )
    assert torch.allclose(
        torch.tensor(uninterrupted.state.gnn_features.node_feature_matrix),
        torch.tensor(state3.gnn_features.node_feature_matrix),
        rtol=1e-6,
        atol=1e-6,
    )


def test_interrupted_batch_commit_is_ignored_and_retry_completes(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    bootstrap_path, batch1, _, _ = _write_inputs(inputs)
    config = _configuration("orchestrator-recovery")
    bootstrap = _bootstrap(inputs, bootstrap_path, config)
    output = tmp_path / "pipeline"

    class InterruptBeforeCommit(RuntimeError):
        pass

    def interrupt(stage, _staging_path):
        if stage == "update_before_commit":
            raise InterruptBeforeCommit(stage)

    with pytest.raises(InterruptBeforeCommit):
        _run(
            inputs,
            output,
            config,
            bootstrap,
            bootstrap_path,
            (batch1,),
            stage_observer=interrupt,
        )
    current = load_latest_completed_snapshot(output, expected_configuration=config)
    assert current is not None and current.manifest.sequence_index == 0
    orphan = output / "snapshots" / "batches" / ".00001-fingerprint1.staging-interrupted"
    orphan.mkdir(parents=True)
    (orphan / "snapshot_manifest.json").write_text("partial", encoding="utf-8")
    assert load_latest_completed_snapshot(output, expected_configuration=config).manifest.sequence_index == 0

    recovered = _run(inputs, output, config, bootstrap, bootstrap_path, (batch1,))
    assert recovered.completed_batch_ids == ("fingerprint1",)
    assert (recovered.state.snapshot_directory / "COMPLETED.json").is_file()
    assert orphan.is_dir()


def test_resume_rejects_configuration_and_checkpoint_artifact_mismatch(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    bootstrap_path, batch1, _, _ = _write_inputs(inputs)
    config = _configuration("orchestrator-mismatch")
    bootstrap = _bootstrap(inputs, bootstrap_path, config)
    output = tmp_path / "pipeline"
    result = _run(inputs, output, config, bootstrap, bootstrap_path, (batch1,))

    incompatible = _configuration("different-dataset")
    with pytest.raises(ValueError, match="configuration"):
        load_completed_snapshot(result.state.snapshot_directory, expected_configuration=incompatible)

    corrupt_copy = tmp_path / "corrupt-snapshot"
    shutil.copytree(result.state.snapshot_directory, corrupt_copy)
    checkpoint = corrupt_copy / "ae_checkpoint.pt"
    checkpoint.write_bytes(checkpoint.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="missing or corrupt"):
        load_completed_snapshot(corrupt_copy, expected_configuration=config)

    missing_copy = tmp_path / "missing-snapshot-artifact"
    shutil.copytree(result.state.snapshot_directory, missing_copy)
    missing_feature = missing_copy / dict(result.state.manifest.artifact_index)[
        "gnn_features"
    ]
    missing_feature.unlink()
    with pytest.raises(ValueError, match="missing or corrupt"):
        load_completed_snapshot(missing_copy, expected_configuration=config)


def test_directory_discovery_is_natural_and_excludes_bootstrap(tmp_path):
    bootstrap = tmp_path / "fingerprint0.txt"
    first = tmp_path / "fingerprint1.txt"
    second = tmp_path / "fingerprint2.txt"
    tenth = tmp_path / "fingerprint10.txt"
    for path in (bootstrap, first, second, tenth):
        path.write_text("", encoding="utf-8")
    paths = discover_update_paths(
        input_directory=tmp_path,
        pattern="fingerprint*.txt",
        bootstrap_paths=(bootstrap,),
    )
    assert tuple(path.name for path in paths) == (
        "fingerprint1.txt",
        "fingerprint2.txt",
        "fingerprint10.txt",
    )


def test_importing_orchestrator_does_not_create_output_directory(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repository)
    subprocess.run(
        [sys.executable, "-c", "import pred"],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert not (tmp_path / "output").exists()


def test_non_32_latent_flows_through_bootstrap_update_and_ap_transition(tmp_path):
    inputs = tmp_path / "latent-inputs"
    inputs.mkdir()
    bootstrap_path, update_path, _update2, _update3 = _write_inputs(inputs)
    config = _configuration("latent-width-test", latent_dim=16)
    bootstrap = _bootstrap(inputs, bootstrap_path, config)

    result = _run(
        inputs,
        tmp_path / "latent-output",
        config,
        bootstrap,
        bootstrap_path,
        (update_path,),
    )
    state = result.state

    assert result.completed_batch_ids == ("fingerprint1",)
    assert state.encoder_features.feature_dim == 16
    assert state.graph_state.encoder_feature_dim == 16
    assert state.graph_state.node_feature_dim == 18
    assert state.gnn_features.node_feature_dim == 18
    assert all(len(row) == 18 for row in state.gnn_features.node_feature_matrix)
    assert state.transition_state.encoder_feature_dim == 16
    assert state.transition_state.graph_feature_dim == 18
    assert len(state.graph_state.sample_node_ids) == len(bootstrap.scans)
    assert state.transition_state.reference_scan_ids == tuple(
        scan.scan_id for scan in bootstrap.scans
    )
    checkpoint = load_torch_state(state.updater_checkpoint_path)
    updater_manifest = FingerprintUpdateManifest.model_validate_json(
        checkpoint["manifest_json"]
    )
    assert updater_manifest.encoder_feature_dim == 16
    assert updater_manifest.graph_feature_dim == 18


def test_orchestration_rejects_mismatched_gnn_feature_width():
    config = _configuration("latent-width-mismatch", latent_dim=16)
    incompatible = GNNModelConfig(num_layers=1, dropout_probability=0.0)
    with pytest.raises(ValueError, match=r"AE latent_dim \+ 2"):
        OrchestrationConfig.model_validate(
            {
                **config.model_dump(mode="python"),
                "gnn_model": incompatible,
            }
        )
