import pytest
import torch
import torch.nn.functional as F

from ae import EncoderFeatureArtifact
from graph_build import (
    GraphBuildConfig,
    GraphState,
    _graph_id,
    build_graph_state,
)
from gufu_artifacts import canonical_json_digest
from gufu_data import PreprocessingConfig, prepare_dataset_files
from sage_ve import (
    EdgeFeatureSAGE,
    GNNModelConfig,
    GNNTrainingError,
    GNNTrainingConfig,
    equation4_loss,
    graph_state_to_tensors,
    load_gnn_checkpoint,
    load_gnn_features,
    train_gnn,
)


def graph_fixture(tmp_path, *, feature_shift=0, dataset_id="sage-test"):
    root = tmp_path / f"input-{feature_shift}"
    root.mkdir(parents=True, exist_ok=True)
    source = root / "bootstrap.txt"
    source.write_text(
        "0,0 001122334455,-50\n10,20 001122334455,-60\n",
        encoding="utf-8",
    )
    prepared = prepare_dataset_files(
        [source],
        PreprocessingConfig(
            dataset_id=dataset_id,
            batch_id=f"bootstrap-{feature_shift}",
            batch_role="bootstrap_reference",
            normalization_mode="paper_ae",
        ),
        source_root=root,
    )
    scan_ids = tuple(scan.scan_id for scan in prepared.scans)
    by_id = {}
    for index, scan_id in enumerate(scan_ids):
        row = [0.0] * 32
        row[index + feature_shift] = 1.0
        row[(index + 4 + feature_shift) % 32] = 0.25
        by_id[scan_id] = tuple(row)
    feature_artifact = EncoderFeatureArtifact(
        dataset_id=prepared.config.dataset_id,
        batch_id=prepared.config.batch_id,
        source_prepared_sha256=canonical_json_digest(prepared),
        scan_ids=scan_ids,
        feature_matrix=tuple(by_id[scan_id] for scan_id in scan_ids),
        checkpoint_sha256="a" * 64,
        bssid_vocabulary=prepared.bssid_vocabulary,
        vocabulary_sha256=canonical_json_digest(prepared.bssid_vocabulary),
        normalization=prepared.config.normalization_mode,
        dtype="float32",
        created_at_utc="2030-01-01T00:00:00+00:00",
    )
    graph = build_graph_state(
        prepared,
        feature_artifact,
        GraphBuildConfig(virtual_edge_sigma=1.0),
    )
    return prepared, feature_artifact, graph


def normalize_relu(values):
    return F.normalize(F.relu(values), p=2.0, dim=1, eps=1e-12)


def test_one_layer_matches_hand_calculated_edge_message_node_and_radio_updates(tmp_path):
    _prepared, _encoder_features, graph = graph_fixture(tmp_path)
    config = GNNModelConfig(num_layers=1, dropout_probability=0.0, initialization_seed=5)
    model = EdgeFeatureSAGE(config)
    with torch.no_grad():
        model.W0[0].zero_()
        model.W0[0][:, 34:] = torch.eye(34)  # message = ReLU(neighbor z)
        model.W1[0].zero_()
        model.W1[0][:, 34:] = torch.eye(34)  # node = normalize(ReLU(mean messages))
        model.W2[0].zero_()
        model.W2[0][:, :34] = torch.eye(34)  # radio edge = normalized left endpoint

    model.eval()
    inputs = graph_state_to_tensors(graph)
    with torch.inference_mode():
        output = model(inputs)

    rows = graph.id_to_row
    sample_a, sample_b = graph.sample_node_ids
    ap_id = graph.ap_node_ids[0]
    expected_a = normalize_relu(
        torch.tensor([graph.node_features[rows[ap_id]]], dtype=torch.float32)
    )[0]
    expected_b = expected_a
    expected_ap = normalize_relu(
        torch.tensor(
            [[
                (max(graph.node_features[rows[sample_a]][dim], 0.0)
                 + max(graph.node_features[rows[sample_b]][dim], 0.0))
                / 2.0
                for dim in range(34)
            ]],
            dtype=torch.float32,
        )
    )[0]
    expected_by_id = {
        sample_a: expected_a,
        sample_b: expected_b,
        ap_id: expected_ap,
    }
    for node_id, row in rows.items():
        assert torch.allclose(
            output.node_features[row], expected_by_id[node_id], rtol=0.0, atol=1e-7
        )

    radio_order = graph.radio_edges
    expected_radio = torch.stack(
        [
            normalize_relu(
                output.node_features[rows[edge.sample_node_id]].unsqueeze(0)
            )[0]
            for edge in radio_order
        ]
    )
    assert torch.allclose(output.radio_edge_features, expected_radio, rtol=0.0, atol=1e-7)
    assert output.virtual_edge_features_initial.shape == (0, 34)

    model.zero_grad(set_to_none=True)
    differentiable_output = model(graph)
    equation4_loss(differentiable_output.node_features, graph).backward()
    # Eq. 4 is node-only, so a one-layer run's final W2 edge update has no
    # downstream use in the written algorithm and therefore receives no grad.
    assert model.W2[0].grad is None


def test_edge_vectors_change_messages_and_eq4_uses_finite_differentiable_sum(tmp_path):
    _prepared, _encoder_features, graph = graph_fixture(tmp_path)
    model = EdgeFeatureSAGE(
        GNNModelConfig(num_layers=1, dropout_probability=0.0, initialization_seed=8)
    )
    with torch.no_grad():
        model.W0[0].zero_()
        model.W0[0][:, :34] = torch.eye(34)  # message depends on edge vector
        model.W0[0][:, 34:] = torch.eye(34)  # and on the neighbor node feature
        model.W1[0].zero_()
        model.W1[0][:, 34:] = torch.eye(34)
        model.W2[0].zero_()
    model.eval()
    inputs = graph_state_to_tensors(graph)
    with torch.inference_mode():
        base = model(inputs)
        changed_vectors = inputs.edge_features.clone()
        changed_vectors[0] = changed_vectors[0] + 0.75
        changed = model(inputs._replace(edge_features=changed_vectors))
        changed_nodes = inputs.node_features.clone()
        changed_nodes[0] = changed_nodes[0] + 0.75
        changed_input = model(inputs._replace(node_features=changed_nodes))
    assert not torch.allclose(base.node_features, changed.node_features)
    assert not torch.allclose(base.node_features, changed_input.node_features)
    mismatched_dtype = graph_state_to_tensors(graph, dtype=torch.float64)
    with pytest.raises(ValueError, match="model dtype"):
        model(mismatched_dtype)
    malformed_edges = inputs._replace(edge_left_rows=inputs.edge_left_rows[:-1])
    with pytest.raises(ValueError, match="edge endpoints/types"):
        model(malformed_edges)


def test_equation4_sums_each_physical_and_virtual_edge_once_and_clamps_zero(tmp_path):
    prepared, encoder_features, _graph = graph_fixture(tmp_path)
    all_edges_graph = build_graph_state(
        prepared,
        encoder_features,
        GraphBuildConfig(virtual_edge_sigma=-1.0),
    )
    assert all_edges_graph.radio_edges
    assert all_edges_graph.virtual_edges
    model = EdgeFeatureSAGE(
        GNNModelConfig(num_layers=2, dropout_probability=0.0, initialization_seed=71)
    )
    model.eval()
    with torch.inference_mode():
        model_output = model(all_edges_graph)
    expected_virtual_vectors = torch.tensor(
        [edge.edge_feature for edge in all_edges_graph.virtual_edges],
        dtype=torch.float32,
    )
    assert torch.equal(
        model_output.virtual_edge_features_initial, expected_virtual_vectors
    )
    values = torch.randn(
        (len(all_edges_graph.node_ids), 34), dtype=torch.float32, requires_grad=True
    )
    row_by_id = all_edges_graph.id_to_row
    dot_products = []
    for edge in all_edges_graph.radio_edges:
        dot_products.append(
            torch.dot(
                values[row_by_id[edge.sample_node_id]],
                values[row_by_id[edge.ap_node_id]],
            )
        )
    for edge in all_edges_graph.virtual_edges:
        dot_products.append(
            torch.dot(
                values[row_by_id[edge.sample_node_id_left]],
                values[row_by_id[edge.sample_node_id_right]],
            )
        )
    epsilon = 1.0e-12
    expected = -torch.log(F.relu(torch.stack(dot_products)).clamp_min(epsilon)).sum()
    actual = equation4_loss(values, all_edges_graph, epsilon=epsilon)
    assert torch.allclose(actual, expected, rtol=1.0e-6, atol=1.0e-6)
    assert len(dot_products) == len(all_edges_graph.radio_edges) + len(
        all_edges_graph.virtual_edges
    )

    zero_embeddings = torch.zeros_like(values, requires_grad=True)
    zero_loss = equation4_loss(zero_embeddings, all_edges_graph, epsilon=epsilon)
    assert torch.isfinite(zero_loss)
    assert zero_loss.item() == pytest.approx(
        -(len(dot_products)) * torch.log(torch.tensor(epsilon)).item()
    )
    zero_loss.backward()
    assert torch.count_nonzero(zero_embeddings.grad).item() == 0

    trainable = EdgeFeatureSAGE(
        GNNModelConfig(num_layers=2, dropout_probability=0.0, initialization_seed=12)
    )
    output = trainable(all_edges_graph)
    loss = equation4_loss(output.node_features, all_edges_graph)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0
    loss.backward()
    gradients = [parameter.grad for parameter in trainable.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient).item() for gradient in gradients)


def test_training_checkpoint_reload_and_final_eval_export_are_id_aligned(tmp_path):
    _prepared, _encoder_features, graph = graph_fixture(tmp_path)
    checkpoint_path = tmp_path / "gnn.pt"
    feature_path = tmp_path / "gnn_features.json"
    model = EdgeFeatureSAGE(
        GNNModelConfig(num_layers=2, dropout_probability=0.2, initialization_seed=17)
    )
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    first_edge_update_before = model.W2[0].detach().clone()
    result = train_gnn(
        graph,
        model,
        checkpoint_path,
        feature_path,
        training_config=GNNTrainingConfig(epochs=3, learning_rate=0.01, seed=91),
    )

    assert checkpoint_path.is_file() and feature_path.is_file()
    assert len(result.loss_history) == 3
    assert all(torch.isfinite(torch.tensor(value)) for value in result.loss_history)
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(parameters_before, model.parameters())
    )
    assert not torch.equal(first_edge_update_before, model.W2[0].detach())
    assert result.checkpoint.manifest.resumable is False
    assert result.checkpoint.manifest.selected_epoch == 3
    assert result.features.node_ids == graph.node_ids
    assert result.features.node_id_to_row == graph.node_id_to_row
    assert len(result.features.node_feature_matrix) == len(graph.node_ids)
    assert all(len(row) == 34 for row in result.features.node_feature_matrix)
    assert result.features.radio_edge_ids == tuple(
        edge.edge_id for edge in graph.radio_edges
    )
    assert len(result.features.radio_edge_feature_matrix) == len(graph.radio_edges)
    assert result.features.checkpoint_sha256 == result.checkpoint.checkpoint_sha256
    assert result.features.graph_id == graph.graph_id
    assert load_gnn_features(feature_path) == result.features

    reloaded = load_gnn_checkpoint(checkpoint_path, graph)
    reloaded.model.eval()
    with torch.inference_mode():
        reloaded_output = reloaded.model(graph)
    assert torch.allclose(
        reloaded_output.node_features,
        torch.tensor(result.features.node_feature_matrix, dtype=torch.float32),
        rtol=0.0,
        atol=1e-7,
    )
    assert torch.allclose(
        reloaded_output.radio_edge_features,
        torch.tensor(result.features.radio_edge_feature_matrix, dtype=torch.float32),
        rtol=0.0,
        atol=1e-7,
    )
    with pytest.raises(GNNTrainingError, match="inference-only"):
        train_gnn(
            graph,
            reloaded.model,
            tmp_path / "continued.pt",
            tmp_path / "continued-features.json",
            training_config=GNNTrainingConfig(epochs=1),
        )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        train_gnn(
            graph,
            EdgeFeatureSAGE(model.config),
            checkpoint_path,
            tmp_path / "unused.json",
            training_config=GNNTrainingConfig(epochs=1),
        )


def test_inference_node_rows_remain_joined_by_ids_after_graph_row_reordering(tmp_path):
    _prepared, _encoder_features, graph = graph_fixture(tmp_path)
    model = EdgeFeatureSAGE(
        GNNModelConfig(num_layers=2, dropout_probability=0.0, initialization_seed=33)
    )
    model.eval()
    reversed_samples = tuple(reversed(graph.sample_node_ids))
    node_ids = reversed_samples + graph.ap_node_ids
    old_rows = graph.id_to_row
    permutation = tuple(old_rows[node_id] for node_id in node_ids)
    payload = graph.model_dump(mode="python")
    payload["node_ids"] = node_ids
    payload["sample_node_ids"] = reversed_samples
    payload["node_id_to_row"] = tuple((node_id, row) for row, node_id in enumerate(node_ids))
    payload["node_types"] = ("sample",) * len(reversed_samples) + ("ap",) * len(graph.ap_node_ids)
    for field in (
        "node_features",
        "reference_sample_mask",
        "new_sample_mask",
        "reference_label_mask",
        "reference_coordinate_targets",
    ):
        payload[field] = tuple(payload[field][row] for row in permutation)
    init_by_id = {item.scan_id: item for item in graph.sample_location_initializations}
    payload["sample_location_initializations"] = tuple(
        init_by_id[scan_id] for scan_id in reversed_samples
    )
    payload["graph_id"] = _graph_id(
        source_prepared_sha256=graph.source_prepared_sha256,
        encoder_feature_artifact_sha256=graph.encoder_feature_artifact_sha256,
        build_config=graph.build_config,
        node_ids=node_ids,
    )
    reordered = GraphState.model_validate(payload)

    with torch.inference_mode():
        original_output = model(graph)
        reordered_output = model(reordered)
    original_by_id = {
        node_id: original_output.node_features[row]
        for node_id, row in graph.id_to_row.items()
    }
    reordered_by_id = {
        node_id: reordered_output.node_features[row]
        for node_id, row in reordered.id_to_row.items()
    }
    assert reordered.node_ids != graph.node_ids
    assert original_by_id.keys() == reordered_by_id.keys()
    for node_id in original_by_id:
        assert torch.allclose(
            original_by_id[node_id], reordered_by_id[node_id], rtol=0.0, atol=1e-7
        )


def test_checkpoint_rejects_different_graph_identity(tmp_path):
    _prepared, _features, graph = graph_fixture(tmp_path)
    _prepared_other, _features_other, different_graph = graph_fixture(
        tmp_path, feature_shift=1, dataset_id="other-floor"
    )
    checkpoint_path = tmp_path / "gnn.pt"
    train_gnn(
        graph,
        EdgeFeatureSAGE(
            GNNModelConfig(num_layers=1, dropout_probability=0.0, initialization_seed=2)
        ),
        checkpoint_path,
        tmp_path / "features.json",
        training_config=GNNTrainingConfig(epochs=1, seed=2),
    )

    with pytest.raises(ValueError, match="dataset identity"):
        load_gnn_checkpoint(checkpoint_path, different_graph)


def test_selected_checkpoint_runs_compatible_update_inference_without_retraining(tmp_path):
    prepared, _encoder_features, reference_graph = graph_fixture(tmp_path)
    checkpoint_path = tmp_path / "gnn.pt"
    train_gnn(
        reference_graph,
        EdgeFeatureSAGE(
            GNNModelConfig(num_layers=2, dropout_probability=0.0, initialization_seed=41)
        ),
        checkpoint_path,
        tmp_path / "reference-features.json",
        training_config=GNNTrainingConfig(epochs=2, seed=43),
    )
    update_root = tmp_path / "update-input"
    update_root.mkdir()
    update_source = update_root / "update.txt"
    update_source.write_text("999,999 001122334455,-55\n", encoding="utf-8")
    update = prepare_dataset_files(
        [update_source],
        PreprocessingConfig(
            dataset_id=prepared.config.dataset_id,
            batch_id="update-1",
            batch_role="update_batch",
            base_bssid_vocabulary=prepared.bssid_vocabulary,
            normalization_mode=prepared.config.normalization_mode,
        ),
        source_root=update_root,
    )
    update_scan_id = update.scans[0].scan_id
    update_features = EncoderFeatureArtifact(
        dataset_id=update.config.dataset_id,
        batch_id=update.config.batch_id,
        source_prepared_sha256=canonical_json_digest(update),
        scan_ids=(update_scan_id,),
        feature_matrix=((0.5,) + (0.0,) * 31,),
        checkpoint_sha256=reference_graph.encoder_checkpoint_sha256,
        bssid_vocabulary=update.bssid_vocabulary,
        vocabulary_sha256=canonical_json_digest(update.bssid_vocabulary),
        normalization=update.config.normalization_mode,
        dtype="float32",
        created_at_utc="2030-01-01T00:00:00+00:00",
    )
    update_graph = build_graph_state(
        update,
        update_features,
        GraphBuildConfig(
            virtual_edge_sigma=1.0,
            pseudo_location_bounds=(0.0, 10.0, 0.0, 20.0),
        ),
    )
    loaded = load_gnn_checkpoint(checkpoint_path, update_graph)
    assert loaded.checkpoint_sha256
    assert update_graph.reference_label_mask[update_graph.id_to_row[update_scan_id]] is False
    assert update_graph.sample_location_initializations[0].xy != (999.0, 999.0)
    with pytest.raises(GNNTrainingError, match="bootstrap_reference"):
        train_gnn(
            update_graph,
            EdgeFeatureSAGE(loaded.manifest.model_config_record),
            tmp_path / "forbidden-update-training.pt",
            tmp_path / "forbidden-update-features.json",
        )
    with torch.inference_mode():
        output = loaded.model(update_graph)
    assert output.node_features.shape == (len(update_graph.node_ids), 34)
    assert loaded.manifest.batch_id == reference_graph.batch_id
    assert loaded.manifest.graph_id != update_graph.graph_id


def test_empty_graph_relations_have_explicit_eq4_error(tmp_path):
    root = tmp_path / "empty-input"
    root.mkdir()
    source = root / "empty.txt"
    source.write_text("1,2\n", encoding="utf-8")
    prepared = prepare_dataset_files(
        [source],
        PreprocessingConfig(
            dataset_id="sage-empty",
            batch_id="empty-bootstrap",
            batch_role="bootstrap_reference",
            empty_scan_policy="keep",
        ),
        source_root=root,
    )
    scan_id = prepared.scans[0].scan_id
    features = EncoderFeatureArtifact(
        dataset_id=prepared.config.dataset_id,
        batch_id=prepared.config.batch_id,
        source_prepared_sha256=canonical_json_digest(prepared),
        scan_ids=(scan_id,),
        feature_matrix=((0.0,) * 32,),
        checkpoint_sha256="b" * 64,
        bssid_vocabulary=(),
        vocabulary_sha256=canonical_json_digest(()),
        normalization=prepared.config.normalization_mode,
        dtype="float32",
        created_at_utc="2030-01-01T00:00:00+00:00",
    )
    graph = build_graph_state(
        prepared,
        features,
        GraphBuildConfig(virtual_edge_sigma=1.0),
    )
    with pytest.raises(ValueError, match="no physical or virtual edges"):
        equation4_loss(torch.zeros((1, 34)), graph)
    with pytest.raises(ValueError, match="no physical or virtual edges"):
        train_gnn(
            graph,
            EdgeFeatureSAGE(GNNModelConfig(num_layers=1)),
            tmp_path / "no-edges.pt",
            tmp_path / "no-edges.json",
        )
