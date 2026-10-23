from __future__ import annotations

from dataclasses import dataclass

import pytest
import numpy as np
import torch
from torch import nn

from ae import (
    AE,
    AETrainingConfig,
    EncoderFeatureArtifact,
    encode_prepared_dataset,
    load_ae_checkpoint,
    migrate_ae_checkpoint_vocabulary,
    train_autoencoder,
)
from fingerprint_update import (
    FingerprintUpdateConfig,
    LocationPredictor,
    LocationToFeatureUpdater,
    compute_update_losses,
    compute_ae_update_losses,
    load_fingerprint_update_checkpoint,
    load_updated_fingerprints,
    predict_updated_fingerprints,
    prepare_fingerprint_update,
    train_fingerprint_update,
)
from fingerprint_update import _coordinate_training_tensors
from batch_transition import project_prepared_batch_to_vocabulary
from graph_build import GraphBuildConfig, build_graph_state
from gufu_artifacts import canonical_json_digest
from gufu_data import PreprocessingConfig, PreparedDataset, prepare_dataset_files
from gufu_data import ae_value_to_raw_dbm
from lp import LinkPredictionConfig, predict_edges
from sage_ve import (
    EdgeFeatureSAGE,
    GNNModelConfig,
    GNNTrainingConfig,
    load_gnn_checkpoint,
    load_gnn_features,
    train_gnn,
)
from lp_streaming import (
    load_streaming_lp_input,
    predict_edges_streaming,
    save_streaming_lp_input,
)
from streaming_update import (
    build_streaming_online_graph,
    build_streaming_reference_graph_after_transition,
    infer_streaming_ap_graph_changes,
    iter_streaming_ap_transition_candidates,
    load_streaming_online_graph,
    save_streaming_online_lp_input,
    save_streaming_online_lp_input_with_new_aps,
)
from streaming_gnn import (
    load_streaming_gnn_state,
    reencode_streaming_gnn,
    train_streaming_gnn,
)
from streaming_transition import (
    apply_streaming_ap_delta_to_reference,
    load_streaming_reference_transition,
    materialize_streaming_reference_prepared,
)
from streaming_fingerprint_update import (
    fine_tune_streaming_ae_after_ap_transition,
    load_streaming_ae_update,
    load_streaming_post_transition_ae,
    load_streaming_updater,
    train_streaming_ae_update,
    train_streaming_updaters,
)
from streaming_gnn import (
    _StreamedEquation4,
    _forward_stream,
    _loss_and_node_gradient,
    _parameter_order,
)
from fingerprint_update import equation4_online_loss


@dataclass(frozen=True)
class PipelineFixture:
    root: object
    reference: PreparedDataset
    update: PreparedDataset
    ae_checkpoint: object
    reference_encoder: EncoderFeatureArtifact
    update_encoder: EncoderFeatureArtifact
    graph: object
    gnn_checkpoint: object
    gnn_features: object


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> PipelineFixture:
    root = tmp_path_factory.mktemp("fingerprint-update")
    (root / "reference.txt").write_text(
        "0,0 001122334455,-50 001122334466,-70\n"
        "10,10 001122334455,-55 001122334466,-65\n"
        "20,20 001122334455,-60 001122334466,-60\n",
        encoding="utf-8",
    )
    (root / "update.txt").write_text(
        "88,99 001122334455,-42 001122334466,-78\n"
        "-7,34 001122334455,-68 001122334466,-49\n",
        encoding="utf-8",
    )
    reference = prepare_dataset_files(
        [root / "reference.txt"],
        PreprocessingConfig(
            dataset_id="update-test",
            batch_id="bootstrap",
            batch_role="bootstrap_reference",
            normalization_mode="paper_ae",
        ),
        source_root=root,
    )
    update = prepare_dataset_files(
        [root / "update.txt"],
        PreprocessingConfig(
            dataset_id="update-test",
            batch_id="batch-a",
            batch_role="update_batch",
            base_bssid_vocabulary=reference.bssid_vocabulary,
            normalization_mode="paper_ae",
        ),
        source_root=root,
    )
    ae_path = root / "base-ae.pt"
    train_autoencoder(
        reference,
        ae_path,
        AETrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            seed=41,
            shuffle=False,
            dtype="float32",
        ),
    )
    ref_encoder = encode_prepared_dataset(
        reference, ae_path, root / "ref-features.json", batch_size=2
    )
    update_encoder = encode_prepared_dataset(
        update, ae_path, root / "update-features.json", batch_size=1
    )
    graph_config = GraphBuildConfig(
        virtual_edge_strategy="paper_cosine_threshold",
        virtual_edge_sigma=1.0,
        pseudo_location_seed=13,
    )
    graph = build_graph_state(reference, ref_encoder, graph_config)
    gnn_path = root / "base-gnn.pt"
    train_gnn(
        graph,
        EdgeFeatureSAGE(
            GNNModelConfig(
                num_layers=2,
                dropout_probability=0.0,
                initialization_seed=7,
            )
        ),
        gnn_path,
        root / "ref-gnn-features.json",
        training_config=GNNTrainingConfig(
            epochs=2,
            learning_rate=0.003,
            seed=53,
            objective_epsilon=1.0e-8,
        ),
    )
    return PipelineFixture(
        root=root,
        reference=reference,
        update=update,
        ae_checkpoint=ae_path,
        reference_encoder=ref_encoder,
        update_encoder=update_encoder,
        graph=graph,
        gnn_checkpoint=gnn_path,
        gnn_features=load_gnn_features(root / "ref-gnn-features.json"),
    )


def small_update_config(**overrides) -> FingerprintUpdateConfig:
    values = dict(
        hidden_dim=8,
        alpha=0.5,
        virtual_edge_sigma=-0.999,
        gnn_epochs=1,
        updater_epochs=3,
        ae_epochs=1,
        gnn_learning_rate=0.001,
        updater_learning_rate=0.01,
        ae_learning_rate=0.0001,
        seed=101,
    )
    values.update(overrides)
    return FingerprintUpdateConfig(**values)


def make_context(pipeline, update=None, update_encoder=None, config=None, **kwargs):
    selected_update = pipeline.update if update is None else update
    selected_features = pipeline.update_encoder if update_encoder is None else update_encoder
    return prepare_fingerprint_update(
        pipeline.reference,
        pipeline.graph,
        pipeline.reference_encoder,
        pipeline.gnn_features,
        selected_update,
        selected_features,
        pipeline.ae_checkpoint,
        pipeline.gnn_checkpoint,
        config=small_update_config() if config is None else config,
        **kwargs,
    )


def test_each_paper_update_loss_matches_tiny_reference_and_gradients_reach_both_mlps():
    location = nn.Linear(34, 2)
    feature = nn.Linear(2, 32)
    with torch.no_grad():
        location.weight.zero_()
        location.bias.copy_(torch.tensor([0.5, 0.25]))
        feature.weight.zero_()
        feature.bias.fill_(0.1)
    reference_gnn = torch.ones((2, 34))
    update_gnn = torch.full((2, 34), 2.0)
    y_ref = torch.tensor([[0.0, 0.0], [1.0, 0.5]])
    z_update = torch.full((2, 32), 0.2)
    adjacency = torch.tensor([[1.0, 0.0], [0.5, 0.5]])

    losses, _, _ = compute_update_losses(
        location,
        feature,
        reference_gnn,
        update_gnn,
        y_ref,
        z_update,
        adjacency,
        alpha=0.5,
    )
    expected_lp = torch.sqrt(torch.tensor(0.625))
    expected_lu = torch.sqrt(torch.tensor(0.64))
    expected_cp = torch.sqrt(torch.tensor(0.3125))
    expected_cu = torch.sqrt(torch.tensor(0.64))
    assert torch.allclose(losses["location_supervised"], expected_lp)
    assert torch.allclose(losses["feature_supervised"], expected_lu)
    assert torch.allclose(losses["location_consistency"], expected_cp)
    assert torch.allclose(losses["feature_consistency"], expected_cu)
    assert torch.allclose(
        losses["total"],
        0.5 * (expected_lp + expected_lu + expected_cp + expected_cu),
    )

    location.zero_grad(set_to_none=True)
    feature.zero_grad(set_to_none=True)
    nonconstant = torch.arange(68, dtype=torch.float32).reshape(2, 34) / 68.0
    nonconstant_u = nonconstant + 0.2
    differentiable_losses, _, _ = compute_update_losses(
        location,
        feature,
        nonconstant,
        nonconstant_u,
        y_ref,
        z_update,
        adjacency,
        alpha=0.5,
    )
    differentiable_losses["total"].backward()
    for model in (location, feature):
        grads = [parameter.grad for parameter in model.parameters()]
        assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
        assert any(torch.count_nonzero(grad) for grad in grads)


def test_equation10_terms_match_reference_and_reach_encoder_decoder_gradients():
    torch.manual_seed(5)
    model = AE(3)
    model.eval()
    x = torch.tensor([[[0.1, 0.5, 0.8]], [[0.9, 0.3, 0.2]]])
    target = torch.linspace(-0.2, 0.4, 64).reshape(2, 32)
    losses = compute_ae_update_losses(model, x, target)
    with torch.no_grad():
        z, x_hat = model(x)
        d_target = model.D(target.unsqueeze(1))
        expected_reconstruction = torch.linalg.vector_norm(x_hat - x)
        expected_latent = torch.linalg.vector_norm(z[:, 0, :] - target)
        expected_decoder = torch.linalg.vector_norm(d_target - x_hat)
    assert torch.allclose(losses["reconstruction"], expected_reconstruction)
    assert torch.allclose(losses["latent_consistency"], expected_latent)
    assert torch.allclose(losses["decoder_consistency"], expected_decoder)
    assert torch.allclose(
        losses["total"],
        expected_reconstruction + 0.5 * (expected_latent + expected_decoder),
    )
    losses["total"].backward()
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in model.E.parameters())
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in model.D.parameters())


def test_online_update_training_preserves_reference_ids_locations_and_shapes(pipeline, tmp_path):
    context = make_context(pipeline)
    assert context.online_graph.node_features
    assert len(context.online_graph.update_sample_ids) == len(pipeline.update.scans)
    assert context.adjacency.shape == (
        len(pipeline.update.scans),
        len(pipeline.reference.reference_positions),
    )
    assert torch.allclose(context.adjacency.sum(dim=1), torch.ones(len(pipeline.update.scans)))
    assert context.tensors.node_features.shape == (len(context.online_graph.node_ids), 34)
    assert context.tensors.edge_features.shape == (
        len(context.online_graph.radio_edges) + len(context.online_graph.virtual_edges),
        34,
    )
    before_ids = tuple(scan.scan_id for scan in pipeline.reference.scans)
    before_locations = tuple(
        (position.scan_id, position.x, position.y)
        for position in pipeline.reference.reference_positions
    )

    trained = train_fingerprint_update(
        context,
        tmp_path / "update-models.pt",
        tmp_path / "updated.json",
        small_update_config(),
    )
    loaded = load_updated_fingerprints(tmp_path / "updated.json")
    assert loaded.reference_scan_ids == before_ids
    assert tuple(row.scan_id for row in loaded.rows) == before_ids
    assert tuple(
        (row.scan_id, *row.original_location)
        for row in loaded.rows
        if row.original_location is not None
    ) == before_locations
    assert all(len(row.updated_rssi_dbm) == len(pipeline.reference.bssid_vocabulary) for row in loaded.rows)
    assert len(loaded.predicted_update_locations) == len(pipeline.update.scans)
    assert len(loaded.updated_latent_reference_features) == len(before_locations)
    assert all(len(row) == 32 for row in loaded.updated_latent_reference_features)
    assert set(loaded.reference_scan_ids).isdisjoint(loaded.predicted_update_locations[i].scan_id for i in range(len(loaded.predicted_update_locations)))
    assert all(
        origin == "decoder_prediction"
        for row in loaded.rows
        for origin in row.updated_value_provenance
    )

    base_ae = load_ae_checkpoint(pipeline.ae_checkpoint, pipeline.reference)
    base_gnn = load_gnn_checkpoint(pipeline.gnn_checkpoint, pipeline.graph)
    assert any(
        not torch.equal(base_gnn.model.state_dict()[key], trained.models.gnn_model.state_dict()[key])
        for key in base_gnn.model.state_dict()
    )
    assert any(
        not torch.equal(base_ae.model.E.state_dict()[key], trained.models.ae_model.E.state_dict()[key])
        for key in base_ae.model.E.state_dict()
    )
    assert any(
        not torch.equal(base_ae.model.D.state_dict()[key], trained.models.ae_model.D.state_dict()[key])
        for key in base_ae.model.D.state_dict()
    )
    assert all(torch.isfinite(torch.tensor(value)) for value in trained.models.manifest.gnn_loss_history)
    assert all(torch.isfinite(torch.tensor(value)) for value in trained.models.manifest.ae_loss_history)
    assert trained.models.manifest.update_coordinates_used_as_training_labels is False
    assert trained.models.manifest.temporary_nodes_and_virtual_edges_retired is True


def test_checkpoint_reload_reproduces_inference_and_rejects_identity_mismatch(pipeline, tmp_path):
    context = make_context(pipeline)
    checkpoint = tmp_path / "state.pt"
    result = train_fingerprint_update(
        context,
        checkpoint,
        tmp_path / "rss.json",
        small_update_config(updater_epochs=2),
    )
    loaded_models = load_fingerprint_update_checkpoint(checkpoint, context)
    reloaded = predict_updated_fingerprints(context, loaded_models)
    for left, right in zip(result.artifact.rows, reloaded.rows):
        assert left.scan_id == right.scan_id
        assert left.updated_rssi_dbm == pytest.approx(right.updated_rssi_dbm, abs=1e-7)
    assert result.artifact.predicted_update_locations == reloaded.predicted_update_locations
    assert result.artifact.updated_latent_reference_features == reloaded.updated_latent_reference_features

    # The loader checks the exact update graph/dataset and rejects another batch.
    second_path = pipeline.root / "alternate.txt"
    second_path.write_text(
        "1,2 001122334455,-45 001122334466,-70\n",
        encoding="utf-8",
    )
    second = prepare_dataset_files(
        [second_path],
        PreprocessingConfig(
            dataset_id="update-test",
            batch_id="batch-b",
            batch_role="update_batch",
            base_bssid_vocabulary=pipeline.reference.bssid_vocabulary,
            normalization_mode="paper_ae",
        ),
        source_root=pipeline.root,
    )
    second_features = encode_prepared_dataset(
        second,
        pipeline.ae_checkpoint,
        pipeline.root / "alternate-features.json",
    )
    second_context = make_context(pipeline, update=second, update_encoder=second_features)
    with pytest.raises(ValueError, match="update prepared data|update graph|online graph"):
        load_fingerprint_update_checkpoint(checkpoint, second_context)


def test_hidden_update_coordinates_do_not_change_training_results(pipeline, tmp_path):
    context_a = make_context(pipeline)
    altered_positions = tuple(
        position.model_copy(update={"x": position.x + 900.0, "y": position.y - 400.0})
        for position in pipeline.update.evaluation_positions
    )
    hidden_changed = pipeline.update.model_copy(
        update={"evaluation_positions": altered_positions}
    )
    changed_encoder = pipeline.update_encoder.model_copy(
        update={"source_prepared_sha256": canonical_json_digest(hidden_changed)}
    )
    context_b = make_context(
        pipeline,
        update=hidden_changed,
        update_encoder=changed_encoder,
    )
    result_a = train_fingerprint_update(
        context_a,
        tmp_path / "state-a.pt",
        tmp_path / "result-a.json",
        small_update_config(updater_epochs=2),
    )
    result_b = train_fingerprint_update(
        context_b,
        tmp_path / "state-b.pt",
        tmp_path / "result-b.json",
        small_update_config(updater_epochs=2),
    )
    assert result_a.models.manifest.loss_history == result_b.models.manifest.loss_history
    assert result_a.models.manifest.gnn_loss_history == result_b.models.manifest.gnn_loss_history
    assert result_a.models.manifest.ae_loss_history == result_b.models.manifest.ae_loss_history
    for left, right in zip(result_a.artifact.rows, result_b.artifact.rows):
        assert left.updated_rssi_dbm == pytest.approx(right.updated_rssi_dbm, abs=1e-7)
    assert result_a.artifact.predicted_update_locations == result_b.artifact.predicted_update_locations


def test_changed_batch_signals_change_updated_reference_vectors(pipeline, tmp_path):
    alt_path = pipeline.root / "changed-signals.txt"
    alt_path.write_text(
        "88,99 001122334455,-80 001122334466,-40\n"
        "-7,34 001122334455,-35 001122334466,-85\n",
        encoding="utf-8",
    )
    alt_update = prepare_dataset_files(
        [alt_path],
        PreprocessingConfig(
            dataset_id="update-test",
            batch_id="batch-changed",
            batch_role="update_batch",
            base_bssid_vocabulary=pipeline.reference.bssid_vocabulary,
            normalization_mode="paper_ae",
        ),
        source_root=pipeline.root,
    )
    alt_encoder = encode_prepared_dataset(
        alt_update,
        pipeline.ae_checkpoint,
        pipeline.root / "changed-features.json",
    )
    # Deliberately controlled synthetic latent targets ensure this fixture
    # tests that the updater consumes Z_U, independently of the tiny AE's
    # observed tendency to collapse these two-feature examples to one vector.
    synthetic_update_features = tuple(
        tuple(5.0 + 0.01 * column for column in range(32))
        for _scan in alt_encoder.scan_ids
    )
    alt_encoder = EncoderFeatureArtifact.model_validate(
        {
            **alt_encoder.model_dump(mode="python"),
            "feature_matrix": synthetic_update_features,
        }
    )
    context = make_context(pipeline, update=alt_update, update_encoder=alt_encoder)
    result = train_fingerprint_update(
        context,
        tmp_path / "changed-models.pt",
        tmp_path / "changed-rss.json",
        small_update_config(updater_epochs=40),
    )
    base_context = make_context(pipeline)
    base_result = train_fingerprint_update(
        base_context,
        tmp_path / "base-models.pt",
        tmp_path / "base-rss.json",
        small_update_config(updater_epochs=40),
    )
    assert any(
        left.updated_rssi_dbm != pytest.approx(right.updated_rssi_dbm, abs=1e-5)
        for left, right in zip(result.artifact.rows, base_result.artifact.rows)
    )


def test_no_vocabulary_change_is_enforced(pipeline):
    new_ap_path = pipeline.root / "new-ap.txt"
    new_ap_path.write_text(
        "2,3 001122334455,-50 001122334466,-60 001122334477,-45\n",
        encoding="utf-8",
    )
    with_new_ap = prepare_dataset_files(
        [new_ap_path],
        PreprocessingConfig(
            dataset_id="update-test",
            batch_id="batch-new-ap",
            batch_role="update_batch",
            base_bssid_vocabulary=pipeline.reference.bssid_vocabulary,
            normalization_mode="paper_ae",
        ),
        source_root=pipeline.root,
    )
    with pytest.raises(ValueError, match="no-vocabulary-change"):
        prepare_fingerprint_update(
            pipeline.reference,
            pipeline.graph,
            pipeline.reference_encoder,
            pipeline.gnn_features,
            with_new_ap,
            pipeline.update_encoder,
            pipeline.ae_checkpoint,
            pipeline.gnn_checkpoint,
            config=small_update_config(),
        )


def test_update_uses_current_link_prediction_snapshot_and_inverse_normalization(pipeline):
    edge_delta = predict_edges(
        pipeline.graph,
        pipeline.gnn_features,
        LinkPredictionConfig(
            max_iterations=2,
            convergence_epsilon=1.0e9,
            link_threshold=0.1,
        ),
    )
    assert edge_delta.outcome != "failed"
    assert edge_delta.updated_graph is not None
    context = make_context(pipeline, reference_snapshot=edge_delta.updated_graph)
    assert context.online_graph.current_reference_snapshot_id == edge_delta.updated_graph.graph_id
    assert context.online_graph.reference_sample_ids == pipeline.graph.sample_node_ids
    assert ae_value_to_raw_dbm(0.5, "paper_ae") == pytest.approx(-60.0)
    assert ae_value_to_raw_dbm(-0.5, "legacy_raw_div_120") == pytest.approx(-60.0)


def test_online_virtual_similarity_keeps_encoder_signal_separate_from_graph_vectors(pipeline):
    context = make_context(pipeline)
    graph = context.online_graph
    assert graph.similarity_space == "z_encoder_signal_only_v1"
    assert graph.node_feature_dim == graph.encoder_feature_dim + 2
    assert len(graph.sample_signal_features) == len(graph.sample_node_ids)
    signal_by_id = dict(zip(graph.sample_node_ids, graph.sample_signal_features))
    reference_rows = pipeline.graph.id_to_row
    update_feature_by_id = dict(
        zip(pipeline.update_encoder.scan_ids, pipeline.update_encoder.feature_matrix)
    )
    for scan_id in graph.reference_sample_ids:
        assert signal_by_id[scan_id] == pytest.approx(
            pipeline.graph.node_features[reference_rows[scan_id]][: graph.encoder_feature_dim]
        )
    for scan_id in graph.update_sample_ids:
        assert signal_by_id[scan_id] == pytest.approx(update_feature_by_id[scan_id])
    for edge in graph.virtual_edges:
        left = np.asarray(signal_by_id[edge.left_sample_id])
        right = np.asarray(signal_by_id[edge.right_sample_id])
        expected = float(
            np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
        )
        assert edge.cosine_similarity == pytest.approx(expected, abs=1e-12)
        assert len(edge.feature) == graph.node_feature_dim


def test_streaming_online_graph_matches_updater_by_stable_ids(pipeline, tmp_path):
    config = LinkPredictionConfig(
        max_iterations=2,
        convergence_epsilon=1.0e9,
        link_threshold=0.1,
    )
    lp_input = tmp_path / "lp-input"
    lp_result_path = tmp_path / "lp-result"
    save_streaming_lp_input(pipeline.graph, pipeline.gnn_features, lp_input)
    lp_result = predict_edges_streaming(lp_input, lp_result_path, config)
    assert lp_result.manifest.outcome != "failed"
    batch_graph_path = tmp_path / "online-graph"
    build_streaming_online_graph(
        lp_input,
        lp_result_path,
        pipeline.update,
        pipeline.update_encoder,
        batch_graph_path,
        virtual_edge_sigma=pipeline.graph.build_config.virtual_edge_sigma,
    )
    streamed = load_streaming_online_graph(batch_graph_path)

    eager_lp = predict_edges(pipeline.graph, pipeline.gnn_features, config)
    assert eager_lp.updated_graph is not None
    eager_context = make_context(
        pipeline,
        reference_snapshot=eager_lp.updated_graph,
        config=small_update_config(
            virtual_edge_sigma=pipeline.graph.build_config.virtual_edge_sigma
        ),
    )
    expected = eager_context.online_graph
    actual_manifest = streamed.manifest
    assert actual_manifest.node_ids == expected.node_ids
    assert actual_manifest.online_graph_id.startswith("onlinegraph:")
    assert actual_manifest.labeled_reference_ids == expected.labeled_reference_ids
    assert actual_manifest.similarity_feature_space == "z_encoder_signal_only_v1"
    np.testing.assert_allclose(
        streamed.sample_signal_features,
        np.asarray(expected.sample_signal_features),
        rtol=0,
        atol=1e-12,
    )

    row_by_id = {node_id: row for row, node_id in enumerate(actual_manifest.node_ids)}
    for node_id, expected_row in zip(expected.node_ids, expected.node_features):
        np.testing.assert_allclose(
            streamed.node_features[row_by_id[node_id]], expected_row, rtol=0, atol=1e-7
        )

    actual_radio = {}
    for index, (sample_row, ap_row) in enumerate(streamed.radio_endpoints):
        key = (
            actual_manifest.node_ids[int(sample_row)],
            actual_manifest.node_ids[int(ap_row)],
        )
        actual_radio[key] = (
            float(streamed.radio_weights[index]),
            streamed.radio_edge_features[index],
        )
    assert set(actual_radio) == {
        (edge.sample_node_id, edge.ap_node_id) for edge in expected.radio_edges
    }
    for edge in expected.radio_edges:
        weight, feature = actual_radio[(edge.sample_node_id, edge.ap_node_id)]
        assert weight == pytest.approx(edge.scalar_weight)
        np.testing.assert_allclose(feature, edge.feature, rtol=0, atol=1e-7)

    actual_virtual = {
        (
            actual_manifest.sample_node_ids[int(left)],
            actual_manifest.sample_node_ids[int(right)],
        ): float(similarity)
        for (left, right), similarity in zip(
            streamed.virtual_endpoints, streamed.virtual_similarities
        )
    }
    expected_virtual = {
        (edge.left_sample_id, edge.right_sample_id): edge.cosine_similarity
        for edge in expected.virtual_edges
    }
    signal_by_id = dict(zip(expected.sample_node_ids, expected.sample_signal_features))
    for pair, similarity in expected_virtual.items():
        left, right = (np.asarray(signal_by_id[scan_id]) for scan_id in pair)
        expected_signal_cosine = float(
            np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right))
        )
        assert similarity == pytest.approx(expected_signal_cosine, abs=1e-12)
    assert actual_virtual.keys() == expected_virtual.keys()
    for pair, similarity in expected_virtual.items():
        assert actual_virtual[pair] == pytest.approx(similarity, abs=1e-12)

    dense_adjacency = np.zeros(actual_manifest.adjacency_shape, dtype=np.float64)
    indptr = streamed.update_reference_indptr
    for update_row in range(actual_manifest.adjacency_shape[0]):
        begin, end = int(indptr[update_row]), int(indptr[update_row + 1])
        dense_adjacency[update_row, streamed.update_reference_cols[begin:end]] = (
            streamed.update_reference_values[begin:end]
        )
    np.testing.assert_allclose(
        dense_adjacency,
        eager_context.adjacency.numpy(),
        rtol=0,
        atol=1e-7,
    )


def test_streaming_gnn_chunked_gradient_and_reload_match_full_graph(pipeline, tmp_path):
    config = LinkPredictionConfig(
        max_iterations=2,
        convergence_epsilon=1.0e9,
        link_threshold=0.1,
    )
    lp_input = tmp_path / "lp-input-gnn"
    lp_result_path = tmp_path / "lp-result-gnn"
    save_streaming_lp_input(pipeline.graph, pipeline.gnn_features, lp_input)
    predict_edges_streaming(lp_input, lp_result_path, config)
    online_path = tmp_path / "online-graph-gnn"
    build_streaming_online_graph(
        lp_input,
        lp_result_path,
        pipeline.update,
        pipeline.update_encoder,
        online_path,
        virtual_edge_sigma=pipeline.graph.build_config.virtual_edge_sigma,
    )

    eager_lp = predict_edges(pipeline.graph, pipeline.gnn_features, config)
    assert eager_lp.updated_graph is not None
    context = make_context(
        pipeline,
        reference_snapshot=eager_lp.updated_graph,
        config=small_update_config(
            virtual_edge_sigma=pipeline.graph.build_config.virtual_edge_sigma
        ),
    )
    source = load_gnn_checkpoint(pipeline.gnn_checkpoint, pipeline.graph)
    mapped_online = load_streaming_online_graph(online_path)
    one_layer_config = GNNModelConfig(
        graph_feature_dim=mapped_online.manifest.node_feature_dim,
        edge_feature_dim=mapped_online.manifest.node_feature_dim,
        node_output_dim=mapped_online.manifest.node_feature_dim,
        edge_output_dim=mapped_online.manifest.node_feature_dim,
        num_layers=1,
        dropout_probability=0.0,
        initialization_seed=19,
    )
    one_layer_stream = EdgeFeatureSAGE(one_layer_config)
    one_layer_eval_directory = tmp_path / "one-layer-eval"
    one_layer_eval_directory.mkdir()
    one_layer_paths, _, _, _ = _forward_stream(
        mapped_online,
        one_layer_config,
        _parameter_order(one_layer_stream),
        one_layer_eval_directory,
        chunk_size=2,
        training_seed=None,
    )
    one_layer_eager_eval = EdgeFeatureSAGE(one_layer_config)
    one_layer_eager_eval.load_state_dict(one_layer_stream.state_dict())
    one_layer_eager_eval.eval()
    one_layer_output = one_layer_eager_eval(context.tensors).node_features
    np.testing.assert_allclose(
        np.load(one_layer_paths[-1]),
        one_layer_output.detach().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    one_layer_output.retain_grad()
    one_layer_reference_loss = equation4_online_loss(
        one_layer_output, context.tensors, epsilon=1.0e-8
    )
    one_layer_reference_loss.backward()
    loss_check_dir = tmp_path / "one-layer-loss-check"
    loss_check_dir.mkdir()
    checked_loss, checked_gradient_path = _loss_and_node_gradient(
        mapped_online,
        one_layer_paths[-1],
        loss_check_dir,
        epsilon=1.0e-8,
        chunk_size=2,
        device=torch.device("cpu"),
    )
    assert checked_loss == pytest.approx(
        float(one_layer_reference_loss.detach()), abs=1e-5
    )
    np.testing.assert_allclose(
        np.load(checked_gradient_path),
        one_layer_output.grad.detach().numpy(),
        rtol=1e-5,
        atol=1e-6,
    )
    one_layer_parameters = _parameter_order(one_layer_stream)
    one_layer_loss = _StreamedEquation4.apply(
        mapped_online, one_layer_config, 1.0e-8, 1234, 2, *one_layer_parameters
    )
    one_layer_loss.backward()
    one_layer_eager = EdgeFeatureSAGE(one_layer_config)
    one_layer_eager_loss = equation4_online_loss(
        one_layer_eager(context.tensors).node_features, context.tensors, epsilon=1.0e-8
    )
    assert float(one_layer_loss.detach()) == pytest.approx(
        float(one_layer_eager_loss.detach()), rel=1e-5, abs=1e-5
    )
    one_layer_eager_loss.backward()
    for name, expected_param in sorted(
        one_layer_eager.named_parameters(), key=lambda item: (item[0] != "W1.0", item[0])
    ):
        observed_param = dict(one_layer_stream.named_parameters())[name]
        if expected_param.grad is None:
            assert observed_param.grad is None
        else:
            torch.testing.assert_close(
                observed_param.grad,
                expected_param.grad,
                rtol=2e-4,
                atol=2e-5,
                msg=lambda message: f"one layer {name}: {message}",
            )
    full_eval_model = EdgeFeatureSAGE(source.manifest.model_config_record)
    full_eval_model.load_state_dict(source.model.state_dict(), strict=True)
    full_eval_model.eval()
    diagnostic_dir = tmp_path / "stream-forward-diagnostic"
    diagnostic_dir.mkdir()
    streamed_paths, _, streamed_aggregates, _ = _forward_stream(
        mapped_online,
        full_eval_model.config,
        _parameter_order(full_eval_model),
        diagnostic_dir,
        chunk_size=2,
        training_seed=None,
    )
    graph_tensors = context.tensors
    np.testing.assert_allclose(
        np.load(streamed_paths[0]), graph_tensors.node_features.numpy(), rtol=0, atol=1e-7
    )
    edge_count = len(graph_tensors.edge_ids)
    edge_rows = torch.arange(edge_count)
    source_rows = torch.cat((graph_tensors.edge_right_rows, graph_tensors.edge_left_rows))
    target_rows = torch.cat((graph_tensors.edge_left_rows, graph_tensors.edge_right_rows))
    message_edge_rows = torch.cat((edge_rows, edge_rows))
    messages = torch.nn.functional.leaky_relu(
        torch.nn.functional.linear(
            torch.cat(
                (
                    graph_tensors.edge_features[message_edge_rows],
                    graph_tensors.node_features[source_rows],
                ),
                dim=1,
            ),
            full_eval_model.W0[0],
        ),
        negative_slope=0.1,
    )
    aggregate = graph_tensors.node_features.new_zeros(
        (graph_tensors.node_features.shape[0], graph_tensors.node_features.shape[1])
    )
    aggregate.index_add_(0, target_rows, messages)
    degrees = torch.bincount(target_rows, minlength=aggregate.shape[0]).clamp_min(1)
    np.testing.assert_allclose(
        np.load(streamed_aggregates[0]), aggregate.detach().numpy(), rtol=1e-6, atol=1e-7
    )
    aggregate = aggregate / degrees[:, None]
    manual_layer1 = torch.nn.functional.normalize(
        torch.relu(
            torch.nn.functional.linear(
                torch.cat((graph_tensors.node_features, aggregate), dim=1),
                full_eval_model.W1[0],
            )
        ),
        p=2,
        dim=1,
        eps=1e-12,
    )
    np.testing.assert_allclose(
        np.load(streamed_paths[1]), manual_layer1.detach().numpy(), rtol=1e-5, atol=1e-6
    )
    full_eval_features = full_eval_model(context.tensors).node_features.detach().numpy()
    np.testing.assert_allclose(
        np.load(streamed_paths[-1]), full_eval_features, rtol=1e-5, atol=1e-6
    )
    streamed_grad_model = EdgeFeatureSAGE(source.manifest.model_config_record)
    streamed_grad_model.load_state_dict(source.model.state_dict(), strict=True)
    streamed_parameters = _parameter_order(streamed_grad_model)
    streamed_loss = _StreamedEquation4.apply(
        mapped_online,
        streamed_grad_model.config,
        1.0e-8,
        71 + 65537,
        2,
        *streamed_parameters,
    )
    streamed_loss.backward()
    expected_grad_model = EdgeFeatureSAGE(source.manifest.model_config_record)
    expected_grad_model.load_state_dict(source.model.state_dict(), strict=True)
    expected_output = expected_grad_model(context.tensors)
    expected_loss = equation4_online_loss(
        expected_output.node_features, context.tensors, epsilon=1.0e-8
    )
    expected_loss.backward()
    for name, expected_gradient in expected_grad_model.named_parameters():
        streamed_gradient = dict(streamed_grad_model.named_parameters())[name].grad
        torch.testing.assert_close(
            streamed_gradient,
            expected_gradient.grad,
            rtol=2e-4,
            atol=2e-5,
            msg=lambda message: f"{name}: {message}",
        )
    training_config = GNNTrainingConfig(
        epochs=1,
        learning_rate=0.003,
        weight_decay=0.0,
        seed=71,
        objective_epsilon=1.0e-8,
    )
    streamed = train_streaming_gnn(
        online_path,
        source,
        tmp_path / "streamed-gnn-state",
        training_config,
        chunk_size=2,
        log_every_epochs=1,
    )

    eager_model = EdgeFeatureSAGE(source.manifest.model_config_record)
    eager_model.load_state_dict(source.model.state_dict(), strict=True)
    eager_optimizer = torch.optim.Adam(
        eager_model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    eager_model.train()
    eager_optimizer.zero_grad(set_to_none=True)
    eager_output = eager_model(context.tensors)
    eager_loss = equation4_online_loss(
        eager_output.node_features,
        context.tensors,
        epsilon=training_config.objective_epsilon,
    )
    eager_loss.backward()
    eager_optimizer.step()
    assert streamed.manifest.loss_history[0] == pytest.approx(
        float(eager_loss.detach()), rel=2e-5, abs=2e-5
    )
    for name, expected in eager_model.state_dict().items():
        torch.testing.assert_close(
            streamed.model.state_dict()[name], expected, rtol=3e-4, atol=2e-5,
            msg=lambda message: f"{name}: {message}",
        )

    loaded = load_streaming_gnn_state(
        tmp_path / "streamed-gnn-state", online_path
    )
    assert loaded.manifest.artifact_id == streamed.manifest.artifact_id
    np.testing.assert_allclose(
        loaded.node_embeddings,
        streamed.node_embeddings,
        rtol=0,
        atol=0,
    )
    reencoded_nodes, reencoded_edges = reencode_streaming_gnn(
        online_path,
        tmp_path / "streamed-gnn-state",
        tmp_path / "reencoded-stream-gnn",
        chunk_size=3,
    )
    np.testing.assert_allclose(
        reencoded_nodes, streamed.node_embeddings, rtol=1e-6, atol=1e-7
    )
    np.testing.assert_allclose(
        reencoded_edges, streamed.radio_edge_features, rtol=1e-5, atol=2e-7
    )

    online_lp_input = tmp_path / "online-lp-input"
    online_lp_manifest = save_streaming_online_lp_input(
        lp_input,
        online_path,
        learned_gnn_state=streamed,
        output_directory=online_lp_input,
    )
    loaded_lp_manifest, mapped_lp = load_streaming_lp_input(online_lp_input)
    assert loaded_lp_manifest.input_id == online_lp_manifest.input_id
    assert loaded_lp_manifest.node_ids == mapped_online.manifest.node_ids
    np.testing.assert_allclose(
        mapped_lp.node_features,
        streamed.node_embeddings,
        rtol=0,
        atol=0,
    )
    assert loaded_lp_manifest.gnn_checkpoint_sha256 == streamed.manifest.checkpoint_sha256
    assert loaded_lp_manifest.reference_label_mask[-len(loaded_lp_manifest.ap_node_ids):] == (
        False,
    ) * len(loaded_lp_manifest.ap_node_ids)
    online_lp_result = predict_edges_streaming(
        online_lp_input,
        tmp_path / "online-lp-result",
        LinkPredictionConfig(
            max_iterations=2,
            convergence_epsilon=1.0e9,
            link_threshold=0.1,
        ),
    )
    assert online_lp_result.manifest.parent_graph_id == online_lp_manifest.graph_id
    assert online_lp_result.manifest.gnn_checkpoint_sha256 == streamed.manifest.checkpoint_sha256

    updater_config = small_update_config(
        virtual_edge_sigma=1.0,
        gnn_epochs=1,
        updater_epochs=1,
        seed=71,
    )
    updater_state_path = tmp_path / "streamed-updaters"
    streamed_updaters = train_streaming_updaters(
        online_path,
        streamed,
        pipeline.reference,
        pipeline.update,
        pipeline.update_encoder,
        updater_state_path,
        updater_config,
        source_ae_checkpoint_sha256=pipeline.reference_encoder.checkpoint_sha256,
        chunk_size=1,
        log_every_epochs=1,
    )
    loaded_updaters = load_streaming_updater(
        updater_state_path, mapped_online, streamed
    )
    assert loaded_updaters.manifest.updater_id == streamed_updaters.manifest.updater_id
    assert loaded_updaters.manifest.update_ids == mapped_online.manifest.update_sample_ids
    assert loaded_updaters.predicted_update_locations.shape == (2, 2)
    assert loaded_updaters.updated_reference_latents.shape == (3, 32)

    # Compare the stream-accumulated global Frobenius losses and Adam step
    # against the existing whole-tensor reference calculation on this tiny
    # fixture. This fixture has no virtual links, so the two consistency terms
    # are zero while the supervised paths exercise both MLPs.
    online_output = streamed.model(context.tensors).node_features.detach()
    row_by_id = dict(context.online_graph.node_id_to_row)
    ref_rows = torch.tensor(
        [row_by_id[scan_id] for scan_id in context.labeled_reference_ids],
        dtype=torch.long,
    )
    update_rows = torch.tensor(
        [row_by_id[scan_id] for scan_id in context.online_graph.update_sample_ids],
        dtype=torch.long,
    )
    y_ref, adjacency = _coordinate_training_tensors(context)
    torch.manual_seed(updater_config.seed)
    eager_location = LocationPredictor(updater_config.hidden_dim, 34)
    eager_feature = LocationToFeatureUpdater(updater_config.hidden_dim, 32)
    eager_optimizer = torch.optim.Adam(
        list(eager_location.parameters()) + list(eager_feature.parameters()),
        lr=updater_config.updater_learning_rate,
    )
    eager_optimizer.zero_grad(set_to_none=True)
    eager_losses, _, _ = compute_update_losses(
        eager_location,
        eager_feature,
        online_output.index_select(0, ref_rows),
        online_output.index_select(0, update_rows),
        y_ref,
        context.update_encoder_targets,
        adjacency,
        alpha=updater_config.alpha,
    )
    eager_losses["total"].backward()
    eager_optimizer.step()
    assert streamed_updaters.manifest.loss_history[0].total == pytest.approx(
        float(eager_losses["total"].detach()), rel=3e-5, abs=3e-5
    )
    for name, expected in eager_location.state_dict().items():
        torch.testing.assert_close(
            streamed_updaters.location_model.state_dict()[name],
            expected,
            rtol=2e-4,
            atol=2e-5,
        )
    for name, expected in eager_feature.state_dict().items():
        torch.testing.assert_close(
            streamed_updaters.feature_model.state_dict()[name],
            expected,
            rtol=2e-4,
            atol=2e-5,
        )

    ae_update_path = tmp_path / "streamed-ae-update"
    streamed_ae_update = train_streaming_ae_update(
        pipeline.reference,
        pipeline.update,
        pipeline.ae_checkpoint,
        loaded_updaters,
        ae_update_path,
        updater_config,
        chunk_size=1,
        log_every_epochs=1,
    )
    reloaded_ae_update = load_streaming_ae_update(
        ae_update_path, pipeline.reference, loaded_updaters
    )
    portable_ae_checkpoint = load_ae_checkpoint(
        ae_update_path / "updated_ae.pt", pipeline.reference, device="cpu"
    )
    assert reloaded_ae_update.manifest.update_id == streamed_ae_update.manifest.update_id
    assert reloaded_ae_update.manifest.reference_scan_ids == tuple(
        scan.scan_id for scan in pipeline.reference.scans
    )
    assert reloaded_ae_update.manifest.labeled_reference_ids == tuple(
        row.scan_id for row in pipeline.reference.reference_positions
    )
    assert reloaded_ae_update.updated_rssi_dbm.shape == (3, 2)
    assert reloaded_ae_update.source_rssi_dbm.shape == (3, 2)
    assert reloaded_ae_update.observation_mask.shape == (3, 2)
    assert reloaded_ae_update.value_provenance.shape == (3, 2)
    assert np.isfinite(reloaded_ae_update.updated_rssi_dbm).all()
    assert np.all(reloaded_ae_update.value_provenance == 1)
    assert reloaded_ae_update.optimizer_state
    assert reloaded_ae_update.manifest.objective == "eq10_exact_global_frobenius_chunk_dropout_v1"
    assert portable_ae_checkpoint.manifest.checkpoint_objective == "streaming_eq10_update_v1"
    assert portable_ae_checkpoint.manifest.training_resume_compatible is False
    extended_vocabulary = pipeline.reference.bssid_vocabulary + ("001122334477",)
    migrated_ae = migrate_ae_checkpoint_vocabulary(
        ae_update_path / "updated_ae.pt",
        extended_vocabulary,
        tmp_path / "streaming-eq10-vocabulary-migration.pt",
        seed=37,
        source_model=portable_ae_checkpoint.model,
        source_optimizer_state=portable_ae_checkpoint.optimizer_state,
    )
    assert migrated_ae.manifest.bssid_vocabulary == extended_vocabulary
    assert migrated_ae.manifest.checkpoint_objective == "streaming_eq10_update_v1"
    assert migrated_ae.manifest.vocabulary_migration is not None
    assert migrated_ae.model.input_dim == len(extended_vocabulary)


def test_streaming_online_lp_extension_initializes_new_ap_by_full_identity(
    pipeline, tmp_path
):
    """A newly observed full BSSID is joined after old-vocabulary inference."""
    transition_graph = build_graph_state(
        pipeline.reference,
        pipeline.reference_encoder,
        GraphBuildConfig(
            virtual_edge_strategy="paper_cosine_threshold",
            virtual_edge_sigma=-1.0,
            pseudo_location_seed=13,
        ),
    )
    transition_gnn_checkpoint_path = tmp_path / "transition-source-gnn.pt"
    transition_gnn_features = tmp_path / "transition-source-gnn-features.json"
    train_gnn(
        transition_graph,
        EdgeFeatureSAGE(
            GNNModelConfig(
                graph_feature_dim=transition_graph.node_feature_dim,
                edge_feature_dim=transition_graph.node_feature_dim,
                node_output_dim=transition_graph.node_feature_dim,
                edge_output_dim=transition_graph.node_feature_dim,
                num_layers=2,
                dropout_probability=0.0,
                initialization_seed=71,
            )
        ),
        transition_gnn_checkpoint_path,
        transition_gnn_features,
        training_config=GNNTrainingConfig(
            epochs=1,
            learning_rate=0.001,
            seed=71,
            objective_epsilon=1e-8,
        ),
    )
    transition_gnn_feature_artifact = load_gnn_features(transition_gnn_features)
    new_bssid = "001122334477"
    raw_update_path = tmp_path / "new-ap-update.txt"
    raw_update_path.write_text(
        "1,2 001122334455,-60 001122334477,-60\n"
        "3,4 001122334466,-70 001122334477,-80\n",
        encoding="utf-8",
    )
    full_update = prepare_dataset_files(
        [raw_update_path],
        PreprocessingConfig(
            dataset_id=pipeline.reference.config.dataset_id,
            batch_id="new-ap-batch",
            batch_role="update_batch",
            base_bssid_vocabulary=pipeline.reference.bssid_vocabulary,
        ),
        source_root=tmp_path,
    )
    projected_update = project_prepared_batch_to_vocabulary(
        full_update, pipeline.reference.bssid_vocabulary
    )
    update_encoder = encode_prepared_dataset(
        projected_update,
        pipeline.ae_checkpoint,
        tmp_path / "projected-update-features.json",
        batch_size=1,
    )

    reference_lp_input = tmp_path / "new-ap-reference-lp-input"
    reference_lp_result = tmp_path / "new-ap-reference-lp-result"
    save_streaming_lp_input(
        transition_graph, transition_gnn_feature_artifact, reference_lp_input
    )
    predict_edges_streaming(
        reference_lp_input,
        reference_lp_result,
        LinkPredictionConfig(
            max_iterations=2,
            convergence_epsilon=1e9,
            link_threshold=0.0,
        ),
    )

    online_graph_path = tmp_path / "new-ap-online-graph"
    build_streaming_online_graph(
        reference_lp_input,
        reference_lp_result,
        projected_update,
        update_encoder,
        online_graph_path,
        virtual_edge_sigma=-1.0,
    )
    source_gnn = load_gnn_checkpoint(
        transition_gnn_checkpoint_path, transition_graph, device="cpu"
    )
    online_gnn = train_streaming_gnn(
        online_graph_path,
        source_gnn,
        tmp_path / "new-ap-online-gnn",
        GNNTrainingConfig(
            epochs=1,
            learning_rate=0.001,
            seed=71,
            objective_epsilon=1e-8,
        ),
        chunk_size=2,
        device="cpu",
        log_every_epochs=1,
    )
    update_config = FingerprintUpdateConfig(
        hidden_dim=8,
        alpha=0.5,
        virtual_edge_sigma=-1.0,
        updater_epochs=1,
        ae_epochs=1,
        seed=73,
    )
    streamed_updaters = train_streaming_updaters(
        online_graph_path,
        online_gnn,
        pipeline.reference,
        projected_update,
        update_encoder,
        tmp_path / "new-ap-streamed-updaters",
        update_config,
        source_ae_checkpoint_sha256=load_ae_checkpoint(
            pipeline.ae_checkpoint, pipeline.reference
        ).checkpoint_sha256,
        chunk_size=2,
        log_every_epochs=1,
    )
    streamed_ae_update = train_streaming_ae_update(
        pipeline.reference,
        projected_update,
        pipeline.ae_checkpoint,
        streamed_updaters,
        tmp_path / "new-ap-streamed-ae-update",
        update_config,
        chunk_size=2,
        log_every_epochs=1,
    )
    extended_lp_path = tmp_path / "new-ap-transition-lp-input"
    extended = save_streaming_online_lp_input_with_new_aps(
        reference_lp_input,
        online_graph_path,
        learned_gnn_state=online_gnn,
        full_update_prepared=full_update,
        output_directory=extended_lp_path,
    )
    extended, mapped = load_streaming_lp_input(extended_lp_path)

    assert extended.bssid_vocabulary == pipeline.reference.bssid_vocabulary + (new_bssid,)
    assert extended.ap_node_ids[-1] == f"ap:{new_bssid}"
    assert extended.node_ids == extended.sample_node_ids + extended.ap_node_ids
    assert extended.radio_edge_count >= 2
    assert len(extended.reference_sample_mask) == len(extended.node_ids)
    assert len(extended.new_sample_mask) == len(extended.node_ids)
    assert not extended.reference_label_mask[-1]
    assert extended.reference_coordinate_targets[-1] is None

    gnn_row = {node_id: row for row, node_id in enumerate(online_gnn.manifest.node_ids)}
    weights = np.asarray([60.0, 40.0], dtype=np.float64)
    expected_feature = sum(
        weights[row]
        * online_gnn.node_embeddings[gnn_row[full_update.scans[row].scan_id]]
        for row in range(2)
    ) / weights.sum()
    ap_row = extended.node_ids.index(f"ap:{new_bssid}")
    np.testing.assert_allclose(
        mapped.node_features[ap_row], expected_feature, rtol=0, atol=1e-6
    )

    scan_rows = {
        scan_id: row for row, scan_id in enumerate(extended.sample_node_ids)
    }
    new_ap_global_row = ap_row
    incident_weights = []
    for scan in full_update.scans:
        start, stop = mapped.radio_indptr[
            scan_rows[scan.scan_id] : scan_rows[scan.scan_id] + 2
        ]
        start, stop = int(start), int(stop)
        for position in range(start, stop):
            if int(mapped.radio_neighbor_rows[position]) == new_ap_global_row:
                incident_weights.append(float(mapped.radio_neighbor_weights[position]))
    assert sorted(incident_weights) == [40.0, 60.0]

    candidate_db = tmp_path / "ap-transition-candidates.sqlite3"
    candidates = iter_streaming_ap_transition_candidates(
        reference_lp_input,
        reference_lp_result,
        online_graph_path,
        full_update_prepared=full_update,
        extended_lp_manifest=extended,
        sqlite_path=candidate_db,
    )
    candidate_values = tuple(candidates)
    assert candidate_values == tuple(sorted(set(candidate_values)))
    assert candidate_values
    online_graph = load_streaming_online_graph(online_graph_path)
    reference_id_set = set(online_graph.manifest.reference_sample_ids)
    assert all(sample_id in reference_id_set for sample_id, _ in candidate_values)
    assert all(ap_id in extended.ap_node_ids for _, ap_id in candidate_values)
    assert any(ap_id == f"ap:{new_bssid}" for _, ap_id in candidate_values)
    transition_result = predict_edges_streaming(
        extended_lp_path,
        tmp_path / "ap-transition-result",
        LinkPredictionConfig(
            max_iterations=2,
            convergence_epsilon=1e9,
            link_threshold=0.0,
        ),
        candidate_pairs=iter(candidate_values),
    )
    assert transition_result.manifest.candidate_count == len(candidate_values)
    assert all(
        score.sample_node_id in reference_id_set
        for score in transition_result.iter_candidate_scores()
    )
    integrated = infer_streaming_ap_graph_changes(
        reference_lp_input,
        reference_lp_result,
        online_graph_path,
        learned_gnn_state=online_gnn,
        full_update_prepared=full_update,
        input_directory=tmp_path / "ap-transition-wrapped-input",
        candidate_index_path=tmp_path / "ap-transition-wrapped-candidates.sqlite3",
        result_directory=tmp_path / "ap-transition-wrapped-result",
        config=LinkPredictionConfig(
            max_iterations=2,
            convergence_epsilon=1e9,
            link_threshold=0.0,
        ),
    )
    assert integrated.result.manifest.candidate_policy == "explicit_sorted_stable_ids_v1"
    assert integrated.result.manifest.candidate_count == len(candidate_values)
    applied = apply_streaming_ap_delta_to_reference(
        reference_lp_input,
        reference_lp_result,
        integrated.input_directory,
        integrated.result.directory,
        full_update,
        streamed_ae_update,
        tmp_path / "streamed-reference-transition",
    )
    reloaded = load_streaming_reference_transition(applied.directory)
    assert reloaded.manifest.reference_scan_ids == tuple(
        scan.scan_id for scan in pipeline.reference.scans
    )
    assert reloaded.manifest.original_coordinates == tuple(
        next(
            ((position.x, position.y) for position in pipeline.reference.reference_positions if position.scan_id == scan.scan_id),
            None,
        )
        for scan in pipeline.reference.scans
    )
    assert reloaded.manifest.vocabulary_after == pipeline.reference.bssid_vocabulary + (
        new_bssid,
    )
    assert reloaded.manifest.admitted_bssids == (new_bssid,)
    assert reloaded.updated_rssi_dbm.shape == (len(pipeline.reference.scans), 3)
    new_col = reloaded.manifest.vocabulary_after.index(new_bssid)
    assert np.any(reloaded.value_provenance[:, new_col] == 2)
    assert not np.any(reloaded.observation_mask[:, new_col])

    eq10_checkpoint = load_ae_checkpoint(
        tmp_path / "new-ap-streamed-ae-update" / "updated_ae.pt",
        pipeline.reference,
    )
    migrated_path = tmp_path / "new-ap-migrated-ae.pt"
    migrated = migrate_ae_checkpoint_vocabulary(
        eq10_checkpoint.path,
        reloaded.manifest.vocabulary_after,
        migrated_path,
        seed=79,
        source_model=eq10_checkpoint.model,
        source_optimizer_state=eq10_checkpoint.optimizer_state,
    )
    post_transition_ae = fine_tune_streaming_ae_after_ap_transition(
        reloaded,
        streamed_ae_update,
        streamed_updaters,
        migrated.path,
        tmp_path / "new-ap-post-transition-ae",
        update_config,
        chunk_size=2,
        log_every_epochs=1,
    )
    reloaded_post_transition_ae = load_streaming_post_transition_ae(
        post_transition_ae.directory
    )
    transitioned_prepared = materialize_streaming_reference_prepared(
        pipeline.reference, reloaded
    )
    assert tuple(scan.scan_id for scan in transitioned_prepared.scans) == tuple(
        scan.scan_id for scan in pipeline.reference.scans
    )
    assert transitioned_prepared.bssid_vocabulary == reloaded.manifest.vocabulary_after
    assert not transitioned_prepared.evaluation_positions
    assert reloaded_post_transition_ae.manifest.bssid_vocabulary == reloaded.manifest.vocabulary_after
    assert reloaded_post_transition_ae.updated_rssi_dbm.shape == reloaded.updated_rssi_dbm.shape
    assert reloaded_post_transition_ae.model.input_dim == len(reloaded.manifest.vocabulary_after)
    assert np.isfinite(reloaded_post_transition_ae.updated_rssi_dbm).all()
    final_portable_ae = load_ae_checkpoint(
        post_transition_ae.directory / "post_transition_ae.pt"
    )
    assert final_portable_ae.manifest.initialization_mode == "vocabulary_migration"
    assert final_portable_ae.manifest.training_resume_compatible is False
    reference_graph_path = tmp_path / "post-transition-reference-graph"
    rebuilt_graph_manifest = build_streaming_reference_graph_after_transition(
        reference_lp_input,
        reloaded,
        reloaded_post_transition_ae,
        full_update,
        online_gnn,
        reference_graph_path,
        reference_prepared=transitioned_prepared,
        device="cpu",
        chunk_size=2,
    )
    rebuilt_graph = load_streaming_online_graph(reference_graph_path)
    assert rebuilt_graph_manifest.update_sample_ids == ()
    assert rebuilt_graph_manifest.reference_sample_ids == tuple(
        scan.scan_id for scan in pipeline.reference.scans
    )
    assert rebuilt_graph_manifest.bssid_vocabulary == reloaded.manifest.vocabulary_after
    assert rebuilt_graph.node_features.shape == (
        len(pipeline.reference.scans) + len(reloaded.manifest.vocabulary_after),
        pipeline.gnn_features.node_feature_dim,
    )
    post_graph_gnn = train_streaming_gnn(
        reference_graph_path,
        online_gnn,
        tmp_path / "post-transition-reference-gnn",
        GNNTrainingConfig(epochs=1, learning_rate=0.001, seed=83),
        chunk_size=2,
        device="cpu",
        log_every_epochs=1,
    )
    assert post_graph_gnn.manifest.node_ids == rebuilt_graph_manifest.node_ids
    assert post_graph_gnn.manifest.reference_graph_id == rebuilt_graph_manifest.update_graph_id
    current_reference_lp = save_streaming_online_lp_input(
        reference_lp_input,
        reference_graph_path,
        learned_gnn_state=post_graph_gnn,
        output_directory=tmp_path / "post-transition-reference-lp-input",
        reference_prepared=transitioned_prepared,
    )
    assert current_reference_lp.graph_id == rebuilt_graph_manifest.update_graph_id
    assert current_reference_lp.batch_role == "bootstrap_reference"
    assert current_reference_lp.batch_id == full_update.config.batch_id
    assert current_reference_lp.bssid_vocabulary == transitioned_prepared.bssid_vocabulary

