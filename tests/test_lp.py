import math
from pathlib import Path

import pytest

from ae import EncoderFeatureArtifact
from graph_build import GraphBuildConfig, build_graph_state
from gufu_artifacts import canonical_json_digest
from gufu_data import PreprocessingConfig, prepare_dataset_files
from lp import (
    LinkPredictionConfig,
    LinkPredictionError,
    apply_edge_delta,
    load_edge_delta,
    predict_edges,
    save_edge_delta,
)
from sage_ve import GNNFeatureArtifact, GNNModelConfig


def graph_fixture(
    tmp_path: Path,
    lines=None,
    *,
    dataset_id="lp-test",
    virtual_edge_sigma=-1.0,
):
    lines = lines or [
        "0,0 001122334455,-50",
        "1,0 001122334455,-60",
        "2,0",
    ]
    root = tmp_path / f"input-{dataset_id}"
    root.mkdir(parents=True, exist_ok=True)
    source = root / "bootstrap.txt"
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    prepared = prepare_dataset_files(
        [source],
        PreprocessingConfig(
            dataset_id=dataset_id,
            batch_id="bootstrap",
            batch_role="bootstrap_reference",
            normalization_mode="paper_ae",
            empty_scan_policy="keep",
        ),
        source_root=root,
    )
    scans = tuple(scan.scan_id for scan in prepared.scans)
    encoder_rows = tuple(
        tuple(1.0 if column == 0 else 0.0 for column in range(32))
        for _scan_id in scans
    )
    encoder = EncoderFeatureArtifact(
        dataset_id=prepared.config.dataset_id,
        batch_id=prepared.config.batch_id,
        source_prepared_sha256=canonical_json_digest(prepared),
        scan_ids=scans,
        feature_matrix=encoder_rows,
        checkpoint_sha256="a" * 64,
        bssid_vocabulary=prepared.bssid_vocabulary,
        vocabulary_sha256=canonical_json_digest(prepared.bssid_vocabulary),
        normalization=prepared.config.normalization_mode,
        dtype="float32",
        created_at_utc="2030-01-01T00:00:00+00:00",
    )
    graph = build_graph_state(
        prepared,
        encoder,
        GraphBuildConfig(virtual_edge_sigma=virtual_edge_sigma),
    )
    return graph


def learned_features(graph, *, row_values=None, checkpoint="b" * 64):
    row_values = row_values or {}
    vectors = []
    for node_id in graph.node_ids:
        value = row_values.get(node_id, 0.0)
        vectors.append(tuple(float(value) for _ in range(34)))
    return GNNFeatureArtifact(
        graph_id=graph.graph_id,
        graph_build_config=graph.build_config,
        dataset_id=graph.dataset_id,
        batch_id=graph.batch_id,
        source_prepared_sha256=graph.source_prepared_sha256,
        encoder_feature_artifact_sha256=graph.encoder_feature_artifact_sha256,
        encoder_checkpoint_sha256=graph.encoder_checkpoint_sha256,
        normalization=graph.normalization,
        vocabulary_sha256=graph.vocabulary_sha256,
        bssid_vocabulary=graph.bssid_vocabulary,
        checkpoint_sha256=checkpoint,
        model_config_record=GNNModelConfig(num_layers=1, dropout_probability=0.0),
        node_ids=graph.node_ids,
        node_types=graph.node_types,
        node_id_to_row=graph.node_id_to_row,
        node_feature_matrix=tuple(vectors),
        radio_edge_ids=tuple(edge.edge_id for edge in graph.radio_edges),
        radio_edge_feature_matrix=tuple(
            tuple(0.0 for _ in range(34)) for _edge in graph.radio_edges
        ),
        virtual_edge_ids=tuple(edge.edge_id for edge in graph.virtual_edges),
        virtual_edge_feature_matrix_initial=tuple(
            tuple(0.0 for _ in range(34)) for _edge in graph.virtual_edges
        ),
        created_at_utc="2030-01-01T00:00:00+00:00",
    )


def _adjacency(graph):
    radio_max = max((edge.scalar_weight for edge in graph.radio_edges), default=0.0)
    neighbors = {node_id: [] for node_id in graph.node_ids}
    for edge in graph.radio_edges:
        weight = edge.scalar_weight / radio_max if radio_max else 0.0
        neighbors[edge.sample_node_id].append((edge.ap_node_id, weight))
        neighbors[edge.ap_node_id].append((edge.sample_node_id, weight))
    for edge in graph.virtual_edges:
        neighbors[edge.sample_node_id_left].append(
            (edge.sample_node_id_right, edge.cosine_similarity)
        )
        neighbors[edge.sample_node_id_right].append(
            (edge.sample_node_id_left, edge.cosine_similarity)
        )
    return neighbors


def test_eq_11_12_are_one_synchronous_hand_computed_iteration(tmp_path):
    graph = graph_fixture(tmp_path, lines=[
        "0,0 001122334455,-50",
        "1,0 001122334455,-60",
    ])
    initial = {node_id: (index + 1) / 10.0 for index, node_id in enumerate(graph.node_ids)}
    learned = learned_features(graph, row_values=initial)
    result = predict_edges(
        graph,
        learned,
        LinkPredictionConfig(
            max_iterations=1,
            convergence_epsilon=1e9,
            link_threshold=1000.0,
        ),
    )

    neighbors = _adjacency(graph)
    row_by_id = {node_id: row for row, node_id in enumerate(graph.node_ids)}
    old_f = {node_id: initial[node_id] for node_id in graph.node_ids}
    expected_g = {}
    for node_id in graph.node_ids:
        adjacent = neighbors[node_id]
        expected_g[node_id] = (
            sum(old_f[neighbor_id] * weight for neighbor_id, weight in adjacent)
            / len(adjacent)
            if adjacent else old_f[node_id]
        )
    expected_f = {}
    for node_id in graph.node_ids:
        adjacent = neighbors[node_id]
        expected_f[node_id] = (
            1.0 - sum(abs(weight - expected_g[neighbor_id]) for neighbor_id, weight in adjacent)
            / (2 * len(adjacent))
            if adjacent else old_f[node_id]
        )

    for node_id, row in row_by_id.items():
        assert result.goodness_features[row] == pytest.approx((expected_g[node_id],) * 34)
        assert result.fairness_features[row] == pytest.approx((expected_f[node_id],) * 34)
    assert len(set(expected_g.values())) > 1  # each node uses its own neighborhood
    assert result.goodness_features != result.fairness_features  # independent G/F buffers
    assert result.iterations == 1
    assert result.threshold_units == "rssi_plus_120_scalar_weight"
    assert result.radio_weight_max == pytest.approx(70.0)
    assert result.outcome == "converged"
    assert result.changed_edges == ()


def test_supported_missing_edge_is_added_once_with_raw_weight_and_paired_arcs(tmp_path):
    graph = graph_fixture(tmp_path)
    learned = learned_features(graph)
    original_graph = graph.model_copy(deep=True)
    original_features = learned.model_copy(deep=True)
    result = predict_edges(
        graph,
        learned,
        LinkPredictionConfig(
            max_iterations=1,
            convergence_epsilon=1e9,
            link_threshold=0.0,
        ),
    )

    assert result.outcome == "converged"
    assert result.added_edges
    assert result.changed_edges == ()
    assert len({(s.sample_node_id, s.ap_node_id) for s in result.candidate_scores}) == len(
        result.candidate_scores
    )
    assert all(
        score.sample_node_id in graph.sample_node_ids
        and score.ap_node_id in graph.ap_node_ids
        and score.sample_node_id != score.ap_node_id
        for score in result.candidate_scores
    )
    supported_pairs = set()
    for virtual in graph.virtual_edges:
        for radio in graph.radio_edges:
            if radio.sample_node_id == virtual.sample_node_id_left:
                supported_pairs.add((virtual.sample_node_id_right, radio.ap_node_id))
            if radio.sample_node_id == virtual.sample_node_id_right:
                supported_pairs.add((virtual.sample_node_id_left, radio.ap_node_id))
    assert all(
        (edge.sample_node_id, edge.ap_node_id) in supported_pairs
        for edge in result.added_edges
    )
    for edge in result.added_edges:
        assert edge.new_weight == pytest.approx(edge.predicted_weight)
        assert edge.new_weight == pytest.approx(0.0)
        assert edge.new_weight - 120.0 == pytest.approx(-120.0)
    updated = result.updated_graph
    assert updated is not None
    assert len(updated.radio_edges) == len(graph.radio_edges) + len(result.added_edges)
    assert updated.virtual_edges_removed_after_prediction is True
    assert graph == original_graph
    assert learned == original_features

    directed = updated.directed_radio_edges()
    arcs_by_pair = {}
    for left, right, weight in directed:
        pair = frozenset((left, right))
        arcs_by_pair.setdefault(pair, []).append((left, right, weight))
    assert all(len(arcs) == 2 and arcs[0][2] == arcs[1][2] for arcs in arcs_by_pair.values())

    unit_value = 1.0 / math.sqrt(34.0)
    normalized_embeddings = learned_features(
        graph,
        row_values={node_id: unit_value for node_id in graph.node_ids},
    )
    scaled = predict_edges(
        graph,
        normalized_embeddings,
        LinkPredictionConfig(
            max_iterations=1,
            convergence_epsilon=1e9,
            link_threshold=0.1,
        ),
    )
    assert scaled.added_edges
    assert scaled.threshold_units == "rssi_plus_120_scalar_weight"
    assert scaled.updated_graph is not None
    scaled_edges_by_pair = {
        (edge.sample_node_id, edge.ap_node_id): edge
        for edge in scaled.updated_graph.radio_edges
    }
    for change in scaled.added_edges:
        srow = scaled.node_ids.index(change.sample_node_id)
        arow = scaled.node_ids.index(change.ap_node_id)
        expected_raw_weight = scaled.radio_weight_max / 2.0 * (
            sum(
                scaled.goodness_features[srow][column]
                * scaled.fairness_features[arow][column]
                + scaled.fairness_features[srow][column]
                * scaled.goodness_features[arow][column]
                for column in range(34)
            )
        )
        assert change.predicted_weight == pytest.approx(expected_raw_weight)
        assert change.new_weight == pytest.approx(expected_raw_weight)
        assert change.new_weight > scaled.threshold_value
        assert change.new_weight - 120.0 == pytest.approx(
            scaled_edges_by_pair[(change.sample_node_id, change.ap_node_id)].rssi_dbm
        )


def test_low_scores_remove_existing_edges_and_orphan_aps_are_tombstoned(tmp_path):
    graph = graph_fixture(tmp_path, lines=[
        "0,0 001122334455,-50",
        "1,0 001122334455,-60",
    ])
    learned = learned_features(graph)
    result = predict_edges(
        graph,
        learned,
        LinkPredictionConfig(
            max_iterations=1,
            convergence_epsilon=1e9,
            link_threshold=0.1,
        ),
    )

    assert result.outcome == "converged"
    assert len(result.removed_edges) == len(graph.radio_edges)
    assert result.added_edges == ()
    assert result.ap_removals == graph.ap_node_ids
    assert result.updated_graph is not None
    assert result.updated_graph.radio_edges == ()
    assert result.updated_graph.active_ap_node_ids == ()
    assert result.updated_graph.removed_ap_node_ids == graph.ap_node_ids
    assert result.updated_graph.ap_ids == graph.ap_ids  # retain vocabulary identity


def test_no_change_and_failed_iteration_are_distinct_and_failure_is_not_applicable(tmp_path):
    graph = graph_fixture(
        tmp_path,
        lines=["0,0 001122334455,-50", "1,0 001122334455,-60"],
        virtual_edge_sigma=1.0,
    )
    learned = learned_features(graph, row_values={node_id: 0.2 for node_id in graph.node_ids})
    no_change = predict_edges(
        graph,
        learned,
        LinkPredictionConfig(
            max_iterations=1,
            convergence_epsilon=1e9,
            link_threshold=0.0,
        ),
    )
    assert no_change.outcome == "no_change"
    assert no_change.updated_graph is not None
    assert no_change.updated_graph.radio_edges

    stalled = predict_edges(
        graph,
        learned,
        LinkPredictionConfig(max_iterations=1, convergence_epsilon=0.0),
    )
    assert stalled.outcome == "failed"
    assert stalled.termination == "max_iterations"
    assert stalled.failure_reason
    assert stalled.updated_graph is None
    with pytest.raises(LinkPredictionError, match="failed"):
        apply_edge_delta(graph, learned, stalled)


def test_zero_weight_edge_and_isolated_sample_have_finite_defined_state(tmp_path):
    graph = graph_fixture(
        tmp_path,
        lines=[
            "0,0 001122334455,-120",
            "1,0 001122334455,-60",
            "2,0",
        ],
        virtual_edge_sigma=1.0,
    )
    isolated_id = graph.sample_node_ids[2]
    learned = learned_features(
        graph,
        row_values={isolated_id: 0.25},
    )
    result = predict_edges(
        graph,
        learned,
        LinkPredictionConfig(
            max_iterations=1,
            convergence_epsilon=1e9,
            link_threshold=0.0,
        ),
    )
    row = result.node_ids.index(isolated_id)
    assert result.goodness_features[row] == pytest.approx((0.25,) * 34)
    assert result.fairness_features[row] == pytest.approx((0.25,) * 34)
    assert result.radio_weight_max == pytest.approx(60.0)
    assert all(
        math.isfinite(value)
        for matrix in (result.goodness_features, result.fairness_features)
        for vector in matrix
        for value in vector
    )
    assert result.updated_graph is not None


def test_empty_radio_graph_is_immediate_isolated_no_change(tmp_path):
    graph = graph_fixture(
        tmp_path,
        lines=["0,0"],
        virtual_edge_sigma=1.0,
    )
    learned = learned_features(graph, row_values={graph.sample_node_ids[0]: 0.125})
    result = predict_edges(graph, learned)

    assert graph.radio_edges == ()
    assert graph.virtual_edges == ()
    assert result.radio_weight_max == 0.0
    assert result.iterations == 0
    assert result.goodness_residual == 0.0
    assert result.fairness_residual == 0.0
    assert result.outcome == "no_change"
    assert result.candidate_scores == ()
    assert result.updated_graph is not None
    assert result.updated_graph.node_ids == graph.node_ids


def test_artifact_round_trip_and_explicit_delta_application_preserve_identity(tmp_path):
    graph = graph_fixture(tmp_path)
    learned = learned_features(graph)
    result = predict_edges(
        graph,
        learned,
        LinkPredictionConfig(
            max_iterations=1,
            convergence_epsilon=1e9,
            link_threshold=0.0,
        ),
    )
    path = save_edge_delta(result, tmp_path / "lp-result.json")
    restored = load_edge_delta(path)
    assert restored == result
    applied = apply_edge_delta(graph, learned, restored)
    assert applied == restored.updated_graph
    assert applied.graph_id == restored.updated_graph.graph_id
    assert applied.parent_graph_id == graph.graph_id

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_edge_delta(result, path)


def test_missing_or_incompatible_learned_state_fails_clearly(tmp_path):
    graph = graph_fixture(tmp_path)
    with pytest.raises(TypeError, match="GNNFeatureArtifact"):
        predict_edges(graph, None)

    learned = learned_features(graph)
    incompatible = learned.model_copy(update={"graph_id": "graph:" + "c" * 64})
    with pytest.raises(LinkPredictionError, match="graph ID"):
        predict_edges(graph, incompatible)
