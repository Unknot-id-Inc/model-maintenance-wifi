import json
from pathlib import Path

import numpy as np
import pytest

from lp import LinkPredictionConfig, predict_edges
from lp_streaming import (
    load_streaming_lp_input,
    load_streaming_lp_result,
    predict_edges_streaming,
    save_streaming_lp_input,
    validate_streaming_lp_input,
)
from test_lp import graph_fixture, learned_features


def _run_streamed(tmp_path: Path, graph, learned, config):
    input_path = tmp_path / "stream-input"
    result_path = tmp_path / "stream-result"
    input_manifest = save_streaming_lp_input(graph, learned, input_path)
    assert validate_streaming_lp_input(graph, learned, input_path).input_id == input_manifest.input_id
    loaded_manifest, mapped = load_streaming_lp_input(input_path)
    assert loaded_manifest.input_id == input_manifest.input_id
    assert mapped.node_features.shape == (graph.node_count, graph.node_feature_dim)
    assert isinstance(mapped.node_features, np.memmap)
    result = predict_edges_streaming(input_path, result_path, config)
    return input_path, result_path, result


@pytest.mark.parametrize(
    "threshold,feature_value,expected_outcome",
    [
        (0.0, 0.2, "converged"),  # supported missing edges are added
        (0.1, 0.0, "converged"),  # existing links are removed and APs orphaned
        (0.0, 0.2, "no_change"),  # one sample: no supported non-edge candidates
    ],
)
def test_streaming_lp_matches_in_memory_equations_and_updated_graph(
    tmp_path, threshold, feature_value, expected_outcome
):
    lines = (
        ["0,0 001122334455,-50", "1,0"]
        if expected_outcome != "no_change"
        else ["0,0 001122334455,-50"]
    )
    graph = graph_fixture(
        tmp_path,
        lines=lines,
        dataset_id=f"stream-{expected_outcome}-{threshold}-{feature_value}",
        virtual_edge_sigma=-1.0,
    )
    learned = learned_features(
        graph,
        row_values={node_id: feature_value for node_id in graph.node_ids},
    )
    config = LinkPredictionConfig(
        max_iterations=2,
        convergence_epsilon=1e9,
        link_threshold=threshold,
    )

    expected = predict_edges(graph, learned, config)
    input_path, result_path, actual = _run_streamed(
        tmp_path, graph, learned, config
    )
    actual = load_streaming_lp_result(
        result_path, input_directory=input_path
    )

    assert actual.manifest.outcome == expected_outcome
    assert actual.manifest.outcome == expected.outcome
    assert actual.manifest.updated_graph_id == expected.updated_graph.graph_id
    assert actual.manifest.iterations == expected.iterations
    assert actual.manifest.goodness_residual == pytest.approx(
        expected.goodness_residual, abs=1e-12
    )
    assert actual.manifest.fairness_residual == pytest.approx(
        expected.fairness_residual, abs=1e-12
    )
    np.testing.assert_allclose(
        np.load(result_path / "goodness_features.npy"),
        np.asarray(expected.goodness_features),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.load(result_path / "fairness_features.npy"),
        np.asarray(expected.fairness_features),
        rtol=0,
        atol=1e-12,
    )
    assert tuple(actual.iter_candidate_scores()) == expected.candidate_scores
    assert tuple(actual.iter_added_edges()) == expected.added_edges
    assert tuple(actual.iter_removed_edges()) == expected.removed_edges

    updated = expected.updated_graph
    assert updated is not None
    endpoints = np.load(result_path / "updated_edge_endpoints.npy")
    weights = np.load(result_path / "updated_edge_weights.npy")
    features = np.load(result_path / "updated_edge_features.npy")
    sources = np.load(result_path / "updated_edge_feature_sources.npy")
    actual_pairs = []
    for index, (sample_row, ap_row) in enumerate(endpoints):
        sample_id = actual.manifest.node_ids[int(sample_row)]
        ap_id = actual.manifest.node_ids[int(ap_row)]
        actual_pairs.append((sample_id, ap_id, weights[index], features[index], sources[index]))
    assert len(actual_pairs) == len(updated.radio_edges)
    for row, edge in zip(actual_pairs, updated.radio_edges):
        sample_id, ap_id, weight, feature, source = row
        assert (sample_id, ap_id) == (edge.sample_node_id, edge.ap_node_id)
        assert weight == pytest.approx(edge.scalar_weight)
        np.testing.assert_allclose(feature, edge.edge_feature, rtol=0, atol=1e-12)
        assert source == (1 if edge.edge_feature_source == "gnn_endpoint_mean_initialization" else 0)


def test_streaming_empty_and_failed_outputs_are_explicit(tmp_path):
    graph = graph_fixture(
        tmp_path,
        lines=["0,0"],
        dataset_id="stream-empty",
        virtual_edge_sigma=1.0,
    )
    learned = learned_features(graph, row_values={graph.sample_node_ids[0]: 0.25})
    input_path, result_path, result = _run_streamed(
        tmp_path,
        graph,
        learned,
        LinkPredictionConfig(max_iterations=1, convergence_epsilon=0.0),
    )
    assert result.manifest.outcome == "no_change"
    assert result.manifest.iterations == 0
    assert result.manifest.updated_edge_count == 0
    assert result.manifest.updated_graph_id == predict_edges(
        graph,
        learned,
        LinkPredictionConfig(max_iterations=1, convergence_epsilon=0.0),
    ).updated_graph.graph_id
    assert np.load(result_path / "updated_edge_endpoints.npy").shape == (0, 2)
    load_streaming_lp_result(result_path, input_directory=input_path)

    graph = graph_fixture(
        tmp_path,
        lines=["0,0 001122334455,-50", "1,0 001122334455,-60"],
        dataset_id="stream-failed",
        virtual_edge_sigma=1.0,
    )
    learned = learned_features(graph, row_values={node_id: 0.1 for node_id in graph.node_ids})
    failed_input = tmp_path / "failed-input"
    failed_result = tmp_path / "failed-result"
    save_streaming_lp_input(graph, learned, failed_input)
    failed = predict_edges_streaming(
        failed_input,
        failed_result,
        LinkPredictionConfig(max_iterations=1, convergence_epsilon=0.0),
    )
    expected = predict_edges(
        graph,
        learned,
        LinkPredictionConfig(max_iterations=1, convergence_epsilon=0.0),
    )
    assert failed.manifest.outcome == expected.outcome == "failed"
    assert failed.manifest.updated_graph_id is None
    assert failed.manifest.failure_reason == expected.failure_reason
    assert tuple(failed.iter_candidate_scores()) == ()
    assert load_streaming_lp_result(failed_result).manifest.result_id == failed.manifest.result_id


def test_streaming_artifacts_refuse_overwrite_and_detect_corruption(tmp_path):
    graph = graph_fixture(tmp_path, dataset_id="stream-integrity")
    learned = learned_features(graph)
    input_path, result_path, _ = _run_streamed(
        tmp_path,
        graph,
        learned,
        LinkPredictionConfig(max_iterations=1, convergence_epsilon=1e9),
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_streaming_lp_input(graph, learned, input_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        predict_edges_streaming(
            input_path,
            result_path,
            LinkPredictionConfig(max_iterations=1, convergence_epsilon=1e9),
        )

    manifest = json.loads((result_path / "manifest.json").read_text())
    assert manifest["input_id"] == load_streaming_lp_input(input_path)[0].input_id
    with (result_path / "updated_edge_weights.npy").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="missing or corrupt"):
        load_streaming_lp_result(result_path)


def test_streaming_input_rejects_feature_rows_that_changed_under_same_checkpoint(tmp_path):
    graph = graph_fixture(tmp_path, dataset_id="stream-feature-lineage")
    learned = learned_features(graph)
    input_path = tmp_path / "input"
    save_streaming_lp_input(graph, learned, input_path)
    changed_row = list(learned.node_feature_matrix[0])
    changed_row[0] += 0.25
    changed = learned.model_copy(
        update={
            "node_feature_matrix": (tuple(changed_row),) + learned.node_feature_matrix[1:]
        }
    )
    with pytest.raises(ValueError, match="node feature row changed"):
        validate_streaming_lp_input(graph, changed, input_path)


def test_streaming_lp_accepts_disk_ordered_explicit_candidates(tmp_path):
    graph = graph_fixture(
        tmp_path,
        lines=["0,0 001122334455,-50", "1,0"],
        dataset_id="stream-explicit-candidates",
        virtual_edge_sigma=-1.0,
    )
    learned = learned_features(
        graph,
        row_values={node_id: 0.2 for node_id in graph.node_ids},
    )
    input_path = tmp_path / "explicit-input"
    result_path = tmp_path / "explicit-result"
    save_streaming_lp_input(graph, learned, input_path)

    # Only request the missing link for sample 1. The existing link from
    # sample 0 is deliberately outside the candidate stream and must survive.
    requested = (graph.sample_node_ids[1], graph.ap_node_ids[0])
    result = predict_edges_streaming(
        input_path,
        result_path,
        LinkPredictionConfig(
            max_iterations=2,
            convergence_epsilon=1e9,
            link_threshold=0.0,
        ),
        candidate_pairs=iter((requested,)),
    )
    assert result.manifest.candidate_count == 1
    scores = tuple(result.iter_candidate_scores())
    assert len(scores) == 1
    assert (scores[0].sample_node_id, scores[0].ap_node_id) == requested
    assert tuple(result.iter_added_edges())[0].sample_node_id == requested[0]
    endpoints = np.load(result_path / "updated_edge_endpoints.npy")
    pairs = {
        (result.manifest.node_ids[int(sample_row)], result.manifest.node_ids[int(ap_row)])
        for sample_row, ap_row in endpoints
    }
    assert (graph.sample_node_ids[0], graph.ap_node_ids[0]) in pairs
    assert requested in pairs


def test_streaming_lp_explicit_candidates_require_sorted_unique_ids(tmp_path):
    graph = graph_fixture(
        tmp_path,
        lines=["0,0 001122334455,-50", "1,0 001122334455,-60"],
        dataset_id="stream-explicit-candidate-order",
        virtual_edge_sigma=-1.0,
    )
    learned = learned_features(graph)
    input_path = tmp_path / "ordered-input"
    save_streaming_lp_input(graph, learned, input_path)
    pairs = sorted(
        (sample_id, graph.ap_node_ids[0]) for sample_id in graph.sample_node_ids
    )
    with pytest.raises(ValueError, match="strictly sorted and unique"):
        predict_edges_streaming(
            input_path,
            tmp_path / "unordered-result",
            LinkPredictionConfig(max_iterations=1, convergence_epsilon=1e9),
            candidate_pairs=(pairs[1], pairs[0]),
        )
