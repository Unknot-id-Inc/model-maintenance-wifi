from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from ae import EncoderFeatureArtifact
from graph_build import (
    GraphBuildConfig,
    GraphConstructionError,
    build_graph_state,
    load_graph_state,
    save_graph_state,
)
from gufu_artifacts import canonical_json_digest
from gufu_data import PreprocessingConfig, prepare_dataset_files
from util import get_data


def prepare(tmp_path, filename, lines, *, batch_role="bootstrap_reference", base_vocabulary=()):
    root = tmp_path / filename.replace(".txt", "-root")
    root.mkdir(parents=True, exist_ok=True)
    source = root / filename
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    batch_id = filename.removesuffix(".txt")
    config = PreprocessingConfig(
        dataset_id="site-floor",
        batch_id=batch_id,
        batch_role=batch_role,
        base_bssid_vocabulary=tuple(base_vocabulary),
        normalization_mode="paper_ae",
    )
    return prepare_dataset_files([source], config, source_root=root)


def features_for(prepared, by_scan_id, *, ordered_ids=None):
    scan_ids = tuple(ordered_ids or (scan.scan_id for scan in prepared.scans))
    return EncoderFeatureArtifact(
        dataset_id=prepared.config.dataset_id,
        batch_id=prepared.config.batch_id,
        source_prepared_sha256=canonical_json_digest(prepared),
        scan_ids=scan_ids,
        feature_matrix=tuple(tuple(by_scan_id[scan_id]) for scan_id in scan_ids),
        checkpoint_sha256="a" * 64,
        bssid_vocabulary=prepared.bssid_vocabulary,
        vocabulary_sha256=canonical_json_digest(prepared.bssid_vocabulary),
        normalization=prepared.config.normalization_mode,
        dtype="float32",
        created_at_utc="2030-01-01T00:00:00+00:00",
    )


def vector(index, *, value=1.0):
    result = [0.0] * 32
    result[index] = value
    return tuple(result)


def test_ap_initialization_uses_hand_checked_rss_weighted_augmented_features(tmp_path):
    ap_a = "001122334455"
    ap_b = "AABBCCDDEE11"
    prepared = prepare(
        tmp_path,
        "bootstrap.txt",
        [
            f"0,0 {ap_a},-60 {ap_b},-80",
            f"10,20 {ap_a},-90",
        ],
    )
    first, second = (scan.scan_id for scan in prepared.scans)
    latent_by_id = {first: vector(0), second: vector(1)}
    feature_artifact = features_for(prepared, latent_by_id)
    graph = build_graph_state(
        prepared,
        feature_artifact,
        GraphBuildConfig(virtual_edge_sigma=1.0),
    )

    rows = graph.id_to_row
    ap_a_id = f"ap:{ap_a}"
    ap_b_id = f"ap:{ap_b}"
    assert graph.ap_node_ids == (ap_a_id, ap_b_id)
    assert graph.node_feature_tensor().shape == (4, 34)
    assert graph.node_features[rows[ap_a_id]][0] == pytest.approx(2.0 / 3.0)
    assert graph.node_features[rows[ap_a_id]][1] == pytest.approx(1.0 / 3.0)
    assert graph.node_features[rows[ap_a_id]][32:] == pytest.approx((10.0 / 3.0, 20.0 / 3.0))
    assert graph.node_features[rows[ap_b_id]] == pytest.approx(
        graph.node_features[rows[first]]
    )
    edge_by_pair = {(edge.sample_node_id, edge.ap_node_id): edge for edge in graph.radio_edges}
    assert all(
        edge.sample_node_id in graph.sample_node_ids
        and edge.ap_node_id in graph.ap_node_ids
        for edge in graph.radio_edges
    )
    assert edge_by_pair[(first, ap_a_id)].scalar_weight == 60.0
    assert edge_by_pair[(second, ap_a_id)].scalar_weight == 30.0
    assert edge_by_pair[(first, ap_b_id)].scalar_weight == 40.0
    assert edge_by_pair[(first, ap_a_id)].edge_feature == pytest.approx(
        tuple((a + b) / 2.0 for a, b in zip(graph.node_features[rows[first]], graph.node_features[rows[ap_a_id]]))
    )
    assert graph.reference_sample_mask == (True, True, False, False)
    assert graph.new_sample_mask == (False, False, False, False)
    assert graph.reference_label_mask == (True, True, False, False)
    assert graph.reference_coordinate_targets[rows[first]] == (0.0, 0.0)


def test_shuffled_encoder_rows_join_by_scan_id_and_graph_rows_are_not_encoder_rows(tmp_path):
    prepared = prepare(
        tmp_path,
        "shuffle.txt",
        ["3,4 001122334455,-50", "8,9 001122334455,-60"],
    )
    first, second = (scan.scan_id for scan in prepared.scans)
    latent_by_id = {first: vector(2), second: vector(7)}
    feature_artifact = features_for(
        prepared, latent_by_id, ordered_ids=(second, first)
    )
    graph = build_graph_state(
        prepared, feature_artifact, GraphBuildConfig(virtual_edge_sigma=1.0)
    )

    assert graph.sample_node_ids == (first, second)
    assert graph.node_features[graph.id_to_row[first]][2] == 1.0
    assert graph.node_features[graph.id_to_row[second]][7] == 1.0
    assert graph.id_to_row[first] == 0
    assert graph.id_to_row[second] == 1
    assert first != "0" and second != "1"


def test_virtual_edges_use_signal_only_cosine_and_omit_zero_norm_pairs(tmp_path):
    prepared = prepare(
        tmp_path,
        "cosine.txt",
        [
            "0,0 001122334455,-60",
            "100,-100 001122334455,-60",
            "200,0 001122334455,-60",
            "0,200 001122334455,-60",
        ],
    )
    ids = tuple(scan.scan_id for scan in prepared.scans)
    assert len(ids) == len(set(ids)) == 4  # equal XY must not merge physical scans
    latent_by_id = {
        ids[0]: vector(0),
        ids[1]: vector(0),
        ids[2]: (0.0,) * 32,
        ids[3]: (0.0,) * 32,
    }
    graph = build_graph_state(
        prepared,
        features_for(prepared, latent_by_id),
        GraphBuildConfig(virtual_edge_sigma=0.95),
    )

    assert len(graph.virtual_edges) == 1
    edge = graph.virtual_edges[0]
    assert (edge.sample_node_id_left, edge.sample_node_id_right) == tuple(sorted(ids[:2]))
    assert edge.cosine_similarity == pytest.approx(1.0)
    rows = graph.id_to_row
    assert graph.node_features[rows[ids[0]]][32:] == pytest.approx((0.0, 0.0))
    assert graph.node_features[rows[ids[1]]][32:] == pytest.approx((100.0, -100.0))
    assert edge.edge_feature[32:] == pytest.approx((50.0, -50.0))
    assert graph.build_config.similarity_feature_space == "z_encoder_signal_only_v1"
    assert edge.scalar_weight is None
    assert edge.temporary is True
    assert all(edge.sample_node_id_left != edge.sample_node_id_right for edge in graph.virtual_edges)
    assert graph.edge_index_tensor("virtual").shape == (2, 2)
    exact_boundary = build_graph_state(
        prepared,
        features_for(prepared, latent_by_id),
        GraphBuildConfig(virtual_edge_sigma=1.0),
    )
    assert exact_boundary.virtual_edges == ()  # 1.0 is not strictly greater than sigma=1.0


def test_old_augmented_similarity_config_is_rejected():
    with pytest.raises(ValidationError):
        GraphBuildConfig(similarity_feature_space="z_graph_input_augmented")


def test_update_samples_get_seeded_pseudo_xy_without_reading_eval_truth(tmp_path):
    lines_a = ["100,200 001122334455,-60", "300,400 001122334455,-70"]
    prepared_a = prepare(
        tmp_path,
        "update.txt",
        lines_a,
        batch_role="update_batch",
    )
    ids_a = tuple(scan.scan_id for scan in prepared_a.scans)
    latents_a = {scan_id: vector(i) for i, scan_id in enumerate(ids_a)}
    graph_a = build_graph_state(
        prepared_a,
        features_for(prepared_a, latents_a),
        GraphBuildConfig(
            pseudo_location_seed=29,
            pseudo_location_bounds=(0.0, 500.0, 0.0, 500.0),
            virtual_edge_sigma=1.0,
        ),
    )
    assert graph_a.reference_sample_mask == (False, False, False)
    assert graph_a.new_sample_mask == (True, True, False)
    assert graph_a.reference_label_mask == (False, False, False)
    assert graph_a.reference_coordinate_targets == (None, None, None)
    for initialization in graph_a.sample_location_initializations:
        assert initialization.source == "seeded_pseudo_location"
        assert 0.0 <= initialization.xy[0] < 500.0
        assert 0.0 <= initialization.xy[1] < 500.0
    assert graph_a.node_features[0][32:] != (100.0, 200.0)
    assert graph_a.build_config.pseudo_location_policy == "sha256_uniform_configured_bounds_v1"
    assert graph_a.build_config.pseudo_location_bounds == (0.0, 500.0, 0.0, 500.0)
    assert graph_a.build_config.pseudo_location_seed == 29

    # Same source rows/batch, changed evaluation-only coordinates. The new IDs
    # and corresponding feature artifact are rebuilt, but pseudo locations stay
    # the same because the policy hashes source identity fields, not row content.
    other_root = tmp_path / "other-update-root"
    other_root.mkdir()
    other_source = other_root / "update.txt"
    other_source.write_text(
        "-10,-20 001122334455,-60\n-30,-40 001122334455,-70\n",
        encoding="utf-8",
    )
    prepared_b = prepare_dataset_at_root(other_source, other_root, "update")
    ids_b = tuple(scan.scan_id for scan in prepared_b.scans)
    graph_b = build_graph_state(
        prepared_b,
        features_for(prepared_b, {scan_id: vector(i) for i, scan_id in enumerate(ids_b)}),
        GraphBuildConfig(
            pseudo_location_seed=29,
            pseudo_location_bounds=(0.0, 500.0, 0.0, 500.0),
            virtual_edge_sigma=1.0,
        ),
    )
    assert tuple(item.xy for item in graph_a.sample_location_initializations) == tuple(
        item.xy for item in graph_b.sample_location_initializations
    )


def test_update_pseudo_locations_require_explicit_bounds_not_eval_truth(tmp_path):
    prepared = prepare(
        tmp_path,
        "unbounded-update.txt",
        ["100,200 001122334455,-60"],
        batch_role="update_batch",
    )
    scan_id = prepared.scans[0].scan_id
    with pytest.raises(GraphConstructionError, match="require pseudo_location_bounds"):
        build_graph_state(
            prepared,
            features_for(prepared, {scan_id: vector(0)}),
            GraphBuildConfig(),
        )


def prepare_dataset_at_root(source: Path, root: Path, batch_id: str):
    return prepare_dataset_files(
        [source],
        PreprocessingConfig(
            dataset_id="site-floor",
            batch_id=batch_id,
            batch_role="update_batch",
            normalization_mode="paper_ae",
        ),
        source_root=root,
    )


def test_top_k_is_explicit_alternative_and_clamps_to_available_neighbors(tmp_path):
    prepared = prepare(
        tmp_path,
        "few.txt",
        ["0,0 001122334455,-60", "0,0 001122334455,-60"],
    )
    ids = tuple(scan.scan_id for scan in prepared.scans)
    graph = build_graph_state(
        prepared,
        features_for(prepared, {ids[0]: vector(4), ids[1]: vector(4)}),
        GraphBuildConfig(virtual_edge_strategy="top_k_alternative", top_k=10),
    )

    assert len(graph.virtual_edges) == 1
    assert graph.virtual_edges[0].cosine_similarity == pytest.approx(1.0)
    assert graph.build_config.virtual_edge_strategy == "top_k_alternative"
    with pytest.raises(ValueError, match="requires an explicit positive top_k"):
        GraphBuildConfig(virtual_edge_strategy="top_k_alternative")


def test_empty_scans_remain_isolated_and_unobserved_vocabulary_aps_are_recorded(tmp_path):
    unused_ap = "AABBCCDDEE11"
    prepared = prepare(
        tmp_path,
        "isolated.txt",
        ["0,0", "1,1"],
        base_vocabulary=(unused_ap,),
    )
    ids = tuple(scan.scan_id for scan in prepared.scans)
    graph = build_graph_state(
        prepared,
        features_for(prepared, {ids[0]: vector(0), ids[1]: vector(1)}),
        GraphBuildConfig(virtual_edge_sigma=1.0),
    )

    assert graph.node_ids == ids
    assert graph.ap_node_ids == ()
    assert graph.omitted_unobserved_ap_ids == (f"ap:{unused_ap}",)
    assert graph.radio_edges == ()
    assert graph.edge_index_tensor("radio").shape == (2, 0)
    assert graph.node_feature_tensor().shape == (2, 34)


def test_graph_save_reload_preserves_node_and_edge_identity(tmp_path):
    prepared = prepare(
        tmp_path,
        "roundtrip.txt",
        ["0,0 001122334455,-60", "2,3 001122334455,-70"],
    )
    ids = tuple(scan.scan_id for scan in prepared.scans)
    graph = build_graph_state(
        prepared,
        features_for(prepared, {ids[0]: vector(1), ids[1]: vector(1)}),
        GraphBuildConfig(virtual_edge_sigma=0.95),
    )
    artifact_path = save_graph_state(graph, tmp_path / "graph_state.json")
    restored = load_graph_state(artifact_path)

    assert restored == graph
    assert restored.graph_id == graph.graph_id
    assert restored.node_ids == graph.node_ids
    assert restored.node_id_to_row == graph.node_id_to_row
    assert restored.radio_edges == graph.radio_edges
    assert restored.virtual_edges == graph.virtual_edges
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_graph_state(graph, artifact_path)

    corrupted = graph.model_dump(mode="python")
    corrupted["node_ids"] = (graph.node_ids[0], graph.node_ids[0], *graph.node_ids[2:])
    with pytest.raises(ValidationError, match="graph node IDs must be unique"):
        type(graph).model_validate(corrupted)


def test_encoder_feature_artifact_rejects_duplicate_scan_ids(tmp_path):
    prepared = prepare(
        tmp_path,
        "duplicate-feature-ids.txt",
        ["0,0 001122334455,-60", "1,1 001122334455,-70"],
    )
    ids = tuple(scan.scan_id for scan in prepared.scans)
    with pytest.raises(ValidationError, match="encoder feature scan IDs must be unique"):
        features_for(
            prepared,
            {ids[0]: vector(0), ids[1]: vector(1)},
            ordered_ids=(ids[0], ids[0]),
        )


def test_an_observed_zero_weight_ap_is_rejected_instead_of_given_placeholder_features(tmp_path):
    prepared = prepare(tmp_path, "zero-weight.txt", ["0,0 001122334455,-120"])
    scan_id = prepared.scans[0].scan_id

    with pytest.raises(GraphConstructionError, match=r"total RSS\+120 weight is zero"):
        build_graph_state(
            prepared,
            features_for(prepared, {scan_id: vector(0)}),
            GraphBuildConfig(virtual_edge_sigma=1.0),
        )


def test_get_data_is_a_typed_wrapper_for_the_shared_graph_builder(tmp_path):
    prepared = prepare(tmp_path, "get-data.txt", ["0,0 001122334455,-60"])
    scan_id = prepared.scans[0].scan_id
    artifact = features_for(prepared, {scan_id: vector(3)})
    graph = get_data(prepared, artifact, config=GraphBuildConfig(virtual_edge_sigma=1.0))

    assert graph.sample_node_ids == (scan_id,)
    assert graph.node_feature_tensor().shape == (2, 34)
    with pytest.raises(TypeError, match="requires a PreparedDataset"):
        get_data("legacy.edgelist", artifact)
