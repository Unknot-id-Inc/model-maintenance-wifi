import json
from pathlib import Path

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from ae import EncoderFeatureArtifact
from graph_build import GraphBuildConfig, build_graph_state
from gufu_artifacts import canonical_json_digest
from gufu_data import PreprocessingConfig, prepare_dataset_files
from lp import LinkPredictionConfig
from lp_streaming import predict_edges_streaming, save_streaming_lp_input
from sage_ve import GNNFeatureArtifact, GNNModelConfig
from spatial_normalization import (
    fit_spatial_transform,
    fit_spatial_transform_from_gf_file,
    load_spatial_transform,
)
from streaming_update import build_streaming_online_graph, load_streaming_online_graph


GEOMETRY = np.asarray([(0.0, 0.0), (100.0, 0.0), (0.0, 10.0), (100.0, 10.0)])


def test_known_geometry_uses_one_shared_max_radius_scale():
    transform = fit_spatial_transform(GEOMETRY, "GF", coordinate_units="survey")
    assert transform.metadata.center == pytest.approx((50.0, 5.0))
    assert transform.metadata.scale == pytest.approx(np.sqrt(50**2 + 5**2))
    normalized = transform.transform(GEOMETRY)
    assert np.max(np.linalg.norm(normalized, axis=1)) == pytest.approx(1.0)
    assert normalized[1, 0] - normalized[0, 0] == pytest.approx(100 / transform.metadata.scale)
    assert normalized[2, 1] - normalized[0, 1] == pytest.approx(10 / transform.metadata.scale)


def test_duplicate_positions_do_not_change_fit_or_merge_scan_records(tmp_path):
    once = fit_spatial_transform([(0, 0), (10, 5)], "floor")
    repeated = fit_spatial_transform([(0, 0), (10, 5), (10, 5), (10, 5)], "floor")
    assert repeated.metadata.center == once.metadata.center
    assert repeated.metadata.scale == once.metadata.scale
    assert repeated.metadata.labeled_row_count == 4
    assert repeated.metadata.unique_position_count == 2

    prepared = _prepare(
        tmp_path,
        "duplicates.txt",
        ["0,0 001122334455,-50", "10,5 001122334455,-60", "10,5 001122334455,-70"],
    )
    graph = build_graph_state(prepared, _features(prepared), GraphBuildConfig(virtual_edge_sigma=1.0))
    assert len(graph.sample_node_ids) == 3
    assert len(set(graph.sample_node_ids)) == 3


def test_forward_inverse_round_trip_and_uniform_distance_scaling():
    transform = fit_spatial_transform(GEOMETRY, "GF")
    normalized = transform.transform(GEOMETRY)
    assert transform.inverse_transform(normalized) == pytest.approx(GEOMETRY)
    unit_x = transform.transform(np.asarray([[50.0, 5.0], [51.0, 5.0]]))
    unit_y = transform.transform(np.asarray([[50.0, 5.0], [50.0, 6.0]]))
    assert np.linalg.norm(unit_x[1] - unit_x[0]) == pytest.approx(1 / transform.metadata.scale)
    assert np.linalg.norm(unit_y[1] - unit_y[0]) == pytest.approx(1 / transform.metadata.scale)


def test_translation_and_positive_unit_scaling_preserve_normalized_geometry():
    reference = fit_spatial_transform(GEOMETRY, "GF")
    translated = fit_spatial_transform(GEOMETRY + np.asarray([500.0, -37.0]), "GF")
    converted = fit_spatial_transform(GEOMETRY * 0.3048, "GF")
    assert translated.transform(GEOMETRY + [500.0, -37.0]) == pytest.approx(
        reference.transform(GEOMETRY)
    )
    assert converted.transform(GEOMETRY * 0.3048) == pytest.approx(
        reference.transform(GEOMETRY)
    )


def test_persistence_preserves_transform_and_refuses_implicit_overwrite(tmp_path):
    transform = fit_spatial_transform(GEOMETRY, "GF", coordinate_units="unknown")
    path = tmp_path / "GF.json"
    transform.save(path)
    loaded = load_spatial_transform(path)
    assert loaded.metadata == transform.metadata
    assert loaded.transform(GEOMETRY) == pytest.approx(transform.transform(GEOMETRY))
    assert loaded.transform_sha256 == transform.transform_sha256
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        transform.save(path)
    transform.save(path, overwrite=True)


def test_loading_rejects_invalid_metadata_and_graph_rejects_wrong_site(tmp_path):
    transform = fit_spatial_transform(GEOMETRY, "GF")
    path = tmp_path / "bad.json"
    payload = transform.metadata.model_dump(mode="json")
    payload["scale"] = 0.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValidationError, match="strictly positive"):
        load_spatial_transform(path)

    good_path = tmp_path / "GF.json"
    transform.save(good_path)
    wrong_site = GraphBuildConfig(
        spatial_normalization_enabled=True,
        spatial_transform_path=str(good_path),
    )
    from graph_build import GraphConstructionError, resolve_spatial_transform

    with pytest.raises(GraphConstructionError, match="does not match"):
        resolve_spatial_transform(wrong_site, "another-site")


def test_frozen_graph_configuration_rejects_replaced_transform_file(tmp_path):
    path = tmp_path / "GF.json"
    fit_spatial_transform(GEOMETRY, "GF").save(path)
    config = GraphBuildConfig(
        spatial_normalization_enabled=True,
        spatial_transform_path=str(path),
    )
    from graph_build import resolve_spatial_transform

    frozen_config, _ = resolve_spatial_transform(config, "GF")
    fit_spatial_transform(GEOMETRY + 1.0, "GF").save(path, overwrite=True)
    with pytest.raises(ValueError, match="differs from the transform recorded"):
        resolve_spatial_transform(frozen_config, "GF")


def test_fitted_transform_is_frozen_and_does_not_clip_later_coordinates():
    transform = fit_spatial_transform(GEOMETRY, "GF")
    center = transform.metadata.center
    scale = transform.metadata.scale
    outside = transform.transform(np.asarray([[250.0, 5.0]]))
    assert outside[0, 0] > 1.0
    assert transform.metadata.center == center
    assert transform.metadata.scale == scale


@pytest.mark.parametrize(
    "coordinates, message",
    [
        ([], "empty labeled dataset"),
        ([1.0, 2.0], r"shape \(N, 2\)"),
        ([(0.0, 0.0), (float("nan"), 1.0)], "finite"),
        ([(3.0, 4.0), (3.0, 4.0)], "two unique fitting positions"),
    ],
)
def test_invalid_fit_inputs_are_rejected_clearly(coordinates, message):
    with pytest.raises(ValueError, match=message):
        fit_spatial_transform(coordinates, "GF")


def test_collinear_positions_and_nearby_distinct_positions_are_valid():
    transform = fit_spatial_transform([(0.0, 2.0), (1e-12, 2.0), (4.0, 2.0)], "line")
    assert transform.metadata.unique_position_count == 3
    assert transform.metadata.scale > 0.0
    assert transform.metadata.center[1] == 2.0


def test_tensor_batches_dtype_gradients_and_optional_gpu():
    transform = fit_spatial_transform(GEOMETRY, "GF")
    batch = torch.tensor(GEOMETRY.reshape(2, 2, 2), dtype=torch.float32, requires_grad=True)
    normalized = transform.transform(batch)
    assert normalized.shape == (2, 2, 2)
    assert normalized.device.type == "cpu"
    assert normalized.dtype == torch.float32
    normalized.square().sum().backward()
    assert batch.grad is not None
    assert torch.isfinite(batch.grad).all()
    assert transform.inverse_transform(normalized).shape == batch.shape
    fitted_from_tensor = fit_spatial_transform(
        torch.tensor(GEOMETRY, dtype=torch.float64), "GF"
    )
    assert fitted_from_tensor.metadata.center == transform.metadata.center

    if torch.cuda.is_available():
        gpu_transform = fit_spatial_transform(GEOMETRY, "GF").to("cuda")
        gpu_coordinates = torch.tensor(GEOMETRY, dtype=torch.float64, device="cuda")
        gpu_result = gpu_transform.transform(gpu_coordinates)
        assert gpu_result.device.type == "cuda"
        assert gpu_result.dtype == torch.float64
        assert gpu_result.shape == gpu_coordinates.shape


def test_gf_file_adapter_reads_only_leading_coordinate_token(tmp_path):
    source = tmp_path / "fingerprint0.txt"
    source.write_text(
        " 0,0\t001122334455,-55,extra,fields 001122334466,-70,other\n\n"
        "10.0,20.0   001122334455,-45,1,2\n"
        " 5,10 001122334477,-60\n",
        encoding="utf-8",
    )
    transform = fit_spatial_transform_from_gf_file(source, "GF")
    assert transform.metadata.labeled_row_count == 3
    assert transform.metadata.unique_position_count == 3
    assert transform.metadata.raw_coordinate_minima == (0.0, 0.0)
    assert transform.metadata.raw_coordinate_maxima == (10.0, 20.0)
    assert transform.metadata.provenance["source_sha256"]

    malformed = tmp_path / "bad.txt"
    malformed.write_text("0,1,2 001122334455,-50\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.txt:1: coordinate token"):
        fit_spatial_transform_from_gf_file(malformed, "GF")


def test_normalized_gnn_features_keep_signal_and_raw_labels_and_signal_edges(tmp_path):
    lines = [
        "0,0 001122334455,-50 001122334466,-70 001122334477,-80",
        "100,0 001122334455,-51 001122334466,-69 001122334477,-79",
        "0,10 001122334455,-80 001122334466,-50 001122334477,-65",
        "100,10 001122334455,-70 001122334477,-50 001122334466,-60",
    ]
    prepared = _prepare(tmp_path, "reference.txt", lines)
    encoder = _features(prepared, width=5)
    transform_path = tmp_path / "GF.json"
    transform = fit_spatial_transform(GEOMETRY, prepared.config.dataset_id)
    transform.save(transform_path)

    raw_graph = build_graph_state(
        prepared,
        encoder,
        GraphBuildConfig(virtual_edge_sigma=0.9, minimum_shared_aps=1),
    )
    normalized_graph = build_graph_state(
        prepared,
        encoder,
        GraphBuildConfig(
            virtual_edge_sigma=0.9,
            minimum_shared_aps=1,
            spatial_normalization_enabled=True,
            spatial_transform_path=str(transform_path),
            position_weight=0.75,
        ),
    )
    assert normalized_graph.node_feature_dim == encoder.feature_dim + 2
    assert normalized_graph.graph_feature_space == "z_encoder_concat_normalized_location_input"
    assert normalized_graph.build_config.spatial_transform_sha256 == transform.transform_sha256
    for row, scan_id in enumerate(normalized_graph.sample_node_ids):
        raw_feature = raw_graph.node_features[raw_graph.id_to_row[scan_id]]
        normalized_feature = normalized_graph.node_features[row]
        assert normalized_feature[: encoder.feature_dim] == raw_feature[: encoder.feature_dim]
        assert normalized_graph.reference_coordinate_targets[row] == raw_graph.reference_coordinate_targets[row]
        assert normalized_feature[-2:] == pytest.approx(
            tuple(value * 0.75 for value in transform.transform(raw_graph.reference_coordinate_targets[row]))
        )
    assert [
        (edge.sample_node_id_left, edge.sample_node_id_right, edge.cosine_similarity)
        for edge in normalized_graph.virtual_edges
    ] == [
        (edge.sample_node_id_left, edge.sample_node_id_right, edge.cosine_similarity)
        for edge in raw_graph.virtual_edges
    ]
    assert normalized_graph.negative_pairs == raw_graph.negative_pairs
    for ap_id in normalized_graph.ap_node_ids:
        row = normalized_graph.id_to_row[ap_id]
        incident = [edge for edge in normalized_graph.radio_edges if edge.ap_node_id == ap_id]
        weighted_xy = np.average(
            [normalized_graph.node_features[normalized_graph.id_to_row[e.sample_node_id]][-2:] for e in incident],
            axis=0,
            weights=[e.scalar_weight for e in incident],
        )
        assert normalized_graph.node_features[row][-2:] == pytest.approx(weighted_xy)


def test_unavailable_update_positions_use_zero_feature_block_and_keep_raw_metadata(tmp_path):
    reference = _prepare(
        tmp_path,
        "reference.txt",
        ["0,0 001122334455,-50", "10,10 001122334455,-60"],
    )
    transform_path = tmp_path / "site.json"
    fit_spatial_transform([(0, 0), (10, 10)], reference.config.dataset_id).save(transform_path)
    update = _prepare(
        tmp_path,
        "update.txt",
        ["900,700 001122334455,-55"],
        batch_role="update_batch",
    )
    graph = build_graph_state(
        update,
        _features(update, width=4),
        GraphBuildConfig(
            virtual_edge_sigma=1.0,
            spatial_normalization_enabled=True,
            spatial_transform_path=str(transform_path),
            pseudo_location_bounds=(0.0, 10.0, 0.0, 10.0),
        ),
    )
    assert graph.reference_label_mask == (False, False)
    assert graph.reference_coordinate_targets == (None, None)
    assert graph.node_features[0][-2:] == (0.0, 0.0)
    assert graph.sample_location_initializations[0].source == "seeded_pseudo_location"
    assert graph.sample_location_initializations[0].xy != (0.0, 0.0)


def test_streamed_update_graph_uses_same_transform_and_preserves_raw_initialization(tmp_path):
    reference_lines = [
        "0,0 001122334455,-50 001122334466,-70 001122334477,-80",
        "10,10 001122334455,-60 001122334466,-65 001122334477,-75",
    ]
    reference = _prepare(tmp_path, "stream-reference.txt", reference_lines)
    encoder = _features(reference, width=4)
    transform_path = tmp_path / "GF.json"
    transform = fit_spatial_transform([(0, 0), (10, 10)], "GF")
    transform.save(transform_path)
    graph = build_graph_state(
        reference,
        encoder,
        GraphBuildConfig(
            virtual_edge_sigma=0.9,
            minimum_shared_aps=1,
            spatial_normalization_enabled=True,
            spatial_transform_path=str(transform_path),
        ),
    )
    dimension = graph.node_feature_dim
    model_config = GNNModelConfig(
        graph_feature_dim=dimension,
        edge_feature_dim=dimension,
        node_output_dim=dimension,
        edge_output_dim=dimension,
        num_layers=1,
    )
    learned = GNNFeatureArtifact(
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
        checkpoint_sha256="b" * 64,
        model_config_record=model_config,
        node_ids=graph.node_ids,
        node_types=graph.node_types,
        node_id_to_row=graph.node_id_to_row,
        node_feature_dim=dimension,
        node_feature_matrix=graph.node_features,
        radio_edge_ids=tuple(edge.edge_id for edge in graph.radio_edges),
        radio_edge_feature_dim=dimension,
        radio_edge_feature_matrix=tuple(edge.edge_feature for edge in graph.radio_edges),
        virtual_edge_ids=tuple(edge.edge_id for edge in graph.virtual_edges),
        virtual_edge_feature_dim=dimension,
        virtual_edge_feature_matrix_initial=tuple(edge.edge_feature for edge in graph.virtual_edges),
        created_at_utc="2030-01-01T00:00:00+00:00",
    )
    lp_input = tmp_path / "lp-input"
    save_streaming_lp_input(graph, learned, lp_input)
    lp_result_path = tmp_path / "lp-result"
    result = predict_edges_streaming(
        lp_input,
        lp_result_path,
        LinkPredictionConfig(
            max_iterations=1,
            convergence_epsilon=1e9,
            link_threshold=0.0,
        ),
    )
    assert result.manifest.outcome != "failed"

    update = _prepare(
        tmp_path,
        "stream-update.txt",
        ["900,700 001122334455,-55 001122334466,-68 001122334477,-78"],
        batch_role="update_batch",
    )
    update_encoder = _features(update, width=4)
    online_path = tmp_path / "online-graph"
    build_streaming_online_graph(
        lp_input,
        lp_result_path,
        update,
        update_encoder,
        online_path,
        virtual_edge_sigma=0.9,
    )
    streamed = load_streaming_online_graph(online_path)
    update_id = update.scans[0].scan_id
    update_row = streamed.manifest.node_ids.index(update_id)
    assert tuple(streamed.node_features[update_row, -2:]) == (0.0, 0.0)
    sample_row = streamed.manifest.sample_node_ids.index(update_id)
    assert streamed.manifest.sample_position_available_mask[sample_row] is False
    initialization = streamed.manifest.sample_location_initializations[sample_row]
    assert initialization.source == "seeded_pseudo_location"
    assert initialization.xy != (0.0, 0.0)
    assert streamed.manifest.graph_build_config.spatial_transform_sha256 == transform.transform_sha256


def _prepare(tmp_path, filename, lines, *, batch_role="bootstrap_reference"):
    root = tmp_path / f"{filename}-root"
    root.mkdir(parents=True, exist_ok=True)
    source = root / filename
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return prepare_dataset_files(
        [source],
        PreprocessingConfig(
            dataset_id="GF",
            batch_id=filename.removesuffix(".txt"),
            batch_role=batch_role,
            normalization_mode="paper_ae",
        ),
        source_root=root,
    )


def _features(prepared, width=5):
    ids = tuple(scan.scan_id for scan in prepared.scans)
    features = tuple(
        tuple(float((row + 1) * (column + 1)) for column in range(width))
        for row in range(len(ids))
    )
    return EncoderFeatureArtifact(
        dataset_id=prepared.config.dataset_id,
        batch_id=prepared.config.batch_id,
        source_prepared_sha256=canonical_json_digest(prepared),
        scan_ids=ids,
        feature_matrix=features,
        feature_dim=width,
        checkpoint_sha256="a" * 64,
        bssid_vocabulary=prepared.bssid_vocabulary,
        vocabulary_sha256=canonical_json_digest(prepared.bssid_vocabulary),
        normalization=prepared.config.normalization_mode,
        dtype="float32",
        created_at_utc="2030-01-01T00:00:00+00:00",
    )
