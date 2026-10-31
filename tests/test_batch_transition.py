from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from ae import (
    AETrainingConfig,
    encode_prepared_dataset,
    load_encoder_features,
    migrate_ae_checkpoint_vocabulary,
    train_autoencoder,
)
from batch_transition import (
    APGraphModification,
    NoSharedAPError,
    TransitionConfig,
    TransitionEncoderFeatureArtifact,
    TransitionGNNFeatureArtifact,
    TransitionRadioEdge,
    VocabularyTransitionState,
    apply_batch_transition,
    bootstrap_transition_state,
    classify_and_plan_ap_vocabulary,
    infer_ap_graph_changes,
    project_prepared_batch_to_vocabulary,
    remap_node_rows_by_id,
    remap_rssi_columns_by_bssid,
    validate_transition_artifacts,
)
from fingerprint_update import (
    FingerprintUpdateConfig,
    load_fingerprint_update_checkpoint,
    prepare_fingerprint_update,
    train_fingerprint_update,
)
from graph_build import GraphBuildConfig, build_graph_state
from gufu_artifacts import canonical_json_digest
from gufu_data import PreprocessingConfig, PreparedDataset, prepare_dataset_files
from lp import LinkEdgeChange, LinkPredictionConfig
from sage_ve import (
    EdgeFeatureSAGE,
    GNNModelConfig,
    GNNTrainingConfig,
    load_gnn_features,
    train_gnn,
)
from spatial_normalization import fit_spatial_transform


AP_A = "001122334455"
AP_B = "001122334466"
AP_C = "AA1122334455"


@dataclass(frozen=True)
class TransitionFixture:
    root: object
    reference: PreparedDataset
    full_batch: PreparedDataset
    projection: PreparedDataset
    ae_checkpoint: object
    graph: object
    gnn_features: object
    context: object
    updater_models: object
    updated_fingerprints: object
    state: VocabularyTransitionState
    inference: object


@pytest.fixture(scope="module")
def transition_fixture(tmp_path_factory) -> TransitionFixture:
    root = tmp_path_factory.mktemp("batch-transition")
    (root / "reference.txt").write_text(
        "0,0 001122334455,-50 001122334466,-67\n"
        "10,10 001122334455,-53 001122334466,-64\n"
        "20,20 001122334455,-58 001122334466,-61\n",
        encoding="utf-8",
    )
    (root / "update.txt").write_text(
        "8,8 001122334455,-51 001122334466,-66 AA1122334455,-45\n"
        "15,15 001122334455,-57 AA1122334455,-48\n",
        encoding="utf-8",
    )
    reference = prepare_dataset_files(
        [root / "reference.txt"],
        PreprocessingConfig(
            dataset_id="transition-test",
            batch_id="bootstrap",
            batch_role="bootstrap_reference",
        ),
        source_root=root,
    )
    full_batch = prepare_dataset_files(
        [root / "update.txt"],
        PreprocessingConfig(
            dataset_id="transition-test",
            batch_id="batch-one",
            batch_role="update_batch",
            base_bssid_vocabulary=reference.bssid_vocabulary,
        ),
        source_root=root,
    )
    projection = project_prepared_batch_to_vocabulary(
        full_batch, reference.bssid_vocabulary
    )
    ae_path = root / "base-ae.pt"
    train_autoencoder(
        reference,
        ae_path,
        AETrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            seed=11,
            shuffle=False,
        ),
    )
    reference_encoder = encode_prepared_dataset(
        reference, ae_path, root / "reference-encoder.json", batch_size=2
    )
    update_encoder = encode_prepared_dataset(
        projection, ae_path, root / "update-encoder.json", batch_size=1
    )
    spatial_path = root / "spatial" / "transition-test.json"
    fit_spatial_transform(
        [(position.x, position.y) for position in reference.training_reference_positions()],
        reference.config.dataset_id,
        provenance={"source": "test bootstrap training positions"},
    ).save(spatial_path)
    graph_config = GraphBuildConfig(
        virtual_edge_strategy="paper_cosine_threshold",
        virtual_edge_sigma=0.995,
        negative_similarity_threshold=0.994,
        minimum_shared_aps=1,
        pseudo_location_seed=9,
        spatial_normalization_enabled=True,
        spatial_transform_path=str(spatial_path),
    )
    graph = build_graph_state(reference, reference_encoder, graph_config)
    gnn_path = root / "base-gnn.pt"
    train_gnn(
        graph,
        EdgeFeatureSAGE(
            GNNModelConfig(
                num_layers=2,
                dropout_probability=0.0,
                initialization_seed=5,
            )
        ),
        gnn_path,
        root / "reference-gnn.json",
        training_config=GNNTrainingConfig(
            epochs=1,
            learning_rate=0.002,
            seed=13,
            objective_epsilon=1e-8,
        ),
    )
    gnn_features = load_gnn_features(root / "reference-gnn.json")
    state = bootstrap_transition_state(reference, graph, gnn_features, ae_path)
    context = prepare_fingerprint_update(
        reference_prepared=reference,
        reference_graph=graph,
        reference_encoder_features=reference_encoder,
        reference_gnn_features=gnn_features,
        update_prepared=projection,
        update_encoder_features=update_encoder,
        ae_checkpoint_path=ae_path,
        gnn_checkpoint_path=gnn_path,
        config=FingerprintUpdateConfig(
            hidden_dim=8,
            virtual_edge_sigma=0.995,
            gnn_epochs=1,
            updater_epochs=2,
            ae_epochs=1,
            gnn_learning_rate=0.001,
            updater_learning_rate=0.01,
            ae_learning_rate=0.0001,
            seed=17,
        ),
    )
    update_model_path = root / "fingerprint-update.pt"
    update_fingerprint_path = root / "fingerprint-update.json"
    trained = train_fingerprint_update(
        context, update_model_path, update_fingerprint_path
    )
    updater_models = load_fingerprint_update_checkpoint(update_model_path, context)
    inference = infer_ap_graph_changes(
        state,
        context,
        updater_models,
        full_batch,
        TransitionConfig(
            ae_finetune_epochs=1,
            link_prediction=LinkPredictionConfig(
                max_iterations=4,
                convergence_epsilon=1e9,
                link_threshold=0.0,
            ),
        ),
    )
    return TransitionFixture(
        root,
        reference,
        full_batch,
        projection,
        ae_path,
        graph,
        gnn_features,
        context,
        updater_models,
        trained.artifact,
        state,
        inference,
    )


def _replace_state(state: VocabularyTransitionState, **updates) -> VocabularyTransitionState:
    values = state.model_dump(mode="python")
    values.update(updates)
    values["graph_id"] = "graph:" + canonical_json_digest(
        {
            "dataset_id": values["dataset_id"],
            "reference_scan_ids": values["reference_scan_ids"],
            "bssid_vocabulary": values["bssid_vocabulary"],
            "node_ids": values["node_ids"],
            "node_types": values["node_types"],
            "graph_input_node_features": values["graph_input_node_features"],
            "graph_node_features": values["graph_node_features"],
            "radio_edges": [
                edge.model_dump(mode="json") if hasattr(edge, "model_dump") else edge
                for edge in values["radio_edges"]
            ],
            "reference_graph_id": values["reference_graph_id"],
        }
    )
    values["state_id"] = "transition:" + canonical_json_digest(
        {
            key: value.model_dump(mode="json") if hasattr(value, "model_dump") else value
            for key, value in values.items()
            if key not in {"state_id", "format"}
        }
    )
    return VocabularyTransitionState.model_validate(values)


def test_old_encoder_projection_preserves_raw_new_ap_data_and_missing_masks(transition_fixture):
    fixture = transition_fixture
    projected = fixture.projection
    assert projected.bssid_vocabulary == fixture.reference.bssid_vocabulary
    assert tuple(scan.scan_id for scan in projected.scans) == tuple(
        scan.scan_id for scan in fixture.full_batch.scans
    )
    assert AP_C in fixture.full_batch.bssid_vocabulary
    assert AP_C not in projected.bssid_vocabulary
    assert all(len(scan.rss_dbm) == len(fixture.reference.bssid_vocabulary) for scan in projected.scans)
    assert projected.scans[1].observation_mask[1] is False
    assert projected.scans[1].rss_dbm[1] == -120.0
    assert projected.scans[0].rss_dbm[
        projected.bssid_vocabulary.index(AP_A)
    ] == fixture.full_batch.scans[0].rss_dbm[
        fixture.full_batch.bssid_vocabulary.index(AP_A)
    ]
    assert fixture.full_batch.scans[0].rss_dbm[
        fixture.full_batch.bssid_vocabulary.index(AP_C)
    ] == -45.0
    with pytest.raises(NoSharedAPError, match="zero APs in common"):
        project_prepared_batch_to_vocabulary(fixture.full_batch, ("FF1122334455",))


def test_identity_remaps_preserve_columns_and_graph_rows_after_reordering():
    values, masks, origins = remap_rssi_columns_by_bssid(
        ((-40.0, -80.0),),
        ((True, False),),
        (("measured", "missing"),),
        (AP_A, AP_B),
        (AP_B, AP_A, AP_C),
    )
    assert values == ((-80.0, -40.0, -120.0),)
    assert masks == ((False, True, False),)
    assert origins == (("missing", "measured", "missing"),)
    assert remap_node_rows_by_id(
        ("sample:b", "sample:a"), ((2.0,), (1.0,)), ("sample:a", "sample:b")
    ) == ((1.0,), (2.0,))
    with pytest.raises(ValueError, match="no learned graph feature"):
        remap_node_rows_by_id(("sample:a",), ((1.0,),), ("sample:b",))


def test_ap_classification_separates_missing_new_returning_and_forgetting(transition_fixture):
    fixture = transition_fixture
    new_edge = TransitionRadioEdge(
        edge_id=f"radio:{fixture.state.reference_scan_ids[0]}|ap:{AP_C}",
        sample_node_id=fixture.state.reference_scan_ids[0],
        bssid=AP_C,
        scalar_weight=72.0,
        edge_feature=(0.25,) * 34,
        source="new_ap_endpoint_mean",
    )
    add_change = LinkEdgeChange(
        edge_id=new_edge.edge_id,
        sample_node_id=new_edge.sample_node_id,
        ap_node_id=f"ap:{AP_C}",
        old_weight=None,
        new_weight=72.0,
        predicted_weight=72.0,
    )
    modification = APGraphModification(
        online_graph_id=fixture.inference.node_features.online_graph_id,
        source_state_id=fixture.state.state_id,
        input_batch_id=fixture.full_batch.config.batch_id,
        source_model_checkpoint_sha256=fixture.updater_models.checkpoint_sha256,
        source_vocabulary_sha256=fixture.state.vocabulary_sha256,
        config=LinkPredictionConfig(link_threshold=0.0),
        candidate_scores=(),
        added_reference_edges=(add_change,),
        removed_reference_edges=(),
        removed_ap_bssids=(),
        final_reference_edges=fixture.state.radio_edges + (new_edge,),
    )
    plan = classify_and_plan_ap_vocabulary(fixture.state, fixture.full_batch, modification)
    assert set(plan.shared_bssids) == {AP_A, AP_B}
    assert plan.newly_observed_bssids == (AP_C,)
    assert plan.missing_from_batch_bssids == ()
    assert plan.vocabulary_after == fixture.state.bssid_vocabulary + (AP_C,)

    shared_only_plan = classify_and_plan_ap_vocabulary(
        fixture.state,
        fixture.projection,
        modification.model_copy(update={"added_reference_edges": (), "final_reference_edges": fixture.state.radio_edges}),
    )
    assert shared_only_plan.newly_observed_bssids == ()
    assert shared_only_plan.vocabulary_after == fixture.state.bssid_vocabulary

    bssid_b_edges = tuple(edge for edge in fixture.state.radio_edges if edge.bssid == AP_B)
    retained = tuple(edge for edge in fixture.state.radio_edges if edge.bssid != AP_B)
    (fixture.root / "disappearing.txt").write_text(
        "1,1 001122334455,-54\n", encoding="utf-8"
    )
    disappearing = prepare_dataset_files(
        [fixture.root / "disappearing.txt"],
        PreprocessingConfig(
            dataset_id="transition-test",
            batch_id="batch-disappearing",
            batch_role="update_batch",
            base_bssid_vocabulary=fixture.state.bssid_vocabulary,
        ),
        source_root=fixture.root,
    )
    remove_b = APGraphModification(
        online_graph_id=fixture.inference.node_features.online_graph_id,
        source_state_id=fixture.state.state_id,
        input_batch_id=disappearing.config.batch_id,
        source_model_checkpoint_sha256=fixture.updater_models.checkpoint_sha256,
        source_vocabulary_sha256=fixture.state.vocabulary_sha256,
        config=LinkPredictionConfig(link_threshold=0.0),
        candidate_scores=(),
        added_reference_edges=(),
        removed_reference_edges=(),
        removed_ap_bssids=(AP_B,),
        final_reference_edges=retained,
    )
    # AP B's absence is observed separately from the actual forgetting delta.
    missing_plan = classify_and_plan_ap_vocabulary(fixture.state, disappearing, remove_b)
    assert missing_plan.missing_from_batch_bssids == (AP_B,)
    assert missing_plan.removed_by_forgetting_bssids == (AP_B,)
    assert bssid_b_edges
    missing_only = remove_b.model_copy(
        update={"removed_ap_bssids": (), "final_reference_edges": fixture.state.radio_edges}
    )
    missing_without_forgetting = classify_and_plan_ap_vocabulary(
        fixture.state, disappearing, missing_only
    )
    assert missing_without_forgetting.missing_from_batch_bssids == (AP_B,)
    assert missing_without_forgetting.removed_by_forgetting_bssids == ()
    assert missing_without_forgetting.vocabulary_after == fixture.state.bssid_vocabulary


def test_transition_runs_inference_migrates_and_reexports_aligned_artifacts(transition_fixture, tmp_path):
    fixture = transition_fixture
    destination = tmp_path / "transition-v1"
    sentinel = tmp_path / "keep-user-data.txt"
    sentinel.write_text("keep me\n", encoding="utf-8")
    result = apply_batch_transition(
        fixture.state,
        fixture.full_batch,
        fixture.updated_fingerprints,
        fixture.inference,
        fixture.ae_checkpoint,
        destination,
        TransitionConfig(
            ae_finetune_epochs=1,
            link_prediction=LinkPredictionConfig(
                max_iterations=4,
                convergence_epsilon=1e9,
                link_threshold=0.0,
            ),
        ),
        updater_models=fixture.updater_models,
    )
    assert result.state.reference_scan_ids == tuple(scan.scan_id for scan in fixture.reference.scans)
    assert result.state.reference_locations == fixture.state.reference_locations
    assert result.state.batch_history[-1].scan_ids == tuple(scan.scan_id for scan in fixture.full_batch.scans)
    assert result.state.bssid_vocabulary == result.fingerprints.bssid_vocabulary
    assert AP_C in result.state.bssid_vocabulary
    assert any(row.bssid == AP_C for row in result.state.ap_registry)
    assert result.state.encoder_cache_status == "valid"
    assert result.state.gnn_cache_status == "valid"
    assert (destination / "prepared_batch.json").is_file()
    assert (destination / "updater_models.pt").is_file()
    assert result.manifest.updater_checkpoint_sha256 == fixture.updater_models.checkpoint_sha256
    assert (destination / "encoder_features.json").is_file()
    assert (destination / "gnn_features.json").is_file()
    assert sentinel.read_text(encoding="utf-8") == "keep me\n"
    assert result.migrated_ae_checkpoint.manifest.model.input_dim == len(result.state.bssid_vocabulary)
    assert result.migrated_ae_checkpoint.manifest.vocabulary_migration.finetune_epochs == 1
    assert all(len(row.updated_rssi_dbm) == len(result.state.bssid_vocabulary) for row in result.fingerprints.rows)
    old_features = load_encoder_features(fixture.root / "reference-encoder.json")
    with pytest.raises(ValueError, match="incompatible vocabulary|incompatible AE checkpoint"):
        validate_transition_artifacts(result.state, encoder_features=old_features)
    current_encoder = TransitionEncoderFeatureArtifact.model_validate_json(
        (destination / "encoder_features.json").read_text(encoding="utf-8")
    )
    current_gnn = TransitionGNNFeatureArtifact.model_validate_json(
        (destination / "gnn_features.json").read_text(encoding="utf-8")
    )
    validate_transition_artifacts(
        result.state,
        encoder_features=current_encoder,
        gnn_features=current_gnn,
    )
    reordered_gnn = current_gnn.model_copy(update={"node_ids": tuple(reversed(current_gnn.node_ids))})
    with pytest.raises(ValueError, match="incompatible node IDs/order"):
        validate_transition_artifacts(result.state, gnn_features=reordered_gnn)
    if len(current_gnn.radio_edge_ids) > 1:
        reordered_edges = current_gnn.model_copy(
            update={"radio_edge_ids": tuple(reversed(current_gnn.radio_edge_ids))}
        )
        with pytest.raises(ValueError, match="incompatible radio edge IDs/order"):
            validate_transition_artifacts(result.state, gnn_features=reordered_edges)
    with pytest.raises(ValueError, match="batch ID was already used"):
        apply_batch_transition(
            result.state,
            fixture.full_batch,
            fixture.updated_fingerprints,
            fixture.inference,
            fixture.ae_checkpoint,
            tmp_path / "duplicate-batch",
            TransitionConfig(
                ae_finetune_epochs=1,
                link_prediction=fixture.inference.modification.config,
            ),
            updater_models=fixture.updater_models,
        )


def test_ae_vocabulary_migration_restores_returning_ap_and_supports_successive_orders(transition_fixture, tmp_path):
    fixture = transition_fixture
    original = load_fingerprint_update_checkpoint(
        fixture.updater_models.checkpoint_path, fixture.context
    )
    from ae import load_ae_checkpoint

    baseline = load_ae_checkpoint(fixture.ae_checkpoint)
    original_encoder_slice = baseline.model.E[0].weight[:, fixture.reference.bssid_vocabulary.index(AP_B)].detach().clone()
    dropped_path = tmp_path / "vocab-ap-b-dropped.pt"
    dropped = migrate_ae_checkpoint_vocabulary(
        fixture.ae_checkpoint,
        (AP_A,),
        dropped_path,
        seed=21,
    )
    assert AP_B in dropped.ap_parameter_archive
    returned_path = tmp_path / "vocab-ap-b-returned.pt"
    returned = migrate_ae_checkpoint_vocabulary(
        dropped_path,
        (AP_A, AP_B),
        returned_path,
        seed=22,
        reactivated_bssids=(AP_B,),
    )
    assert returned.manifest.vocabulary_migration.reactivated_bssids == (AP_B,)
    assert (returned.model.E[0].weight[:, 1] == original_encoder_slice).all()
    reordered_path = tmp_path / "vocab-reordered.pt"
    reordered = migrate_ae_checkpoint_vocabulary(
        returned_path,
        (AP_B, AP_A),
        reordered_path,
        seed=23,
    )
    assert reordered.manifest.bssid_vocabulary == (AP_B, AP_A)
    assert (reordered.model.E[0].weight[:, 0] == original_encoder_slice).all()
    assert original.checkpoint_sha256 == fixture.updater_models.checkpoint_sha256


def test_failed_migration_rolls_back_staging_and_parent_state(transition_fixture, tmp_path):
    fixture = transition_fixture
    destination = tmp_path / "failed-transition"
    parent_state_id = fixture.state.state_id
    bad_optimizer = {**fixture.updater_models.optimizer_states, "updated_ae_adam": {"state": {}, "param_groups": []}}
    bad_models = replace(fixture.updater_models, optimizer_states=bad_optimizer)
    with pytest.raises(ValueError, match="source AE optimizer state is malformed|single Adam"):
        apply_batch_transition(
            fixture.state,
            fixture.full_batch,
            fixture.updated_fingerprints,
            fixture.inference,
            fixture.ae_checkpoint,
            destination,
            TransitionConfig(
                ae_finetune_epochs=1,
                link_prediction=fixture.inference.modification.config,
            ),
            updater_models=bad_models,
        )
    assert not destination.exists()
    assert not tuple(tmp_path.glob(".failed-transition.staging-*"))
    assert fixture.state.state_id == parent_state_id


def test_unlinked_new_ap_is_not_misreported_as_forgotten(transition_fixture, tmp_path):
    fixture = transition_fixture
    no_link_change = fixture.inference.modification.model_copy(
        update={
            "added_reference_edges": (),
            "removed_reference_edges": (),
            "removed_ap_bssids": (),
            "final_reference_edges": fixture.state.radio_edges,
        }
    )
    result = apply_batch_transition(
        fixture.state,
        fixture.full_batch,
        fixture.updated_fingerprints,
        replace(fixture.inference, modification=no_link_change),
        fixture.ae_checkpoint,
        tmp_path / "unlinked-new-ap",
        TransitionConfig(
            ae_finetune_epochs=1,
            link_prediction=no_link_change.config,
        ),
        updater_models=fixture.updater_models,
    )
    new_ap = next(row for row in result.state.ap_registry if row.bssid == AP_C)
    assert new_ap.status == "observed_unlinked"
    assert AP_C not in result.state.bssid_vocabulary
    assert AP_C not in result.summary.removed_by_forgetting_bssids


def test_returning_ap_reactivates_after_prior_forgetting_transition(transition_fixture, tmp_path):
    fixture = transition_fixture
    config = TransitionConfig(
        ae_finetune_epochs=1,
        link_prediction=fixture.inference.modification.config,
    )
    keep_edges = tuple(edge for edge in fixture.state.radio_edges if edge.bssid != AP_B)
    removed_edges = tuple(edge for edge in fixture.state.radio_edges if edge.bssid == AP_B)
    removed_changes = tuple(
        LinkEdgeChange(
            edge_id=edge.edge_id,
            sample_node_id=edge.sample_node_id,
            ap_node_id=f"ap:{edge.bssid}",
            old_weight=edge.scalar_weight,
            new_weight=None,
            predicted_weight=0.0,
        )
        for edge in removed_edges
    )
    removal = APGraphModification(
        online_graph_id=fixture.inference.node_features.online_graph_id,
        source_state_id=fixture.state.state_id,
        input_batch_id=fixture.full_batch.config.batch_id,
        source_model_checkpoint_sha256=fixture.updater_models.checkpoint_sha256,
        source_vocabulary_sha256=fixture.state.vocabulary_sha256,
        config=config.link_prediction,
        candidate_scores=(),
        added_reference_edges=(),
        removed_reference_edges=removed_changes,
        removed_ap_bssids=(AP_B,),
        final_reference_edges=keep_edges,
    )
    after_removal = apply_batch_transition(
        fixture.state,
        fixture.full_batch,
        fixture.updated_fingerprints,
        replace(fixture.inference, modification=removal),
        fixture.ae_checkpoint,
        tmp_path / "transition-remove-b",
        config,
        updater_models=fixture.updater_models,
    )
    assert after_removal.state.bssid_vocabulary == (AP_A,)
    tombstone = next(row for row in after_removal.state.ap_registry if row.bssid == AP_B)
    assert tombstone.status == "tombstoned"
    assert tombstone.removed_by_batch_id == fixture.full_batch.config.batch_id

    (fixture.root / "returning.txt").write_text(
        "4,4 001122334455,-52 001122334466,-49\n", encoding="utf-8"
    )
    returning_batch = prepare_dataset_files(
        [fixture.root / "returning.txt"],
        PreprocessingConfig(
            dataset_id="transition-test",
            batch_id="batch-two",
            batch_role="update_batch",
            base_bssid_vocabulary=after_removal.state.bssid_vocabulary,
        ),
        source_root=fixture.root,
    )
    returned_edge = TransitionRadioEdge(
        edge_id=f"radio:{after_removal.state.reference_scan_ids[0]}|ap:{AP_B}",
        sample_node_id=after_removal.state.reference_scan_ids[0],
        bssid=AP_B,
        scalar_weight=71.0,
        edge_feature=(0.5,) * 34,
        source="new_ap_endpoint_mean",
    )
    returned_change = LinkEdgeChange(
        edge_id=returned_edge.edge_id,
        sample_node_id=returned_edge.sample_node_id,
        ap_node_id=f"ap:{AP_B}",
        old_weight=None,
        new_weight=71.0,
        predicted_weight=71.0,
    )
    returning_delta = APGraphModification(
        online_graph_id=fixture.inference.node_features.online_graph_id,
        source_state_id=after_removal.state.state_id,
        input_batch_id=returning_batch.config.batch_id,
        source_model_checkpoint_sha256=fixture.updater_models.checkpoint_sha256,
        source_vocabulary_sha256=after_removal.state.vocabulary_sha256,
        config=config.link_prediction,
        candidate_scores=(),
        added_reference_edges=(returned_change,),
        removed_reference_edges=(),
        removed_ap_bssids=(),
        final_reference_edges=after_removal.state.radio_edges + (returned_edge,),
    )
    return_plan = classify_and_plan_ap_vocabulary(
        after_removal.state, returning_batch, returning_delta
    )
    assert return_plan.shared_bssids == (AP_A,)
    assert return_plan.newly_observed_bssids == (AP_B,)
    assert return_plan.returning_bssids == (AP_B,)
    assert return_plan.vocabulary_after == (AP_A, AP_B)

    reactivated = migrate_ae_checkpoint_vocabulary(
        after_removal.migrated_ae_checkpoint.path,
        return_plan.vocabulary_after,
        tmp_path / "transition-return-b.pt",
        seed=99,
        reactivated_bssids=return_plan.returning_bssids,
    )
    assert reactivated.manifest.vocabulary_migration.reactivated_bssids == (AP_B,)
    assert AP_B not in reactivated.ap_parameter_archive
