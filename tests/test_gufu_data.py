import pickle

import pytest
from pydantic import ValidationError

from gufu_data import (
    LEGACY_REGENERATION_MESSAGE,
    LegacyArtifactError,
    MalformedRecordError,
    PreprocessingConfig,
    load_legacy_pickle_pair,
    load_prepared_dataset,
    prepare_dataset_files,
    raw_dbm_row_to_ae,
    rssi_dbm_to_graph_weight,
    save_prepared_dataset,
)


def config(batch_id="b0", role="bootstrap_reference", **overrides):
    values = {
        "dataset_id": "siteA-floor1",
        "batch_id": batch_id,
        "batch_role": role,
    }
    values.update(overrides)
    return PreprocessingConfig(**values)


def write(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def test_raw_round_trip_retains_provenance_ids_masks_and_named_conversions(tmp_path):
    source = write(
        tmp_path / "input" / "GF" / "fingerprint0.txt",
        "10.0,20.0 00:11:22:33:44:55,-60,other 00-11-22-33-44-66,-90\n",
    )
    prepared = prepare_dataset_files(
        [source], config(), source_root=tmp_path / "input"
    )

    assert prepared.scans[0].source.source_file == "GF/fingerprint0.txt"
    assert prepared.scans[0].source.source_row == 1
    assert prepared.scans[0].rss_dbm == (-60.0, -90.0)
    assert prepared.scans[0].observation_mask == (True, True)
    assert prepared.ae_matrix() == ((0.5, 0.25),)
    assert raw_dbm_row_to_ae((-60.0, -120.0), (True, False)) == (0.5, 0.0)
    assert raw_dbm_row_to_ae(
        (-60.0, -120.0), (True, False), "legacy_raw_div_120"
    ) == (-0.5, -1.0)
    assert rssi_dbm_to_graph_weight(-60.0, True) == 60.0
    assert rssi_dbm_to_graph_weight(-120.0, False) is None

    artifact = save_prepared_dataset(prepared, tmp_path / "runs" / "prepared.json")
    restored = load_prepared_dataset(artifact)
    assert restored == prepared


def test_same_coordinates_do_not_merge_distinct_scans(tmp_path):
    source = write(
        tmp_path / "fingerprints.txt",
        "4.0,8.0 001122334455,-50\n4.0,8.0 001122334455,-51\n",
    )
    prepared = prepare_dataset_files([source], config(), source_root=tmp_path)

    assert len(prepared.scans) == 2
    assert prepared.scans[0].scan_id != prepared.scans[1].scan_id
    assert [item.source.source_row for item in prepared.scans] == [1, 2]


def test_full_bssid_identity_keeps_shared_prefixes_distinct(tmp_path):
    source = write(
        tmp_path / "f.txt",
        "0,0 AABBCCDDEE11,-50 AABBCCDDEE12,-60\n",
    )
    prepared = prepare_dataset_files([source], config(), source_root=tmp_path)

    assert prepared.bssid_vocabulary == ("AABBCCDDEE11", "AABBCCDDEE12")
    assert prepared.ap_ids == ("ap:AABBCCDDEE11", "ap:AABBCCDDEE12")
    assert prepared.scans[0].observation_mask == (True, True)


def test_mac_renderings_canonicalize_consistently(tmp_path):
    source = write(
        tmp_path / "f.txt",
        "0,0 aa:bb:cc:dd:ee:ff,-60\n0,0 AABB.CCDD.EEFF,-61\n",
    )
    prepared = prepare_dataset_files([source], config(), source_root=tmp_path)

    assert prepared.bssid_vocabulary == ("AABBCCDDEEFF",)
    assert len(prepared.scans) == 2
    assert prepared.scans[0].scan_id != prepared.scans[1].scan_id


def test_file_order_does_not_change_ids_or_vocabulary_and_batches_are_namespaced(tmp_path):
    a = write(tmp_path / "input" / "a.txt", "1,1 001122334455,-50\n")
    b = write(tmp_path / "input" / "b.txt", "2,2 AABBCCDDEE11,-70\n")
    first = prepare_dataset_files([a, b], config(), source_root=tmp_path / "input")
    reversed_input = prepare_dataset_files(
        [b, a], config(), source_root=tmp_path / "input"
    )
    next_batch = prepare_dataset_files(
        [a, b], config(batch_id="b1"), source_root=tmp_path / "input"
    )

    assert first == reversed_input
    assert first.bssid_vocabulary == ("001122334455", "AABBCCDDEE11")
    assert {scan.scan_id for scan in first.scans}.isdisjoint(
        {scan.scan_id for scan in next_batch.scans}
    )


def test_update_scan_identity_excludes_evaluation_only_coordinates(tmp_path):
    source = write(tmp_path / "fingerprint1.txt", "1,2 001122334455,-60\n")
    before = prepare_dataset_files(
        [source], config(batch_id="fingerprint1", role="update_batch"), source_root=tmp_path
    )
    write(source, "999,888 001122334455,-60\n")
    after = prepare_dataset_files(
        [source], config(batch_id="fingerprint1", role="update_batch"), source_root=tmp_path
    )

    assert before.scans[0].scan_id == after.scans[0].scan_id
    assert before.scans[0].source.source_record_sha256 != after.scans[0].source.source_record_sha256
    assert before.scans[0].source.scan_identity_sha256 == after.scans[0].source.scan_identity_sha256
    assert before.evaluation_positions != after.evaluation_positions


def test_configured_base_vocabulary_is_preserved_and_new_aps_append_sorted(tmp_path):
    source = write(
        tmp_path / "f.txt",
        "0,0 001122334466,-50 001122334455,-60\n",
    )
    prepared = prepare_dataset_files(
        [source],
        config(base_bssid_vocabulary=("AABBCCDDEE11", "001122334466")),
        source_root=tmp_path,
    )

    assert prepared.bssid_vocabulary == (
        "AABBCCDDEE11",
        "001122334466",
        "001122334455",
    )
    assert prepared.ap_ids == tuple(f"ap:{mac}" for mac in prepared.bssid_vocabulary)
    assert prepared.scans[0].observation_mask == (False, True, True)


def test_missing_measurements_use_mask_not_rssi_value_guess(tmp_path):
    source = write(
        tmp_path / "f.txt",
        "0,0 001122334455,-120\n1,1 AABBCCDDEE11,-55\n",
    )
    prepared = prepare_dataset_files([source], config(), source_root=tmp_path)

    assert prepared.scans[0].rss_dbm == (-120.0, -120.0)
    assert prepared.scans[0].observation_mask == (True, False)
    assert prepared.scans[1].observation_mask == (False, True)
    assert prepared.ae_matrix()[0] == (0.0, 0.0)


def test_one_pruning_policy_drives_counts_vocabulary_and_scan_matrix(tmp_path):
    source = write(
        tmp_path / "f.txt",
        "0,0 00:11:22:33:44:55,-40 02:11:22:33:44:66,-50 06:11:22:33:44:77,-60 08:11:22:33:44:88,-60\n",
    )
    prepared = prepare_dataset_files(
        [source],
        config(prune_randomized_bssids=True, rssi_threshold_dbm=-55.0),
        source_root=tmp_path,
    )

    assert prepared.stats.readings_pruned == 2
    assert prepared.stats.readings_below_threshold == 1
    assert prepared.bssid_vocabulary == ("001122334455",)
    assert prepared.scans[0].observation_mask == (True,)
    assert len(prepared.scans[0].rss_dbm) == len(prepared.bssid_vocabulary)


def test_duplicate_canonical_bssids_follow_the_named_policy(tmp_path):
    source = write(
        tmp_path / "f.txt",
        "0,0 001122334455,-70 00:11:22:33:44:55,-50\n",
    )
    strongest = prepare_dataset_files([source], config(), source_root=tmp_path)
    dropped = prepare_dataset_files(
        [source],
        config(duplicate_bssid_policy="error"),
        source_root=tmp_path,
    )

    assert strongest.scans[0].rss_dbm == (-50.0,)
    assert strongest.stats.duplicate_bssid_entries == 1
    assert dropped.scans == ()
    assert dropped.dropped_records[0].reason == "duplicate_bssid_in_scan"


def test_threshold_and_empty_scan_policies_are_explicit(tmp_path):
    source = write(
        tmp_path / "f.txt",
        "0,0 001122334455,-80\n1,1\n2,2 AABBCCDDEE11,-50\n",
    )
    kept = prepare_dataset_files(
        [source], config(rssi_threshold_dbm=-70.0), source_root=tmp_path
    )
    dropped = prepare_dataset_files(
        [source],
        config(rssi_threshold_dbm=-70.0, empty_scan_policy="drop"),
        source_root=tmp_path,
    )

    assert len(kept.scans) == 3
    assert kept.stats.readings_below_threshold == 1
    assert kept.scans[0].observation_mask == (False,)
    assert kept.scans[1].observation_mask == (False,)
    assert kept.scans[2].observation_mask == (True,)
    assert len(dropped.scans) == 1
    assert dropped.stats.empty_scans_dropped == 2
    assert [record.source_row for record in dropped.dropped_records] == [1, 2]


def test_malformed_record_drop_preserves_physical_source_row_and_error_mode(tmp_path):
    source = write(
        tmp_path / "f.txt",
        "not,a,coordinate 001122334455,-50\n1,2 001122334455,-60\n",
    )
    dropped = prepare_dataset_files([source], config(), source_root=tmp_path)
    assert len(dropped.scans) == 1
    assert dropped.scans[0].source.source_row == 2
    assert dropped.dropped_records[0].source_row == 1
    assert dropped.stats.malformed_records_dropped == 1

    with pytest.raises(MalformedRecordError, match="f.txt:1"):
        prepare_dataset_files(
            [source],
            config(malformed_record_policy="error"),
            source_root=tmp_path,
        )


def test_duplicate_source_paths_are_rejected(tmp_path):
    source = write(tmp_path / "f.txt", "0,0 001122334455,-50\n")
    with pytest.raises(ValueError, match="duplicate input file"):
        prepare_dataset_files([source, source], config(), source_root=tmp_path)


def test_prepared_contract_rejects_duplicate_scan_ids(tmp_path):
    source = write(tmp_path / "f.txt", "0,0 001122334455,-50\n1,1 AABBCCDDEE11,-60\n")
    prepared = prepare_dataset_files([source], config(), source_root=tmp_path)
    duplicate = prepared.model_dump(mode="python")
    duplicate["scans"] = (duplicate["scans"][0], duplicate["scans"][0])

    with pytest.raises(ValidationError, match="scan IDs must be unique"):
        type(prepared).model_validate(duplicate)


def test_update_coordinates_are_evaluation_only(tmp_path):
    source = write(tmp_path / "update.txt", "3,7 001122334455,-50\n")
    prepared = prepare_dataset_files(
        [source],
        config(batch_id="u1", role="update_batch"),
        source_root=tmp_path,
    )

    assert prepared.scans[0].coordinate_role == "evaluation_only"
    assert prepared.reference_positions == ()
    assert prepared.training_reference_positions() == ()
    assert prepared.evaluation_positions[0].scan_id == prepared.scans[0].scan_id


def test_jsonl_timestamps_convert_float_seconds_to_integer_milliseconds(tmp_path):
    source = write(
        tmp_path / "f.jsonl",
        '{"timestamp": 12.3456, "data": [{"bssid": "00:11:22:33:44:55", "rssi": -60}]}\n',
    )
    prepared = prepare_dataset_files(
        [source],
        config(input_format="gufu_jsonl"),
        source_root=tmp_path,
    )
    assert prepared.scans[0].timestamp == 12346
    saved = save_prepared_dataset(prepared, tmp_path / "timestamped.json")
    assert load_prepared_dataset(saved).scans[0].timestamp == 12346


def test_existing_files_are_preserved_and_artifacts_do_not_overwrite(tmp_path):
    source = write(tmp_path / "f.txt", "0,0 001122334455,-50\n")
    prepared = prepare_dataset_files([source], config(), source_root=tmp_path)
    run_dir = tmp_path / "existing-run"
    run_dir.mkdir()
    user_file = write(run_dir / "notes.txt", "keep this\n")
    artifact = save_prepared_dataset(prepared, run_dir / "prepared.json")
    original_artifact = artifact.read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        save_prepared_dataset(prepared, artifact)
    assert artifact.read_bytes() == original_artifact
    assert user_file.read_text(encoding="utf-8") == "keep this\n"


def test_trusted_legacy_adapter_requires_opt_in_and_recoverable_identity(tmp_path):
    observations_path = tmp_path / "observations.pkl"
    aps_path = tmp_path / "aps.pkl"
    observations = [["u1", "fingerprint0.txt", [1.0, 2.0], {"001122334455": -55.0}]]
    aps = [["i1", "001122334455", None, []]]
    observations_path.write_bytes(pickle.dumps(observations))
    aps_path.write_bytes(pickle.dumps(aps))

    with pytest.raises(PermissionError, match="trusted_local=True"):
        load_legacy_pickle_pair(
            observations_path, aps_path, config=config(), source_row_by_legacy_id={"u1": 1}
        )
    with pytest.raises(LegacyArtifactError, match="source row"):
        load_legacy_pickle_pair(
            observations_path, aps_path, trusted_local=True, config=config()
        )

    prepared = load_legacy_pickle_pair(
        observations_path,
        aps_path,
        trusted_local=True,
        config=config(),
        source_row_by_legacy_id={"u1": 9},
    )
    assert prepared.scans[0].source.source_kind == "legacy_pickle"
    assert prepared.scans[0].source.source_row == 9
    assert prepared.reference_positions[0].scan_id == prepared.scans[0].scan_id


def test_legacy_adapter_refuses_truncated_bssids_with_regeneration_guidance(tmp_path):
    observations_path = tmp_path / "observations.pkl"
    aps_path = tmp_path / "aps.pkl"
    observations_path.write_bytes(
        pickle.dumps([["u1", "fingerprint0.txt", [1.0, 2.0], {"CCDDEEFF": -55.0}]])
    )
    aps_path.write_bytes(pickle.dumps([["i1", "CCDDEEFF", None, []]]))

    with pytest.raises(LegacyArtifactError, match="complete supported BSSID") as error:
        load_legacy_pickle_pair(
            observations_path,
            aps_path,
            trusted_local=True,
            config=config(),
            source_row_by_legacy_id={"u1": 1},
        )
    assert LEGACY_REGENERATION_MESSAGE in str(error.value)
