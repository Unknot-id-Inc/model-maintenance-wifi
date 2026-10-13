import pytest
import torch

from ae import (
    AE,
    AEModelConfig,
    AETrainingConfig,
    encode_prepared_dataset,
    load_ae_checkpoint,
    load_encoder_features,
    migrate_ae_checkpoint_vocabulary,
    reconstruction_diagnostic,
    train_autoencoder,
)
from gufu_data import PreprocessingConfig, prepare_dataset_files
from gufu_artifacts import atomic_torch_save, load_torch_state


def synthetic_prepared(tmp_path, *, bssid_offset=0, normalization="paper_ae", role="bootstrap_reference"):
    vocab = [f"{index + bssid_offset:012X}" for index in range(673)]
    source = tmp_path / f"{role}-{bssid_offset}-{normalization}.txt"
    lines = []
    for row, (x, y) in enumerate(((0, 0), (1, 1), (2, 2)), start=1):
        readings = " ".join(f"{mac},-{45 + row + index % 20}" for index, mac in enumerate(vocab))
        lines.append(f"{x},{y} {readings}")
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")
    config = PreprocessingConfig(
        dataset_id="synthetic-floor",
        batch_id=f"{role}-{bssid_offset}-{normalization}",
        batch_role=role,
        normalization_mode=normalization,
    )
    return prepare_dataset_files([source], config, source_root=tmp_path)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_673_feature_training_batches_are_finite_and_checkpointed(tmp_path, batch_size):
    prepared = synthetic_prepared(tmp_path)
    checkpoint_path = tmp_path / f"ae-batch-{batch_size}.pt"

    result = train_autoencoder(
        prepared,
        checkpoint_path,
        AETrainingConfig(epochs=1, batch_size=batch_size, learning_rate=0.001, seed=31),
    )

    assert checkpoint_path.is_file()
    assert result.epochs_run == 1
    assert len(result.loss_history) == 1
    assert torch.isfinite(torch.tensor(result.loss_history)).all()
    model = result.checkpoint.model
    sample = torch.tensor(prepared.ae_matrix(), dtype=torch.float32).unsqueeze(1)
    model.train()
    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    parameters_before = [parameter.detach().clone() for parameter in model.parameters()]
    latent, reconstruction = model(sample)
    assert latent.shape == (3, 1, 32)
    assert reconstruction.shape == (3, 1, 673)
    loss = torch.nn.functional.mse_loss(reconstruction, sample)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    optimizer.step()
    parameters_after = list(model.parameters())
    assert all(torch.isfinite(parameter).all() for parameter in parameters_after)
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(parameters_before, parameters_after)
    )


def test_leaky_relu_and_kaiming_initialization_keep_encoder_gradient_path_open():
    torch.manual_seed(0)
    model = AE(input_dim=87)
    activations = [layer for layer in model.modules() if isinstance(layer, torch.nn.LeakyReLU)]
    assert len(activations) == 4
    assert all(layer.negative_slope == pytest.approx(0.1) for layer in activations)

    convolutions = [
        layer
        for layer in model.modules()
        if isinstance(layer, (torch.nn.Conv1d, torch.nn.ConvTranspose1d))
    ]
    assert len(convolutions) == 8
    for layer in convolutions:
        assert torch.count_nonzero(layer.weight) > 0
        assert torch.count_nonzero(layer.bias) == 0

    sample = torch.rand(3, 1, 87)
    _, reconstruction = model(sample)
    torch.nn.functional.mse_loss(reconstruction, sample).backward()
    assert model.E[0].weight.grad is not None
    assert torch.count_nonzero(model.E[0].weight.grad) > 0
    assert model.D[0].weight.grad is not None
    assert torch.count_nonzero(model.D[0].weight.grad) > 0
    assert AEModelConfig(input_dim=87).architecture_id == "gufu_ae_leaky_relu_kaiming_uniform_v2"


def test_autoencoder_training_visits_every_prepared_scan(tmp_path, monkeypatch):
    prepared = synthetic_prepared(tmp_path)
    training_batch_sizes = []
    original_forward = AE.forward

    def record_training_batches(model, values):
        if model.training:
            training_batch_sizes.append(values.shape[0])
        return original_forward(model, values)

    monkeypatch.setattr(AE, "forward", record_training_batches)
    train_autoencoder(
        prepared,
        tmp_path / "all-scans-ae.pt",
        AETrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=0.001,
            seed=13,
            shuffle=True,
        ),
        progress_eval_scans=1,
    )

    assert sum(training_batch_sizes) == len(prepared.scans) == 3
    assert training_batch_sizes == [2, 1]


def test_mse_and_frobenius_are_explicit_checkpointed_training_choices(
    tmp_path, capsys
):
    prepared = synthetic_prepared(tmp_path)
    common = dict(epochs=1, batch_size=2, learning_rate=0.001, seed=29, shuffle=False)

    mse = train_autoencoder(
        prepared,
        tmp_path / "ae-mse.pt",
        AETrainingConfig(**common, loss="mean_squared_error"),
        log_every_epochs=1,
        progress_eval_scans=1,
    )
    mse_log = capsys.readouterr().out
    assert mse.checkpoint.manifest.training_config.loss == "mean_squared_error"
    assert (
        mse.checkpoint.manifest.loss_history_definition
        == "element_weighted_batch_mse_per_epoch"
    )
    assert "train_mse=" in mse_log

    frobenius = train_autoencoder(
        prepared,
        tmp_path / "ae-frobenius.pt",
        AETrainingConfig(**common, loss="batch_frobenius_norm"),
        log_every_epochs=1,
        progress_eval_scans=1,
    )
    frobenius_log = capsys.readouterr().out
    assert (
        frobenius.checkpoint.manifest.training_config.loss
        == "batch_frobenius_norm"
    )
    assert (
        frobenius.checkpoint.manifest.loss_history_definition
        == "mean_of_batch_frobenius_norms_per_epoch"
    )
    assert "train_batch_frobenius_mean=" in frobenius_log
    assert all(torch.isfinite(torch.tensor(value)) for value in frobenius.loss_history)

    # Resume is continuation of the same objective, not an objective switch.
    with pytest.raises(ValueError, match="settings must match"):
        train_autoencoder(
            prepared,
            tmp_path / "ae-mse.pt",
            AETrainingConfig(**{**common, "epochs": 2}, loss="batch_frobenius_norm"),
            initialization="resume",
            source_checkpoint_path=tmp_path / "ae-mse.pt",
        )


def test_dense_width_and_dropout_are_used_saved_reloaded_and_migrated(tmp_path):
    prepared = synthetic_prepared(tmp_path)
    source_path = tmp_path / "custom-ae.pt"
    config = AETrainingConfig(
        epochs=1,
        batch_size=2,
        learning_rate=0.001,
        seed=41,
        shuffle=False,
        dense_width=64,
        latent_dim=32,
        dropout_probability=0.2,
    )
    trained = train_autoencoder(prepared, source_path, config)
    reloaded = load_ae_checkpoint(source_path, prepared)

    assert trained.checkpoint.manifest.model.dense_width == 64
    assert trained.checkpoint.manifest.model.latent_dim == 32
    assert trained.checkpoint.manifest.model.dropout_probability == 0.2
    assert trained.checkpoint.manifest.training_config.dense_width == 64
    assert reloaded.model.E[0].out_features == 64
    assert reloaded.model.E[1].out_features == 32
    assert reloaded.model.D[7].in_features == 32
    assert reloaded.model.D[7].out_features == 64
    assert [layer.p for layer in reloaded.model.modules() if isinstance(layer, torch.nn.Dropout)] == [0.2, 0.2]
    features = encode_prepared_dataset(
        prepared, source_path, tmp_path / "custom-encoder.json", batch_size=1
    )
    assert features.scan_ids == tuple(scan.scan_id for scan in prepared.scans)
    assert len(features.feature_matrix) == len(prepared.scans)
    assert all(len(row) == 32 for row in features.feature_matrix)

    target_vocabulary = prepared.bssid_vocabulary[:-1]
    migrated = migrate_ae_checkpoint_vocabulary(
        source_path,
        target_vocabulary,
        tmp_path / "custom-ae-migrated.pt",
        seed=43,
    )
    assert migrated.model.input_dim == len(target_vocabulary)
    assert migrated.manifest.model.dense_width == 64
    assert migrated.manifest.model.dropout_probability == 0.2
    assert migrated.model.E[0].out_features == 64
    assert migrated.model.D[7].out_features == 64

    variable_width = AE(
        673, config=AEModelConfig(input_dim=673, dense_width=64, latent_dim=16)
    )
    latent, reconstructed = variable_width(torch.zeros((2, 1, 673)))
    assert latent.shape == (2, 1, 16)
    assert reconstructed.shape == (2, 1, 673)
    assert AETrainingConfig(latent_dim=16).latent_dim == 16


def test_shuffled_training_exports_one_final_eval_embedding_per_scan_in_id_order(tmp_path):
    prepared = synthetic_prepared(tmp_path)
    checkpoint_path = tmp_path / "ae.pt"
    features_path = tmp_path / "encoder_features.json"
    trained = train_autoencoder(
        prepared,
        checkpoint_path,
        AETrainingConfig(epochs=2, batch_size=1, learning_rate=0.001, seed=7, shuffle=True),
    )

    artifact = encode_prepared_dataset(
        prepared, checkpoint_path, features_path, batch_size=2
    )

    expected_ids = tuple(scan.scan_id for scan in prepared.scans)
    assert artifact.scan_ids == expected_ids
    assert len(artifact.feature_matrix) == len(prepared.scans) == 3
    assert all(len(row) == 32 for row in artifact.feature_matrix)
    assert artifact.checkpoint_sha256 == trained.checkpoint.checkpoint_sha256
    assert artifact.bssid_vocabulary == prepared.bssid_vocabulary
    assert artifact.vocabulary_sha256 == trained.checkpoint.manifest.vocabulary_sha256
    assert load_encoder_features(features_path) == artifact

    reloaded = load_ae_checkpoint(checkpoint_path, prepared)
    assert reloaded.checkpoint_sha256 == trained.checkpoint.checkpoint_sha256
    reloaded.model.eval()
    values = torch.tensor(prepared.ae_matrix(), dtype=torch.float32).unsqueeze(1)
    with torch.inference_mode():
        latent, reconstruction = reloaded.model(values)
    assert torch.allclose(
        latent[:, 0, :], torch.tensor(artifact.feature_matrix), rtol=0.0, atol=1e-7
    )
    assert reconstruction.shape == values.shape
    assert reloaded.manifest.epochs_completed == 2
    assert reloaded.manifest.loss_history == trained.loss_history
    assert reloaded.manifest.training_config.loss == "mean_squared_error"
    assert (
        reloaded.manifest.loss_history_definition
        == "element_weighted_batch_mse_per_epoch"
    )
    diagnostic = reconstruction_diagnostic(prepared, checkpoint_path, max_scans=2, batch_size=1)
    assert diagnostic["scan_count"] == 2
    assert diagnostic["input_dim"] == 673
    assert torch.isfinite(torch.tensor([diagnostic["frobenius_norm"], diagnostic["rmse"]])).all()


def test_incompatible_vocabulary_and_normalization_are_rejected(tmp_path):
    prepared = synthetic_prepared(tmp_path)
    wrong_vocabulary = synthetic_prepared(tmp_path, bssid_offset=1)
    wrong_normalization = synthetic_prepared(
        tmp_path, normalization="legacy_raw_div_120"
    )
    checkpoint_path = tmp_path / "ae.pt"
    train_autoencoder(
        prepared,
        checkpoint_path,
        AETrainingConfig(epochs=1, batch_size=3, learning_rate=0.001, seed=4),
    )

    with pytest.raises(ValueError, match="vocabulary"):
        load_ae_checkpoint(checkpoint_path, wrong_vocabulary)
    with pytest.raises(ValueError, match="normalization"):
        load_ae_checkpoint(checkpoint_path, wrong_normalization)
    with pytest.raises(ValueError, match="vocabulary"):
        encode_prepared_dataset(
            wrong_vocabulary,
            checkpoint_path,
            tmp_path / "should-not-exist.json",
        )
    assert not (tmp_path / "should-not-exist.json").exists()


def test_pre_leaky_relu_checkpoint_architecture_is_rejected(tmp_path):
    prepared = synthetic_prepared(tmp_path)
    checkpoint_path = tmp_path / "ae-v2.pt"
    train_autoencoder(
        prepared,
        checkpoint_path,
        AETrainingConfig(epochs=1, batch_size=3, learning_rate=0.001, seed=17),
    )
    payload = load_torch_state(checkpoint_path)
    payload["manifest"]["model"]["architecture_id"] = (
        "gufu_ae_repaired_same_padding_v1"
    )
    incompatible_path = tmp_path / "ae-relu-v1.pt"
    atomic_torch_save(payload, incompatible_path)

    with pytest.raises(ValueError, match="invalid AE checkpoint manifest"):
        load_ae_checkpoint(incompatible_path, prepared)


def test_resume_restores_optimizer_progress_and_rng_state(tmp_path):
    prepared = synthetic_prepared(tmp_path)
    split_path = tmp_path / "split.pt"
    continuous_path = tmp_path / "continuous.pt"
    common = dict(batch_size=2, learning_rate=0.001, seed=83, shuffle=True)
    train_autoencoder(prepared, split_path, AETrainingConfig(epochs=1, **common))
    resumed = train_autoencoder(
        prepared,
        split_path,
        AETrainingConfig(epochs=2, **common),
        initialization="resume",
        source_checkpoint_path=split_path,
    )
    continuous = train_autoencoder(
        prepared, continuous_path, AETrainingConfig(epochs=2, **common)
    )

    assert resumed.epochs_run == 1
    assert resumed.loss_history == continuous.loss_history
    assert resumed.checkpoint.manifest.initialization_mode == "resume"
    assert resumed.checkpoint.manifest.source_checkpoint_sha256 is not None
    for resumed_parameter, continuous_parameter in zip(
        resumed.checkpoint.model.parameters(), continuous.checkpoint.model.parameters()
    ):
        assert torch.equal(resumed_parameter, continuous_parameter)


def test_weights_only_is_distinct_from_resume_and_update_batch_cannot_train(tmp_path):
    prepared = synthetic_prepared(tmp_path)
    source = tmp_path / "source.pt"
    target = tmp_path / "weights-only.pt"
    config = AETrainingConfig(epochs=1, batch_size=3, learning_rate=0.001, seed=2)
    train_autoencoder(prepared, source, config)
    weights_only_config = config.model_copy(update={"dropout_probability": 0.2})
    weights_only = train_autoencoder(
        prepared,
        target,
        weights_only_config,
        initialization="weights_only",
        source_checkpoint_path=source,
    )
    assert weights_only.checkpoint.manifest.initialization_mode == "weights_only"
    assert weights_only.checkpoint.manifest.epochs_completed == 1
    assert weights_only.checkpoint.manifest.source_checkpoint_sha256 is not None
    assert weights_only.checkpoint.manifest.model.dropout_probability == 0.2

    update = synthetic_prepared(
        tmp_path, bssid_offset=1000, normalization="paper_ae", role="update_batch"
    )
    with pytest.raises(ValueError, match="bootstrap_reference"):
        train_autoencoder(update, tmp_path / "update.pt", config)


def test_existing_checkpoint_and_feature_artifacts_are_not_overwritten(tmp_path):
    prepared = synthetic_prepared(tmp_path)
    checkpoint = tmp_path / "ae.pt"
    feature_path = tmp_path / "features.json"
    train_autoencoder(
        prepared,
        checkpoint,
        AETrainingConfig(epochs=1, batch_size=3, learning_rate=0.001, seed=9),
    )
    first_checkpoint = checkpoint.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        train_autoencoder(
            prepared,
            checkpoint,
            AETrainingConfig(epochs=1, batch_size=3, learning_rate=0.001, seed=9),
        )
    assert checkpoint.read_bytes() == first_checkpoint

    encode_prepared_dataset(prepared, checkpoint, feature_path)
    first_features = feature_path.read_bytes()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        encode_prepared_dataset(prepared, checkpoint, feature_path)
    assert feature_path.read_bytes() == first_features
