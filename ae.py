"""GU-FU autoencoder and persisted training/inference APIs.

This local AE retains the repaired shape and dropout layout, uses LeakyReLU and
Kaiming-uniform convolution initialization to avoid a dead decoder path, and
supports either element-mean squared error or the paper's minibatch Frobenius
objective through an explicit training setting.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch import nn

from gufu_artifacts import (
    atomic_torch_save,
    atomic_write_json,
    canonical_json_digest,
    load_torch_state,
    read_json_text,
    sha256_file,
)
from gufu_data import NormalizationMode, PreparedDataset, canonicalize_bssid
from training_progress import (
    DEFAULT_LOG_EVERY_K_EPOCHS,
    log_training_epoch,
    should_log_training_epoch,
    validate_log_interval,
)

ARCHITECTURE_ID = "gufu_ae_leaky_relu_kaiming_uniform_v2"
CHECKPOINT_FORMAT = "gufu_ae_checkpoint_v1"
FEATURE_FORMAT = "gufu_encoder_features_v1"
AE_PROGRESS_DIAGNOSTIC_SCANS = 4
AE_DECODER_OUTPUT_LAYER_INDEX = 8


class AEModelConfig(BaseModel):
    """Complete shape/architecture record for the preserved local AE."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    architecture_id: Literal["gufu_ae_leaky_relu_kaiming_uniform_v2"] = ARCHITECTURE_ID
    input_dim: int = Field(ge=1)
    dense_width: int = Field(default=256, ge=1)
    latent_dim: int = Field(default=32, ge=1)
    input_channels: Literal[1] = 1
    encoder_kernel_sizes: tuple[int, ...] = (3, 3, 3, 11)
    encoder_padding: tuple[int, ...] = (1, 1, 1, 5)
    decoder_kernel_sizes: tuple[int, ...] = (11, 3, 3, 3)
    decoder_padding: tuple[int, ...] = (5, 1, 1, 1)
    dropout_probability: float = Field(default=0.5, ge=0.0, lt=1.0)
    activation: Literal["leaky_relu"] = "leaky_relu"
    leaky_relu_negative_slope: float = 0.1
    convolution_weight_initialization: Literal[
        "kaiming_uniform_fan_in_leaky_relu_a_0_1"
    ] = "kaiming_uniform_fan_in_leaky_relu_a_0_1"
    convolution_bias_initialization: Literal["zeros"] = "zeros"
    encoder_dropout_after_convolution: Literal[2] = 2
    decoder_dropout_after_convolution: Literal[2] = 2
    padding_decision: Literal["symmetric_length_preserving_padding_inferred"] = (
        "symmetric_length_preserving_padding_inferred"
    )
    dropout_decision: Literal[
        "dropout_replaces_second_convolution_activation_inferred_from_local_repair"
    ] = "dropout_replaces_second_convolution_activation_inferred_from_local_repair"
    return_contract: Literal["(latent, reconstruction)"] = "(latent, reconstruction)"

    @model_validator(mode="after")
    def validate_architecture_record(self):
        if len(self.encoder_kernel_sizes) != 4 or len(self.encoder_padding) != 4:
            raise ValueError("the repaired encoder must record four convolutions")
        if len(self.decoder_kernel_sizes) != 4 or len(self.decoder_padding) != 4:
            raise ValueError("the repaired decoder must record four convolutions")
        return self


class AETrainingConfig(BaseModel):
    """Reproducible epoch-boundary Adam training settings."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    epochs: int = Field(default=10, ge=1)
    batch_size: int = Field(default=32, ge=1)
    learning_rate: float = Field(default=0.01, gt=0.0)
    optimizer_betas: tuple[float, float] = (0.9, 0.999)
    optimizer_epsilon: float = Field(default=1e-8, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    amsgrad: bool = False
    seed: int = 0
    shuffle: bool = True
    dtype: Literal["float32", "float64"] = "float32"
    optimizer: Literal["Adam"] = "Adam"
    dense_width: int = Field(default=256, ge=1)
    latent_dim: int = Field(default=32, ge=1)
    dropout_probability: float = Field(default=0.5, ge=0.0, lt=1.0)
    loss: Literal["mean_squared_error", "batch_frobenius_norm"] = "mean_squared_error"

    @field_validator("learning_rate")
    @classmethod
    def learning_rate_is_finite(cls, value: float) -> float:
        if not torch.isfinite(torch.tensor(value)).item():
            raise ValueError("learning_rate must be finite")
        return value

    @field_validator("dropout_probability")
    @classmethod
    def dropout_probability_is_finite(cls, value: float) -> float:
        if not torch.isfinite(torch.tensor(value)).item():
            raise ValueError("dropout_probability must be finite")
        return value

    @field_validator("optimizer_epsilon", "weight_decay")
    @classmethod
    def optimizer_scalars_are_finite(cls, value: float) -> float:
        if not torch.isfinite(torch.tensor(value)).item():
            raise ValueError("optimizer values must be finite")
        return value

    @field_validator("optimizer_betas")
    @classmethod
    def adam_betas_are_valid(cls, value: tuple[float, float]) -> tuple[float, float]:
        if len(value) != 2 or any(
            not torch.isfinite(torch.tensor(beta)).item() or not 0.0 <= beta < 1.0
            for beta in value
        ):
            raise ValueError("Adam betas must be finite values in [0, 1)")
        return value


class DatasetIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    dataset_id: str
    batch_id: str
    batch_role: Literal["bootstrap_reference"]
    prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scan_ids: tuple[str, ...]
    scan_count: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_ids(self):
        if self.scan_count != len(self.scan_ids):
            raise ValueError("dataset scan count does not match its scan IDs")
        if len(set(self.scan_ids)) != len(self.scan_ids):
            raise ValueError("dataset identity contains duplicate scan IDs")
        return self


class AECheckpointTrainingConfig(AETrainingConfig):
    """Reader for current MSE and historical Frobenius checkpoint metadata."""

    loss: Literal["mean_squared_error", "batch_frobenius_norm"] = "mean_squared_error"


class AEVocabularyMigrationRecord(BaseModel):
    """Auditable AP-identity transfer for one vocabulary width change."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    source_vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_vocabulary: tuple[str, ...]
    target_vocabulary: tuple[str, ...]
    added_bssids: tuple[str, ...]
    removed_bssids: tuple[str, ...]
    reactivated_bssids: tuple[str, ...]
    seed: int
    source_model_identity: str | None = None
    transition_batch_id: str | None = None
    finetune_epochs: int = Field(default=0, ge=0)
    finetune_seed: int | None = None
    finetune_loss_history: tuple[float, ...] = ()
    parameter_policy: Literal[
        "copy_shared_by_bssid_restore_tombstone_or_seed_new_ap_slices_v1"
    ] = "copy_shared_by_bssid_restore_tombstone_or_seed_new_ap_slices_v1"
    optimizer_policy: Literal[
        "reset_resized_tensor_state_preserve_unaffected_adam_state_v1"
    ] = "reset_resized_tensor_state_preserve_unaffected_adam_state_v1"

    @model_validator(mode="after")
    def aligned_migration(self):
        for vocabulary in (self.source_vocabulary, self.target_vocabulary):
            if len(set(vocabulary)) != len(vocabulary) or any(
                canonicalize_bssid(value) != value for value in vocabulary
            ):
                raise ValueError(
                    "AE migration vocabularies must use unique full BSSIDs"
                )
        source, target = set(self.source_vocabulary), set(self.target_vocabulary)
        if set(self.added_bssids) != target - source:
            raise ValueError("AE migration added BSSIDs do not match vocabulary delta")
        if set(self.removed_bssids) != source - target:
            raise ValueError(
                "AE migration removed BSSIDs do not match vocabulary delta"
            )
        if len(set(self.reactivated_bssids)) != len(self.reactivated_bssids) or not set(
            self.reactivated_bssids
        ) <= set(self.added_bssids):
            raise ValueError("reactivated BSSIDs must be a unique subset of additions")
        if (
            canonical_json_digest(self.source_vocabulary)
            != self.source_vocabulary_sha256
        ):
            raise ValueError("source AE migration vocabulary hash is inconsistent")
        if (
            canonical_json_digest(self.target_vocabulary)
            != self.target_vocabulary_sha256
        ):
            raise ValueError("target AE migration vocabulary hash is inconsistent")
        if len(self.finetune_loss_history) != self.finetune_epochs:
            raise ValueError(
                "migration fine-tune losses must align with completed epochs"
            )
        if any(
            not torch.isfinite(torch.tensor(value)).item()
            for value in self.finetune_loss_history
        ):
            raise ValueError("migration fine-tune losses must be finite")
        if self.finetune_epochs and (
            self.transition_batch_id is None or self.finetune_seed is None
        ):
            raise ValueError("fine-tuned migration must record its batch and seed")
        return self


class AECheckpointManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    model: AEModelConfig
    normalization: NormalizationMode
    bssid_vocabulary: tuple[str, ...]
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_dataset: DatasetIdentity
    training_config: AECheckpointTrainingConfig
    initialization_mode: Literal[
        "fresh", "weights_only", "resume", "vocabulary_migration", "streaming_update"
    ]
    source_checkpoint_sha256: str | None = None
    vocabulary_migration: AEVocabularyMigrationRecord | None = None
    checkpoint_objective: Literal[
        "pretraining_configured_loss_v1", "streaming_eq10_update_v1"
    ] = "pretraining_configured_loss_v1"
    update_loss_components: tuple[tuple[float, float, float], ...] = ()
    training_resume_compatible: bool = True
    epochs_completed: int = Field(ge=0)
    loss_history: tuple[float, ...]
    loss_history_definition: Literal[
        "element_weighted_batch_mse_per_epoch",
        "mean_of_batch_frobenius_norms_per_epoch",
        "eq10_total_frobenius_objective_per_epoch",
    ] = "element_weighted_batch_mse_per_epoch"
    optimizer_state_present: bool
    random_state_present: bool
    device_used: str
    created_at_utc: str

    @model_validator(mode="after")
    def validate_checkpoint_contract(self):
        for field_name in ("dense_width", "latent_dim", "dropout_probability"):
            if getattr(self.model, field_name) != getattr(
                self.training_config, field_name
            ):
                raise ValueError(
                    f"AE model and training configuration disagree on {field_name}"
                )
        expected_history_definition = (
            "eq10_total_frobenius_objective_per_epoch"
            if self.checkpoint_objective == "streaming_eq10_update_v1"
            else {
                "mean_squared_error": "element_weighted_batch_mse_per_epoch",
                "batch_frobenius_norm": "mean_of_batch_frobenius_norms_per_epoch",
            }[self.training_config.loss]
        )
        if self.loss_history_definition != expected_history_definition:
            raise ValueError(
                "AE loss history definition does not match the configured loss"
            )
        if len(self.bssid_vocabulary) != self.model.input_dim:
            raise ValueError("model input width must equal ordered BSSID vocabulary")
        if canonical_json_digest(self.bssid_vocabulary) != self.vocabulary_sha256:
            raise ValueError("checkpoint vocabulary hash is inconsistent")
        if self.epochs_completed != len(self.loss_history):
            raise ValueError("loss history must contain one scalar per completed epoch")
        if any(
            not torch.isfinite(torch.tensor(value)).item()
            for value in self.loss_history
        ):
            raise ValueError("loss history values must be finite")
        if self.epochs_completed > self.training_config.epochs:
            raise ValueError("completed epoch count exceeds configured target")
        if self.checkpoint_objective == "streaming_eq10_update_v1":
            if self.initialization_mode not in {"streaming_update", "vocabulary_migration"}:
                raise ValueError("Eq. (10) checkpoints must declare streaming update or its vocabulary migration")
            if self.training_resume_compatible:
                raise ValueError("Eq. (10) checkpoint is not a generic AE epoch-boundary resume")
            if self.source_checkpoint_sha256 is None:
                raise ValueError("Eq. (10) checkpoint must identify its source AE checkpoint")
            if len(self.update_loss_components) != self.epochs_completed:
                raise ValueError("Eq. (10) component history must align with completed epochs")
            if any(
                len(row) != 3 or any(not torch.isfinite(torch.tensor(value)).item() for value in row)
                for row in self.update_loss_components
            ):
                raise ValueError("Eq. (10) component history must contain finite triples")
        elif self.update_loss_components:
            raise ValueError("pretraining checkpoints cannot carry Eq. (10) component history")
        if self.initialization_mode == "resume" and not self.training_resume_compatible:
            raise ValueError("this checkpoint is not compatible with generic AE training resume")
        if self.initialization_mode == "resume":
            if self.source_checkpoint_sha256 is None:
                raise ValueError(
                    "resume checkpoint must identify its parent checkpoint"
                )
        if self.initialization_mode == "vocabulary_migration":
            if (
                self.source_checkpoint_sha256 is None
                or self.vocabulary_migration is None
            ):
                raise ValueError(
                    "vocabulary migration checkpoint must identify its source and mapping"
                )
            if self.vocabulary_migration.target_vocabulary != self.bssid_vocabulary:
                raise ValueError(
                    "migrated AE vocabulary does not match checkpoint vocabulary"
                )
        elif self.vocabulary_migration is not None:
            raise ValueError(
                "only vocabulary-migration checkpoints may carry migration metadata"
            )
        if not self.optimizer_state_present or not self.random_state_present:
            raise ValueError(
                "operational AE checkpoints must preserve optimizer and RNG state"
            )
        return self


class EncoderFeatureArtifact(BaseModel):
    """ID-aligned, eval-mode encoder output from one fixed checkpoint."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    artifact_format: Literal["gufu_encoder_features_v1"] = FEATURE_FORMAT
    feature_space: Literal["z_encoder"] = "z_encoder"
    dataset_id: str
    batch_id: str
    source_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scan_ids: tuple[str, ...]
    feature_matrix: tuple[tuple[float, ...], ...]
    feature_dim: int = Field(default=32, ge=1)
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bssid_vocabulary: tuple[str, ...]
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalization: NormalizationMode
    dtype: Literal["float32", "float64"]
    inference_mode: Literal["eval_inference_mode"] = "eval_inference_mode"
    created_at_utc: str

    @model_validator(mode="after")
    def validate_feature_contract(self):
        if len(set(self.scan_ids)) != len(self.scan_ids):
            raise ValueError("encoder feature scan IDs must be unique")
        if len(self.scan_ids) != len(self.feature_matrix):
            raise ValueError("one encoder feature row is required per scan ID")
        if canonical_json_digest(self.bssid_vocabulary) != self.vocabulary_sha256:
            raise ValueError("feature vocabulary hash is inconsistent")
        if any(len(row) != self.feature_dim for row in self.feature_matrix):
            raise ValueError("encoder feature rows must match the declared feature_dim")
        if any(
            not torch.isfinite(torch.tensor(value)).item()
            for row in self.feature_matrix
            for value in row
        ):
            raise ValueError("encoder features must be finite")
        return self


@dataclass(frozen=True)
class LoadedAECheckpoint:
    model: AE
    manifest: AECheckpointManifest
    checkpoint_sha256: str
    path: Path
    optimizer_state: dict
    random_state: dict
    ap_parameter_archive: dict[str, dict[str, torch.Tensor]]


@dataclass(frozen=True)
class AETrainingResult:
    checkpoint: LoadedAECheckpoint
    loss_history: tuple[float, ...]
    epochs_run: int


class AE(nn.Module):
    def __init__(self, input_dim: int, *, config: AEModelConfig | None = None):
        super().__init__()
        if not isinstance(input_dim, int) or input_dim < 1:
            raise ValueError("input_dim must be a positive integer")
        selected_config = config or AEModelConfig(input_dim=input_dim)
        if not isinstance(selected_config, AEModelConfig):
            raise TypeError("config must be an AEModelConfig")
        if selected_config.input_dim != input_dim:
            raise ValueError("AE input_dim must match its model configuration")
        self.config = selected_config
        self.input_dim = input_dim
        self.E = nn.Sequential(
            nn.Linear(input_dim, selected_config.dense_width),
            # nn.ReLU(),
            nn.Linear(selected_config.dense_width, selected_config.latent_dim),
            nn.Conv1d(1, 1, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(1, 1, 3, padding=1),
            nn.Dropout(p=selected_config.dropout_probability),
            nn.Conv1d(1, 1, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv1d(1, 1, 11, padding=5),
        )
        self.D = nn.Sequential(
            nn.ConvTranspose1d(1, 1, 11, padding=5),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose1d(1, 1, 3, padding=1),
            nn.Dropout(p=selected_config.dropout_probability),
            nn.ConvTranspose1d(1, 1, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.ConvTranspose1d(1, 1, 3, padding=1),
            nn.Linear(selected_config.latent_dim, selected_config.dense_width),
            # nn.ReLU(),
            nn.Linear(selected_config.dense_width, input_dim),
        )
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.kaiming_uniform_(
                    layer.weight,
                    a=0.1,
                    mode="fan_in",
                    nonlinearity="leaky_relu",
                )
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor):
        if x.ndim != 3 or x.shape[1:] != (1, self.input_dim):
            raise ValueError(
                f"Expected [batch, 1, {self.input_dim}], got {tuple(x.shape)}"
            )
        z = self.E(x)
        output = self.D(z)
        if z.shape != (x.shape[0], 1, self.config.latent_dim):
            raise RuntimeError(f"AE latent shape contract failed: {tuple(z.shape)}")
        if output.shape != x.shape:
            raise RuntimeError(
                f"AE reconstruction shape contract failed: {tuple(output.shape)}"
            )
        return z, output


def ae_input_values(
    prepared: PreparedDataset,
    mode: NormalizationMode | None = None,
) -> tuple[tuple[float, ...], ...]:
    """Return the artifact's scan/AP ordered RSS matrix in a named AE space."""
    return prepared.ae_matrix(mode)


def _torch_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"unsupported AE dtype: {name}")


def _device(value: str | torch.device) -> torch.device:
    selected = torch.device(value)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"requested CUDA device is unavailable: {selected}")
    return selected


def _prepared_inputs(
    prepared: PreparedDataset, *, require_bootstrap: bool = False
) -> tuple[torch.Tensor, tuple[str, ...]]:
    if require_bootstrap and prepared.config.batch_role != "bootstrap_reference":
        raise ValueError("AE training requires a bootstrap_reference PreparedDataset")
    if not prepared.scans:
        raise ValueError("AE requires at least one retained scan")
    if not prepared.bssid_vocabulary:
        raise ValueError("AE requires a non-empty ordered BSSID vocabulary")
    matrix = torch.tensor(ae_input_values(prepared), dtype=torch.float64)
    if matrix.ndim != 2 or matrix.shape != (
        len(prepared.scans),
        len(prepared.bssid_vocabulary),
    ):
        raise ValueError("prepared AE matrix shape does not match scans and vocabulary")
    if not torch.isfinite(matrix).all().item():
        raise ValueError("prepared AE matrix contains non-finite values")
    ids = tuple(scan.scan_id for scan in prepared.scans)
    if len(set(ids)) != len(ids):
        raise ValueError("prepared dataset contains duplicate scan IDs")
    return matrix, ids


def _eval_reconstruction_metrics(
    model: AE,
    values: torch.Tensor,
    *,
    scan_count: int,
    batch_size: int,
) -> tuple[int, float, float]:
    """Evaluate a fixed prefix of scans with dropout disabled.

    Frobenius norm is the L2 norm across every selected scan and AP value;
    RMSE is normalized per scalar value. The model's prior train/eval mode is
    restored so this diagnostic does not alter subsequent training behavior.
    """
    if type(scan_count) is not int or scan_count < 1:
        raise ValueError("scan_count must be a positive integer")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    selected_count = min(scan_count, values.shape[0])
    if selected_count < 1:
        raise ValueError("reconstruction metrics require at least one scan")
    parameter = next(model.parameters())
    model_was_training = model.training
    model.eval()
    try:
        reconstructions: list[torch.Tensor] = []
        with torch.inference_mode():
            for start in range(0, selected_count, batch_size):
                batch = (
                    values[start : min(start + batch_size, selected_count)]
                    .to(device=parameter.device, dtype=parameter.dtype)
                    .unsqueeze(1)
                )
                _latent, reconstruction = model(batch)
                reconstructions.append(reconstruction)
            target = (
                values[:selected_count]
                .to(device=parameter.device, dtype=parameter.dtype)
                .unsqueeze(1)
            )
            reconstructed = torch.cat(reconstructions, dim=0)
            if reconstructed.shape != target.shape:
                raise AssertionError(
                    "diagnostic reconstruction and target shapes differ"
                )
            difference = reconstructed - target
            frobenius = float(torch.linalg.vector_norm(difference).cpu().item())
            rmse = float(torch.sqrt(torch.mean(difference.square())).cpu().item())
            if not torch.isfinite(torch.tensor([frobenius, rmse])).all().item():
                raise FloatingPointError("reconstruction diagnostic is non-finite")
            return selected_count, frobenius, rmse
    finally:
        model.train(model_was_training)


def _dataset_identity(
    prepared: PreparedDataset, scan_ids: tuple[str, ...]
) -> DatasetIdentity:
    return DatasetIdentity(
        dataset_id=prepared.config.dataset_id,
        batch_id=prepared.config.batch_id,
        batch_role="bootstrap_reference",
        prepared_sha256=canonical_json_digest(prepared),
        scan_ids=scan_ids,
        scan_count=len(scan_ids),
    )


def _model_config(
    prepared: PreparedDataset, training: AETrainingConfig | None = None
) -> AEModelConfig:
    selected = training or AETrainingConfig()
    return AEModelConfig(
        input_dim=len(prepared.bssid_vocabulary),
        dense_width=selected.dense_width,
        latent_dim=selected.latent_dim,
        dropout_probability=selected.dropout_probability,
    )


def _vocab_hash(prepared: PreparedDataset) -> str:
    return canonical_json_digest(prepared.bssid_vocabulary)


def _check_inference_compatibility(
    manifest: AECheckpointManifest, prepared: PreparedDataset
) -> None:
    if prepared.config.dataset_id != manifest.training_dataset.dataset_id:
        raise ValueError("checkpoint dataset_id does not match prepared data")
    if prepared.bssid_vocabulary != manifest.bssid_vocabulary:
        raise ValueError(
            "checkpoint ordered BSSID vocabulary does not match prepared data"
        )
    if _vocab_hash(prepared) != manifest.vocabulary_sha256:
        raise ValueError(
            "checkpoint BSSID vocabulary hash does not match prepared data"
        )
    if prepared.config.normalization_mode != manifest.normalization:
        raise ValueError("checkpoint normalization does not match prepared data")
    if len(prepared.bssid_vocabulary) != manifest.model.input_dim:
        raise ValueError(
            "checkpoint input dimension does not match prepared vocabulary"
        )


def _state_payload(
    model: AE,
    optimizer: torch.optim.Optimizer,
    *,
    cpu_rng_state: torch.Tensor,
    shuffle_rng_state: torch.Tensor,
    cuda_rng_state: torch.Tensor | None,
) -> dict:
    return {
        "format": CHECKPOINT_FORMAT,
        "encoder_state_dict": model.E.state_dict(),
        "decoder_state_dict": model.D.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "random_state": {
            "torch_cpu": cpu_rng_state,
            "shuffle_generator": shuffle_rng_state,
            "torch_cuda": cuda_rng_state,
        },
    }


def _load_raw_checkpoint(
    path: str | Path, *, map_location: str | torch.device = "cpu"
) -> tuple[dict, AECheckpointManifest]:
    payload = load_torch_state(path, map_location=map_location)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError("unsupported AE checkpoint format")
    try:
        manifest = AECheckpointManifest.model_validate(payload["manifest"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid AE checkpoint manifest: {exc}") from exc
    for name in (
        "encoder_state_dict",
        "decoder_state_dict",
        "optimizer_state_dict",
        "random_state",
    ):
        if name not in payload or not isinstance(payload[name], dict):
            raise ValueError(f"AE checkpoint is missing {name}")
    if (
        "torch_cpu" not in payload["random_state"]
        or "shuffle_generator" not in payload["random_state"]
    ):
        raise ValueError("AE checkpoint is missing resumable random state")
    if not isinstance(
        payload["optimizer_state_dict"].get("state"), dict
    ) or not isinstance(payload["optimizer_state_dict"].get("param_groups"), list):
        raise ValueError("AE checkpoint optimizer state is malformed")
    random_state = payload["random_state"]
    if not isinstance(random_state["torch_cpu"], torch.Tensor) or not isinstance(
        random_state["shuffle_generator"], torch.Tensor
    ):
        raise ValueError("AE checkpoint RNG state is malformed")
    if manifest.device_used.startswith("cuda") and not isinstance(
        random_state.get("torch_cuda"), torch.Tensor
    ):
        raise ValueError("CUDA AE checkpoint is missing CUDA RNG state")
    return payload, manifest


def load_ae_checkpoint(
    path: str | Path,
    prepared: PreparedDataset | None = None,
    *,
    device: str | torch.device = "cpu",
    dtype: Literal["float32", "float64"] | None = None,
) -> LoadedAECheckpoint:
    """Load portable weights and validate data compatibility before inference."""
    source = Path(path)
    payload, manifest = _load_raw_checkpoint(source, map_location="cpu")
    if prepared is not None:
        _check_inference_compatibility(manifest, prepared)
    selected_dtype = dtype or manifest.training_config.dtype
    model = AE(manifest.model.input_dim, config=manifest.model).to(
        device=_device(device), dtype=_torch_dtype(selected_dtype)
    )
    model.E.load_state_dict(payload["encoder_state_dict"], strict=True)
    model.D.load_state_dict(payload["decoder_state_dict"], strict=True)
    model.eval()
    return LoadedAECheckpoint(
        model=model,
        manifest=manifest,
        checkpoint_sha256=sha256_file(source),
        path=source,
        optimizer_state=payload["optimizer_state_dict"],
        random_state=payload["random_state"],
        ap_parameter_archive=payload.get("ap_parameter_archive", {}),
    )


def migrate_ae_checkpoint_vocabulary(
    source_checkpoint_path: str | Path,
    target_bssid_vocabulary: tuple[str, ...] | list[str],
    target_checkpoint_path: str | Path,
    *,
    seed: int,
    reactivated_bssids: tuple[str, ...] | list[str] = (),
    source_model: AE | None = None,
    source_optimizer_state: dict | None = None,
    source_model_identity: str | None = None,
    overwrite: bool = False,
) -> LoadedAECheckpoint:
    """Migrate AE input/output slices by full BSSID identity.

    The shared first-encoder columns and final-decoder rows/biases retain their
    learned values. New AP slices use a reproducible fresh initialization;
    returning APs recover their archived slices if previously tombstoned.
    Adam state for the three resized tensors is cleared as a whole because one
    scalar step counter cannot preserve a different optimizer age per AP
    slice. Unaffected parameter state and all shape-independent model weights
    are retained.
    """
    source_path = Path(source_checkpoint_path)
    target_path = Path(target_checkpoint_path)
    source_payload, source_manifest = _load_raw_checkpoint(source_path)
    target_vocabulary = tuple(
        canonicalize_bssid(value) for value in target_bssid_vocabulary
    )
    if not target_vocabulary or len(set(target_vocabulary)) != len(target_vocabulary):
        raise ValueError(
            "migrated AE vocabulary must be non-empty and contain unique BSSIDs"
        )
    source_vocabulary = source_manifest.bssid_vocabulary
    vocabulary_changed = target_vocabulary != source_vocabulary
    reactivated = tuple(canonicalize_bssid(value) for value in reactivated_bssids)
    added = tuple(
        value for value in target_vocabulary if value not in set(source_vocabulary)
    )
    removed = tuple(
        value for value in source_vocabulary if value not in set(target_vocabulary)
    )
    if len(set(reactivated)) != len(reactivated) or not set(reactivated) <= set(added):
        raise ValueError(
            "reactivated BSSIDs must be a unique subset of newly added identities"
        )
    if target_path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite migrated AE checkpoint: {target_path}"
        )

    archive: dict[str, dict[str, torch.Tensor]] = {
        bssid: {name: tensor.detach().cpu().clone() for name, tensor in slices.items()}
        for bssid, slices in source_payload.get("ap_parameter_archive", {}).items()
    }
    source_encoder = (
        source_payload["encoder_state_dict"]
        if source_model is None
        else {
            key: value.detach().cpu().clone()
            for key, value in source_model.E.state_dict().items()
        }
    )
    source_decoder = (
        source_payload["decoder_state_dict"]
        if source_model is None
        else {
            key: value.detach().cpu().clone()
            for key, value in source_model.D.state_dict().items()
        }
    )
    if source_model is not None and source_model.input_dim != len(source_vocabulary):
        raise ValueError("source AE model width does not match its declared vocabulary")
    source_encoder_input = source_encoder["0.weight"]
    decoder_output_prefix = str(AE_DECODER_OUTPUT_LAYER_INDEX)
    decoder_output_weight_name = f"{decoder_output_prefix}.weight"
    decoder_output_bias_name = f"{decoder_output_prefix}.bias"
    source_decoder_output = source_decoder[decoder_output_weight_name]
    source_decoder_bias = source_decoder[decoder_output_bias_name]
    for column, bssid in enumerate(source_vocabulary):
        if bssid in set(removed):
            archive[bssid] = {
                "encoder_input_weight": source_encoder_input[:, column]
                .detach()
                .cpu()
                .clone(),
                "decoder_output_weight": source_decoder_output[column, :]
                .detach()
                .cpu()
                .clone(),
                "decoder_output_bias": source_decoder_bias[column]
                .detach()
                .cpu()
                .clone(),
            }

    dtype = _torch_dtype(source_manifest.training_config.dtype)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        migrated_config = source_manifest.model.model_copy(
            update={"input_dim": len(target_vocabulary)}
        )
        migrated = AE(len(target_vocabulary), config=migrated_config).to(dtype=dtype)
    migrated_encoder = migrated.E.state_dict()
    migrated_decoder = migrated.D.state_dict()
    for name, tensor in source_encoder.items():
        if name != "0.weight":
            if migrated_encoder[name].shape != tensor.shape:
                raise ValueError(
                    f"unexpected encoder shape change during AP migration: {name}"
                )
            migrated_encoder[name] = tensor.detach().clone().to(dtype=dtype)
    for name, tensor in source_decoder.items():
        if name not in (decoder_output_weight_name, decoder_output_bias_name):
            if migrated_decoder[name].shape != tensor.shape:
                raise ValueError(
                    f"unexpected decoder shape change during AP migration: {name}"
                )
            migrated_decoder[name] = tensor.detach().clone().to(dtype=dtype)

    old_rows = {bssid: row for row, bssid in enumerate(source_vocabulary)}
    for target_column, bssid in enumerate(target_vocabulary):
        if bssid in old_rows:
            source_column = old_rows[bssid]
            migrated_encoder["0.weight"][:, target_column] = source_encoder_input[
                :, source_column
            ]
            migrated_decoder[decoder_output_weight_name][target_column, :] = (
                source_decoder_output[source_column, :]
            )
            migrated_decoder[decoder_output_bias_name][target_column] = (
                source_decoder_bias[source_column]
            )
        elif bssid in archive:
            slices = archive.pop(bssid)
            expected = {
                "encoder_input_weight": (source_manifest.model.dense_width,),
                "decoder_output_weight": (source_manifest.model.dense_width,),
                "decoder_output_bias": (),
            }
            for name, shape in expected.items():
                if name not in slices or tuple(slices[name].shape) != shape:
                    raise ValueError(
                        f"archived AE parameter slice for {bssid} is malformed: {name}"
                    )
            migrated_encoder["0.weight"][:, target_column] = slices[
                "encoder_input_weight"
            ].to(dtype=dtype)
            migrated_decoder[decoder_output_weight_name][target_column, :] = slices[
                "decoder_output_weight"
            ].to(dtype=dtype)
            migrated_decoder[decoder_output_bias_name][target_column] = slices[
                "decoder_output_bias"
            ].to(dtype=dtype)

    migrated.E.load_state_dict(migrated_encoder, strict=True)
    migrated.D.load_state_dict(migrated_decoder, strict=True)

    optimizer_state = copy.deepcopy(
        source_optimizer_state
        if source_optimizer_state is not None
        else source_payload["optimizer_state_dict"]
    )
    if not isinstance(optimizer_state, dict) or not isinstance(
        optimizer_state.get("param_groups"), list
    ):
        raise ValueError("source AE optimizer state is malformed")
    if len(optimizer_state["param_groups"]) != 1:
        raise ValueError(
            "AE vocabulary migration supports the recorded single Adam parameter group"
        )
    parameter_names = tuple(name for name, _parameter in migrated.named_parameters())
    resized_names = {
        "E.0.weight",
        f"D.{AE_DECODER_OUTPUT_LAYER_INDEX}.weight",
        f"D.{AE_DECODER_OUTPUT_LAYER_INDEX}.bias",
    }
    parameter_ids = tuple(optimizer_state["param_groups"][0]["params"])
    if len(parameter_ids) != len(parameter_names):
        raise ValueError(
            "source Adam parameter mapping does not match the AE architecture"
        )
    for parameter_id, parameter_name in zip(parameter_ids, parameter_names):
        if vocabulary_changed and parameter_name in resized_names:
            optimizer_state["state"].pop(parameter_id, None)

    migration = AEVocabularyMigrationRecord(
        source_vocabulary_sha256=source_manifest.vocabulary_sha256,
        target_vocabulary_sha256=canonical_json_digest(target_vocabulary),
        source_vocabulary=source_vocabulary,
        target_vocabulary=target_vocabulary,
        added_bssids=added,
        removed_bssids=removed,
        reactivated_bssids=reactivated,
        seed=seed,
        source_model_identity=source_model_identity,
    )
    manifest = source_manifest.model_copy(
        update={
            "model": source_manifest.model.model_copy(
                update={"input_dim": len(target_vocabulary)}
            ),
            "bssid_vocabulary": target_vocabulary,
            "vocabulary_sha256": migration.target_vocabulary_sha256,
            "initialization_mode": "vocabulary_migration",
            "source_checkpoint_sha256": sha256_file(source_path),
            "vocabulary_migration": migration,
            "created_at_utc": datetime.now(UTC).isoformat(),
        }
    )
    migrated_payload = {
        "format": CHECKPOINT_FORMAT,
        "encoder_state_dict": migrated.E.state_dict(),
        "decoder_state_dict": migrated.D.state_dict(),
        "optimizer_state_dict": optimizer_state,
        "random_state": source_payload["random_state"],
        "ap_parameter_archive": archive,
        "manifest": manifest.model_dump(mode="python"),
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_torch_save(migrated_payload, target_path, overwrite=overwrite)
    return load_ae_checkpoint(target_path)


def train_autoencoder(
    prepared: PreparedDataset,
    checkpoint_path: str | Path,
    config: AETrainingConfig | None = None,
    *,
    initialization: Literal["fresh", "weights_only", "resume"] = "fresh",
    source_checkpoint_path: str | Path | None = None,
    device: str | torch.device = "cpu",
    overwrite: bool = False,
    log_every_epochs: int = DEFAULT_LOG_EVERY_K_EPOCHS,
    progress_eval_scans: int = AE_PROGRESS_DIAGNOSTIC_SCANS,
) -> AETrainingResult:
    """Train on a prepared bootstrap dataset and persist complete resumable state.

    ``fresh`` initializes all weights. ``weights_only`` copies encoder/decoder
    weights from a compatible checkpoint but starts a new optimizer/progress.
    ``resume`` restores weights, Adam, epoch history, and CPU/CUDA/shuffle RNG
    state at an epoch boundary. Existing destinations are never replaced except
    by explicit ``overwrite=True`` or a validated resume operation.

    ``training.loss`` selects mean-reduced MSE or the paper's raw minibatch
    Frobenius norm. Each mode records its own explicit epoch-history reduction:
    element-weighted mean squared error, or arithmetic mean of the minibatch
    Frobenius norms. The latter retains batch-size-dependent scaling.
    On logged epochs, ``progress_eval_scans`` leading scans are evaluated with
    dropout disabled; their Frobenius norm and RMSE remain separate diagnostics.
    """
    training = config or AETrainingConfig()
    log_every_epochs = validate_log_interval(log_every_epochs)
    if type(progress_eval_scans) is not int or progress_eval_scans < 1:
        raise ValueError("progress_eval_scans must be a positive integer")
    values, scan_ids = _prepared_inputs(prepared, require_bootstrap=True)
    model_cfg = _model_config(prepared, training)
    dataset_identity = _dataset_identity(prepared, scan_ids)
    vocabulary = prepared.bssid_vocabulary
    vocabulary_hash = _vocab_hash(prepared)
    selected_device = _device(device)
    dtype = _torch_dtype(training.dtype)
    target_path = Path(checkpoint_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    source: Path | None = None
    source_payload: dict | None = None
    source_manifest: AECheckpointManifest | None = None
    source_digest: str | None = None
    if initialization == "fresh":
        if source_checkpoint_path is not None:
            raise ValueError("fresh initialization does not accept a source checkpoint")
    else:
        if source_checkpoint_path is None:
            raise ValueError(f"{initialization} requires source_checkpoint_path")
        source = Path(source_checkpoint_path)
        source_payload, source_manifest = _load_raw_checkpoint(
            source, map_location="cpu"
        )
        source_digest = sha256_file(source)
        _check_inference_compatibility(source_manifest, prepared)
        if source_manifest.model != model_cfg:
            if initialization != "weights_only" or source_manifest.model.model_dump(
                exclude={"dropout_probability"}
            ) != model_cfg.model_dump(exclude={"dropout_probability"}):
                raise ValueError("source checkpoint AE architecture does not match")
        if initialization == "resume":
            if source_manifest.training_dataset != dataset_identity:
                raise ValueError(
                    "resume requires the exact original prepared dataset identity"
                )
            if training.model_dump(
                exclude={"epochs"}
            ) != source_manifest.training_config.model_dump(exclude={"epochs"}):
                raise ValueError(
                    "resume settings must match the source checkpoint apart from target epochs"
                )
            if training.epochs <= source_manifest.epochs_completed:
                raise ValueError("resume target epochs must exceed completed epochs")
            if (
                not source_manifest.optimizer_state_present
                or not source_manifest.random_state_present
            ):
                raise ValueError("source checkpoint does not contain resumable state")
            if not source_manifest.training_resume_compatible:
                raise ValueError("source checkpoint is not a compatible generic AE resume boundary")
        elif initialization == "weights_only":
            if source_manifest.epochs_completed < 1:
                raise ValueError("weights_only source must contain trained weights")
        else:
            raise ValueError(f"unsupported initialization mode: {initialization}")

    if initialization == "resume":
        if source is None:
            raise AssertionError("validated resume must have a source path")
        # Resume is an explicit request to advance this compatible checkpoint.
        if (
            target_path.exists()
            and target_path.resolve() != source.resolve()
            and not overwrite
        ):
            raise FileExistsError(f"refusing to overwrite artifact: {target_path}")
    elif target_path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite artifact: {target_path}")

    cuda_device_index = (
        (
            selected_device.index
            if selected_device.index is not None
            else torch.cuda.current_device()
        )
        if selected_device.type == "cuda"
        else None
    )
    cuda_devices = [cuda_device_index] if cuda_device_index is not None else []
    history: list[float] = []
    start_epoch = 0
    rng_context = (
        torch.random.fork_rng(devices=cuda_devices, enabled=True)
        if selected_device.type == "cuda"
        else torch.random.fork_rng(devices=[], enabled=True)
    )
    with rng_context:
        torch.random.default_generator.manual_seed(training.seed)
        if selected_device.type == "cuda":
            with torch.cuda.device(selected_device):
                torch.cuda.manual_seed(training.seed)
        model = AE(model_cfg.input_dim, config=model_cfg).to(
            device=selected_device, dtype=dtype
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training.learning_rate,
            betas=training.optimizer_betas,
            eps=training.optimizer_epsilon,
            weight_decay=training.weight_decay,
            amsgrad=training.amsgrad,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(training.seed)
        if source_payload is not None and source_manifest is not None:
            model.E.load_state_dict(source_payload["encoder_state_dict"], strict=True)
            model.D.load_state_dict(source_payload["decoder_state_dict"], strict=True)
            if initialization == "resume":
                if source_manifest.device_used != str(selected_device):
                    raise ValueError(
                        "resume requires the source checkpoint's original device"
                    )
                optimizer.load_state_dict(source_payload["optimizer_state_dict"])
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if isinstance(value, torch.Tensor):
                            state[key] = value.to(selected_device)
                history.extend(source_manifest.loss_history)
                start_epoch = source_manifest.epochs_completed
                saved_rng = source_payload["random_state"]
                torch.set_rng_state(saved_rng["torch_cpu"].cpu())
                generator.set_state(saved_rng["shuffle_generator"].cpu())
                if selected_device.type == "cuda":
                    cuda_state = saved_rng.get("torch_cuda")
                    if not isinstance(cuda_state, torch.Tensor):
                        raise ValueError(
                            "CUDA resume checkpoint is missing CUDA RNG state"
                        )
                    torch.cuda.set_rng_state(cuda_state.cpu(), device=selected_device)
            # weights_only deliberately starts fresh optimizer/progress/RNG.

        model.train()
        for _epoch in range(start_epoch, training.epochs):
            if training.shuffle:
                order = torch.randperm(values.shape[0], generator=generator)
            else:
                order = torch.arange(values.shape[0])
            epoch_squared_error = 0.0
            epoch_value_count = 0
            batch_frobenius_losses: list[float] = []
            batch_count = 0
            for start in range(0, values.shape[0], training.batch_size):
                indices = order[start : start + training.batch_size]
                batch = (
                    values.index_select(0, indices)
                    .to(device=selected_device, dtype=dtype)
                    .unsqueeze(1)
                )
                if batch.ndim != 3 or batch.shape[1:] != (1, model_cfg.input_dim):
                    raise ValueError(
                        f"AE training batch shape is invalid: {tuple(batch.shape)}"
                    )
                optimizer.zero_grad(set_to_none=True)
                latent, reconstruction = model(batch)
                if (
                    latent.shape != (batch.shape[0], 1, model_cfg.latent_dim)
                    or reconstruction.shape != batch.shape
                ):
                    raise ValueError("AE forward output violates its shape contract")
                residual = reconstruction - batch
                if training.loss == "mean_squared_error":
                    loss = torch.nn.functional.mse_loss(
                        reconstruction, batch, reduction="mean"
                    )
                else:
                    loss = torch.linalg.vector_norm(residual)
                if not torch.isfinite(loss).item():
                    raise FloatingPointError("AE loss is non-finite")
                loss.backward()
                if any(
                    parameter.grad is not None
                    and not torch.isfinite(parameter.grad).all().item()
                    for parameter in model.parameters()
                ):
                    raise FloatingPointError("AE gradient is non-finite")
                optimizer.step()
                epoch_squared_error += float(
                    residual.detach().square().sum().cpu().item()
                )
                epoch_value_count += residual.numel()
                if training.loss == "batch_frobenius_norm":
                    batch_frobenius_losses.append(float(loss.detach().cpu().item()))
                batch_count += 1
            if not batch_count or not epoch_value_count:
                raise RuntimeError("AE epoch produced no batches")
            if training.loss == "mean_squared_error":
                epoch_loss = epoch_squared_error / epoch_value_count
            else:
                epoch_loss = sum(batch_frobenius_losses) / len(batch_frobenius_losses)
            history.append(epoch_loss)
            epoch_number = _epoch + 1
            epoch_metrics: dict[str, int | float] = {
                "learning_rate": optimizer.param_groups[0]["lr"]
            }
            if training.loss == "mean_squared_error":
                epoch_metrics["train_mse"] = history[-1]
                epoch_metrics["train_rmse"] = history[-1] ** 0.5
            else:
                epoch_metrics["train_batch_frobenius_mean"] = history[-1]
            if should_log_training_epoch(
                epoch_number, training.epochs, log_every_epochs
            ):
                _, eval_frobenius, eval_rmse = _eval_reconstruction_metrics(
                    model,
                    values,
                    scan_count=progress_eval_scans,
                    batch_size=64,
                )
                epoch_metrics.update(
                    {
                        "eval_frobenius_norm": eval_frobenius,
                        "eval_rmse": eval_rmse,
                    }
                )
            log_training_epoch(
                "AE",
                epoch_number,
                training.epochs,
                epoch_metrics,
                every_k_epochs=log_every_epochs,
            )

        cpu_rng = torch.get_rng_state().cpu().clone()
        shuffle_rng = generator.get_state().cpu().clone()
        cuda_rng = (
            torch.cuda.get_rng_state(selected_device).cpu().clone()
            if selected_device.type == "cuda"
            else None
        )
        payload = _state_payload(
            model,
            optimizer,
            cpu_rng_state=cpu_rng,
            shuffle_rng_state=shuffle_rng,
            cuda_rng_state=cuda_rng,
        )

    manifest = AECheckpointManifest(
        model=model_cfg,
        normalization=prepared.config.normalization_mode,
        bssid_vocabulary=vocabulary,
        vocabulary_sha256=vocabulary_hash,
        training_dataset=dataset_identity,
        training_config=AECheckpointTrainingConfig.model_validate(
            training.model_dump()
        ),
        initialization_mode=initialization,
        source_checkpoint_sha256=source_digest,
        epochs_completed=len(history),
        loss_history=tuple(history),
        loss_history_definition=(
            "element_weighted_batch_mse_per_epoch"
            if training.loss == "mean_squared_error"
            else "mean_of_batch_frobenius_norms_per_epoch"
        ),
        optimizer_state_present=True,
        random_state_present=True,
        device_used=str(selected_device),
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    payload["manifest"] = manifest.model_dump(mode="python")
    atomic_torch_save(
        payload, target_path, overwrite=(overwrite or initialization == "resume")
    )
    loaded = load_ae_checkpoint(
        target_path, prepared, device=selected_device, dtype=training.dtype
    )
    return AETrainingResult(
        checkpoint=loaded,
        loss_history=tuple(history),
        epochs_run=training.epochs - start_epoch,
    )


def encode_prepared_dataset(
    prepared: PreparedDataset,
    checkpoint_path: str | Path,
    feature_path: str | Path,
    *,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
    dtype: Literal["float32", "float64"] | None = None,
    overwrite: bool = False,
) -> EncoderFeatureArtifact:
    """Run a separate eval/inference pass over every retained scan and save IDs."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    values, scan_ids = _prepared_inputs(prepared)
    checkpoint = load_ae_checkpoint(
        checkpoint_path, prepared, device=device, dtype=dtype
    )
    selected_dtype = dtype or checkpoint.manifest.training_config.dtype
    model = checkpoint.model
    model.eval()
    rows: list[tuple[float, ...]] = []
    with torch.inference_mode():
        for start in range(0, len(scan_ids), batch_size):
            batch = (
                values[start : start + batch_size]
                .to(device=_device(device), dtype=_torch_dtype(selected_dtype))
                .unsqueeze(1)
            )
            if batch.ndim != 3 or batch.shape[1:] != (
                1,
                checkpoint.manifest.model.input_dim,
            ):
                raise ValueError(
                    f"AE inference batch shape is invalid: {tuple(batch.shape)}"
                )
            latent, reconstruction = model(batch)
            if (
                latent.shape
                != (batch.shape[0], 1, checkpoint.manifest.model.latent_dim)
                or reconstruction.shape != batch.shape
            ):
                raise ValueError("AE inference output violates its shape contract")
            rows.extend(
                tuple(float(value) for value in row)
                for row in latent[:, 0, :].cpu().tolist()
            )
    if len(rows) != len(scan_ids):
        raise AssertionError(
            "encoder export row count does not match retained scan count"
        )
    artifact = EncoderFeatureArtifact(
        dataset_id=prepared.config.dataset_id,
        batch_id=prepared.config.batch_id,
        source_prepared_sha256=canonical_json_digest(prepared),
        scan_ids=scan_ids,
        feature_matrix=tuple(rows),
        feature_dim=checkpoint.manifest.model.latent_dim,
        checkpoint_sha256=checkpoint.checkpoint_sha256,
        bssid_vocabulary=prepared.bssid_vocabulary,
        vocabulary_sha256=_vocab_hash(prepared),
        normalization=prepared.config.normalization_mode,
        dtype=selected_dtype,
        created_at_utc=datetime.now(UTC).isoformat(),
    )
    atomic_write_json(artifact, feature_path, overwrite=overwrite)
    return artifact


def load_encoder_features(path: str | Path) -> EncoderFeatureArtifact:
    """Load and validate a serialized encoder feature matrix."""
    return EncoderFeatureArtifact.model_validate_json(read_json_text(path))


def reconstruction_diagnostic(
    prepared: PreparedDataset,
    checkpoint_path: str | Path,
    *,
    max_scans: int = AE_PROGRESS_DIAGNOSTIC_SCANS,
    batch_size: int = 64,
    device: str | torch.device = "cpu",
) -> dict[str, int | float | str]:
    """Return a small eval-mode reconstruction summary without training state."""
    if type(max_scans) is not int or max_scans < 1:
        raise ValueError("max_scans must be a positive integer")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    values, _scan_ids = _prepared_inputs(prepared)
    checkpoint = load_ae_checkpoint(checkpoint_path, prepared, device=device)
    selected_count, frobenius, rmse = _eval_reconstruction_metrics(
        checkpoint.model,
        values,
        scan_count=max_scans,
        batch_size=batch_size,
    )
    return {
        "scan_count": selected_count,
        "input_dim": checkpoint.manifest.model.input_dim,
        "latent_dim": checkpoint.manifest.model.latent_dim,
        "frobenius_norm": frobenius,
        "rmse": rmse,
        "checkpoint_sha256": checkpoint.checkpoint_sha256,
    }


if __name__ == "__main__":
    raise SystemExit(
        "Use train_autoencoder with a milestone 1 PreparedDataset; raw-file training is unsupported."
    )
