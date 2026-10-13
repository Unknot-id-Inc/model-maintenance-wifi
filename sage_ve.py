"""Edge-feature GNN training and export for the milestone 3 ``GraphState``.

The implementation follows the node and physical-edge updates in GUFU
Algorithm 1. PyG's ``SAGEConv`` is intentionally not used: its forward method
does not take RSSI scalar weights in the third positional argument, and it has
no input for the learned edge vectors required by Algorithm 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, NamedTuple
from uuid import uuid4

import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from graph_build import GraphBuildConfig, GraphState
from gufu_artifacts import (
    atomic_torch_save,
    atomic_write_json,
    canonical_json_digest,
    load_torch_state,
    read_json_text,
    sha256_file,
)
from training_progress import (
    DEFAULT_LOG_EVERY_K_EPOCHS,
    log_training_epoch,
    validate_log_interval,
)

GNN_ARCHITECTURE_ID = "gufu_algorithm1_edge_feature_mean_v1"
GNN_CHECKPOINT_FORMAT = "gufu_gnn_checkpoint_v1"
GNN_FEATURE_FORMAT = "gufu_gnn_features_v1"
DEFAULT_GNN_DIM = 34


class GNNTrainingError(ValueError):
    """A graph or objective cannot be trained under the declared GNN policy."""


class GNNModelConfig(BaseModel):
    """Algorithm 1 widths and named choices absent from its pseudocode."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    architecture_id: Literal["gufu_algorithm1_edge_feature_mean_v1"] = (
        GNN_ARCHITECTURE_ID
    )
    graph_feature_dim: int = Field(default=DEFAULT_GNN_DIM, ge=1)
    edge_feature_dim: int = Field(default=DEFAULT_GNN_DIM, ge=1)
    node_output_dim: int = Field(default=DEFAULT_GNN_DIM, ge=1)
    edge_output_dim: int = Field(default=DEFAULT_GNN_DIM, ge=1)
    dtype: Literal["float32"] = "float32"
    num_layers: int = Field(default=2, ge=1)
    dropout_probability: float = Field(default=0.5, ge=0.0, lt=1.0)
    initialization_seed: int = 0
    aggregation: Literal["mean_edge_neighbor_message_v1"] = (
        "mean_edge_neighbor_message_v1"
    )
    activation: Literal["relu"] = "relu"
    edge_update_scope: Literal["physical_radio_edges_only"] = (
        "physical_radio_edges_only"
    )
    physical_edge_endpoint_order: Literal["sample_then_ap"] = "sample_then_ap"
    virtual_edge_update_policy: Literal["retain_initial_mean_then_remove"] = (
        "retain_initial_mean_then_remove"
    )
    edge_scalar_weight_usage: Literal[
        "ap_initialization_only_not_algorithm1_or_eq4"
    ] = "ap_initialization_only_not_algorithm1_or_eq4"
    optimizer: Literal["Adam"] = "Adam"
    dropout_placement: Literal[
        "after_relu_before_mean_or_l2_normalization_inferred"
    ] = "after_relu_before_mean_or_l2_normalization_inferred"
    zero_vector_policy: Literal["l2_normalize_eps_preserve_zero"] = (
        "l2_normalize_eps_preserve_zero"
    )
    layer_width_policy: Literal[
        "dimension_preserving_34d_inference",
        "dimension_preserving_graph_feature_width_v1",
    ] = "dimension_preserving_graph_feature_width_v1"

    @model_validator(mode="after")
    def validate_dimension_contract(self):
        dimensions = (
            self.graph_feature_dim,
            self.edge_feature_dim,
            self.node_output_dim,
            self.edge_output_dim,
        )
        if len(set(dimensions)) != 1:
            raise ValueError(
                "this GraphSAGE variant requires equal node/edge input and output widths"
            )
        return self


class GNNTrainingConfig(BaseModel):
    """Single full-graph independent experiment; no implicit batch resume."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    epochs: int = Field(default=50, ge=1)
    learning_rate: float = Field(default=0.01, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    seed: int = 0
    optimizer: Literal["Adam"] = "Adam"
    objective: Literal["eq4_relu_log_sum_clamped_v1"] = "eq4_relu_log_sum_clamped_v1"
    objective_epsilon: float = Field(default=1.0e-12, gt=0.0)
    objective_reduction: Literal["sum_over_each_undirected_edge_once"] = (
        "sum_over_each_undirected_edge_once"
    )
    selection_policy: Literal["final_training_epoch"] = "final_training_epoch"
    run_mode: Literal["independent_experiment_no_resume"] = (
        "independent_experiment_no_resume"
    )

    @model_validator(mode="after")
    def finite_scalars(self):
        if not torch.isfinite(torch.tensor(self.learning_rate)).item():
            raise ValueError("learning_rate must be finite")
        if not torch.isfinite(torch.tensor(self.weight_decay)).item():
            raise ValueError("weight_decay must be finite")
        if not torch.isfinite(torch.tensor(self.objective_epsilon)).item():
            raise ValueError("objective_epsilon must be finite")
        return self


class GNNCheckpointManifest(BaseModel):
    """Strict provenance stored beside tensor-only model weights."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_gnn_checkpoint_v1"] = GNN_CHECKPOINT_FORMAT
    architecture_id: Literal["gufu_algorithm1_edge_feature_mean_v1"] = (
        GNN_ARCHITECTURE_ID
    )
    model_config_record: GNNModelConfig
    training_config: GNNTrainingConfig
    run_id: str
    dataset_id: str
    batch_id: str
    batch_role: Literal["bootstrap_reference"]
    normalization: Literal["paper_ae", "legacy_raw_div_120"]
    graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    graph_build_config: GraphBuildConfig
    source_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_feature_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bssid_vocabulary: tuple[str, ...]
    node_ids: tuple[str, ...]
    node_types: tuple[Literal["sample", "ap"], ...]
    edge_topology_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    radio_edge_ids: tuple[str, ...]
    virtual_edge_ids: tuple[str, ...]
    loss_history: tuple[float, ...]
    epochs_completed: int = Field(ge=1)
    selected_epoch: int = Field(ge=1)
    final_epoch_selected: Literal[True] = True
    continuation_mode: Literal[
        "independent_graph_training", "weights_only_transition_retarget"
    ] = "independent_graph_training"
    source_checkpoint_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    transition_state_id: str | None = Field(
        default=None, pattern=r"^transition:[0-9a-f]{64}$"
    )
    resumable: Literal[False] = False
    training_device: str
    torch_version: str
    created_at_utc: str

    @model_validator(mode="after")
    def validate_history(self):
        if self.epochs_completed != len(self.loss_history):
            raise ValueError(
                "checkpoint loss history must contain one scalar per epoch"
            )
        if self.selected_epoch != self.epochs_completed:
            raise ValueError("this milestone selects the declared final training epoch")
        if any(
            not torch.isfinite(torch.tensor(value)).item()
            for value in self.loss_history
        ):
            raise ValueError("checkpoint loss history must contain finite scalars")
        if len(self.node_ids) != len(self.node_types):
            raise ValueError("node IDs/types in checkpoint must align")
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("checkpoint node IDs must be unique")
        if len(set(self.radio_edge_ids + self.virtual_edge_ids)) != len(
            self.radio_edge_ids + self.virtual_edge_ids
        ):
            raise ValueError("checkpoint edge IDs must be unique")
        if self.continuation_mode == "weights_only_transition_retarget":
            if (
                self.source_checkpoint_sha256 is None
                or self.transition_state_id is None
            ):
                raise ValueError(
                    "retargeted GNN checkpoint must bind its source and transition state"
                )
        elif (
            self.source_checkpoint_sha256 is not None
            or self.transition_state_id is not None
        ):
            raise ValueError(
                "only retargeted GNN checkpoints may record transition lineage"
            )
        return self


class GNNFeatureArtifact(BaseModel):
    """Final eval-mode node and physical-edge features, keyed by stable IDs."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_gnn_features_v1"] = GNN_FEATURE_FORMAT
    feature_space: Literal["z_gnn_node"] = "z_gnn_node"
    edge_feature_space: Literal["z_gnn_physical_edge"] = "z_gnn_physical_edge"
    graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    graph_build_config: GraphBuildConfig
    dataset_id: str
    batch_id: str
    source_prepared_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_feature_artifact_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoder_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalization: Literal["paper_ae", "legacy_raw_div_120"]
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bssid_vocabulary: tuple[str, ...]
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_config_record: GNNModelConfig
    node_ids: tuple[str, ...]
    node_types: tuple[Literal["sample", "ap"], ...]
    node_id_to_row: tuple[tuple[str, int], ...]
    node_feature_dim: int = Field(default=DEFAULT_GNN_DIM, ge=1)
    node_feature_matrix: tuple[tuple[float, ...], ...]
    radio_edge_ids: tuple[str, ...]
    radio_edge_feature_dim: int = Field(default=DEFAULT_GNN_DIM, ge=1)
    radio_edge_feature_matrix: tuple[tuple[float, ...], ...]
    virtual_edge_ids: tuple[str, ...]
    virtual_edge_feature_dim: int = Field(default=DEFAULT_GNN_DIM, ge=1)
    virtual_edge_feature_matrix_initial: tuple[tuple[float, ...], ...]
    virtual_edges_removed_after_aggregation: Literal[True] = True
    inference_mode: Literal["model_eval_torch_inference_mode"] = (
        "model_eval_torch_inference_mode"
    )
    created_at_utc: str

    @model_validator(mode="after")
    def validate_feature_rows(self):
        count = len(self.node_ids)
        if len(set(self.node_ids)) != count:
            raise ValueError("exported node IDs must be unique")
        if len(self.node_types) != count:
            raise ValueError("exported node types must align with node IDs")
        if self.node_id_to_row != tuple(
            (node_id, row) for row, node_id in enumerate(self.node_ids)
        ):
            raise ValueError(
                "exported node mapping must exactly follow feature row order"
            )
        if self.node_feature_dim != self.model_config_record.node_output_dim:
            raise ValueError(
                "GNN node output width does not match its model configuration"
            )
        if (
            self.radio_edge_feature_dim != self.model_config_record.edge_output_dim
            or self.virtual_edge_feature_dim != self.model_config_record.edge_output_dim
        ):
            raise ValueError(
                "GNN edge output widths do not match its model configuration"
            )
        if len(self.node_feature_matrix) != count or any(
            len(row) != self.node_feature_dim for row in self.node_feature_matrix
        ):
            raise ValueError(
                "node feature matrix must match its declared feature width"
            )
        if len(self.radio_edge_ids) != len(self.radio_edge_feature_matrix) or any(
            len(row) != self.radio_edge_feature_dim
            for row in self.radio_edge_feature_matrix
        ):
            raise ValueError("physical edge features must align with radio edge IDs")
        if len(self.virtual_edge_ids) != len(
            self.virtual_edge_feature_matrix_initial
        ) or any(
            len(row) != self.virtual_edge_feature_dim
            for row in self.virtual_edge_feature_matrix_initial
        ):
            raise ValueError(
                "initial virtual edge features must align with virtual edge IDs"
            )
        if (
            self.bssid_vocabulary
            and canonical_json_digest(self.bssid_vocabulary) != self.vocabulary_sha256
        ):
            raise ValueError(
                "exported vocabulary hash does not match its ordered vocabulary"
            )
        for matrix in (
            self.node_feature_matrix,
            self.radio_edge_feature_matrix,
            self.virtual_edge_feature_matrix_initial,
        ):
            if any(
                not torch.isfinite(torch.tensor(value)).item()
                for row in matrix
                for value in row
            ):
                raise ValueError("exported graph features must be finite")
        return self


class GraphTensorInput(NamedTuple):
    """Tensor view; RSS scalar weights are intentionally not model arguments."""

    node_features: Tensor
    edge_features: Tensor
    edge_left_rows: Tensor
    edge_right_rows: Tensor
    radio_edge_count: int
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    radio_edge_ids: tuple[str, ...]
    virtual_edge_ids: tuple[str, ...]


class GNNOutput(NamedTuple):
    node_features: Tensor
    radio_edge_features: Tensor
    virtual_edge_features_initial: Tensor


@dataclass(frozen=True)
class LoadedGNNCheckpoint:
    model: EdgeFeatureSAGE
    manifest: GNNCheckpointManifest
    checkpoint_path: Path
    checkpoint_sha256: str


@dataclass(frozen=True)
class GNNTrainingResult:
    checkpoint: LoadedGNNCheckpoint
    features: GNNFeatureArtifact
    loss_history: tuple[float, ...]


class EdgeFeatureSAGE(nn.Module):
    """Algorithm 1 node/edge updates with an explicit edge-vector message."""

    def __init__(self, config: GNNModelConfig | None = None):
        super().__init__()
        self.config = config or GNNModelConfig()
        if not isinstance(self.config, GNNModelConfig):
            raise TypeError("config must be a validated GNNModelConfig")
        self.feature_dim = self.config.graph_feature_dim
        self.W0 = nn.ParameterList()
        self.W1 = nn.ParameterList()
        self.W2 = nn.ParameterList()
        # Use a local RNG fork so creating a model does not perturb caller state.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.initialization_seed)
            for _ in range(self.config.num_layers):
                w0 = nn.Parameter(torch.empty(self.feature_dim, 2 * self.feature_dim))
                w1 = nn.Parameter(torch.empty(self.feature_dim, 2 * self.feature_dim))
                w2 = nn.Parameter(torch.empty(self.feature_dim, 3 * self.feature_dim))
                nn.init.xavier_uniform_(w0)
                nn.init.xavier_uniform_(w1)
                nn.init.xavier_uniform_(w2)
                self.W0.append(w0)
                self.W1.append(w1)
                self.W2.append(w2)
        self._inference_only = False
        self._training_started = False
        self._training_completed = False

    def forward(self, graph: GraphState | GraphTensorInput) -> GNNOutput:
        parameter = self.W0[0]
        tensors = (
            graph
            if isinstance(graph, GraphTensorInput)
            else graph_state_to_tensors(
                graph, device=parameter.device, dtype=parameter.dtype
            )
        )
        if tensors.node_features.shape[1] != self.feature_dim:
            raise ValueError(f"GNN requires {self.feature_dim}-D graph node features")
        if tensors.edge_features.shape[1] != self.feature_dim:
            raise ValueError(
                f"GNN requires {self.feature_dim}-D initialized edge vectors"
            )
        if (
            tensors.node_features.dtype != parameter.dtype
            or tensors.edge_features.dtype != parameter.dtype
        ):
            raise ValueError(
                f"GNN inputs must use the model dtype {parameter.dtype}; "
                f"got nodes={tensors.node_features.dtype}, edges={tensors.edge_features.dtype}"
            )
        if (
            tensors.node_features.device != parameter.device
            or tensors.edge_features.device != parameter.device
            or tensors.edge_left_rows.device != parameter.device
            or tensors.edge_right_rows.device != parameter.device
        ):
            raise ValueError(
                "GNN input tensors and model parameters must share one device"
            )
        edge_count = tensors.edge_features.shape[0]
        if tensors.node_features.shape[0] != len(tensors.node_ids):
            raise ValueError("GNN node tensor rows must align with node IDs")
        if edge_count != len(tensors.edge_ids):
            raise ValueError("GNN edge tensor rows must align with edge IDs")
        if (
            tensors.edge_left_rows.shape != (edge_count,)
            or tensors.edge_right_rows.shape != (edge_count,)
            or not 0 <= tensors.radio_edge_count <= edge_count
            or tensors.radio_edge_ids != tensors.edge_ids[: tensors.radio_edge_count]
            or tensors.virtual_edge_ids != tensors.edge_ids[tensors.radio_edge_count :]
        ):
            raise ValueError("GNN edge endpoints/types do not align with edge IDs")

        node_features = tensors.node_features
        edge_features = tensors.edge_features
        if edge_count:
            message_edge_rows = torch.cat(
                (
                    torch.arange(edge_count, device=edge_features.device),
                    torch.arange(edge_count, device=edge_features.device),
                )
            )
            message_source_rows = torch.cat(
                (tensors.edge_right_rows, tensors.edge_left_rows)
            )
            message_target_rows = torch.cat(
                (tensors.edge_left_rows, tensors.edge_right_rows)
            )
        else:
            message_edge_rows = torch.empty(
                (0,), device=edge_features.device, dtype=torch.long
            )
            message_source_rows = message_target_rows = message_edge_rows

        for layer in range(self.config.num_layers):
            if edge_count:
                message_inputs = torch.cat(
                    (
                        edge_features[message_edge_rows],
                        node_features[message_source_rows],
                    ),
                    dim=1,
                )
                messages = F.leaky_relu(
                    F.linear(message_inputs, self.W0[layer]), negative_slope=0.1
                )
                messages = F.dropout(
                    messages,
                    p=self.config.dropout_probability,
                    training=self.training,
                )
                aggregate = node_features.new_zeros(
                    (node_features.shape[0], self.feature_dim)
                )
                aggregate.index_add_(0, message_target_rows, messages)
                degree = torch.bincount(
                    message_target_rows, minlength=node_features.shape[0]
                ).to(dtype=node_features.dtype)
                degree = degree.clamp_min(1.0).unsqueeze(1)
                aggregate = aggregate / degree
            else:
                # Isolated nodes have an explicit zero-neighbor aggregate.
                aggregate = node_features.new_zeros(
                    (node_features.shape[0], self.feature_dim)
                )

            node_inputs = torch.cat((node_features, aggregate), dim=1)
            node_features = F.relu(F.linear(node_inputs, self.W1[layer]))
            node_features = F.dropout(
                node_features,
                p=self.config.dropout_probability,
                training=self.training,
            )
            node_features = F.normalize(node_features, p=2.0, dim=1, eps=1e-12)

            # Algorithm 1 updates physical radio edges only. Virtual vectors
            # stay at their initial endpoint mean for the temporary pass.
            radio_count = tensors.radio_edge_count
            if radio_count:
                radio_left = tensors.edge_left_rows[:radio_count]
                radio_right = tensors.edge_right_rows[:radio_count]
                updated_radio = F.relu(
                    F.linear(
                        torch.cat(
                            (
                                node_features[radio_left],
                                edge_features[:radio_count],
                                node_features[radio_right],
                            ),
                            dim=1,
                        ),
                        self.W2[layer],
                    )
                )
                updated_radio = F.dropout(
                    updated_radio,
                    p=self.config.dropout_probability,
                    training=self.training,
                )
                updated_radio = F.normalize(updated_radio, p=2.0, dim=1, eps=1e-12)
                edge_features = torch.cat(
                    (updated_radio, edge_features[radio_count:]), dim=0
                )

        return GNNOutput(
            node_features=node_features,
            radio_edge_features=edge_features[: tensors.radio_edge_count],
            virtual_edge_features_initial=edge_features[tensors.radio_edge_count :],
        )


def graph_state_to_tensors(
    graph_state: GraphState,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> GraphTensorInput:
    """Convert the typed graph to tensors with stable ID-to-row joins.

    Physical RSS scalar weights and virtual ``None`` weights remain graph
    metadata. Algorithm 1 receives the graph's vector-valued edge features,
    whose width is encoder_feature_dim + 2, not those scalar weights.
    """
    feature_dim = graph_state.node_feature_dim
    if graph_state.node_feature_dim < 1:
        raise ValueError("graph feature width must be positive")
    row_by_id = dict(graph_state.node_id_to_row)
    radio_edges = graph_state.radio_edges
    virtual_edges = graph_state.virtual_edges
    edge_ids = tuple(edge.edge_id for edge in radio_edges) + tuple(
        edge.edge_id for edge in virtual_edges
    )
    edge_vectors = tuple(edge.edge_feature for edge in radio_edges) + tuple(
        edge.edge_feature for edge in virtual_edges
    )
    left_ids = tuple(edge.sample_node_id for edge in radio_edges) + tuple(
        edge.sample_node_id_left for edge in virtual_edges
    )
    right_ids = tuple(edge.ap_node_id for edge in radio_edges) + tuple(
        edge.sample_node_id_right for edge in virtual_edges
    )
    return GraphTensorInput(
        node_features=torch.tensor(
            graph_state.node_features, dtype=dtype, device=device
        ).reshape((-1, feature_dim)),
        edge_features=torch.tensor(edge_vectors, dtype=dtype, device=device).reshape(
            (-1, feature_dim)
        ),
        edge_left_rows=torch.tensor(
            [row_by_id[node_id] for node_id in left_ids],
            dtype=torch.long,
            device=device,
        ),
        edge_right_rows=torch.tensor(
            [row_by_id[node_id] for node_id in right_ids],
            dtype=torch.long,
            device=device,
        ),
        radio_edge_count=len(radio_edges),
        node_ids=graph_state.node_ids,
        edge_ids=edge_ids,
        radio_edge_ids=tuple(edge.edge_id for edge in radio_edges),
        virtual_edge_ids=tuple(edge.edge_id for edge in virtual_edges),
    )


def equation4_loss(
    node_features: Tensor,
    graph_state: GraphState,
    *,
    epsilon: float = 1.0e-12,
) -> Tensor:
    """Eq. 4: ``-sum(log(ReLU(z_u dot z_v)))`` over unique graph edges.

    A lower clamp at ``epsilon`` is the named numerical inference needed to
    keep the paper's ReLU/log expression finite at zero. It adds no pairs or
    alternate activation; entries at/below zero have zero ReLU gradient.
    Canonical undirected radio and virtual edge records each contribute once.
    """
    if not torch.isfinite(torch.tensor(epsilon)).item() or epsilon <= 0.0:
        raise ValueError("Eq. 4 epsilon must be finite and positive")
    if not graph_state.radio_edges and not graph_state.virtual_edges:
        raise GNNTrainingError(
            "Eq. 4 has no physical or virtual edges; the graph has no training relations"
        )
    if node_features.ndim != 2 or node_features.shape != (
        len(graph_state.node_ids),
        graph_state.node_feature_dim,
    ):
        raise ValueError(
            "Eq. 4 node embedding width must match the graph feature width"
        )
    row_by_id = dict(graph_state.node_id_to_row)
    left_ids = [edge.sample_node_id for edge in graph_state.radio_edges]
    right_ids = [edge.ap_node_id for edge in graph_state.radio_edges]
    left_ids.extend(edge.sample_node_id_left for edge in graph_state.virtual_edges)
    right_ids.extend(edge.sample_node_id_right for edge in graph_state.virtual_edges)
    left_rows = torch.tensor(
        [row_by_id[node_id] for node_id in left_ids],
        dtype=torch.long,
        device=node_features.device,
    )
    right_rows = torch.tensor(
        [row_by_id[node_id] for node_id in right_ids],
        dtype=torch.long,
        device=node_features.device,
    )
    dot_products = (node_features[left_rows] * node_features[right_rows]).sum(dim=1)
    activated = F.relu(dot_products)
    return -torch.log(activated.clamp_min(epsilon)).sum()


def _edge_topology_digest(graph_state: GraphState) -> str:
    return canonical_json_digest(
        {
            "radio": tuple(
                (edge.edge_id, edge.sample_node_id, edge.ap_node_id)
                for edge in graph_state.radio_edges
            ),
            "virtual": tuple(
                (edge.edge_id, edge.sample_node_id_left, edge.sample_node_id_right)
                for edge in graph_state.virtual_edges
            ),
        }
    )


def _validate_checkpoint_graph(
    manifest: GNNCheckpointManifest, graph_state: GraphState
) -> None:
    checks = (
        (manifest.dataset_id == graph_state.dataset_id, "dataset identity"),
        (
            manifest.model_config_record.graph_feature_dim
            == graph_state.node_feature_dim,
            "graph feature width",
        ),
        (
            manifest.model_config_record.edge_feature_dim
            == graph_state.node_feature_dim,
            "edge feature width",
        ),
        (manifest.normalization == graph_state.normalization, "normalization"),
        (
            manifest.encoder_checkpoint_sha256 == graph_state.encoder_checkpoint_sha256,
            "encoder checkpoint",
        ),
        (
            manifest.graph_build_config == graph_state.build_config,
            "graph configuration",
        ),
        (manifest.vocabulary_sha256 == graph_state.vocabulary_sha256, "vocabulary"),
        (manifest.bssid_vocabulary == graph_state.bssid_vocabulary, "vocabulary order"),
    )
    for matches, label in checks:
        if not matches:
            raise ValueError(f"GNN checkpoint is incompatible with graph {label}")
    if graph_state.graph_id == manifest.graph_id:
        exact_checks = (
            (manifest.batch_id == graph_state.batch_id, "batch identity"),
            (
                manifest.source_prepared_sha256 == graph_state.source_prepared_sha256,
                "prepared data",
            ),
            (
                manifest.encoder_feature_artifact_sha256
                == graph_state.encoder_feature_artifact_sha256,
                "encoder feature artifact",
            ),
            (manifest.node_ids == graph_state.node_ids, "node IDs/order"),
            (manifest.node_types == graph_state.node_types, "node types"),
            (
                manifest.edge_topology_sha256 == _edge_topology_digest(graph_state),
                "edge topology",
            ),
            (
                manifest.radio_edge_ids
                == tuple(edge.edge_id for edge in graph_state.radio_edges),
                "physical edge IDs/order",
            ),
            (
                manifest.virtual_edge_ids
                == tuple(edge.edge_id for edge in graph_state.virtual_edges),
                "virtual edge IDs/order",
            ),
        )
        for matches, label in exact_checks:
            if not matches:
                raise ValueError(f"GNN checkpoint is incompatible with graph {label}")
    # A non-identical graph is valid for inference when the complete model
    # input contract above still matches. Its own graph/batch identity is
    # written into the feature artifact; the checkpoint retains its training
    # graph identity. No parameters or optimizer state are continued here.


def _manifest_for_graph(
    graph_state: GraphState,
    model_config: GNNModelConfig,
    training_config: GNNTrainingConfig,
    loss_history: tuple[float, ...],
    training_device: torch.device,
) -> GNNCheckpointManifest:
    return GNNCheckpointManifest(
        model_config_record=model_config,
        training_config=training_config,
        run_id=uuid4().hex,
        dataset_id=graph_state.dataset_id,
        batch_id=graph_state.batch_id,
        batch_role="bootstrap_reference",
        normalization=graph_state.normalization,
        graph_id=graph_state.graph_id,
        graph_build_config=graph_state.build_config,
        source_prepared_sha256=graph_state.source_prepared_sha256,
        encoder_feature_artifact_sha256=graph_state.encoder_feature_artifact_sha256,
        encoder_checkpoint_sha256=graph_state.encoder_checkpoint_sha256,
        vocabulary_sha256=graph_state.vocabulary_sha256,
        bssid_vocabulary=graph_state.bssid_vocabulary,
        node_ids=graph_state.node_ids,
        node_types=graph_state.node_types,
        edge_topology_sha256=_edge_topology_digest(graph_state),
        radio_edge_ids=tuple(edge.edge_id for edge in graph_state.radio_edges),
        virtual_edge_ids=tuple(edge.edge_id for edge in graph_state.virtual_edges),
        loss_history=loss_history,
        epochs_completed=len(loss_history),
        selected_epoch=len(loss_history),
        training_device=str(training_device),
        torch_version=torch.__version__,
        created_at_utc=datetime.now(UTC).isoformat(),
    )


def _save_checkpoint(
    model: EdgeFeatureSAGE,
    manifest: GNNCheckpointManifest,
    path: str | Path,
    *,
    overwrite: bool,
) -> Path:
    state_dict = {
        name: tensor.detach().to(device="cpu").contiguous()
        for name, tensor in model.state_dict().items()
    }
    return atomic_torch_save(
        {
            "format": GNN_CHECKPOINT_FORMAT,
            "manifest_json": manifest.model_dump_json(),
            "model_state": state_dict,
        },
        path,
        overwrite=overwrite,
    )


def retarget_gnn_checkpoint(
    model: EdgeFeatureSAGE,
    graph_state: GraphState,
    training_config: GNNTrainingConfig,
    loss_history: tuple[float, ...],
    checkpoint_path: str | Path,
    *,
    source_checkpoint_sha256: str,
    transition_state_id: str,
    overwrite: bool = False,
) -> LoadedGNNCheckpoint:
    """Bind learned GNN weights to a migrated AE/graph identity without retraining.

    The configured GNN feature width is independent of AP vocabulary resizing. This
    explicit retarget step preserves its selected weights while binding the
    checkpoint to the new encoder artifact, graph ordering and transition.
    The source updater checkpoint remains the provenance source for weights.
    """
    if graph_state.batch_role != "bootstrap_reference":
        raise ValueError(
            "retargeted GNN state must describe the current reference graph"
        )
    if not isinstance(model, EdgeFeatureSAGE):
        raise TypeError("retargeted model must be an EdgeFeatureSAGE")
    if not source_checkpoint_sha256 or not transition_state_id.startswith(
        "transition:"
    ):
        raise ValueError(
            "retargeted GNN checkpoint requires source and transition identities"
        )
    if not loss_history or len(loss_history) != training_config.epochs:
        raise ValueError(
            "retargeted GNN must preserve the source GNN loss history/configuration"
        )
    manifest = _manifest_for_graph(
        graph_state,
        model.config,
        training_config,
        loss_history,
        torch.device("cpu"),
    ).model_copy(
        update={
            "continuation_mode": "weights_only_transition_retarget",
            "source_checkpoint_sha256": source_checkpoint_sha256,
            "transition_state_id": transition_state_id,
        }
    )
    path = Path(checkpoint_path)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite retargeted GNN checkpoint: {path}"
        )
    _save_checkpoint(model, manifest, path, overwrite=overwrite)
    return load_gnn_checkpoint(path, graph_state)


def load_gnn_checkpoint(
    checkpoint_path: str | Path,
    graph_state: GraphState,
    *,
    device: torch.device | str = "cpu",
) -> LoadedGNNCheckpoint:
    """Load weights-only inference state after exact graph compatibility checks."""
    path = Path(checkpoint_path)
    payload = load_torch_state(path, map_location="cpu")
    if payload.get("format") != GNN_CHECKPOINT_FORMAT:
        raise ValueError("unsupported GNN checkpoint format")
    manifest_json = payload.get("manifest_json")
    if not isinstance(manifest_json, str):
        raise ValueError("GNN checkpoint is missing its strict manifest")
    manifest = GNNCheckpointManifest.model_validate_json(manifest_json)
    _validate_checkpoint_graph(manifest, graph_state)
    model = EdgeFeatureSAGE(manifest.model_config_record)
    state_dict = payload.get("model_state")
    if not isinstance(state_dict, dict):
        raise ValueError("GNN checkpoint is missing model weights")
    if any(
        not isinstance(value, Tensor) or value.dtype != torch.float32
        for value in state_dict.values()
    ):
        raise ValueError(
            "GNN checkpoint weights must all use the recorded float32 dtype"
        )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    model._inference_only = True
    model._training_completed = True
    return LoadedGNNCheckpoint(
        model=model,
        manifest=manifest,
        checkpoint_path=path,
        checkpoint_sha256=sha256_file(path),
    )


def load_streaming_gnn_checkpoint(
    checkpoint_path: str | Path,
    streaming_lp_input_directory: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> LoadedGNNCheckpoint:
    """Load a bootstrap GNN using its compact LP bundle as graph identity.

    This avoids reconstructing the large in-memory ``GraphState`` solely to
    validate a checkpoint before the streamed sequential driver begins.
    """
    from lp_streaming import load_streaming_lp_input

    lp_manifest, _mapped = load_streaming_lp_input(streaming_lp_input_directory)
    path = Path(checkpoint_path)
    payload = load_torch_state(path, map_location="cpu")
    if payload.get("format") != GNN_CHECKPOINT_FORMAT:
        raise ValueError("unsupported GNN checkpoint format")
    manifest_json = payload.get("manifest_json")
    if not isinstance(manifest_json, str):
        raise ValueError("GNN checkpoint is missing its strict manifest")
    manifest = GNNCheckpointManifest.model_validate_json(manifest_json)
    digest = sha256_file(path)
    if (
        lp_manifest.batch_role != "bootstrap_reference"
        or manifest.graph_id != lp_manifest.graph_id
        or manifest.dataset_id != lp_manifest.dataset_id
        or manifest.batch_id != lp_manifest.batch_id
        or manifest.normalization != lp_manifest.normalization
        or manifest.bssid_vocabulary != lp_manifest.bssid_vocabulary
        or manifest.vocabulary_sha256 != lp_manifest.vocabulary_sha256
        or manifest.encoder_checkpoint_sha256 != lp_manifest.encoder_checkpoint_sha256
        or manifest.node_ids != lp_manifest.node_ids
        or manifest.model_config_record.graph_feature_dim != lp_manifest.node_feature_dim
        or manifest.graph_build_config != lp_manifest.graph_build_config
        or digest != lp_manifest.gnn_checkpoint_sha256
    ):
        raise ValueError("GNN checkpoint is incompatible with compact streaming LP identity")
    state_dict = payload.get("model_state")
    if not isinstance(state_dict, dict):
        raise ValueError("GNN checkpoint is missing model weights")
    if any(
        not isinstance(value, Tensor) or value.dtype != torch.float32
        for value in state_dict.values()
    ):
        raise ValueError("GNN checkpoint weights must use float32")
    model = EdgeFeatureSAGE(manifest.model_config_record)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    model._inference_only = True
    model._training_completed = True
    return LoadedGNNCheckpoint(
        model=model,
        manifest=manifest,
        checkpoint_path=path,
        checkpoint_sha256=digest,
    )


def _feature_artifact(
    graph_state: GraphState,
    output: GNNOutput,
    model_config: GNNModelConfig,
    checkpoint_sha256: str,
) -> GNNFeatureArtifact:
    def rows(tensor: Tensor) -> tuple[tuple[float, ...], ...]:
        return tuple(
            tuple(float(value) for value in row)
            for row in tensor.detach().to(device="cpu", dtype=torch.float64).tolist()
        )

    return GNNFeatureArtifact(
        graph_id=graph_state.graph_id,
        graph_build_config=graph_state.build_config,
        dataset_id=graph_state.dataset_id,
        batch_id=graph_state.batch_id,
        source_prepared_sha256=graph_state.source_prepared_sha256,
        encoder_feature_artifact_sha256=graph_state.encoder_feature_artifact_sha256,
        encoder_checkpoint_sha256=graph_state.encoder_checkpoint_sha256,
        normalization=graph_state.normalization,
        vocabulary_sha256=graph_state.vocabulary_sha256,
        bssid_vocabulary=graph_state.bssid_vocabulary,
        checkpoint_sha256=checkpoint_sha256,
        model_config_record=model_config,
        node_ids=graph_state.node_ids,
        node_types=graph_state.node_types,
        node_id_to_row=graph_state.node_id_to_row,
        node_feature_dim=graph_state.node_feature_dim,
        node_feature_matrix=rows(output.node_features),
        radio_edge_ids=tuple(edge.edge_id for edge in graph_state.radio_edges),
        radio_edge_feature_dim=graph_state.node_feature_dim,
        radio_edge_feature_matrix=rows(output.radio_edge_features),
        virtual_edge_ids=tuple(edge.edge_id for edge in graph_state.virtual_edges),
        virtual_edge_feature_dim=graph_state.node_feature_dim,
        virtual_edge_feature_matrix_initial=rows(output.virtual_edge_features_initial),
        created_at_utc=datetime.now(UTC).isoformat(),
    )


def encode_graph_state(
    graph_state: GraphState,
    checkpoint_path: str | Path,
    feature_path: str | Path,
    *,
    device: torch.device | str = "cpu",
    overwrite_features: bool = False,
) -> GNNFeatureArtifact:
    """Export all graph rows from the exact checkpoint in eval/inference mode."""
    loaded = load_gnn_checkpoint(checkpoint_path, graph_state, device=device)
    model = loaded.model
    model.eval()
    tensors = graph_state_to_tensors(
        graph_state,
        device=device,
        dtype=next(model.parameters()).dtype,
    )
    with torch.inference_mode():
        output = model(tensors)
    if output.node_features.shape != (
        len(graph_state.node_ids),
        graph_state.node_feature_dim,
    ):
        raise RuntimeError("GNN inference returned an unexpected node feature shape")
    if len(tensors.node_ids) != len(output.node_features):
        raise RuntimeError("GNN inference node rows lost their graph ID alignment")
    artifact = _feature_artifact(
        graph_state,
        output,
        loaded.manifest.model_config_record,
        loaded.checkpoint_sha256,
    )
    atomic_write_json(artifact, feature_path, overwrite=overwrite_features)
    return artifact


def load_gnn_features(path: str | Path) -> GNNFeatureArtifact:
    """Read and validate an ID-aligned learned graph-feature artifact."""
    return GNNFeatureArtifact.model_validate_json(read_json_text(path))


def train_gnn(
    graph_state: GraphState,
    model: EdgeFeatureSAGE,
    checkpoint_path: str | Path,
    feature_path: str | Path,
    *,
    training_config: GNNTrainingConfig | None = None,
    device: torch.device | str = "cpu",
    overwrite_checkpoint: bool = False,
    overwrite_features: bool = False,
    log_every_epochs: int = DEFAULT_LOG_EVERY_K_EPOCHS,
) -> GNNTrainingResult:
    """Train one fresh model on a bootstrap graph and export its final weights.

    The supplied model is trained as-is; parameters are never reset in this
    function. Independent experiments must use separately initialized model
    objects and distinct artifact paths. Loaded checkpoints are inference-only
    in this milestone; there is no batch-continuation claim or optimizer resume.
    """
    if not isinstance(model, EdgeFeatureSAGE):
        raise TypeError("model must be an explicitly constructed EdgeFeatureSAGE")
    if model._inference_only or model._training_started or model._training_completed:
        raise GNNTrainingError(
            "train_gnn accepts a new model only; loaded/completed runs are inference-only"
        )
    if graph_state.batch_role != "bootstrap_reference":
        raise GNNTrainingError(
            "GNN training requires a bootstrap_reference graph; update batches are not training inputs"
        )
    if not graph_state.radio_edges and not graph_state.virtual_edges:
        raise GNNTrainingError(
            "cannot train Eq. 4 on a graph with no physical or virtual edges"
        )
    if graph_state.node_feature_dim != model.config.graph_feature_dim:
        raise GNNTrainingError("model graph feature width does not match GraphState")
    if next(model.parameters()).dtype != torch.float32:
        raise GNNTrainingError(
            "this GNN checkpoint format supports float32 parameters only"
        )
    selected_training = training_config or GNNTrainingConfig()
    log_every_epochs = validate_log_interval(log_every_epochs)
    checkpoint_target = Path(checkpoint_path)
    features_target = Path(feature_path)
    if checkpoint_target.exists() and not overwrite_checkpoint:
        raise FileExistsError(
            f"refusing to overwrite GNN checkpoint: {checkpoint_target}"
        )
    if features_target.exists() and not overwrite_features:
        raise FileExistsError(
            f"refusing to overwrite GNN feature artifact: {features_target}"
        )

    selected_device = torch.device(device)
    model.to(selected_device)
    tensors = graph_state_to_tensors(
        graph_state,
        device=selected_device,
        dtype=next(model.parameters()).dtype,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=selected_training.learning_rate,
        weight_decay=selected_training.weight_decay,
    )
    loss_history: list[float] = []
    model._training_started = True
    cuda_devices = []
    if selected_device.type == "cuda":
        cuda_devices = [
            selected_device.index
            if selected_device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(selected_training.seed)
        if cuda_devices:
            with torch.cuda.device(cuda_devices[0]):
                torch.cuda.manual_seed(selected_training.seed)
        for _epoch in range(1, selected_training.epochs + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(tensors)
            loss = equation4_loss(
                output.node_features,
                graph_state,
                epsilon=selected_training.objective_epsilon,
            )
            if not torch.isfinite(loss):
                raise GNNTrainingError("Eq. 4 loss is non-finite")
            loss.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.grad is not None
            ]
            if not gradients or any(
                not torch.isfinite(gradient).all().item() for gradient in gradients
            ):
                raise GNNTrainingError("GNN produced no finite parameter gradients")
            if not any(torch.count_nonzero(gradient).item() for gradient in gradients):
                raise GNNTrainingError(
                    "Eq. 4/ReLU produced no gradient signal; no trained checkpoint was saved"
                )
            optimizer.step()
            loss_history.append(float(loss.detach().to(device="cpu").item()))
            log_training_epoch(
                "GNN",
                _epoch,
                selected_training.epochs,
                {
                    "loss": loss_history[-1],
                    "nodes": graph_state.node_count,
                    "radio_edges": graph_state.physical_edge_count,
                    "virtual_edges": graph_state.virtual_edge_count,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
                every_k_epochs=log_every_epochs,
            )

    model.eval()
    model._training_completed = True
    history = tuple(loss_history)
    manifest = _manifest_for_graph(
        graph_state, model.config, selected_training, history, selected_device
    )
    _save_checkpoint(
        model,
        manifest,
        checkpoint_target,
        overwrite=overwrite_checkpoint,
    )
    # Reload the serialized selected checkpoint before export, so feature rows
    # demonstrably come from persisted final weights rather than a live trainer.
    loaded = load_gnn_checkpoint(checkpoint_target, graph_state, device=selected_device)
    features = encode_graph_state(
        graph_state,
        checkpoint_target,
        features_target,
        device=selected_device,
        overwrite_features=overwrite_features,
    )
    return GNNTrainingResult(
        checkpoint=loaded,
        features=features,
        loss_history=history,
    )


if __name__ == "__main__":
    raise SystemExit(
        "Use the graph-state training API from gufu.py; standalone edgelist/random-feature training was removed."
    )
