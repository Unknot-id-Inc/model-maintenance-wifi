"""Chunked full-graph GNN training and inference over the streamed online graph.

The implementation applies mean softplus losses to normalized cosine scores
for positive graph edges and RSSI-mined negative scan pairs, matching the
in-memory trainer. Numeric graph state and per-epoch activations live in NumPy
memory maps; only one edge/node chunk and the small model are resident at once.
Dropout uses the explicitly named, deterministic chunk RNG policy recorded in
the checkpoint because chunked RNG calls cannot promise PyTorch's full-tensor
dropout bitstream.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

from gufu_artifacts import (
    atomic_torch_save,
    atomic_write_json,
    canonical_json_digest,
    load_torch_state,
    read_json_text,
    sha256_file,
)
from lp_streaming import StreamArrayArtifact
from graph_build import GraphBuildConfig
from sage_ve import (
    EdgeFeatureSAGE,
    GNNModelConfig,
    GNNTrainingConfig,
    LoadedGNNCheckpoint,
)
from streaming_update import MappedStreamingOnlineGraph, load_streaming_online_graph
from training_progress import (
    DEFAULT_LOG_EVERY_K_EPOCHS,
    log_training_epoch,
    log_training_start,
    validate_log_interval,
)


STREAM_GNN_FORMAT = "gufu_streaming_gnn_v7"
STREAM_GNN_TRAINING_POLICY = "chunked_full_graph_softplus_cosine_terminal_node_refinement_v7"


class StreamingGNNManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    format: Literal["gufu_streaming_gnn_v7"] = STREAM_GNN_FORMAT
    artifact_id: str = Field(pattern=r"^streamgnn:[0-9a-f]{64}$")
    online_graph_id: str = Field(pattern=r"^onlinegraph:[0-9a-f]{64}$")
    dataset_id: str
    reference_batch_id: str
    update_batch_id: str
    reference_graph_id: str = Field(pattern=r"^graph:[0-9a-f]{64}$")
    reference_lp_result_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_gnn_checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    graph_build_config: GraphBuildConfig
    vocabulary_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    node_ids: tuple[str, ...]
    radio_edge_count: int = Field(ge=0)
    virtual_edge_count: int = Field(ge=0)
    negative_pair_count: int = Field(ge=0)
    model_config_record: GNNModelConfig
    training_config: GNNTrainingConfig
    training_policy: Literal[
        "chunked_full_graph_softplus_cosine_terminal_node_refinement_v7"
    ] = STREAM_GNN_TRAINING_POLICY
    dropout_rng_policy: Literal["per_epoch_layer_kind_chunk_sequence_v1"] = (
        "per_epoch_layer_kind_chunk_sequence_v1"
    )
    resume_policy: Literal["weights_only_fresh_adam_each_batch"] = (
        "weights_only_fresh_adam_each_batch"
    )
    epochs_completed: int = Field(ge=1)
    loss_history: tuple[float, ...]
    checkpoint_name: Literal["gnn_state.pt"] = "gnn_state.pt"
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    arrays: tuple[StreamArrayArtifact, ...]

    @model_validator(mode="after")
    def aligned(self):
        if len(set(self.node_ids)) != len(self.node_ids):
            raise ValueError("streaming GNN node IDs must be unique")
        if self.epochs_completed != len(self.loss_history):
            raise ValueError("streaming GNN loss history must contain one scalar per epoch")
        if any(not math.isfinite(value) for value in self.loss_history):
            raise ValueError("streaming GNN loss history must be finite")
        if self.model_config_record.graph_feature_dim < 1:
            raise ValueError("streaming GNN feature width must be positive")
        if self.graph_build_config.spatial_normalization_enabled and (
            self.graph_build_config.spatial_transform_metadata is None
            or self.graph_build_config.spatial_transform_sha256 is None
        ):
            raise ValueError("streaming GNN checkpoint must carry resolved spatial transform metadata")
        expected = {
            "node_embeddings.npy": (len(self.node_ids), self.model_config_record.node_output_dim),
            "radio_edge_features.npy": (
                self.radio_edge_count,
                self.model_config_record.edge_output_dim,
            ),
        }
        if {array.name: array.shape for array in self.arrays} != expected:
            raise ValueError("streaming GNN output array shapes are incompatible")
        if self.artifact_id != f"streamgnn:{canonical_json_digest(self.model_dump(mode='json', exclude={'artifact_id'}))}":
            raise ValueError("streaming GNN artifact identity does not match its manifest")
        return self


@dataclass(frozen=True)
class StreamingGNNState:
    directory: Path
    manifest: StreamingGNNManifest
    model: EdgeFeatureSAGE
    node_embeddings: np.ndarray
    radio_edge_features: np.ndarray


def _open_array(path: Path, shape: tuple[int, ...], dtype="float32") -> np.ndarray:
    if math.prod(shape) == 0:
        array = np.empty(shape, dtype=dtype)
        np.save(path, array, allow_pickle=False)
        return array
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _artifact(path: Path, name: str) -> StreamArrayArtifact:
    try:
        values = np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError:
        values = np.load(path, allow_pickle=False)
    return StreamArrayArtifact(
        name=name,
        dtype=values.dtype.name,
        shape=tuple(int(value) for value in values.shape),
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
    )


def _jsonable(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def _tensor(values, device: torch.device, *, requires_grad=False) -> Tensor:
    # Copy only the current bounded chunk from its mmap into a writable tensor.
    return torch.tensor(np.asarray(values), dtype=torch.float32, device=device, requires_grad=requires_grad)


def _seed(seed: int, epoch: int, layer: int, kind: int) -> int:
    return (int(seed) + 1_000_003 * epoch + 10_007 * layer + 101 * kind) % (2**63 - 1)


def _generator(seed: int) -> torch.Generator:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def _dropout_mask(
    rows: int, width: int, probability: float, generator: torch.Generator | None, device
) -> Tensor | None:
    if probability == 0.0:
        return None
    if generator is None:
        raise AssertionError("training dropout requires an explicit generator")
    return (
        torch.rand((rows, width), generator=generator, device="cpu")
        .ge(probability)
        .to(dtype=torch.float32, device=device)
        .div_(1.0 - probability)
    )


def _apply_dropout(values: Tensor, mask: Tensor | None) -> Tensor:
    return values if mask is None else values * mask


def _edge_endpoints(graph: MappedStreamingOnlineGraph, start: int, stop: int):
    radio_count = graph.manifest.radio_edge_count
    radio_stop = min(stop, radio_count)
    if start < radio_stop:
        first = np.asarray(graph.radio_endpoints[start:radio_stop], dtype=np.int64)
    else:
        first = np.empty((0, 2), dtype=np.int64)
    virtual_start = max(start, radio_count) - radio_count
    virtual_stop = max(stop - radio_count, 0)
    if virtual_start < virtual_stop:
        second = np.asarray(graph.virtual_endpoints[virtual_start:virtual_stop], dtype=np.int64)
    else:
        second = np.empty((0, 2), dtype=np.int64)
    return first if not len(second) else second if not len(first) else np.concatenate((first, second))


def _edge_features(
    graph: MappedStreamingOnlineGraph,
    radio_features: np.ndarray,
    start: int,
    stop: int,
):
    radio_count = graph.manifest.radio_edge_count
    radio_stop = min(stop, radio_count)
    if start < radio_stop:
        first = np.asarray(radio_features[start:radio_stop], dtype=np.float32)
    else:
        first = np.empty((0, graph.manifest.node_feature_dim), dtype=np.float32)
    virtual_start = max(start, radio_count) - radio_count
    virtual_stop = max(stop - radio_count, 0)
    if virtual_start < virtual_stop:
        second = np.asarray(
            graph.virtual_edge_features[virtual_start:virtual_stop], dtype=np.float32
        )
    else:
        second = np.empty((0, graph.manifest.node_feature_dim), dtype=np.float32)
    return first if not len(second) else second if not len(first) else np.concatenate((first, second))


def _forward_stream(
    graph: MappedStreamingOnlineGraph,
    model_config: GNNModelConfig,
    parameters: tuple[Tensor, ...],
    work_directory: Path,
    *,
    chunk_size: int,
    training_seed: int | None,
    epoch: int = 0,
):
    """Run Algorithm 1 into mapped activation files and return layer paths."""
    device = parameters[0].device
    count_nodes = len(graph.manifest.node_ids)
    count_radio = graph.manifest.radio_edge_count
    count_virtual = graph.manifest.virtual_edge_count
    count_edges = count_radio + count_virtual
    dim = model_config.graph_feature_dim
    degree = np.zeros((count_nodes,), dtype=np.int64)
    for start in range(0, count_edges, chunk_size):
        endpoints = _edge_endpoints(graph, start, min(start + chunk_size, count_edges))
        np.add.at(degree, endpoints[:, 0], 1)
        np.add.at(degree, endpoints[:, 1], 1)

    h0 = _open_array(work_directory / "h_layer_0.npy", (count_nodes, dim))
    for start in range(0, count_nodes, chunk_size):
        stop = min(start + chunk_size, count_nodes)
        h0[start:stop] = np.asarray(graph.node_features[start:stop], dtype=np.float32)
    if isinstance(h0, np.memmap):
        h0.flush()
    h_paths = [work_directory / "h_layer_0.npy"]
    e_paths = []
    e0 = _open_array(work_directory / "e_radio_layer_0.npy", (count_radio, dim))
    for start in range(0, count_radio, chunk_size):
        stop = min(start + chunk_size, count_radio)
        e0[start:stop] = np.asarray(graph.radio_edge_features[start:stop], dtype=np.float32)
    if isinstance(e0, np.memmap):
        e0.flush()
    e_paths.append(work_directory / "e_radio_layer_0.npy")
    h_current, e_current = h0, e0
    aggregate_paths = []

    for layer in range(model_config.num_layers):
        aggregate_path = work_directory / f"aggregate_layer_{layer}.npy"
        aggregate = _open_array(aggregate_path, (count_nodes, dim))
        aggregate[:] = 0.0
        message_generator = (
            _generator(_seed(training_seed, epoch, layer, 1))
            if training_seed is not None
            else None
        )
        # The first directed half is right->left, the second left->right, as
        # in EdgeFeatureSAGE.forward. This fixes summation and dropout order.
        for direction in range(2):
            for start in range(0, count_edges, chunk_size):
                stop = min(start + chunk_size, count_edges)
                edge_rows = np.arange(start, stop, dtype=np.int64)
                endpoints = _edge_endpoints(graph, start, stop)
                source = endpoints[:, 1] if direction == 0 else endpoints[:, 0]
                target = endpoints[:, 0] if direction == 0 else endpoints[:, 1]
                edge_features = _edge_features(graph, e_current, start, stop)
                source_features = np.asarray(h_current[source], dtype=np.float32)
                edge_tensor = _tensor(edge_features, device)
                source_tensor = _tensor(source_features, device)
                message_inputs = torch.cat((edge_tensor, source_tensor), dim=1)
                messages = F.leaky_relu(
                    F.linear(message_inputs, parameters[layer * 3]), negative_slope=0.1
                )
                mask = _dropout_mask(
                    len(edge_rows), dim, model_config.dropout_probability,
                    message_generator, device,
                ) if training_seed is not None else None
                messages = _apply_dropout(messages, mask)
                np.add.at(aggregate, target, messages.detach().cpu().numpy())
        if isinstance(aggregate, np.memmap):
            aggregate.flush()
        aggregate_paths.append(aggregate_path)

        h_next_path = work_directory / f"h_layer_{layer + 1}.npy"
        h_next = _open_array(h_next_path, (count_nodes, dim))
        node_generator = (
            _generator(_seed(training_seed, epoch, layer, 2))
            if training_seed is not None
            else None
        )
        for start in range(0, count_nodes, chunk_size):
            stop = min(start + chunk_size, count_nodes)
            self_features = _tensor(h_current[start:stop], device)
            mean_aggregate = np.asarray(aggregate[start:stop], dtype=np.float32) / np.maximum(
                degree[start:stop], 1
            ).astype(np.float32)[:, None]
            aggregate_features = _tensor(mean_aggregate, device)
            node_inputs = torch.cat((self_features, aggregate_features), dim=1)
            node_values = F.leaky_relu(
                F.linear(node_inputs, parameters[layer * 3 + 1]), negative_slope=0.1
            )
            mask = _dropout_mask(
                stop - start, dim, model_config.dropout_probability,
                node_generator, device,
            ) if training_seed is not None else None
            node_values = F.normalize(
                _apply_dropout(node_values, mask), p=2.0, dim=1, eps=1e-12
            )
            h_next[start:stop] = node_values.detach().cpu().numpy()
        if isinstance(h_next, np.memmap):
            h_next.flush()

        e_next_path = work_directory / f"e_radio_layer_{layer + 1}.npy"
        e_next = _open_array(e_next_path, (count_radio, dim))
        edge_generator = (
            _generator(_seed(training_seed, epoch, layer, 3))
            if training_seed is not None
            else None
        )
        for start in range(0, count_radio, chunk_size):
            stop = min(start + chunk_size, count_radio)
            endpoints = np.asarray(graph.radio_endpoints[start:stop], dtype=np.int64)
            left = _tensor(h_next[endpoints[:, 0]], device)
            old_edge = _tensor(e_current[start:stop], device)
            right = _tensor(h_next[endpoints[:, 1]], device)
            edge_inputs = torch.cat((left, old_edge, right), dim=1)
            edge_values = F.leaky_relu(
                F.linear(edge_inputs, parameters[layer * 3 + 2]), negative_slope=0.1
            )
            mask = _dropout_mask(
                stop - start, dim, model_config.dropout_probability,
                edge_generator, device,
            ) if training_seed is not None else None
            edge_values = F.normalize(
                _apply_dropout(edge_values, mask), p=2.0, dim=1, eps=1e-12
            )
            e_next[start:stop] = edge_values.detach().cpu().numpy()
        if isinstance(e_next, np.memmap):
            e_next.flush()
        h_current, e_current = h_next, e_next
        h_paths.append(h_next_path)
        e_paths.append(e_next_path)

    # Terminal node refinement: consume the final physical-edge vectors so
    # the last W2 update participates in the node-only training objective.
    terminal_layer = model_config.num_layers
    terminal_w0_index = 3 * terminal_layer
    terminal_w1_index = terminal_w0_index + 1
    terminal_aggregate_path = work_directory / "aggregate_terminal.npy"
    terminal_aggregate = _open_array(terminal_aggregate_path, (count_nodes, dim))
    terminal_aggregate[:] = 0.0
    terminal_message_generator = (
        _generator(_seed(training_seed, epoch, terminal_layer, 1))
        if training_seed is not None
        else None
    )
    for direction in range(2):
        for start in range(0, count_edges, chunk_size):
            stop = min(start + chunk_size, count_edges)
            endpoints = _edge_endpoints(graph, start, stop)
            source = endpoints[:, 1] if direction == 0 else endpoints[:, 0]
            target = endpoints[:, 0] if direction == 0 else endpoints[:, 1]
            edge_features = _edge_features(graph, e_current, start, stop)
            source_features = np.asarray(h_current[source], dtype=np.float32)
            message_inputs = torch.cat(
                (_tensor(edge_features, device), _tensor(source_features, device)),
                dim=1,
            )
            messages = F.leaky_relu(
                F.linear(message_inputs, parameters[terminal_w0_index]),
                negative_slope=0.1,
            )
            mask = _dropout_mask(
                stop - start,
                dim,
                model_config.dropout_probability,
                terminal_message_generator,
                device,
            ) if training_seed is not None else None
            messages = _apply_dropout(messages, mask)
            np.add.at(terminal_aggregate, target, messages.detach().cpu().numpy())
    if isinstance(terminal_aggregate, np.memmap):
        terminal_aggregate.flush()
    aggregate_paths.append(terminal_aggregate_path)

    terminal_h_path = work_directory / "h_layer_terminal.npy"
    terminal_h = _open_array(terminal_h_path, (count_nodes, dim))
    terminal_node_generator = (
        _generator(_seed(training_seed, epoch, terminal_layer, 2))
        if training_seed is not None
        else None
    )
    for start in range(0, count_nodes, chunk_size):
        stop = min(start + chunk_size, count_nodes)
        self_features = _tensor(h_current[start:stop], device)
        mean_aggregate = np.asarray(
            terminal_aggregate[start:stop], dtype=np.float32
        ) / np.maximum(degree[start:stop], 1).astype(np.float32)[:, None]
        node_inputs = torch.cat(
            (self_features, _tensor(mean_aggregate, device)), dim=1
        )
        node_values = F.leaky_relu(
            F.linear(node_inputs, parameters[terminal_w1_index]),
            negative_slope=0.1,
        )
        mask = _dropout_mask(
            stop - start,
            dim,
            model_config.dropout_probability,
            terminal_node_generator,
            device,
        ) if training_seed is not None else None
        node_values = F.normalize(
            _apply_dropout(node_values, mask), p=2.0, dim=1, eps=1e-12
        )
        terminal_h[start:stop] = node_values.detach().cpu().numpy()
    if isinstance(terminal_h, np.memmap):
        terminal_h.flush()
    h_paths.append(terminal_h_path)

    return h_paths, e_paths, aggregate_paths, degree


def _loss_and_node_gradient(
    graph: MappedStreamingOnlineGraph,
    final_node_path: Path,
    work_directory: Path,
    *,
    chunk_size: int,
    device: torch.device,
    temperature: float = 0.1,
    negative_weight: float = 1.0,
    epsilon: float | None = None,
):
    count_nodes = len(graph.manifest.node_ids)
    count_edges = graph.manifest.radio_edge_count + graph.manifest.virtual_edge_count
    count_negatives = graph.manifest.negative_pair_count
    if count_edges == 0:
        raise ValueError("cosine softplus loss requires at least one positive pair")
    if count_negatives == 0:
        raise ValueError("cosine softplus loss requires at least one negative pair")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("contrastive temperature must be finite and positive")
    if not math.isfinite(negative_weight) or negative_weight < 0.0:
        raise ValueError("negative_weight must be finite and nonnegative")
    final_nodes = np.load(final_node_path, mmap_mode="r", allow_pickle=False)
    gradient = _open_array(work_directory / "gradient_h_final.npy", (count_nodes, graph.manifest.node_feature_dim))
    gradient[:] = 0.0
    loss_value = 0.0

    def accumulate_pair_loss(endpoints: np.ndarray, *, positive: bool, pair_count: int):
        nonlocal loss_value
        if len(endpoints) == 0:
            return
        with torch.enable_grad():
            left = _tensor(final_nodes[endpoints[:, 0]], device).requires_grad_(True)
            right = _tensor(final_nodes[endpoints[:, 1]], device).requires_grad_(True)
            left_normalized = F.normalize(left, p=2, dim=-1)
            right_normalized = F.normalize(right, p=2, dim=-1)
            cosines = (left_normalized * right_normalized).sum(dim=-1)
            if positive:
                pair_loss = F.softplus(-cosines / temperature)
                coefficient = 1.0 / float(pair_count)
            else:
                pair_loss = F.softplus(cosines / temperature)
                coefficient = negative_weight / float(pair_count)
            scaled_loss = pair_loss.sum() * coefficient
            grad_left, grad_right = torch.autograd.grad(scaled_loss, (left, right))
        loss_value += float(scaled_loss.detach().cpu())
        np.add.at(gradient, endpoints[:, 0], grad_left.detach().cpu().numpy())
        np.add.at(gradient, endpoints[:, 1], grad_right.detach().cpu().numpy())

    for start in range(0, count_edges, chunk_size):
        stop = min(start + chunk_size, count_edges)
        endpoints = _edge_endpoints(graph, start, stop)
        accumulate_pair_loss(endpoints, positive=True, pair_count=count_edges)
    for start in range(0, count_negatives, chunk_size):
        stop = min(start + chunk_size, count_negatives)
        endpoints = np.asarray(graph.negative_endpoints[start:stop], dtype=np.int64)
        accumulate_pair_loss(endpoints, positive=False, pair_count=count_negatives)
    if isinstance(gradient, np.memmap):
        gradient.flush()
    return loss_value, work_directory / "gradient_h_final.npy"


def _accumulate_parameter_gradient(parameter: Tensor, gradient: Tensor, gradients: dict[int, Tensor]):
    key = id(parameter)
    if key in gradients:
        gradients[key].add_(gradient.detach())
    else:
        gradients[key] = gradient.detach().clone()


def _backward_terminal_refinement(
    graph: MappedStreamingOnlineGraph,
    model_config: GNNModelConfig,
    parameters: tuple[Tensor, ...],
    h_paths: list[Path],
    e_paths: list[Path],
    aggregate_paths: list[Path],
    degree: np.ndarray,
    final_gradient_path: Path,
    work_directory: Path,
    *,
    chunk_size: int,
    seed: int,
    epoch: int,
    device: torch.device,
):
    """Backpropagate the node loss through the terminal node refinement."""
    count_nodes = len(graph.manifest.node_ids)
    count_radio = graph.manifest.radio_edge_count
    count_edges = count_radio + graph.manifest.virtual_edge_count
    dim = model_config.graph_feature_dim
    terminal_layer = model_config.num_layers
    terminal_w0_index = 3 * terminal_layer
    terminal_w1_index = terminal_w0_index + 1
    h_input = np.load(h_paths[-2], mmap_mode="r", allow_pickle=False)
    e_input = np.load(e_paths[-1], mmap_mode="r", allow_pickle=False)
    aggregate = np.load(aggregate_paths[-1], mmap_mode="r", allow_pickle=False)
    grad_h_upstream = np.load(final_gradient_path, mmap_mode="r", allow_pickle=False)
    gradients: dict[int, Tensor] = {}

    # Differentiate the terminal node projection with respect to its node and
    # aggregate inputs, then propagate the aggregate gradient into messages.
    grad_h_input_path = work_directory / "gradient_h_terminal_input.npy"
    grad_h_input = _open_array(grad_h_input_path, (count_nodes, dim))
    grad_a_path = work_directory / "gradient_a_terminal.npy"
    grad_a = _open_array(grad_a_path, (count_nodes, dim))
    node_generator = _generator(_seed(seed, epoch, terminal_layer, 2))
    terminal_w1 = parameters[terminal_w1_index]
    for start in range(0, count_nodes, chunk_size):
        stop = min(start + chunk_size, count_nodes)
        self_features = _tensor(h_input[start:stop], device)
        mean_aggregate = np.asarray(aggregate[start:stop], dtype=np.float32) / np.maximum(
            degree[start:stop], 1
        ).astype(np.float32)[:, None]
        inputs = torch.cat(
            (self_features, _tensor(mean_aggregate, device)), dim=1
        ).requires_grad_(True)
        values = F.leaky_relu(F.linear(inputs, terminal_w1), negative_slope=0.1)
        mask = _dropout_mask(
            stop - start, dim, model_config.dropout_probability,
            node_generator, device,
        )
        values = F.normalize(
            _apply_dropout(values, mask), p=2.0, dim=1, eps=1e-12
        )
        upstream = _tensor(grad_h_upstream[start:stop], device)
        scalar = (values * upstream).sum()
        grad_inputs, grad_weight = torch.autograd.grad(scalar, (inputs, terminal_w1))
        _accumulate_parameter_gradient(terminal_w1, grad_weight, gradients)
        values_grad = grad_inputs.detach().cpu().numpy()
        grad_h_input[start:stop] = values_grad[:, :dim]
        grad_a[start:stop] = values_grad[:, dim:]
    for array in (grad_h_input, grad_a):
        if isinstance(array, np.memmap):
            array.flush()

    grad_h_message_path = work_directory / "gradient_h_terminal_message.npy"
    grad_h_message = _open_array(grad_h_message_path, (count_nodes, dim))
    grad_h_message[:] = grad_h_input
    grad_e_path = work_directory / "gradient_e_terminal_output.npy"
    grad_e = _open_array(grad_e_path, (count_radio, dim))
    grad_e[:] = 0.0
    aggregate_gradient = np.load(grad_a_path, mmap_mode="r", allow_pickle=False)
    message_generator = _generator(_seed(seed, epoch, terminal_layer, 1))
    terminal_w0 = parameters[terminal_w0_index]
    for direction in range(2):
        for start in range(0, count_edges, chunk_size):
            stop = min(start + chunk_size, count_edges)
            endpoints = _edge_endpoints(graph, start, stop)
            source = endpoints[:, 1] if direction == 0 else endpoints[:, 0]
            target = endpoints[:, 0] if direction == 0 else endpoints[:, 1]
            source_features = _tensor(h_input[source], device, requires_grad=True)
            edge_features = _tensor(
                _edge_features(graph, e_input, start, stop),
                device,
                requires_grad=True,
            )
            inputs = torch.cat((edge_features, source_features), dim=1).requires_grad_(True)
            values = F.leaky_relu(F.linear(inputs, terminal_w0), negative_slope=0.1)
            mask = _dropout_mask(
                stop - start, dim, model_config.dropout_probability,
                message_generator, device,
            )
            values = _apply_dropout(values, mask)
            valid_degree = np.maximum(degree[target], 1).astype(np.float32)
            upstream_np = np.asarray(aggregate_gradient[target], dtype=np.float32) / valid_degree[:, None]
            upstream = _tensor(upstream_np, device)
            scalar = (values * upstream).sum()
            grad_inputs, grad_weight = torch.autograd.grad(scalar, (inputs, terminal_w0))
            _accumulate_parameter_gradient(terminal_w0, grad_weight, gradients)
            grad_values = grad_inputs.detach().cpu().numpy()
            np.add.at(grad_h_message, source, grad_values[:, dim:])
            radio_stop = min(stop, count_radio)
            if start < radio_stop:
                radio_rows = np.arange(start, radio_stop)
                np.add.at(grad_e, radio_rows, grad_values[: radio_stop - start, :dim])
    for array in (grad_h_message, grad_e):
        if isinstance(array, np.memmap):
            array.flush()
    return grad_h_message_path, grad_e_path, gradients


def _backward_stream(
    graph: MappedStreamingOnlineGraph,
    model_config: GNNModelConfig,
    parameters: tuple[Tensor, ...],
    h_paths: list[Path],
    e_paths: list[Path],
    aggregate_paths: list[Path],
    degree: np.ndarray,
    final_gradient_path: Path,
    work_directory: Path,
    *,
    chunk_size: int,
    seed: int,
    epoch: int,
    device: torch.device,
):
    count_nodes = len(graph.manifest.node_ids)
    count_radio = graph.manifest.radio_edge_count
    count_virtual = graph.manifest.virtual_edge_count
    count_edges = count_radio + count_virtual
    dim = model_config.graph_feature_dim
    grad_h_path, grad_e_out_path, gradients = _backward_terminal_refinement(
        graph,
        model_config,
        parameters,
        h_paths,
        e_paths,
        aggregate_paths,
        degree,
        final_gradient_path,
        work_directory,
        chunk_size=chunk_size,
        seed=seed,
        epoch=epoch,
        device=device,
    )
    grad_h_out = np.load(grad_h_path, mmap_mode="r", allow_pickle=False)

    for layer in reversed(range(model_config.num_layers)):
        h_input = np.load(h_paths[layer], mmap_mode="r", allow_pickle=False)
        h_output = np.load(h_paths[layer + 1], mmap_mode="r", allow_pickle=False)
        e_input = np.load(e_paths[layer], mmap_mode="r", allow_pickle=False)

        # Backpropagate the terminal/next node layer's message gradient through
        # this layer's physical-edge update.
        grad_h_extra_path = work_directory / f"gradient_h_edge_{layer}.npy"
        grad_h_extra = _open_array(grad_h_extra_path, (count_nodes, dim))
        grad_h_extra[:] = 0.0
        grad_e_input = _open_array(
            work_directory / f"gradient_e_input_{layer}.npy", (count_radio, dim)
        )
        grad_e_input[:] = 0.0
        if grad_e_out_path is not None and count_radio:
            grad_e_out = np.load(grad_e_out_path, mmap_mode="r", allow_pickle=False)
            edge_generator = _generator(_seed(seed, epoch, layer, 3))
            weight = parameters[layer * 3 + 2]
            for start in range(0, count_radio, chunk_size):
                stop = min(start + chunk_size, count_radio)
                endpoints = np.asarray(graph.radio_endpoints[start:stop], dtype=np.int64)
                left = _tensor(h_output[endpoints[:, 0]], device, requires_grad=True)
                old_edge = _tensor(e_input[start:stop], device, requires_grad=True)
                right = _tensor(h_output[endpoints[:, 1]], device, requires_grad=True)
                inputs = torch.cat((left, old_edge, right), dim=1).requires_grad_(True)
                values = F.leaky_relu(F.linear(inputs, weight), negative_slope=0.1)
                mask = _dropout_mask(
                    stop - start, dim, model_config.dropout_probability,
                    edge_generator, device,
                )
                values = F.normalize(
                    _apply_dropout(values, mask), p=2.0, dim=1, eps=1e-12
                )
                upstream = _tensor(grad_e_out[start:stop], device)
                scalar = (values * upstream).sum()
                grad_inputs, grad_weight = torch.autograd.grad(scalar, (inputs, weight))
                _accumulate_parameter_gradient(weight, grad_weight, gradients)
                grad_values = grad_inputs.detach().cpu().numpy()
                np.add.at(grad_h_extra, endpoints[:, 0], grad_values[:, :dim])
                np.add.at(grad_e_input, np.arange(start, stop), grad_values[:, dim : 2 * dim])
                np.add.at(grad_h_extra, endpoints[:, 1], grad_values[:, 2 * dim :])

        grad_h_combined_path = work_directory / f"gradient_h_combined_{layer}.npy"
        grad_h_combined = _open_array(grad_h_combined_path, (count_nodes, dim))
        current_grad_h = grad_h_out
        for start in range(0, count_nodes, chunk_size):
            stop = min(start + chunk_size, count_nodes)
            grad_h_combined[start:stop] = current_grad_h[start:stop] + grad_h_extra[start:stop]
        if isinstance(grad_h_combined, np.memmap):
            grad_h_combined.flush()

        # Node update derivative matches EdgeFeatureSAGE's leaky-ReLU update.
        aggregate = np.load(aggregate_paths[layer], mmap_mode="r", allow_pickle=False)
        grad_h_input_path = work_directory / f"gradient_h_input_{layer}.npy"
        grad_h_input = _open_array(grad_h_input_path, (count_nodes, dim))
        grad_a_path = work_directory / f"gradient_a_{layer}.npy"
        grad_a = _open_array(grad_a_path, (count_nodes, dim))
        node_generator = _generator(_seed(seed, epoch, layer, 2))
        weight = parameters[layer * 3 + 1]
        grad_h_upstream = np.load(grad_h_combined_path, mmap_mode="r", allow_pickle=False)
        for start in range(0, count_nodes, chunk_size):
            stop = min(start + chunk_size, count_nodes)
            self_features = _tensor(h_input[start:stop], device)
            mean_aggregate = np.asarray(aggregate[start:stop], dtype=np.float32) / np.maximum(
                degree[start:stop], 1
            ).astype(np.float32)[:, None]
            aggregate_features = _tensor(mean_aggregate, device)
            inputs = torch.cat((self_features, aggregate_features), dim=1).requires_grad_(True)
            values = F.leaky_relu(F.linear(inputs, weight), negative_slope=0.1)
            mask = _dropout_mask(
                stop - start, dim, model_config.dropout_probability,
                node_generator, device,
            )
            values = F.normalize(
                _apply_dropout(values, mask), p=2.0, dim=1, eps=1e-12
            )
            upstream = _tensor(grad_h_upstream[start:stop], device)
            scalar = (values * upstream).sum()
            grad_inputs, grad_weight = torch.autograd.grad(scalar, (inputs, weight))
            _accumulate_parameter_gradient(weight, grad_weight, gradients)
            values_grad = grad_inputs.detach().cpu().numpy()
            grad_h_input[start:stop] = values_grad[:, :dim]
            grad_a[start:stop] = values_grad[:, dim:]
        for array in (grad_h_input, grad_a):
            if isinstance(array, np.memmap):
                array.flush()

        # Message derivative scatters mean-aggregate gradients to source nodes
        # and to the edge-vector rows. Physical edge-vector gradients continue
        # through W2 at the previous layer; virtual edge vectors are constants.
        grad_h_message_path = work_directory / f"gradient_h_message_{layer}.npy"
        grad_h_message = _open_array(grad_h_message_path, (count_nodes, dim))
        for start in range(0, count_nodes, chunk_size):
            stop = min(start + chunk_size, count_nodes)
            grad_h_message[start:stop] = grad_h_input[start:stop]
        if isinstance(grad_h_message, np.memmap):
            grad_h_message.flush()
        agg_gradient = np.load(grad_a_path, mmap_mode="r", allow_pickle=False)
        message_generator = _generator(_seed(seed, epoch, layer, 1))
        weight = parameters[layer * 3]
        for direction in range(2):
            for start in range(0, count_edges, chunk_size):
                stop = min(start + chunk_size, count_edges)
                endpoints = _edge_endpoints(graph, start, stop)
                source = endpoints[:, 1] if direction == 0 else endpoints[:, 0]
                target = endpoints[:, 0] if direction == 0 else endpoints[:, 1]
                source_features = _tensor(h_input[source], device, requires_grad=True)
                edge_features = _tensor(
                    _edge_features(graph, e_input, start, stop),
                    device,
                    requires_grad=True,
                )
                inputs = torch.cat((edge_features, source_features), dim=1).requires_grad_(True)
                values = F.leaky_relu(F.linear(inputs, weight), negative_slope=0.1)
                mask = _dropout_mask(
                    stop - start, dim, model_config.dropout_probability,
                    message_generator, device,
                )
                values = _apply_dropout(values, mask)
                valid_degree = np.maximum(degree[target], 1).astype(np.float32)
                upstream_np = np.asarray(agg_gradient[target], dtype=np.float32) / valid_degree[:, None]
                upstream = _tensor(upstream_np, device)
                scalar = (values * upstream).sum()
                grad_inputs, grad_weight = torch.autograd.grad(scalar, (inputs, weight))
                _accumulate_parameter_gradient(weight, grad_weight, gradients)
                grad_values = grad_inputs.detach().cpu().numpy()
                np.add.at(grad_h_message, source, grad_values[:, dim:])
                radio_start = start
                radio_stop = min(stop, count_radio)
                if radio_start < radio_stop:
                    local_edge_rows = np.arange(radio_start, radio_stop)
                    np.add.at(
                        grad_e_input,
                        local_edge_rows,
                        grad_values[: radio_stop - radio_start, :dim],
                    )
        if isinstance(grad_h_message, np.memmap):
            grad_h_message.flush()
        if isinstance(grad_e_input, np.memmap):
            grad_e_input.flush()
        grad_h_out = grad_h_message
        grad_e_out_path = work_directory / f"gradient_e_input_{layer}.npy"

    return tuple(gradients.get(id(parameter)) for parameter in parameters)


class _StreamedEquation4(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        graph,
        model_config,
        epsilon,
        seed,
        chunk_size,
        *objective_and_parameters,
    ):
        parameter_count = 3 * model_config.num_layers + 2
        optional_input_count = len(objective_and_parameters) - parameter_count
        if optional_input_count not in (0, 2):
            raise ValueError("streamed contrastive loss received invalid objective arguments")
        if optional_input_count == 0:
            temperature = 0.1
            negative_weight = 1.0
            parameters = objective_and_parameters
        else:
            temperature, negative_weight = objective_and_parameters[:2]
            parameters = objective_and_parameters[2:]
        if len(parameters) != parameter_count:
            raise ValueError("streamed contrastive model parameter count is inconsistent")
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("contrastive temperature must be finite and positive")
        if not math.isfinite(negative_weight) or negative_weight < 0.0:
            raise ValueError("negative_weight must be finite and nonnegative")
        if graph.manifest.radio_edge_count + graph.manifest.virtual_edge_count == 0:
            raise ValueError("streamed contrastive loss requires positive pairs")
        work_directory = Path(tempfile.mkdtemp(prefix="gufu-stream-gnn-"))
        try:
            h_paths, e_paths, aggregate_paths, degree = _forward_stream(
                graph,
                model_config,
                tuple(parameters),
                work_directory,
                chunk_size=chunk_size,
                training_seed=seed,
                epoch=0,
            )
            # Distinguish epochs by deriving the seed before entry. The caller
            # passes a unique per-epoch seed; epoch zero is then fixed here.
            loss_result = _loss_and_node_gradient(
                graph,
                h_paths[-1],
                work_directory,
                chunk_size=chunk_size,
                device=parameters[0].device,
                temperature=temperature,
                negative_weight=negative_weight,
            )
            loss_value, grad_path = loss_result
            ctx.graph = graph
            ctx.model_config = model_config
            ctx.seed = seed
            ctx.chunk_size = chunk_size
            ctx.optional_input_count = optional_input_count
            ctx.work_directory = work_directory
            ctx.h_paths = h_paths
            ctx.e_paths = e_paths
            ctx.aggregate_paths = aggregate_paths
            ctx.degree = degree
            ctx.grad_path = grad_path
            ctx.parameter_count = len(parameters)
            ctx.save_for_backward(*parameters)
            return torch.tensor(loss_value, dtype=parameters[0].dtype, device=parameters[0].device)
        except Exception:
            # The directory is created solely for this incomplete epoch.
            shutil.rmtree(work_directory, ignore_errors=True)
            raise

    @staticmethod
    def backward(ctx, grad_output):
        parameters = ctx.saved_tensors
        try:
            with torch.enable_grad():
                gradients = list(_backward_stream(
                    ctx.graph,
                    ctx.model_config,
                    tuple(parameters),
                    ctx.h_paths,
                    ctx.e_paths,
                    ctx.aggregate_paths,
                    ctx.degree,
                    ctx.grad_path,
                    ctx.work_directory,
                    chunk_size=ctx.chunk_size,
                    seed=ctx.seed,
                    epoch=0,
                    device=parameters[0].device,
                ))
            return (
                None,
                None,
                None,
                None,
                None,
                *(None for _ in range(ctx.optional_input_count)),
                *(gradient * grad_output if gradient is not None else None for gradient in gradients),
            )
        finally:
            shutil.rmtree(ctx.work_directory, ignore_errors=True)


def _parameter_order(model: EdgeFeatureSAGE) -> tuple[Tensor, ...]:
    result = []
    for layer in range(model.config.num_layers):
        result.extend((model.W0[layer], model.W1[layer], model.W2[layer]))
    result.extend((model.W0_terminal, model.W1_terminal))
    return tuple(result)


def _stream_inference(
    graph: MappedStreamingOnlineGraph,
    model: EdgeFeatureSAGE,
    target_directory: Path,
    *,
    chunk_size: int,
):
    parameters = _parameter_order(model)
    model.eval()
    h_paths, e_paths, _aggregate, _degree = _forward_stream(
        graph,
        model.config,
        parameters,
        target_directory,
        chunk_size=chunk_size,
        training_seed=None,
    )
    node_path = target_directory / "node_embeddings.npy"
    radio_path = target_directory / "radio_edge_features.npy"
    shutil.copyfile(h_paths[-1], node_path)
    shutil.copyfile(e_paths[-1], radio_path)
    return node_path, radio_path


def infer_streaming_gnn_embeddings(
    graph_directory: str | Path,
    model: EdgeFeatureSAGE,
    node_rows: np.ndarray | tuple[int, ...] | list[int],
    *,
    chunk_size: int = 4096,
    device: str | torch.device = "cpu",
    temporary_parent: str | Path | None = None,
) -> np.ndarray:
    """Infer selected node rows, keeping full-graph intermediate arrays temporary."""
    if chunk_size < 1:
        raise ValueError("streaming GNN inference chunk_size must be positive")
    graph = load_streaming_online_graph(graph_directory)
    if model.config.graph_feature_dim != graph.manifest.node_feature_dim:
        raise ValueError("streaming GNN inference model width differs from graph features")
    rows = np.asarray(node_rows, dtype=np.int64)
    if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= len(graph.manifest.node_ids)):
        raise ValueError("streaming GNN inference rows are outside the graph")
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"requested CUDA device is unavailable: {selected_device}")
    model = model.to(device=selected_device, dtype=torch.float32).eval()
    parent = None if temporary_parent is None else Path(temporary_parent)
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(
        tempfile.mkdtemp(prefix="gufu-gnn-inference-", dir=parent)
    )
    try:
        node_path, _radio_path = _stream_inference(
            graph, model, temporary_directory, chunk_size=chunk_size
        )
        embeddings = np.load(node_path, mmap_mode="r", allow_pickle=False)
        return np.asarray(embeddings[rows], dtype=np.float32).copy()
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)


def infer_streaming_gnn_state(
    graph_directory: str | Path,
    source_state: StreamingGNNState,
    output_directory: str | Path,
    *,
    chunk_size: int = 4096,
    device: str | torch.device = "cpu",
) -> StreamingGNNState:
    """Persist source-GNN embeddings recomputed once over another graph.

    The checkpoint weights and their training history remain those of
    ``source_state``; only graph-bound node and edge embeddings are recomputed.
    This is used when a downstream updater must consume one-pass GNN outputs
    for both references and incoming scans on the same graph.
    """
    if chunk_size < 1:
        raise ValueError("streaming GNN inference chunk_size must be positive")
    graph = load_streaming_online_graph(graph_directory)
    if graph.manifest.reference_graph_id != source_state.manifest.reference_graph_id:
        raise ValueError("inference graph and source GNN use different reference graph lineage")
    if graph.manifest.reference_gnn_checkpoint_sha256 != source_state.manifest.checkpoint_sha256:
        raise ValueError("inference graph was built from a different source GNN checkpoint")
    config = source_state.manifest.model_config_record
    if config.graph_feature_dim != graph.manifest.node_feature_dim:
        raise ValueError("source GNN width is incompatible with inference graph")
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"requested CUDA device is unavailable: {selected_device}")

    target = Path(output_directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite inferred streaming GNN state: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        model = EdgeFeatureSAGE(config).to(device=selected_device, dtype=torch.float32)
        model.load_state_dict(source_state.model.state_dict(), strict=True)
        model.eval()
        with tempfile.TemporaryDirectory(prefix="gufu-stream-gnn-rebind-") as work:
            node_path, radio_path = _stream_inference(
                graph, model, Path(work), chunk_size=chunk_size
            )
            shutil.copyfile(node_path, staging / "node_embeddings.npy")
            shutil.copyfile(radio_path, staging / "radio_edge_features.npy")
        shutil.copyfile(
            source_state.directory / source_state.manifest.checkpoint_name,
            staging / "gnn_state.pt",
        )
        arrays = (
            _artifact(staging / "node_embeddings.npy", "node_embeddings.npy"),
            _artifact(staging / "radio_edge_features.npy", "radio_edge_features.npy"),
        )
        fields = dict(
            online_graph_id=graph.manifest.online_graph_id,
            dataset_id=graph.manifest.dataset_id,
            reference_batch_id=graph.manifest.reference_batch_id,
            update_batch_id=graph.manifest.update_batch_id,
            reference_graph_id=graph.manifest.reference_graph_id,
            reference_lp_result_id=graph.manifest.reference_lp_result_id,
            source_gnn_checkpoint_sha256=source_state.manifest.checkpoint_sha256,
            graph_build_config=graph.manifest.graph_build_config,
            vocabulary_sha256=graph.manifest.vocabulary_sha256,
            node_ids=graph.manifest.node_ids,
            radio_edge_count=graph.manifest.radio_edge_count,
            virtual_edge_count=graph.manifest.virtual_edge_count,
            negative_pair_count=graph.manifest.negative_pair_count,
            model_config_record=config,
            training_config=source_state.manifest.training_config,
            epochs_completed=source_state.manifest.epochs_completed,
            loss_history=source_state.manifest.loss_history,
            checkpoint_sha256=sha256_file(staging / "gnn_state.pt"),
            arrays=arrays,
        )
        provisional = StreamingGNNManifest.model_construct(
            artifact_id="streamgnn:" + "0" * 64,
            **fields,
        )
        artifact_id = "streamgnn:" + canonical_json_digest(
            provisional.model_dump(mode="json", exclude={"artifact_id"})
        )
        manifest = StreamingGNNManifest(artifact_id=artifact_id, **fields)
        atomic_write_json(manifest, staging / "manifest.json")
        atomic_write_json(
            {
                "format": "gufu_streaming_artifact_complete_v1",
                "identity": manifest.artifact_id,
                "manifest_sha256": sha256_file(staging / "manifest.json"),
            },
            staging / "COMPLETED.json",
        )
        os.replace(staging, target)
        return load_streaming_gnn_state(target, graph_directory, device=selected_device)
    except Exception:
        raise


def infer_streaming_gnn_checkpoint_embeddings(
    graph_directory: str | Path,
    checkpoint_path: str | Path,
    model_config: GNNModelConfig,
    node_rows: np.ndarray | tuple[int, ...] | list[int],
    *,
    expected_checkpoint_sha256: str,
    chunk_size: int = 4096,
    device: str | torch.device = "cpu",
    temporary_parent: str | Path | None = None,
) -> np.ndarray:
    """Run a saved standard or streamed GNN checkpoint over a streamed graph."""
    checkpoint = Path(checkpoint_path)
    if not checkpoint.is_file() or sha256_file(checkpoint) != expected_checkpoint_sha256:
        raise ValueError("GNN evaluation checkpoint is missing or has unexpected provenance")
    payload = load_torch_state(checkpoint, map_location="cpu")
    if set(payload) == {"state_dict"} and isinstance(payload["state_dict"], dict):
        state_dict = payload["state_dict"]
    elif isinstance(payload.get("model_state"), dict):
        state_dict = payload["model_state"]
    else:
        raise ValueError("GNN evaluation checkpoint has an unsupported payload")
    model = EdgeFeatureSAGE(model_config)
    model.load_state_dict(state_dict, strict=True)
    model._inference_only = True
    return infer_streaming_gnn_embeddings(
        graph_directory,
        model,
        node_rows,
        chunk_size=chunk_size,
        device=device,
        temporary_parent=temporary_parent,
    )


def train_streaming_gnn(
    graph_directory: str | Path,
    source_checkpoint: LoadedGNNCheckpoint | StreamingGNNState,
    output_directory: str | Path,
    training_config: GNNTrainingConfig,
    *,
    chunk_size: int = 4096,
    device: str | torch.device = "cpu",
    log_every_epochs: int = DEFAULT_LOG_EVERY_K_EPOCHS,
) -> StreamingGNNState:
    """Weights-only continuation with a fresh Adam optimizer and streamed graph."""
    if chunk_size < 1:
        raise ValueError("streaming GNN chunk_size must be positive")
    log_every_epochs = validate_log_interval(log_every_epochs)
    graph = load_streaming_online_graph(graph_directory)
    selected_device = torch.device(device)
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"requested CUDA device is unavailable: {selected_device}")
    source_manifest = source_checkpoint.manifest
    config = source_manifest.model_config_record
    source_checkpoint_sha256 = (
        source_checkpoint.manifest.checkpoint_sha256
        if isinstance(source_checkpoint, StreamingGNNState)
        else source_checkpoint.checkpoint_sha256
    )
    source_reference_graph_id = (
        source_manifest.reference_graph_id
        if isinstance(source_checkpoint, StreamingGNNState)
        else source_manifest.graph_id
    )
    if config.graph_feature_dim != graph.manifest.node_feature_dim:
        raise ValueError("source GNN width is incompatible with streamed graph")
    if source_checkpoint_sha256 != graph.manifest.reference_gnn_checkpoint_sha256:
        raise ValueError("source GNN checkpoint does not match streamed graph lineage")
    if source_reference_graph_id != graph.manifest.reference_graph_id:
        raise ValueError("source GNN checkpoint belongs to a different reference graph")
    model = EdgeFeatureSAGE(config).to(device=selected_device, dtype=torch.float32)
    model.load_state_dict(source_checkpoint.model.state_dict(), strict=True)
    model._inference_only = False
    model._training_started = True
    model._training_completed = False
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    parameters = _parameter_order(model)
    history = []
    model.train()
    log_training_start(
        "Streaming batch GNN",
        {
            "nodes": len(graph.manifest.node_ids),
            "radio_edges": graph.manifest.radio_edge_count,
            "virtual_edges": graph.manifest.virtual_edge_count,
            "negative_pairs": graph.manifest.negative_pair_count,
            "chunk_size": chunk_size,
        },
    )
    for epoch in range(1, training_config.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = _StreamedEquation4.apply(
            graph,
            config,
            training_config.objective_epsilon,
            training_config.seed + epoch * 65537,
            chunk_size,
            training_config.contrastive_temperature,
            training_config.negative_weight,
            *parameters,
        )
        if not torch.isfinite(loss).item():
            raise FloatingPointError("streamed contrastive loss is non-finite")
        loss.backward()
        grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not grads or any(not torch.isfinite(gradient).all().item() for gradient in grads):
            raise FloatingPointError("streamed GNN gradients are missing or non-finite")
        if not any(torch.count_nonzero(gradient).item() for gradient in grads):
            raise RuntimeError("streamed contrastive loss produced no parameter gradient")
        optimizer.step()
        scalar_loss = float(loss.detach().cpu())
        history.append(scalar_loss)
        log_training_epoch(
            "Streaming batch GNN",
            epoch,
            training_config.epochs,
            {
                "loss": scalar_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            },
            every_k_epochs=log_every_epochs,
        )
    model.eval()
    model._training_completed = True

    target = Path(output_directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite streaming GNN state: {target}")
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.staging-", dir=target.parent))
    try:
        # Final features are recomputed with the selected final weights and
        # eval-mode dropout, never taken from an in-training epoch.
        with tempfile.TemporaryDirectory(prefix="gufu-stream-gnn-export-") as work:
            node_path, radio_path = _stream_inference(
                graph, model, Path(work), chunk_size=chunk_size
            )
            shutil.copyfile(node_path, staging / "node_embeddings.npy")
            shutil.copyfile(radio_path, staging / "radio_edge_features.npy")
        checkpoint_path = staging / "gnn_state.pt"
        atomic_torch_save(
            {
                "state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            },
            checkpoint_path,
        )
        arrays = (
            _artifact(staging / "node_embeddings.npy", "node_embeddings.npy"),
            _artifact(staging / "radio_edge_features.npy", "radio_edge_features.npy"),
        )
        fields = dict(
            online_graph_id=graph.manifest.online_graph_id,
            dataset_id=graph.manifest.dataset_id,
            reference_batch_id=graph.manifest.reference_batch_id,
            update_batch_id=graph.manifest.update_batch_id,
            reference_graph_id=(
                graph.manifest.update_graph_id
                if not graph.manifest.update_sample_ids
                else graph.manifest.reference_graph_id
            ),
            reference_lp_result_id=graph.manifest.reference_lp_result_id,
            source_gnn_checkpoint_sha256=source_checkpoint_sha256,
            graph_build_config=graph.manifest.graph_build_config,
            vocabulary_sha256=graph.manifest.vocabulary_sha256,
            node_ids=graph.manifest.node_ids,
            radio_edge_count=graph.manifest.radio_edge_count,
            virtual_edge_count=graph.manifest.virtual_edge_count,
            negative_pair_count=graph.manifest.negative_pair_count,
            model_config_record=config,
            training_config=training_config,
            epochs_completed=len(history),
            loss_history=tuple(history),
            checkpoint_sha256=sha256_file(checkpoint_path),
            arrays=arrays,
        )
        provisional = StreamingGNNManifest.model_construct(
            artifact_id="streamgnn:" + "0" * 64,
            **fields,
        )
        artifact_id = "streamgnn:" + canonical_json_digest(
            provisional.model_dump(mode="json", exclude={"artifact_id"})
        )
        manifest = StreamingGNNManifest(artifact_id=artifact_id, **fields)
        atomic_write_json(manifest, staging / "manifest.json")
        atomic_write_json(
            {
                "format": "gufu_streaming_artifact_complete_v1",
                "identity": manifest.artifact_id,
                "manifest_sha256": sha256_file(staging / "manifest.json"),
            },
            staging / "COMPLETED.json",
        )
        os.replace(staging, target)
        return load_streaming_gnn_state(target, graph_directory, device=selected_device)
    except Exception:
        # Keep an incomplete staging directory for diagnosis; it has no marker.
        raise


def load_streaming_gnn_state(
    checkpoint_directory: str | Path,
    graph_directory: str | Path,
    *,
    device: str | torch.device = "cpu",
    verify_hashes: bool = True,
) -> StreamingGNNState:
    root = Path(checkpoint_directory)
    manifest_path = root / "manifest.json"
    marker_path = root / "COMPLETED.json"
    if not manifest_path.is_file() or not marker_path.is_file():
        raise ValueError(f"streaming GNN artifact is incomplete: {root}")
    manifest = StreamingGNNManifest.model_validate_json(read_json_text(manifest_path))
    marker = json.loads(read_json_text(marker_path))
    if marker.get("identity") != manifest.artifact_id or marker.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("streaming GNN completion marker does not match manifest")
    graph = load_streaming_online_graph(graph_directory, verify_hashes=verify_hashes)
    if (
        manifest.online_graph_id != graph.manifest.online_graph_id
        or manifest.node_ids != graph.manifest.node_ids
        or manifest.vocabulary_sha256 != graph.manifest.vocabulary_sha256
        or manifest.dataset_id != graph.manifest.dataset_id
        or manifest.update_batch_id != graph.manifest.update_batch_id
        or manifest.negative_pair_count != graph.manifest.negative_pair_count
        or manifest.graph_build_config != graph.manifest.graph_build_config
    ):
        raise ValueError("streaming GNN checkpoint is incompatible with online graph identity/order")
    checkpoint_path = root / manifest.checkpoint_name
    if not checkpoint_path.is_file() or (verify_hashes and sha256_file(checkpoint_path) != manifest.checkpoint_sha256):
        raise ValueError("streaming GNN checkpoint is missing or corrupt")
    payload = load_torch_state(checkpoint_path, map_location="cpu")
    if set(payload) != {"state_dict"}:
        raise ValueError("streaming GNN checkpoint has an unsupported payload")
    model = EdgeFeatureSAGE(manifest.model_config_record)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device=torch.device(device), dtype=torch.float32).eval()
    model._inference_only = True
    arrays = {}
    for item in manifest.arrays:
        path = root / item.name
        if not path.is_file() or path.stat().st_size != item.size_bytes:
            raise ValueError(f"streaming GNN output array is missing: {item.name}")
        if verify_hashes and sha256_file(path) != item.sha256:
            raise ValueError(f"streaming GNN output array hash mismatch: {item.name}")
        value = np.load(
            path,
            mmap_mode="r" if math.prod(item.shape) else None,
            allow_pickle=False,
        )
        if tuple(value.shape) != item.shape or value.dtype.name != item.dtype:
            raise ValueError(f"streaming GNN output shape/dtype mismatch: {item.name}")
        arrays[item.name] = value
    return StreamingGNNState(
        directory=root,
        manifest=manifest,
        model=model,
        node_embeddings=arrays["node_embeddings.npy"],
        radio_edge_features=arrays["radio_edge_features.npy"],
    )


def reencode_streaming_gnn(
    graph_directory: str | Path,
    selected_state_directory: str | Path,
    output_directory: str | Path,
    *,
    chunk_size: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Recompute complete node/radio-edge features from a selected checkpoint."""
    graph = load_streaming_online_graph(graph_directory)
    state = load_streaming_gnn_state(selected_state_directory, graph_directory)
    target = Path(output_directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite streaming GNN re-encoding: {target}")
    target.mkdir()
    try:
        node_path, radio_path = _stream_inference(graph, state.model, target, chunk_size=chunk_size)
        return (
            np.load(node_path, mmap_mode="r", allow_pickle=False),
            np.load(
                radio_path,
                mmap_mode="r" if graph.manifest.radio_edge_count else None,
                allow_pickle=False,
            ),
        )
    except Exception:
        # This directory was just created by this call and has no completion marker.
        raise


__all__ = [
    "StreamingGNNManifest",
    "StreamingGNNState",
    "infer_streaming_gnn_checkpoint_embeddings",
    "infer_streaming_gnn_embeddings",
    "infer_streaming_gnn_state",
    "load_streaming_gnn_state",
    "reencode_streaming_gnn",
    "train_streaming_gnn",
]
