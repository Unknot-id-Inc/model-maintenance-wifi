# %%
# 1. Imports, explicit run configuration, seeds, device, and output location
# Inputs: the checked-in GUFU text files and the run choices below.
# Outputs: immutable stage configuration and a reusable, validated run directory.

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from ae import AETrainingConfig
from batch_transition import TransitionConfig
from fingerprint_update import FingerprintUpdateConfig
from graph_build import GraphBuildConfig, resolve_spatial_transform
from gufu_artifacts import atomic_write_json
from gufu_data import PreprocessingConfig
from lp import LinkPredictionConfig
from pred import OrchestrationConfig
from sage_ve import GNNModelConfig, GNNTrainingConfig

# Select "real_dataset" for the normal data/budget configuration or
# "cpu_smoke" for short CPU training budgets over the same real input files.
DATA_PROFILE = "real_dataset"
# "lp_streaming" uses disk-backed link prediction AND update stages in cells
# 6-8. Cell 7 commits the first batch and cell 8 resumes that snapshot.
# "stepwise" exposes bootstrap artifacts and uses the small-graph in-memory
# demonstration. "full" delegates the complete run to pred.run_sequential.
RUN_MODE = "lp_streaming"
# Print scalar training metrics every K epochs and at each training stage's end.
LOG_EVERY_K_EPOCHS = 2
# Worker processes used for normalized-RSSI relation mining in each update.
RSSI_PAIR_NUM_WORKERS = 4
# Choose one bootstrap AE objective. Switching it selects a separate RUN_ID and
# checkpoint set, so one objective cannot silently reuse the other's weights.
AE_LOSS = "mean_squared_error"  # or "batch_frobenius_norm"
# AE architecture controls. Graph width follows ``latent_dim + 2``; every
# checkpoint and derived artifact remains scoped to this run's dimension.
AE_DROPOUT_PROBABILITY = 0.1
AE_DENSE_WIDTH = 256
AE_LATENT_DIM = 32

AE_LOSS_RUN_TAG = {
    "mean_squared_error": "mse",
    "batch_frobenius_norm": "frobenius",
}.get(AE_LOSS)

if AE_LOSS_RUN_TAG is None:
    raise ValueError("AE_LOSS must be 'mean_squared_error' or 'batch_frobenius_norm'")

DATASET_ID = "GF"
BOOTSTRAP_BATCH_ID = "fingerprint0"
INPUT_ROOT = Path("input")
BOOTSTRAP_PATH = INPUT_ROOT / DATASET_ID / f"{BOOTSTRAP_BATCH_ID}.txt"
BOOTSTRAP_PATHS = (BOOTSTRAP_PATH,)
UPDATE_INPUT_DIRECTORY = INPUT_ROOT / DATASET_ID
UPDATE_INPUT_PATTERN = "fingerprint[1-9]*.txt"
DEMONSTRATION_UPDATE_PATH = INPUT_ROOT / DATASET_ID / "fingerprint1.txt"

if DATA_PROFILE == "real_dataset":
    DEVICE = "cpu"  # Change explicitly to a supported CUDA device when desired.
    SEED = 0
    AE_ACTION = "train"  # Set to "train" to train through the selected run mode.
    GNN_ACTION = "train"  # Set to "train" to train through the selected run mode.
    AE_TRAINING_CONFIG = AETrainingConfig(
        epochs=200,
        batch_size=32,
        learning_rate=0.01,
        seed=SEED,
        shuffle=True,
        dropout_probability=AE_DROPOUT_PROBABILITY,
        dense_width=AE_DENSE_WIDTH,
        latent_dim=AE_LATENT_DIM,
        loss=AE_LOSS,
    )
    GNN_TRAINING_CONFIG = GNNTrainingConfig(
        epochs=50,
        learning_rate=0.01,
        weight_decay=0.0,
        seed=SEED,
        contrastive_temperature=0.1,
        negative_weight=1.0,
    )
    LINK_CONFIG = LinkPredictionConfig(
        max_iterations=2000, convergence_epsilon=1e-3, link_threshold=0.1
    )
    UPDATE_CONFIG = FingerprintUpdateConfig(
        alpha=0.5,
        virtual_edge_sigma=0.9,
        contrastive_temperature=0.1,
        negative_weight=1.0,
        gnn_epochs=1,
        updater_epochs=100,
        ae_epochs=25,
        seed=SEED,
    )
    TRANSITION_CONFIG = TransitionConfig(
        link_prediction=LINK_CONFIG,
        ae_finetune_epochs=25,
        ae_finetune_seed=1701,
        reinitialize_seed=1702,
    )
elif DATA_PROFILE == "cpu_smoke":
    # This profile uses input/GF/fingerprint*.txt; it does not synthesize data.
    DEVICE = "cpu"
    SEED = 23
    AE_ACTION = "train"
    GNN_ACTION = "train"
    AE_TRAINING_CONFIG = AETrainingConfig(
        epochs=1,
        batch_size=32,
        learning_rate=0.001,
        seed=SEED,
        shuffle=False,
        dropout_probability=AE_DROPOUT_PROBABILITY,
        dense_width=AE_DENSE_WIDTH,
        latent_dim=AE_LATENT_DIM,
        loss=AE_LOSS,
    )
    GNN_TRAINING_CONFIG = GNNTrainingConfig(
        epochs=1,
        learning_rate=0.001,
        weight_decay=0.0,
        seed=SEED,
        contrastive_temperature=0.1,
        negative_weight=1.0,
    )
    LINK_CONFIG = LinkPredictionConfig(
        max_iterations=8, convergence_epsilon=1e9, link_threshold=0.0
    )
    UPDATE_CONFIG = FingerprintUpdateConfig(
        hidden_dim=16,
        alpha=0.5,
        virtual_edge_sigma=0.9,
        contrastive_temperature=0.1,
        negative_weight=1.0,
        gnn_epochs=1,
        updater_epochs=1,
        ae_epochs=1,
        seed=SEED,
    )
    TRANSITION_CONFIG = TransitionConfig(
        link_prediction=LINK_CONFIG,
        ae_finetune_epochs=1,
        ae_finetune_seed=SEED + 1,
        reinitialize_seed=SEED + 2,
    )
else:
    raise ValueError("DATA_PROFILE must be 'real_dataset' or 'cpu_smoke'")

if DATA_PROFILE == "cpu_smoke":
    # One explicit batch from the real input collection keeps the smoke bounded.
    SELECTED_UPDATE_PATHS = (DEMONSTRATION_UPDATE_PATH,)
    SELECTED_UPDATE_DIRECTORY = None
else:
    SELECTED_UPDATE_PATHS = None
    SELECTED_UPDATE_DIRECTORY = UPDATE_INPUT_DIRECTORY

if RUN_MODE not in {"stepwise", "full", "lp_streaming"}:
    raise ValueError("RUN_MODE must be 'stepwise', 'full', or 'lp_streaming'")
if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError(f"requested CUDA device is unavailable: {DEVICE}")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
if DEVICE == "cpu":
    torch.set_num_threads(1)

# Stable run IDs make cell re-execution reuse the same validated artifacts.
# Change this string deliberately to create a separate run and train fresh models.
RUN_ID = (
    f"cpu-smoke-real-GF-fingerprint0-ae-v2-{AE_LOSS_RUN_TAG}-rssi-pairs-v1-spatial-v1-softplus-cosine-t0p1-nw1-"
    f"d{AE_DENSE_WIDTH}-z{AE_LATENT_DIM}-drop{str(AE_DROPOUT_PROBABILITY).replace('.', 'p')}"
    if DATA_PROFILE == "cpu_smoke"
    else f"real-GF-fingerprint0-ae-v2-{AE_LOSS_RUN_TAG}-rssi-pairs-v1-spatial-v1-softplus-cosine-t0p1-nw1-"
    f"d{AE_DENSE_WIDTH}-z{AE_LATENT_DIM}-drop{str(AE_DROPOUT_PROBABILITY).replace('.', 'p')}"
)
RUN_DIR = Path("output") / "notebook_runs" / DATASET_ID / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=True)
PIPELINE_PROGRESS_LOG_PATH = RUN_DIR / "pipeline_progress.log"
PREPARED_PATH = RUN_DIR / "prepared_bootstrap.json"
ENCODER_FEATURE_PATH = RUN_DIR / "encoder_features.json"
GRAPH_STATE_PATH = RUN_DIR / "graph_state.json"
GNN_FEATURE_PATH = RUN_DIR / "gnn_features.json"
STREAMING_LP_INPUT_DIR = RUN_DIR / "link_prediction_input"
STREAMING_LP_RESULT_DIR = RUN_DIR / "link_prediction_result"

EXTERNAL_AE_CHECKPOINT = (
    Path("output") / "checkpoints" / DATASET_ID / BOOTSTRAP_BATCH_ID / "ae.pt"
)
EXTERNAL_GNN_CHECKPOINT = (
    Path("output") / "checkpoints" / DATASET_ID / BOOTSTRAP_BATCH_ID / "gnn.pt"
)
STEPWISE_AE_CHECKPOINT = RUN_DIR / "checkpoints" / "ae.pt"
STEPWISE_GNN_CHECKPOINT = RUN_DIR / "checkpoints" / "gnn.pt"
AE_CHECKPOINT_PATH = (
    STEPWISE_AE_CHECKPOINT if AE_ACTION == "train" else EXTERNAL_AE_CHECKPOINT
)
GNN_CHECKPOINT_PATH = (
    STEPWISE_GNN_CHECKPOINT if GNN_ACTION == "train" else EXTERNAL_GNN_CHECKPOINT
)

PREPROCESSING_CONFIG = PreprocessingConfig(
    dataset_id=DATASET_ID,
    batch_id=BOOTSTRAP_BATCH_ID,
    batch_role="bootstrap_reference",
    input_format="gufu_db",
    rssi_threshold_dbm=None,
    prune_randomized_bssids=False,
    malformed_record_policy="drop",
    empty_scan_policy="keep",
    duplicate_bssid_policy="strongest",
    normalization_mode="paper_ae",
)
GRAPH_CONFIG = GraphBuildConfig(
    virtual_edge_strategy="paper_cosine_threshold",
    virtual_edge_sigma=0.9,
    similarity_feature_space="normalized_rssi_cosine_min_overlap_v1",
    negative_similarity_threshold=0.5,
    minimum_shared_aps=3,
    max_negative_pairs_per_sample=8,
    pseudo_location_seed=SEED,
    spatial_normalization_enabled=True,
    spatial_transform_path=str(Path("artifacts") / "spatial" / f"{DATASET_ID}.json"),
    position_weight=1.0,
)
GRAPH_CONFIG, _SPATIAL_TRANSFORM = resolve_spatial_transform(GRAPH_CONFIG, DATASET_ID)
GNN_MODEL_CONFIG = GNNModelConfig(
    graph_feature_dim=AE_LATENT_DIM + 2,
    edge_feature_dim=AE_LATENT_DIM + 2,
    node_output_dim=AE_LATENT_DIM + 2,
    edge_output_dim=AE_LATENT_DIM + 2,
    num_layers=2,
    dropout_probability=0.1,
    initialization_seed=SEED,
)
ORCHESTRATION_CONFIG = OrchestrationConfig(
    dataset_id=DATASET_ID,
    input_format=PREPROCESSING_CONFIG.input_format,
    rssi_threshold_dbm=PREPROCESSING_CONFIG.rssi_threshold_dbm,
    prune_randomized_bssids=PREPROCESSING_CONFIG.prune_randomized_bssids,
    malformed_record_policy=PREPROCESSING_CONFIG.malformed_record_policy,
    empty_scan_policy=PREPROCESSING_CONFIG.empty_scan_policy,
    duplicate_bssid_policy=PREPROCESSING_CONFIG.duplicate_bssid_policy,
    normalization_mode=PREPROCESSING_CONFIG.normalization_mode,
    ae_action=AE_ACTION,
    gnn_action=GNN_ACTION,
    ae_training=AE_TRAINING_CONFIG,
    graph_build=GRAPH_CONFIG,
    gnn_model=GNN_MODEL_CONFIG,
    gnn_training=GNN_TRAINING_CONFIG,
    link_prediction=LINK_CONFIG,
    fingerprint_update=UPDATE_CONFIG,
    transition=TRANSITION_CONFIG,
    device=DEVICE,
)

from streaming_pipeline import StreamingPipelineConfig

STREAMING_PIPELINE_CONFIG = StreamingPipelineConfig(
    dataset_id=DATASET_ID,
    preprocessing=PREPROCESSING_CONFIG,
    graph_build=GRAPH_CONFIG,
    gnn_training=GNN_TRAINING_CONFIG,
    link_prediction=LINK_CONFIG,
    fingerprint_update=UPDATE_CONFIG,
    transition=TRANSITION_CONFIG,
    device=DEVICE,
    seed=SEED,
    gnn_chunk_size=4096,
    updater_chunk_size=4096,
    ae_chunk_size=256,
    encoder_batch_size=AE_TRAINING_CONFIG.batch_size,
    log_every_epochs=LOG_EVERY_K_EPOCHS,
    rssi_pair_num_workers=RSSI_PAIR_NUM_WORKERS,
)
STREAMING_PIPELINE_CONFIG_PATH = RUN_DIR / "streaming_pipeline_config.json"
if STREAMING_PIPELINE_CONFIG_PATH.exists():
    existing_streaming_config = StreamingPipelineConfig.model_validate_json(
        STREAMING_PIPELINE_CONFIG_PATH.read_text(encoding="utf-8")
    )
    if existing_streaming_config != STREAMING_PIPELINE_CONFIG:
        raise ValueError(
            "streaming pipeline configuration changed for this RUN_ID; choose a new RUN_ID"
        )
else:
    atomic_write_json(STREAMING_PIPELINE_CONFIG, STREAMING_PIPELINE_CONFIG_PATH)

if RUN_MODE == "stepwise":
    PIPELINE_OUTPUT_ROOT = RUN_DIR / "one_batch_demonstration"
elif RUN_MODE == "lp_streaming":
    PIPELINE_OUTPUT_ROOT = RUN_DIR / "streaming_pipeline"
elif DATA_PROFILE == "cpu_smoke":
    PIPELINE_OUTPUT_ROOT = RUN_DIR / "cpu_smoke_pipeline"
else:
    PIPELINE_OUTPUT_ROOT = Path("output") / "pipelines" / DATASET_ID / RUN_ID
PIPELINE_RESUME = True

RUN_EVALUATION = False  # Evaluation is an optional, post-update cell 9 action.
print(f"Profile={DATA_PROFILE}; mode={RUN_MODE}; device={DEVICE}; seed={SEED}")
print(f"Bootstrap AE training objective={AE_LOSS}")
print(
    f"AE architecture settings: dense_width={AE_DENSE_WIDTH}; "
    f"latent_dim={AE_LATENT_DIM}; dropout_probability={AE_DROPOUT_PROBABILITY}"
)
print(
    f"Training progress: every {LOG_EVERY_K_EPOCHS} epochs, plus each stage's final epoch"
)
print(f"RSSI relation mining workers: {RSSI_PAIR_NUM_WORKERS}")
print(f"Bootstrap input: {BOOTSTRAP_PATH}")
print(
    f"Update selection: {SELECTED_UPDATE_PATHS or SELECTED_UPDATE_DIRECTORY} "
    f"(pattern={UPDATE_INPUT_PATTERN if SELECTED_UPDATE_DIRECTORY else 'explicit one-batch smoke'})"
)
print(f"Notebook artifacts: {RUN_DIR} (change RUN_ID for a fresh isolated run)")
print(f"Pipeline progress log: {PIPELINE_PROGRESS_LOG_PATH}")
print(f"Sequential pipeline root: {PIPELINE_OUTPUT_ROOT} (resume={PIPELINE_RESUME})")

# %%
# 2. Prepare/load the authoritative bootstrap dataset
# Inputs: BOOTSTRAP_PATH and PREPROCESSING_CONFIG from cell 1.
# Outputs: validated PreparedDataset and PREPARED_PATH; no downstream parser.

from gufu_artifacts import canonical_json_digest
from gufu_data import (
    load_prepared_dataset,
    prepare_dataset_files,
    save_prepared_dataset,
)

if not BOOTSTRAP_PATH.is_file():
    raise FileNotFoundError(f"bootstrap fingerprint file is missing: {BOOTSTRAP_PATH}")
source_prepared_dataset = prepare_dataset_files(
    BOOTSTRAP_PATHS, PREPROCESSING_CONFIG, source_root=INPUT_ROOT
)
if PREPARED_PATH.is_file():
    prepared_dataset = load_prepared_dataset(PREPARED_PATH)
    if canonical_json_digest(prepared_dataset) != canonical_json_digest(
        source_prepared_dataset
    ):
        raise ValueError(
            "saved prepared artifact no longer matches the selected input/config; "
            "change RUN_ID for a fresh run"
        )
else:
    prepared_dataset = source_prepared_dataset
    save_prepared_dataset(prepared_dataset, PREPARED_PATH)
    prepared_dataset = load_prepared_dataset(PREPARED_PATH)

assert prepared_dataset.config == PREPROCESSING_CONFIG
assert prepared_dataset.config.batch_role == "bootstrap_reference"
assert prepared_dataset.bssid_vocabulary == tuple(
    sorted(prepared_dataset.bssid_vocabulary)
)
assert len(prepared_dataset.ap_ids) == len(prepared_dataset.bssid_vocabulary)
assert len({scan.scan_id for scan in prepared_dataset.scans}) == len(
    prepared_dataset.scans
)
assert all(
    len(scan.rss_dbm) == len(prepared_dataset.bssid_vocabulary)
    and len(scan.observation_mask) == len(prepared_dataset.bssid_vocabulary)
    for scan in prepared_dataset.scans
)
print(f"Prepared input: {BOOTSTRAP_PATH} -> {PREPARED_PATH}")
print(prepared_dataset.summary())
print(
    f"Observed measurements={prepared_dataset.stats.observed_readings}; "
    f"mask shape=({len(prepared_dataset.scans)}, {len(prepared_dataset.bssid_vocabulary)}); "
    f"dropped rows={prepared_dataset.stats.dropped_records}"
)
print(
    f"Ordered AP IDs: {prepared_dataset.ap_ids[:8]}{' ...' if len(prepared_dataset.ap_ids) > 8 else ''}"
)

# %%
# 3. Train/load the AE and inspect its complete encoder output
# Inputs: the cell 2 PreparedDataset and the selected AE checkpoint.
# Outputs in stepwise mode: compatible AE checkpoint, every scan's configured latent feature,
# and a small reconstruction diagnostic. Full mode performs this once in cell 8.

from ae import (
    encode_prepared_dataset,
    load_ae_checkpoint,
    reconstruction_diagnostic,
    train_autoencoder,
)
from gufu_artifacts import canonical_json_digest

ae_checkpoint = None
ae_features = None
ae_diagnostic = None
if RUN_MODE == "full":
    print(
        "Full mode: AE training/loading and complete inference are delegated to cell 8."
    )
else:
    if AE_ACTION == "train":
        if AE_CHECKPOINT_PATH.is_file():
            ae_checkpoint = load_ae_checkpoint(
                AE_CHECKPOINT_PATH, prepared_dataset, device=DEVICE
            )
            if (
                ae_checkpoint.manifest.training_config.model_dump()
                != AE_TRAINING_CONFIG.model_dump()
            ):
                raise ValueError(
                    "existing stepwise AE checkpoint uses a different training config; "
                    "start a fresh run directory"
                )
            print("Reusing the validated stepwise AE checkpoint.")
        else:
            ae_checkpoint = train_autoencoder(
                prepared_dataset,
                AE_CHECKPOINT_PATH,
                AE_TRAINING_CONFIG,
                initialization="fresh",
                device=DEVICE,
                log_every_epochs=LOG_EVERY_K_EPOCHS,
            ).checkpoint
    elif AE_ACTION == "load":
        ae_checkpoint = load_ae_checkpoint(
            AE_CHECKPOINT_PATH, prepared_dataset, device=DEVICE
        )
    else:
        raise ValueError("AE_ACTION must be 'train' or 'load'")

    if ENCODER_FEATURE_PATH.is_file():
        from ae import load_encoder_features

        ae_features = load_encoder_features(ENCODER_FEATURE_PATH)
    else:
        ae_features = encode_prepared_dataset(
            prepared_dataset,
            AE_CHECKPOINT_PATH,
            ENCODER_FEATURE_PATH,
            batch_size=AE_TRAINING_CONFIG.batch_size,
            device=DEVICE,
        )
    ae_diagnostic = reconstruction_diagnostic(
        prepared_dataset, AE_CHECKPOINT_PATH, max_scans=4, batch_size=2, device=DEVICE
    )
    assert ae_features.scan_ids == tuple(
        scan.scan_id for scan in prepared_dataset.scans
    )
    assert len(ae_features.feature_matrix) == len(prepared_dataset.scans)
    assert all(
        len(row) == ae_features.feature_dim for row in ae_features.feature_matrix
    )
    assert (
        ae_features.feature_dim
        == ae_checkpoint.manifest.model.latent_dim
        == AE_LATENT_DIM
    )
    assert ae_features.bssid_vocabulary == prepared_dataset.bssid_vocabulary
    assert ae_features.vocabulary_sha256 == canonical_json_digest(
        prepared_dataset.bssid_vocabulary
    )
    assert ae_features.normalization == prepared_dataset.config.normalization_mode
    assert ae_features.source_prepared_sha256 == canonical_json_digest(prepared_dataset)
    assert ae_features.checkpoint_sha256 == ae_checkpoint.checkpoint_sha256
    assert (
        ae_checkpoint.manifest.normalization
        == prepared_dataset.config.normalization_mode
    )
    assert ae_checkpoint.manifest.model.input_dim == len(
        prepared_dataset.bssid_vocabulary
    )
    print(f"AE input: {PREPARED_PATH}")
    print(f"AE checkpoint: {AE_CHECKPOINT_PATH}")
    print(f"Complete encoder feature artifact: {ENCODER_FEATURE_PATH}")
    print(
        f"Encoded {len(ae_features.scan_ids)} scans; shape="
        f"({len(ae_features.feature_matrix)}, {ae_features.feature_dim}); normalization={ae_features.normalization}"
    )
    print(
        f"AE checkpoint SHA-256={ae_checkpoint.checkpoint_sha256}; "
        f"epochs={ae_checkpoint.manifest.epochs_completed}; "
        f"final loss={ae_checkpoint.manifest.loss_history[-1]:.6g}"
    )
    print(f"Reconstruction diagnostic: {ae_diagnostic}")

# %%
# 4. Build/load the graph and inspect stable identity and typed edges
# Inputs: prepared scans plus cell 3 AE features joined by scan ID.
# Outputs in stepwise mode: validated GraphState with typed physical and
# temporary virtual edges. Full mode builds this once in cell 8.

from ae import load_encoder_features
from graph_build import build_graph_state, load_graph_state, save_graph_state

graph_state = None
encoder_feature_artifact = None
if RUN_MODE == "full":
    print("Full mode: graph construction is delegated to cell 8.")
else:
    encoder_feature_artifact = load_encoder_features(ENCODER_FEATURE_PATH)
    candidate_graph_state = build_graph_state(
        prepared_dataset, encoder_feature_artifact, GRAPH_CONFIG
    )
    if GRAPH_STATE_PATH.is_file():
        graph_state = load_graph_state(GRAPH_STATE_PATH)
        if graph_state.graph_id != candidate_graph_state.graph_id:
            raise ValueError(
                "saved graph does not match current prepared data/features/config; "
                "start a fresh run directory"
            )
    else:
        graph_state = candidate_graph_state
        save_graph_state(graph_state, GRAPH_STATE_PATH)
        graph_state = load_graph_state(GRAPH_STATE_PATH)
    assert graph_state.sample_node_ids == tuple(
        scan.scan_id for scan in prepared_dataset.scans
    )
    assert graph_state.node_id_to_row == tuple(
        (node_id, row) for row, node_id in enumerate(graph_state.node_ids)
    )
    assert graph_state.node_feature_tensor().shape == (
        graph_state.node_count,
        graph_state.node_feature_dim,
    )
    assert graph_state.node_feature_dim == graph_state.encoder_feature_dim + 2
    assert graph_state.bssid_vocabulary == prepared_dataset.bssid_vocabulary
    assert graph_state.normalization == prepared_dataset.config.normalization_mode
    assert (
        graph_state.encoder_checkpoint_sha256
        == encoder_feature_artifact.checkpoint_sha256
    )
    assert graph_state.encoder_feature_artifact_sha256 == canonical_json_digest(
        encoder_feature_artifact
    )
    assert graph_state.source_prepared_sha256 == canonical_json_digest(prepared_dataset)
    print(f"Graph input: {PREPARED_PATH}, {ENCODER_FEATURE_PATH}")
    print(f"Graph artifact: {GRAPH_STATE_PATH}; identity={graph_state.graph_id}")
    print(
        f"Nodes={graph_state.node_count} (samples={len(graph_state.sample_node_ids)}, "
        f"APs={len(graph_state.ap_node_ids)}); feature shape="
        f"({graph_state.node_count}, {graph_state.node_feature_dim}); "
        f"radio edges={graph_state.physical_edge_count}; "
        f"temporary virtual edges={graph_state.virtual_edge_count}"
    )

# %%
# 5. Train/load the GNN and inspect learned state and provenance
# Inputs: the cell 4 GraphState and selected model/checkpoint configuration.
# Outputs in stepwise mode: selected GNN checkpoint and eval-mode node/edge
# features aligned to graph IDs. Full mode performs this once in cell 8.

from sage_ve import (
    EdgeFeatureSAGE,
    encode_graph_state,
    load_gnn_checkpoint,
    load_gnn_features,
    train_gnn,
)

gnn_checkpoint = None
gnn_features = None
if RUN_MODE == "full":
    print("Full mode: GNN training/loading and export are delegated to cell 8.")
else:
    if GNN_ACTION == "train":
        if GNN_CHECKPOINT_PATH.is_file():
            gnn_checkpoint = load_gnn_checkpoint(
                GNN_CHECKPOINT_PATH, graph_state, device=DEVICE
            )
            if (
                gnn_checkpoint.manifest.model_config_record != GNN_MODEL_CONFIG
                or gnn_checkpoint.manifest.training_config != GNN_TRAINING_CONFIG
            ):
                raise ValueError(
                    "existing stepwise GNN checkpoint uses a different model/training config; "
                    "start a fresh run directory"
                )
            print("Reusing the validated stepwise GNN checkpoint.")
        else:
            training_result = train_gnn(
                graph_state,
                EdgeFeatureSAGE(GNN_MODEL_CONFIG),
                GNN_CHECKPOINT_PATH,
                GNN_FEATURE_PATH,
                training_config=GNN_TRAINING_CONFIG,
                device=DEVICE,
                log_every_epochs=LOG_EVERY_K_EPOCHS,
            )
            gnn_checkpoint = training_result.checkpoint
            gnn_features = training_result.features
    elif GNN_ACTION == "load":
        gnn_checkpoint = load_gnn_checkpoint(
            GNN_CHECKPOINT_PATH, graph_state, device=DEVICE
        )
    else:
        raise ValueError("GNN_ACTION must be 'train' or 'load'")

    if gnn_features is None:
        if GNN_FEATURE_PATH.is_file():
            gnn_features = load_gnn_features(GNN_FEATURE_PATH)
        else:
            gnn_features = encode_graph_state(
                graph_state,
                GNN_CHECKPOINT_PATH,
                GNN_FEATURE_PATH,
                device=DEVICE,
            )
    assert gnn_features.graph_id == graph_state.graph_id
    assert gnn_features.node_ids == graph_state.node_ids
    assert gnn_features.node_id_to_row == graph_state.node_id_to_row
    assert len(gnn_features.node_feature_matrix) == graph_state.node_count
    assert all(
        len(row) == graph_state.node_feature_dim
        for row in gnn_features.node_feature_matrix
    )
    assert gnn_features.radio_edge_ids == tuple(
        edge.edge_id for edge in graph_state.radio_edges
    )
    assert (
        len(gnn_features.radio_edge_feature_matrix) == graph_state.physical_edge_count
    )
    assert gnn_features.virtual_edge_ids == tuple(
        edge.edge_id for edge in graph_state.virtual_edges
    )
    assert gnn_features.vocabulary_sha256 == graph_state.vocabulary_sha256
    assert gnn_features.normalization == graph_state.normalization
    assert (
        gnn_features.encoder_checkpoint_sha256 == graph_state.encoder_checkpoint_sha256
    )
    assert gnn_features.checkpoint_sha256 == gnn_checkpoint.checkpoint_sha256
    assert gnn_checkpoint.manifest.graph_id == graph_state.graph_id
    assert gnn_checkpoint.manifest.vocabulary_sha256 == graph_state.vocabulary_sha256
    print(f"GNN input: {GRAPH_STATE_PATH}; checkpoint: {GNN_CHECKPOINT_PATH}")
    print(f"Learned state artifact: {GNN_FEATURE_PATH}")
    print(
        f"Learned nodes={len(gnn_features.node_ids)} × "
        f"{graph_state.node_feature_dim}; learned radio-edge vectors="
        f"{len(gnn_features.radio_edge_feature_matrix)}; "
        f"checkpoint={gnn_features.checkpoint_sha256}"
    )
    print(
        f"GNN training graph={gnn_checkpoint.manifest.graph_id}; "
        f"epochs={gnn_checkpoint.manifest.epochs_completed}; "
        f"final loss={gnn_checkpoint.manifest.loss_history[-1]:.6g}"
    )

# %%
# 6. Inspect link modification using the matching graph and learned features
# Inputs: cell 4 GraphState, cell 5 ID-aligned GNNFeatureArtifact, LINK_CONFIG.
# Outputs: compact disk-backed G/F, candidate, delta, and updated-edge artifacts.
# The mathematical policy is the same as lp.predict_edges; only storage changes.
# Full cell-8 orchestration still uses its legacy in-memory result contract.

from itertools import islice

from lp_streaming import (
    load_streaming_lp_input,
    load_streaming_lp_result,
    predict_edges_streaming,
    save_streaming_lp_input,
    validate_streaming_lp_input,
)


def _report_pipeline_progress(message):
    print(message, flush=True)
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with PIPELINE_PROGRESS_LOG_PATH.open("a", encoding="utf-8") as progress_log:
        progress_log.write(f"{timestamp} {message}\n")
        progress_log.flush()


streaming_lp_result = None
if RUN_MODE == "full":
    print(
        "Full mode: cell 8 currently uses the orchestration API's in-memory "
        "link-prediction snapshot contract. Use stepwise cell 6 for the "
        "disk-backed LP stage; the downstream full orchestrator is not yet out-of-core."
    )
else:
    if globals().get("graph_state") is None:
        from graph_build import load_graph_state

        graph_state = load_graph_state(GRAPH_STATE_PATH)
    if globals().get("gnn_features") is None:
        from sage_ve import load_gnn_features

        gnn_features = load_gnn_features(GNN_FEATURE_PATH)
    if globals().get("gnn_checkpoint") is None:
        from sage_ve import load_gnn_checkpoint

        gnn_checkpoint = load_gnn_checkpoint(
            GNN_CHECKPOINT_PATH, graph_state, device=DEVICE
        )
    assert graph_state.graph_id == gnn_features.graph_id
    assert gnn_features.node_ids == graph_state.node_ids
    assert gnn_features.checkpoint_sha256 == gnn_checkpoint.checkpoint_sha256
    possible_sample_ap_pairs = len(graph_state.sample_node_ids) * len(
        graph_state.ap_node_ids
    )
    _report_pipeline_progress(
        "[link prediction] starting candidate scoring: "
        f"{len(graph_state.sample_node_ids)} samples × "
        f"{len(graph_state.ap_node_ids)} APs (upper bound="
        f"{possible_sample_ap_pairs:,}); "
        f"radio edges={graph_state.physical_edge_count:,}, "
        f"virtual edges={graph_state.virtual_edge_count:,}"
    )
    if STREAMING_LP_INPUT_DIR.exists():
        _report_pipeline_progress(
            "[link prediction] validating the saved streaming input bundle"
        )
        input_manifest = validate_streaming_lp_input(
            graph_state, gnn_features, STREAMING_LP_INPUT_DIR
        )
    else:
        _report_pipeline_progress(
            "[link prediction] creating the streaming input bundle"
        )
        input_manifest = save_streaming_lp_input(
            graph_state, gnn_features, STREAMING_LP_INPUT_DIR
        )
    assert input_manifest.graph_id == graph_state.graph_id
    assert input_manifest.gnn_checkpoint_sha256 == gnn_features.checkpoint_sha256
    assert input_manifest.vocabulary_sha256 == graph_state.vocabulary_sha256
    if STREAMING_LP_RESULT_DIR.exists():
        _report_pipeline_progress(
            "[link prediction] validating the saved proposal; the solver will be skipped if it matches"
        )
        streaming_lp_result = load_streaming_lp_result(
            STREAMING_LP_RESULT_DIR,
            input_directory=STREAMING_LP_INPUT_DIR,
        )
        if streaming_lp_result.manifest.config != LINK_CONFIG:
            raise ValueError(
                "existing streaming link-prediction result uses a different "
                "configuration; change RUN_ID for a new result"
            )
    else:
        _report_pipeline_progress(
            "[link prediction] no saved proposal found; starting a new run"
        )
        streaming_lp_result = predict_edges_streaming(
            STREAMING_LP_INPUT_DIR,
            STREAMING_LP_RESULT_DIR,
            LINK_CONFIG,
            progress_callback=_report_pipeline_progress,
        )
    lp_manifest = streaming_lp_result.manifest
    assert lp_manifest.parent_graph_id == graph_state.graph_id
    assert lp_manifest.gnn_checkpoint_sha256 == gnn_features.checkpoint_sha256
    assert lp_manifest.vocabulary_sha256 == graph_state.vocabulary_sha256
    _report_pipeline_progress(
        f"Link proposal for graph={lp_manifest.parent_graph_id}: "
        f"outcome={lp_manifest.outcome}, termination={lp_manifest.termination}, "
        f"iterations={lp_manifest.iterations}, residuals="
        f"({lp_manifest.goodness_residual:.6g}, {lp_manifest.fairness_residual:.6g})"
    )
    _report_pipeline_progress(
        f"candidate scores={lp_manifest.candidate_count:,}, "
        f"added={lp_manifest.added_count:,}, removed={lp_manifest.removed_count:,}, "
        f"AP removals={len(lp_manifest.ap_removals):,}; "
        f"threshold={lp_manifest.threshold_value} {lp_manifest.threshold_units}; "
        f"result radio edges={lp_manifest.updated_edge_count:,}"
    )
    if lp_manifest.failure_reason:
        _report_pipeline_progress(f"Link prediction failure: {lp_manifest.failure_reason}")
    _report_pipeline_progress(f"Streaming input: {STREAMING_LP_INPUT_DIR}")
    _report_pipeline_progress(f"Streaming result: {STREAMING_LP_RESULT_DIR}")
    for proposal in islice(streaming_lp_result.iter_added_edges(), 10):
        _report_pipeline_progress(
            f"ADD {proposal.sample_node_id} -- {proposal.ap_node_id}: "
            f"weight={proposal.new_weight:.5g}"
        )
    for proposal in islice(streaming_lp_result.iter_removed_edges(), 10):
        _report_pipeline_progress(
            f"REMOVE {proposal.sample_node_id} -- {proposal.ap_node_id}: "
            f"old={proposal.old_weight:.5g}; predicted={proposal.predicted_weight:.5g}"
        )
    _report_pipeline_progress(
        "[link prediction] compact proposal is complete. In lp_streaming mode, "
        "cells 7-8 pass this exact LP input/result into the disk-backed batch "
        "orchestrator; no in-memory LP rerun is used.",
    )
    # Release the expanded Python objects before iterating the disk-backed LP
    # result. Their validated source artifacts remain in RUN_DIR.
    del graph_state, gnn_features, gnn_checkpoint

# %%
# 7. Demonstrate the updater and commit the first batch through the selected API
# Inputs: bootstrap prepared data, the checkpoint/LP artifacts from cells 3-6,
# and the first selected update file. In lp_streaming mode this is the first
# committed snapshot; cell 8 resumes this exact snapshot and continues.

from pred import run_sequential
from streaming_pipeline import run_streaming_sequential, select_update_paths

demonstration_result = None
demonstration_stage_events = []
selected_streaming_update_paths = select_update_paths(
    update_paths=SELECTED_UPDATE_PATHS,
    update_directory=SELECTED_UPDATE_DIRECTORY,
    pattern=UPDATE_INPUT_PATTERN,
    bootstrap_paths=BOOTSTRAP_PATHS,
)
streaming_bootstrap_gnn = None
if RUN_MODE == "lp_streaming":
    if not selected_streaming_update_paths:
        print(
            "No update files selected; cell 8 will validate/bootstrap the streaming run."
        )
    else:
        from sage_ve import load_streaming_gnn_checkpoint

        streaming_bootstrap_gnn = load_streaming_gnn_checkpoint(
            GNN_CHECKPOINT_PATH,
            STREAMING_LP_INPUT_DIR,
            device=DEVICE,
        )
        first_update_path = selected_streaming_update_paths[0]
        demonstration_result = run_streaming_sequential(
            PIPELINE_OUTPUT_ROOT,
            STREAMING_PIPELINE_CONFIG,
            bootstrap_prepared=prepared_dataset,
            bootstrap_ae_checkpoint_path=AE_CHECKPOINT_PATH,
            bootstrap_gnn_checkpoint=streaming_bootstrap_gnn,
            bootstrap_lp_input_directory=STREAMING_LP_INPUT_DIR,
            bootstrap_lp_result_directory=STREAMING_LP_RESULT_DIR,
            bootstrap_paths=BOOTSTRAP_PATHS,
            update_paths=(first_update_path,),
            source_root=INPUT_ROOT,
            progress_callback=_report_pipeline_progress,
        )
        print(
            f"First streamed batch={first_update_path}; "
            f"snapshot={demonstration_result.latest_snapshot_directory}; "
            f"completed={demonstration_result.completed_batch_ids}; "
            f"skipped={demonstration_result.skipped_completed_batch_ids}"
        )
elif RUN_MODE == "full":
    print("Full mode: the in-memory one-batch demonstration is delegated to cell 8.")
else:
    if not DEMONSTRATION_UPDATE_PATH.is_file():
        raise FileNotFoundError(
            f"configured one-batch demonstration input is missing: {DEMONSTRATION_UPDATE_PATH}"
        )
    # Stepwise cells may have trained models. The isolated orchestrator loads
    # those exact checkpoints, so it does not train the bootstrap models twice.
    demonstration_config = ORCHESTRATION_CONFIG.model_copy(
        update={"ae_action": "load", "gnn_action": "load"}
    )

    def _report_demonstration_stage(stage, staging_path):
        demonstration_stage_events.append(stage)
        next_stage = {
            "bootstrap_preprocessing": "bootstrap AE training/loading and encoding",
            "bootstrap_ae_encoding": "bootstrap graph construction",
            "bootstrap_graph_aggregation": "bootstrap GNN training/loading and encoding",
            "bootstrap_gnn": "bootstrap link prediction and edge-delta serialization",
            "bootstrap_edge_prediction": "bootstrap transition-state creation",
            "bootstrap_transition_state": "bootstrap snapshot manifest and commit",
            "bootstrap_before_commit": "snapshot artifact hashing and completion marker",
            "before_completion_marker": "snapshot completion marker and atomic publish",
            "update_shared_ae_encoding": "update graph construction/aggregation",
            "update_graph_aggregation": "fingerprint-updater training",
            "update_updater_training": "AP inference and edge modification",
            "update_edge_prediction": "AE vocabulary migration and fingerprint reconstruction",
            "update_ae_realign": "post-transition AE/GNN feature refresh",
            "update_post_transition_features": "temporary-node cleanup and validation",
            "update_cleanup": "batch snapshot manifest and commit",
            "update_before_commit": "snapshot artifact hashing and completion marker",
        }.get(stage, "orchestrator next step")
        print(
            f"[pipeline] completed stage={stage}; "
            f"next_expected={next_stage}; "
            f"staging_snapshot={Path(staging_path).name}",
            flush=True,
        )

    print(
        "[pipeline] starting isolated one-batch demonstration; "
        f"update={DEMONSTRATION_UPDATE_PATH}; "
        f"AE checkpoint={AE_CHECKPOINT_PATH}; "
        f"GNN checkpoint={GNN_CHECKPOINT_PATH}",
        flush=True,
    )
    demonstration_result = run_sequential(
        PIPELINE_OUTPUT_ROOT,
        demonstration_config,
        bootstrap_prepared=prepared_dataset,
        bootstrap_paths=BOOTSTRAP_PATHS,
        bootstrap_source_paths=BOOTSTRAP_PATHS,
        update_paths=(DEMONSTRATION_UPDATE_PATH,),
        source_root=INPUT_ROOT,
        ae_checkpoint_path=AE_CHECKPOINT_PATH,
        gnn_checkpoint_path=GNN_CHECKPOINT_PATH,
        resume=True,
        stage_observer=_report_demonstration_stage,
        log_every_epochs=LOG_EVERY_K_EPOCHS,
    )
    demonstration_state = demonstration_result.state
    assert demonstration_state.prepared.config.dataset_id == DATASET_ID
    assert demonstration_state.transition_state.reference_scan_ids == tuple(
        scan.scan_id for scan in prepared_dataset.scans
    )
    assert demonstration_state.manifest.update_scan_count >= 1
    assert (
        demonstration_state.manifest.bssid_vocabulary
        == demonstration_state.prepared.bssid_vocabulary
    )
    assert demonstration_state.encoder_features.scan_ids == tuple(
        scan.scan_id for scan in demonstration_state.prepared.scans
    )
    assert (
        demonstration_state.gnn_features.node_ids
        == demonstration_state.graph_state.node_ids
    )
    print(f"Incoming batch: {DEMONSTRATION_UPDATE_PATH}")
    print(
        f"Isolated demonstration root: {demonstration_state.snapshot_directory.parent.parent}"
    )
    print(f"Stages completed in this call: {demonstration_stage_events}")
    print(
        f"Updated reference rows retained={len(demonstration_state.prepared.scans)}; "
        f"temporary update scans consumed={demonstration_state.manifest.update_scan_count}; "
        f"active APs={len(demonstration_state.prepared.bssid_vocabulary)}"
    )
    for record in demonstration_state.manifest.stage_records:
        print(f"  {record.stage}: {record.status}; metrics={dict(record.metrics)}")

# %%
# 8. Run/resume the sequential orchestrator and inspect reference rows
# lp_streaming resumes the disk-backed snapshot from cell 7 and processes the
# remaining ordered files. Full/stepwise modes use pred.run_sequential.

if RUN_MODE == "lp_streaming":
    from ae import load_ae_checkpoint
    from lp_streaming import load_streaming_lp_input
    from sage_ve import load_streaming_gnn_checkpoint
    from streaming_gnn import load_streaming_gnn_state
    from streaming_pipeline import run_streaming_sequential, select_update_paths
    from streaming_update import load_streaming_online_graph

    if streaming_bootstrap_gnn is None:
        streaming_bootstrap_gnn = load_streaming_gnn_checkpoint(
            GNN_CHECKPOINT_PATH, STREAMING_LP_INPUT_DIR, device=DEVICE
        )
    selected_streaming_update_paths = select_update_paths(
        update_paths=SELECTED_UPDATE_PATHS,
        update_directory=SELECTED_UPDATE_DIRECTORY,
        pattern=UPDATE_INPUT_PATTERN,
        bootstrap_paths=BOOTSTRAP_PATHS,
    )
    pipeline_result = run_streaming_sequential(
        PIPELINE_OUTPUT_ROOT,
        STREAMING_PIPELINE_CONFIG,
        bootstrap_prepared=prepared_dataset,
        bootstrap_ae_checkpoint_path=AE_CHECKPOINT_PATH,
        bootstrap_gnn_checkpoint=streaming_bootstrap_gnn,
        bootstrap_lp_input_directory=STREAMING_LP_INPUT_DIR,
        bootstrap_lp_result_directory=STREAMING_LP_RESULT_DIR,
        bootstrap_paths=BOOTSTRAP_PATHS,
        update_paths=selected_streaming_update_paths,
        source_root=INPUT_ROOT,
        progress_callback=_report_pipeline_progress,
    )
    latest = pipeline_result.latest_snapshot
    latest_directory = pipeline_result.latest_snapshot_directory
    if latest is None or latest_directory is None:
        print("No update batch was selected; no batch snapshot has been committed.")
    else:
        current_prepared = load_prepared_dataset(
            latest_directory / latest.current_state.reference_prepared,
            value_provenance_path=latest_directory
            / latest.current_state.reference_value_provenance,
        )
        current_graph = load_streaming_online_graph(
            latest_directory / latest.current_state.reference_graph
        )
        current_gnn = load_streaming_gnn_state(
            latest_directory / latest.current_state.reference_gnn,
            latest_directory / latest.current_state.reference_graph,
            device=DEVICE,
        )
        current_lp, _mapped = load_streaming_lp_input(
            latest_directory / latest.current_state.reference_lp_input
        )
        current_ae = load_ae_checkpoint(
            latest_directory / latest.current_state.ae_checkpoint,
            current_prepared,
            device=DEVICE,
        )
        current_scan_ids = tuple(scan.scan_id for scan in current_prepared.scans)
        assert current_graph.manifest.reference_sample_ids == current_scan_ids
        assert current_gnn.manifest.node_ids == current_graph.manifest.node_ids
        assert current_lp.sample_node_ids == current_scan_ids
        assert current_lp.graph_id == current_graph.manifest.update_graph_id
        assert (
            current_lp.gnn_checkpoint_sha256 == current_gnn.manifest.checkpoint_sha256
        )
        assert current_lp.bssid_vocabulary == current_prepared.bssid_vocabulary
        assert current_ae.manifest.vocabulary_sha256 == current_lp.vocabulary_sha256
        assert (
            current_ae.manifest.normalization
            == current_prepared.config.normalization_mode
        )
        assert len(current_scan_ids) == len(prepared_dataset.scans)
        print(f"Streaming snapshot: {latest_directory}")
        print(f"Completed in this call: {pipeline_result.completed_batch_ids}")
        print(f"Already complete: {pipeline_result.skipped_completed_batch_ids}")
        print(
            f"Reference fingerprints={len(current_scan_ids)}; "
            f"active APs={len(current_prepared.bssid_vocabulary)}; "
            f"graph nodes={len(current_graph.manifest.node_ids)}; "
            f"radio edges={current_graph.manifest.radio_edge_count}; "
            f"virtual edges={current_graph.manifest.virtual_edge_count}"
        )
        print("Stage status: " + ", ".join(item.name for item in latest.stage_records))
        for name, value in latest.metrics:
            print(f"  {name}={value:.6g}")
else:
    if RUN_MODE == "full":
        if not BOOTSTRAP_PATH.is_file():
            raise FileNotFoundError(
                f"bootstrap fingerprint file is missing: {BOOTSTRAP_PATH}"
            )
        pipeline_result = run_sequential(
            PIPELINE_OUTPUT_ROOT,
            ORCHESTRATION_CONFIG,
            bootstrap_prepared=prepared_dataset,
            bootstrap_paths=BOOTSTRAP_PATHS,
            bootstrap_source_paths=BOOTSTRAP_PATHS,
            update_paths=SELECTED_UPDATE_PATHS,
            update_directory=SELECTED_UPDATE_DIRECTORY,
            update_pattern=UPDATE_INPUT_PATTERN,
            source_root=INPUT_ROOT,
            ae_checkpoint_path=(
                EXTERNAL_AE_CHECKPOINT if AE_ACTION == "load" else None
            ),
            gnn_checkpoint_path=(
                EXTERNAL_GNN_CHECKPOINT if GNN_ACTION == "load" else None
            ),
            resume=PIPELINE_RESUME,
            log_every_epochs=LOG_EVERY_K_EPOCHS,
        )
    elif demonstration_result is not None:
        pipeline_result = demonstration_result
    else:
        raise RuntimeError("stepwise mode requires the completed cell 7 handoff")

    from ae import load_ae_checkpoint
    from sage_ve import load_gnn_checkpoint

    pipeline_state = pipeline_result.state
    pipeline_ae_checkpoint = load_ae_checkpoint(
        pipeline_state.ae_checkpoint_path, pipeline_state.prepared, device=DEVICE
    )
    pipeline_gnn_checkpoint = load_gnn_checkpoint(
        pipeline_state.gnn_checkpoint_path, pipeline_state.graph_state, device=DEVICE
    )
    selected_scan_ids = tuple(scan.scan_id for scan in pipeline_state.prepared.scans)
    assert len(set(selected_scan_ids)) == len(selected_scan_ids)
    assert pipeline_state.graph_state.sample_node_ids == selected_scan_ids
    assert pipeline_state.transition_state.reference_scan_ids == selected_scan_ids
    assert pipeline_state.encoder_features.scan_ids == selected_scan_ids
    assert pipeline_state.gnn_features.node_ids == pipeline_state.graph_state.node_ids
    assert (
        pipeline_state.prepared.bssid_vocabulary
        == pipeline_state.graph_state.bssid_vocabulary
    )
    assert (
        pipeline_state.transition_state.bssid_vocabulary
        == pipeline_state.prepared.bssid_vocabulary
    )
    assert pipeline_state.manifest.vocabulary_sha256 == canonical_json_digest(
        pipeline_state.prepared.bssid_vocabulary
    )
    assert (
        pipeline_state.encoder_features.normalization
        == pipeline_state.prepared.config.normalization_mode
    )
    assert (
        pipeline_state.graph_state.normalization
        == pipeline_state.prepared.config.normalization_mode
    )
    assert (
        pipeline_state.encoder_features.checkpoint_sha256
        == pipeline_state.graph_state.encoder_checkpoint_sha256
    )
    assert (
        pipeline_state.encoder_features.source_prepared_sha256
        == canonical_json_digest(pipeline_state.prepared)
    )
    assert pipeline_state.graph_state.source_prepared_sha256 == canonical_json_digest(
        pipeline_state.prepared
    )
    assert (
        pipeline_state.graph_state.encoder_feature_artifact_sha256
        == canonical_json_digest(pipeline_state.encoder_features)
    )
    assert pipeline_state.gnn_features.source_prepared_sha256 == canonical_json_digest(
        pipeline_state.prepared
    )
    assert (
        pipeline_state.gnn_features.encoder_feature_artifact_sha256
        == canonical_json_digest(pipeline_state.encoder_features)
    )
    assert (
        pipeline_state.gnn_features.checkpoint_sha256
        == pipeline_gnn_checkpoint.checkpoint_sha256
    )
    assert (
        pipeline_gnn_checkpoint.manifest.graph_id == pipeline_state.graph_state.graph_id
    )
    assert (
        pipeline_gnn_checkpoint.manifest.vocabulary_sha256
        == pipeline_state.manifest.vocabulary_sha256
    )
    assert (
        pipeline_ae_checkpoint.checkpoint_sha256
        == pipeline_state.graph_state.encoder_checkpoint_sha256
    )
    assert (
        pipeline_ae_checkpoint.manifest.normalization
        == pipeline_state.prepared.config.normalization_mode
    )
    assert pipeline_state.gnn_features.graph_id == pipeline_state.graph_state.graph_id
    assert pipeline_state.manifest.reference_scan_count == len(selected_scan_ids)
    print(f"Pipeline snapshot: {pipeline_state.snapshot_directory}")
    print(f"Completed now: {pipeline_result.completed_batch_ids}")
    print(f"Already complete: {pipeline_result.skipped_completed_batch_ids}")
    print(
        f"Updated reference fingerprints: {len(pipeline_state.prepared.scans)} original IDs; "
        f"AP vocabulary={len(pipeline_state.prepared.bssid_vocabulary)}; "
        f"GNN nodes={len(pipeline_state.gnn_features.node_ids)}"
    )
    print(
        "Stage status: "
        + ", ".join(
            f"{record.stage}={record.status}"
            for record in pipeline_state.manifest.stage_records
        )
    )

# %%
# 9. Optional evaluation, isolated from all fitting and update inputs
# Inputs: cell 8's completed snapshot, held-out coordinates in its original
# incoming-batch artifact, and predicted locations from the updater result.
# Output: a measured-reference RSSI diagnostic and a separate paired
# localization metric; neither feeds model state.

from fingerprint_update import load_updated_fingerprints
from gufu_data import load_prepared_dataset
from gufu_evaluation import (
    evaluate_measured_reference_rssi,
    evaluate_streamed_reference_rssi,
    evaluate_streamed_update_localization,
    evaluate_update_localization,
)

if RUN_MODE == "lp_streaming":
    if not RUN_EVALUATION:
        print("Evaluation disabled. Set RUN_EVALUATION=True to score the latest batch.")
    elif pipeline_result.latest_snapshot is None:
        print("No completed update snapshot is available to evaluate.")
    else:
        import json

        latest = pipeline_result.latest_snapshot
        snapshot_root = pipeline_result.latest_snapshot_directory
        evaluation_batch = load_prepared_dataset(
            snapshot_root / "input_prepared_batch.json"
        )
        updater_manifest = json.loads(
            (snapshot_root / "updaters" / "manifest.json").read_text(encoding="utf-8")
        )
        predicted_locations = np.load(
            snapshot_root / "updaters" / "predicted_update_locations.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        assert tuple(updater_manifest["update_ids"]) == tuple(
            scan.scan_id for scan in evaluation_batch.scans
        )
        localization_metric = evaluate_streamed_update_localization(
            tuple(updater_manifest["update_ids"]),
            predicted_locations,
            evaluation_batch.evaluation_positions,
        )
        print(
            "Held-out localization (estimator="
            f"{localization_metric.estimator}): n={localization_metric.paired_scans}, "
            f"2-D RMSE={localization_metric.rmse:.6g} "
            f"{localization_metric.coordinate_units}"
        )
        before_provenance_path = (
            snapshot_root / "reference_value_provenance_before.json"
        )
        source_reference = load_prepared_dataset(
            snapshot_root / "reference_prepared_before.json",
            value_provenance_path=(
                before_provenance_path if before_provenance_path.is_file() else None
            ),
        )
        final_reference_rssi = np.load(
            snapshot_root / "post_transition_ae" / "updated_rssi_dbm.npy",
            mmap_mode="r",
            allow_pickle=False,
        )
        try:
            rssi_metric = evaluate_streamed_reference_rssi(
                source_reference,
                final_reference_rssi,
                latest.vocabulary_after,
            )
        except ValueError as error:
            print(f"Measured-provenance RSSI diagnostic unavailable: {error}")
        else:
            print(
                "Reference RSSI reconstruction (measured-provenance cells; "
                f"n={rssi_metric.measured_cells}): RMSE={rssi_metric.rmse_dbm:.6g} dBm, "
                f"MAE={rssi_metric.mae_dbm:.6g} dBm"
            )
elif not RUN_EVALUATION:
    print(
        "Evaluation disabled. Set RUN_EVALUATION=True to score held-out update coordinates."
    )
else:
    artifact_paths = dict(pipeline_state.manifest.artifact_index)
    required = {"input_prepared_batch", "updated_fingerprints"}
    if not required.issubset(artifact_paths):
        print(
            "No completed update batch with evaluation artifacts is available to score."
        )
    else:
        snapshot_root = pipeline_state.snapshot_directory
        evaluation_batch = load_prepared_dataset(
            snapshot_root / artifact_paths["input_prepared_batch"]
        )
        updated_artifact = load_updated_fingerprints(
            snapshot_root / artifact_paths["updated_fingerprints"]
        )
        assert evaluation_batch.config.batch_role == "update_batch"
        assert updated_artifact.update_scan_ids == tuple(
            row.scan_id for row in updated_artifact.predicted_update_locations
        )
        assert updated_artifact.update_prepared_sha256 == canonical_json_digest(
            evaluation_batch
        )
        rssi_metric = evaluate_measured_reference_rssi(updated_artifact)
        print(
            "Measured-reference RSSI agreement (observed dBm cells; in-sample, "
            f"not held out): n={rssi_metric.measured_cells}, "
            f"RMSE={rssi_metric.rmse_dbm:.6g} dBm, "
            f"MAE={rssi_metric.mae_dbm:.6g} dBm"
        )
        if not evaluation_batch.evaluation_positions:
            print(
                "No held-out coordinate labels; localization metric was not computed."
            )
        else:
            location_metric = evaluate_update_localization(
                updated_artifact,
                evaluation_batch.evaluation_positions,
            )
            print(
                "Held-out localization (estimator="
                f"{location_metric.estimator}): n={location_metric.paired_scans}, "
                f"2-D RMSE={location_metric.rmse:.6g} "
                f"{location_metric.coordinate_units}"
            )
