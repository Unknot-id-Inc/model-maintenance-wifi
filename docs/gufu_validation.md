# GUFU pipeline validation and operation

This document records the reproducible commands, data contracts, compatibility rules, and validation limits for the connected pipeline. The implementation notebook remains [`gufu.py`](../gufu.py); stage logic lives in Python modules. The source and named implementation inferences for the paper equations remain in [`gufu_repair_plan.md`](gufu_repair_plan.md).

## Environment and setup

The validated environment is the existing `py314` micromamba environment; no packages were installed or upgraded:

| Package | Version |
|---|---:|
| Python | 3.14.6 |
| PyTorch | 2.12.0+cu130 |
| torch-geometric | 2.8.1 |
| torch-sparse | 0.6.18+pt212cu130 |
| NumPy | 2.5.3 |
| NetworkX | 3.6.1 |
| SciPy | 1.18.0 |
| Pydantic | 2.13.4 |
| pytest | 9.1.1 |

Use the explicit interpreter because the interactive shell's default Python is a separate 3.9 environment:

```bash
GUFU_PY=/home/oem/micromamba/envs/py314/bin/python
PYTHONDONTWRITEBYTECODE=1 "$GUFU_PY" -m pytest -q -p no:cacheprovider tests
git diff --check
```

No notebook editor package is installed in this environment. `gufu.py` is syntax checked and executed as a Python script; the tests parse every `# %%` cell and check the nine-cell order. Editor-specific interactive execution is not claimed.

## Stage inputs and outputs

All row and node joins use explicit stable IDs. Shapes below exclude JSON metadata and are always accompanied by the vocabulary or ID list that defines row/column order.

| Stage | Inputs | Output and key shape | Consumer |
|---|---|---|---|
| Prepare (`gufu_data`) | Raw DB/JSONL files and strict `PreprocessingConfig` | `PreparedDataset`, `rss_dbm [N,A]`, `observation_mask [N,A]`, full ordered BSSID/AP IDs, source row hashes, label roles, dropped-row report | AE, graph builder, batch transition |
| AE (`ae.py`) | Bootstrap `PreparedDataset`; selected model/training config | `gufu_ae_checkpoint_v1` with encoder+decoder and full training state; `gufu_encoder_features_v1`, `feature_matrix [N,32]` keyed by `scan_ids` | Graph builder, updater |
| Graph (`graph_build.py`) | Prepared values and matching final AE artifact | `GraphState`, ordinary node matrix `[V,34]`, row map and typed physical/virtual edges | GNN, link prediction |
| GNN (`sage_ve.py`) | `GraphState` and explicit GNN configuration | `gufu_gnn_checkpoint_v5`; `gufu_gnn_features_v1`, nodes `[V,34]`, physical edge vectors `[E_radio,34]`, all keyed by graph IDs | Link prediction, batch updater |
| Link modification (`lp.py`) | Matching graph and learned GNN features | `gufu_link_prediction_v1`: goodness/fairness rows `[V,34]`, candidate scores and explicit added/removed/changed/AP deltas; optional `gufu_lp_graph_snapshot_v1` | Fingerprint updater and transition |
| Fingerprint update (`fingerprint_update.py`) | Prior reference state, full incoming scans, compatible AE/GNN artifacts and optional LP snapshot | `gufu_fingerprint_update_checkpoint_v4`; `gufu_updated_fingerprints_v1`, one row per original reference scan plus separate predicted update positions | AP/vocabulary transition |
| AP transition (`batch_transition.py`) | Prior immutable state, current incoming batch, updater outputs and edge changes | Versioned transition state, RSS/mask/provenance remapped by BSSID, migrated AE/GNN lineage, transition results | Next batch |
| Orchestration (`pred.py`) | Bootstrap and explicitly ordered/naturally discovered update files, complete configuration | Immutable checksum manifest and completion marker per bootstrap/batch snapshot | Fresh-process resume; notebook and CLI |

### Checkpoint compatibility and resume boundary

- AE checkpoint loading validates the architecture, exact ordered BSSID vocabulary and digest, normalization, dataset identity, input width, and model configuration. `resume` restores optimizer/progress/random state; `weights_only` is initialization and makes no resume claim.
- Current AE checkpoints use architecture `gufu_ae_leaky_relu_kaiming_uniform_v2` and record either element-mean MSE or minibatch Frobenius training. Existing ReLU architecture checkpoints are rejected because their forward computation differs. Section 4.1 prints the Frobenius objective; MSE is the selectable local alternative. The trained encoder is a downstream graph input, so its learned feature geometry is part of checkpoint lineage.
- GNN checkpoint loading validates graph ID, ordered node IDs/types and map, radio/virtual edge topology, vocabulary, normalization, source AE checkpoint, and model configuration. Independent graph runs are distinct. The per-batch GNN is initialized from the prior selected trained weights with a fresh optimizer; the final state is re-encoded in inference mode.
- Fingerprint updater checkpoints bind the reference/update IDs, graph, edge lists, vocabulary, normalization, AE/GNN checkpoints, update configuration, optimizer states, and random state. Across completed batches, the two updater MLPs carry weights forward while they train on the new batch with new optimizers. Mid-batch training resume is not supported.
- AP vocabulary migration joins every RSS column, mask, graph row, decoder row/bias, and cache by full canonical BSSID or stable node ID. Encoder/GNN features are regenerated after a transition. APs removed by forgetting remain in the AP registry as tombstones and can be restored if they return. A batch missing an AP measurement does not by itself tombstone that AP.
- Pipeline resume is supported between completed snapshots. The loader validates the completion marker, manifest/configuration hash, every artifact hash, parent snapshot/state IDs, and model/data lineage before it exposes state. An incomplete staging directory is ignored. A conflicting completed output root is rejected; use a new run root for an independent run.

## Notebook and CLI use

`gufu.py` has nine `# %%` cells:

1. Imports, explicit paths/configuration, seeds, device, run mode, output location.
2. Prepare/load bootstrap data and inspect labels, masks, vocabulary, and dropped rows.
3. Train/load AE; inspect the compatible checkpoint, all encoder rows, and reconstruction diagnostic.
4. Build/load graph; inspect ID mapping, AP initialization, and physical/virtual counts.
5. Train/load GNN; inspect learned node/edge feature provenance.
6. Inspect the streamed link delta; it performs no destructive write.
7. In `lp_streaming`, commit the first incoming batch using cell 6's exact streamed LP result; in `stepwise`, run one isolated in-memory demonstration.
8. Resume the same selected driver; streaming mode starts from the completed cell 7 snapshot and processes the rest of the ordered batches.
9. Optionally report measured-reference RSSI agreement and held-out localization after fitting.

In the default `RUN_MODE="lp_streaming"`, cells 7–8 consume the exact streamed LP input/result from cell 6 and do not call the in-memory `pred.run_sequential` path. Run cells **1–8** in order. Cell 7 commits the first selected update; cell 8 resumes the same output root, skips that completed input, and processes any remaining batches from its snapshot. Cell 1 saves the matching configuration to `RUN_DIR/streaming_pipeline_config.json`. `DATA_PROFILE="cpu_smoke"` uses one epoch and one selected update from the configured real text files; it changes budgets and virtual-edge threshold, not the data source. `DATA_PROFILE="real_dataset"` uses the configured larger budgets and natural directory selection. Cells 1–5 still build/load ordinary prepared, graph, and GNN artifacts in memory, so the streaming mode does not guarantee arbitrarily large bootstrap graphs will fit.

`RUN_MODE="full"` delegates update work to the in-memory `pred.run_sequential` path in cell 8. `RUN_MODE="stepwise"` exposes cells 3–6 and uses an isolated in-memory demonstration in cell 7. In cell 1, set `AE_LOSS="mean_squared_error"` or `AE_LOSS="batch_frobenius_norm"`; the derived `RUN_ID` and artifact paths differ so their weights/features are not mixed. Changing that selector while resuming an AE checkpoint is rejected: start the other objective's separate run from fresh initialization, or explicitly choose a compatible weights-only initialization in a direct API call. A stable `RUN_ID` reuses validated artifacts; change it to choose a new isolated run. `AE_LOSS` selects training only; if `AE_ACTION="load"`, the loaded checkpoint's recorded objective remains authoritative. Both `AE_ACTION` and `GNN_ACTION` default to `"train"`, so a clean run does not assume checkpoints exist. Training budgets are visible in cell 1.

The supported script entry point accepts the same serialized `OrchestrationConfig` used by the notebook:

```bash
/home/oem/micromamba/envs/py314/bin/python pred.py --help
/home/oem/micromamba/envs/py314/bin/python pred.py \
  --config-json /path/to/orchestration-config.json \
  --output-root /path/to/new-or-resumable-run \
  --bootstrap input/GF/fingerprint0.txt \
  --bootstrap-batch-id fingerprint0 \
  --updates input/GF/fingerprint1.txt input/GF/fingerprint2.txt \
  --source-root input
```

Explicit `--updates` order is preserved. Directory discovery uses numeric-aware filename ordering and excludes the bootstrap path. Importing `pred` has no training or output-directory side effects. Snapshots are additive and retain prior completed batches.

The disk-backed continuation uses `streaming_pipeline.py`, the same API called by notebook cells 7–8:

```bash
PY=/home/oem/micromamba/envs/py314/bin/python
RUN_ID=real-GF-fingerprint0-ae-v2-mse-d256-z32-drop0p1
BASE="output/notebook_runs/GF/$RUN_ID"
"$PY" -m streaming_pipeline --help
"$PY" -m streaming_pipeline --config "$BASE/streaming_pipeline_config.json" --output "$BASE/streaming_pipeline" --prepared "$BASE/prepared_bootstrap.json" --ae-checkpoint "$BASE/checkpoints/ae.pt" --gnn-checkpoint "$BASE/checkpoints/gnn.pt" --lp-input "$BASE/link_prediction_input" --lp-result "$BASE/link_prediction_result" --bootstrap-path input/GF/fingerprint0.txt --update-directory input/GF --update-pattern 'fingerprint[1-9]*.txt' --source-root input
```

For fresh-process resume, retain `--config`, `--output`, the same ordered update inputs, and source root. With an existing `LATEST.json`, the CLI validates and loads the latest completed snapshot without needing bootstrap checkpoint arguments again.

## Paper-to-code equations and named inferences

The detailed source audit and dimensions are in the plan. This condensed table distinguishes paper requirements from implementation choices. Inferred rows are tested locally, but are not author-verified reproductions.

| Source | Paper operation | Implemented mapping | Status / unresolved point |
|---|---|---|---|
| Section 4.2, Algorithm 1 | Initialize sample node `z_s` from encoder code plus location; initialize AP nodes by weighted neighboring-sample features; update edge vectors from endpoints. | Sample graph rows are `[z_encoder (32), location_input (2)]`; AP rows are `RSS+120` weighted means; graph node and edge vectors are 34-D; scalar weights stay separate. | AP averaging and edge feature flow have unit hand calculations. The source's dimensions and some notation do not all align; 34-D is the implementation's resolved shape. Unlabeled locations use a row/path/seed-hashed point inside bootstrap-derived or explicit trusted bounds. |
| Equation (4) | Sum edge endpoint dot products through `-log(ReLU(dot))`. | Sum once per undirected typed edge, clamp the nonnegative dot by explicit epsilon before `log`; gradients pass through learned node outputs and layer parameters used by those outputs. | The exact reduction/clamp at zero is an explicit finite-gradient inference; local reference calculation tested. The final edge-vector layer has no gradient from node-only Eq. (4) in a one-layer run, as the printed objective implies. |
| Section 5.2, Equations (5)–(9) | Train location→position and position→feature mappings with supervised and adjacency consistency terms. | Inputs: GNN `[N,34]` and `[K,34]`; bootstrap scaled coordinates `[N,2]`; fixed new encoder targets `[K,32]`; row-normalized cross-virtual adjacency `[K,N]`. Losses use Frobenius/vector norms, fixed targets are detached, and predictions remain differentiable. | Input/output orientation and `A` normalization are inferred and recorded in the plan. Update/evaluation truth is not supplied to training. The source has notation/orientation ambiguity; the tested variant is not called author-verified. |
| Equation (10) | Update AE using RSS reconstruction and encoder/decoder consistency with updated latent targets. | RSS/AE tensors `[N,1,A]`; fixed updated latent target `[N,32]`; decoder input is the learned `z_update_mlp` output in the explicitly named 32-D decoder space. Inverse normalization produces dBm output. | Direct compatible 32-D latent alignment is tested. Weighting and target schedule ambiguity is documented; training traces show intended encoder/decoder gradients. |
| Sections 5–6 | Shared-AP inference precedes new-node/AP initialization, edge change, vocabulary/model update, fingerprint reconstruction. | Batch order is: project to old vocabulary and encode; transient graph/updaters; infer AP edges; migrate vocabulary/model state by identity; reconstruct; regenerate final AE/GNN artifacts; remove transient nodes at snapshot boundary. | Paper does not specify a unique weight-copy policy or persisted lineage. Parameter migration with archived returning AP slices and transactional state is an explicit implementation choice. |
| Section 6, Equations (11)–(13), Algorithm 2 | Initialize goodness/fairness from learned nodes, update through weighted neighborhoods, score AP/sample links, forget weak edges/orphan APs. | Synchronous Eq. (11)/(12), independent cloned buffers, node-specific neighborhoods, normalized radio weights, Eq. (13) denormalized to `RSS+120`, bounded convergence, supported missing-edge candidates plus separately scored existing edges, explicit delta. | Algorithm 2 filters existing links before its removal branch, so that branch is unreachable. Existing edges are scored separately to make the Section 6.2 removal prose reachable. Virtual scalar weights are unspecified; stored cosine is the named normalized LP weight. Threshold unit reconciliation is also inferred. |

For bootstrap AE training, the source objective is Frobenius norm. MSE averages squared error over batch and feature dimensions, giving a batch-size-stable scale for similarly distributed errors; Frobenius mode uses the raw norm of each minibatch residual and thus scales with minibatch size. Either objective can yield a good reconstruction, but changing it changes optimization and does not make latent representations interchangeable: the learned 32-D features are consumed by graph construction and link prediction, so model architecture, checkpoint lineage, and feature geometry matter too.

### Historical GNN loss inspection

The previously inspected notebook run `output/notebook_runs/GF/real-GF-fingerprint0-ae-v2-mse` provides evidence for interpreting its apparently stalled GNN curve. At that inspection, its graph contained 1,065 nodes, 20,474 physical RSSI edges, and 76,557 virtual sample edges (76,636 possible sample pairs), or 97,031 terms in the configured Equation (4) sum. The saved 500-epoch curve fell from about 126,851 on epoch 1 to roughly 76,000–82,000 by later epochs, then fluctuates rather than decreasing smoothly. This is an edge-count sum, not a mean, so raw loss values also change with graph size/density.

The trainer reports the objective while `model.train()` enables dropout at probability 0.5 after message/node ReLUs and before L2 normalization. Repeated train-mode evaluations of the saved weights ranged about 74,381–85,597; eval-mode output from the same checkpoint gave Equation (4) loss about 2.51. Eval embeddings had nonzero pairwise edge dots clustered very close to 1 (mean about 0.999974), indicating that the positive-edge objective has driven connected nodes toward near-aligned vectors. This supports two separate explanations: dropout makes the reported per-epoch objective noisy, and the objective itself rewards collapse because it has only attraction terms. The nearly complete virtual sample graph intensifies that pressure and can erase useful node distinctions.

This is not a proof that every run will behave identically. It is an inspection of a saved real-data checkpoint from before the objective update documented here. At final verification the graph and GNN checkpoint files under that run path were absent, so these artifact measurements could not be independently reloaded again. At the time, the objective was `-sum(log(clamp(ReLU(z_u · z_v))))` over positive graph edges, with no negative-pair term and a raw sum that scaled with edge count. The last `W2` edge-vector update in a two-layer model also received no direct gradient because that objective reads node embeddings only. No GNN objective, dropout, or graph change was made during that earlier diagnostic task. The active implementation now uses mean softplus losses on normalized cosine similarities for positive edges and RSSI-mined negative pairs, as described in the README. The historical measurements below should not be read as measurements of that newer objective.

Evaluation keeps unlike outcomes separate. `evaluate_measured_reference_rssi` reports dBm RMSE/MAE on observed reference cells only; this is an in-sample consistency diagnostic, not held-out RSSI accuracy. `evaluate_update_localization` reports paired 2-D RMSE in source coordinate units for `location_mlp_from_final_z_gnn`, joining only a separate evaluation-only label table by scan ID. Neither metric feeds a fitting or transition API. No published localization accuracy is claimed from these plumbing checks.

## Validation record (milestone 10)

The deterministic orchestration fixture uses a labeled bootstrap with repeated coordinates and missing RSSI cells; three update batches are JSONL with no coordinates at all. It adds AP C, varies input BSSID order, carries the updated vocabulary/model into later batches, and checks reference identity, temporary-node cleanup, graph/AP feature aggregation, and edge prediction against the trained checkpoint's GNN artifact. Separate tests perform identity-based vocabulary reorder, AP forgetting/return, and transition rollback. The update coordinates used by localization tests are built separately as `EvaluationPosition` values and are never passed to fitting.

The suite checks equations against tiny hand calculations for AP initialization, the GNN layer and Equation (4), link goodness/fairness and weight units, all updater losses/gradient paths, and RSSI/localization metric estimators. It rejects missing/corrupt snapshots, wrong graph/node/vocabulary/checkpoint lineage, and failed migration. A changed learned GNN row changes the link result; wrong IDs and vocabulary are rejected. The orchestration test resumes batches two and three in a new `pred.py` process, compares the final encoder/GNN matrices with uninterrupted CPU execution at `rtol=atol=1e-6`, and verifies an unrelated file survives.

Exact checks run after milestone 10 changes:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/oem/micromamba/envs/py314/bin/python \
  -m pytest -q -p no:cacheprovider tests
git diff --check
PYTHONPYCACHEPREFIX=/tmp/gufu_m10_pycache PYTHONDONTWRITEBYTECODE=1 \
  /home/oem/micromamba/envs/py314/bin/python \
  -m py_compile gufu.py pred.py gufu_evaluation.py
```

The full suite result was **78 passed**, with 19 PyTorch 2.12 `torch.jit.script` deprecation warnings. The clean-root `gufu.py` script smoke used copies of `input/GF/fingerprint0.txt` and `fingerprint1.txt`, CPU, one epoch per model/updater stage, and `virtual_edge_sigma=1.0` (strict threshold, therefore no virtual edges). It retained 392 reference scans, parsed 673 APs and 20,474 measured cells, passed 2,078 update scans through the transient graph, and exported 1,065 final graph nodes. Edge prediction returned `no_change`; updater training, AP transition, AE realignment, feature refresh, temporary-node cleanup, and snapshot commit still completed. A second fresh script process validated the snapshot and skipped the already completed batch. This is a bounded real-file integration smoke, not a full-paper configuration or accuracy run.

The script smoke ran in a clean temporary root from an isolated copy of `gufu.py`; only `DATA_PROFILE` was changed from `real_dataset` to its supported `cpu_smoke` option. The repository's `output/` tree and checked-in input files were not used as destinations or modified. The exact run root was `/tmp/gufu_m10_clean_20261012`.

### Coverage and unresolved scope by milestone

| Milestone | Implementation | Tests actually run | Notebook cell(s) | Artifacts | Remaining scientific or verification ambiguity |
|---:|---|---|---|---|---|
| 1. Shared preprocessing | `gufu_data.py`; prep/AE/pred/notebook use one prepared representation | `tests/test_gufu_data.py`, full suite | 1–2 | Prepared JSON, provenance and value-origin sidecar | RSS threshold/pruning defaults are explicit; source dataset semantics beyond supported formats remain data-specific. |
| 2. AE persistence | `ae.py` training/load/encoding/checkpoint APIs; current architecture v2 uses LeakyReLU/Kaiming and selectable MSE/Frobenius objectives | `tests/test_ae.py` (13 tests), updated full suite | 3 | AE `.pt`, encoder feature JSON | Padding/dropout placement remain named architecture inferences; latent width stays 32 for downstream compatibility. |
| 3. Graph construction | `graph_build.py` ID-joined deterministic AP/sample graph | `tests/test_graph_build.py`, full suite | 4 | Graph state JSON | Unlabeled pseudo-location policy is seeded inference; paper does not define it. |
| 4. GNN | `sage_ve.py` edge-aware Algorithm 1 path and exported learned state | `tests/test_sage_ve.py`, full suite | 5 | GNN `.pt`, ID-aligned feature JSON | Equation (4) zero clamp/reduction; one-layer final edge-vector parameters are not trained by the node-only loss. |
| 5. Link prediction | `lp.py` typed edge delta and pure apply function | `tests/test_lp.py`, full suite | 6 | Edge delta JSON, optional learned graph snapshot | Virtual-edge weight, threshold units, and removal candidate reconciliation are explicit inferences, not author verified. |
| 6. Fingerprint updater | `fingerprint_update.py` GNN/MLP/AE update and reference-only RSS output | `tests/test_fingerprint_update.py`, full suite | 7 | Compound updater checkpoint and updated fingerprint JSON | Adjacency orientation/normalization, loss weighting, and target schedule are tested inferences. |
| 7. AP/batch transitions | `batch_transition.py` versioned BSSID remap, model migration, tombstones/return | `tests/test_batch_transition.py`, full suite | 7 | Transition state/result, migrated AE, refreshed caches | Paper says retrain at AP changes but does not specify a unique parameter migration policy. |
| 8. Orchestration | `pred.py` bootstrap, update, sequential CLI/API and transactional snapshots | `tests/test_orchestrator.py`, including fresh-process resume | 8 | Checksummed snapshots and completion markers | Resume boundary is between completed batches; no mid-batch optimizer resume. |
| 9. Notebook integration | Existing `gufu.py`, nine cells calling shared APIs | `tests/test_gufu_notebook.py`, syntax check, clean-root script smoke | 1–9 | Notebook run directories and pipeline snapshots | Interactive editor execution is not tested because notebook/JupyterLab are absent; script execution covers the cell file top to bottom. |
| 10. End-to-end validation/docs | Deterministic unlabeled multi-batch fixture, provenance assertions, separate metrics and this guide | Original full suite **78 passed**; AE follow-up full suite **81 passed**; `git diff --check`; `py_compile`; earlier clean-root CPU smoke and rerun | 1–9 | Test snapshots under pytest temp paths; earlier smoke snapshots under `/tmp/gufu_m10_clean_20261012` | The real-file smoke predates AE v2. Full 392+2,078+ real input was run only at one-epoch CPU smoke settings with no virtual edges. Full-data `sigma=0.95` resource use, real-data convergence, interactive notebook execution, and held-out localization accuracy remain unverified. |

No environment upgrade, branch change, commit, push, recursive deletion, or write to the repository's existing output data was performed.

## AE training update (2026-10-13)

The shared bootstrap AE uses `gufu_ae_leaky_relu_kaiming_uniform_v2`: LeakyReLU slope `0.1`, Kaiming-uniform Conv1d/ConvTranspose1d weights (`fan_in`, `a=0.1`), and zero convolution biases. Cell 1 exposes dropout probability (currently `0.1`; the direct training API defaults to `0.5`) and dense width (`256`); both are recorded in the checkpoint and affect the actual model layers. The connected graph/update APIs currently require a `32`-D latent, so `AE_LATENT_DIM` is exposed as a documented fixed value and rejects other sizes. Bootstrap training uses every prepared scan according to the configured shuffle setting. Cell 1 selects MSE or minibatch Frobenius loss. MSE records element-weighted epoch MSE; Frobenius records the unweighted mean of raw minibatch norms, which varies with minibatch size. The separate Equation (10) updater loss remains Frobenius-based.

The architecture ID prevents loading old ReLU checkpoints as if they were equivalent. `gufu.py` now uses a fresh default run ID ending in `ae-v2-mse`, preserving earlier output roots. A regression test confirms nonzero gradients reach the encoder and first decoder convolution, checks convolution biases start at zero, and verifies that all prepared scans enter training. Existing vocabulary-transition tests found and drove a correction to a stale decoder-output state-dict index; they now pass with architecture v2.

Cell 1 exposes dropout probability and dense width, and both values configure actual AE layers and are included in checkpoint/run identity. It also shows `AE_LATENT_DIM=32`; another width is rejected because graph and updater modules currently require a 32-D encoder vector.

Commands run after this follow-up:

```bash
PYTHONDONTWRITEBYTECODE=1 /home/oem/micromamba/envs/py314/bin/python \
  -m pytest -q -p no:cacheprovider tests
PYTHONPYCACHEPREFIX=/tmp/gufu_ae_update_pycache PYTHONDONTWRITEBYTECODE=1 \
  /home/oem/micromamba/envs/py314/bin/python \
  -m py_compile ae.py gufu.py batch_transition.py fingerprint_update.py pred.py
git diff --check
```

Observed: **81 passed**, with 19 PyTorch 2.12 `torch.jit.script` deprecation warnings; `py_compile` and `git diff --check` passed. The synthetic integration suite exercised model training, downstream graph/GNN/link/update APIs, vocabulary migration, and orchestration. The interactive `# %%` cells and bounded real-data smoke were not rerun with the new AE architecture. The prepared matrix loss includes the missing-reading sentinel at currently missing AP cells; it is not mask-weighted.

## Disk-backed continuation after cell 6 (2026-10-13)

The earlier LP-only behavior stopped after cell 6. That did protect the link-prediction calculation from candidate-sized Python allocations, but left a user who wanted to update fingerprints needing the old expanded in-memory path. The current `lp_streaming` mode now carries cell 6's exact LP bundle/result and selected GNN checkpoint through the streamed updater, AP/vocabulary transition, AE/GNN refresh, and per-batch snapshot. Cells 7 and 8 both call `streaming_pipeline.run_streaming_sequential`; cell 7 commits the first selected batch, and cell 8 resumes that same root and continues. A no-edge-change batch does not terminate the outer update sequence. Cell 9 remains optional and reads evaluation coordinates only after fitting.

Cell 1 writes the immutable `StreamingPipelineConfig` to `RUN_DIR/streaming_pipeline_config.json`; editing configuration under the same run ID is rejected. For a clean small run, set `DATA_PROFILE="cpu_smoke"`, leave `RUN_MODE="lp_streaming"`, and run cells **1–8**. This profile still reads the configured real input text files, uses one selected batch, and uses one-epoch budgets. For the larger real-data budgets, retain `DATA_PROFILE="real_dataset"`; update discovery is natural filename order and excludes the bootstrap file. Cell 9 can be run with `RUN_EVALUATION=True` after cell 8 when evaluation-only positions exist. The CLI equivalent and argument list are in [streaming_link_prediction.md](streaming_link_prediction.md).

Snapshots contain: the before/after reference prepared artifacts and provenance; incoming prepared batch; cell-6 source LP artifacts on the first batch; online graph/GNN and updater states; Eq. (10) result; AP transition; migrated/post-transition AE; refreshed reference graph/GNN and LP input; source-file hashes; stage metrics and config; and manifest/completion markers. `LATEST.json` is advanced only after the complete snapshot is atomically published. Restart validates the full artifact hashes, model/vocabulary/node identities and parent chain, then processes only inputs not already committed. Updater and streaming GNN warm starts carry weights across batches with a fresh Adam optimizer at each batch boundary; mid-batch training resume is not supported.

The continuation keeps large numeric graph, activation, adjacency, RSSI, and edge tables in memory-mapped files and uses SQLite for candidate/reference edge indexing. It avoids constructing the expanded in-memory LP delta, updated graph snapshot, and online edge dictionaries. This is not a full guarantee that graphs larger than RAM can pass through all of GUFU: cells 1–5 still materialize the bootstrap `PreparedDataset`, `GraphState`, and GNN artifacts, and the downstream code retains PreparedDataset scan/RSSI metadata (including an incoming batch) in Python. Virtual-edge enumeration remains quadratic. No production-scale peak-RSS test has been run. Thus `lp_streaming` now goes beyond cell 6 without reverting to the same full graph materialization that motivated LP streaming, while full out-of-core preprocessing/bootstrap remains future work.

Validation for this continuation:

- `tests/test_streaming_pipeline.py` performs actual AE/GNN/updater training, streamed LP, new-AP admission, vocabulary migration, reference graph/GNN refresh, snapshot commit, multiple sequential updates, resume/uninterrupted comparison, interrupted-stage recovery, unrelated-file preservation, hidden-coordinate invariance, and fresh-process CLI resume on tiny fixtures.
- `tests/test_gufu_notebook.py` parses all nine cells and checks the streamed notebook handoff uses the shared sequential API and saves configuration.
- The notebook/script file itself was not executed in this continuation. The prior milestone's real-data `cpu_smoke` predates these streaming cells; no large-data OOM or peak-memory claim is made here.

The focused validation command `PYTHONDONTWRITEBYTECODE=1 /home/oem/micromamba/envs/py314/bin/python -m pytest -q -p no:cacheprovider tests/test_streaming_pipeline.py tests/test_gufu_evaluation.py tests/test_gufu_notebook.py tests/test_lp_streaming.py` passed **16 tests**. Full-suite, compile, and diff checks for this follow-up are recorded in the final run report below.

Follow-up verification actually completed:

- Full suite: `PYTHONDONTWRITEBYTECODE=1 /home/oem/micromamba/envs/py314/bin/python -m pytest -q -p no:cacheprovider tests` — **101 passed**, with 19 PyTorch 2.12 / Python 3.14 `torch.jit.script` deprecation warnings.
- Static: `py_compile` passed for `streaming_pipeline.py`, `streaming_update.py`, `streaming_gnn.py`, `streaming_transition.py`, `streaming_fingerprint_update.py`, `sage_ve.py`, `gufu_evaluation.py`, `gufu.py`, and the streaming/notebook tests. `git diff --check` passed.
- The supported `gufu.py` script workflow was run from a temporary working copy at `/tmp/gufu-lpstream-followup-ZJCpwt`, with only `DATA_PROFILE` changed to `cpu_smoke`. It used copies of the real GF bootstrap and first update files; no source input or repository output path was written. Cells 1–8 completed. Bootstrap prepared 392 scans / 673 APs / 20,474 observed readings; the temporary batch contained 2,078 scans; final reference state retained 392 scans and exported 1,065 nodes. It used one epoch and strict `virtual_edge_sigma=1.0`, so no virtual edges or AP vocabulary/edge changes occurred in this smoke. The snapshot completed and cell 8 skipped the already committed cell-7 batch. No localization evaluation or full-budget training was run.
- The actual AP addition, vocabulary migration, and later-batch continuation were exercised in `tests/test_streaming_pipeline.py` on tiny fixtures; the real-data smoke does not establish accuracy or production-scale memory behavior.

The notebook editor's interactive execution was not tested. The `# %%` source parsed and was executed as a Python script. Peak RSS on a graph larger than available memory remains unverified; bootstrap graph preparation is still in memory and the streamed online graph builder's exact virtual-pair scan remains quadratic.
