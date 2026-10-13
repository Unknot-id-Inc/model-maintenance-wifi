# Disk-backed link prediction

`lp_streaming.py` runs the same Section 6 implementation policy as
`lp.predict_edges`, using read-only NumPy memory maps for node vectors, CSR
adjacency, and physical-edge vectors. It writes candidate scores and edge
deltas as JSON Lines and the updated radio-edge table as NumPy arrays. It does
not construct the legacy candidate-sized Python tuples/lists, `EdgeDelta`, or
`UpdatedGraphSnapshot` during prediction.

This preserves the current code's behavior: the radio/virtual neighbor weights,
stable lexical neighbor order, G/F initialization and synchronous updates,
residual stopping rule, candidate-edge support rule, threshold units, edge
addition/removal decisions, edge-vector initialization, AP forgetting, and
updated graph digest all match `lp.predict_edges`. This is equivalence to the
local implementation policy, including its documented inferred choices; it is
not a new validation of author intent.

## Use

The direct API takes one existing `GraphState` and matching `GNNFeatureArtifact`
to write a compact input bundle once. Later invocations can run from the bundle
alone:

```python
from lp import LinkPredictionConfig
from lp_streaming import (
    predict_edges_streaming,
    save_streaming_lp_input,
)

save_streaming_lp_input(graph_state, gnn_features, "run/lp_input")
result = predict_edges_streaming(
    "run/lp_input",
    "run/lp_result",
    LinkPredictionConfig(max_iterations=2000, link_threshold=0.1),
)
print(result.manifest.outcome, result.manifest.candidate_count)
for edge in result.iter_added_edges():
    consume(edge)
```

Transition callers can optionally pass an ordered iterator of stable ID pairs
through `candidate_pairs=`. The iterator must be lexicographically sorted by
`(sample_id, ap_id)` and unique. This lets a later AP-transition stage keep its
paper-specified candidate rule in a disk-backed index and score only those
pairs, without building a candidate-sized Python set. When omitted, the
ordinary Section 6 support rule is used.

For one batch transition, `streaming_update.save_streaming_online_lp_input_with_new_aps`
extends the streamed old-vocabulary online graph with raw measurements for
new or returning APs. The AP vector is the RSS+120 weighted mean of the final
online GNN sample vectors; temporary edge vectors use endpoint means. The
paired `iter_streaming_ap_transition_candidates` stores existing reference
edges and update-to-reference virtual-supported new-AP pairs in a SQLite
index, and `infer_streaming_ap_graph_changes` sends that sorted disk cursor to
the streamed scorer. Its result records the explicit candidate policy and a
candidate-set digest. `streaming_transition.apply_streaming_ap_delta_to_reference`
consumes that JSONL delta without constructing `APGraphModification`,
updates a SQLite reference-edge table, determines forgetting/admission by AP
identity, and writes mapped RSS, measured-mask, and provenance arrays aligned
to the original reference IDs/coordinates. New-edge RSS values remain marked
as edge predictions with false measured masks. `streaming_pipeline.py` carries
this state through AE vocabulary migration and fine-tuning, reference graph/GNN
refresh, new reference LP input creation, and an immutable completed snapshot.
The next batch reads that snapshot's AE, graph, GNN, vocabulary, and LP input.

Both directories are immutable and refuse overwrite. Loaders verify completion
markers, artifact hashes, shapes/dtypes, dataset/graph/checkpoint/vocabulary
lineage, and input/output identity. Candidate and change records are iterators;
they must not be converted to lists on large graphs. Arrays such as
`goodness_features.npy`, `fairness_features.npy`, and
`updated_edge_endpoints.npy` can be opened with `numpy.load(..., mmap_mode="r")`.
The updated-edge endpoint array contains input node-row indices; consult the
manifest's stable `node_ids`. Added physical-edge vectors use the same
endpoint-mean initialization as the in-memory path. Existing vectors and
weights are retained. APs with zero resulting radio degree are listed in
`ap_removals` and excluded from `active_node_rows.npy`.

In `gufu.py`, `RUN_MODE = "lp_streaming"` is the default. Cells 1–5 prepare the
bootstrap dataset and train/load the bootstrap AE/GNN. Cell 6 writes or
validates `RUN_DIR/link_prediction_input`, then streams the proposal to
`RUN_DIR/link_prediction_result`. Cell 7 sends the exact cell-6 input/result
and trained GNN checkpoint into `streaming_pipeline.run_streaming_sequential`
for the first selected batch. Cell 8 calls the same API against the same output
root; it verifies and skips the batch committed in cell 7, then processes the
remaining ordered inputs from the last completed reference state. Cell 9 can
evaluate the latest update against a separate evaluation-only coordinate table
and report measured-reference RSSI agreement as a distinct dBm metric. This
mode does not call the legacy in-memory LP or `pred.run_sequential`.

Keep `RUN_MODE="lp_streaming"` and run cells 1–8 in order. Set
`RUN_EVALUATION=True` only before cell 9 if update records have evaluation-only
coordinates. `DATA_PROFILE="cpu_smoke"` selects one update and one-epoch
budgets using the configured real text inputs; it changes budgets and virtual
edge threshold, not the data source. `real_dataset` uses the configured larger
budgets and naturally selects `fingerprint[1-9]*.txt`. Change `RUN_ID` for an
independent artifact root. Re-running cells 7–8 with the same ID, config, and
inputs resumes/skips completed immutable snapshots; incompatible artifacts
fail lineage validation. `stepwise` and `full` are separate in-memory paths,
not continuation modes for a streamed result.

Fresh-process execution uses `python -m streaming_pipeline` with the same
serialized `StreamingPipelineConfig`. A new run consumes prepared bootstrap,
AE/GNN checkpoints, cell-6 LP input/result, and update paths. A resumed run
needs the existing pipeline output root, matching config, and ordered input
paths; it loads current model and graph state from the last completed snapshot.
The CLI's `--help` lists its arguments.

## Memory and scale limits

During the prediction pass, feature and adjacency values are disk-backed.
Resident numeric work is bounded by the G/F ping-pong row matrices' mapped
pages, one node's neighborhood and feature row, and one candidate record at a
time; ID and provenance metadata still scales with node count. Candidate scores
are preserved on disk, so disk use scales with the number of scored pairs.
Runtime still scales with the number of G/F iterations and traversed edges,
and the number of candidates can be large (especially with dense virtual
connectivity).

The streaming continuation stores the online graph, GNN activations/features,
edge deltas, sparse update/reference adjacency, fingerprint arrays, and
post-transition reference graph in mapped numeric arrays; candidate and
reference-edge indexes use SQLite. It avoids materializing the candidate
`EdgeDelta`, expanded `UpdatedGraphSnapshot`, or expanded online graph in
cells 7–8. GNN/updater/AE numeric training passes process configured chunks.

This is not yet an end-to-end graph-larger-than-RAM guarantee. Cells 1–5 still
construct/load ordinary `PreparedDataset`, `GraphState`, and GNN artifacts in
memory before the compact cell-6 bundle exists. The continuation also retains
`PreparedDataset` scan/RSSI metadata in Python objects during chunked model
work; identity metadata grows with graph size, and virtual-edge enumeration
remains quadratic even though its candidate index is disk-backed. These
boundaries can still exhaust memory for a sufficiently large dataset or dense
virtual graph. No production-scale peak-RSS benchmark has been run. This path
continues beyond LP without falling back to the legacy expanded graph/delta
flow and substantially reduces downstream numeric working memory; it does not
yet make every stage strictly out-of-core for graphs orders of magnitude
larger than RAM.

## Tests run for this implementation

`tests/test_lp_streaming.py` compares the disk-backed pass with `lp.predict_edges`
on tiny fixtures: G/F matrices and residuals, candidate ordering/scores,
additions/removals, endpoint weights/vectors, AP orphan handling, and updated
graph IDs. It also covers no-edge and non-converged cases, artifact lineage,
corruption detection, and refusal to overwrite. `tests/test_streaming_pipeline.py`
runs actual AE/GNN/updater training, AP changes, transition, refresh, snapshot,
multi-batch resume, interrupted-save recovery, no-edge-change followed by a
later batch, hidden-coordinate invariance, and fresh-process resume on tiny
fixtures. `tests/test_gufu_notebook.py` checks the nine-cell structure and mode
handoffs. These establish local calculations and state connectivity, not
large-graph throughput, peak memory, or published model performance.
