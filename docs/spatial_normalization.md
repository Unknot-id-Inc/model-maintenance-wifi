# Per-site spatial normalization for model-facing coordinates

The graph feature vector remains `signal_embedding || x || y`; its positional
channels are produced by one frozen transform per site. The offline and
streaming location/feature updaters use that same transform for coordinate
targets and inputs, and invert it when exporting predicted locations. Signal
embeddings are passed through unchanged. RSSI cosine pair mining and virtual
edge construction continue to use the normalized RSSI scans only.

## Bootstrap fitting and run-local output

The `gufu.py` bootstrap flow fits the transform after preparing the initial
reference dataset. It uses only `training_reference_positions`, then saves the
result under
`output/notebook_runs/GF/<RUN_ID>/artifacts/spatial/GF.json` before building the
graph. Update labels and evaluation-only positions do not enter the fit. The
run ID includes hashes of the selected input files, so changing the GF files
creates a separate output root and cannot silently reuse an old transform or
checkpoint set.

The standalone fitter remains available to direct callers. It reads only the
leading `x,y` token from GF rows and refuses to replace an existing file unless
`--overwrite` is supplied:

```bash
python spatial_normalization.py fit \
  --input input/GF/fingerprint0.txt \
  --site-id GF \
  --output /tmp/GF.json \
  --coordinate-units unknown
```

The command reports labeled and unique-position counts, surveyed coordinate
bounds, center, scalar scale, maximum normalized fitting radius, and output
path. Bounds describe the input survey; they do not describe a reconstructed
floor plan. The transform assumes x and y are in the same Cartesian units and
does not convert geographic coordinates.

For another site, use that site's initial labeled reference subset, a matching
`--site-id`, and a separate output path. Never combine unrelated coordinate
frames.

The core API can also fit directly from an `(N, 2)` NumPy array or PyTorch
tensor:

```python
from spatial_normalization import fit_spatial_transform

transform = fit_spatial_transform(
    labeled_xy,
    site_id="GF",
    coordinate_units="unknown",
    provenance={"source": "initial labeled training subset"},
)
transform.save("/tmp/GF.json")
```

The fitted map is `normalized = (xy - center) / scale`, where the center is
the mean of exact unique fitting positions and the one shared scale is their
maximum radius from that center. `inverse_transform` omits `position_weight`
and recovers the original coordinate frame. Later positions are not clipped or
used to refit the transform.

## Load during graph construction and training

The active `gufu.py` configuration points to the transform inside the current
run's output directory and includes its metadata and SHA-256 identity in the
graph and pipeline configuration. Graph construction and updater loading check
that saved identity before applying it. Re-running with the same inputs reuses
the same run root; changing any selected input file selects a different root.

Other callers can opt in through `GraphBuildConfig`:

```python
config = GraphBuildConfig(
    spatial_normalization_enabled=True,
    spatial_transform_path="output/my-run/artifacts/spatial/GF.json",
    position_weight=1.0,
)
graph = build_graph_state(prepared, encoder_features, config)
```

When enabled, a missing, malformed, or wrong-site transform is an error for
direct graph-building callers; `gufu.py` creates its run-local transform from
prepared bootstrap labels before graph construction. GNN checkpoints store the
resolved `GraphBuildConfig`, which contains the site transform metadata and digest.
Streaming graph and GNN manifests carry the same configuration. Feature cache
identities also depend on the resolved configuration and graph identity.
Both fingerprint-updater paths require this saved transform and use it for
reference coordinate supervision, cross-reference coordinate targets,
location-to-feature inputs, and inverse-transforming predictions. Their
checkpoint manifests bind the same digest; warm-starting with a different
site transform is rejected.

Only positions available through the graph's existing `reference_label_mask`
enter the normalized sample feature block. Unavailable and hidden positions
use `[0, 0]` in those two input channels; seeded pseudo positions remain in the
raw location-initialization metadata for existing interfaces. Availability is
carried by the mask, not inferred from a coordinate being zero or from the
stored pseudo coordinate. Raw coordinate labels, exported predictions, and
evaluation metrics retain their original order and units. The updater models
train in the same normalized coordinate space as the GNN's positional channels;
they use `inverse_transform` when returning a predicted location. Their
coordinate normalization is never refit from an update batch. Updater
checkpoints record the transform digest and reject a different transform.
AP-node features continue to be initialized by averaging the
already-normalized sample features, so the transform is applied exactly once.

The transform preserves two positional channels. `position_weight` multiplies
the normalized coordinates at feature assembly and is independent of the
geometric scale saved by the fitter. The signal embedding is not rescaled or
modified.
