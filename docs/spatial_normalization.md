# Per-site spatial normalization for GNN inputs

The graph feature vector remains `signal_embedding || x || y`; the positional
channels are now optionally produced by one frozen transform per site. Signal
embeddings are passed through unchanged. RSSI cosine pair mining and virtual
edge construction continue to use the normalized RSSI scans only.

## Fit the initial reference transform

Fit only from the initial labeled reference subset. If a caller splits the
initial survey into training and held-out rows, make that split first and fit
from the training rows only. Do not fit from update files, predicted positions,
hidden evaluation labels, or pseudo-location initializations. The GF adapter
reads the leading `x,y` token only, so comma-delimited Wi-Fi tokens later in a
row are not interpreted as coordinate columns.

From the repository root, fit the supplied GF reference file with:

```bash
python spatial_normalization.py fit \
  --input input/GF/fingerprint0.txt \
  --site-id GF \
  --output artifacts/spatial/GF.json \
  --coordinate-units unknown
```

The command reports labeled and unique-position counts, surveyed coordinate
bounds, center, scalar scale, maximum normalized fitting radius, and output
path. It refuses to replace an existing transform unless `--overwrite` is
given. Bounds describe the input survey; they do not describe a reconstructed
floor plan. The transform assumes x and y are in the same Cartesian units and
does not convert geographic coordinates.

For another site, use that site's initial reference file, a matching
`--site-id`, and a separate output path such as
`artifacts/spatial/SITE.json`. Never combine unrelated coordinate frames.

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
transform.save("artifacts/spatial/GF.json")
```

The fitted map is `normalized = (xy - center) / scale`, where the center is
the mean of exact unique fitting positions and the one shared scale is their
maximum radius from that center. `inverse_transform` omits `position_weight`
and recovers the original coordinate frame. Later positions are not clipped or
used to refit the transform.

## Load during graph construction and training

The active `gufu.py` configuration enables the frozen transform at
`artifacts/spatial/GF.json` with `position_weight=1.0`. It resolves the file at
startup, checks that its `site_id` matches `dataset_id`, and includes the
metadata and SHA-256 identity in the graph and run configuration. Graph
construction also checks the saved transform identity before applying it.
Changing the file therefore requires a new run configuration; the active
notebook run ID includes `spatial-v1` so raw-coordinate checkpoints are not
reused as normalized-coordinate checkpoints.

Other callers can opt in through `GraphBuildConfig`:

```python
config = GraphBuildConfig(
    spatial_normalization_enabled=True,
    spatial_transform_path="artifacts/spatial/GF.json",
    position_weight=1.0,
)
graph = build_graph_state(prepared, encoder_features, config)
```

When enabled, a missing, malformed, or wrong-site transform is an error; graph
construction never fits one implicitly. GNN checkpoints store the resolved
`GraphBuildConfig`, which contains the site transform metadata and digest.
Streaming graph and GNN manifests carry the same configuration. Feature cache
identities also depend on the resolved configuration and graph identity.

Only positions available through the graph's existing `reference_label_mask`
enter the normalized sample feature block. Unavailable and hidden positions
use `[0, 0]` in those two input channels; seeded pseudo positions remain in the
raw location-initialization metadata for existing interfaces. Availability is
carried by the mask, not inferred from a coordinate being zero or from the
stored pseudo coordinate. Raw coordinate
targets and evaluation values keep their original order and units. AP-node
features continue to be initialized by averaging the already-normalized sample
features, so the transform is applied exactly once.

The transform preserves two positional channels. `position_weight` multiplies
the normalized coordinates at feature assembly and is independent of the
geometric scale saved by the fitter. The signal embedding is not rescaled or
modified.
