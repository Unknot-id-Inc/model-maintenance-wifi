"""Frozen, per-site spatial normalization for GUFU GNN input features.

The fitter is deliberately independent of the GUFU input formats. The GF text
adapter below reads only the leading ``x,y`` token from each nonblank line.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn

from gufu_artifacts import atomic_write_json, canonical_json_digest, sha256_file


SPATIAL_TRANSFORM_SCHEMA_VERSION = 1
SPATIAL_TRANSFORM_METHOD = "unique_centroid_max_radius"


class SpatialTransformMetadata(BaseModel):
    """Human-readable provenance and fitted statistics for one site."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: int = SPATIAL_TRANSFORM_SCHEMA_VERSION
    site_id: str
    method: str = SPATIAL_TRANSFORM_METHOD
    center: tuple[float, float]
    scale: float
    coordinate_units: str = "unknown"
    axis_order: tuple[str, str] = ("x", "y")
    labeled_row_count: int = Field(ge=1)
    unique_position_count: int = Field(ge=2)
    raw_coordinate_minima: tuple[float, float]
    raw_coordinate_maxima: tuple[float, float]
    provenance: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_metadata(self):
        if self.schema_version != SPATIAL_TRANSFORM_SCHEMA_VERSION:
            raise ValueError("unsupported spatial transform schema_version")
        if not self.site_id.strip():
            raise ValueError("site_id must not be blank")
        if self.method != SPATIAL_TRANSFORM_METHOD:
            raise ValueError(f"method must be {SPATIAL_TRANSFORM_METHOD!r}")
        if self.axis_order != ("x", "y"):
            raise ValueError('axis_order must be ["x", "y"]')
        if not self.coordinate_units.strip():
            raise ValueError("coordinate_units must not be blank; use 'unknown' if unspecified")
        if len(self.center) != 2 or not all(math.isfinite(value) for value in self.center):
            raise ValueError("center must contain two finite coordinates")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("scale must be finite and strictly positive")
        if self.unique_position_count > self.labeled_row_count:
            raise ValueError("unique position count cannot exceed labeled row count")
        if not all(math.isfinite(value) for value in (*self.raw_coordinate_minima, *self.raw_coordinate_maxima)):
            raise ValueError("raw coordinate bounds must be finite")
        if any(lo > hi for lo, hi in zip(self.raw_coordinate_minima, self.raw_coordinate_maxima)):
            raise ValueError("raw coordinate minima must not exceed maxima")
        return self


class FittedSpatialTransform(nn.Module):
    """A fixed differentiable transform with center and scale as buffers."""

    def __init__(self, metadata: SpatialTransformMetadata):
        super().__init__()
        self.metadata = metadata
        self.register_buffer("center", torch.tensor(metadata.center, dtype=torch.float64))
        self.register_buffer("scale", torch.tensor(metadata.scale, dtype=torch.float64))

    @property
    def site_id(self) -> str:
        return self.metadata.site_id

    @property
    def coordinate_units(self) -> str:
        return self.metadata.coordinate_units

    @property
    def transform_sha256(self) -> str:
        return canonical_json_digest(self.metadata.model_dump(mode="json"))

    @property
    def max_normalized_fitting_radius(self) -> float:
        return self.metadata.scale / self.metadata.scale

    def forward(self, xy: Tensor) -> Tensor:
        """Apply the unweighted map while preserving batch dimensions/gradients."""
        _validate_tensor_shape(xy, "coordinates")
        if not xy.is_floating_point():
            raise TypeError("coordinate tensors must use a floating-point dtype")
        center = self.center.to(device=xy.device, dtype=xy.dtype)
        scale = self.scale.to(device=xy.device, dtype=xy.dtype)
        return (xy - center) / scale

    def transform(self, xy):
        """Transform a tensor, NumPy array, or coordinate sequence.

        Tensor inputs are passed through the differentiable module path and
        retain their device and dtype. Array-like inputs return a NumPy array.
        """
        if isinstance(xy, Tensor):
            return self(xy)
        values = np.asarray(xy, dtype=np.float64)
        _validate_array_shape(values, "coordinates")
        result = self(torch.as_tensor(values, dtype=torch.float64, device=self.center.device))
        return result.detach().cpu().numpy()

    def inverse_transform(self, normalized_xy):
        """Invert the unweighted map, preserving tensor autograd when supplied."""
        if isinstance(normalized_xy, Tensor):
            _validate_tensor_shape(normalized_xy, "normalized coordinates")
            if not normalized_xy.is_floating_point():
                raise TypeError("coordinate tensors must use a floating-point dtype")
            center = self.center.to(device=normalized_xy.device, dtype=normalized_xy.dtype)
            scale = self.scale.to(device=normalized_xy.device, dtype=normalized_xy.dtype)
            return normalized_xy * scale + center
        values = np.asarray(normalized_xy, dtype=np.float64)
        _validate_array_shape(values, "normalized coordinates")
        return values * np.asarray(self.metadata.scale) + np.asarray(self.metadata.center)

    def save(self, path: str | Path, *, overwrite: bool = False) -> Path:
        return atomic_write_json(self.metadata, path, overwrite=overwrite)


def _validate_array_shape(values: np.ndarray, name: str) -> None:
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError(f"{name} must have last dimension 2 (x, y)")


def _validate_tensor_shape(values: Tensor, name: str) -> None:
    if values.ndim < 1 or values.shape[-1] != 2:
        raise ValueError(f"{name} must have last dimension 2 (x, y)")


def _as_float64_array(xy) -> np.ndarray:
    if isinstance(xy, Tensor):
        values = xy.detach().to(device="cpu", dtype=torch.float64).numpy()
    else:
        try:
            values = np.asarray(xy, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("fitting coordinates must be a numeric (N, 2) array") from exc
    if values.size == 0:
        raise ValueError("cannot fit a spatial transform from an empty labeled dataset")
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"fitting coordinates must have shape (N, 2); got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("fitting coordinates must contain only finite values")
    return np.asarray(values, dtype=np.float64)


def fit_spatial_transform(
    xy,
    site_id: str,
    *,
    coordinate_units: str = "unknown",
    provenance: Mapping[str, Any] | None = None,
) -> FittedSpatialTransform:
    """Fit unique-centroid/max-radius normalization on labeled coordinates."""
    if not isinstance(site_id, str) or not site_id.strip():
        raise ValueError("site_id must be a nonblank string")
    values = _as_float64_array(xy)
    # np.unique compares the exact float64 coordinate pairs; no rounding or
    # tolerance-based spatial merging is performed.
    unique = np.unique(values, axis=0)
    if unique.shape[0] < 2:
        raise ValueError(
            "at least two unique fitting positions are required to establish spatial extent"
        )
    center = unique.mean(axis=0, dtype=np.float64)
    radii = np.linalg.norm(unique - center, axis=1)
    scale = float(np.max(radii))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("fitted spatial scale must be finite and strictly positive")
    metadata = SpatialTransformMetadata(
        site_id=site_id,
        center=(float(center[0]), float(center[1])),
        scale=scale,
        coordinate_units=coordinate_units,
        labeled_row_count=int(values.shape[0]),
        unique_position_count=int(unique.shape[0]),
        raw_coordinate_minima=(float(values[:, 0].min()), float(values[:, 1].min())),
        raw_coordinate_maxima=(float(values[:, 0].max()), float(values[:, 1].max())),
        provenance=dict(provenance) if provenance is not None else None,
    )
    return FittedSpatialTransform(metadata)


def fit_spatial_transform_from_gf_file(
    path: str | Path,
    site_id: str,
    *,
    coordinate_units: str = "unknown",
) -> FittedSpatialTransform:
    """Fit from the leading whitespace-delimited ``x,y`` token in GF rows."""
    source = Path(path)
    coordinates: list[tuple[float, float]] = []
    try:
        stream = source.open("r", encoding="utf-8")
    except OSError as exc:
        raise OSError(f"cannot read GF coordinate file {source}: {exc}") from exc
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            first_token = line.split(maxsplit=1)[0]
            parts = first_token.split(",")
            if len(parts) != 2:
                raise ValueError(
                    f"{source}:{line_number}: coordinate token must be exactly x,y; got {first_token!r}"
                )
            try:
                x, y = float(parts[0]), float(parts[1])
            except ValueError as exc:
                raise ValueError(
                    f"{source}:{line_number}: coordinate token must contain two finite floats; got {first_token!r}"
                ) from exc
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(
                    f"{source}:{line_number}: coordinate token must contain two finite floats; got {first_token!r}"
                )
            coordinates.append((x, y))
    if not coordinates:
        raise ValueError(f"{source}: no nonblank coordinate rows were found")
    return fit_spatial_transform(
        coordinates,
        site_id,
        coordinate_units=coordinate_units,
        provenance={"source_file": str(source.resolve()), "source_sha256": sha256_file(source)},
    )


def load_spatial_transform(path: str | Path) -> FittedSpatialTransform:
    """Load and validate an already-fitted JSON transform; never refit."""
    source = Path(path)
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load spatial transform {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"spatial transform {source} must contain a JSON object")
    metadata = SpatialTransformMetadata.model_validate_json(source.read_text(encoding="utf-8"))
    return FittedSpatialTransform(metadata)


def normalized_position_feature(
    transform: FittedSpatialTransform,
    position_raw,
    *,
    available: bool,
):
    """Normalize an available position or return zeros for an unavailable one.

    Unavailable positions always produce finite zeros. Tensor inputs preserve
    their device, dtype, and gradient path through the transform.
    """
    if isinstance(position_raw, Tensor):
        _validate_tensor_shape(position_raw, "raw positions")
        if not position_raw.is_floating_point():
            raise TypeError("coordinate tensors must use a floating-point dtype")
        if not available:
            return torch.zeros_like(position_raw)
        return transform.transform(position_raw)
    values = np.asarray(position_raw, dtype=np.float64)
    if values.shape != (2,):
        raise ValueError("raw position must have shape (2,) in x, y order")
    if not available:
        return np.zeros((2,), dtype=np.float64)
    return transform.transform(values)


def _fit_command(args: argparse.Namespace) -> int:
    transform = fit_spatial_transform_from_gf_file(
        args.input,
        args.site_id,
        coordinate_units=args.coordinate_units,
    )
    transform.save(args.output, overwrite=args.overwrite)
    meta = transform.metadata
    print(f"Labeled rows: {meta.labeled_row_count}")
    print(f"Unique positions: {meta.unique_position_count}")
    print(f"Coordinate bounds: x=[{meta.raw_coordinate_minima[0]:.17g}, {meta.raw_coordinate_maxima[0]:.17g}], "
          f"y=[{meta.raw_coordinate_minima[1]:.17g}, {meta.raw_coordinate_maxima[1]:.17g}]")
    print(f"Fitted center: ({meta.center[0]:.17g}, {meta.center[1]:.17g})")
    print(f"Fitted scale: {meta.scale:.17g}")
    print(f"Maximum normalized fitting radius: {transform.max_normalized_fitting_radius:.17g}")
    print(f"Output: {Path(args.output)}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit a site transform from an initial GF file")
    fit.add_argument("--input", required=True, type=Path)
    fit.add_argument("--site-id", required=True)
    fit.add_argument("--output", required=True, type=Path)
    fit.add_argument("--coordinate-units", default="unknown")
    fit.add_argument("--overwrite", action="store_true", help="explicitly replace an existing transform")
    fit.set_defaults(handler=_fit_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
