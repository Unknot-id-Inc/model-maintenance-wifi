"""Authoritative parsing and prepared-data contract for GUFU.

Raw parsing belongs here. Downstream stages should consume ``PreparedDataset``
and its named conversions instead of interpreting fingerprint files or pickles.
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
from pathlib import Path
from typing import Literal, Sequence

from data.base.models.base import TimestampedData
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gufu_artifacts import atomic_write_json, read_json_text, sha256_bytes


NormalizationMode = Literal["paper_ae", "legacy_raw_div_120"]
BatchRole = Literal["bootstrap_reference", "update_batch"]
CoordinateRole = Literal["bootstrap_train", "evaluation_only", "unavailable"]
InputFormat = Literal["gufu_db", "gufu_jsonl"]

LEGACY_REGENERATION_MESSAGE = (
    "This legacy artifact does not preserve recoverable full BSSID and source-row "
    "identity. Regenerate it from the original raw fingerprint files with "
    "gufu_data.prepare_dataset_files; do not guess truncated MAC prefixes or row IDs."
)


class PreparedDataError(ValueError):
    """Base error for invalid or unrecoverable prepared input."""


class MalformedRecordError(PreparedDataError):
    """A raw row cannot be parsed under the declared input format."""


class LegacyArtifactError(PreparedDataError):
    """A trusted legacy pickle cannot be mapped without inventing identity."""


class PreprocessingConfig(BaseModel):
    """Explicit and serializable policy for one dataset batch."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    dataset_id: str = Field(min_length=1)
    batch_id: str = Field(min_length=1)
    batch_role: BatchRole
    input_format: InputFormat = "gufu_db"
    base_bssid_vocabulary: tuple[str, ...] = ()
    rssi_threshold_dbm: float | None = None
    prune_randomized_bssids: bool = False
    malformed_record_policy: Literal["drop", "error"] = "drop"
    empty_scan_policy: Literal["keep", "drop"] = "keep"
    duplicate_bssid_policy: Literal["strongest", "last", "error"] = "strongest"
    normalization_mode: NormalizationMode = "paper_ae"

    @field_validator("dataset_id", "batch_id")
    @classmethod
    def identity_is_nonblank(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", value):
            raise ValueError("dataset_id and batch_id must be path-safe identifiers")
        return value

    @field_validator("rssi_threshold_dbm")
    @classmethod
    def threshold_is_finite(cls, value):
        if value is not None and not math.isfinite(value):
            raise ValueError("RSSI threshold must be finite")
        return value

    @field_validator("base_bssid_vocabulary", mode="before")
    @classmethod
    def canonical_base_vocabulary(cls, value):
        if not isinstance(value, (tuple, list)):
            raise ValueError("base_bssid_vocabulary must be an ordered sequence")
        canonical = tuple(canonicalize_bssid(item) for item in value)
        if len(set(canonical)) != len(canonical):
            raise ValueError("base_bssid_vocabulary contains duplicate BSSIDs")
        return canonical


class SourceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    source_file: str
    source_row: int = Field(ge=1)
    source_file_sha256: str
    source_record_sha256: str
    # Update-batch IDs use content that excludes held-out coordinates; the
    # complete raw-row digest above remains available for provenance.
    scan_identity_sha256: str | None = None
    source_kind: Literal["raw_text", "legacy_pickle"] = "raw_text"

    @field_validator("source_file_sha256", "source_record_sha256")
    @classmethod
    def hashes_are_sha256(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("provenance hashes must be lowercase SHA-256 hex")
        return value

    @field_validator("scan_identity_sha256")
    @classmethod
    def optional_scan_identity_is_sha256(cls, value):
        if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("scan identity must be lowercase SHA-256 hex")
        return value


class DroppedRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    source_file: str
    source_row: int = Field(ge=1)
    reason: str
    source_record_sha256: str


class PreparationStats(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    rows_seen: int = Field(ge=0)
    prepared_scans: int = Field(ge=0)
    dropped_records: int = Field(ge=0)
    malformed_records_dropped: int = Field(ge=0)
    empty_scans_dropped: int = Field(ge=0)
    readings_below_threshold: int = Field(ge=0)
    readings_pruned: int = Field(ge=0)
    duplicate_bssid_entries: int = Field(ge=0)
    observed_readings: int = Field(ge=0)


class PreparedScan(TimestampedData, extra="forbid", strict=True, frozen=True):
    """One scan, with a typed timestamp and RSS columns aligned to the dataset.

    A float supplied as ``timestamp`` is interpreted as seconds and converted
    to integer milliseconds; integer values follow ``TimestampedData``'s
    millisecond convention. Current ``gufu_db`` rows have no timestamp.
    """

    timestamp: int | None = None
    scan_id: str
    dataset_id: str
    batch_id: str
    batch_role: BatchRole
    source: SourceProvenance
    coordinate_role: CoordinateRole
    has_coordinates: bool
    rss_dbm: tuple[float, ...]
    observation_mask: tuple[bool, ...]
    # State snapshots may carry inferred RSS as usable AE inputs while this
    # measurement mask remains false. Orchestration stores this sidecar beside
    # the prepared artifact, keeping bootstrap artifact hashes stable.
    value_provenance: tuple[
        Literal["measured", "decoder_prediction", "edge_prediction", "missing"], ...
    ] | None = Field(default=None, exclude=True)

    @field_validator("timestamp", mode="before")
    @classmethod
    def seconds_to_milliseconds(cls, value):
        if value is None or type(value) is int:
            return value
        if type(value) is float and math.isfinite(value):
            return int(round(value * 1000.0))
        raise ValueError("timestamp must be integer milliseconds or float seconds")


class ReferencePosition(BaseModel):
    """Immutable bootstrap coordinate used as a training label."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    scan_id: str
    x: float
    y: float


class EvaluationPosition(BaseModel):
    """Immutable update-batch truth, kept out of training labels."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    scan_id: str
    x: float
    y: float


class PreparedDataset(BaseModel):
    """Validated immutable scan/AP state consumed by all later stages."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    schema_version: Literal[1] = 1
    config: PreprocessingConfig
    bssid_vocabulary: tuple[str, ...]
    ap_ids: tuple[str, ...]
    scans: tuple[PreparedScan, ...]
    reference_positions: tuple[ReferencePosition, ...]
    evaluation_positions: tuple[EvaluationPosition, ...]
    source_files: tuple[str, ...]
    dropped_records: tuple[DroppedRecord, ...]
    stats: PreparationStats

    @model_validator(mode="after")
    def validate_contract(self):
        if len(set(self.bssid_vocabulary)) != len(self.bssid_vocabulary):
            raise ValueError("BSSID vocabulary contains duplicates")
        if any(canonicalize_bssid(mac) != mac for mac in self.bssid_vocabulary):
            raise ValueError("BSSID vocabulary must use canonical full MAC identities")
        expected_ap_ids = tuple(ap_id_for_bssid(mac) for mac in self.bssid_vocabulary)
        if self.ap_ids != expected_ap_ids:
            raise ValueError("AP IDs must align one-to-one with ordered BSSID vocabulary")
        scan_ids = tuple(scan.scan_id for scan in self.scans)
        if len(set(scan_ids)) != len(scan_ids):
            raise ValueError("scan IDs must be unique")
        source_rows = tuple(
            (scan.source.source_file, scan.source.source_row) for scan in self.scans
        )
        if len(set(source_rows)) != len(source_rows):
            raise ValueError("source file/row identity must be unique within a batch")
        if len(set(self.source_files)) != len(self.source_files):
            raise ValueError("source file manifest contains duplicates")
        if any(scan.source.source_file not in self.source_files for scan in self.scans):
            raise ValueError("scan source is missing from source file manifest")
        if any(record.source_file not in self.source_files for record in self.dropped_records):
            raise ValueError("dropped record source is missing from source file manifest")
        scan_by_id = {scan.scan_id: scan for scan in self.scans}
        width = len(self.bssid_vocabulary)
        for scan in self.scans:
            if scan.dataset_id != self.config.dataset_id or scan.batch_id != self.config.batch_id:
                raise ValueError("scan dataset/batch provenance disagrees with config")
            if scan.batch_role != self.config.batch_role:
                raise ValueError("scan role disagrees with config")
            expected_scan_id = stable_scan_id(
                self.config.dataset_id,
                self.config.batch_id,
                scan.source.source_file,
                scan.source.source_row,
                scan.source.scan_identity_sha256 or scan.source.source_record_sha256,
            )
            if scan.scan_id != expected_scan_id:
                raise ValueError("scan ID does not match its stable source identity")
            if len(scan.rss_dbm) != width or len(scan.observation_mask) != width:
                raise ValueError("RSS and observation mask must match vocabulary width")
            if any(not math.isfinite(value) for value in scan.rss_dbm):
                raise ValueError("RSS values must be finite")
            if scan.value_provenance is None:
                for value, observed in zip(scan.rss_dbm, scan.observation_mask):
                    if not observed and value != -120.0:
                        raise ValueError("unobserved RSS columns must use the -120 dBm fill")
            else:
                if len(scan.value_provenance) != width:
                    raise ValueError("per-cell RSS provenance must align to the AP vocabulary")
                for value, observed, origin in zip(
                    scan.rss_dbm, scan.observation_mask, scan.value_provenance
                ):
                    if origin == "measured" and not observed:
                        raise ValueError("measured RSS provenance requires an observed mask")
                    if origin == "missing" and (observed or value != -120.0):
                        raise ValueError("missing RSS provenance requires mask=False and -120 dBm")
                    # The observation mask records whether the source scan had
                    # a physical measurement. A later decoder/edge estimate
                    # may replace the current RSS value while that historical
                    # measurement fact remains true.
            expected_role = (
                "bootstrap_train"
                if scan.batch_role == "bootstrap_reference" and scan.has_coordinates
                else "evaluation_only"
                if scan.batch_role == "update_batch" and scan.has_coordinates
                else "unavailable"
            )
            if scan.coordinate_role != expected_role:
                raise ValueError("coordinate role/availability is inconsistent")
        ref_ids = tuple(position.scan_id for position in self.reference_positions)
        eval_ids = tuple(position.scan_id for position in self.evaluation_positions)
        if len(set(ref_ids)) != len(ref_ids) or len(set(eval_ids)) != len(eval_ids):
            raise ValueError("coordinate tables must not contain duplicate scan IDs")
        if set(ref_ids) & set(eval_ids):
            raise ValueError("reference and evaluation positions must be separate")
        for scan_id in ref_ids:
            scan = scan_by_id.get(scan_id)
            if scan is None or scan.coordinate_role != "bootstrap_train":
                raise ValueError("reference position must name a bootstrap training scan")
        for scan_id in eval_ids:
            scan = scan_by_id.get(scan_id)
            if scan is None or scan.coordinate_role != "evaluation_only":
                raise ValueError("evaluation position must name an evaluation-only scan")
        if set(ref_ids) != {
            scan.scan_id for scan in self.scans if scan.coordinate_role == "bootstrap_train"
        }:
            raise ValueError("bootstrap labels are missing from reference_positions")
        if set(eval_ids) != {
            scan.scan_id for scan in self.scans if scan.coordinate_role == "evaluation_only"
        }:
            raise ValueError("evaluation truth is missing from evaluation_positions")
        if self.stats.prepared_scans != len(self.scans):
            raise ValueError("prepared scan count disagrees with stats")
        if self.stats.dropped_records != len(self.dropped_records):
            raise ValueError("dropped record count disagrees with stats")
        if self.stats.rows_seen != len(self.scans) + len(self.dropped_records):
            raise ValueError("row count does not reconcile with scans and dropped records")
        if self.stats.observed_readings != sum(
            sum(scan.observation_mask) for scan in self.scans
        ):
            raise ValueError("observed reading count disagrees with masks")
        return self

    def ae_matrix(
        self, mode: NormalizationMode | None = None
    ) -> tuple[tuple[float, ...], ...]:
        """Return named AE input values in scan/vocabulary order."""
        selected_mode = mode or self.config.normalization_mode
        return tuple(
            raw_dbm_row_to_ae(
                scan.rss_dbm, ae_input_observation_mask(scan), selected_mode
            )
            for scan in self.scans
        )

    def mask_matrix(self) -> tuple[tuple[bool, ...], ...]:
        return tuple(scan.observation_mask for scan in self.scans)

    def training_reference_positions(self) -> tuple[ReferencePosition, ...]:
        """Bootstrap-only labels; evaluation truth is never returned here."""
        return self.reference_positions

    def summary(self) -> dict[str, int | str]:
        return {
            "dataset_id": self.config.dataset_id,
            "batch_id": self.config.batch_id,
            "batch_role": self.config.batch_role,
            "scan_count": len(self.scans),
            "ap_count": len(self.bssid_vocabulary),
            "reference_label_count": len(self.reference_positions),
            "evaluation_truth_count": len(self.evaluation_positions),
            "dropped_record_count": len(self.dropped_records),
        }


def canonicalize_bssid(value: str) -> str:
    """Canonicalize standard full MAC renderings to 12 uppercase hex digits."""
    if not isinstance(value, str):
        raise ValueError("BSSID must be text")
    token = value.strip()
    if re.fullmatch(r"[0-9A-Fa-f]{12}", token):
        compact = token
    elif re.fullmatch(r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", token):
        compact = token.replace(":", "")
    elif re.fullmatch(r"(?:[0-9A-Fa-f]{2}-){5}[0-9A-Fa-f]{2}", token):
        compact = token.replace("-", "")
    elif re.fullmatch(r"[0-9A-Fa-f]{4}(?:\.[0-9A-Fa-f]{4}){2}", token):
        compact = token.replace(".", "")
    else:
        raise ValueError(f"not a complete supported BSSID: {value!r}")
    return compact.upper()


def ap_id_for_bssid(canonical_bssid: str) -> str:
    mac = canonicalize_bssid(canonical_bssid)
    return f"ap:{mac}"


def raw_dbm_row_to_ae(
    raw_dbm: Sequence[float],
    observation_mask: Sequence[bool],
    mode: NormalizationMode = "paper_ae",
) -> tuple[float, ...]:
    """Convert raw dBm to a named AE feature space using the explicit mask."""
    if len(raw_dbm) != len(observation_mask):
        raise ValueError("raw RSS and observation mask lengths differ")
    result: list[float] = []
    for value, observed in zip(raw_dbm, observation_mask):
        if not math.isfinite(value):
            raise ValueError("RSS values must be finite")
        if not observed:
            result.append(0.0 if mode == "paper_ae" else -1.0)
        elif mode == "paper_ae":
            result.append((value + 120.0) / 120.0)
        elif mode == "legacy_raw_div_120":
            result.append(value / 120.0)
        else:  # protects non-Pydantic direct callers
            raise ValueError(f"unsupported AE normalization mode: {mode}")
    return tuple(result)


def ae_input_observation_mask(scan: PreparedScan) -> tuple[bool, ...]:
    """Include explicit inferred values for AE input without changing measured masks."""
    if scan.value_provenance is None:
        return scan.observation_mask
    if len(scan.value_provenance) != len(scan.observation_mask):
        raise ValueError("RSS provenance width does not match the observation mask")
    return tuple(
        observed or origin in {"decoder_prediction", "edge_prediction"}
        for observed, origin in zip(scan.observation_mask, scan.value_provenance)
    )


def ae_value_to_raw_dbm(value: float, mode: NormalizationMode) -> float:
    """Invert one named AE RSS value; missingness remains a separate mask."""
    if not math.isfinite(value):
        raise ValueError("AE RSS values must be finite")
    if mode == "paper_ae":
        return value * 120.0 - 120.0
    if mode == "legacy_raw_div_120":
        return value * 120.0
    raise ValueError(f"unsupported AE normalization mode: {mode}")


def rssi_dbm_to_graph_weight(raw_dbm: float, observed: bool) -> float | None:
    """Return the paper's RSS+120 radio weight, or None for a missing reading."""
    if not observed:
        return None
    if not math.isfinite(raw_dbm):
        raise ValueError("RSS values must be finite")
    return raw_dbm + 120.0


def is_randomized_bssid(canonical_bssid: str) -> bool:
    first_octet = int(canonicalize_bssid(canonical_bssid)[:2], 16)
    return bool(first_octet & 0x02)


def _sha256(data: bytes) -> str:
    return sha256_bytes(data)


def stable_scan_id(
    dataset_id: str,
    batch_id: str,
    source_file: str,
    source_row: int,
    source_record_sha256: str,
) -> str:
    identity = "\0".join(
        (dataset_id, batch_id, source_file, str(source_row), source_record_sha256)
    ).encode("utf-8")
    return "scan:" + _sha256(identity)


def _safe_component(text: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", text):
        raise ValueError("dataset and batch IDs must be path-safe values")
    return text


def _source_names(paths: Sequence[Path], source_root: Path | None) -> dict[Path, str]:
    resolved = [path.resolve() for path in paths]
    if source_root is not None:
        root = source_root.resolve()
        names = {}
        for path in resolved:
            try:
                names[path] = path.relative_to(root).as_posix()
            except ValueError as exc:
                raise ValueError(f"source file {path} is outside source_root {root}") from exc
    else:
        common_dir = Path(os.path.commonpath([str(path.parent) for path in resolved]))
        names = {path: path.relative_to(common_dir).as_posix() for path in resolved}
    if len(set(names.values())) != len(names):
        raise ValueError("input files have duplicate source identities")
    return names


def _parse_db_record(line: str):
    fields = line.strip().split(maxsplit=1)
    if not fields:
        raise ValueError("blank_record")
    coords = fields[0].split(",")
    if len(coords) != 2:
        raise ValueError("coordinate_field_must_have_two_values")
    xy = tuple(float(part) for part in coords)
    if not all(math.isfinite(value) for value in xy):
        raise ValueError("coordinates_must_be_finite")
    pairs = [] if len(fields) == 1 else fields[1].split()
    readings = []
    for pair in pairs:
        parts = pair.split(",")
        if len(parts) < 2:
            raise ValueError("reading_must_contain_bssid_and_rssi")
        bssid = canonicalize_bssid(parts[0])
        rssi = float(parts[1])
        if not math.isfinite(rssi):
            raise ValueError("rssi_must_be_finite")
        readings.append((bssid, rssi))
    return xy, readings, None


def _parse_jsonl_record(line: str):
    payload = json.loads(line)
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("JSON record must contain a data list")
    raw_xy = payload.get("coordinates")
    xy = None
    if raw_xy is not None:
        if not isinstance(raw_xy, list) or len(raw_xy) != 2:
            raise ValueError("coordinates_must_be_a_two_value_array")
        xy = tuple(float(value) for value in raw_xy)
        if not all(math.isfinite(value) for value in xy):
            raise ValueError("coordinates_must_be_finite")
    readings = []
    for item in payload["data"]:
        if not isinstance(item, dict) or "bssid" not in item or "rssi" not in item:
            raise ValueError("each data entry must contain bssid and rssi")
        bssid = canonicalize_bssid(item["bssid"])
        rssi = float(item["rssi"])
        if not math.isfinite(rssi):
            raise ValueError("rssi_must_be_finite")
        readings.append((bssid, rssi))
    timestamp = None
    if "sysTimeMs" in payload:
        raw_time = payload["sysTimeMs"]
        if type(raw_time) is not int:
            raise ValueError("sysTimeMs_must_be_integer_milliseconds")
        timestamp = raw_time
    elif "timestamp" in payload:
        raw_time = payload["timestamp"]
        if type(raw_time) not in (int, float):
            raise ValueError("timestamp_must_be_numeric")
        timestamp = raw_time
    return xy, readings, timestamp


def prepare_dataset_files(
    files: Sequence[str | Path],
    config: PreprocessingConfig,
    *,
    source_root: str | Path | None = None,
) -> PreparedDataset:
    """Parse files once into stable scan/AP identities and explicit masks.

    ``source_row`` is the physical, one-based line number. File ordering does
    not affect scan IDs, vocabulary order, or output scan order.
    """
    if not files:
        raise ValueError("at least one input file is required")
    _safe_component(config.dataset_id)
    _safe_component(config.batch_id)
    paths = [Path(item) for item in files]
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("duplicate input file paths are not allowed")
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
    root = Path(source_root) if source_root is not None else None
    source_names = _source_names(resolved, root)
    ordered = sorted(resolved, key=lambda path: source_names[path])

    raw_rows: list[dict] = []
    dropped: list[DroppedRecord] = []
    rows_seen = malformed_dropped = empty_dropped = 0
    below_threshold = pruned = duplicates = observed_readings = 0
    bssid_set: set[str] = set()

    for path in ordered:
        raw_bytes = path.read_bytes()
        try:
            text = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise PreparedDataError(f"input file is not UTF-8: {source_names[path]}") from exc
        file_digest = _sha256(raw_bytes)
        for row_number, line in enumerate(text.splitlines(), start=1):
            rows_seen += 1
            row_digest = _sha256(line.encode("utf-8"))
            try:
                if config.input_format == "gufu_db":
                    xy, readings, timestamp = _parse_db_record(line)
                else:
                    xy, readings, timestamp = _parse_jsonl_record(line)
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                reason = str(exc) or type(exc).__name__
                if config.malformed_record_policy == "error":
                    raise MalformedRecordError(
                        f"{source_names[path]}:{row_number}: {reason}"
                    ) from exc
                dropped.append(
                    DroppedRecord(
                        source_file=source_names[path],
                        source_row=row_number,
                        reason=reason,
                        source_record_sha256=row_digest,
                    )
                )
                malformed_dropped += 1
                continue

            values: dict[str, float] = {}
            row_dropped = 0
            for mac, rssi in readings:
                if config.prune_randomized_bssids and is_randomized_bssid(mac):
                    pruned += 1
                    continue
                if config.rssi_threshold_dbm is not None and rssi < config.rssi_threshold_dbm:
                    below_threshold += 1
                    continue
                if mac in values:
                    duplicates += 1
                    if config.duplicate_bssid_policy == "error":
                        row_dropped = 1
                        break
                    if config.duplicate_bssid_policy == "strongest":
                        values[mac] = max(values[mac], rssi)
                    else:
                        values[mac] = rssi
                else:
                    values[mac] = rssi
            if row_dropped:
                reason = "duplicate_bssid_in_scan"
                if config.malformed_record_policy == "error":
                    raise MalformedRecordError(
                        f"{source_names[path]}:{row_number}: {reason}"
                    )
                dropped.append(
                    DroppedRecord(
                        source_file=source_names[path],
                        source_row=row_number,
                        reason=reason,
                        source_record_sha256=row_digest,
                    )
                )
                malformed_dropped += 1
                continue
            if not values and config.empty_scan_policy == "drop":
                dropped.append(
                    DroppedRecord(
                        source_file=source_names[path],
                        source_row=row_number,
                        reason="empty_scan_after_filtering",
                        source_record_sha256=row_digest,
                    )
                )
                empty_dropped += 1
                continue

            source_file = source_names[path]
            scan_identity_sha256 = None
            if config.batch_role == "update_batch":
                scan_identity_sha256 = _sha256(
                    json.dumps(
                        {
                            "timestamp": timestamp,
                            "readings": sorted(values.items()),
                        },
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode("utf-8")
                )
            scan_id = stable_scan_id(
                config.dataset_id,
                config.batch_id,
                source_file,
                row_number,
                scan_identity_sha256 or row_digest,
            )
            for bssid in values:
                bssid_set.add(bssid)
            source = SourceProvenance(
                source_file=source_file,
                source_row=row_number,
                source_file_sha256=file_digest,
                source_record_sha256=row_digest,
                scan_identity_sha256=scan_identity_sha256,
            )
            raw_rows.append(
                {
                    "scan_id": scan_id,
                    "source": source,
                    "coordinate_xy": xy,
                    "timestamp": timestamp,
                    "values": values,
                }
            )

    if not config.base_bssid_vocabulary:
        vocabulary = tuple(sorted(bssid_set))
    else:
        base = config.base_bssid_vocabulary
        vocabulary = base + tuple(sorted(bssid_set.difference(base)))
    index = {bssid: idx for idx, bssid in enumerate(vocabulary)}

    scans: list[PreparedScan] = []
    reference_positions: list[ReferencePosition] = []
    evaluation_positions: list[EvaluationPosition] = []
    for row in raw_rows:
        xy = row["coordinate_xy"]
        role: CoordinateRole = (
            "unavailable"
            if xy is None
            else "bootstrap_train"
            if config.batch_role == "bootstrap_reference"
            else "evaluation_only"
        )
        rss = [-120.0] * len(vocabulary)
        mask = [False] * len(vocabulary)
        for mac, value in row["values"].items():
            col = index[mac]
            rss[col] = value
            mask[col] = True
        observed_readings += sum(mask)
        scan = PreparedScan(
            timestamp=row["timestamp"],
            scan_id=row["scan_id"],
            dataset_id=config.dataset_id,
            batch_id=config.batch_id,
            batch_role=config.batch_role,
            source=row["source"],
            coordinate_role=role,
            has_coordinates=xy is not None,
            rss_dbm=tuple(rss),
            observation_mask=tuple(mask),
        )
        scans.append(scan)
        if role == "bootstrap_train":
            reference_positions.append(
                ReferencePosition(scan_id=scan.scan_id, x=xy[0], y=xy[1])
            )
        elif role == "evaluation_only":
            evaluation_positions.append(
                EvaluationPosition(scan_id=scan.scan_id, x=xy[0], y=xy[1])
            )

    scans.sort(key=lambda scan: (scan.source.source_file, scan.source.source_row))
    return PreparedDataset(
        config=config,
        bssid_vocabulary=vocabulary,
        ap_ids=tuple(ap_id_for_bssid(mac) for mac in vocabulary),
        scans=tuple(scans),
        reference_positions=tuple(reference_positions),
        evaluation_positions=tuple(evaluation_positions),
        source_files=tuple(source_names[path] for path in ordered),
        dropped_records=tuple(dropped),
        stats=PreparationStats(
            rows_seen=rows_seen,
            prepared_scans=len(scans),
            dropped_records=len(dropped),
            malformed_records_dropped=malformed_dropped,
            empty_scans_dropped=empty_dropped,
            readings_below_threshold=below_threshold,
            readings_pruned=pruned,
            duplicate_bssid_entries=duplicates,
            observed_readings=observed_readings,
        ),
    )


def save_prepared_dataset(
    dataset: PreparedDataset,
    path: str | Path,
    *,
    overwrite: bool = False,
    value_provenance_path: str | Path | None = None,
) -> Path:
    """Write JSON without replacing an existing artifact unless explicitly asked."""
    has_cell_provenance = any(scan.value_provenance is not None for scan in dataset.scans)
    sidecar = Path(value_provenance_path) if value_provenance_path is not None else None
    if has_cell_provenance and sidecar is None:
        raise ValueError("updated reference values require a separate value-provenance sidecar")
    if not overwrite and (Path(path).exists() or (sidecar is not None and sidecar.exists())):
        raise FileExistsError(f"refusing to overwrite prepared artifact or provenance sidecar: {path}")
    try:
        if sidecar is not None:
            atomic_write_json(
                {
                    "format": "gufu_rss_value_provenance_v1",
                    "scan_ids": tuple(scan.scan_id for scan in dataset.scans),
                    "values": tuple(scan.value_provenance for scan in dataset.scans),
                },
                sidecar,
                overwrite=overwrite,
            )
        return atomic_write_json(dataset, path, overwrite=overwrite)
    except FileExistsError as exc:
        raise FileExistsError(f"refusing to overwrite prepared artifact: {path}") from exc


def load_prepared_dataset(
    path: str | Path,
    *,
    value_provenance_path: str | Path | None = None,
) -> PreparedDataset:
    payload = json.loads(read_json_text(path))
    if value_provenance_path is not None:
        sidecar = json.loads(read_json_text(value_provenance_path))
        scan_ids = tuple(item["scan_id"] for item in payload.get("scans", ()))
        if (
            sidecar.get("format") != "gufu_rss_value_provenance_v1"
            or tuple(sidecar.get("scan_ids", ())) != scan_ids
            or len(sidecar.get("values", ())) != len(scan_ids)
        ):
            raise ValueError("prepared value-provenance sidecar has incompatible scan identity/order")
        for scan, values in zip(payload["scans"], sidecar["values"]):
            if values is None or len(values) != len(scan["rss_dbm"]):
                raise ValueError("prepared value-provenance sidecar has an invalid AP row")
            scan["value_provenance"] = values
    # Strict tuple fields reject the Python lists produced by ``json.loads``;
    # validate the serialized JSON representation so JSON arrays are parsed
    # into the declared immutable tuple fields.
    return PreparedDataset.model_validate_json(json.dumps(payload, allow_nan=False))


def load_legacy_pickle_pair(
    observations_path: str | Path,
    aps_path: str | Path,
    *,
    trusted_local: bool = False,
    config: PreprocessingConfig,
    source_row_by_legacy_id: dict[str, int] | None = None,
) -> PreparedDataset:
    """Adapt validated full-identity local pickles when row metadata is supplied.

    Pickle deserialization can execute code. Callers must explicitly assert the
    files are trusted local artifacts. Historical truncated MACs and missing
    source-row maps are refused with raw-regeneration guidance.
    """
    if not trusted_local:
        raise PermissionError("pickle loading requires trusted_local=True")
    row_map = source_row_by_legacy_id or {}
    obs_path = Path(observations_path)
    ap_path = Path(aps_path)
    obs_bytes = obs_path.read_bytes()
    ap_bytes = ap_path.read_bytes()
    try:
        observations = pickle.loads(obs_bytes)
        aps = pickle.loads(ap_bytes)
    except Exception as exc:
        raise LegacyArtifactError(f"cannot deserialize trusted legacy pickles: {exc}") from exc
    if not isinstance(observations, (list, tuple)) or not isinstance(aps, (list, tuple)):
        raise LegacyArtifactError(f"legacy observation/AP pickles must be sequences. {LEGACY_REGENERATION_MESSAGE}")

    ap_by_legacy_id: dict[str, str] = {}
    ap_bssids: set[str] = set()
    for item in aps:
        try:
            legacy_ap_id, raw_bssid = item[0], item[1]
            if (
                not isinstance(legacy_ap_id, str)
                or not legacy_ap_id
                or not isinstance(raw_bssid, str)
            ):
                raise ValueError("AP ID and BSSID must be strings")
            mac = canonicalize_bssid(raw_bssid)
        except Exception as exc:
            raise LegacyArtifactError(f"invalid legacy AP mapping: {exc}. {LEGACY_REGENERATION_MESSAGE}") from exc
        if legacy_ap_id in ap_by_legacy_id or mac in ap_bssids:
            raise LegacyArtifactError(f"duplicate AP ID/BSSID in legacy pickle. {LEGACY_REGENERATION_MESSAGE}")
        ap_by_legacy_id[legacy_ap_id] = mac
        ap_bssids.add(mac)

    rows: list[dict] = []
    seen_ids: set[str] = set()
    all_bssids: set[str] = set()
    duplicate_entries = 0
    if not isinstance(row_map, dict) or any(
        not isinstance(key, str) or type(value) is not int or value < 1
        for key, value in row_map.items()
    ):
        raise LegacyArtifactError(
            "source-row crosswalk must map legacy string IDs to positive integer rows. "
            + LEGACY_REGENERATION_MESSAGE
        )
    for item in observations:
        if isinstance(item, dict):
            legacy_id = item.get("legacy_id", item.get("id"))
            source_file = item.get("source_file", item.get("device_id"))
            xy = item.get("coordinates", item.get("ground_truth"))
            readings = item.get("rss_dbm", item.get("readings"))
        elif isinstance(item, (list, tuple)) and len(item) >= 4:
            legacy_id, source_file, xy, readings = item[:4]
        else:
            raise LegacyArtifactError(f"invalid legacy observation record. {LEGACY_REGENERATION_MESSAGE}")
        if not isinstance(legacy_id, str) or not legacy_id or legacy_id in seen_ids:
            raise LegacyArtifactError(f"invalid/duplicate legacy scan ID. {LEGACY_REGENERATION_MESSAGE}")
        seen_ids.add(legacy_id)
        if not isinstance(source_file, str) or not source_file:
            raise LegacyArtifactError(f"legacy source file is missing. {LEGACY_REGENERATION_MESSAGE}")
        if legacy_id not in row_map or type(row_map[legacy_id]) is not int or row_map[legacy_id] < 1:
            raise LegacyArtifactError(
                f"source row for legacy scan {legacy_id!r} cannot be recovered. {LEGACY_REGENERATION_MESSAGE}"
            )
        if not isinstance(readings, dict):
            raise LegacyArtifactError(f"legacy readings must be a mapping. {LEGACY_REGENERATION_MESSAGE}")
        normalized: dict[str, float] = {}
        for raw_mac, raw_rssi in readings.items():
            try:
                mac = canonicalize_bssid(raw_mac)
                value = float(raw_rssi)
                if not math.isfinite(value):
                    raise ValueError("non-finite RSS")
            except Exception as exc:
                raise LegacyArtifactError(
                    f"legacy BSSID/RSS mapping is unrecoverable ({raw_mac!r}): {exc}. {LEGACY_REGENERATION_MESSAGE}"
                ) from exc
            if mac in normalized:
                duplicate_entries += 1
                if config.duplicate_bssid_policy == "error":
                    raise LegacyArtifactError(
                        f"duplicate canonical BSSID in legacy scan {legacy_id!r}. "
                        f"{LEGACY_REGENERATION_MESSAGE}"
                    )
                if config.duplicate_bssid_policy == "strongest":
                    normalized[mac] = max(normalized[mac], value)
                else:
                    normalized[mac] = value
            else:
                normalized[mac] = value
        all_bssids.update(normalized)
        if xy in (None, [None, None], (None, None)):
            xy_value = None
        elif isinstance(xy, (list, tuple)) and len(xy) == 2:
            try:
                xy_value = tuple(float(value) for value in xy)
            except (TypeError, ValueError) as exc:
                raise LegacyArtifactError(
                    f"invalid legacy coordinates. {LEGACY_REGENERATION_MESSAGE}"
                ) from exc
            if not all(math.isfinite(value) for value in xy_value):
                raise LegacyArtifactError(f"invalid legacy coordinates. {LEGACY_REGENERATION_MESSAGE}")
        else:
            raise LegacyArtifactError(f"invalid legacy coordinate mapping. {LEGACY_REGENERATION_MESSAGE}")
        canonical_record = json.dumps(
            {"id": legacy_id, "source": source_file, "xy": xy_value, "readings": normalized},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        rows.append(
            {
                "legacy_id": legacy_id,
                "source_file": source_file,
                "source_row": row_map[legacy_id],
                "xy": xy_value,
                "readings": normalized,
                "record_hash": _sha256(canonical_record),
            }
        )
    if set(row_map) - seen_ids:
        raise LegacyArtifactError(
            "source-row map contains unknown legacy scan IDs. "
            + LEGACY_REGENERATION_MESSAGE
        )
    if all_bssids != ap_bssids:
        raise LegacyArtifactError(
            "legacy AP pickle vocabulary does not exactly match observation mappings. "
            + LEGACY_REGENERATION_MESSAGE
        )
    if config.batch_role not in ("bootstrap_reference", "update_batch"):
        raise LegacyArtifactError("legacy batch role must be explicit")

    dropped: list[DroppedRecord] = []
    below_threshold = readings_pruned = empty_dropped = 0
    retained_bssids: set[str] = set()
    for row in rows:
        retained: dict[str, float] = {}
        for mac, value in row["readings"].items():
            if config.prune_randomized_bssids and is_randomized_bssid(mac):
                readings_pruned += 1
                continue
            if config.rssi_threshold_dbm is not None and value < config.rssi_threshold_dbm:
                below_threshold += 1
                continue
            retained[mac] = value
        row["prepared_readings"] = retained
        if not retained and config.empty_scan_policy == "drop":
            dropped.append(
                DroppedRecord(
                    source_file=row["source_file"],
                    source_row=row["source_row"],
                    reason="empty_scan_after_filtering",
                    source_record_sha256=row["record_hash"],
                )
            )
            empty_dropped += 1
            continue
        retained_bssids.update(retained)

    kept_rows = [
        row
        for row in rows
        if not (
            not row["prepared_readings"] and config.empty_scan_policy == "drop"
        )
    ]
    if config.base_bssid_vocabulary:
        base = config.base_bssid_vocabulary
        vocabulary = base + tuple(sorted(retained_bssids.difference(base)))
    else:
        vocabulary = tuple(sorted(retained_bssids))
    col_by_bssid = {mac: i for i, mac in enumerate(vocabulary)}
    pair_hash = _sha256(obs_bytes + b"\0" + ap_bytes)
    scans: list[PreparedScan] = []
    refs: list[ReferencePosition] = []
    evals: list[EvaluationPosition] = []
    for row in kept_rows:
        scan_id = stable_scan_id(
            config.dataset_id,
            config.batch_id,
            row["source_file"],
            row["source_row"],
            row["record_hash"],
        )
        source = SourceProvenance(
            source_file=row["source_file"],
            source_row=row["source_row"],
            source_file_sha256=pair_hash,
            source_record_sha256=row["record_hash"],
            source_kind="legacy_pickle",
        )
        xy = row["xy"]
        role: CoordinateRole = (
            "unavailable"
            if xy is None
            else "bootstrap_train"
            if config.batch_role == "bootstrap_reference"
            else "evaluation_only"
        )
        rss = [-120.0] * len(vocabulary)
        mask = [False] * len(vocabulary)
        for mac, value in row["prepared_readings"].items():
            index = col_by_bssid[mac]
            rss[index] = value
            mask[index] = True
        scan = PreparedScan(
            scan_id=scan_id,
            dataset_id=config.dataset_id,
            batch_id=config.batch_id,
            batch_role=config.batch_role,
            source=source,
            coordinate_role=role,
            has_coordinates=xy is not None,
            rss_dbm=tuple(rss),
            observation_mask=tuple(mask),
        )
        scans.append(scan)
        if role == "bootstrap_train":
            refs.append(ReferencePosition(scan_id=scan_id, x=xy[0], y=xy[1]))
        elif role == "evaluation_only":
            evals.append(EvaluationPosition(scan_id=scan_id, x=xy[0], y=xy[1]))
    scans.sort(key=lambda scan: (scan.source.source_file, scan.source.source_row))
    return PreparedDataset(
        config=config,
        bssid_vocabulary=vocabulary,
        ap_ids=tuple(ap_id_for_bssid(mac) for mac in vocabulary),
        scans=tuple(scans),
        reference_positions=tuple(refs),
        evaluation_positions=tuple(evals),
        source_files=tuple(sorted({row["source_file"] for row in rows})),
        dropped_records=tuple(dropped),
        stats=PreparationStats(
            rows_seen=len(rows),
            prepared_scans=len(scans),
            dropped_records=len(dropped),
            malformed_records_dropped=0,
            empty_scans_dropped=empty_dropped,
            readings_below_threshold=below_threshold,
            readings_pruned=readings_pruned,
            duplicate_bssid_entries=duplicate_entries,
            observed_readings=sum(sum(scan.observation_mask) for scan in scans),
        ),
    )

