"""Deterministic pair mining from normalized RSS measurements.

Missing AP readings are represented as zero in the normalized feature vector,
and the observation masks prevent pairs with too little shared evidence from
being compared. Positive pairs are thresholded; negative pairs are bounded per
sample and selected as semi-hard negatives just below the negative threshold.
"""

from __future__ import annotations

import math
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator


RSSI_SIMILARITY_SPACE = "normalized_rssi_cosine_min_overlap_v1"


class NegativeSamplePair(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    left_sample_id: str
    right_sample_id: str
    signal_cosine_similarity: float = Field(ge=-1.0, le=1.0)
    negative_strength: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def canonical_pair(self):
        if self.left_sample_id >= self.right_sample_id:
            raise ValueError("negative-pair endpoints must be distinct and canonically ordered")
        if not math.isfinite(self.signal_cosine_similarity) or not math.isfinite(
            self.negative_strength
        ):
            raise ValueError("negative-pair scores must be finite")
        return self


def normalized_rssi_cosine(
    left: Sequence[float],
    right: Sequence[float],
    left_mask: Sequence[bool],
    right_mask: Sequence[bool],
    *,
    minimum_shared_aps: int,
) -> float | None:
    """Cosine of full zero-filled normalized vectors after an overlap check.

    Cosine uses the same fixed AP space as ``PreparedDataset.ae_matrix()``.
    Masks are used only to reject pairs with too few jointly measured APs; this
    preserves the metric users inspect while avoiding unstable one-AP scores.
    """
    if len(left) != len(right) or len(left) != len(left_mask) or len(left) != len(right_mask):
        raise ValueError("RSSI vectors and observation masks must share one AP width")
    if minimum_shared_aps < 1:
        raise ValueError("minimum_shared_aps must be positive")
    if sum(bool(a) and bool(b) for a, b in zip(left_mask, right_mask)) < minimum_shared_aps:
        return None
    left_values = tuple(float(value) for value in left)
    right_values = tuple(float(value) for value in right)
    if any(not math.isfinite(value) for value in left_values + right_values):
        raise ValueError("normalized RSSI vectors must contain finite values")
    left_norm = math.sqrt(sum(value * value for value in left_values))
    right_norm = math.sqrt(sum(value * value for value in right_values))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    similarity = sum(a * b for a, b in zip(left_values, right_values)) / (
        left_norm * right_norm
    )
    return max(-1.0, min(1.0, similarity))


def negative_strength(similarity: float, negative_similarity_threshold: float) -> float:
    """Map lower RSS cosine to a stronger, bounded negative-pair weight."""
    denominator = max(1.0 + negative_similarity_threshold, 1.0e-12)
    severity = (negative_similarity_threshold - similarity) / denominator
    severity = max(0.0, min(1.0, severity))
    # Keep selected negatives useful near the boundary while giving clearly
    # dissimilar scans more influence.
    return 0.5 + 0.5 * severity


class BoundedNegativeSampler:
    """Keep deterministic semi-hard candidates with a strict degree cap."""

    def __init__(
        self,
        sample_ids: Sequence[str],
        *,
        negative_similarity_threshold: float,
        max_pairs_per_sample: int,
    ):
        if not -1.0 < negative_similarity_threshold <= 1.0:
            raise ValueError("negative similarity threshold must be in (-1, 1]")
        if max_pairs_per_sample < 1:
            raise ValueError("max_pairs_per_sample must be positive")
        self.sample_ids = tuple(sample_ids)
        self.threshold = float(negative_similarity_threshold)
        self.cap = int(max_pairs_per_sample)
        self._best_by_sample: list[list[tuple[float, int]]] = [
            [] for _ in self.sample_ids
        ]

    def add(self, left_row: int, right_row: int, similarity: float) -> None:
        if left_row == right_row:
            return
        if not math.isfinite(similarity) or similarity > self.threshold:
            return
        for row, other in ((left_row, right_row), (right_row, left_row)):
            candidates = self._best_by_sample[row]
            candidates.append((float(similarity), other))
            candidates.sort(key=lambda item: (-item[0], self.sample_ids[item[1]]))
            del candidates[self.cap :]

    def selected(self) -> tuple[NegativeSamplePair, ...]:
        candidates: dict[tuple[int, int], float] = {}
        for left_row, rows in enumerate(self._best_by_sample):
            for similarity, right_row in rows:
                pair = (min(left_row, right_row), max(left_row, right_row))
                candidates[pair] = similarity
        ordered = sorted(
            ((similarity, left, right) for (left, right), similarity in candidates.items()),
            key=lambda item: (
                -item[0],
                self.sample_ids[item[1]],
                self.sample_ids[item[2]],
            ),
        )
        degrees = [0] * len(self.sample_ids)
        selected: list[NegativeSamplePair] = []
        for similarity, left_row, right_row in ordered:
            if degrees[left_row] >= self.cap or degrees[right_row] >= self.cap:
                continue
            left_id, right_id = sorted(
                (self.sample_ids[left_row], self.sample_ids[right_row])
            )
            selected.append(
                NegativeSamplePair(
                    left_sample_id=left_id,
                    right_sample_id=right_id,
                    signal_cosine_similarity=similarity,
                    negative_strength=negative_strength(similarity, self.threshold),
                )
            )
            degrees[left_row] += 1
            degrees[right_row] += 1
        return tuple(
            sorted(selected, key=lambda pair: (pair.left_sample_id, pair.right_sample_id))
        )


def mine_rssi_pairs(
    sample_ids: Sequence[str],
    signal_features: Sequence[Sequence[float]],
    observation_masks: Sequence[Sequence[bool]],
    *,
    virtual_edge_sigma: float,
    negative_similarity_threshold: float,
    minimum_shared_aps: int,
    max_negative_pairs_per_sample: int,
    virtual_edge_strategy: Literal["paper_cosine_threshold", "top_k_alternative"] = "paper_cosine_threshold",
    top_k: int | None = None,
) -> tuple[tuple[tuple[str, str, float], ...], tuple[NegativeSamplePair, ...]]:
    """Mine positives and a bounded negative set from one ordered RSS matrix."""
    ids = tuple(sample_ids)
    features = tuple(tuple(float(value) for value in row) for row in signal_features)
    masks = tuple(tuple(bool(value) for value in row) for row in observation_masks)
    if len(ids) != len(features) or len(ids) != len(masks):
        raise ValueError("sample IDs, RSSI features, and masks must align")
    if len(set(ids)) != len(ids):
        raise ValueError("sample IDs must be unique for pair mining")
    widths = {len(row) for row in features} | {len(row) for row in masks}
    if len(widths) > 1:
        raise ValueError("RSSI features and masks must have one fixed AP width")
    if virtual_edge_strategy == "top_k_alternative":
        if top_k is None or top_k < 1:
            raise ValueError("top_k_alternative requires a positive top_k")
    sampler = BoundedNegativeSampler(
        ids,
        negative_similarity_threshold=negative_similarity_threshold,
        max_pairs_per_sample=max_negative_pairs_per_sample,
    )
    positives: list[tuple[str, str, float]] = []
    neighbors: list[list[tuple[str, float]]] = [[] for _ in ids]
    for left_row, left_id in enumerate(ids):
        for right_row in range(left_row + 1, len(ids)):
            similarity = normalized_rssi_cosine(
                features[left_row],
                features[right_row],
                masks[left_row],
                masks[right_row],
                minimum_shared_aps=minimum_shared_aps,
            )
            if similarity is None:
                continue
            right_id = ids[right_row]
            if virtual_edge_strategy == "paper_cosine_threshold":
                if similarity > virtual_edge_sigma:
                    canonical_left, canonical_right = sorted((left_id, right_id))
                    positives.append((canonical_left, canonical_right, similarity))
            else:
                neighbors[left_row].append((right_id, similarity))
                neighbors[right_row].append((left_id, similarity))
            if similarity <= negative_similarity_threshold:
                sampler.add(left_row, right_row, similarity)

    if virtual_edge_strategy == "top_k_alternative":
        chosen: set[tuple[str, str]] = set()
        for row, candidates in enumerate(neighbors):
            candidates.sort(key=lambda item: (-item[1], item[0]))
            for neighbor_id, _similarity in candidates[:top_k]:
                chosen.add(tuple(sorted((ids[row], neighbor_id))))
        by_id = {sample_id: row for row, sample_id in enumerate(ids)}
        positives = []
        for left_id, right_id in sorted(chosen):
            left_row, right_row = by_id[left_id], by_id[right_id]
            similarity = normalized_rssi_cosine(
                features[left_row],
                features[right_row],
                masks[left_row],
                masks[right_row],
                minimum_shared_aps=minimum_shared_aps,
            )
            if similarity is not None:
                positives.append((left_id, right_id, similarity))
    positives.sort(key=lambda item: (item[0], item[1]))
    return tuple(positives), sampler.selected()
