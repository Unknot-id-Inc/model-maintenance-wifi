"""Small, consistent console reporting for model-training epochs."""

from __future__ import annotations

import math
from collections.abc import Mapping


DEFAULT_LOG_EVERY_K_EPOCHS = 10


def validate_log_interval(every_k_epochs: int) -> int:
    """Return a valid positive integer epoch-reporting interval."""
    if type(every_k_epochs) is not int or every_k_epochs < 1:
        raise ValueError("log_every_epochs must be a positive integer")
    return every_k_epochs


def should_log_training_epoch(
    epoch: int,
    total_epochs: int,
    every_k_epochs: int = DEFAULT_LOG_EVERY_K_EPOCHS,
) -> bool:
    """Return whether this epoch is on the reporting cadence or is final."""
    validate_log_interval(every_k_epochs)
    if type(epoch) is not int or type(total_epochs) is not int:
        raise ValueError("epoch numbers must be integers")
    if total_epochs < 1 or epoch < 1 or epoch > total_epochs:
        raise ValueError("epoch must be between 1 and total_epochs")
    return epoch % every_k_epochs == 0 or epoch == total_epochs


def log_training_epoch(
    stage: str,
    epoch: int,
    total_epochs: int,
    metrics: Mapping[str, int | float],
    *,
    every_k_epochs: int = DEFAULT_LOG_EVERY_K_EPOCHS,
) -> bool:
    """Print one scalar-metric line every K epochs and always on the final epoch.

    Returns whether a line was emitted. Callers pass detached Python scalars so
    progress reporting cannot retain autograd graphs.
    """
    if not stage.strip():
        raise ValueError("training stage name must not be blank")
    if not should_log_training_epoch(epoch, total_epochs, every_k_epochs):
        return False

    formatted: list[str] = []
    for name, raw_value in metrics.items():
        if not name.strip():
            raise ValueError("metric names must not be blank")
        if type(raw_value) not in (int, float) or not math.isfinite(float(raw_value)):
            raise ValueError(f"training metric {name!r} must be a finite Python scalar")
        value = str(raw_value) if type(raw_value) is int else f"{raw_value:.6g}"
        formatted.append(f"{name}={value}")
    suffix = " | " + " | ".join(formatted) if formatted else ""
    print(f"[{stage}] epoch {epoch}/{total_epochs}{suffix}", flush=True)
    return True
