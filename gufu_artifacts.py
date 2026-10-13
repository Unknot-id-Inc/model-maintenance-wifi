"""Small shared helpers for portable, no-overwrite GUFU artifacts."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_digest(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def atomic_write_bytes(
    path: str | Path, content: bytes, *, overwrite: bool = False
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temp_name, target)
        else:
            try:
                os.link(temp_name, target)
            except FileExistsError:
                raise FileExistsError(f"refusing to overwrite artifact: {target}")
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return target


def atomic_write_json(
    value: Any, path: str | Path, *, overwrite: bool = False
) -> Path:
    if isinstance(value, BaseModel):
        encoded = value.model_dump_json(indent=2).encode("utf-8")
    else:
        encoded = json.dumps(
            value,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    return atomic_write_bytes(path, encoded + b"\n", overwrite=overwrite)


def read_json_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def atomic_torch_save(
    value: dict, path: str | Path, *, overwrite: bool = False
) -> Path:
    import torch

    buffer = io.BytesIO()
    torch.save(value, buffer)
    return atomic_write_bytes(path, buffer.getvalue(), overwrite=overwrite)


def load_torch_state(path: str | Path, *, map_location="cpu") -> dict:
    """Load tensor/primitive checkpoints only; never unpickle arbitrary objects."""
    import torch

    value = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(value, dict):
        raise ValueError("checkpoint root must be a mapping")
    return value
