"""Cached downloads with provenance logging.

Every fetched artefact gets a JSON sidecar ``<file>.provenance.json`` with
URL, SHA-256, byte size, retrieval timestamp and licence, so data/ is fully
reproducible and auditable without committing any raw data.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

CHUNK = 1 << 20


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".provenance.json")


def download(
    url: str,
    dest: Path,
    licence: str = "",
    note: str = "",
    force: bool = False,
    timeout: int = 600,
) -> Path:
    """Download ``url`` to ``dest`` unless a cached copy with sidecar exists.

    Returns ``dest``. Partial downloads are written to a .part file and only
    moved into place on success, so the cache never contains truncated files.
    """
    dest = Path(dest)
    if dest.exists() and sidecar_path(dest).exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(part, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=CHUNK):
                fh.write(chunk)
    part.replace(dest)
    record = {
        "url": url,
        "sha256": sha256_of(dest),
        "bytes": dest.stat().st_size,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "licence": licence,
        "note": note,
    }
    with open(sidecar_path(dest), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
    return dest
