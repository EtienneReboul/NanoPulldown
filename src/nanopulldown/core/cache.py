"""
nanopulldown.core.cache
=======================
The per-unique-sequence annotation cache (data/annotation/_cache/<kind>/).
"""
from __future__ import annotations

import json
from pathlib import Path

from .io import seq_md5


def annotation_cache_dir(cfg: dict, kind: str) -> Path:
    """data/annotation/_cache/<kind>/ -- per-UNIQUE-SEQUENCE annotation cache
    (<md5(sequence)>.<ext>), shared by every pair that contains the sequence.
    The bait appears in every pair, so without this it would be re-annotated
    once per pair. Delete the directory to force a refresh."""
    d = Path(cfg["dirs"]["annotation"]) / "_cache" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached_disorder_tracks(cfg: dict, seq: str) -> dict[str, list[float]]:
    """AIUPred tracks ({"disorder": [...], "binding": [...], "linker": [...]},
    one value per residue) of a sequence from the annotation cache; {} if absent."""
    path = annotation_cache_dir(cfg, "disorder") / f"{seq_md5(seq)}.json"
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
