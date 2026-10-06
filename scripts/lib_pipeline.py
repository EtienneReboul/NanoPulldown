#!/usr/bin/env python3
"""
scripts/lib_pipeline.py
========================
Shared helpers for both stage Snakefiles and their scripts: config loading
(config.yaml + optional config.local.yaml overlay), per-pair spec loading
(configs/<bait>__<prey>.yaml), and small chain/sequence utilities.

Adapted from ab_initio_pipeline's scripts/lib_pipeline.py: drops the HPC/
scheduler/PLIP normalisation (nanopulldown runs one ESMFold2 backend, one
seed, locally, protein-protein only) and renames anchor/partner chains to
bait/prey chains to match AlphaPulldown's own vocabulary.

Kept dependency-free (stdlib + pyyaml) so it can be imported from any of the
per-rule conda envs.
"""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Config ──────────────────────────────────────────────────────────────────

def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(repo_root: Path | str = REPO_ROOT) -> dict:
    """config.yaml with config.local.yaml deep-merged on top when present."""
    root = Path(repo_root)
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    local = root / "config.local.yaml"
    if local.exists():
        cfg = _deep_merge(cfg, yaml.safe_load(local.read_text()) or {})
    return cfg


# ── Per-pair spec ────────────────────────────────────────────────────────────

def load_pair(name: str, repo_root: Path | str = REPO_ROOT) -> dict:
    """configs/<name>.yaml, with a few normalisations:
      - bait_chains / prey_chains always lists
      - annotation_reviewed defaults False
    """
    root = Path(repo_root)
    spec = yaml.safe_load((root / "configs" / f"{name}.yaml").read_text())

    spec["bait_chains"] = list(spec.get("bait_chains", []))
    spec["prey_chains"] = list(spec.get("prey_chains", []))
    spec.setdefault("annotation_reviewed", False)
    spec.setdefault("domains", {})
    return spec


# ── Chain helpers ──────────────────────────────────────────────────────────

def protein_chains(spec: dict) -> list[dict]:
    """Every chain in a pair spec's `sequences:` list (protein-protein only,
    so this is every entry — kept as a function for parity with call sites
    that read like ab_initio_pipeline's protein_chains(spec))."""
    return list(spec.get("sequences", []))


def seq_md5(seq: str) -> str:
    return hashlib.md5(seq.encode("ascii")).hexdigest()


def annotation_cache_dir(cfg: dict, kind: str) -> Path:
    """data/annotation/_cache/<kind>/ -- per-UNIQUE-SEQUENCE annotation cache
    (<md5(sequence)>.<ext>), shared by every pair that contains the sequence.
    The bait appears in every pair, so without this it would be re-annotated
    once per pair. Delete the directory to force a refresh."""
    d = Path(cfg["dirs"]["annotation"]) / "_cache" / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + rename so concurrent jobs never see a partial cache entry."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    tmp.replace(path)


def pair_tokens(spec: dict) -> int:
    """Total token count (= residues, protein-only) ESMFold2 sees for a pair.
    Folding memory grows ~quadratically with this -- see esmfold2.max_tokens."""
    return sum(len(c["sequence"]) for c in protein_chains(spec))


def chain_by_id(spec: dict, cid: str) -> dict | None:
    for s in spec.get("sequences", []):
        if s.get("id") == cid:
            return s
    return None


def bait_chain_ids(spec: dict) -> list[str]:
    return list(spec.get("bait_chains", []))


def prey_chain_ids(spec: dict) -> list[str]:
    return list(spec.get("prey_chains", []))
