"""
nanopulldown.core.config
========================
Config loading: config.yaml with config.local.yaml deep-merged on top when
present, then the per-run overlay named by $NANOPULLDOWN_RUN_CONFIG (the CLI's
--run-config sets it).

Every path in the config (dirs:, configs/...) is relative to the repo root,
and every job runs with the repo root as its working directory (Snakemake
does), so the repo root is found from there: $NANOPULLDOWN_REPO, else the
nearest directory up from the cwd that holds config.yaml, else the checkout
this package lives in (src/nanopulldown/core/ -> ../../..).

Stdlib + pyyaml only, so it can be imported from any of the per-rule conda envs.
"""
from __future__ import annotations

import copy
import os
from collections.abc import Mapping
from pathlib import Path

import yaml


def repo_root() -> Path:
    env = os.environ.get("NANOPULLDOWN_REPO")
    if env:
        return Path(env).resolve()
    cwd = Path.cwd().resolve()
    for d in (cwd, *cwd.parents):
        if (d / "config.yaml").exists() and (d / "configs").is_dir():
            return d
    return Path(__file__).resolve().parents[3]


REPO_ROOT = repo_root()


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
    """config.yaml with config.local.yaml deep-merged on top when present,
    then the per-run overlay named by $NANOPULLDOWN_RUN_CONFIG (e.g.
    config_q9m548_e3_fbox.yaml: its own `pairs:` and `dirs:`) when set."""
    root = Path(repo_root)
    cfg = yaml.safe_load((root / "config.yaml").read_text())
    local = root / "config.local.yaml"
    if local.exists():
        cfg = _deep_merge(cfg, yaml.safe_load(local.read_text()) or {})
    run_cfg = os.environ.get("NANOPULLDOWN_RUN_CONFIG")
    if run_cfg:
        cfg = _deep_merge(cfg, yaml.safe_load((root / run_cfg).read_text()) or {})
    return cfg



class LazyConfig(Mapping):
    """load_config() on first access -- for modules that keep a module-level
    CFG, so importing them reads no file (and picks up a --run-config set
    after import)."""

    def __init__(self, repo_root: Path | str | None = None):
        self._root, self._cfg = repo_root, None

    def _load(self) -> dict:
        if self._cfg is None:
            self._cfg = load_config(self._root or REPO_ROOT)
        return self._cfg

    def __getitem__(self, k):
        return self._load()[k]

    def __iter__(self):
        return iter(self._load())

    def __len__(self):
        return len(self._load())
