"""
nanopulldown.core.domains
=========================
Residue -> domain lookups over a pair spec's curated `domains:` block, shared
by the domain x domain figures (PAE confidence, PLIP contacts).
"""
from __future__ import annotations


def domain_map(spec, chain, label=None):
    """resnr -> '<protein>:<domain>' (label = the chain's display name)."""
    label = label or chain
    segs = (spec.get("domains") or {}).get(chain)
    if segs:
        def _lookup(resnr):
            for d in segs:
                if int(d["start"]) <= resnr <= int(d["end"]) and d["kind"] != "morf":
                    return f"{label}:{d['name']}"
            return f"{label}:?"
        return _lookup
    return lambda r: f"{label}:{((r - 1) // 50) * 50 + 1}"
