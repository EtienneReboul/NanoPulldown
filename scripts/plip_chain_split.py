#!/usr/bin/env python3
"""
scripts/plip_chain_split.py — Stage 2d (PLIP workaround)
=========================================================
PLIP 3.0 (pharmai/plip) segfaults (exit 139) in --chains mode on any
structure with a chain longer than ~880 residues, whatever the atom count or
memory: in dne1_dcp_ip_fragments, all 3448 pairs whose longest chain was
<= 809 aa passed and all 24 with >= 889 aa crashed. Giving the long chain
two (or more) chain ids, each under the limit, avoids it.

  split    rewrites the PDB in place: every chain longer than --max-len is
           cut into consecutive pieces of <= --max-len residues, each new
           piece getting an unused chain id (residue numbers and atom order
           unchanged). Writes the piece -> original chain map next to the PDB
           (<pdb>.chainmap.json) and prints the PLIP --chains argument with
           the pieces in their side's group. No long chain: PDB untouched,
           empty map, same --chains as before.
  restore  rewrites a PLIP report so only the original chain ids appear
           (residue tokens RES:C:451 and the RESCHAIN / RESCHAIN_LIG table
           columns), so rule plip_parse and everything downstream see the
           pair's own chains.

Stdlib only (runs in the controller env, like the rest of rule run_plip).
"""
from __future__ import annotations

import argparse
import json
import re
import string
from pathlib import Path

ATOM = ("ATOM", "HETATM", "ANISOU", "TER")


def _chain_residues(lines: list[str]) -> dict[str, list[str]]:
    res: dict[str, list[str]] = {}
    for ln in lines:
        if ln.startswith(("ATOM", "HETATM")):
            key = ln[22:27]                       # resSeq + iCode
            seen = res.setdefault(ln[21], [])
            if not seen or seen[-1] != key:
                seen.append(key)
    return res


def split(pdb: Path, bait: list[str], prey: list[str], max_len: int) -> str:
    lines = pdb.read_text().splitlines(keepends=True)
    residues = _chain_residues(lines)
    free = [c for c in string.ascii_uppercase + string.ascii_lowercase + string.digits if c not in residues]
    piece_of: dict[tuple[str, str], str] = {}     # (chain, residue key) -> new chain id
    chain_map: dict[str, str] = {}                # new chain id -> original chain id
    for chain, keys in residues.items():
        if len(keys) <= max_len:
            continue
        n_pieces = -(-len(keys) // max_len)
        size = -(-len(keys) // n_pieces)          # even pieces
        ids = [chain] + [free.pop(0) for _ in range(n_pieces - 1)]   # first piece keeps its id
        chain_map.update({new: chain for new in ids[1:]})
        for i, key in enumerate(keys):
            if i >= size:
                piece_of[(chain, key)] = ids[i // size]
    if chain_map:
        out, last = [], None
        for ln in lines:
            if ln.startswith(ATOM) and len(ln) > 21:
                new = piece_of.get((ln[21], ln[22:27]))
                if ln.startswith("TER"):
                    new = last
                if new:
                    ln = ln[:21] + new + ln[22:]
                    last = new
                elif not ln.startswith("TER"):
                    last = None
            out.append(ln)
        pdb.write_text("".join(out))
    Path(f"{pdb}.chainmap.json").write_text(json.dumps(chain_map))

    def group(side: list[str]) -> list[str]:
        return side + [new for new, orig in chain_map.items() if orig in side]
    return str([group(bait), group(prey)])


def restore(report: Path, chain_map: dict[str, str]) -> int:
    if not chain_map:
        return 0
    token = re.compile(r"(?<=[A-Z]:)([" + "".join(map(re.escape, chain_map)) + r"])(?=:-?\d)")
    out, cols, n = [], [], 0
    for ln in report.read_text().splitlines(keepends=True):
        if ln.startswith("| RESNR"):
            names = [c.strip() for c in ln.strip().strip("|").split("|")]
            cols = [i for i, c in enumerate(names) if c in ("RESCHAIN", "RESCHAIN_LIG")]
        elif ln.startswith("| ") and cols:
            cells = ln.rstrip("\n").split("|")
            for i in cols:
                cell = cells[i + 1]
                if cell.strip() in chain_map:
                    cells[i + 1] = cell.replace(cell.strip(), chain_map[cell.strip()], 1)
                    n += 1
            ln = "|".join(cells) + ("\n" if ln.endswith("\n") else "")
        elif not ln.startswith(("|", "+")):
            cols = cols if ln.strip() == "" else []
            ln, k = token.subn(lambda m: chain_map[m.group(1)], ln)
            n += k
        out.append(ln)
    report.write_text("".join(out))
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--pdb", type=Path, required=True)
    s.add_argument("--bait", required=True, help="comma-joined chain ids")
    s.add_argument("--prey", required=True, help="comma-joined chain ids")
    s.add_argument("--max-len", type=int, default=850)
    r = sub.add_parser("restore")
    r.add_argument("--report", type=Path, required=True)
    r.add_argument("--map", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "split":
        print(split(a.pdb, a.bait.split(","), a.prey.split(","), a.max_len))
    else:
        restore(a.report, json.loads(a.map.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
