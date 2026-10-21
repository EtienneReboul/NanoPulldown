"""
nanopulldown.cli
================
`nanopulldown <group> <command> [args]` (or `python -m nanopulldown ...`).

Stdlib only, and nothing below a command is imported until that command
runs: every per-rule conda env (minimize has no plotting stack, for one)
can load the dispatcher, and each command pulls in only its own deps. A
command is a module-level `main()` that parses `sys.argv[1:]` itself, so the
modules also stay runnable on their own.

Global option: --run-config config_<run>.yaml (same as setting
$NANOPULLDOWN_RUN_CONFIG, the per-run config overlay).
"""
from __future__ import annotations

import importlib
import os
import sys

# group -> {command: ("module:function", help)}; a group of None is a top-level command
COMMANDS: dict[str | None, dict[str, tuple[str, str]]] = {
    "inputs": {
        "expand": ("inputs.expand:main", "resolve the config's inputs into configs/pairs.parquet"),
        "resolve": ("inputs.resolve:main", "preview what the config's `inputs:` selects"),
        "fragments": ("inputs.fragments:main", "preview the domain fragments of some proteins"),
        "decoys": ("inputs.decoys:main", "draw (or preview) the run's matched decoy prey set"),
        "positives": ("inputs.controls:main", "preview the baits' known partners by evidence tier (positive controls)"),
        "interactors": ("services.interactors:main", "preview the known interactors of protein(s)"),
        "uniprot": ("services.uniprot:main", "fetch UniProt sequences / metadata / queries"),
    },
    "annotate": {
        "domains": ("annotate.batch:domains_main", "fill the per-sequence domain cache (InterPro/ScanProsite)"),
        "disorder": ("annotate.batch:disorder_main", "fill the per-sequence AIUPred disorder cache"),
        "merge": ("annotate.merge:main", "one pair's proposed annotation.yaml, from the caches"),
        "domains-tsv": ("annotate.domains:main", "one pair's raw domain hits as TSV (debugging)"),
        "disorder-tsv": ("annotate.disorder:main", "one pair's AIUPred tracks as TSV (debugging)"),
    },
    None: {
        "fold": ("fold.esmfold2:main", "fold pairs with ESMFold2-Fast (MLX); ONE model at a time"),
        "minimize": ("structure.minimize:main", "OpenMM energy minimization of one sample"),
        "metrics": ("structure.metrics:main", "ipSAE / iLIS / Pinc / pair ipTM of one pair"),
        "clashes": ("structure.clashes:main", "steric clashes, raw vs minimized samples"),
        "pydock": ("structure.pydock:main", "pyDock bait-prey energy of one pair's minimized samples"),
        "calibrate": ("report.calibration:main", "decoy-calibrated conformal p-values + BH per bait"),
    },
    "plip": {
        "split-chains": ("plip.chain_split:split_main", "give long chains extra ids (PLIP segfaults on them)"),
        "restore-chains": ("plip.chain_split:restore_main", "map a PLIP report's split chains back"),
        "cxc-config": ("plip.cxc_config:main", "the JSON config `pliparser csv2cxc` reads"),
        "aggregate": ("plip.aggregate:main", "per-sample summaries -> plip_contacts.parquet"),
    },
    "plot": {
        "domain-bar": ("report.plots.domain_bar:main", "domain map + AIUPred tracks of one protein"),
        "domain-heatmap": ("report.plots.domain_heatmap:main", "bait x prey domain PAE confidence"),
        "plip-heatmap": ("report.plots.plip_heatmap:main", "bait x prey domain PLIP contacts"),
        "minimize-energy": ("report.plots.minimize_energy:main", "minimization energy traces of one pair"),
        "violins": ("report.plots.violins:main", "interactive metric distributions"),
        "clash-test": ("report.plots.clash_test:main", "raw vs minimized clash test (Wilcoxon)"),
    },
    "table": {
        "results": ("report.tables:results_main", "the run-wide results table"),
        "minimize-failures": ("report.tables:minimize_failures_main", "per-pair minimization failure rates"),
        "by-bait": ("report.tables:by_bait_main", "one bait's rows of run-wide pair tables"),
    },
    "report": {
        "tables": ("report.bundle:tables_main", "datavzrd table bundle (dark skin)"),
        "build": ("report.bundle:build_main", "snakemake --report zip + dark-theme fixes (never fails)"),
        "style": ("report.bundle:style_main", "dark-theme fixes inside an existing report zip"),
    },
}

GROUP_HELP = {
    "inputs": "build the run's pairs (before Snakemake; needs internet)",
    "annotate": "sequence annotation (domains, disorder)",
    "plip": "PLIP contacts",
    "plot": "report figures",
    "table": "report tables",
    "report": "the HTML report",
}


def _usage() -> str:
    lines = ["usage: nanopulldown [--run-config CONFIG] <command> [args]",
             "       nanopulldown [--run-config CONFIG] <group> <command> [args]", ""]
    for name, (_, h) in COMMANDS[None].items():
        lines.append(f"  {name:<20} {h}")
    for group, cmds in COMMANDS.items():
        if group is None:
            continue
        lines += ["", f"{group}: {GROUP_HELP[group]}"]
        for name, (_, h) in cmds.items():
            lines.append(f"  {group} {name:<{19 - len(group)}} {h}")
    lines += ["", "`nanopulldown <command> --help` for a command's own options."]
    return "\n".join(lines)


def _resolve(argv: list[str]) -> tuple[str, str, list[str]]:
    """(prog, "module:function", remaining args), or exit with usage."""
    head = argv[0] if argv else ""
    if head in COMMANDS[None]:
        return head, COMMANDS[None][head][0], argv[1:]
    if head in COMMANDS and head is not None:
        cmds = COMMANDS[head]
        sub = argv[1] if len(argv) > 1 else ""
        if sub in cmds:
            return f"{head} {sub}", cmds[sub][0], argv[2:]
        known = ", ".join(cmds)
        sys.exit(f"nanopulldown {head}: unknown command {sub!r} (one of: {known})")
    print(_usage(), file=sys.stderr if head else sys.stdout)
    sys.exit(0 if head in ("", "-h", "--help") else 2)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    while argv and argv[0].startswith("--run-config"):
        opt = argv.pop(0)
        if "=" in opt:
            os.environ["NANOPULLDOWN_RUN_CONFIG"] = opt.split("=", 1)[1]
        elif argv:
            os.environ["NANOPULLDOWN_RUN_CONFIG"] = argv.pop(0)
        else:
            sys.exit("nanopulldown: --run-config needs a file")
    prog, target, rest = _resolve(argv)
    mod, func = target.split(":")
    fn = getattr(importlib.import_module(f"nanopulldown.{mod}"), func)
    # commands parse sys.argv themselves; argparse takes `prog` from argv[0]
    sys.argv = [f"nanopulldown {prog}", *rest]
    return int(fn() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
