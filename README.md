# nanopulldown

A local, protein-protein-only bait/prey screening pipeline: reuses
AlphaPulldown's UniProt-ID-driven bait/prey screening strategy, folds
locally with **ESMFold2-Fast** on Apple Silicon (MLX/Metal) instead of
AlphaFold-multimer/AF3 — no MSA, no templates, no cluster. Provide rescoring
and downstream analysis with PLIP.

One seed, `esmfold2.num_diffusion_samples` diffusion samples (1 by default; see `esmfold2.max_tokens` for the pair-length limit)
per pair, no pose clustering, one PAE-based domain×domain confidence heatmap
per sample, PLIP contact identification per minimized sample, ONE report at
the end.

## Pipeline

One `Snakefile`, one `rule all`, one command — everything runs on one
machine (no cluster profile, no manual checkpoint to flip) straight
through from sequence annotation to the final report.

**`--resources gpu=1` is required on every invocation, not optional.**
ESMFold2-Fast loads ~25GB per instance into Apple Silicon's shared unified
memory. Nothing else guards against multiple concurrent loads -- there is
no lock or safety net in the script itself, only this flag telling
Snakemake's scheduler to cap `run_esmfold2` at one job at a time. Omit it
and `--cores 4` will happily load 4 models at once; this  will likely results in a
OOM-crash.

```text
┌─ preprocessing (needs internet) ───────────────────────────────────────────┐
│  sequence annotation:  domains (ScanProsite + InterProScan v6 MATCH API)   │
│                        disorder + MoRF (AIUPred docker)                    │
│                     ─► data/annotation/<pair>/annotation.yaml              │
│                        (proposed `domains:`; curate into configs/<pair>.yaml│
│                        whenever convenient — doesn't block folding)        │
├─ folding (Apple Silicon GPU) ───────────────────────────────────────────────┤
│  ESMFold2-Fast: 1 seed, N diffusion samples, no MSA/templates              │
│  ─► metadata compression (arrays.h5 + model_metadata.parquet)             │
│  ─► OpenMM minimization                     │
│  ─► rescoring: ipSAE / iLIS / Pinc + ESMFold2's own pair ipTM             │
│  ─► reports/report.zip  (unzip → report.html) — the ONE report            │
└──────────────────────────────────────────────────────────────────────────┘

snakemake --use-conda --cores 4 --resources gpu=1 docker_heavy=3
```

## Quick start

```bash
# 0. controller env (once)
conda env create -f envs/controller.yaml && conda activate nanopulldown

# 1. define bait/prey lists (AlphaPulldown syntax: ID | ID:N | ID:start-stop | ID:N:start-stop)
#    edit configs/baits.txt and configs/preys.txt, then:
python scripts/expand_pairs.py          # skips pairs over esmfold2.max_tokens (listed in configs/skipped_too_long.tsv)
#    edit config.yaml's `pairs:` list to include the new configs/<pair>.yaml stems

# 2. run the whole pipeline (Apple Silicon GPU via MLX/Metal for folding)
#    --resources gpu=1 is REQUIRED (see above) -- --cores alone doesn't
#    know run_esmfold2's ~25GB model load is one shared GPU resource, and
#    nothing else in this pipeline stops it dispatching several at once.
#    docker_heavy=N (config.yaml's scheduling.docker_heavy_pool) keeps
#    minimize_cif/run_plip from running alongside a model load -- run_esmfold2
#    claims the whole pool -- then lets up to N of them run concurrently
#    once folding for a pair is done. Tune N to your machine; 3 caused a
#    swap/OOM crash on the dev Mac when it was 3 PLIP containers at once
#    with no folding running (see git history) -- start lower (e.g. 2) and
#    watch `vm.swapusage` / Activity Monitor before raising it.
snakemake --use-conda --cores 4 --resources gpu=1 docker_heavy=3
#    -> reports/report.zip

# anytime before or after: review data/annotation/<pair>/annotation.yaml and
# curate the `domains:` block into configs/<pair>.yaml -- optional, only
# affects domain-map figures, never blocks a run
```

## Layout

| Path                                         | What                                                                                             |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `config.yaml`                              | all shared defaults, both stages                                                                 |
| `config.local.yaml`                        | per-machine overrides (gitignored; copy`.example`) — set `annotate.interproscan.email` here |
| `configs/baits.txt`, `configs/preys.txt` | one protein spec per line; expanded into pairs by`scripts/expand_pairs.py`                     |
| `configs/<bait>__<prey>.yaml`              | per-pair chain spec + curated domains                                                            |
| `envs/`                                    | one conda env per rule group, built by`--use-conda`                                            |
| `tools/`                                   | vendored:`ipsae`, `pinc` (both from ab_initio_pipeline, unchanged)                           |
| `scripts/`                                 | see each script's own docstring for what it reuses/adapts/replaces from ab_initio_pipeline       |
| `Snakefile`                                | the whole pipeline, one default`rule all`                                                      |
| `report/`                                  | `custom.css` (dark theme, from ab_initio_pipeline), `datavzrd/nanopulldown.datavzrd.yaml`    |
| `data/ results/ logs/ reports/`            | runtime outputs (gitignored)                                                                     |

## Worked example

`configs/example_toy__example_toy.yaml` (a ubiquitin homodimer smoke test,
same protein ab_initio_pipeline uses for its own worked example) is kept as
a template for the schema in `configs/_schema.md`.

## Implementation notes worth knowing before relying on this in production

- **ESMFold2's MLX path**: this pipeline assumes the Apple Silicon MLX model
  populates `result.pae` / `result.pair_chains_iptm` exactly like the
  Torch/CUDA path does — confirm once against a live run.
- **AIUPred docker CLI**: `scripts/annotate_disorder_aiupred.py` assumes a
  `-i/-t/-o` CLI contract; confirm with
  `docker run --rm ghcr.io/doszilab/aiupred:cpu --help` and adjust if it
  differs.
- **InterProScan v6 MATCH API**: `scripts/annotate_domains.py`'s MD5-cache
  lookup endpoint/schema should be confirmed against current EBI docs.
- **OpenMM OpenCL platform**: `scripts/minimize_openmm.py` prefers OpenCL
  (Mac Metal-backed GPU accel) with a CPU fallback — confirm OpenCL actually
  initializes on the target Mac.
- **PLIP docker image on Apple Silicon**: `docker.io/pharmai/plip` has no
  arm64 build, so `plip.docker_platform` defaults to `linux/amd64` (x86
  emulation via Docker Desktop's Rosetta) — slower than a native image, and
  needs "Use Rosetta for x86/amd64 emulation" enabled in Docker Desktop.
