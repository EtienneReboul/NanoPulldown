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
│                        (proposed `domains:`; curate into                   │
│                        configs/curated_domains.yaml                        │
│                        whenever convenient — doesn't block folding)        │
├─ folding (Apple Silicon GPU) ───────────────────────────────────────────────┤
│  ESMFold2-Fast: 1 seed, N diffusion samples, no MSA/templates              │
│  ─► raw/sample_<i>.cif.gz + raw/fold.parquet (scores, PAE, pLDDT)         │
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

# 1. declare where baits/preys come from in a run config's `inputs:` block --
#    list files (AlphaPulldown syntax: ID | ID:N | ID:start-stop | ID:N:start-stop),
#    inline accessions, UniProt GO/InterPro queries, IP-MS tables; see
#    scripts/resolve_inputs.py and config_bpm6_curated.yaml. Every run also
#    gets controls (config.yaml `controls:`, "Controls" below): the baits'
#    known direct partners as positives and a matched decoy set as negatives.
#    Preview the counts with --dry-run, then:
NANOPULLDOWN_RUN_CONFIG=config_<run>.yaml python scripts/expand_pairs.py
#    -> configs/pairs.parquet rows (skips pairs over esmfold2.max_tokens),
#       configs/proteins.parquet (UniProt names), configs/inputs_<run>.parquet,
#       and the run's pair list (point the config's `pairs_file:` at it)
#    (no `inputs:` block: configs/baits.txt x configs/preys.txt, as before)

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
#    For a long (multi-day) run use the full command of "Launching a long
#    run" below: --keep-going / --retries / --rerun-incomplete, caffeinate.
snakemake --use-conda --cores 4 --resources gpu=1 docker_heavy=2 \
  --keep-going --retries 2 --rerun-incomplete
#    -> reports/report.zip

# anytime before or after: review data/annotation/<pair>/annotation.yaml and
# curate the `domains:` block into configs/curated_domains.yaml -- optional, only
# affects domain-map figures, never blocks a run
```

## Launching a long run

A screen of thousands of pairs folds for days, so launch it detached, from a
fresh login shell with the controller env activated:

```bash
nohup env -i HOME=$HOME USER=$USER TERM=xterm zsh -lic '
  cd ~/projects/NanoPulldown && conda activate nanopulldown &&
  NANOPULLDOWN_RUN_CONFIG=config_<run>.yaml caffeinate -dimsu \
    snakemake --use-conda --cores 4 --resources gpu=1 docker_heavy=2 \
      --keep-going --retries 2 --rerun-incomplete' \
  > logs_<run>/snakemake_caffeinate.log 2>&1 &
```

- `--keep-going`: without it, ONE failed job (e.g. a PLIP container dying on
  a large pair) stops Snakemake from scheduling anything else; the run then
  sits idle until someone looks. With it, only that pair's downstream steps
  are skipped.
- `--retries 2`: reruns transiently failing jobs (docker, OpenCL, web
  services) before counting them as failed. A fold chunk skips the pairs
  it already folded, so a retry does not redo work.
- `--rerun-incomplete`: reruns jobs a crash or kill left half-written.
- `caffeinate -dimsu`: keeps the Mac awake for the whole run.
- `conda activate nanopulldown`, in a clean shell: do NOT put the
  controller env on `PATH` by hand instead. Its python then stays ahead of
  the rule envs `--use-conda` activates, and `run_esmfold2` dies at once on
  `No module named 'mlx'`.
- Check it is really folding a few minutes after launch:
  `grep "model loaded" logs_<run>/folding/run_esmfold2/*.log`, then
  `grep "^\[fold_batch\]"` on the newest chunk log for per-pair progress.
  List failures with `grep "Error in rule" logs_<run>/snakemake_caffeinate.log`.
- To stop: `pkill -f bin/snakemake; pkill -f run_esmfold2_batch.py`. After a
  hard kill, `NANOPULLDOWN_RUN_CONFIG=config_<run>.yaml snakemake --unlock`
  before relaunching.
- In fragment mode, `expand_pairs.py` writes nothing if any protein's domain
  annotation fails. InterPro can answer 408 for an accession on every retry
  (P93736, P56757 did). Give such proteins their domains in
  `configs/fragment_domains.yaml` (e.g. Pfam from
  `https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/uniprot/<ACC>`)
  and rerun; the other annotations are cached.

## Fragment mode (domain x domain, as in yeast two-hybrid)

Set `fragments.enabled: true` in a run overlay (see `config_drm2_fragments.yaml`)
and every bait and prey is cut into its annotated domains before pairing:
the screen folds bait domain x prey domain instead of full length x full
length, the way two-hybrid constructs are made. Proteins too long to fold
whole under `esmfold2.max_tokens` become screenable by domain (DRM2's
interactors AGO4/AGO6/TPR2/CYP65 are skipped full length, folded as fragments).

```bash
# preview the cuts (nothing written)
NANOPULLDOWN_RUN_CONFIG=config_drm2_fragments.yaml python scripts/fragment_domains.py Q9M548 Q9ZVD5
#   Q9ZVD5:57-458   402 aa  interpro  Protein argonaute, N-terminal + Argonaute, linker 1 domain + PAZ domain
#   Q9ZVD5:462-924  463 aa  interpro  Argonaute-like, PIWI domain + Piwi domain
NANOPULLDOWN_RUN_CONFIG=config_drm2_fragments.yaml python scripts/expand_pairs.py   # or --fragments
NANOPULLDOWN_RUN_CONFIG=config_drm2_fragments.yaml snakemake --use-conda --cores 4 --resources gpu=1 docker_heavy=2
```

- Domains are the ones the domain-map figures show (InterPro representative
  set, `scripts/merge_annotation.py`). Abutting or overlapping domains are
  merged into one fragment (`merge_gap`), as are repeats of the same domain
  (`repeat_gap`). Domains with a linker between them stay separate. Short
  blocks are dropped (`min_length`), and `padding` residues are added on each
  side. `configs/fragment_domains.yaml` replaces them by hand per
  accession, e.g. with the constructs of a published screen.
- `sides:` picks which side is cut. `include_full_length:` adds the whole
  protein as a control. `without_domains:` keeps (`full_length`) or drops
  (`skip`) proteins with no domain. Specs that already give regions
  (`ID:start-stop`) are kept as given.
- Each fragment is an ordinary region pair (`Q9M548x285-626__Q9ZVD5x462-924`),
  so the rest of the pipeline is unchanged. The fragment's domain name is
  stored in `configs/pairs.parquet`, shown in figure labels, and added as
  `bait_fragment` / `prey_fragment` columns in the results table. Every
  fragment is listed in `configs/fragments_<run>.parquet`.
- Fragment chains are numbered from 1 in structures and figures, not in
  full-length numbering: add the fragment start minus 1 to map back.
  Their domain annotation is the full-length entry's, projected onto the
  fragment. No per-fragment InterProScan submission is made.

## Controls (positives and decoys, calibrated hit calling)

`scripts/expand_pairs.py` adds two control sets to every run's preys and
pairs them with every bait (config.yaml `controls:`; a run overlay can turn
either off, e.g. `controls: {decoys: {enabled: false}}`). They are tagged in
`prey_sources` of the results table, and the Snakefile's `decoy_calibration`
rule (in the report, "Controls") reads them.

**Positives (`known`): the baits' known direct partners.** They come from
UniProt, IntAct, MINT, BioGRID, STRING and Complex Portal
(`scripts/fetch_interactors.py`, snapshot
`data/interactors/interactors_<run>.tsv`). Only partners with direct
evidence are kept. That means at least one publication with a PSI-MI
*direct interaction* type, such as BioGRID's two-hybrid and reconstituted
complexes, or one of its descendants (covalent binding, enzymatic reactions
such as ubiquitination). A two-hybrid detection method, or a descendant of
it, also counts. Co-IP and affinity capture-MS (*association*, *physical
association*) do not count, and neither does Complex Portal co-membership.
STRING alone does not count either: its API gives no per-evidence detail,
so its score is only carried as `known_string_score`. Of DRM2's 12 reported
partners, 2 pass. `direct_methods: ["MI:0018", "MI:0090"]` also accepts
protein-complementation assays (BiFC, split luciferase).
`direct_only: false` takes every reported partner, as the former
`interactors:` input source did (`config_drm2_ppi.yaml`). Positives are cut
like any prey in fragment mode. A screen prey that is a known partner gets
the `known_*` columns either way.

**Negatives (`decoy`): a matched decoy set** (`scripts/build_decoys.py`).
It is built the way DUD-E builds one for docking: real, folded proteins
matched to the preys but unable to meet the bait. They are Arabidopsis
proteins found only in plastid stroma, thylakoid lumen, mitochondrial
matrix, apoplast, vacuolar lumen, ER lumen, peroxisomal matrix or the
extracellular region. They have no nuclear or cytosolic annotation and no
known interaction with the baits, direct or not. In fragment mode they are
cut with the run's own `fragments:` rules. They are then matched to the
run's screen prey fragments on length and on Cys+His content: zinc-binding
RING/U-box blocks score differently from large globular domains. The match
uses only the preys that fit under `max_tokens`, and leaves out the
positives.

```bash
# preview: positives, decoy match report, pair counts (needs internet; writes nothing but snapshots)
NANOPULLDOWN_RUN_CONFIG=config_<run>.yaml python scripts/expand_pairs.py --dry-run
NANOPULLDOWN_RUN_CONFIG=config_<run>.yaml python scripts/controls.py          # the positives only
NANOPULLDOWN_RUN_CONFIG=config_<run>.yaml python scripts/fetch_interactors.py Q9M548 --direct

# expand + fold as usual: the controls are paired with every bait
NANOPULLDOWN_RUN_CONFIG=config_<run>.yaml python scripts/expand_pairs.py
#    -> configs/decoys_<run>.txt + data/decoys/decoys_<run>.parquet (+ .json)
NANOPULLDOWN_RUN_CONFIG=config_<run>.yaml snakemake --use-conda --cores 4 --resources gpu=1 docker_heavy=2
#    -> reports_<run>/folding/decoy_calibration/: per bait and metric, conformal
#       p-values against the decoys, BH q-values, hits; recall of the positives
```

- **Decoys are drawn once per run.** They are reused while
  `controls.decoys` and the cutting rules are unchanged. Adding preys to a
  run therefore never re-draws, and re-folds, its decoys. A warning says
  when the preys have changed since the draw.
  `expand_pairs.py --rebuild-decoys` (or `build_decoys.py [--n N --seed S]`)
  draws again.
- **Cost:** each decoy is paired with every bait fragment, so the controls
  add `n` × bait fragments folds. The default `n: 500` with 10 bait
  fragments is 5,000 pairs. Lower `n` or switch decoys off for exploratory
  runs.
- **How many decoys:** the smallest conformal p is 1/(n+1). For BH at FDR α
  over m preys to accept k hits, n must be at least about m/(k·α). With
  ~700 preys per bait, 30 hits need ~500 decoys, and 10 hits need ~1,400.
- **Reading the calibration** (`index.html`, `summary.csv`, `recall.csv`):
  - `auc_prey_vs_decoy` close to 0.5 means most preys behave like decoys.
    The median/MAD or adjusted-boxplot fences on the screen alone are then
    justified.
  - `auc_positive_vs_decoy` shows how well a metric separates known binders
    from the null.
  - `recall` (per bait protein) is the fraction of known partners with at
    least one hit pair. In fragment mode most fragment pairs of a true
    partner miss the interface, so recall is the number to compare metrics
    on.
- The decoy pool query, the interactor lookup and the PSI-MI term
  expansion are snapshotted (`data/uniprot_retrieval/decoy_pool_<run>.*`,
  `data/interactors/interactors_<run>.*`, `data/interactors/psi_mi_*.json`).
  The draw is seeded, so a control set can be rebuilt exactly.

## Layout

| Path                                         | What                                                                                             |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| `config.yaml`                              | all shared defaults, both stages                                                                 |
| `config.local.yaml`                        | per-machine overrides (gitignored; copy`.example`) — set `annotate.interproscan.email` here |
| `configs/baits.txt`, `configs/preys.txt` | one protein spec per line; a `list:` input source (or the default lists) for`scripts/expand_pairs.py` |
| `configs/pairs.parquet`                    | every expanded pair's chain spec, one row per pair (generated; schema in `configs/_schema.md`)   |
| `configs/proteins.parquet`                 | UniProt names / gene / AGI locus per accession, and the display names reports use (generated)   |
| `configs/inputs_<run>.parquet`             | which input source picked each protein, plus IP-MS columns (generated)                           |
| `configs/protein_names.yaml`               | hand overrides of display names, by accession (tracked)                                          |
| `configs/curated_domains.yaml`             | hand-curated domain maps, by pair (tracked)                                                      |
| `configs/fragment_domains.yaml`            | fragment mode: hand-picked domain fragments, by accession (tracked)                              |
| `configs/fragments_<run>.parquet`          | fragment mode: every bait/prey fragment a run pairs (generated)                                  |
| `envs/`                                    | one conda env per rule group, built by`--use-conda`                                            |
| `tools/`                                   | vendored:`ipsae`, `pinc` (both from ab_initio_pipeline, unchanged)                           |
| `scripts/`                                 | see each script's own docstring for what it reuses/adapts/replaces from ab_initio_pipeline       |
| `Snakefile`                                | the whole pipeline, one default`rule all`                                                      |
| `report/`                                  | `custom.css` (dark theme, from ab_initio_pipeline), `datavzrd/nanopulldown.datavzrd.yaml`    |
| `data/ results/ logs/ reports/`            | runtime outputs (gitignored)                                                                     |

## Worked example

`example_toy__example_toy` (a ubiquitin homodimer smoke test, same protein
ab_initio_pipeline uses for its own worked example) is kept in
`configs/curated_domains.yaml` as a template for a curated entry; its spec is in
`configs/pairs.parquet` (see `configs/_schema.md`).

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
