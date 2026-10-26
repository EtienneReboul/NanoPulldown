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
│  ─► pyDock energy of minimized samples; Z(AF conf) − Z(pyDock-1VDW)       │
│  ─► reports/report.zip  (unzip → report.html) — the ONE report            │
└──────────────────────────────────────────────────────────────────────────┘

snakemake --use-conda --cores 4 --resources gpu=1 docker_heavy=3
```

## Start a new project

```bash
cookiecutter gh:EtienneReboul/NanoPulldown --directory cookiecutter   # or a local path to cookiecutter/
```

asks for the baits, preys and options and writes a ready-to-run project:
its run overlay and bait/prey lists, plus this pipeline (see
`cookiecutter/README.md`).

## Quick start

```bash
# 0. controller env (once), with the `nanopulldown` command on PATH
conda env create -f envs/controller.yaml && conda activate nanopulldown
pip install -e .        # or, without installing: PYTHONPATH=src python -m nanopulldown ...

# 1. declare where baits/preys come from in a run config's `inputs:` block --
#    list files (AlphaPulldown syntax: ID | ID:N | ID:start-stop | ID:N:start-stop),
#    inline accessions, UniProt GO/InterPro queries, IP-MS tables; see
#    nanopulldown.inputs.resolve and config_bpm6_curated.yaml. Every run also
#    gets controls (config.yaml `controls:`, "Controls" below): the baits'
#    known direct partners as positives and a matched decoy set as negatives.
#    Preview the counts with --dry-run, then:
nanopulldown --run-config config_<run>.yaml inputs expand
#    -> configs/pairs.parquet rows (skips pairs over <fold.engine>.max_tokens),
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
- To stop: `pkill -f bin/snakemake; pkill -f "nanopulldown fold"`. After a
  hard kill, `NANOPULLDOWN_RUN_CONFIG=config_<run>.yaml snakemake --unlock`
  before relaunching.
- In fragment mode, `inputs expand` writes nothing if any protein's domain
  annotation fails. InterPro can answer 408 for an accession on every retry
  (P93736, P56757 did). Give such proteins their domains in
  `configs/fragment_domains.yaml` (e.g. Pfam from
  `https://www.ebi.ac.uk/interpro/api/entry/pfam/protein/uniprot/<ACC>`)
  and rerun; the other annotations are cached.

## Running on the IFB cluster

The whole pipeline also runs on the IFB Core Cluster
(`core.cluster.france-bioinformatique.fr`) under SLURM, with folding on its
GPUs. The point is ensembles: [config.ifb.yaml.example](config.ifb.yaml.example)
folds every pair as **5 independent runs** (`num_seeds: 5`, 1 sample each) at
ESMFold2's regular settings (20 loops, 200 steps) with `lm_dropout: 0.3`, about
9x the Mac's per-pair compute. The Mac defaults in config.yaml are unchanged.

What differs from the Mac:

| Aspect | Mac | IFB |
| --- | --- | --- |
| fold backend | MLX (`esmfold2.backend: mlx`, [envs/esmfold2.yaml](envs/esmfold2.yaml)) | torch on CUDA (`torch`, [envs/esmfold2_cuda.yaml](envs/esmfold2_cuda.yaml)) |
| rule software | conda envs (`--use-conda`) | the same conda envs, built once by `setup.sh` |
| PLIP / AIUPred images | `docker run` | `apptainer run plip.sif` / `aiupred.sif` (no docker daemon) |
| scheduling | `--resources gpu=1 docker_heavy=N` | [profiles/ifb](profiles/ifb/config.yaml): SLURM, up to 8 GPU jobs |

**Inode quota.** `/shared/projects/npf_abinitio` is capped at 500k inodes,
and a full project fails **every** write in it. Measured on Linux builds of
these env files, the rule envs take ~146k (analysis 48k incl. pliparser,
esmfold2_cuda 74k, minimize 24k), and a pair at 5 samples ~120 more (results,
fold cache, logs, figures, Snakemake metadata). Check the headroom first:
`getfattr --only-values -n ceph.dir.rentries /shared/projects/npf_abinitio`.

Fold jobs pick their GPU from config.yaml's `hpc.gpu_classes`, by the chunk's
largest pair: an L40S up to 1200 tokens, else an H200 (nodes 7 and 9
excluded). Their walltime is estimated from the Mac timings. CPU jobs go to
`fast`, sized in the profile.

### One-time IFB setup

1. Copy the checkout to the cluster, and `configs/pairs.parquet` (not tracked).
2. `cp config.ifb.yaml.example config.local.yaml`, and add the Mac
   config.local.yaml's `annotate:` block.
3. As a SLURM job, not on the shared login node (the command is in the
   script's header): `scripts/ifb/setup.sh`. It pulls `plip.sif` and
   `aiupred.sif`, builds the conda envs with `snakemake --conda-create-envs-only`
   (conda's package cache in node-local scratch, not on the project; it
   refuses to start without ~170k free inodes), and downloads the
   ESMFold2-Fast weights and `ccd.pkl` into
   `/shared/projects/npf_abinitio/hf_cache`, so the GPU nodes run offline.

### Launching on IFB

From the checkout on the login node, inside `tmux`/`screen`. The
controller is the existing `abcfold-npf-pipeline` env (snakemake 9.24, SLURM
plugin 2.7.1, pyarrow); it has no `conda`, so IFB's system one goes on PATH
after it:

```bash
export PATH=/shared/projects/npf_abinitio/conda/envs/abcfold-npf-pipeline/bin:$PATH:/shared/software/miniconda_deployment/bin
export CONDA_CHANNEL_PRIORITY=strict   # as setup.sh, if a missing env gets built
NANOPULLDOWN_RUN_CONFIG=config_<run>.yaml snakemake --profile profiles/ifb
```

- Run overlays merge **after** config.local.yaml, so a value in
  config_<run>.yaml (e.g. `esmfold2.max_tokens: 1200`) overrides the IFB one.
  Put IFB-specific overrides in the overlay itself when they conflict.
- `nanopulldown inputs expand` leaves out pairs over `esmfold2.max_tokens`.
  Expand with the IFB limit (2000) to fold the larger pairs on the H200s.
- IFB results (5 samples, dropout, more loops/steps) are a different series
  from the Mac's 1-sample folds: give them their own `dirs:` in the overlay.
- Progress: `squeue -u $USER`, and `grep "^\[fold_batch\]"` on the chunk logs
  as on the Mac.

## Running on CAIUS (Strasbourg)

The same ensemble setup as IFB (5 runs x 1 sample, 20 loops / 200 steps,
`lm_dropout: 0.3`) also runs on CAIUS, the Université de Strasbourg mésocentre
(`hpc-login.u-strasbg.fr`, account `ibmp`), on the H200 nodes hpc-n977-981
(8 GPUs, 64 cores, 2.3 TB each; partition `publicgpu`, 24 h max, no
preemption there). What differs from IFB:

| Aspect | IFB | CAIUS |
| --- | --- | --- |
| package manager | the shared `abcfold-npf-pipeline` env + system conda | micromamba (`~/.local/bin`), controller env `~/micromamba/envs/nanopulldown` from [envs/controller.yaml](envs/controller.yaml) |
| compute-node internet | yes | **no**: installs run on the login node, and so does `annotate_domains_batch` (web calls) |
| GPU request | `--gpus=h200:1` | untyped gres: `--gpus=1 --constraint=gpuh200` (`hpc.gpu_constraints`) |
| walltime | 3 days | 24 h: fold chunks cut at 23 h estimated (`hpc.fold_chunk_minutes`), by an H200 fit of the IFB timings (`hpc.fold_time_fit`, ~n^3) |
| storage | 500k inodes | 500 GiB home (BeeGFS), no file limit (`diskquota`) |

Setup, once, from a clone on hpc-login (`git clone
https://github.com/EtienneReboul/NanoPulldown.git`): copy the untracked
`configs/pairs.parquet`, `configs/proteins.parquet` and
`configs/inputs_<run>.parquet` from the Mac, then

```bash
nice -n 19 scripts/caius/setup.sh   # micromamba, controller env, config.local.yaml, images, rule envs, weights
```

and add the Mac config.local.yaml's `annotate:` block to the generated
config.local.yaml. Launch inside tmux:

```bash
NANOPULLDOWN_RUN_CONFIG=config_<run>.yaml scripts/caius/run.sh 2>&1 | tee logs_<run>.caius.log
```

## Fold engine: ESMFold2-Fast or ColabFold

`fold.engine: esmfold2` (the default) folds without MSAs, on the Mac or on CUDA.
`fold.engine: colabfold` folds with ColabFold (AlphaFold2-multimer v3 + MMseqs2
MSAs), CUDA only. Both write the same `results/<pair>/raw/` (`sample_<i>.cif.gz`
+ `fold.parquet`), so minimization, PLIP, pyDock, the metrics and the report
are shared ([nanopulldown/fold/colabfold.py](src/nanopulldown/fold/colabfold.py)).
Set the engine in a run overlay **together with the run's own `dirs:`**: a pair
folded by one engine looks finished to the other.

```yaml
# config_<run>_colabfold.yaml
fold: {engine: colabfold}
colabfold:
  num_models: 5          # samples per pair = num_models x num_seeds
  num_seeds: 1
  msa: {mode: server}    # or local (see below)
dirs: {postproc: results_<run>_colabfold, reports: reports_<run>_colabfold, logs: logs_<run>_colabfold}
```

Two rules replace `run_esmfold2`:

- `colabfold_msa` runs ONE batch, a paired + unpaired MSA per complex, cached by content in
  `data/msa/<key>.a3m`. Only complexes missing from the cache are searched, so
  adding preys or rerunning costs nothing for the rest.
  - `msa.mode: server`: `colabfold_batch --msa-only` against ColabFold's public
    MMseqs2 API. Needs internet; meant for small runs, and refuses more than
    `msa.server_max_queries` (500) new complexes. On CAIUS, `scripts/caius/run.sh`
    runs it on the login node.
  - `msa.mode: local`: `colabfold_search` on local ColabFold databases with
    MMseqs2-GPU (`msa.gpu: 1`, databases held in GPU memory with `gpu_server`).
    This is one GPU job sized by `colabfold.msa.slurm` (IFB, CAIUS, any SLURM GPU cluster).
- `run_colabfold_chunk_NNN` runs one `colabfold_batch` over `colabfold.pairs_per_load`
  pairs' MSAs per GPU job, then converts the results. ColabFold's own `.done.txt`
  markers make a timed-out chunk resume at its next pair. The conversion renames
  chains to the spec's ids, rescales pLDDT to 0-1, and reads per-chain-pair ipTM
  from `--calc-extra-ptm`.

One-time cluster setup for ColabFold, after the rule envs exist (`envs/colabfold.yaml`
is the CUDA env, `envs/colabfold_msa.yaml` the portable server-MSA client). Run it as a job:

```bash
sbatch --cpus-per-task=16 --mem=128G --time=24:00:00 --output=logs/colabfold_setup.log \
  scripts/colabfold/setup.sh weights databases check
```

`weights` puts AF2-multimer v3 into `$CF_DATA` (set `colabfold.data_dir`).
`databases` runs ColabFold's `setup_databases.sh` with `GPU=1` into `$CF_DB`
(set `colabfold.msa.db_dir`). That is about 1 TB but few files: PDB templates are
skipped, since their ~200k mmCIFs alone would exceed IFB's inode cap. CAIUS home
(500 GiB) is too small for it. `check` verifies that the env's `mmseqs` has GPU support.
`config.ifb.yaml.example` and `config.caius.yaml.example` carry the cluster
settings. `colabfold.time_fit` (walltime per model) is an unmeasured
AF2-multimer guess with a 3x safety factor; refit it after a first run.

## Fragment mode (domain x domain, as in yeast two-hybrid)

Set `fragments.enabled: true` in a run overlay (see `config_drm2_fragments.yaml`)
and every bait and prey is cut into its annotated domains before pairing:
the screen folds bait domain x prey domain instead of full length x full
length, the way two-hybrid constructs are made. Proteins too long to fold
whole under `esmfold2.max_tokens` become screenable by domain (DRM2's
interactors AGO4/AGO6/TPR2/CYP65 are skipped full length, folded as fragments).

```bash
# preview the cuts (nothing written)
nanopulldown --run-config config_drm2_fragments.yaml inputs fragments Q9M548 Q9ZVD5
#   Q9ZVD5:57-458   402 aa  interpro  Protein argonaute, N-terminal + Argonaute, linker 1 domain + PAZ domain
#   Q9ZVD5:462-924  463 aa  interpro  Argonaute-like, PIWI domain + Piwi domain
nanopulldown --run-config config_drm2_fragments.yaml inputs expand   # or --fragments
NANOPULLDOWN_RUN_CONFIG=config_drm2_fragments.yaml snakemake --use-conda --cores 4 --resources gpu=1 docker_heavy=2
```

- Domains are the ones the domain-map figures show (InterPro representative
  set, `nanopulldown.annotate.merge`). Abutting or overlapping domains are
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

`nanopulldown.inputs.expand` adds two control sets to every run's preys and
pairs them with every bait (config.yaml `controls:`; a run overlay can turn
either off, e.g. `controls: {decoys: {enabled: false}}`). They are tagged in
`prey_sources` of the results table, and the Snakefile's `decoy_calibration`
rule (in the report, "Controls") reads them.

**Positives (`known`): the baits' best-supported known partners.** They
come from UniProt, IntAct, MINT, BioGRID, STRING and Complex Portal
(`nanopulldown.services.interactors`, snapshot
`data/interactors/interactors_<run>.tsv`), plus a hand-curated list. Each
(bait, partner) gets an evidence tier, and the positives are the partners
at `min_tier` or better (default `high`):

| Tier | Rule (config.yaml `controls.positives`) |
|---|---|
| `curated` | listed under `curated:` (`{bait: [{partner, pmids, note}]}`) |
| `high` | direct evidence in ≥ 2 publications (`high_min_publications`), or direct evidence with an MIscore ≥ 0.6 (`high_miscore`), or an X-ray / NMR structure (`structure_methods`) |
| `medium` | direct evidence in 1 publication (`min_direct_publications`) |
| `low` | reported, nothing direct |

Evidence is *direct* when its detection method is two-hybrid (or a
descendant; `direct_methods`), or its PSI-MI type is *direct interaction*
(or a descendant: covalent binding, ubiquitination ...) and it was not seen
by FRET, BRET or BiFC only (`proximity_methods`: BioGRID labels them
"direct interaction", but they only show the proteins are within ~10 nm).
Co-IP and affinity capture-MS (*association*, *physical association*),
Complex Portal co-membership and STRING are never direct.

The MIscore is IntAct's confidence score (Villaveces et al. 2015,
*Database*; [EBI-IntAct/miscore](https://github.com/EBI-IntAct/miscore)):
the mean of a method, an interaction-type and a publication score, 0-1. It
is recomputed over IntAct, MINT and BioGRID together, so BioGRID-only
interactions get one, with each (paper, method) counted once however many
databases curated it. On IntAct records alone the port reproduces IntAct's
own score (`known_intact_miscore`, kept for comparison) to 0.005 for 21 of
24 partners checked; the others differ by IntAct counting two records of
one paper, or an unknown type in IntAct's older ontology. One paper scores
about 0.4; several papers or orthogonal methods push it past 0.6.

STRING is annotation only: its *experiments* channel is BioGRID and IntAct
again (BPM6-ATHB-6: combined 0.508 = experiments 0.508, from the one FRET
paper). Its combined score and channels (`known_string_escore`, `_dscore`,
`_tscore`: experiments, curated databases, text-mining) are carried as
columns. UniProt's interaction comments give no method, type or PubMed id,
so they only count towards `known_n_databases`.

BPM6 at the default `high`: CUL3A (curated) only. RAP2-4 (one pull-down),
BPM1 and BPM3 (one two-hybrid paper) are `medium`; ATHB-6 (one FRET paper)
is `low`. `min_tier: medium` restores roughly the former direct-only rule,
and `min_tier: low` takes every reported partner, as the former
`interactors:` input source did (`config_drm2_ppi.yaml`; the old
`direct_only: true/false` still reads as medium/low). Positives are cut like
any prey in fragment mode. A screen prey that is a known partner gets the
`known_*` columns either way.

```bash
nanopulldown --run-config config_<run>.yaml inputs positives   # every partner with tier, MIscore, STRING channels
nanopulldown inputs interactors A1L4W5 --min-tier medium --refresh   # --refresh: re-fetch (adds IntAct MIscore + STRING channels to an older snapshot)
```

**Negatives (`decoy`): a matched decoy set** (`nanopulldown.inputs.decoys`).
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
nanopulldown --run-config config_<run>.yaml inputs expand --dry-run
nanopulldown --run-config config_<run>.yaml inputs positives          # the positives only
nanopulldown --run-config config_<run>.yaml inputs interactors Q9M548 --direct

# expand + fold as usual: the controls are paired with every bait
nanopulldown --run-config config_<run>.yaml inputs expand
#    -> configs/decoys_<run>.txt + data/decoys/decoys_<run>.parquet (+ .json)
NANOPULLDOWN_RUN_CONFIG=config_<run>.yaml snakemake --use-conda --cores 4 --resources gpu=1 docker_heavy=2
#    -> reports_<run>/folding/decoy_calibration/: per bait and metric, conformal
#       p-values against the decoys, BH q-values, hits; recall of the positives
```

- **Decoys are drawn once per run.** They are reused while
  `controls.decoys` and the cutting rules are unchanged. Adding preys to a
  run therefore never re-draws, and re-folds, its decoys. A warning says
  when the preys have changed since the draw.
  `nanopulldown inputs expand --rebuild-decoys` (or `nanopulldown inputs decoys
  [--n N --seed S]`) draws again.
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
| `configs/baits.txt`, `configs/preys.txt` | one protein spec per line; a `list:` input source (or the default lists) for`nanopulldown.inputs.expand` |
| `configs/pairs.parquet`                    | every expanded pair's chain spec, one row per pair (generated; schema in `configs/_schema.md`)   |
| `configs/proteins.parquet`                 | UniProt names / gene / AGI locus per accession, and the display names reports use (generated)   |
| `configs/inputs_<run>.parquet`             | which input source picked each protein, plus IP-MS columns (generated)                           |
| `configs/protein_names.yaml`               | hand overrides of display names, by accession (tracked)                                          |
| `configs/curated_domains.yaml`             | hand-curated domain maps, by pair (tracked)                                                      |
| `configs/fragment_domains.yaml`            | fragment mode: hand-picked domain fragments, by accession (tracked)                              |
| `configs/fragments_<run>.parquet`          | fragment mode: every bait/prey fragment a run pairs (generated)                                  |
| `envs/`                                    | `analysis` (annotation, metrics, pliparser, figures, tables), `esmfold2` (MLX) / `esmfold2_cuda` (IFB), `minimize`; built by `--use-conda` |
| `src/nanopulldown/`                        | the package + `nanopulldown` CLI every rule calls (see "Code layout" below)                       |
| `scripts/legacy/`                          | one-off migrations and comparisons, not part of the pipeline                                     |
| `cookiecutter/`                            | project template: `cookiecutter cookiecutter/` starts a new screen                               |
| `Snakefile`                                | the whole pipeline, one default`rule all`                                                      |
| `report/`                                  | `custom.css` (dark theme, from ab_initio_pipeline), `datavzrd/nanopulldown.datavzrd.yaml`    |
| `data/ results/ logs/ reports/`            | runtime outputs (gitignored)                                                                     |

## Code layout

Everything the pipeline runs is one package, `src/nanopulldown/`, behind one
CLI (`nanopulldown --help` lists every command; `nanopulldown <cmd> --help`
its options). The Snakefile calls `python -m nanopulldown <cmd>` with `src/`
on `PYTHONPATH`, so the rule envs need no install. Modules only import
downward, and each layer adds as few dependencies as it can:

| Module         | What                                                                        | Deps (beyond stdlib)          |
| -------------- | --------------------------------------------------------------------------- | ----------------------------- |
| `cli`          | the dispatcher; imports a command's module only when it runs                | none                          |
| `core`         | config, pair specs, protein names, fold table, cache, small io helpers      | pyyaml (+ pyarrow, lazily)    |
| `services`     | UniProt, interactor databases, polite rate-limited HTTP                     | requests                      |
| `annotate`     | domains (InterPro/ScanProsite), disorder (AIUPred docker), per-pair merge   | requests                      |
| `inputs`       | `inputs:` resolution, fragments, decoys, positives -> `configs/pairs.parquet` | requests, pandas, numpy     |
| `fold`         | ESMFold2-Fast (MLX / CUDA), or ColabFold MSAs + folds converted to the same table | esm, mlx / colabfold, gemmi |
| `structure`    | OpenMM minimization, clashes, interface metrics (+ vendored ipSAE, Pinc), pyDock energy | openmm / gemmi, scipy, numpy / lightdock |
| `plip`         | pliparser's cxc config, per-pair contact table (the PLIP image run and the pliparser calls stay shell in the Snakefile) | none / pandas |
| `report`       | figures (`report.plots`), tables, decoy calibration, datavzrd + report zip  | matplotlib, plotly, pandas    |

```bash
nanopulldown --run-config config_<run>.yaml inputs expand --dry-run
nanopulldown plot domain-heatmap --pair <pair> --fold results/<pair>/raw/fold.parquet \
  --annotation data/annotation/<pair>/annotation.yaml --out /tmp/heatmap.svg
```

The package moved out of the old flat `scripts/` with this release, which
changes every rule's shell command and conda env. On an existing run, pass
`--rerun-triggers mtime params input` once so Snakemake does not re-run
finished jobs (folding included) only because their command text changed.

## Worked example

`example_toy__example_toy` (a ubiquitin homodimer smoke test, same protein
ab_initio_pipeline uses for its own worked example) is kept in
`configs/curated_domains.yaml` as a template for a curated entry; its spec is in
`configs/pairs.parquet` (see `configs/_schema.md`).

## Implementation notes worth knowing before relying on this in production

- **ESMFold2's MLX path**: this pipeline assumes the Apple Silicon MLX model
  populates `result.pae` / `result.pair_chains_iptm` exactly like the
  Torch/CUDA path does — confirm once against a live run.
- **AIUPred docker CLI**: `nanopulldown.annotate.disorder` assumes a
  `-i/-t/-o` CLI contract; confirm with
  `docker run --rm ghcr.io/doszilab/aiupred:cpu --help` and adjust if it
  differs.
- **InterProScan v6 MATCH API**: `nanopulldown.annotate.domains`'s MD5-cache
  lookup endpoint/schema should be confirmed against current EBI docs.
- **OpenMM OpenCL platform**: `nanopulldown.structure.minimize` prefers OpenCL
  (Mac Metal-backed GPU accel) with a CPU fallback — confirm OpenCL actually
  initializes on the target Mac.
- **PLIP docker image on Apple Silicon**: `docker.io/pharmai/plip` has no
  arm64 build, so `plip.docker_platform` defaults to `linux/amd64` (x86
  emulation via Docker Desktop's Rosetta) — slower than a native image, and
  needs "Use Rosetta for x86/amd64 emulation" enabled in Docker Desktop.
