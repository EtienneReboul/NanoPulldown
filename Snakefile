"""
Snakefile — nanopulldown (local, single-machine pipeline)
============================================================
One straight-through pipeline, no manual checkpoint in the middle: for
every pair in config.yaml's `pairs:` list, sequence annotation feeds
straight into folding.

  1. preprocessing (needs internet)
    Sequence annotation. Pairs themselves (bait/prey UniProt sequences
    resolved, chains assigned) come from `nanopulldown inputs expand`, run once
    ahead of Snakemake, into configs/pairs.parquet -- see the README and
    configs/_schema.md. This stage only annotates their already-resolved
    sequences.
      annotate_domains_batch / annotate_disorder_batch
                         once per UNIQUE sequence into data/annotation/_cache/
                         (throttled InterPro API + ScanProsite; one AIUPred container run per 200 seqs)
      merge_annotation   per pair, straight from the cache -> data/annotation/<pair>/annotation.yaml -- a
                          proposed `domains:` block; hand-curate it into
                          configs/curated_domains.yaml whenever convenient
                          (see configs/_schema.md), it isn't required to proceed
      annotate_cdd       CDD domains + their conserved-feature sites (InterPro API
                         `?residues`, cached per sequence) of every chain ->
                         data/annotation/cdd_sites.parquet, read against PLIP
                         contacts by the results table and the CDD figures
      report figures (domain map + AIUPred) -- one per distinct protein,
      filed under Baits / Preys in the one report.zip (built after folding,
      below).

  2. folding (Apple Silicon GPU via MLX/Metal, or CUDA on the IFB cluster)
    fold.engine: esmfold2 (default) -- ESMFold2-Fast folding (esmfold2.num_seeds
    runs of esmfold2.num_diffusion_samples diffusion samples, no MSA/templates);
    or fold.engine: colabfold -- ColabFold / AlphaFold2-multimer with MSAs
    (CUDA only) -> OpenMM minimization -> PAE-based rescoring
    (ipSAE/iLIS/Pinc) -> ONE report. Structures are stored gzipped
    (.cif.gz / .pdb.gz -- ChimeraX and gemmi open them as-is).
      run_esmfold2       one seed, N diffusion samples -> raw/sample_<i>.cif.gz +
                         raw/fold.parquet (scores, pae/plddt float16, topology;
                         schema in nanopulldown/core/fold_table.py)
      -- or, fold.engine: colabfold --
      colabfold_msa      one batch: a paired+unpaired MSA per complex, cached by
                         content in colabfold.msa.cache_dir -- ColabFold's MMseqs2
                         web server (small runs) or local MMseqs2-GPU search
                         (colabfold_search --gpu 1, a GPU job on IFB / CAIUS)
      run_colabfold_chunk_NNN  colabfold_batch over a chunk's MSAs, converted to
                         the same raw/sample_<i>.cif.gz + raw/fold.parquet
                         (num_models x num_seeds samples per pair)
      compute_metrics    ipSAE / iLIS / Pinc + ESMFold2's own pair ipTM ->
                         results/<pair>/interface_metrics.parquet
      minimize_cif       OpenMM energy minimization -> minimized/sample_<i>.pdb.gz
                         + sample_<i>_energy.parquet, no ChimeraX / antechamber
      run_plip           PLIP (docker / apptainer) on each minimized sample, bait_chains
                         vs prey_chains -> per-sample TXT report (the
                         decompressed input and PLIP's protonated copy are
                         deleted afterwards)
      plip_parse         pliparser -> per-sample ChimeraX .cxc (+ temporary contact CSVs)
      aggregate_plip     -> results/<pair>/plip_contacts.parquet (all samples)
      pydock_energy      pyDock bait-prey energy (LightDock's cpydock) of each
                         minimized sample -> results/<pair>/pydock_energy.parquet;
                         the results table combines it with 0.8 ipTM + 0.2 pTM
                         into z-scores (Rodriguez-Lumbreras et al 2026, Eq. 1)
      + report figures (domain x domain confidence heatmaps, minimize energy,
        PLIP contact heatmaps, PLIP contacts on CDD sites per pair and per
        bait) and reports/report.zip (onsuccess hook; unzip -> report.html)
        -- the ONLY report this pipeline produces.

Every step is a `nanopulldown` CLI command (src/nanopulldown/, run as
`python -m nanopulldown` with src/ on PYTHONPATH, so no per-env install);
`nanopulldown --help` lists them, and each runs by hand the same way.

Usage (one command, both stages):
  snakemake --use-conda --cores 4
On the IFB SLURM cluster (CUDA folding, same conda envs; README "Running on
the IFB cluster"):
  snakemake --profile profiles/ifb
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(workflow.basedir)
sys.path.insert(0, str(REPO / "src"))
from nanopulldown.core.config import load_config  # noqa: E402
from nanopulldown.core.containers import run_command  # noqa: E402
from nanopulldown.core.fold_table import engine_samples, fold_engine, fold_max_tokens  # noqa: E402
from nanopulldown.core.metric_refs import REPORT_METRICS, caption_rst  # noqa: E402
from nanopulldown.core.proteins import label_digest, load_proteins, pair_labels, protein_name  # noqa: E402
from nanopulldown.core.specs import (inputs_table_path, load_pairs, pair_tokens,  # noqa: E402
                                     protein_chains, spec_digest)

# Jobs (in any --use-conda env) import the package from the checkout itself.
# PYTHONPATH goes IN the command: Snakemake deletes it from a conda job's
# environment (conda_block_conflicting_envvars), so os.environ would not reach it.
NP = f'PYTHONPATH="{REPO / "src"}" python -m nanopulldown'

configfile: str(REPO / "config.yaml")
if (REPO / "config.local.yaml").exists():
    configfile: str(REPO / "config.local.yaml")

report: "report/folding.rst"

CFG = load_config(REPO)
DIRS = CFG["dirs"]
EF = CFG["esmfold2"]
# fold.engine: esmfold2 | colabfold -- which of the two fold rule sets below
# exists. Give a run per engine its own `dirs:` (run overlay): both write
# results/<pair>/raw/, and a finished fold of one engine looks finished to the other.
ENGINE = fold_engine(CFG)
CF = CFG.get("colabfold") or {}
MAX_TOKENS = fold_max_tokens(CFG)

# Software: every rule env is a conda env (envs/*.yaml, --use-conda) on the
# Mac AND on the IFB cluster; only the fold env differs by machine (MLX vs CUDA).
# Third-party tool images (PLIP, AIUPred) run via execution.container_runtime.
FOLD_BACKEND = EF.get("backend", "mlx")
ESMFOLD_ENV = "esmfold2_cuda" if FOLD_BACKEND == "torch" else "esmfold2"

# Specs come from configs/pairs.parquet (+ configs/curated_domains.yaml). A
# pair is dropped (with a warning) when it isn't in the table -- inputs expand
# does not add pairs over esmfold2.max_tokens -- or when it is over max_tokens
# anyway (specs expanded before the limit existed): ESMFold2 memory grows
# ~quadratically with tokens and an over-limit fold aborts the whole process
# on the Metal buffer cap. 0 = no limit.
_ALL_SPECS = load_pairs(REPO)
# `pairs_file:` (written by `nanopulldown inputs expand` from the config's
# `inputs:`) takes precedence over an inline `pairs:` list.
if CFG.get("pairs_file"):
    _PAIR_LIST = [ln.split("#", 1)[0].strip() for ln in (REPO / CFG["pairs_file"]).read_text().splitlines()]
    _PAIR_LIST = [p for p in _PAIR_LIST if p]
else:
    _PAIR_LIST = CFG.get("pairs") or []
PAIRS, SPECS, _no_spec, _too_long = [], {}, [], []
for _p in _PAIR_LIST:
    if _p not in _ALL_SPECS:
        _no_spec.append(_p)
        continue
    _spec = _ALL_SPECS[_p]
    if MAX_TOKENS and pair_tokens(_spec) > MAX_TOKENS:
        _too_long.append(_p)
        continue
    PAIRS.append(_p)
    SPECS[_p] = _spec
if _no_spec:
    print(f"[Snakefile] WARNING: {len(_no_spec)} pair(s) in config.yaml are not in configs/pairs.parquet "
          f"(not expanded, or skipped as too long) and are ignored, e.g. {_no_spec[:3]}", file=sys.stderr)
if _too_long:
    print(f"[Snakefile] {len(_too_long)} pair(s) exceed {ENGINE}.max_tokens={MAX_TOKENS} and are ignored",
          file=sys.stderr)

# Rules don't take a spec FILE as input any more (one shared table would make
# every job look outdated whenever a pair is added). Instead they carry content
# digests as params, so Snakemake's params rerun-trigger reruns exactly the
# jobs whose pair changed: SPEC_KEY = chains/sequences/roles (what folding and
# annotation see), CUR_KEY = curated domains (what the figures also use).
SPEC_KEY = {p: spec_digest(SPECS[p]) for p in PAIRS}
CUR_KEY = {p: spec_digest(SPECS[p], "curation") for p in PAIRS}

# Display names (configs/proteins.parquet + configs/protein_names.yaml): paths
# stay accession-keyed, names only label figures/tables. LABEL_KEY reruns a
# pair's figures when one of its proteins is renamed.
PROTEINS = load_proteins(REPO)
LABELS = {p: pair_labels(SPECS[p], PROTEINS) for p in PAIRS}
LABEL_KEY = {p: label_digest(SPECS[p], PROTEINS) for p in PAIRS}
INPUTS_TABLE = inputs_table_path(CFG, REPO)

# Report sections are split per bait PROTEIN (all its fragments together):
# per-pair figures get the bait as their subcategory, run-level pages are
# built once per bait from that bait's rows (<reports>/folding/by_bait/<acc>/).
BAITS = sorted({SPECS[p]["bait"]["uniprot"] for p in PAIRS})
PAIRS_OF = {b: [p for p in PAIRS if SPECS[p]["bait"]["uniprot"] == b] for b in BAITS}
BAIT_LABEL = {b: protein_name(b, PROTEINS) if b in protein_name(b, PROTEINS) else f"{protein_name(b, PROTEINS)} ({b})"
              for b in BAITS}


def bait_subcategory(wc):
    return BAIT_LABEL[SPECS[wc.pair]["bait"]["uniprot"]]


wildcard_constraints:
    pair="[^/]+",
    bait="[^/]+",

# Controls (nanopulldown/inputs/controls.py, added by inputs expand): decoy_calibration
# runs only when this run's preys include decoys -- a run expanded before
# the controls existed (or with controls.decoys off) builds as before.
_CTL = CFG.get("controls") or {}
DECOY_TAG = (_CTL.get("decoys") or {}).get("name", "decoy")
POSITIVE_TAG = (_CTL.get("positives") or {}).get("name", "known")


def _run_has_decoys() -> bool:
    if not INPUTS_TABLE.exists():
        return False
    import pyarrow.parquet as pq
    t = pq.read_table(INPUTS_TABLE, columns=["role", "sources"]).to_pydict()
    return any(r == "prey" and DECOY_TAG in (s or []) for r, s in zip(t["role"], t["sources"]))


HAS_DECOYS = _run_has_decoys()


def pair_report_labels(wc):
    # bait + prey only: more columns crowd the report's figure lists out of view
    lb = LABELS[wc.pair]
    return {"bait": lb["bait"], "prey": lb["prey"]}

# ── stage 1: preprocessing ───────────────────────────────────────────────
ANN = Path(DIRS["annotation"])
PRE_REP = Path(DIRS["reports"]) / "preprocessing"
PRE_LOG = Path(DIRS["logs"]) / "preprocessing"

# ── stage 2: folding ─────────────────────────────────────────────────────
POST = Path(DIRS["postproc"])
FOLD_REP = Path(DIRS["reports"]) / "folding"
FOLD_LOG = Path(DIRS["logs"]) / "folding"
MET = CFG["metrics"]
MZ = CFG["minimize"]
CL = MZ["clashes"]
PLIP = CFG["plip"]
CXC = PLIP["cxc"]
RPT = CFG["report"]
# figure rules carry the theme as a param, so switching it redraws them
FIG_THEME = f"{RPT.get('figure_theme', 'dark')}/{bool(RPT.get('figure_transparent', False))}"
DOCKER_HEAVY_POOL = int(CFG["scheduling"]["docker_heavy_pool"])
DATAVZRD = bool(RPT.get("datavzrd", False))
BY_BAIT = FOLD_REP / "by_bait"
BAIT_PAGES = (["metric_violins", "plip_type_violins", "clash_test", "cdd_contact_map.svg"]
              + (["decoy_calibration"] if HAS_DECOYS else []) + (["tables"] if DATAVZRD else []))
FMTS = ",".join(RPT["figure_formats"])
N_SAMPLES = engine_samples(CFG)   # esmfold2: num_seeds x num_diffusion_samples; colabfold: num_seeds x num_models
SAMPLE_IDX = list(range(N_SAMPLES))


# Domain map + AIUPred figures are per distinct PROTEIN, not per pair x chain:
# the bait is in every pair (and a homo-oligomer repeats its chain), so a
# per-chain figure repeated the same bait plot hundreds of times. A protein is
# (role, sequence, curated domains) -- the same UniProt with other regions or
# another pair's curation gets its own figure. Each is drawn from the first
# pair it occurs in; the report files them under Baits / Preys.
def _domain_proteins():
    prots, seen = {}, {}
    for p in PAIRS:
        reviewed = SPECS[p].get("annotation_reviewed", False)
        for c in protein_chains(SPECS[p]):
            cur = (SPECS[p].get("domains") or {}).get(c["id"]) if reviewed else None
            key = (c["role"], c["sequence"], json.dumps(cur, sort_keys=True))
            if key in seen:
                prots[seen[key]]["pairs"].append(p)
                continue
            slug = c.get("name") or f"{c['role']}_{p}_{c['id']}"
            if slug in prots:
                slug = f"{slug}_{hashlib.sha1(repr(key).encode()).hexdigest()[:6]}"
            seen[key] = slug
            acc = SPECS[p][c["role"]]["uniprot"]
            prots[slug] = {"role": c["role"], "pair": p, "chain": c["id"], "pairs": [p], "uniprot": acc}
    return prots


DOMAIN_PROTEINS = _domain_proteins()


rule all:
    default_target: True
    input:
        [str(ANN / p / "annotation.yaml") for p in PAIRS],
        [str(PRE_REP / "domain_bar" / f"{prot}.svg") for prot in DOMAIN_PROTEINS],
        [str(POST / p / "interface_metrics.parquet") for p in PAIRS],
        [str(FOLD_REP / p / "domain_heatmap.svg") for p in PAIRS],
        [str(FOLD_REP / p / "minimize_energy.svg") for p in PAIRS],
        [str(POST / p / "plip" / f"sample_{i}_report" / f"sample_{i}.cxc")
         for p in PAIRS for i in SAMPLE_IDX],
        str(FOLD_REP / "clash_test_stats.csv"),
        str(FOLD_REP / "clash_counts.csv"),
        [str(FOLD_REP / p / "plip_contacts.svg") for p in PAIRS],
        [str(FOLD_REP / p / "plip_cdd.svg") for p in PAIRS],
        # MSA-based engines only (ESMFold2 has no MSA)
        [str(FOLD_REP / "msa_unpaired" / f"{prot}.svg") for prot in DOMAIN_PROTEINS] if ENGINE == "colabfold" else [],
        [str(FOLD_REP / p / "msa_paired.svg") for p in PAIRS] if ENGINE == "colabfold" else [],
        str(FOLD_REP / "results_table.csv"),
        str(FOLD_REP / "minimize_failure_rate.csv"),
        # per-bait report sections (the run-wide pages above are on-demand targets)
        [str(BY_BAIT / b / page) for b in BAITS for page in BAIT_PAGES],


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — preprocessing
# ═══════════════════════════════════════════════════════════════════════════

ANN_CACHE = ANN / "_cache"
ALL_PAIRS = " ".join(PAIRS)
ALL_SPECS_KEY = hashlib.sha1(" ".join(f"{p}:{SPEC_KEY[p]}" for p in PAIRS).encode()).hexdigest()[:12]


# Annotation is per UNIQUE SEQUENCE, not per pair: the bait is in every pair,
# so per-pair jobs would re-query the web services / re-run the AIUPred
# container ~2x per pair for work that is identical. These two batch rules fill
# a per-sequence cache once (web calls in a small thread pool; AIUPred in a few
# multi-FASTA container runs); the per-pair rules below just assemble from it.
# They run concurrently with each other and, via `priority`, ahead of folding.
rule annotate_domains_batch:
    output:
        done=str(ANN_CACHE / "domains.done"),
    log:
        str(PRE_LOG / "annotate_domains_batch.log"),
    conda:
        "envs/analysis.yaml"
    priority: 50
    params:
        pairs=ALL_PAIRS, specs=ALL_SPECS_KEY,
    shell:
        r"""
        {NP} annotate domains --pairs {params.pairs} --done {output.done} > {log} 2>&1
        """


rule annotate_disorder_batch:
    output:
        done=str(ANN_CACHE / "disorder.done"),
    log:
        str(PRE_LOG / "annotate_disorder_batch.log"),
    conda:
        "envs/analysis.yaml"
    priority: 50
    params:
        pairs=ALL_PAIRS, specs=ALL_SPECS_KEY,
    resources:
        # docker_heavy=1: never overlaps a run_esmfold2 model load (that rule
        # claims the whole pool). CPU-only -- no gpu= claim, the GPU stays free.
        docker_heavy=1,
    shell:
        r"""
        {NP} annotate disorder --pairs {params.pairs} --done {output.done} > {log} 2>&1
        """


# Per pair, straight from the per-sequence cache the batch rules filled (with
# a live lookup for any sequence they missed).
rule merge_annotation:
    input:
        domains=str(ANN_CACHE / "domains.done"),
        disorder=str(ANN_CACHE / "disorder.done"),
    output:
        yaml=str(ANN / "{pair}" / "annotation.yaml"),
    log:
        str(PRE_LOG / "merge_annotation" / "{pair}.log"),
    conda:
        "envs/analysis.yaml"
    params:
        spec=lambda wc: SPEC_KEY[wc.pair],
        curation=lambda wc: CUR_KEY[wc.pair],
    shell:
        r"""
        {NP} annotate merge --pair {wildcards.pair} --out {output.yaml} > {log} 2>&1
        """


# CDD sites are fetched here, not in annotate_domains_batch: that would rewrite
# domains.done and so rerun every merge_annotation and the figures under it.
rule annotate_cdd:
    input:
        domains=str(ANN_CACHE / "domains.done"),
    output:
        str(ANN / "cdd_sites.parquet"),
    log:
        str(PRE_LOG / "annotate_cdd.log"),
    conda:
        "envs/analysis.yaml"
    priority: 50
    params:
        pairs=ALL_PAIRS, specs=ALL_SPECS_KEY,
    shell:
        r"""
        {NP} annotate cdd --pairs {params.pairs} --out {output[0]} > {log} 2>&1
        """


rule fig_domain_bar:
    input:
        annotation=lambda wc: str(ANN / DOMAIN_PROTEINS[wc.protein]["pair"] / "annotation.yaml"),
    output:
        report(str(PRE_REP / "domain_bar" / "{protein}.svg"),
               category="Domain map + AIUPred",
               subcategory=lambda wc: "Baits" if DOMAIN_PROTEINS[wc.protein]["role"] == "bait" else "Preys",
               labels=lambda wc: {
                   "protein": protein_name(DOMAIN_PROTEINS[wc.protein]["uniprot"], PROTEINS),
                   "name": protein_name(DOMAIN_PROTEINS[wc.protein]["uniprot"], PROTEINS, "long"),
                   "UniProt": DOMAIN_PROTEINS[wc.protein]["uniprot"],
                   "pairs": str(len(DOMAIN_PROTEINS[wc.protein]["pairs"]))}),
    conda:
        "envs/analysis.yaml"
    params:
        theme=FIG_THEME,
        fmts=",".join(CFG["report"]["figure_formats"]),
        cutoff=CFG["annotate"]["consensus"]["disorder_cutoff"],
        pair=lambda wc: DOMAIN_PROTEINS[wc.protein]["pair"],
        chain=lambda wc: DOMAIN_PROTEINS[wc.protein]["chain"],
        spec=lambda wc: SPEC_KEY[DOMAIN_PROTEINS[wc.protein]["pair"]],
        curation=lambda wc: CUR_KEY[DOMAIN_PROTEINS[wc.protein]["pair"]],
        labels=lambda wc: LABEL_KEY[DOMAIN_PROTEINS[wc.protein]["pair"]],
    shell:
        r"""
        {NP} plot domain-bar \
          --pair {params.pair} --annotation {input.annotation} --chain {params.chain} \
          --cutoff {params.cutoff} \
          --out {output[0]} --formats {params.fmts}
        """


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — folding
# ═══════════════════════════════════════════════════════════════════════════

# All N samples are declared as outputs (N is static: esmfold2.num_seeds x
# num_diffusion_samples) so minimize_cif's per-sample input can wire
# back to this rule. fold.parquet is written last, so it also marks the pair
# as complete.
# Fold granularity. pairs_per_load = 0: one process (and one ~25 GB model load,
# with its swap spike) per pair. > 0: pairs are split into chunks of that size
# and each chunk is ONE job that loads the model once and folds its pairs in a
# loop (`nanopulldown fold --pairs ...`). A chunk's downstream steps (minimize,
# PLIP, ...) only start once the whole chunk is done, so keep chunks to ~1-2 h.
PAIRS_PER_LOAD = int((CF if ENGINE == "colabfold" else EF).get("pairs_per_load", 0))

HPC = CFG.get("hpc") or {}
# Shell-level globals, not params: a new param would mark every finished fold
# outdated under the README's `--rerun-triggers mtime params input`.
# Pre-downloaded weights (esmfold2.hf_home): the GPU nodes need no internet.
# (ColabFold: none of these -- its weights come from colabfold.data_dir.)
FOLD_HF_ENV = f'HF_HOME="{EF["hf_home"]}" HF_HUB_OFFLINE=1 ' if EF.get("hf_home") and ENGINE == "esmfold2" else ""
# esmfold2.conda_lib_first (colabfold.conda_lib_first): the env's lib/ ahead on LD_LIBRARY_PATH. The torch
# wheel loads libstdc++ before numpy and, lacking an RPATH to the env, gets the
# system one -- too old on RHEL 8 (CAIUS: numpy needs GLIBCXX_3.4.29).
# CUDA: expandable segments against allocator fragmentation -- at the H200
# limit a chunk's 3rd pair (2,798 tokens) hit OOM with 27 GiB reserved but
# unallocated, left over from its first two pairs (CAIUS 2026-10-22).
if FOLD_BACKEND == "torch" and ENGINE == "esmfold2":
    FOLD_HF_ENV += "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
if (CF if ENGINE == "colabfold" else EF).get("conda_lib_first"):
    FOLD_HF_ENV += ('LD_LIBRARY_PATH="$(python -c \'import sys; print(sys.prefix)\')/lib'
                    '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ')


# Per-fold time fit at the Mac reference settings (10 loops / 100 steps / 1
# sample): t(n) = base_s + coef_s * (n/600)^exponent. Default: the Mac fit
# (5,768 folds). hpc.fold_time_fit overrides it per machine -- the CUDA kernels
# scale more steeply (H200: ~n^3, config.caius.yaml.example).
# ColabFold: colabfold.time_fit is the time of ONE model at the configured
# recycles, same form; its default is a rough AF2-multimer guess (unmeasured
# here) -- refit it per machine after a first run.
if ENGINE == "colabfold":
    FOLD_FIT = {"base_s": 10.0, "coef_s": 40.0, "exponent": 2.0, **(CF.get("time_fit") or {})}
else:
    FOLD_FIT = {"base_s": 2.6, "coef_s": 46.8, "exponent": 2.16, **(HPC.get("fold_time_fit") or {})}


def fold_minutes(pairs):
    """Walltime estimate (minutes) of one fold job holding `pairs`: model load
    (hpc.fold_load_minutes) + the FOLD_FIT per-fold time, scaled ~linearly by
    loops (trunk, ~90%) and steps (diffusion) and by the number of runs (ColabFold:
    by models x seeds), times hpc.fold_runtime_safety (ColabFold: colabfold.runtime_safety)."""
    if ENGINE == "colabfold":
        scale = N_SAMPLES
    else:
        scale = (0.9 * (int(EF["num_loops"]) + 1) / 11 + 0.1 * int(EF["num_sampling_steps"]) / 100) \
            * int(EF.get("num_seeds", 1)) * (1 + 0.1 * (int(EF["num_diffusion_samples"]) - 1))
    secs = sum(FOLD_FIT["base_s"] + FOLD_FIT["coef_s"] * (pair_tokens(SPECS[p]) / 600) ** FOLD_FIT["exponent"]
               for p in pairs) * scale
    # ColabFold's time fit is a guess, not a cluster's measured fit: its own safety factor
    safety = CF.get("runtime_safety", 3.0) if ENGINE == "colabfold" else HPC.get("fold_runtime_safety", 3.0)
    return float(HPC.get("fold_load_minutes", 10)) + secs / 60 * float(safety)


# hpc.gpu_request: "gpus" (default) -> the plugin's --gpus (gpu=1 + gpu_model);
# "gres" -> --gres=gpu:1, for SLURM without select/cons_tres (CAIUS rejects
# --gpus). The plugin adds --gpus for any gpu > 0, so gres mode claims gpu=0 --
# the Mac's single-model guard (gpu=1) is unaffected, it never sets this.
GPU_GRES = HPC.get("gpu_request", "gpus") == "gres"
FOLD_GPU_CLAIM = 0 if GPU_GRES else 1


def fold_resources(pairs):
    """SLURM resources of one fold job holding `pairs` (snakemake-executor-plugin-slurm
    keys; the local executor ignores them). gpu=1 + gpu_model -> --gpus=<model>:1;
    on clusters with untyped GPU gres (CAIUS: gpu:8), hpc.gpu_constraints maps the
    model to a node feature instead (h200 -> --constraint=gpuh200, --gpus=1).
    Walltime: fold_minutes(), capped at hpc.fold_runtime_max (the partition's limit).
    ColabFold: colabfold.gpu_classes, when set, replaces hpc.gpu_classes (AF2's
    memory per token is not ESMFold2's)."""
    n = [pair_tokens(SPECS[p]) for p in pairs]
    classes = (ENGINE == "colabfold" and CF.get("gpu_classes")) or HPC.get("gpu_classes") or [{}]
    cls = next((c for c in classes if not c.get("max_tokens") or max(n) <= c["max_tokens"]), classes[-1])
    return gpu_job_resources(cls, min(float(HPC.get("fold_runtime_max", 4320)),
                                      max(float(HPC.get("fold_runtime_min", 60)), fold_minutes(pairs))))


def gpu_job_resources(cls, minutes):
    """One GPU job's SLURM resources from a gpu class ({gpu_model, mem_mb, cpus,
    partition, constraint}) and a walltime -- shared by the fold jobs and
    colabfold_msa. A class's own `constraint` (a SLURM feature expression, e.g.
    "gpuv100|gpua100") replaces gpu_model: any of several GPU types."""
    res = dict(
        gpu_model=cls.get("gpu_model", ""),
        slurm_partition=cls.get("partition") or HPC.get("gpu_partition", "gpu"),
        mem_mb=int(cls.get("mem_mb", 64000)),
        cpus_per_task=int(cls.get("cpus", 8)),
        runtime=int(minutes),
    )
    if GPU_GRES:
        res["gres"] = "gpu:1"
    feature = cls.get("constraint") or (HPC.get("gpu_constraints") or {}).get(res["gpu_model"])
    if feature:
        res["gpu_model"], res["constraint"] = "", feature
    if HPC.get("exclude_nodes"):
        res["slurm_extra"] = f"--exclude={HPC['exclude_nodes']}"
    return res


# Fold chunks (pairs_per_load > 0, and always for ColabFold).
# Already-folded and still-to-fold pairs are chunked separately (list order
# otherwise), so pairs added to a run -- a raised max_tokens, new preys --
# never share a chunk with finished ones: a chunk reruns as a whole, and
# rerunning one that holds finished pairs relinks them and so reruns
# everything downstream of them (minimize, PLIP, ...). With
# `--rerun-triggers mtime`, a chunk whose outputs all exist is not rerun even
# though its membership changed.
# hpc.fold_chunk_minutes (> 0) also closes a chunk before its fold_minutes()
# estimate passes that budget -- under a short partition walltime (CAIUS
# publicgpu: 24 h) the largest pairs then go a few per job.
def _chunks(pairs, size):
    budget = float(HPC.get("fold_chunk_minutes", 0))
    out = []
    for p in pairs:
        if out and len(out[-1]) < size and not (budget and fold_minutes(out[-1] + [p]) > budget):
            out[-1].append(p)
        else:
            out.append([p])
    return out


# The chunk plan is fixed by the controller. On a cluster, each job re-parses
# this Snakefile when it starts; by then other chunks may have finished,
# which moves their pairs from _todo to _folded and renumbers the chunks --
# chunk_K would fold another chunk's pairs (or none) and the controller then
# misses chunk_K's outputs. So the controller writes its plan and jobs
# (remote mode) read it back. Dry runs never write it: one run while a
# controller is active would swap the plan under its queued jobs.
def fold_plan(size, plan_name):
    plan_file = FOLD_LOG / plan_name
    dry = workflow.dryrun or any(a in ("--dry-run", "--dryrun") or (a.startswith("-") and not a.startswith("--") and "n" in a)
                                 for a in sys.argv[1:])
    if workflow.remote_exec and plan_file.exists():
        return json.loads(plan_file.read_text())
    folded = [p for p in PAIRS if (POST / p / "raw" / "fold.parquet").exists()]
    # still-to-fold pairs chunked by size: a fold job's GPU class (fold_resources)
    # follows its largest pair, so mixing sizes would send small pairs to the
    # few big GPUs; on the Mac it just evens out chunk durations
    todo = sorted((p for p in PAIRS if p not in set(folded)), key=lambda p: pair_tokens(SPECS[p]))
    plan = _chunks(folded, size) + _chunks(todo, size)
    if not workflow.remote_exec and not dry:
        plan_file.parent.mkdir(parents=True, exist_ok=True)
        plan_file.write_text(json.dumps(plan))
    return plan


if ENGINE == "colabfold":
    # ── ColabFold: MSAs (one batch), then folds (chunks) ────────────────────
    # colabfold_msa searches only the complexes missing from the content-keyed
    # MSA cache, so adding pairs reruns it cheaply; the fold chunks take its
    # .done file as ancient() input, so that rerun does not refold finished
    # pairs. A pair's MSA identity is in the chunk's params instead (msas=).
    from nanopulldown.fold.colabfold import a3m_path, msa_key, msa_settings, run_params as cf_run_params  # noqa: E402

    CF_MSA = CF.get("msa") or {}
    CF_MSA_MODE = msa_settings(CF)["mode"]
    MSA_DONE = POST / "_colabfold" / "msa.done"
    MSA_KEY = {p: msa_key(SPECS[p], CF) for p in PAIRS}
    CF_KEY = hashlib.sha1(json.dumps(cf_run_params(CF), sort_keys=True).encode()).hexdigest()[:12]
    # local search on a GPU (colabfold.msa.gpu: 1): a GPU job sized by
    # colabfold.msa.slurm; server mode: a plain CPU job that needs internet
    # (CAIUS: run from the login node by scripts/caius/run.sh)
    MSA_GPU = CF_MSA_MODE == "local" and int(CF_MSA.get("gpu", 0)) == 1
    MSA_RES = {}
    if MSA_GPU:
        _ms = CF_MSA.get("slurm") or {}
        MSA_RES = dict(gpu=FOLD_GPU_CLAIM, docker_heavy=DOCKER_HEAVY_POOL,
                       **gpu_job_resources(_ms, float(_ms.get("runtime", 720))))
    elif CF_MSA_MODE == "local":
        _ms = CF_MSA.get("slurm") or {}
        MSA_RES = dict(mem_mb=int(_ms.get("mem_mb", 128000)), cpus_per_task=int(_ms.get("cpus", 32)),
                       runtime=int(_ms.get("runtime", 720)))

    rule colabfold_msa:
        output:
            done=str(MSA_DONE),
        log:
            str(FOLD_LOG / "colabfold_msa.log"),
        conda:
            # server mode only needs the client (CPU jax, any platform); local
            # search runs mmseqs from the CUDA fold env
            "envs/colabfold.yaml" if CF_MSA_MODE == "local" else "envs/colabfold_msa.yaml"
        priority: 50
        params:
            pairs=ALL_PAIRS, specs=ALL_SPECS_KEY,
            settings=json.dumps(msa_settings(CF), sort_keys=True),
        resources:
            **MSA_RES,
        shell:
            r"""
            {NP} colabfold msa --pairs {params.pairs} --done {output.done} > {log} 2>&1
            """

    for _k, _chunk in enumerate(fold_plan(max(PAIRS_PER_LOAD, 1), "fold_chunks_colabfold.json")):
        rule:
            name: f"run_colabfold_chunk_{_k:03d}"
            input:
                msa=ancient(str(MSA_DONE)),
            output:
                [str(POST / _p / "raw" / f"sample_{_s}.cif.gz") for _p in _chunk for _s in SAMPLE_IDX],
                [str(POST / _p / "raw" / "fold.parquet") for _p in _chunk],
            log:
                str(FOLD_LOG / "run_colabfold" / f"chunk_{_k:03d}.log"),
            conda:
                "envs/colabfold.yaml"
            resources:
                # same one-GPU-job guard as ESMFold2's chunks
                gpu=FOLD_GPU_CLAIM,
                docker_heavy=DOCKER_HEAVY_POOL,
                **fold_resources(_chunk),
            params:
                pairs=" ".join(_chunk),
                specs=" ".join(SPEC_KEY[_p] for _p in _chunk),
                msas=" ".join(MSA_KEY[_p] for _p in _chunk),
                settings=CF_KEY,
                root=str(POST),
            shell:
                r"""
                {FOLD_HF_ENV}{NP} colabfold fold --pairs {params.pairs} --out-root {params.root} > {log} 2>&1
                """

elif PAIRS_PER_LOAD <= 0:
    rule run_esmfold2:
        output:
            cifs=expand(str(POST / "{{pair}}" / "raw" / "sample_{s}.cif.gz"), s=SAMPLE_IDX),
            fold=str(POST / "{pair}" / "raw" / "fold.parquet"),
        log:
            str(FOLD_LOG / "run_esmfold2" / "{pair}.log"),
        conda:
            f"envs/{ESMFOLD_ENV}.yaml"
        resources:
            # gpu=1 caps run_esmfold2 at one concurrent job -- REQUIRED, not a
            # hint: the -Fast checkpoint loads ~25GB into Apple Silicon's shared
            # unified memory, and nothing else guards against multiple
            # concurrent loads (this OOM-crashed the Mac once already). Only
            # takes effect when the run is launched with `--resources gpu=1`
            # (see README) -- always pass it. docker_heavy claims the ENTIRE
            # pool (see config.yaml's `scheduling:`) so minimize_cif/run_plip
            # -- which each only claim 1 -- can't run concurrently with a model
            # load; once this rule finishes, up to docker_heavy_pool of them
            # run at once. Also requires a matching `--resources` flag.
            gpu=FOLD_GPU_CLAIM,
            docker_heavy=DOCKER_HEAVY_POOL,
            # on SLURM (profiles/ifb): GPU model, partition, memory, walltime
            **{k: (lambda wc, k=k: fold_resources([wc.pair]).get(k, ""))
               for k in ("gpu_model", "slurm_partition", "mem_mb", "cpus_per_task", "runtime", "slurm_extra")
               # the plugin passes `-C ''` for an empty constraint: declare it only where used
               + (("constraint",) if HPC.get("gpu_constraints") else ()) + (("gres",) if GPU_GRES else ())},
        params:
            out=lambda wc: str(POST / wc.pair / "raw"),
            spec=lambda wc: SPEC_KEY[wc.pair],
        shell:
            r"""
            : > {log}
            if [ "{FOLD_BACKEND}" = mlx ]; then
              python -c "import mlx_lm" 2>/dev/null || pip install --no-deps \
                "mlx-lm @ git+https://github.com/faustomilletari/mlx-lm.git@main" >> {log} 2>&1
            fi
            {FOLD_HF_ENV}{NP} fold --pair {wildcards.pair} --out-dir {params.out} >> {log} 2>&1
            """

else:
    for _k, _chunk in enumerate(fold_plan(PAIRS_PER_LOAD, "fold_chunks.json")):
        rule:
            name: f"run_esmfold2_chunk_{_k:03d}"
            output:
                [str(POST / _p / "raw" / f"sample_{_s}.cif.gz") for _p in _chunk for _s in SAMPLE_IDX],
                [str(POST / _p / "raw" / "fold.parquet") for _p in _chunk],
            log:
                str(FOLD_LOG / "run_esmfold2" / f"chunk_{_k:03d}.log"),
            conda:
                f"envs/{ESMFOLD_ENV}.yaml"
            resources:
                # same single-model-at-a-time guard as the per-pair rule above
                gpu=FOLD_GPU_CLAIM,
                docker_heavy=DOCKER_HEAVY_POOL,
                # on SLURM (profiles/ifb): GPU model, partition, memory, walltime
                **fold_resources(_chunk),
            params:
                pairs=" ".join(_chunk),
                specs=" ".join(SPEC_KEY[_p] for _p in _chunk),
                root=str(POST),
            shell:
                r"""
                : > {log}
                if [ "{FOLD_BACKEND}" = mlx ]; then
                  python -c "import mlx_lm" 2>/dev/null || pip install --no-deps \
                    "mlx-lm @ git+https://github.com/faustomilletari/mlx-lm.git@main" >> {log} 2>&1
                fi
                {FOLD_HF_ENV}{NP} fold --pairs {params.pairs} --out-root {params.root} >> {log} 2>&1
                """


rule compute_metrics:
    input:
        fold=str(POST / "{pair}" / "raw" / "fold.parquet"),
    output:
        parquet=str(POST / "{pair}" / "interface_metrics.parquet"),
    log:
        str(FOLD_LOG / "compute_metrics" / "{pair}.log"),
    conda:
        "envs/analysis.yaml"
    params:
        enabled=",".join(MET["enabled"]),
        pae=MET["pae_cutoff"], cb=MET["contact_cutoff_cb"], pairs=MET["pairs"],
        bait=lambda wc: ",".join(SPECS[wc.pair]["bait_chains"]),
        prey=lambda wc: ",".join(SPECS[wc.pair]["prey_chains"]),
    shell:
        r"""
        {NP} metrics \
          --pair {wildcards.pair} \
          --fold {input.fold} --results-root {POST} \
          --out {output.parquet} \
          --enabled {params.enabled} --pae-cutoff {params.pae} \
          --contact-cutoff-cb {params.cb} --pairs {params.pairs} \
          --bait-chains {params.bait} --prey-chains {params.prey} \
          > {log} 2>&1
        """


rule minimize_cif:
    input:
        cif=str(POST / "{pair}" / "raw" / "sample_{sample}.cif.gz"),
    output:
        pdb=str(POST / "{pair}" / "minimized" / "sample_{sample}.pdb.gz"),
        energy=str(POST / "{pair}" / "minimized" / "sample_{sample}_energy.parquet"),
    log:
        str(FOLD_LOG / "minimize" / "{pair}" / "sample_{sample}.log"),
    conda:
        "envs/minimize.yaml"
    resources:
        # Shares config.yaml's docker_heavy pool with run_esmfold2/run_plip
        # -- see run_esmfold2's resources: comment.
        docker_heavy=1,
    shell:
        r"""
        {NP} minimize {input.cif} {output.pdb} > {log} 2>&1
        """


# Violin pages: best-fragment points in fragment runs, else one point per pair
# (its mean over samples). From the run config, not guessed from the table:
# configs/pairs.parquet is shared across runs, so a full-length run can reuse a
# fragment run's spec for the same pair name, fragment label included (4 DNE1
# pairs whose "fragment" was the whole prey).
VIOLIN_MODE = "--fragments best" if (CFG.get("fragments") or {}).get("enabled") else "--fragments all --aggregate mean"

# `docker run ... <image>` on the Mac, `apptainer run ... plip.sif` on IFB. A
# global, not a param (see FOLD_HF_ENV); the image/memory/platform params stay
# so a change to them still reruns PLIP.
PLIP_RUN = " ".join(run_command(CFG, image=PLIP["image"], sif="plip", args=[],
                                mounts={"$(pwd)": "/work"}, workdir="/work",
                                platform=PLIP.get("docker_platform", ""), memory=PLIP["docker_memory"]))


# ── 2e. PLIP contact identification ──────────────────────────────────────
# --chains treats bait_chains as one interaction partner and prey_chains as
# the other (protein-protein only, so no --dnareceptor, and no fix_pdb pass
# -- `nanopulldown minimize`'s PDBFixer step already adds missing atoms/hydrogens,
# unlike ab_initio_pipeline's ChimeraX minimization). PLIP can't read .pdb.gz
# (it opens it in binary mode and fails), so the minimized structure is
# decompressed into the report dir for the run, then deleted together with
# PLIP's own <name>_protonated.pdb (~2 MB/sample, read by nothing).
rule run_plip:
    input:
        pdb=str(POST / "{pair}" / "minimized" / "sample_{sample}.pdb.gz"),
    output:
        report=str(POST / "{pair}" / "plip" / "sample_{sample}_report" / "sample_{sample}_report.txt"),
    params:
        image=PLIP["image"],
        memory=PLIP["docker_memory"],
        platform=(f"--platform {PLIP['docker_platform']}" if PLIP.get("docker_platform") else ""),
        bait=lambda wc: ",".join(SPECS[wc.pair]["bait_chains"]),
        prey=lambda wc: ",".join(SPECS[wc.pair]["prey_chains"]),
        max_chain_len=int(PLIP.get("max_chain_len", 850)),
        outdir=str(POST / "{pair}" / "plip" / "sample_{sample}_report"),
    log:
        str(FOLD_LOG / "plip" / "{pair}" / "sample_{sample}.log"),
    resources:
        # Shares config.yaml's docker_heavy pool with run_esmfold2/minimize_cif
        # -- see run_esmfold2's resources: comment. Each PLIP container also
        # runs under amd64/Rosetta emulation (config.yaml's plip.docker_platform),
        # which is the heaviest of the three per-instance -- keep
        # docker_heavy_pool modest (config.yaml default: 3) until you've
        # watched a few runs at that concurrency.
        docker_heavy=1,
    shell:
        r"""
        mkdir -p {params.outdir}
        pdb="{params.outdir}/sample_{wildcards.sample}.pdb"
        gunzip -c {input.pdb} > "$pdb"
        # chains over max_chain_len get extra ids for PLIP only (it segfaults on them)
        chains=$({NP} plip split-chains --pdb "$pdb" \
          --bait {params.bait} --prey {params.prey} --max-len {params.max_chain_len})
        {PLIP_RUN} \
          -f "$pdb" -t --chains "$chains" -o {params.outdir} > {log} 2>&1 \
          || {{ rm -f "$pdb" "$pdb.chainmap.json"; exit 1; }}
        {NP} plip restore-chains --report {output.report} --map "$pdb.chainmap.json"
        rm -f "$pdb" "$pdb.chainmap.json" "{params.outdir}/sample_{wildcards.sample}_protonated.pdb"
        """


# PLIP TXT report -> ChimeraX visualization script (open the .cxc in ChimeraX).
# pliparser's per-type contact CSVs and its cxc config are only needed while
# csv2cxc runs (the .cxc is self-contained), so they live in a scratch dir;
# the summary CSV is kept, as temp(), until aggregate_plip has folded it into
# the pair's plip_contacts.parquet. Bait chains are one colored group, prey
# chains the other; the minimized .pdb.gz is referenced by absolute path
# (ChimeraX opens gzip directly).
rule plip_parse:
    input:
        report=str(POST / "{pair}" / "plip" / "sample_{sample}_report" / "sample_{sample}_report.txt"),
        pdb=str(POST / "{pair}" / "minimized" / "sample_{sample}.pdb.gz"),
    output:
        cxc=str(POST / "{pair}" / "plip" / "sample_{sample}_report" / "sample_{sample}.cxc"),
        summary=temp(str(POST / "{pair}" / "plip" / "sample_{sample}_report" / "sample_{sample}_summary.csv")),
    params:
        tmp=str(POST / "{pair}" / "plip" / "sample_{sample}_report" / "_pliparser"),
        bait=lambda wc: ",".join(SPECS[wc.pair]["bait_chains"]),
        prey=lambda wc: ",".join(SPECS[wc.pair]["prey_chains"]),
        bait_color=CXC["bait_color"], prey_color=CXC["prey_color"],
        transparency=CXC["transparency"],
    conda:
        "envs/analysis.yaml"
    log:
        str(FOLD_LOG / "pliparser" / "{pair}" / "sample_{sample}.log"),
    shell:
        r"""
        rm -rf {params.tmp} && mkdir -p {params.tmp}/csv
        pliparser plip2csv --input {input.report} --output {params.tmp}/csv/ > {log} 2>&1
        {NP} plip cxc-config --pdb {input.pdb} --csv-dir {params.tmp}/csv \
          --out {params.tmp}/cxc-config.json --bait-chains {params.bait} --prey-chains {params.prey} \
          --bait-color {params.bait_color} --prey-color {params.prey_color} \
          --transparency {params.transparency} >> {log} 2>&1
        pliparser csv2cxc --config {params.tmp}/cxc-config.json --output {output.cxc} >> {log} 2>&1
        mv {params.tmp}/csv/summary.csv {output.summary}
        rm -rf {params.tmp}
        """


rule aggregate_plip:
    input:
        csvs=lambda wc: expand(
            str(POST / "{{pair}}" / "plip" / "sample_{s}_report" / "sample_{s}_summary.csv"), s=SAMPLE_IDX),
    output:
        str(POST / "{pair}" / "plip_contacts.parquet"),
    conda:
        "envs/analysis.yaml"
    log:
        str(FOLD_LOG / "aggregate_plip" / "{pair}.log"),
    shell:
        r"""
        {NP} plip aggregate {input.csvs} --pair {wildcards.pair} --out {output[0]} > {log} 2>&1
        """


rule fig_plip_contacts:
    input:
        contacts=str(POST / "{pair}" / "plip_contacts.parquet"),
        annotation=str(ANN / "{pair}" / "annotation.yaml"),
    output:
        report(str(FOLD_REP / "{pair}" / "plip_contacts.svg"),
               category="PLIP contacts", subcategory=bait_subcategory, labels=pair_report_labels),
    conda:
        "envs/analysis.yaml"
    params:
        theme=FIG_THEME,
        fmts=FMTS,
        spec=lambda wc: SPEC_KEY[wc.pair],
        curation=lambda wc: CUR_KEY[wc.pair],
        labels=lambda wc: LABEL_KEY[wc.pair],
    shell:
        r"""
        {NP} plot plip-heatmap --pair {wildcards.pair} \
          --contacts {input.contacts} \
          --annotation {input.annotation} \
          --out {output[0]} --formats {params.fmts}
        """


# PLIP contact frequency per residue of each chain against its CDD domains and
# conserved-feature sites (nanopulldown.report.cdd).
rule fig_plip_cdd:
    input:
        contacts=str(POST / "{pair}" / "plip_contacts.parquet"),
        cdd=str(ANN / "cdd_sites.parquet"),
        annotation=str(ANN / "{pair}" / "annotation.yaml"),
    output:
        report(str(FOLD_REP / "{pair}" / "plip_cdd.svg"),
               category="PLIP contacts on CDD sites", subcategory=bait_subcategory, labels=pair_report_labels),
    conda:
        "envs/analysis.yaml"
    params:
        theme=FIG_THEME,
        fmts=FMTS, n=N_SAMPLES,
        spec=lambda wc: SPEC_KEY[wc.pair],
        curation=lambda wc: CUR_KEY[wc.pair],
        labels=lambda wc: LABEL_KEY[wc.pair],
    shell:
        r"""
        {NP} plot plip-cdd --pair {wildcards.pair} \
          --contacts {input.contacts} --cdd {input.cdd} --annotation {input.annotation} \
          --num-samples {params.n} --out {output[0]} --formats {params.fmts}
        """


rule fig_domain_heatmap:
    input:
        fold=str(POST / "{pair}" / "raw" / "fold.parquet"),
        annotation=str(ANN / "{pair}" / "annotation.yaml"),
    output:
        report(str(FOLD_REP / "{pair}" / "domain_heatmap.svg"),
               category="Bait x prey domain confidence", subcategory=bait_subcategory,
               labels=pair_report_labels),
    conda:
        "envs/analysis.yaml"
    params:
        theme=FIG_THEME,
        pae=MET["pae_cutoff"], fmts=FMTS,
        spec=lambda wc: SPEC_KEY[wc.pair],
        curation=lambda wc: CUR_KEY[wc.pair],
        labels=lambda wc: LABEL_KEY[wc.pair],
    shell:
        r"""
        {NP} plot domain-heatmap --pair {wildcards.pair} \
          --fold {input.fold} --annotation {input.annotation} \
          --pae-cutoff {params.pae} --out {output[0]} --formats {params.fmts}
        """


# ColabFold-style MSA coverage (nanopulldown.report.plots.msa_coverage), only
# with an MSA-based engine, in two report sections: each distinct protein's
# unpaired MSA (Baits / Preys, like the domain map + AIUPred figures; drawn
# from the a3m of the first pair it is in -- the unpaired search is per
# sequence) and each pair's paired MSA (per bait). The a3m is not a declared
# output of colabfold_msa (it lives in the shared content-keyed cache), so the
# rules wait on msa.done and take the a3m path as a param -- a new MSA changes
# the path and redraws the figure.
if ENGINE == "colabfold":
    rule fig_msa_unpaired:
        input:
            msa=ancient(str(MSA_DONE)),
        output:
            report(str(FOLD_REP / "msa_unpaired" / "{protein}.svg"),
                   category="MSA coverage (unpaired)",
                   subcategory=lambda wc: "Baits" if DOMAIN_PROTEINS[wc.protein]["role"] == "bait" else "Preys",
                   labels=lambda wc: {
                       "protein": protein_name(DOMAIN_PROTEINS[wc.protein]["uniprot"], PROTEINS),
                       "name": protein_name(DOMAIN_PROTEINS[wc.protein]["uniprot"], PROTEINS, "long"),
                       "UniProt": DOMAIN_PROTEINS[wc.protein]["uniprot"],
                       "pairs": str(len(DOMAIN_PROTEINS[wc.protein]["pairs"]))}),
        conda:
            "envs/analysis.yaml"
        params:
            theme=FIG_THEME,
            fmts=FMTS,
            pair=lambda wc: DOMAIN_PROTEINS[wc.protein]["pair"],
            chain=lambda wc: DOMAIN_PROTEINS[wc.protein]["chain"],
            a3m=lambda wc: str(a3m_path(SPECS[DOMAIN_PROTEINS[wc.protein]["pair"]], CF)),
            labels=lambda wc: LABEL_KEY[DOMAIN_PROTEINS[wc.protein]["pair"]],
        shell:
            r"""
            {NP} plot msa-unpaired --pair {params.pair} --chain {params.chain} --a3m {params.a3m} \
              --out {output[0]} --formats {params.fmts}
            """

    rule fig_msa_paired:
        input:
            msa=ancient(str(MSA_DONE)),
        output:
            report(str(FOLD_REP / "{pair}" / "msa_paired.svg"),
                   category="MSA coverage (paired)", subcategory=bait_subcategory, labels=pair_report_labels),
        conda:
            "envs/analysis.yaml"
        params:
            theme=FIG_THEME,
            fmts=FMTS,
            a3m=lambda wc: str(a3m_path(SPECS[wc.pair], CF)),
            labels=lambda wc: LABEL_KEY[wc.pair],
        shell:
            r"""
            {NP} plot msa-paired --pair {wildcards.pair} --a3m {params.a3m} \
              --out {output[0]} --formats {params.fmts}
            """


rule fig_minimize_energy:
    input:
        energy=lambda wc: expand(str(POST / wc.pair / "minimized" / "sample_{s}_energy.parquet"), s=SAMPLE_IDX),
    output:
        svg=report(str(FOLD_REP / "{pair}" / "minimize_energy.svg"),
                   category="Minimization energy", subcategory=bait_subcategory, labels=pair_report_labels),
        table=str(FOLD_REP / "{pair}" / "minimize_failure_rate.csv"),
    conda:
        "envs/analysis.yaml"
    params:
        theme=FIG_THEME,
        n=N_SAMPLES, maxe=MZ["max_abs_energy"], fmts=FMTS,
        labels=lambda wc: LABEL_KEY[wc.pair],
    shell:
        r"""
        {NP} plot minimize-energy --pair {wildcards.pair} \
          --results-root {POST} --num-samples {params.n} --max-abs-energy {params.maxe} \
          --out {output.svg} --out-table {output.table} --formats {params.fmts}
        """


# ChimeraX-style clash counting (`nanopulldown clashes`) on the raw ESMFold2
# sample and its minimized counterpart, then a paired Wilcoxon test.
rule count_clashes:
    input:
        raw=lambda wc: expand(str(POST / wc.pair / "raw" / "sample_{s}.cif.gz"), s=SAMPLE_IDX),
        minimized=lambda wc: expand(str(POST / wc.pair / "minimized" / "sample_{s}.pdb.gz"), s=SAMPLE_IDX),
    output:
        str(POST / "{pair}" / "clashes.csv"),
    conda:
        "envs/analysis.yaml"
    params:
        n=N_SAMPLES, cutoff=CL["overlap_cutoff"], hb=CL["hbond_allowance"],
        sep=CL["bond_separation"],
    shell:
        r"""
        {NP} clashes --pair {wildcards.pair} \
          --results-root {POST} --num-samples {params.n} --out {output[0]} \
          --overlap-cutoff {params.cutoff} --hbond-allowance {params.hb} \
          --bond-separation {params.sep}
        """


# pyDock binding energy between the bait chains and the prey chains of every
# minimized sample (the AMBER charges need the hydrogens minimization adds).
rule pydock_energy:
    input:
        minimized=lambda wc: expand(str(POST / wc.pair / "minimized" / "sample_{s}.pdb.gz"), s=SAMPLE_IDX),
    output:
        str(POST / "{pair}" / "pydock_energy.parquet"),
    log:
        str(FOLD_LOG / "pydock" / "{pair}.log"),
    conda:
        "envs/pydock.yaml"
    params:
        n=N_SAMPLES,
        bait=lambda wc: ",".join(SPECS[wc.pair]["bait_chains"]),
        prey=lambda wc: ",".join(SPECS[wc.pair]["prey_chains"]),
    shell:
        r"""
        {NP} pydock --pair {wildcards.pair} \
          --results-root {POST} --num-samples {params.n} \
          --bait-chains {params.bait} --prey-chains {params.prey} \
          --out {output[0]} > {log} 2>&1
        """


rule fig_clash_test:
    input:
        [str(POST / p / "clashes.csv") for p in PAIRS],
    output:
        html=directory(str(FOLD_REP / "clash_test")),
        stats=str(FOLD_REP / "clash_test_stats.csv"),
        counts=str(FOLD_REP / "clash_counts.csv"),
    conda:
        "envs/analysis.yaml"
    params:
        alt=CL["alternative"],
    shell:
        r"""
        {NP} plot clash-test --tables {input} --outdir {output.html} \
          --out-stats {output.stats} --out-counts {output.counts} \
          --alternative {params.alt}
        """


rule collect_results_table:
    input:
        metrics=[str(POST / p / "interface_metrics.parquet") for p in PAIRS],
        plip=[str(POST / p / "plip_contacts.parquet") for p in PAIRS],
        pydock=[str(POST / p / "pydock_energy.parquet") for p in PAIRS],
        cdd=str(ANN / "cdd_sites.parquet"),
    output:
        str(FOLD_REP / "results_table.csv"),
    conda:
        "envs/analysis.yaml"
    params:
        pairs=" ".join(PAIRS),
        inputs=str(INPUTS_TABLE),
        labels=hashlib.sha1(" ".join(LABEL_KEY[p] for p in PAIRS).encode()).hexdigest()[:12],
        inputs_mtime=INPUTS_TABLE.stat().st_mtime if INPUTS_TABLE.exists() else 0,
    shell:
        r"""
        {NP} table results --pairs {params.pairs} \
          --results-root {POST} --inputs {params.inputs} --cdd {input.cdd} --out {output[0]}
        """


rule fig_metric_violins_interactive:
    input:
        str(FOLD_REP / "results_table.csv"),
    output:
        directory(str(FOLD_REP / "metric_violins_interactive")),
    conda:
        "envs/analysis.yaml"
    shell:
        r"""
        {NP} plot violins {VIOLIN_MODE} --table {input[0]} --outdir {output[0]}
        """


# PLIP contacts per interaction type: same interactive violin figure as the
# metric distributions, over the results table's plip_* count columns. The
# total (plip_contacts) is also one of the metric-distribution panels.
PLIP_TYPE_COLS = ["plip_hbond", "plip_hydrophobic", "plip_salt_bridge", "plip_water_bridge",
                  "plip_pi_stacking", "plip_pi_cation", "plip_halogen", "plip_metal"]
PLIP_TYPE_LABELS = ["hydrogen bonds", "hydrophobic", "salt bridges", "water bridges",
                    "π-stacking", "π-cation", "halogen bonds", "metal complexes"]


rule fig_plip_type_violins:
    input:
        str(FOLD_REP / "results_table.csv"),
    output:
        directory(str(FOLD_REP / "plip_type_violins")),
    conda:
        "envs/analysis.yaml"
    params:
        cols=" ".join(PLIP_TYPE_COLS),
        labels=" ".join(f"'{l}'" for l in PLIP_TYPE_LABELS),
    shell:
        r"""
        {NP} plot violins {VIOLIN_MODE} --table {input[0]} \
          --outdir {output[0]} --metrics {params.cols} --labels {params.labels} \
          --skip-all-zero --ncols 3 \
          --title "PLIP bait-prey contacts per interaction type (all pairs, samples, chain pairs)"
        """


# Controls: conformal p-values of every prey against the bait's decoys, BH
# hits, and the recall of the known direct partners (`nanopulldown calibrate`).
rule decoy_calibration:
    input:
        str(FOLD_REP / "results_table.csv"),
    output:
        directory(str(FOLD_REP / "decoy_calibration")),
    conda:
        "envs/analysis.yaml"
    log:
        str(FOLD_LOG / "decoy_calibration.log"),
    params:
        decoy=DECOY_TAG,
        positive=POSITIVE_TAG,
    shell:
        r"""
        {NP} calibrate --table {input[0]} --outdir {output[0]} \
          --decoy-tag {params.decoy} --positive-tag {params.positive} > {log} 2>&1
        """


rule collect_minimize_failures:
    input:
        [str(FOLD_REP / p / "minimize_failure_rate.csv") for p in PAIRS],
    output:
        str(FOLD_REP / "minimize_failure_rate.csv"),
    conda:
        "envs/analysis.yaml"
    params:
        labels=hashlib.sha1(" ".join(LABEL_KEY[p] for p in PAIRS).encode()).hexdigest()[:12],
    shell:
        r"""
        {NP} table minimize-failures --tables {input} --out {output[0]}
        """


# ── interactive tables — datavzrd bundle embedded in the .zip report ───────
# ── per-bait report sections ─────────────────────────────────────────────
# Each run-level page is built once per bait protein (all its fragments)
# from that bait's rows only, and filed under the bait as report subcategory,
# so baits never mix. The run-wide pages (metric_violins_interactive,
# plip_type_violins, decoy_calibration, clash_test, tables) stay buildable on
# demand but are no longer in the report.

rule bait_tables:
    input:
        results=str(FOLD_REP / "results_table.csv"),
        failures=str(FOLD_REP / "minimize_failure_rate.csv"),
    output:
        results=str(BY_BAIT / "{bait}" / "results_table.csv"),
        failures=str(BY_BAIT / "{bait}" / "minimize_failure_rate.csv"),
    conda:
        "envs/analysis.yaml"
    shell:
        r"""
        {NP} table by-bait --bait {wildcards.bait} --tables {input.results} {input.failures} \
          --outdir $(dirname {output.results})
        """


# Report captions of the violin pages: each score's definition and DOI, split
# into global fold confidence and bait-prey interface, from
# nanopulldown.core.metric_refs (the same registry the pages and the Parquet
# metadata cite from). report/captions/metric_refs.rst only renders this param.
METRIC_CAPTION = caption_rst(
    REPORT_METRICS,
    intro="One violin per score, per group of preys (known partners / screen preys / decoys). "
          "Scores of the whole predicted complex and scores of the bait-prey interface are "
          "plotted separately.")
PLIP_TYPE_CAPTION = caption_rst(
    PLIP_TYPE_COLS, intro="Bait-prey PLIP contacts of each minimized model, one violin per interaction type.")


rule fig_metric_violins_bait:
    input:
        str(BY_BAIT / "{bait}" / "results_table.csv"),
    output:
        report(directory(str(BY_BAIT / "{bait}" / "metric_violins")),
               category="Metric distributions", subcategory=lambda wc: BAIT_LABEL[wc.bait],
               labels=lambda wc: {"bait": BAIT_LABEL[wc.bait]}, htmlindex="index.html",
               caption="report/captions/metric_refs.rst"),
    conda:
        "envs/analysis.yaml"
    params:
        title=lambda wc: f"Metric distributions — {BAIT_LABEL[wc.bait]} (all pairs, samples, chain pairs)",
        caption=METRIC_CAPTION,
    shell:
        r"""
        {NP} plot violins {VIOLIN_MODE} --table {input[0]} --outdir {output[0]} --title "{params.title}"
        """


rule fig_plip_type_violins_bait:
    input:
        str(BY_BAIT / "{bait}" / "results_table.csv"),
    output:
        report(directory(str(BY_BAIT / "{bait}" / "plip_type_violins")),
               category="PLIP contacts by interaction type", subcategory=lambda wc: BAIT_LABEL[wc.bait],
               labels=lambda wc: {"bait": BAIT_LABEL[wc.bait]}, htmlindex="index.html",
               caption="report/captions/metric_refs.rst"),
    conda:
        "envs/analysis.yaml"
    params:
        caption=PLIP_TYPE_CAPTION,
        cols=" ".join(PLIP_TYPE_COLS),
        labels=" ".join(f"'{l}'" for l in PLIP_TYPE_LABELS),
        title=lambda wc: f"PLIP bait-prey contacts per interaction type — {BAIT_LABEL[wc.bait]} "
                         "(all pairs, samples, chain pairs)",
    shell:
        r"""
        {NP} plot violins {VIOLIN_MODE} --table {input[0]} \
          --outdir {output[0]} --metrics {params.cols} --labels {params.labels} \
          --skip-all-zero --ncols 3 --title "{params.title}"
        """


# Every prey's PLIP contacts on the bait, in full-length bait numbering, against
# the bait's CDD sites. Paths come from the table (--results-root), not the
# command line: a fragment run's bait has thousands of pairs.
rule fig_bait_cdd_map:
    input:
        table=str(BY_BAIT / "{bait}" / "results_table.csv"),
        contacts=lambda wc: [str(POST / p / "plip_contacts.parquet") for p in PAIRS_OF[wc.bait]],
        cdd=str(ANN / "cdd_sites.parquet"),
    output:
        report(str(BY_BAIT / "{bait}" / "cdd_contact_map.svg"),
               category="PLIP contacts on CDD sites", subcategory=lambda wc: BAIT_LABEL[wc.bait],
               labels=lambda wc: {"bait": BAIT_LABEL[wc.bait], "prey": "all preys"}),
    conda:
        "envs/analysis.yaml"
    params:
        theme=FIG_THEME,
        fmts=FMTS, n=N_SAMPLES,
        decoy=DECOY_TAG,
        positive=POSITIVE_TAG,
    shell:
        r"""
        {NP} plot bait-cdd-map --bait {wildcards.bait} --table {input.table} \
          --results-root {POST} --cdd {input.cdd} --num-samples {params.n} \
          --decoy-tag {params.decoy} --positive-tag {params.positive} \
          --out {output[0]} --formats {params.fmts}
        """


rule fig_clash_test_bait:
    input:
        lambda wc: [str(POST / p / "clashes.csv") for p in PAIRS_OF[wc.bait]],
    output:
        html=report(directory(str(BY_BAIT / "{bait}" / "clash_test")),
                    category="Minimization clashes", subcategory=lambda wc: BAIT_LABEL[wc.bait],
                    labels=lambda wc: {"bait": BAIT_LABEL[wc.bait]}, htmlindex="index.html"),
        stats=str(BY_BAIT / "{bait}" / "clash_test_stats.csv"),
        counts=str(BY_BAIT / "{bait}" / "clash_counts.csv"),
    conda:
        "envs/analysis.yaml"
    params:
        alt=CL["alternative"],
    shell:
        r"""
        {NP} plot clash-test --tables {input} --outdir {output.html} \
          --out-stats {output.stats} --out-counts {output.counts} \
          --alternative {params.alt}
        """


rule decoy_calibration_bait:
    input:
        str(BY_BAIT / "{bait}" / "results_table.csv"),
    output:
        report(directory(str(BY_BAIT / "{bait}" / "decoy_calibration")),
               category="Controls: decoy calibration and known partners",
               subcategory=lambda wc: BAIT_LABEL[wc.bait],
               labels=lambda wc: {"bait": BAIT_LABEL[wc.bait]}, htmlindex="index.html"),
    conda:
        "envs/analysis.yaml"
    log:
        str(FOLD_LOG / "decoy_calibration" / "{bait}.log"),
    params:
        decoy=DECOY_TAG,
        positive=POSITIVE_TAG,
    shell:
        r"""
        {NP} calibrate --table {input[0]} --outdir {output[0]} \
          --decoy-tag {params.decoy} --positive-tag {params.positive} > {log} 2>&1
        """


rule datavzrd_bait:
    input:
        config=str(REPO / "report" / "datavzrd" / "nanopulldown.datavzrd.yaml"),
        css=str(REPO / "src" / "nanopulldown" / "report" / "assets" / "datavzrd-dark.css"),
        results=str(BY_BAIT / "{bait}" / "results_table.csv"),
        failures=str(BY_BAIT / "{bait}" / "minimize_failure_rate.csv"),
        clash_stats=str(BY_BAIT / "{bait}" / "clash_test_stats.csv"),
        clash_counts=str(BY_BAIT / "{bait}" / "clash_counts.csv"),
    output:
        report(directory(str(BY_BAIT / "{bait}" / "tables")),
               category="Interactive tables", subcategory=lambda wc: BAIT_LABEL[wc.bait],
               labels=lambda wc: {"bait": BAIT_LABEL[wc.bait]}, htmlindex="index.html"),
    conda:
        "envs/analysis.yaml"
    log:
        str(FOLD_LOG / "datavzrd" / "{bait}.log"),
    shell:
        r"""
        {NP} report tables --config {input.config} --outdir {output[0]} --css {input.css} > {log} 2>&1
        """


rule datavzrd_folding:
    input:
        config=str(REPO / "report" / "datavzrd" / "nanopulldown.datavzrd.yaml"),
        css=str(REPO / "src" / "nanopulldown" / "report" / "assets" / "datavzrd-dark.css"),
        results=str(FOLD_REP / "results_table.csv"),
        failures=str(FOLD_REP / "minimize_failure_rate.csv"),
        clash_stats=str(FOLD_REP / "clash_test_stats.csv"),
        clash_counts=str(FOLD_REP / "clash_counts.csv"),
    output:
        directory(str(FOLD_REP / "tables")),
    conda:
        "envs/analysis.yaml"
    log:
        str(FOLD_LOG / "datavzrd.log"),
    shell:
        r"""
        {NP} report tables --config {input.config} --outdir {output[0]} --css {input.css} > {log} 2>&1
        """


onsuccess:
    shell(f"{NP} report build --repo {REPO} --out {Path(DIRS['reports']) / 'report.zip'}")
