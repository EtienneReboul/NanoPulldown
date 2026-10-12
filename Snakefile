"""
Snakefile — nanopulldown (local, single-machine pipeline)
============================================================
One straight-through pipeline, no manual checkpoint in the middle: for
every pair in config.yaml's `pairs:` list, sequence annotation feeds
straight into folding.

  1. preprocessing (needs internet)
    Sequence annotation. Pairs themselves (bait/prey UniProt sequences
    resolved, chains assigned) come from scripts/expand_pairs.py, run once
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
      report figures (domain map + AIUPred) -- one per distinct protein,
      filed under Baits / Preys in the one report.zip (built after folding,
      below).

  2. folding (Apple Silicon GPU via MLX/Metal)
    ESMFold2-Fast folding (one seed, esmfold2.num_diffusion_samples
    diffusion samples, no MSA/templates) -> OpenMM minimization -> PAE-based
    rescoring (ipSAE/iLIS/Pinc) -> ONE report. Structures are stored gzipped
    (.cif.gz / .pdb.gz -- ChimeraX and gemmi open them as-is).
      run_esmfold2       one seed, N diffusion samples -> raw/sample_<i>.cif.gz +
                         raw/fold.parquet (scores, pae/plddt float16, topology;
                         schema in scripts/fold_io.py)
      compute_metrics    ipSAE / iLIS / Pinc + ESMFold2's own pair ipTM ->
                         results/<pair>/interface_metrics.parquet
      minimize_cif       OpenMM energy minimization -> minimized/sample_<i>.pdb.gz
                         + sample_<i>_energy.parquet, no ChimeraX / antechamber
      run_plip           PLIP (docker) on each minimized sample, bait_chains
                         vs prey_chains -> per-sample TXT report (the
                         decompressed input and PLIP's protonated copy are
                         deleted afterwards)
      plip_parse         pliparser -> per-sample ChimeraX .cxc (+ temporary contact CSVs)
      aggregate_plip     -> results/<pair>/plip_contacts.parquet (all samples)
      + report figures (domain x domain confidence heatmaps, minimize energy,
        PLIP contact heatmaps) and reports/report.zip (onsuccess hook; unzip
        -> report.html) -- the ONLY report this pipeline produces.

Usage (one command, both stages):
  snakemake --use-conda --cores 4
"""
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(workflow.basedir)
sys.path.insert(0, str(REPO / "scripts"))
from lib_pipeline import (inputs_table_path, label_digest, load_config, load_pairs,  # noqa: E402
                          load_proteins, pair_labels, pair_tokens, protein_chains,
                          protein_name, spec_digest)

configfile: str(REPO / "config.yaml")
if (REPO / "config.local.yaml").exists():
    configfile: str(REPO / "config.local.yaml")

report: "report/folding.rst"

CFG = load_config(REPO)
DIRS = CFG["dirs"]
EF = CFG["esmfold2"]
MAX_TOKENS = int(EF.get("max_tokens", 0))

# Specs come from configs/pairs.parquet (+ configs/curated_domains.yaml). A
# pair is dropped (with a warning) when it isn't in the table -- expand_pairs
# does not add pairs over esmfold2.max_tokens -- or when it is over max_tokens
# anyway (specs expanded before the limit existed): ESMFold2 memory grows
# ~quadratically with tokens and an over-limit fold aborts the whole process
# on the Metal buffer cap. 0 = no limit.
_ALL_SPECS = load_pairs(REPO)
# `pairs_file:` (written by scripts/expand_pairs.py from the config's
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
    print(f"[Snakefile] {len(_too_long)} pair(s) exceed esmfold2.max_tokens={MAX_TOKENS} and are ignored",
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


def pair_report_labels(wc):
    lb = LABELS[wc.pair]
    return {"bait": lb["bait"], "prey": lb["prey"], "prey name": lb["prey_long"], "pair": wc.pair}

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
FMTS = ",".join(RPT["figure_formats"])
N_SAMPLES = int(EF["num_diffusion_samples"])
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
        str(FOLD_REP / "clash_test"),
        str(FOLD_REP / "clash_test_stats.csv"),
        str(FOLD_REP / "clash_counts.csv"),
        [str(FOLD_REP / p / "plip_contacts.svg") for p in PAIRS],
        str(FOLD_REP / "results_table.csv"),
        str(FOLD_REP / "metric_violins_interactive"),
        str(FOLD_REP / "plip_type_violins"),
        str(FOLD_REP / "minimize_failure_rate.csv"),
        ([str(FOLD_REP / "tables")] if DATAVZRD else []),


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
        "envs/annotate.yaml"
    priority: 50
    params:
        pairs=ALL_PAIRS, specs=ALL_SPECS_KEY,
    shell:
        r"""
        python scripts/annotate_domains_batch.py --pairs {params.pairs} --done {output.done} \
          > {log} 2>&1
        """


rule annotate_disorder_batch:
    output:
        done=str(ANN_CACHE / "disorder.done"),
    log:
        str(PRE_LOG / "annotate_disorder_batch.log"),
    conda:
        "envs/annotate.yaml"
    priority: 50
    params:
        pairs=ALL_PAIRS, specs=ALL_SPECS_KEY,
    resources:
        # docker_heavy=1: never overlaps a run_esmfold2 model load (that rule
        # claims the whole pool). CPU-only -- no gpu= claim, the GPU stays free.
        docker_heavy=1,
    shell:
        r"""
        python scripts/annotate_disorder_batch.py --pairs {params.pairs} --done {output.done} \
          > {log} 2>&1
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
        "envs/annotate.yaml"
    params:
        spec=lambda wc: SPEC_KEY[wc.pair],
        curation=lambda wc: CUR_KEY[wc.pair],
    shell:
        r"""
        python scripts/merge_annotation.py \
          --pair {wildcards.pair} --out {output.yaml} \
          > {log} 2>&1
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
        "envs/report.yaml"
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
        python scripts/plot_domain_bar.py \
          --pair {params.pair} --annotation {input.annotation} --chain {params.chain} \
          --cutoff {params.cutoff} \
          --out {output[0]} --formats {params.fmts}
        """


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — folding
# ═══════════════════════════════════════════════════════════════════════════

# All N diffusion samples are declared as outputs (N is static, from
# esmfold2.num_diffusion_samples) so minimize_cif's per-sample input can wire
# back to this rule. fold.parquet is written last, so it also marks the pair
# as complete.
# Fold granularity. pairs_per_load = 0: one process (and one ~25 GB model load,
# with its swap spike) per pair. > 0: pairs are split into chunks of that size
# and each chunk is ONE job that loads the model once and folds its pairs in a
# loop (scripts/run_esmfold2_batch.py). A chunk's downstream steps (minimize,
# PLIP, ...) only start once the whole chunk is done, so keep chunks to ~1-2 h.
PAIRS_PER_LOAD = int(EF.get("pairs_per_load", 0))

if PAIRS_PER_LOAD <= 0:
    rule run_esmfold2:
        output:
            cifs=expand(str(POST / "{{pair}}" / "raw" / "sample_{s}.cif.gz"), s=SAMPLE_IDX),
            fold=str(POST / "{pair}" / "raw" / "fold.parquet"),
        log:
            str(FOLD_LOG / "run_esmfold2" / "{pair}.log"),
        conda:
            "envs/esmfold2.yaml"
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
            gpu=1,
            docker_heavy=DOCKER_HEAVY_POOL,
        params:
            out=lambda wc: str(POST / wc.pair / "raw"),
            spec=lambda wc: SPEC_KEY[wc.pair],
        shell:
            r"""
            python -c "import mlx_lm" 2>/dev/null || pip install --no-deps \
              "mlx-lm @ git+https://github.com/faustomilletari/mlx-lm.git@main" \
              > {log} 2>&1
            python scripts/run_esmfold2.py \
              --pair {wildcards.pair} \
              --out-dir {params.out} \
              >> {log} 2>&1
            """

else:
    for _k, _chunk in enumerate(PAIRS[i:i + PAIRS_PER_LOAD] for i in range(0, len(PAIRS), PAIRS_PER_LOAD)):
        rule:
            name: f"run_esmfold2_chunk_{_k:03d}"
            output:
                [str(POST / _p / "raw" / f"sample_{_s}.cif.gz") for _p in _chunk for _s in SAMPLE_IDX],
                [str(POST / _p / "raw" / "fold.parquet") for _p in _chunk],
            log:
                str(FOLD_LOG / "run_esmfold2" / f"chunk_{_k:03d}.log"),
            conda:
                "envs/esmfold2.yaml"
            resources:
                # same single-model-at-a-time guard as the per-pair rule above
                gpu=1,
                docker_heavy=DOCKER_HEAVY_POOL,
            params:
                pairs=" ".join(_chunk),
                specs=" ".join(SPEC_KEY[_p] for _p in _chunk),
                root=str(POST),
            shell:
                r"""
                python -c "import mlx_lm" 2>/dev/null || pip install --no-deps \
                  "mlx-lm @ git+https://github.com/faustomilletari/mlx-lm.git@main" \
                  > {log} 2>&1
                python scripts/run_esmfold2_batch.py \
                  --pairs {params.pairs} --out-root {params.root} \
                  >> {log} 2>&1
                """


rule compute_metrics:
    input:
        fold=str(POST / "{pair}" / "raw" / "fold.parquet"),
    output:
        parquet=str(POST / "{pair}" / "interface_metrics.parquet"),
    log:
        str(FOLD_LOG / "compute_metrics" / "{pair}.log"),
    conda:
        "envs/metrics.yaml"
    params:
        enabled=",".join(MET["enabled"]),
        pae=MET["pae_cutoff"], cb=MET["contact_cutoff_cb"], pairs=MET["pairs"],
        bait=lambda wc: ",".join(SPECS[wc.pair]["bait_chains"]),
        prey=lambda wc: ",".join(SPECS[wc.pair]["prey_chains"]),
    shell:
        r"""
        python scripts/compute_interface_metrics.py \
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
        python scripts/minimize_openmm.py {input.cif} {output.pdb} > {log} 2>&1
        """


# ── 2e. PLIP contact identification ──────────────────────────────────────
# --chains treats bait_chains as one interaction partner and prey_chains as
# the other (protein-protein only, so no --dnareceptor, and no fix_pdb pass
# -- minimize_openmm.py's PDBFixer step already adds missing atoms/hydrogens,
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
        chains=lambda wc: '--chains "[{},{}]"'.format(
            SPECS[wc.pair]["bait_chains"], SPECS[wc.pair]["prey_chains"]),
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
        docker run --rm --memory={params.memory} {params.platform} \
          -v $(pwd):/work -w /work {params.image} \
          -f "$pdb" -t {params.chains} -o {params.outdir} > {log} 2>&1 \
          || {{ rm -f "$pdb"; exit 1; }}
        rm -f "$pdb" "{params.outdir}/sample_{wildcards.sample}_protonated.pdb"
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
        "envs/pliparser.yaml"
    log:
        str(FOLD_LOG / "pliparser" / "{pair}" / "sample_{sample}.log"),
    shell:
        r"""
        rm -rf {params.tmp} && mkdir -p {params.tmp}/csv
        pliparser plip2csv --input {input.report} --output {params.tmp}/csv/ > {log} 2>&1
        python scripts/write_cxc_config.py --pdb {input.pdb} --csv-dir {params.tmp}/csv \
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
        "envs/report.yaml"
    log:
        str(FOLD_LOG / "aggregate_plip" / "{pair}.log"),
    shell:
        r"""
        python scripts/aggregate_plip_summaries.py {input.csvs} --pair {wildcards.pair} \
          --out {output[0]} > {log} 2>&1
        """


rule fig_plip_contacts:
    input:
        contacts=str(POST / "{pair}" / "plip_contacts.parquet"),
        annotation=str(ANN / "{pair}" / "annotation.yaml"),
    output:
        report(str(FOLD_REP / "{pair}" / "plip_contacts.svg"),
               category="PLIP contacts", labels=pair_report_labels),
    conda:
        "envs/report.yaml"
    params:
        theme=FIG_THEME,
        fmts=FMTS,
        spec=lambda wc: SPEC_KEY[wc.pair],
        curation=lambda wc: CUR_KEY[wc.pair],
        labels=lambda wc: LABEL_KEY[wc.pair],
    shell:
        r"""
        python scripts/plot_plip_heatmap.py --pair {wildcards.pair} \
          --contacts {input.contacts} \
          --annotation {input.annotation} \
          --out {output[0]} --formats {params.fmts}
        """


rule fig_domain_heatmap:
    input:
        fold=str(POST / "{pair}" / "raw" / "fold.parquet"),
        annotation=str(ANN / "{pair}" / "annotation.yaml"),
    output:
        report(str(FOLD_REP / "{pair}" / "domain_heatmap.svg"),
               category="Bait x prey domain confidence",
               labels=pair_report_labels),
    conda:
        "envs/report.yaml"
    params:
        theme=FIG_THEME,
        pae=MET["pae_cutoff"], fmts=FMTS,
        spec=lambda wc: SPEC_KEY[wc.pair],
        curation=lambda wc: CUR_KEY[wc.pair],
        labels=lambda wc: LABEL_KEY[wc.pair],
    shell:
        r"""
        python scripts/plot_domain_heatmap.py --pair {wildcards.pair} \
          --fold {input.fold} --annotation {input.annotation} \
          --pae-cutoff {params.pae} --out {output[0]} --formats {params.fmts}
        """


rule fig_minimize_energy:
    input:
        energy=lambda wc: expand(str(POST / wc.pair / "minimized" / "sample_{s}_energy.parquet"), s=SAMPLE_IDX),
    output:
        svg=report(str(FOLD_REP / "{pair}" / "minimize_energy.svg"),
                   category="Minimization energy", labels=pair_report_labels),
        table=str(FOLD_REP / "{pair}" / "minimize_failure_rate.csv"),
    conda:
        "envs/report.yaml"
    params:
        theme=FIG_THEME,
        n=N_SAMPLES, maxe=MZ["max_abs_energy"], fmts=FMTS,
        labels=lambda wc: LABEL_KEY[wc.pair],
    shell:
        r"""
        python scripts/plot_minimize_energy.py --pair {wildcards.pair} \
          --results-root {POST} --num-samples {params.n} --max-abs-energy {params.maxe} \
          --out {output.svg} --out-table {output.table} --formats {params.fmts}
        """


# ChimeraX-style clash counting (scripts/count_clashes.py) on the raw ESMFold2
# sample and its minimized counterpart, then a paired Wilcoxon test.
rule count_clashes:
    input:
        raw=lambda wc: expand(str(POST / wc.pair / "raw" / "sample_{s}.cif.gz"), s=SAMPLE_IDX),
        minimized=lambda wc: expand(str(POST / wc.pair / "minimized" / "sample_{s}.pdb.gz"), s=SAMPLE_IDX),
    output:
        str(POST / "{pair}" / "clashes.csv"),
    conda:
        "envs/clashes.yaml"
    params:
        n=N_SAMPLES, cutoff=CL["overlap_cutoff"], hb=CL["hbond_allowance"],
        sep=CL["bond_separation"],
    shell:
        r"""
        python scripts/count_clashes.py --pair {wildcards.pair} \
          --results-root {POST} --num-samples {params.n} --out {output[0]} \
          --overlap-cutoff {params.cutoff} --hbond-allowance {params.hb} \
          --bond-separation {params.sep}
        """


rule fig_clash_test:
    input:
        [str(POST / p / "clashes.csv") for p in PAIRS],
    output:
        html=report(directory(str(FOLD_REP / "clash_test")),
                    category="Minimization clashes", htmlindex="index.html"),
        stats=str(FOLD_REP / "clash_test_stats.csv"),
        counts=str(FOLD_REP / "clash_counts.csv"),
    conda:
        "envs/report.yaml"
    params:
        alt=CL["alternative"],
    shell:
        r"""
        python scripts/plot_clash_test.py --tables {input} --outdir {output.html} \
          --out-stats {output.stats} --out-counts {output.counts} \
          --alternative {params.alt}
        """


rule collect_results_table:
    input:
        metrics=[str(POST / p / "interface_metrics.parquet") for p in PAIRS],
        plip=[str(POST / p / "plip_contacts.parquet") for p in PAIRS],
    output:
        str(FOLD_REP / "results_table.csv"),
    conda:
        "envs/report.yaml"
    params:
        pairs=" ".join(PAIRS),
        inputs=str(INPUTS_TABLE),
        labels=hashlib.sha1(" ".join(LABEL_KEY[p] for p in PAIRS).encode()).hexdigest()[:12],
        inputs_mtime=INPUTS_TABLE.stat().st_mtime if INPUTS_TABLE.exists() else 0,
    shell:
        r"""
        python scripts/collect_results_table.py --pairs {params.pairs} \
          --results-root {POST} --inputs {params.inputs} --out {output[0]}
        """


rule fig_metric_violins_interactive:
    input:
        str(FOLD_REP / "results_table.csv"),
    output:
        report(directory(str(FOLD_REP / "metric_violins_interactive")),
               category="Metric distributions", htmlindex="index.html"),
    conda:
        "envs/report.yaml"
    shell:
        r"""
        python scripts/plot_metric_violins_interactive.py --table {input[0]} \
          --outdir {output[0]}
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
        report(directory(str(FOLD_REP / "plip_type_violins")),
               category="PLIP contacts by interaction type", htmlindex="index.html"),
    conda:
        "envs/report.yaml"
    params:
        cols=" ".join(PLIP_TYPE_COLS),
        labels=" ".join(f"'{l}'" for l in PLIP_TYPE_LABELS),
    shell:
        r"""
        python scripts/plot_metric_violins_interactive.py --table {input[0]} \
          --outdir {output[0]} --metrics {params.cols} --labels {params.labels} \
          --skip-all-zero --ncols 3 \
          --title "PLIP bait-prey contacts per interaction type (all pairs, samples, chain pairs)"
        """


rule collect_minimize_failures:
    input:
        [str(FOLD_REP / p / "minimize_failure_rate.csv") for p in PAIRS],
    output:
        str(FOLD_REP / "minimize_failure_rate.csv"),
    conda:
        "envs/report.yaml"
    params:
        labels=hashlib.sha1(" ".join(LABEL_KEY[p] for p in PAIRS).encode()).hexdigest()[:12],
    shell:
        r"""
        python scripts/collect_minimize_failures.py --tables {input} --out {output[0]}
        """


# ── interactive tables — datavzrd bundle embedded in the .zip report ───────
rule datavzrd_folding:
    input:
        config=str(REPO / "report" / "datavzrd" / "nanopulldown.datavzrd.yaml"),
        css=str(REPO / "scripts" / "datavzrd-dark.css"),
        results=str(FOLD_REP / "results_table.csv"),
        failures=str(FOLD_REP / "minimize_failure_rate.csv"),
        clash_stats=str(FOLD_REP / "clash_test_stats.csv"),
        clash_counts=str(FOLD_REP / "clash_counts.csv"),
    output:
        report(directory(str(FOLD_REP / "tables")),
               category="Interactive tables", htmlindex="index.html"),
    conda:
        "envs/report.yaml"
    log:
        str(FOLD_LOG / "datavzrd.log"),
    shell:
        r"""
        (
          cd "$(dirname {input.results})" \
            && datavzrd "{input.config}" --overwrite-output --output tables
        ) > {log} 2>&1
        python scripts/style_datavzrd.py {output[0]} {input.css} >> {log} 2>&1
        """


onsuccess:
    shell(f"bash {REPO}/scripts/build_report.sh {REPO} {Path(DIRS['reports']) / 'report.zip'}")
