"""
Snakefile — nanopulldown (local, single-machine pipeline)
============================================================
One straight-through pipeline, no manual checkpoint in the middle: for
every pair in config.yaml's `pairs:` list, sequence annotation feeds
straight into folding.

  1. preprocessing (needs internet)
    Sequence annotation. Pairs themselves (bait/prey UniProt sequences
    resolved, chains assigned) come from scripts/expand_pairs.py, run once
    ahead of Snakemake -- see the README. This stage only annotates
    configs/<pair>.yaml's already-resolved `sequences:`.
      annotate_domains_batch / annotate_disorder_batch
                         once per UNIQUE sequence into data/annotation/_cache/
                         (throttled InterPro API + ScanProsite; one AIUPred container run per 200 seqs)
      annotate_domains   per pair, assembled from the cache -> domains.raw.tsv
      annotate_disorder  per pair, assembled from the cache -> disorder.tsv
      merge_annotation   -> data/annotation/<pair>/annotation.yaml -- a
                          proposed `domains:` block; hand-curate it into
                          configs/<pair>.yaml whenever convenient (see
                          configs/_schema.md), it isn't required to proceed
      report figures (domain map, disorder/binding) -- viewed directly as
      SVGs, not packaged into a report (nanopulldown's one report.zip comes
      from folding, below).

  2. folding (Apple Silicon GPU via MLX/Metal)
    ESMFold2-Fast folding (one seed, esmfold2.num_diffusion_samples
    diffusion samples, no MSA/templates) -> metadata compression -> OpenMM
    minimization -> PAE-based rescoring (ipSAE/iLIS/Pinc) -> ONE report.
      run_esmfold2       one seed, N diffusion samples -> raw/sample_<i>.cif +
                         sample_<i>_confidences.json
      compress_metadata  -> results/metadata/<pair>/{model_metadata.parquet,arrays.h5}
      compute_metrics    ipSAE / iLIS / Pinc + ESMFold2's own pair ipTM ->
                         results/<pair>/interface_metrics.parquet
      minimize_cif       OpenMM energy minimization (+ *_energy.csv trace),
                         no ChimeraX / antechamber
      run_plip           PLIP (docker) on each minimized sample, bait_chains
                         vs prey_chains -> per-sample TXT report
      plip_to_csv        pliparser -> per-sample summary.csv
      aggregate_plip     -> results/<pair>/plip_summary.csv (all samples)
      + report figures (domain x domain confidence heatmaps, minimize energy,
        PLIP contact heatmaps) and reports/report.zip (onsuccess hook; unzip
        -> report.html) -- the ONLY report this pipeline produces.

Usage (one command, both stages):
  snakemake --use-conda --cores 4
"""
import sys
from pathlib import Path

REPO = Path(workflow.basedir)
sys.path.insert(0, str(REPO / "scripts"))
from lib_pipeline import load_config, load_pair, pair_tokens, protein_chains  # noqa: E402

configfile: str(REPO / "config.yaml")
if (REPO / "config.local.yaml").exists():
    configfile: str(REPO / "config.local.yaml")

report: "report/folding.rst"

CFG = load_config(REPO)
DIRS = CFG["dirs"]
EF = CFG["esmfold2"]
MAX_TOKENS = int(EF.get("max_tokens", 0))

# A pair is dropped (with a warning) when its spec is missing -- expand_pairs
# does not write specs for pairs over esmfold2.max_tokens -- or when it is
# over max_tokens anyway (specs expanded before the limit existed): ESMFold2
# memory grows ~quadratically with tokens and an over-limit fold aborts the
# whole process on the Metal buffer cap. 0 = no limit.
PAIRS, SPECS, _no_spec, _too_long = [], {}, [], []
for _p in CFG["pairs"]:
    if not (REPO / "configs" / f"{_p}.yaml").exists():
        _no_spec.append(_p)
        continue
    _spec = load_pair(_p, REPO)
    if MAX_TOKENS and pair_tokens(_spec) > MAX_TOKENS:
        _too_long.append(_p)
        continue
    PAIRS.append(_p)
    SPECS[_p] = _spec
if _no_spec:
    print(f"[Snakefile] WARNING: {len(_no_spec)} pair(s) in config.yaml have no configs/<pair>.yaml "
          f"(not expanded, or skipped as too long) and are ignored, e.g. {_no_spec[:3]}", file=sys.stderr)
if _too_long:
    print(f"[Snakefile] {len(_too_long)} pair(s) exceed esmfold2.max_tokens={MAX_TOKENS} and are ignored",
          file=sys.stderr)
CHAIN_IDS = {p: [c["id"] for c in protein_chains(SPECS[p])] for p in PAIRS}

# ── stage 1: preprocessing ───────────────────────────────────────────────
ANN = Path(DIRS["annotation"])
PRE_REP = Path(DIRS["reports"]) / "preprocessing"
PRE_LOG = Path(DIRS["logs"]) / "preprocessing"

# ── stage 2: folding ─────────────────────────────────────────────────────
META = Path(DIRS["metadata"])
POST = Path(DIRS["postproc"])
FOLD_REP = Path(DIRS["reports"]) / "folding"
FOLD_LOG = Path(DIRS["logs"]) / "folding"
MET = CFG["metrics"]
MZ = CFG["minimize"]
CL = MZ["clashes"]
PLIP = CFG["plip"]
CXC = PLIP["cxc"]
RPT = CFG["report"]
DOCKER_HEAVY_POOL = int(CFG["scheduling"]["docker_heavy_pool"])
DATAVZRD = bool(RPT.get("datavzrd", False))
FMTS = ",".join(RPT["figure_formats"])
N_SAMPLES = int(EF["num_diffusion_samples"])
SAMPLE_IDX = list(range(N_SAMPLES))


def chain_figs(kind):
    return [str(PRE_REP / p / f"{kind}_{cid}.svg")
            for p in PAIRS for cid in CHAIN_IDS[p]]


rule all:
    default_target: True
    input:
        [str(ANN / p / "annotation.yaml") for p in PAIRS],
        chain_figs("domain_bar"),
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
        str(FOLD_REP / "minimize_failure_rate.csv"),
        ([str(FOLD_REP / "tables")] if DATAVZRD else []),


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — preprocessing
# ═══════════════════════════════════════════════════════════════════════════

ANN_CACHE = ANN / "_cache"
SPEC_FILES = [str(REPO / "configs" / f"{p}.yaml") for p in PAIRS]


# Annotation is per UNIQUE SEQUENCE, not per pair: the bait is in every pair,
# so per-pair jobs would re-query the web services / re-run the AIUPred
# container ~2x per pair for work that is identical. These two batch rules fill
# a per-sequence cache once (web calls in a small thread pool; AIUPred in a few
# multi-FASTA container runs); the per-pair rules below just assemble from it.
# They run concurrently with each other and, via `priority`, ahead of folding.
rule annotate_domains_batch:
    input:
        SPEC_FILES,
    output:
        done=str(ANN_CACHE / "domains.done"),
    log:
        str(PRE_LOG / "annotate_domains_batch.log"),
    conda:
        "envs/annotate.yaml"
    priority: 50
    shell:
        r"""
        python scripts/annotate_domains_batch.py --specs {input} --done {output.done} \
          > {log} 2>&1
        """


rule annotate_disorder_batch:
    input:
        SPEC_FILES,
    output:
        done=str(ANN_CACHE / "disorder.done"),
    log:
        str(PRE_LOG / "annotate_disorder_batch.log"),
    conda:
        "envs/annotate.yaml"
    priority: 50
    resources:
        # docker_heavy=1: never overlaps a run_esmfold2 model load (that rule
        # claims the whole pool). CPU-only -- no gpu= claim, the GPU stays free.
        docker_heavy=1,
    shell:
        r"""
        python scripts/annotate_disorder_batch.py --specs {input} --done {output.done} \
          > {log} 2>&1
        """


rule annotate_domains:
    input:
        spec=str(REPO / "configs" / "{pair}.yaml"),
        cache=str(ANN_CACHE / "domains.done"),
    output:
        tsv=str(ANN / "{pair}" / "domains.raw.tsv"),
    log:
        str(PRE_LOG / "annotate_domains" / "{pair}.log"),
    conda:
        "envs/annotate.yaml"
    shell:
        r"""
        python scripts/annotate_domains.py \
          --pair {wildcards.pair} --spec {input.spec} --out {output.tsv} \
          > {log} 2>&1
        """


rule annotate_disorder:
    input:
        spec=str(REPO / "configs" / "{pair}.yaml"),
        cache=str(ANN_CACHE / "disorder.done"),
    output:
        tsv=str(ANN / "{pair}" / "disorder.tsv"),
    log:
        str(PRE_LOG / "annotate_disorder" / "{pair}.log"),
    conda:
        "envs/annotate.yaml"
    shell:
        r"""
        python scripts/annotate_disorder_aiupred.py \
          --pair {wildcards.pair} --spec {input.spec} --out {output.tsv} \
          > {log} 2>&1
        """


rule merge_annotation:
    input:
        spec=str(REPO / "configs" / "{pair}.yaml"),
        domains=str(ANN / "{pair}" / "domains.raw.tsv"),
        disorder=str(ANN / "{pair}" / "disorder.tsv"),
    output:
        yaml=str(ANN / "{pair}" / "annotation.yaml"),
    log:
        str(PRE_LOG / "merge_annotation" / "{pair}.log"),
    conda:
        "envs/annotate.yaml"
    shell:
        r"""
        python scripts/merge_annotation.py \
          --pair {wildcards.pair} --spec {input.spec} \
          --domains {input.domains} --disorder {input.disorder} \
          --out {output.yaml} \
          > {log} 2>&1
        """


rule fig_domain_bar:
    input:
        spec=str(REPO / "configs" / "{pair}.yaml"),
        annotation=str(ANN / "{pair}" / "annotation.yaml"),
        disorder=str(ANN / "{pair}" / "disorder.tsv"),
    output:
        report(str(PRE_REP / "{pair}" / "domain_bar_{chain}.svg"),
               category="Domain map + AIUPred", labels={"pair": "{pair}", "chain": "{chain}"}),
    conda:
        "envs/report.yaml"
    params:
        fmts=",".join(CFG["report"]["figure_formats"]),
        cutoff=CFG["annotate"]["consensus"]["disorder_cutoff"],
    shell:
        r"""
        python scripts/plot_domain_bar.py \
          --spec {input.spec} --annotation {input.annotation} --chain {wildcards.chain} \
          --disorder {input.disorder} --cutoff {params.cutoff} \
          --out {output[0]} --formats {params.fmts}
        """


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — folding
# ═══════════════════════════════════════════════════════════════════════════

# All N diffusion samples are declared as outputs (N is static, from
# esmfold2.num_diffusion_samples) so minimize_cif's per-sample input can wire
# back to this rule -- a `done` sentinel alone would hide the per-sample CIFs
# from the DAG.
# Fold granularity. pairs_per_load = 0: one process (and one ~25 GB model load,
# with its swap spike) per pair. > 0: pairs are split into chunks of that size
# and each chunk is ONE job that loads the model once and folds its pairs in a
# loop (scripts/run_esmfold2_batch.py). A chunk's downstream steps (minimize,
# PLIP, ...) only start once the whole chunk is done, so keep chunks to ~1-2 h.
PAIRS_PER_LOAD = int(EF.get("pairs_per_load", 0))

if PAIRS_PER_LOAD <= 0:
    rule run_esmfold2:
        input:
            spec=str(REPO / "configs" / "{pair}.yaml"),
        output:
            cifs=expand(str(POST / "{{pair}}" / "raw" / "sample_{s}.cif"), s=SAMPLE_IDX),
            jsons=expand(str(POST / "{{pair}}" / "raw" / "sample_{s}_confidences.json"), s=SAMPLE_IDX),
            done=str(POST / "{pair}" / "raw" / "prediction.done"),
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
        shell:
            r"""
            python -c "import mlx_lm" 2>/dev/null || pip install --no-deps \
              "mlx-lm @ git+https://github.com/faustomilletari/mlx-lm.git@main" \
              > {log} 2>&1
            python scripts/run_esmfold2.py \
              --pair {wildcards.pair} --spec {input.spec} \
              --out-dir {params.out} --done {output.done} \
              >> {log} 2>&1
            """

else:
    for _k, _chunk in enumerate(PAIRS[i:i + PAIRS_PER_LOAD] for i in range(0, len(PAIRS), PAIRS_PER_LOAD)):
        rule:
            name: f"run_esmfold2_chunk_{_k:03d}"
            input:
                [str(REPO / "configs" / f"{_p}.yaml") for _p in _chunk],
            output:
                [str(POST / _p / "raw" / f"sample_{_s}.cif") for _p in _chunk for _s in SAMPLE_IDX],
                [str(POST / _p / "raw" / f"sample_{_s}_confidences.json") for _p in _chunk for _s in SAMPLE_IDX],
                [str(POST / _p / "raw" / "prediction.done") for _p in _chunk],
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


rule compress_metadata:
    input:
        done=str(POST / "{pair}" / "raw" / "prediction.done"),
    output:
        parquet=str(META / "{pair}" / "model_metadata.parquet"),
        h5=str(META / "{pair}" / "arrays.h5"),
    log:
        str(FOLD_LOG / "compress" / "{pair}.log"),
    conda:
        "envs/metrics.yaml"
    shell:
        r"""
        python scripts/compress_esmfold2_metadata.py \
          --pair {wildcards.pair} \
          --results-root {POST} --out-root {META} \
          --delete-originals --skip-merge > {log} 2>&1
        """


rule compute_metrics:
    input:
        metadata=str(META / "{pair}" / "model_metadata.parquet"),
        h5=str(META / "{pair}" / "arrays.h5"),
        spec=str(REPO / "configs" / "{pair}.yaml"),
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
          --metadata-root {META} --results-root {POST} \
          --out {output.parquet} \
          --enabled {params.enabled} --pae-cutoff {params.pae} \
          --contact-cutoff-cb {params.cb} --pairs {params.pairs} \
          --bait-chains {params.bait} --prey-chains {params.prey} \
          > {log} 2>&1
        """


rule minimize_cif:
    input:
        cif=str(POST / "{pair}" / "raw" / "sample_{sample}.cif"),
    output:
        pdb=str(POST / "{pair}" / "minimized" / "sample_{sample}.pdb"),
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
# unlike ab_initio_pipeline's ChimeraX minimization).
rule run_plip:
    input:
        pdb=str(POST / "{pair}" / "minimized" / "sample_{sample}.pdb"),
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
        docker run --rm --memory={params.memory} {params.platform} \
          -v $(pwd):/work -w /work {params.image} \
          -f {input.pdb} -t {params.chains} -o {params.outdir} > {log} 2>&1
        actual="{params.outdir}/$(basename {input.pdb} .pdb)_report.txt"
        [ "$actual" != "{output.report}" ] && mv "$actual" "{output.report}" || true
        """


rule plip_to_csv:
    input:
        report=str(POST / "{pair}" / "plip" / "sample_{sample}_report" / "sample_{sample}_report.txt"),
    output:
        summary=str(POST / "{pair}" / "plip" / "sample_{sample}_report" / "csv" / "summary.csv"),
    params:
        outdir=str(POST / "{pair}" / "plip" / "sample_{sample}_report" / "csv"),
    conda:
        "envs/pliparser.yaml"
    log:
        str(FOLD_LOG / "pliparser" / "{pair}" / "sample_{sample}.log"),
    shell:
        r"""
        mkdir -p {params.outdir}
        pliparser plip2csv --input {input.report} --output {params.outdir}/ > {log} 2>&1
        """


# PLIP contacts -> ChimeraX visualization script (open the .cxc in ChimeraX).
# Bait chains are one colored group, prey chains the other; the minimized PDB
# is referenced by absolute path.
rule plip_to_cxc:
    input:
        csv=str(POST / "{pair}" / "plip" / "sample_{sample}_report" / "csv" / "summary.csv"),
        pdb=str(POST / "{pair}" / "minimized" / "sample_{sample}.pdb"),
    output:
        cxc=str(POST / "{pair}" / "plip" / "sample_{sample}_report" / "sample_{sample}.cxc"),
        config=str(POST / "{pair}" / "plip" / "sample_{sample}_report" / "cxc-config.json"),
    params:
        csv_dir=str(POST / "{pair}" / "plip" / "sample_{sample}_report" / "csv"),
        bait=lambda wc: ",".join(SPECS[wc.pair]["bait_chains"]),
        prey=lambda wc: ",".join(SPECS[wc.pair]["prey_chains"]),
        bait_color=CXC["bait_color"], prey_color=CXC["prey_color"],
        transparency=CXC["transparency"],
    conda:
        "envs/pliparser.yaml"
    log:
        str(FOLD_LOG / "plip_to_cxc" / "{pair}" / "sample_{sample}.log"),
    shell:
        r"""
        python scripts/write_cxc_config.py --pdb {input.pdb} --csv-dir {params.csv_dir} \
          --out {output.config} --bait-chains {params.bait} --prey-chains {params.prey} \
          --bait-color {params.bait_color} --prey-color {params.prey_color} \
          --transparency {params.transparency} > {log} 2>&1
        pliparser csv2cxc --config {output.config} --output {output.cxc} >> {log} 2>&1
        """


rule aggregate_plip:
    input:
        csvs=lambda wc: expand(
            str(POST / "{{pair}}" / "plip" / "sample_{s}_report" / "csv" / "summary.csv"), s=SAMPLE_IDX),
    output:
        str(POST / "{pair}" / "plip_summary.csv"),
    conda:
        "envs/report.yaml"
    log:
        str(FOLD_LOG / "aggregate_plip" / "{pair}.log"),
    shell:
        r"""
        python scripts/aggregate_plip_summaries.py {input.csvs} --out {output[0]} > {log} 2>&1
        """


rule fig_plip_contacts:
    input:
        spec=str(REPO / "configs" / "{pair}.yaml"),
        summary=str(POST / "{pair}" / "plip_summary.csv"),
        annotation=str(ANN / "{pair}" / "annotation.yaml"),
    output:
        report(str(FOLD_REP / "{pair}" / "plip_contacts.svg"),
               category="PLIP contacts", labels={"pair": "{pair}"}),
    conda:
        "envs/report.yaml"
    params:
        fmts=FMTS,
    shell:
        r"""
        python scripts/plot_plip_heatmap.py --pair {wildcards.pair} \
          --spec {input.spec} --summary {input.summary} \
          --annotation {input.annotation} \
          --out {output[0]} --formats {params.fmts}
        """


rule fig_domain_heatmap:
    input:
        spec=str(REPO / "configs" / "{pair}.yaml"),
        metadata=str(META / "{pair}" / "model_metadata.parquet"),
        h5=str(META / "{pair}" / "arrays.h5"),
    output:
        report(str(FOLD_REP / "{pair}" / "domain_heatmap.svg"),
               category="Bait x prey domain confidence",
               labels={"pair": "{pair}"}),
    conda:
        "envs/report.yaml"
    params:
        pae=MET["pae_cutoff"], fmts=FMTS,
    shell:
        r"""
        python scripts/plot_domain_heatmap.py --pair {wildcards.pair} \
          --spec {input.spec} --metadata-root {META} \
          --pae-cutoff {params.pae} --out {output[0]} --formats {params.fmts}
        """


rule fig_minimize_energy:
    input:
        pdbs=lambda wc: expand(str(POST / wc.pair / "minimized" / "sample_{s}.pdb"), s=SAMPLE_IDX),
    output:
        svg=report(str(FOLD_REP / "{pair}" / "minimize_energy.svg"),
                   category="Minimization energy", labels={"pair": "{pair}"}),
        table=str(FOLD_REP / "{pair}" / "minimize_failure_rate.csv"),
    conda:
        "envs/report.yaml"
    params:
        n=N_SAMPLES, maxe=MZ["max_abs_energy"], fmts=FMTS,
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
        raw=lambda wc: expand(str(POST / wc.pair / "raw" / "sample_{s}.cif"), s=SAMPLE_IDX),
        minimized=lambda wc: expand(str(POST / wc.pair / "minimized" / "sample_{s}.pdb"), s=SAMPLE_IDX),
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
        [str(POST / p / "interface_metrics.parquet") for p in PAIRS],
    output:
        str(FOLD_REP / "results_table.csv"),
    conda:
        "envs/report.yaml"
    params:
        pairs=" ".join(PAIRS),
    shell:
        r"""
        python scripts/collect_results_table.py --pairs {params.pairs} \
          --metadata-root {META} --results-root {POST} --out {output[0]}
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


rule collect_minimize_failures:
    input:
        [str(FOLD_REP / p / "minimize_failure_rate.csv") for p in PAIRS],
    output:
        str(FOLD_REP / "minimize_failure_rate.csv"),
    conda:
        "envs/report.yaml"
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
    shell(f"bash {REPO}/scripts/build_report.sh {REPO}")
