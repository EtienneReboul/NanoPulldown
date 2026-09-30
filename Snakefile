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
      annotate_domains   ScanProsite + InterProScan v6 MATCH API -> domains.raw.tsv
      annotate_disorder  AIUPred docker (disorder + binding/MoRF)  -> disorder.tsv
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
      + report figures (domain x domain confidence heatmaps, minimize energy)
        and reports/report.zip (onsuccess hook; unzip -> report.html) -- the
        ONLY report this pipeline produces.

Usage (one command, both stages):
  snakemake --use-conda --cores 4
"""
import sys
from pathlib import Path

REPO = Path(workflow.basedir)
sys.path.insert(0, str(REPO / "scripts"))
from lib_pipeline import load_config, load_pair, protein_chains  # noqa: E402

configfile: str(REPO / "config.yaml")
if (REPO / "config.local.yaml").exists():
    configfile: str(REPO / "config.local.yaml")

report: "report/folding.rst"

CFG = load_config(REPO)
DIRS = CFG["dirs"]
PAIRS = CFG["pairs"]
SPECS = {p: load_pair(p, REPO) for p in PAIRS}
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
EF = CFG["esmfold2"]
MET = CFG["metrics"]
MZ = CFG["minimize"]
RPT = CFG["report"]
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
        chain_figs("disorder"),
        [str(POST / p / "interface_metrics.parquet") for p in PAIRS],
        [str(FOLD_REP / p / "domain_heatmap.svg") for p in PAIRS],
        [str(FOLD_REP / p / "minimize_energy.svg") for p in PAIRS],
        str(FOLD_REP / "results_table.csv"),
        str(FOLD_REP / "minimize_failure_rate.csv"),
        ([str(FOLD_REP / "tables")] if DATAVZRD else []),


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1 — preprocessing
# ═══════════════════════════════════════════════════════════════════════════

rule annotate_domains:
    input:
        spec=str(REPO / "configs" / "{pair}.yaml"),
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
    output:
        str(PRE_REP / "{pair}" / "domain_bar_{chain}.svg"),
    conda:
        "envs/report.yaml"
    params:
        fmts=",".join(CFG["report"]["figure_formats"]),
    shell:
        r"""
        python scripts/plot_domain_bar.py \
          --spec {input.spec} --annotation {input.annotation} --chain {wildcards.chain} \
          --out {output[0]} --formats {params.fmts}
        """


rule fig_disorder:
    input:
        disorder=str(ANN / "{pair}" / "disorder.tsv"),
    output:
        str(PRE_REP / "{pair}" / "disorder_{chain}.svg"),
    conda:
        "envs/report.yaml"
    params:
        fmts=",".join(CFG["report"]["figure_formats"]),
        cutoff=CFG["annotate"]["consensus"]["disorder_cutoff"],
    shell:
        r"""
        python scripts/plot_disorder.py \
          --disorder {input.disorder} --chain {wildcards.chain} \
          --cutoff {params.cutoff} --out {output[0]} --formats {params.fmts}
        """


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2 — folding
# ═══════════════════════════════════════════════════════════════════════════

# All N diffusion samples are declared as outputs (N is static, from
# esmfold2.num_diffusion_samples) so minimize_cif's per-sample input can wire
# back to this rule -- a `done` sentinel alone would hide the per-sample CIFs
# from the DAG.
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
    params:
        out=lambda wc: str(POST / wc.pair / "raw"),
    shell:
        r"""
        python scripts/run_esmfold2.py \
          --pair {wildcards.pair} --spec {input.spec} \
          --out-dir {params.out} --done {output.done} \
          > {log} 2>&1
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
    shell:
        r"""
        python scripts/minimize_openmm.py {input.cif} {output.pdb} > {log} 2>&1
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
