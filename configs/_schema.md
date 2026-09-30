# `configs/<bait>__<prey>.yaml` — per-pair schema

One file per entry in `config.yaml`'s `pairs:` list, named
`<bait_accession>__<prey_accession>.yaml`. Generated for you by
`scripts/expand_pairs.py` from `configs/baits.txt` x `configs/preys.txt`
(AlphaPulldown's bait-file x prey-file combinatorial convention); hand-edit
afterwards to curate `domains:`.

```yaml
name: bait__prey            # must equal the filename stem and the pairs: entry

# ── Bait / prey (AlphaPulldown-style ID:copies:start-stop) ─────────────────
bait: {uniprot: P01258, copies: 1, regions: null}   # regions: [[1,100]] for a fragment
prey: {uniprot: P01579, copies: 1, regions: null}

# ── Chains ────────────────────────────────────────────────────────────────
# Order defines the ESMFold2 StructurePredictionInput chain order. One entry
# per expanded chain (copies > 1 produces one entry per copy, ids A, B, C...
# assigned in bait-then-prey order). Protein-protein only -- no rna/dna/ligand.
sequences:
  - {id: A, name: bait_P01258,  sequence: "MABC...", role: bait}
  - {id: B, name: prey_P01579,  sequence: "MDEF...", role: prey}

bait_chains: [A]
prey_chains: [B]

# ── Annotation review gate ──────────────────────────────────────────────
# Stage 1 writes data/annotation/<pair>/annotation.yaml for you to merge
# into the `domains:` block below. Flip this to true once you've reviewed
# it -- stage 2 refuses to run for a pair still set false.
annotation_reviewed: false

# ── Domain / region map (hand-curated from stage-1 annotation) ──────────
# Per chain id, an ordered list of non-overlapping segments. `kind` is one
# of: domain | linker | disordered | morf. 1-based, inclusive. Used by the
# stage-2 domain x domain PAE heatmaps. "linker" is always a human call made
# during curation -- neither AIUPred track outputs it directly.
domains:
  A:
    - {name: dom1, start: 1, end: 76, kind: domain}
  B:
    - {name: dom1,  start: 1,  end: 60, kind: domain}
    - {name: tail,  start: 61, end: 95, kind: disordered}
    - {name: morf1, start: 78, end: 88, kind: morf}
```

## Worked example

`configs/example_toy__example_toy.yaml` (a ubiquitin homodimer smoke test) is
kept as a worked reference for this schema -- use it as the template when
adding a new `configs/<pair>.yaml`.
