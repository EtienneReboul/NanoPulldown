nanopulldown — bait/prey screening report
==========================================

The ONE report for every bait/prey pair in ``config.yaml``'s ``pairs:``
list, folded locally with ESMFold2-Fast (one seed,
``esmfold2.num_diffusion_samples`` diffusion samples, no MSA, no templates).

* **Domain map** — curated (``annotation_reviewed: true``) segment map per
  chain: domain / linker / disordered / MoRF, from Stage 1's
  ScanProsite + InterProScan v6 annotation.
* **Disorder / binding** — AIUPred's disorder, binding(MoRF) and linker tracks per
  chain.
* **Domain x domain confidence heatmaps** — one panel per diffusion sample,
  cell = mean inverse-PAE between each bait-domain x prey-domain pair
  (AlphaPulldown-style PAE-derived confidence, domain-aggregated the way
  ab_initio_pipeline aggregates PLIP contacts).
* **Results table** (interactive) — per pair, per sample, per bait-prey
  chain pair: pTM / ipTM / ESMFold2's own pair ipTM, plus ipSAE / iLIS /
  Pinc / pDockQ / pDockQ2 from ab_initio_pipeline's PAE-based rescoring
  stack.
* **Metric distributions** — one violin per confidence / interface metric
  (pTM, ipTM, pLDDT, ipSAE, iLIS, Pinc, pDockQ, ...) across all pairs,
  samples and chain pairs in the results table.
* **Minimization energy** — OpenMM energy trace per diffusion sample, plus a
  failure-rate table for any sample whose minimization diverged.

Pairs with ``annotation_reviewed: false`` are refused at the start of this
workflow — review ``data/annotation/<pair>/annotation.yaml`` and curate
``domains:`` in ``configs/<pair>.yaml`` first.
