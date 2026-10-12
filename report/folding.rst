nanopulldown — bait/prey screening report
==========================================

The ONE report for every bait/prey pair in ``config.yaml``'s ``pairs:``
list, folded locally with ESMFold2-Fast (one seed,
``esmfold2.num_diffusion_samples`` diffusion samples, no MSA, no templates).

* **Domain map + AIUPred** — split into **Baits** and **Preys**, one figure
  per distinct protein (not per pair: the bait is drawn once, not once per
  pair it is in). Each figure stacks AIUPred's disorder / binding (MoRF) /
  linker tracks over the segment map (domain / linker / disordered / MoRF,
  from Stage 1's ScanProsite + InterProScan v6 annotation, or the curated
  ``domains:``). The ``pairs`` label counts the pairs the protein is in.
* **Bait x prey domain confidence** — one panel per diffusion sample,
  cell = mean ``1 - PAE/pae_cutoff`` (clipped at 0) between each bait-domain x
  prey-domain pair (AlphaPulldown-style PAE-derived confidence, domain-aggregated
  the way ab_initio_pipeline aggregates PLIP contacts). White = no confident
  relative placement, dark blue = confident.
* **Results table** (interactive) — per pair, per sample, per bait-prey
  chain pair: pTM / ipTM / ESMFold2's own pair ipTM, plus ipSAE / iLIS /
  Pinc / pDockQ / pDockQ2 from ab_initio_pipeline's PAE-based rescoring
  stack.
* **Metric distributions** — one violin per confidence / interface metric
  (pTM, ipTM, pLDDT, ipSAE, iLIS, Pinc, pDockQ, ...) across all pairs,
  samples and chain pairs in the results table.
* **Minimization energy** — OpenMM energy trace per diffusion sample, plus a
  failure-rate table for any sample whose minimization diverged.

Domain maps use the curated ``domains:`` of ``configs/curated_domains.yaml``
for pairs marked ``annotation_reviewed: true``, and the automatic proposal in
``data/annotation/<pair>/annotation.yaml`` otherwise; curation never blocks a run.
