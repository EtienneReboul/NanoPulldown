nanopulldown — bait/prey screening report
==========================================

The ONE report for every bait/prey pair in ``config.yaml``'s ``pairs:``
list, folded locally with ESMFold2-Fast (one seed,
``esmfold2.num_diffusion_samples`` diffusion samples, no MSA, no templates).

* **Domain map + AIUPred** — split into **Baits** and **Preys**, one figure
  per distinct protein (not per pair: the bait is drawn once, not once per
  pair it is in), laid out like InterPro's protein viewer: AIUPred's
  disorder / binding (MoRF) / linker tracks; the protein's family; the
  **domain map** the heatmaps use; one row per InterPro entry with the member
  databases that agree on it; unintegrated signatures; MobiDB-lite, coiled
  coils, signal peptides, TM helices and AIUPred MoRFs. The proposed domain
  map is InterPro's representative domain set (plus InterPro domain entries
  it leaves uncovered), each region named after its InterPro domain entry;
  the curated ``domains:`` replace it once reviewed. The ``pairs`` label
  counts the pairs the protein is in.
* **Bait x prey domain confidence** — one panel per diffusion sample,
  cell = mean ``1 - PAE/pae_cutoff`` (clipped at 0) between each bait-domain x
  prey-domain pair (AlphaPulldown-style PAE-derived confidence, domain-aggregated
  the way ab_initio_pipeline aggregates PLIP contacts). Cells fade into the
  background where there is no confident relative placement and turn bright
  blue where there is.
* **Results table** (interactive) — per pair, per sample, per bait-prey
  chain pair: pTM / ipTM / ESMFold2's own pair ipTM, plus ipSAE / iLIS /
  Pinc / pDockQ / pDockQ2 from ab_initio_pipeline's PAE-based rescoring
  stack.
* **Metric distributions** — one violin per confidence / interface metric
  (pTM, ipTM, pLDDT, ipSAE, iLIS, Pinc, pDockQ, PLIP contacts, ...) across all pairs,
  samples and chain pairs in the results table.
* **PLIP contacts by interaction type** — the bait-prey PLIP contact count
  of every sample split by interaction type (hydrogen bonds, hydrophobic,
  salt bridges, π-stacking, π-cation, ...), one violin per type; the total
  count is the ``plip_contacts`` panel of the metric distributions, so it can
  be compared with the confidence scores as a ranking signal.
* **Minimization energy** — OpenMM energy trace per diffusion sample, plus a
  failure-rate table for any sample whose minimization diverged.

Domain maps use the curated ``domains:`` of ``configs/curated_domains.yaml``
for pairs marked ``annotation_reviewed: true``, and the automatic proposal in
``data/annotation/<pair>/annotation.yaml`` otherwise; curation never blocks a run.
