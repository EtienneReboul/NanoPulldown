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
* **Metric distributions** — one violin per score across all pairs, samples
  and chain pairs in the results table, in two groups: **global fold
  confidence** (pTM, pLDDT: the predicted complex as a whole) and the
  **bait-prey interface** (ipTM, pair ipTM, ipSAE, LIS / cLIS / iLIS, Pinc,
  pDockQ / pDockQ2, PLIP contacts, pyDock-1VDW, AF / pyDock z-score). Each
  score's definition and publication (with DOI) is in the page caption and
  under each group; the same references are stored in the ``references``
  field metadata of the Parquet column that holds the score.
* **PLIP contacts by interaction type** — the bait-prey PLIP contact count
  of every sample split by interaction type (hydrogen bonds, hydrophobic,
  salt bridges, π-stacking, π-cation, ...), one violin per type; the total
  count is the ``plip_contacts`` panel of the metric distributions, so it can
  be compared with the confidence scores as a ranking signal.
* **PLIP contacts on CDD sites** — per pair, one panel per chain: the
  per-residue PLIP contact frequency over the samples, under the chain's CDD
  domains and the conserved-feature sites CDD curates in them ("dimer
  interface", "substrate binding site", ...), each site residue marked
  contacted or not. Per bait, one map of every prey's contacts on the bait
  (full-length numbering, so fragments line up), grouped as known partners /
  preys / decoys and ordered by how many of the bait's site residues they
  touch. The results table carries the same per sample and side: site
  residues contacted, their enrichment over the chain's contact rate, and
  the share of contacts inside CDD domains.
* **MSA coverage** (ColabFold runs only) — the MSAs ColabFold folded from,
  drawn like ColabFold's own "Sequence coverage" plot: one row per aligned
  sequence, sorted and coloured by identity to the query, blank where the
  row has a gap; the black line is the per-position depth. Two sections:

  * **unpaired** — split into **Baits** and **Preys**, one figure per
    distinct protein (as the domain maps, same residue numbering): the
    protein's own homologs, i.e. how well AF2 can fold the chain itself;
  * **paired** — per bait, one figure per pair: only the rows ColabFold
    paired across the chains (same species), chains side by side, with each
    chain's median paired depth. A pair with few or no paired rows gives
    AF2-multimer little coevolution signal for the interface.
* **Minimization energy** — OpenMM energy trace per diffusion sample, plus a
  failure-rate table for any sample whose minimization diverged.

Domain maps use the curated ``domains:`` of ``configs/curated_domains.yaml``
for pairs marked ``annotation_reviewed: true``, and the automatic proposal in
``data/annotation/<pair>/annotation.yaml`` otherwise; curation never blocks a run.
