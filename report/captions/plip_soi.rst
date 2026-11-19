PLIP contacts on sequences of interest (``nanopulldown.report.plots.plip_cdd``):
do the predicted complexes actually touch what the sequence analyses point at?
Per pair, one panel per chain: the per-residue PLIP contact frequency over the
samples, the domain map, then one row per sequence of interest with a tick per
residue (grey: never contacted; faded: contacted, but not on the partner region
the analysis names; solid: contacting that partner region), and in how many
samples it does. One subsection per analysis:

* **CDD sites**: CDD domains and their conserved-feature sites, plus coiled
  coils (solid = knobs-into-holes with the partner's coil). Per bait, a map of
  every prey's contacts on the bait's sites.
* **ELM motifs**: disordered ELM motifs whose binding Pfam domain is on the
  other side (solid = on that domain).
* **SPARROW IDR pairs**: bait IDR x prey IDR pairs scored for physicochemical
  complementarity (solid = on the partner IDR).
* **min-iPAE segments**: residues the fold places confidently on the partner
  (solid = on the partner residues they are placed on).

Only pairs with rows have an ELM, SPARROW or min-iPAE figure.
