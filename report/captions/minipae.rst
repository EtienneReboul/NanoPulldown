Short linear motifs placed by the folds (``nanopulldown.structure.minipae``):
per residue, the minimum inter-chain PAE onto the other side of the pair
(min-iPAE, Omidi et al. 2024 PNAS 121:e2406407121; AF2_MiniPAE), median over
samples. Confident segments are candidate SLiMs when short and disordered.
The decoy p tells how often the bait's decoy preys get as confident a pose.
The ELM support table checks each ELM motif x partner-domain match against
the min-iPAE of the motif onto that domain. This is model confidence, not
experimental evidence: it carries the fold's own false positives.
Whether the segments are actually in contact with the partner is drawn per pair
in **PLIP contacts on sequences of interest** (subsection *min-iPAE segments*).
