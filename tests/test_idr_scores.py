"""Unit tests of the IDR complementarity arithmetic (nanopulldown.annotate.idr_scores)
and the chain projection of nanopulldown.annotate.idr_pairs.

  PYTHONPATH=src python -m pytest tests/test_idr_scores.py     (envs/sparrow.yaml for the SPARROW checks)
"""
import numpy as np
import pytest

from nanopulldown.annotate import idr_scores as S

W = [15, 25]


def summ(seq, phospho=()):
    return S.summary(S.property_vectors(seq, phospho), W)


# ── IDR calling ───────────────────────────────────────────────────────────

def test_define_idrs_merges_short_gaps_and_drops_short_runs():
    d = [0.9] * 20 + [0.1] * 10 + [0.9] * 20 + [0.1] * 11 + [0.9] * 25 + [0.1] * 5
    # 20 + gap 10 (merged) + 20 = residues 1-50; gap 11 splits; the last run is 25 < 30
    assert S.define_idrs(d, 0.5, 10, 30) == [(1, 50)]
    assert S.define_idrs(d, 0.5, 11, 30) == [(1, 86)]
    assert S.define_idrs(d, 0.5, 10, 20) == [(1, 50), (62, 86)]


def test_define_idrs_threshold_inclusive_and_empty():
    assert S.define_idrs([0.5] * 30) == [(1, 30)]
    assert S.define_idrs([0.49] * 100) == []
    assert S.define_idrs([]) == []


# ── scores on toy sequences ───────────────────────────────────────────────

def test_k30_e30_is_maximal_charge_complementarity():
    sc = S.pair_scores(summ("K" * 30), summ("E" * 30), 15)
    assert sc["s_charge"] == pytest.approx(1.0)
    assert sc["local_charge"] == pytest.approx(1.0)
    assert sc["s_pi"] == 0 and sc["s_catpi"] == 0


def test_like_charges_repel():
    sc = S.pair_scores(summ("K" * 30), summ("R" * 30), 15)
    assert sc["s_charge"] == pytest.approx(-1.0)
    assert sc["local_charge"] == pytest.approx(-1.0)


def test_charge_score_is_symmetric_and_ranks_opposite_above_neutral():
    a, b, n = summ("KKAKKSKKGK" * 4), summ("EEDEESDEGE" * 4), summ("GSGSAGSGPS" * 4)
    assert S.pair_scores(a, b, 15)["s_charge"] == pytest.approx(S.pair_scores(b, a, 15)["s_charge"])
    assert S.pair_scores(a, b, 15)["local_charge"] > S.pair_scores(a, n, 15)["local_charge"] == 0


def test_local_charge_finds_blocks_in_a_neutral_idr():
    # net-neutral IDR with a K block and an E block: global score ~0, local score high
    a = "K" * 15 + "GS" * 20 + "E" * 15
    b = "E" * 30
    sa, sb = summ(a), summ(b)
    sc = S.pair_scores(sa, sb, 15)
    assert sc["s_charge"] == pytest.approx(0.0)
    assert sc["local_charge"] == pytest.approx(1.0)
    i, j = S.local_charge_windows(sa, sb, 15)
    assert a[i:i + 15] == "K" * 15


def test_cation_pi_and_pi():
    sa, sb = summ("FYW" * 10), summ("RK" * 15)
    sc = S.pair_scores(sa, sb, 15)
    assert sc["s_catpi"] == pytest.approx(1.0) and sc["local_catpi"] == pytest.approx(1.0)
    assert sc["s_pi"] == 0
    assert S.pair_scores(sa, sa, 15)["s_pi"] == pytest.approx(1.0)
    assert S.mechanism_windows("cation_pi", sa, sb, 15) == (0, 0)


def test_hydrophobic_patch():
    hyd = S.pair_scores(summ("G" * 10 + "ILVILVILVILVILV" + "G" * 10), summ("LIVLIVLIVLIVLIV" + "S" * 20), 15)
    pol = S.pair_scores(summ("S" * 35), summ("S" * 35), 15)
    assert hyd["hydro_local"] > 0.8 > pol["hydro_local"]


def test_phosphate_charge():
    q = S.charge_vector("SKSE", phospho=[0, 2])
    assert list(q) == [-2, 1, -2, -1]
    # phosphorylating a basic IDR's serines flips it acidic -> charge complementarity with K30
    base = "KS" * 15
    ph = [i for i, c in enumerate(base) if c == "S"]
    assert S.pair_scores(summ(base), summ("K" * 30), 15)["s_charge"] < 0
    assert S.pair_scores(summ(base, ph), summ("K" * 30), 15)["s_charge"] > 0


def test_broadcasting_matches_elementwise():
    seqs = ["KKEEGS" * 6, "DEDEDESS" * 5, "FYRKGS" * 6, "LIVAGS" * 6]
    bg = S.stack_summaries([summ(s) for s in seqs], W)
    a = summ("KAKPKKAAKS" * 4)
    vec = S.pair_scores(a, bg, 25)
    for i, s in enumerate(seqs):
        one = S.pair_scores(a, summ(s), 25)
        for k in one:
            assert vec[k][i] == pytest.approx(one[k])


# ── null models ───────────────────────────────────────────────────────────

def test_shuffle_null_rewards_blocky_patterning():
    a = "K" * 15 + "G" * 25          # one basic block
    b = "E" * 15 + "G" * 25          # one acidic block
    sa, sb = summ(a), summ(b)
    ha, hb = S.shuffle_summary(a, 500, W), S.shuffle_summary(b, 500, W)
    obs = S.pair_scores(sa, sb, 15)["local_charge"]
    z, p = S.shuffle_null(obs, S.pair_scores(ha, hb, 15)["local_charge"])
    assert z > 3 and p < 0.01
    # composition (hence global scores) is preserved by shuffling
    assert np.allclose(ha["ncpr"], sa["ncpr"]) and np.allclose(ha["arom"], sa["arom"])


def test_shuffles_are_reproducible():
    a1, a2 = S.shuffle_summary("KEGS" * 10, 50, W), S.shuffle_summary("KEGS" * 10, 50, W)
    assert np.array_equal(a1["ncpr_max_15"], a2["ncpr_max_15"])


def test_percentile_midrank():
    assert S.percentile(5, [1, 2, 3, 4]) == 1.0
    assert S.percentile(0, [1, 2, 3, 4]) == 0.0
    assert S.percentile(2, [1, 2, 3, 4]) == pytest.approx(0.375)


def test_dominant_mechanism_needs_positive_score_and_threshold():
    pcts = {"s_charge": 0.99, "local_charge": 0.97, "s_catpi": 0.5, "local_catpi": 0.6,
            "s_pi": 0.1, "local_pi": 0.2, "hydro_local": 0.98}
    raw = {k: 0.1 for k in pcts} | {"s_charge": -0.1}
    label, mp = S.dominant_mechanism(pcts, raw, 0.95)
    assert mp["charge"] == 0.97                       # s_charge < 0 does not count
    assert label == "hydrophobic"
    assert S.dominant_mechanism({k: 0.5 for k in pcts}, raw, 0.95)[0] == "none"


# ── projection onto fragment chains ───────────────────────────────────────

def test_chain_idrs_projects_onto_regions():
    from nanopulldown.annotate.idr_pairs import chain_idrs

    class P:                                          # a Protein stand-in
        seq = "A" * 100 + "K" * 60 + "A" * 40
        acc, idrs, morfs, phospho = "X", [(101, 160)], [(120, 130)], {125}

    # fragment = residues 1-20 + 131-200: the IDR piece 131-160 is 30 aa, chain residues 21-50
    d = chain_idrs(P, [(1, 20), (131, 200)], "B", "prey", 30)
    assert len(d) == 1 and (d[0]["start"], d[0]["end"]) == (21, 50)
    assert (d[0]["full_start"], d[0]["full_end"]) == (131, 160)
    assert d[0]["sequence"] == "K" * 30
    assert d[0]["morfs"] == [] and d[0]["phospho_idx"] == []
    assert chain_idrs(P, [(1, 20), (141, 200)], "B", "prey", 30) == []     # 20 aa piece dropped
    full = chain_idrs(P, None, "B", "prey", 30)[0]
    assert full["morfs"] == [(120, 130)] and full["phospho_idx"] == [24]


# ── agreement with SPARROW (envs/sparrow.yaml) ────────────────────────────

def test_windows_match_sparrow_profiles():
    sparrow = pytest.importorskip("sparrow")
    seq = "MSEAAQKEEKKPAAEKKAAGSEEDDEDKRKPAEKKAAAKKAKKPAGFYWAKKAKSPKKAKAAKPKKAPKSPAKAKAVK"
    p = sparrow.Protein(seq)
    v = S.property_vectors(seq)
    for w in (15, 25):
        assert np.allclose(S.window_means(v["charge"], w), p.linear_sequence_profile("NCPR", w, end_mode=""))
        assert np.allclose(S.window_means(v["arom"], w), p.linear_sequence_profile("aromatic", w, end_mode=""))
        assert np.allclose(S.window_means(v["hyd"], w) * 9,
                           p.linear_sequence_profile("hydrophobicity", w, end_mode=""))
    assert summ(seq)["ncpr"] == pytest.approx(p.NCPR)
    assert summ(seq)["arom"] == pytest.approx(p.fraction_aromatic)
