"""Unit tests of the min-iPAE SLiM arithmetic (nanopulldown.structure.minipae).

  PYTHONPATH=src python -m pytest tests/test_minipae.py
"""
import numpy as np

from nanopulldown.structure import minipae as M


def toy_pae(n_a=4, n_b=5, low=((1, 2),), fill=20.0, low_val=2.0):
    """Chain A = tokens 0..n_a-1 (partner), B = n_a.. (motif side); PAE(i in A, j in B)
    set to low_val for the (i, j-in-B) pairs in `low`."""
    n = n_a + n_b
    pae = np.full((n, n), fill, np.float32)
    for i, jb in low:
        pae[i, n_a + jb] = low_val
    return pae


def test_min_ipae_uses_partner_rows():
    pae = toy_pae(low=((1, 2), (3, 2), (0, 4)))
    pae[4 + 2, 1] = 0.5                        # the transposed entry (aligned on B) must not count
    v, am = M.min_ipae(pae, motif_idx=range(4, 9), partner_idx=range(0, 4))
    assert v.tolist() == [20, 20, 2, 20, 2]
    assert am[2] in (1, 3) and am[4] == 0


def test_min_ipae_matches_af2_minipae_v7():
    # AF2_MiniPAE v7: receptor first, SLiM chain second, pae[:idp_len, idp_len:].min(axis=0)
    rng = np.random.default_rng(0)
    pae = rng.uniform(0, 30, (9, 9))
    v, _ = M.min_ipae(pae, range(4, 9), range(0, 4))
    assert np.allclose(v, pae[:4, 4:].min(axis=0))


def test_call_segments_threshold_support_gap_len():
    prof = np.array([9, 3, 3, 9, 3, 3, 9, 9, 9, 3, 3, 3, 3, 9, 9, 3])
    sup = np.ones_like(prof, float)
    # gap of 1 merges 2-3 and 5-6; the lone 16 (gap 2) is shorter than min_len
    assert M.call_segments(prof, sup, 4.0, 0.5, max_gap=1, min_len=3) == [(2, 6), (10, 13)]
    assert M.call_segments(prof, sup, 4.0, 0.5, max_gap=0, min_len=3) == [(10, 13)]
    sup[10:12] = 0.2                            # residues 11-12 not supported across samples
    assert M.call_segments(prof, sup, 4.0, 0.5, max_gap=0, min_len=2) == [(2, 3), (5, 6)]


def test_call_segments_strict_threshold():
    assert M.call_segments([4.0, 4.0, 4.0], [1, 1, 1], 4.0, 0.5, 0, 1) == []


def test_greedy_span_like_af2_minipae():
    assert M.greedy_span([9, 2, 9, 9, 3, 9], "ABCDEF", 4.0) == (2, 5, "BcdE")
    assert M.greedy_span([9, 2, 9], "ABC", 4.0) == (None, None, None)


def test_conformal_p_lower_is_better():
    null = [2.0, 5.0, 8.0, np.nan]
    assert M.conformal_p(1.0, null) == 1 / 4
    assert M.conformal_p(5.0, null) == 3 / 4
    assert M.conformal_p(1.0, []) == 1.0


def test_full_positions_regions():
    assert M.full_positions(3, None) == [1, 2, 3]
    assert M.full_positions(5, [(10, 12), (20, 21)]) == [10, 11, 12, 20, 21]
    assert M.full_positions(4, [(10, 12)]) == [1, 2, 3, 4]      # length mismatch: identity


def _rows(pae, chains, iptm):
    return [{"pae": p, "token_chain": np.array(chains), "iptm": t, "ptm": t, "sample_index": k}
            for k, (p, t) in enumerate(zip(pae, iptm))]


def test_pair_profiles_roles_and_best_sample():
    spec = {"bait_chains": ["A"], "prey_chains": ["B"],
            "sequences": [{"id": "A", "sequence": "MKLV"}, {"id": "B", "sequence": "GSPEQ"}]}
    chains = ["A"] * 4 + ["B"] * 5
    p1 = toy_pae(low=((1, 2),), low_val=3.0)
    p2 = toy_pae(low=((3, 2),), low_val=1.0)                # best sample (higher ipTM)
    prof = M.pair_profiles(_rows([p1, p2], chains, [0.3, 0.8]), spec)
    b = prof["B"]
    assert b["role"] == "prey" and b["values"].shape == (2, 5)
    assert b["best"][2] == 1.0
    assert b["partner_chain"][2] == "A" and b["partner_res"][2] == 4   # token 3 = residue 4 of A
    # bait side: rows of B onto columns of A, all 20 here
    assert prof["A"]["values"].min() == 20


def test_pair_profiles_skips_length_mismatch():
    spec = {"bait_chains": ["A"], "prey_chains": ["B"],
            "sequences": [{"id": "A", "sequence": "MKLV"}, {"id": "B", "sequence": "GSPE"}]}
    prof = M.pair_profiles(_rows([toy_pae()], ["A"] * 4 + ["B"] * 5, [0.5]), spec)
    assert "B" not in prof and "A" in prof


def test_domain_min_ipae_restricts_rows():
    chains = ["A"] * 4 + ["B"] * 5
    pae = toy_pae(low=((0, 2), (3, 3)), low_val=2.0)
    rows = _rows([pae], chains, [0.5])
    # motif B 3-4 (tokens 6, 7); domain A 1-2 -> only (0, 2) counts
    assert M.domain_min_ipae(rows, "B", 3, 4, "A", 1, 2).tolist() == [2.0]
    assert M.domain_min_ipae(rows, "B", 4, 4, "A", 1, 2).tolist() == [20.0]
    assert M.domain_min_ipae(rows, "B", 4, 4, "A", None, None).tolist() == [2.0]
