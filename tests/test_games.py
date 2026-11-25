"""Splitting-game values and the dual certificate behind (3/4)^n."""
import numpy as np

from qfhe.games import (BB84, _objective_operator, channel_splitting_value,
                        channel_value, complete_isometry, dual_certificate,
                        naive_values, optimal_channel, optimal_split_value,
                        p_win_isometry, scheme_splitting_game, sdp_dual_bound)

COS4 = np.cos(np.pi / 8) ** 4      # ~0.72855


def test_naive_values_exact():
    nv = naive_values()
    assert np.isclose(nv["z_copy"], 5 / 8)
    assert np.isclose(nv["x_copy"], 5 / 8)
    assert np.isclose(nv["fixed_00"], 3 / 8)
    assert np.isclose(nv["fixed_plus"], 3 / 8)


def test_dual_certificate_psd():
    """The paper's Theorem relies on: spectrum(M) >= 0 and (3/8) I - M >= 0,
    i.e. Y = (3/8) I_2 is dual feasible with Tr Y = 3/4."""
    spec_m, spec_gap = dual_certificate()
    assert spec_m.min() > -1e-10
    assert spec_gap.min() > -1e-10


def test_channel_value_is_three_quarters():
    val, kraus = channel_splitting_value(restarts=12, seed=2)
    assert 0.749 < val <= 0.75 + 1e-6            # primal reaches the dual bound
    assert kraus.shape == (2, 4, 2)
    g = sum(k.conj().T @ k for k in kraus)       # trace preserving
    assert np.allclose(g, np.eye(2), atol=1e-6)


def test_exact_optimal_channel():
    """Explicit optimal channel: exactly TP, value exactly 3/4 (matches the
    dual bound Tr Y), and complementary slackness --- each vec(A_k) is a
    3/8-eigenvector of the objective operator M, i.e. lies in ker((3/8) I - M).
    This witnesses p*_1 = 3/4 with no SDP duality machinery."""
    kraus = optimal_channel()
    assert np.allclose(sum(k.conj().T @ k for k in kraus), np.eye(2), atol=1e-12)
    assert abs(channel_value(kraus) - 0.75) < 1e-12
    M = _objective_operator()
    for A in kraus:
        v = A.reshape(-1, order="F")             # column-stacking vec, matches M
        assert np.allclose(M @ v, (3 / 8) * v, atol=1e-12)


def test_sdp_dual_bound_meets_primal():
    bound, slack = sdp_dual_bound(restarts=6, seed=0)
    assert slack < 1e-3                          # certificate is feasible
    assert 0.75 - 1e-3 <= bound <= 0.77          # converges to Tr Y = 3/4


def test_isometry_optimum_below_channels():
    best, V = optimal_split_value(restarts=8, seed=1)
    assert 0.72 <= best <= 0.75 + 1e-6           # cos^4(pi/8) <= channel value
    assert abs(best - COS4) < 5e-3 or best >= COS4
    assert np.allclose(V.conj().T @ V, np.eye(2), atol=1e-9)


def test_complete_isometry_unitary():
    rng = np.random.default_rng(4)
    z = rng.normal(size=16)
    Q, _ = np.linalg.qr((z[:8] + 1j * z[8:]).reshape(4, 2))
    W = complete_isometry(Q)
    assert np.allclose(W.conj().T @ W, np.eye(4), atol=1e-9)
    assert np.allclose(W[:, 0], Q[:, 0], atol=1e-9)   # documented interleaved layout
    assert np.allclose(W[:, 2], Q[:, 1], atol=1e-9)


def test_scheme_splitting_game_vs_bound():
    """End-to-end game with the measure-and-resend isometry: empirical p_both
    matches (5/8)^n and stays under the certificate bound (3/4)^n."""
    n, trials = 3, 600
    res = scheme_splitting_game(n, trials, seed=1)
    assert np.isclose(res["bound_product"], (5 / 8) ** n)
    assert abs(res["p_both"] - (5 / 8) ** n) < 0.06        # ~4 sigma at 600 draws
    assert res["p_both"] <= (3 / 4) ** n + 0.06
    assert res["p_at_least_one"] >= res["p_both"]
