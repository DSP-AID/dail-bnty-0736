"""Uncloneability games for the pad register (paper Sec. 4).

Key-revealing splitting game G_key (Def. 4.1): a string `a` is encoded in the
BB84 pad register with key theta; the adversary holds ONE copy, splits it
into systems A and B by any quantum operation; both decrypters are then shown
theta and must output a.  This module:

* `p_win_isometry` / `optimal_split_value`: one-bit game value restricted to
  isometries C^2 -> C^2 (x) C^2 (best found ~0.7286, a lower bound on the
  true value);
* `channel_splitting_value`: one-bit value over ALL CPTP splitting channels
  (Kraus rank 2 suffices): exactly 3/4, attained by the explicit channel
  `optimal_channel` (measure-Z-and-reprepare only reaches 5/8);
* `sdp_dual_bound` / `dual_certificate`: SDP upper bound over all attacks -
  the numeric search reaches 0.75, and the analytic certificate Y = (3/8) I_2
  (gap matrix PSD, see `dual_certificate`) gives Tr Y = 3/4, i.e. p*_1 <= 3/4,
  and by tensor power p*_n <= (3/4)^n for arbitrary joint attacks;
* `scheme_splitting_game`: end-to-end n-bit game, encrypt -> split -> both
  decrypt, versus the product bound value^n.

All numbers are computed by this code; none are hard-coded from the literature.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .pad import bb84_encode
from .sim import QSim, H

BB84 = [np.array([1, 0], complex), np.array([0, 1], complex),
        np.array([1, 1], complex) / np.sqrt(2), np.array([1, -1], complex) / np.sqrt(2)]


# ------------------------------------------------------------ one-bit game
def p_win_isometry(V: np.ndarray) -> float:
    """Exact winning probability of the one-bit key-revealing splitting game
    for a 1->2 isometry V (4x2): p = (1/4) sum_u |<uu| V |u>|^2.  Also exact
    for the Kraus-rank-1 channels used as naive baselines below."""
    assert V.shape == (4, 2)
    tot = 0.0
    for u in BB84:
        clone = np.kron(u, u)
        tot += abs(np.vdot(clone, V @ u)) ** 2
    return float(tot / 4)


def _qr_isometry(z: np.ndarray) -> np.ndarray:
    M = (z[:8] + 1j * z[8:]).reshape(4, 2)
    Q, R = np.linalg.qr(M)
    return Q * np.sign(np.diag(R))[None, :]


def optimal_split_value(restarts: int = 40, seed: int = 0) -> tuple[float, np.ndarray]:
    """Best value found over one-bit splitting ISOMETRIES; general channels do
    strictly better (`channel_splitting_value`), so this is a lower bound."""
    rng = np.random.default_rng(seed)
    best, bestV = -1.0, None
    for _ in range(restarts):
        z0 = rng.normal(size=16)
        r = minimize(lambda z: -p_win_isometry(_qr_isometry(z)), z0,
                     method="Nelder-Mead",
                     options={"maxiter": 6000, "xatol": 1e-10, "fatol": 1e-13})
        if -r.fun > best:
            best = -r.fun
            bestV = _qr_isometry(r.x)
    return float(best), bestV


def _objective_operator() -> np.ndarray:
    """M with p_win = Tr[M J] for the Choi matrix J of a channel C^2 -> C^4
    (all BB84 states are real, so no transposes are needed)."""
    M = np.zeros((8, 8), complex)
    for u in BB84:
        M += np.kron(np.outer(u, u.conj()),
                     np.outer(np.kron(u, u), np.kron(u, u).conj()))
    return M / 4


def sdp_dual_bound(restarts: int = 12, seed: int = 0) -> tuple[float, float]:
    """Dual SDP  min Tr Y  s.t.  Y (x) I_4 >= M  (Y Hermitian 2x2).

    Returns (certified_bound, slack): any feasible Y upper-bounds every
    quantum attack; we report Tr Y + 2*max(0, -lambda_min(Y (x) I - M)), a
    valid bound even for an only-epsilons-feet feasible Y.  Strong duality
    holds (Slater), so this equals the primal optimum over all attacks."""
    M = _objective_operator()

    def worst(y: np.ndarray) -> float:
        Y = np.array([[y[0], y[1] + 1j * y[2]], [y[1] - 1j * y[2], y[3]]], complex)
        return float(np.linalg.eigvalsh(np.kron(Y, np.eye(4)) - M).min())

    rng = np.random.default_rng(seed)
    cons = [{"type": "ineq", "fun": worst}]   # concave lambda_min >= 0
    best = (9.0, None)
    for _ in range(restarts):
        y0 = np.array([1.0, 0.0, 0.0, 1.0]) + 0.3 * rng.normal(size=4)
        r = minimize(lambda y: y[0] + y[3], y0, method="SLSQP",
                     constraints=cons,
                     options={"maxiter": 2000, "ftol": 1e-12})
        pen = float(r.x[0] + r.x[3] + 2.0 * max(0.0, -worst(r.x)))
        if pen < best[0]:
            best = (pen, r.x.copy())
    y = best[1]
    return best[0], -worst(y)


def channel_splitting_value(restarts: int = 40, seed: int = 2):
    """One-bit game value over ALL CPTP splitting channels, parametrised by
    Kraus operators A1, A2 (rank 2 suffices for qubit -> two qubits) with the
    trace-preservation constraint A1^H A1 + A2^H A2 = I imposed as an equality
    constraint.  Returns (value, stacked Kraus pair of shape (2, 4, 2))."""

    def mats(z):
        return ((z[:8] + 1j * z[8:16]).reshape(4, 2),
                (z[16:24] + 1j * z[24:32]).reshape(4, 2))

    def val(z):
        X, Y = mats(z)
        return float(np.mean([abs(np.vdot(np.kron(u, u), X @ u)) ** 2
                              + abs(np.vdot(np.kron(u, u), Y @ u)) ** 2
                              for u in BB84]))

    def tp(z):
        X, Y = mats(z)
        G = X.conj().T @ X + Y.conj().T @ Y - np.eye(2)
        return np.array([G[0, 0].real, G[1, 1].real, G[0, 1].real, G[0, 1].imag])

    rng = np.random.default_rng(seed)
    best, bz = -1.0, None
    for _ in range(restarts):
        r = minimize(lambda z: -val(z), rng.normal(size=32) * 0.5, method="SLSQP",
                     constraints=[{"type": "eq", "fun": tp}],
                     options={"maxiter": 800, "ftol": 1e-12})
        if getattr(r, "x", None) is None or abs(tp(r.x)).max() > 1e-7:
            continue
        if val(r.x) > best:
            best, bz = val(r.x), r.x
    return float(best), np.stack(mats(bz))


def optimal_channel() -> np.ndarray:
    """Exact optimal splitting channel Lambda*: Kraus operators A1, A2 (stacked,
    shape (2, 4, 2)) whose columns are the two kernel basis vectors of the gap
    matrix (3/8) I - M reshaped.  A1^H A1 + A2^H A2 = I and the game value is
    exactly 3/4, meeting the dual certificate Tr Y = 3/4."""
    s = 1.0 / np.sqrt(12.0)
    A1 = s * np.array([[3, 0], [0, 1], [0, 1], [1, 0]], dtype=complex)
    A2 = s * np.array([[0, 1], [1, 0], [1, 0], [0, 3]], dtype=complex)
    return np.stack([A1, A2])


def channel_value(kraus: np.ndarray) -> float:
    """Game value of a channel given as a stack of Kraus operators."""
    return float(np.mean([sum(abs(np.vdot(np.kron(u, u), A @ u)) ** 2
                              for A in kraus) for u in BB84]))


def dual_certificate() -> tuple[np.ndarray, np.ndarray]:
    """Analytic dual feasible point Y = (3/8) I_2.  Returns the spectra of the
    objective operator M and of the gap (3/8) I_8 - M; both are nonnegative
    (observed: M in {0,0,0,0,1/8,1/8,3/8,3/8}), which certifies p*_1 <= Tr Y
    = 3/4.  Since the n-bit objective is M^{tensor n}, the tensor-power point
    Y_n = (3/8)^n I_{2^n} is dual-feasible for the n-bit game with
    Tr Y_n = (3/4)^n, giving p*_n <= (3/4)^n for every joint splitting attack."""
    M = _objective_operator()
    return (np.linalg.eigvalsh((M + M.conj().T) / 2),
            np.linalg.eigvalsh((3 / 8) * np.eye(8) - (M + M.conj().T) / 2))


def p_win_fixed(phi: np.ndarray) -> float:
    """Winning probability when both clones are prepared in a fixed (possibly
    entangled) two-qubit state `phi`, ignoring the input entirely."""
    return float(np.mean([abs(np.vdot(np.kron(u, u), phi)) ** 2 for u in BB84]))


def naive_values() -> dict:
    """Baselines for the one-bit game (paper Table 1)."""
    out = {}
    V = np.zeros((4, 2), complex)
    V[0, 0] = 1.0                     # coherent Z copy: |0> -> |00>, |1> -> |11>
    V[3, 1] = 1.0
    out["z_copy"] = p_win_isometry(V)
    out["x_copy"] = p_win_isometry(np.kron(H, H) @ V @ H)
    f00 = np.zeros(4, complex); f00[0] = 1.0          # ignore input, output |00>
    out["fixed_00"] = p_win_fixed(f00)
    fpp = np.full(4, 0.5, complex)                    # ... or |++>
    out["fixed_plus"] = p_win_fixed(fpp)
    return out


# ------------------------------------------------------------------ n bits
def product_value(p1: float, n: int) -> float:
    return float(p1 ** n)


def complete_isometry(V: np.ndarray) -> np.ndarray:
    """4x2 isometry -> 4x4 unitary W with W[:, 2k] = V[:, k]: the op index of
    split_register is 2*q_bit + anc_bit and the ancilla starts blank, so input
    bit q selects column 2q.  Columns 1,3 complete W to a unitary with an
    orthonormal basis of ker(V^H)."""
    W = np.zeros((4, 4), complex)
    W[:, 0], W[:, 2] = V[:, 0], V[:, 1]
    null = np.linalg.svd(V.conj().T)[2][2:]
    for k, row in enumerate(null):
        W[:, 1 + 2 * k] = row.conj()
    return W


def split_register(sim: QSim, qubits: list[int], V: np.ndarray) -> tuple[QSim, list[tuple[int, int]]]:
    """Apply the 1->2 isometry `V` to each of `qubits` of `sim`, consuming one
    fresh blank qubit per input.  Returns the enlarged simulator and the clone
    pairs [(half0_j, half1_j)]."""
    n = sim.n
    m = len(qubits)
    out = QSim(n + m)
    out.state[:: 2 ** m] = sim.state      # appended ancillas are least significant
    W = complete_isometry(V)
    pairs = []
    for j, q in enumerate(qubits):
        out.apply(W, q, n + j)
        pairs.append((q, n + j))
    return out, pairs


def scheme_splitting_game(n: int, trials: int, seed: int = 1,
                          V: np.ndarray | None = None) -> dict:
    """End-to-end game: encrypt x, split the pad register qubit-by-qubit with
    `V`, reveal theta and the transcript to both challengers; both must output
    x (they know p_in, so each guesses x_hat_j = p_j ^ a_hat_j).  Per-bit both-
    correct probability is exactly the one-bit game value, hence for this qubit-
    local attack p_both = p_win(V)^n; the dual certificate bounds ANY attack by
    (3/4)^n."""
    from .scheme import UPQFHE, build_demo_circuit

    rng = np.random.default_rng(seed)
    if V is None:
        V = np.zeros((4, 2), complex)              # measure Z, resend |b>|b>
        V[0, 0] = V[3, 1] = 1.0
    gates = build_demo_circuit(n)
    both = one = 0
    for _ in range(trials):
        sq = UPQFHE(n, rng=rng)
        x = [int(b) for b in rng.integers(0, 2, n)]
        ct = sq.evaluate(sq.encrypt(x), gates)
        big, pairs = split_register(ct["mask"], list(range(n)), V)
        outs = []
        for side in (0, 1):
            a_hat = [int(big.measure_basis(pairs[j][side], sq.theta[j], rng))
                     for j in range(n)]
            x_hat = [u ^ v for u, v in zip(ct["p_in"], a_hat)]
            outs.append(x_hat == x)
        both += int(outs[0] and outs[1])
        one += int(outs[0] or outs[1])
    return {"n": n, "trials": trials, "p_both": both / trials,
            "p_at_least_one": one / trials,
            "bound_product": product_value(p_win_isometry(V), n)}
