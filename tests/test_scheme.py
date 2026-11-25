"""UP-QFHE end-to-end: correctness, blind evaluation, confidentiality."""
import numpy as np
import pytest

from qfhe.scheme import (UPQFHE, build_demo_circuit, eval_classical,
                         eval_quantum, self_check)


def random_circuit(n, rng, depth=8):
    """Random {NOT, AND} circuit with fresh target wires per AND."""
    gates, free = [], list(range(n))
    for _ in range(depth):
        if rng.random() < 0.4:
            gates.append(("not", int(rng.integers(n))))
        elif len(free) >= 3:
            a, b, t = rng.choice(free, 3, replace=False)
            gates.append(("and", int(a), int(b), int(t)))
            free.remove(int(t))          # target becomes computed wire
    return gates


@pytest.mark.parametrize("n", [3, 4, 5, 6])
@pytest.mark.parametrize("pad", ["bb84", "clifford"])
def test_round_trip_simulate(n, pad):
    rng = np.random.default_rng(100 + n)
    sq = UPQFHE(n, pad=pad, rng=rng)
    for _ in range(5):
        x = [int(b) for b in rng.integers(0, 2, n)]
        gates = build_demo_circuit(n) if pad == "bb84" else random_circuit(n, rng)
        ct = sq.evaluate(sq.encrypt(x), gates)
        assert sq.decrypt(ct) == eval_classical(gates, x)


def test_round_trip_no_simulation_large():
    rng = np.random.default_rng(77)
    n = 60
    sq = UPQFHE(n, simulate=False, rng=rng)
    x = [int(b) for b in rng.integers(0, 2, n)]
    gates = build_demo_circuit(n)                       # depth-n ripple
    ct = sq.evaluate(sq.encrypt(x), gates)
    assert sq.decrypt(ct) == eval_classical(gates, x)


def test_x_frame_flips_pad_bit_blindly():
    """Lem. 3.6: X on a payload qubit flips the encrypted bit without any
    access to the pad register, whatever the key basis."""
    rng = np.random.default_rng(5)
    n = 5
    sq = UPQFHE(n, rng=rng)
    x = [int(b) for b in rng.integers(0, 2, n)]
    ct = sq.encrypt(x)
    p_in = list(ct["p_in"])
    gates = [("not", j) for j in range(n)]              # NOT-only: no pad access
    ct2 = sq.evaluate(ct, gates)
    assert ct2["mask"] is ct["mask"]                    # pad never touched
    assert sq.decrypt(ct2) == [b ^ 1 for b in x]
    # physical check: payload now reads p_in ^ 1 while the pad still decodes a
    assert sq.payload_bits(ct2, rng) == [b ^ 1 for b in p_in]


def test_payload_bits_match_transcript():
    rng = np.random.default_rng(6)
    n = 4
    sq = UPQFHE(n, rng=rng)
    x = [int(b) for b in rng.integers(0, 2, n)]
    gates = build_demo_circuit(n)
    ct = sq.evaluate(sq.encrypt(x), gates)
    assert sq.payload_bits(ct, rng) == ct["p_out"]      # evaluator reads p_out


def test_and_target_must_differ():
    with pytest.raises(ValueError):
        eval_classical([("and", 0, 1, 0)], [1, 1, 0])
    with pytest.raises(ValueError):
        from qfhe.sim import QSim
        eval_quantum([("and", 0, 1, 1)], QSim(3))


def test_p_in_uniform_independent_of_x():
    """Information-theoretic input confidentiality: the payload p = x ^ a with
    uniform a is uniform, so its distribution is identical for all x."""
    rng = np.random.default_rng(9)
    n, trials = 6, 2000
    sq = UPQFHE(n, simulate=False, rng=rng)
    dists = []
    for x in ([0] * n, [1] * n):
        cnt = {}
        for _ in range(trials):
            key = tuple(sq.encrypt(x)["p_in"])
            cnt[key] = cnt.get(key, 0) + 1
        dists.append(np.array([cnt.get(tuple(idx), 0) / trials
                               for idx in np.ndindex(*[2] * n)]))
    assert np.abs(dists[0] - 1 / 2 ** n).max() < 0.03   # ~uniform
    assert np.abs(dists[0] - dists[1]).max() < 0.03     # x-independent


def test_self_check():
    for seed in range(4):
        assert self_check(seed=seed)
