"""Pad registers: key-holder decode is exact for both encodings."""
import numpy as np

from qfhe.pad import bb84, clifford


def test_bb84_decode_exact():
    rng = np.random.default_rng(11)
    pad = bb84([int(b) for b in rng.integers(0, 2, 6)])
    for _ in range(30):
        a = [int(b) for b in rng.integers(0, 2, 6)]
        assert pad["decode"](pad["encode"](a), rng) == a


def test_clifford_decode_exact():
    rng = np.random.default_rng(12)
    gates = [[("h", 0), ("cx", 0, 1), ("s", 1), ("cx", 1, 2), ("h", 2)],
             [("cx", 0, 1), ("cx", 1, 2), ("s", 0), ("h", 1)]]
    for g in gates:
        pad = clifford(g, 3)
        for _ in range(10):
            a = [int(b) for b in rng.integers(0, 2, 3)]
            assert pad["decode"](pad["encode"](a), rng) == a


def test_random_clifford_decode_exact():
    from qfhe.pad import random_clifford
    rng = np.random.default_rng(13)
    for m in (2, 3, 4):
        pad = clifford(random_clifford(m, rng), m)
        for _ in range(5):
            a = [int(b) for b in rng.integers(0, 2, m)]
            assert pad["decode"](pad["encode"](a), rng) == a
