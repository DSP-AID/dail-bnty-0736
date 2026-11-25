"""Uncloneable pad registers: encodings of a classical bit string into qubits
whose key cannot be learned by two isolated parties (Broadbent-Lord UE /
quantum-encyclopaedia-cloning-machine formality of arXiv:2607.28561).

Two instantiations, both of the form "apply a key-randomised Clifford to |a>":

  bb84()      key theta in {0,1}^m, encoding  H^{theta_j} |a_j>   (m qubits)
  clifford()  key = random Clifford circuit V on lambda qubits, encoding V|a>

Decryption is exact for both: the holder of the key undoes the randomisation
and reads the bits.  The point of the register is that a *single* copy can be
read correctly by one party, but not by two isolated parties (paper Thm. 3.4).
"""
from __future__ import annotations

import numpy as np

from .sim import QSim, H


def bb84_encode(a: list[int], theta: list[int]) -> QSim:
    """Statevector of the pad register for value `a` in bases `theta`."""
    m = len(a)
    sim = QSim(m)
    for j in range(m):
        if a[j]:
            sim.x(j)                    # |a_j>  (Z basis, bit j = LSB convention)
        if theta[j]:
            sim.h(j)                    # H^theta_j |a_j>
    return sim


def bb84_decode(sim: QSim, theta: list[int], rng) -> list[int]:
    """Key-holder read-out: measure qubit j in basis theta_j (destructive)."""
    return [int(sim.measure_basis(j, int(theta[j]), rng)) for j in range(len(theta))]


def random_clifford(m: int, rng, depth: int | None = None):
    """A random Clifford circuit as a gate list (H, S, CNOT layers)."""
    if depth is None:
        depth = 3 * m
    gates: list[tuple] = []
    for _ in range(depth):
        for q in range(m):
            r = int(rng.integers(3))
            if r == 1:
                gates.append(("h", q))
            elif r == 2:
                gates.append(("s", q))
        for q in range(m - 1):
            if rng.random() < 0.5:
                gates.append(("cx", q, q + 1))
    return gates


def apply_gate_list(sim: QSim, gates: list[tuple], inverse: bool = False) -> None:
    for g in reversed(gates) if inverse else gates:
        if g[0] == "h":
            sim.h(g[1])
        elif g[0] == "s":
            if inverse:
                sim.s(g[1])
                sim.z(g[1])                       # Z S = diag(1,-i) = S^{-1}
            else:
                sim.s(g[1])
        else:
            sim.cx(g[1], g[2])


def clifford_encode(a: list[int], gates: list[tuple]) -> QSim:
    m = len(a)
    sim = QSim(m)
    for j, b in enumerate(a):
        if b:
            sim.x(j)
    apply_gate_list(sim, gates)
    return sim


def clifford_decode(sim: QSim, gates: list[tuple], rng) -> list[int]:
    m = sim.n
    apply_gate_list(sim, gates, inverse=True)
    return [int(sim.measure_z(j, rng)) for j in range(m)]


def bb84(theta: list[int]):
    """Pad-register scheme object keyed by the basis string theta."""
    return {
        "name": "bb84",
        "key": list(theta),
        "m": len(theta),
        "encode": lambda a: bb84_encode(a, theta),
        "decode": lambda sim, rng: bb84_decode(sim, theta, rng),
    }


def clifford(gates: list[tuple], m: int):
    return {
        "name": "clifford",
        "key": gates,
        "m": m,
        "encode": lambda a: clifford_encode(a, gates),
        "decode": lambda sim, rng: clifford_decode(sim, gates, rng),
    }
