"""Minimal dense state-vector simulator for the UP-QFHE scheme.

Supports exactly what the protocol needs: X, Z, H, S, CNOT, CCX, computational
-basis measurement and Bell-basis measurement with mid-circuit collapse.
Qubit 0 is the most significant bit.  n <= ~16 is plenty for the simulation
mode of qfhe/scheme.py (transcript mode scales past that without qubits).
"""
from __future__ import annotations

import numpy as np

X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
CNOT = np.eye(4, dtype=complex)
CNOT[2:, 2:] = X
CCX = np.eye(8, dtype=complex)
CCX[6:, 6:] = X


def _apply(state: np.ndarray, op: np.ndarray, qubits: tuple[int, ...]) -> np.ndarray:
    """Apply a 2^k x 2^k operator acting on `qubits` (big-endian within op)."""
    n = state_ndim(state)
    state = state.reshape([2] * n)
    k = len(qubits)
    op = op.reshape(*([2] * k), *([2] * k))
    moved = np.moveaxis(state, list(qubits), list(range(k)))
    out = np.tensordot(op, moved, axes=(list(range(k, 2 * k)), list(range(k))))
    return np.moveaxis(out, list(range(k)), list(qubits)).reshape(-1)


def state_ndim(state: np.ndarray) -> int:
    return int(np.log2(len(state)))


def reduce_rho(state: np.ndarray, keep: list[int]) -> np.ndarray:
    """Reduced density matrix over `keep` (big-endian order)."""
    n = state_ndim(state)
    rho = np.outer(state, state.conj()).reshape([2] * (2 * n))
    for q in sorted((q for q in range(n) if q not in keep), reverse=True):
        k = rho.ndim // 2
        rho = np.trace(rho, axis1=q, axis2=q + k)
    d = 2 ** len(keep)
    return rho.reshape(d, d)


class QSim:
    def __init__(self, n: int):
        self.n = n
        self.state = np.zeros(2 ** n, dtype=complex)
        self.state[0] = 1.0

    def apply(self, op: np.ndarray, *qubits: int) -> None:
        self.state = _apply(self.state, op, qubits)

    def x(self, q): self.apply(X, q)
    def z(self, q): self.apply(Z, q)
    def h(self, q): self.apply(H, q)
    def s(self, q): self.apply(S, q)
    def cx(self, c, t): self.apply(CNOT, c, t)
    def ccx(self, c1, c2, t): self.apply(CCX, c1, c2, t)

    def _measure_bits(self, qs: list[int], basis_change=None, rng=None) -> tuple[int, ...]:
        """Joint Z-basis measurement of qubits `qs` (after optional single-
        qubit `basis_change` dict {q: op}); collapse and return bit tuple."""
        if basis_change:
            for q, op in basis_change.items():
                self.apply(op, q)
        amps = self.state.reshape([2] * self.n)
        p = np.ones([2] * len(qs))
        for bits in np.ndindex(*[2] * len(qs)):
            sl = [slice(None)] * self.n
            for q, b in zip(qs, bits):
                sl[q] = b
            sub = amps[tuple(sl)]
            p[bits] = float(np.vdot(sub, sub).real)
        flat = np.clip(p.flatten(), 0.0, None)
        flat /= flat.sum()
        bits = np.unravel_index(int(rng.choice(len(flat), p=flat)), p.shape)
        new = np.zeros_like(amps)
        sl = [slice(None)] * self.n
        for q, b in zip(qs, bits):
            sl[q] = int(b)
        new[tuple(sl)] = amps[tuple(sl)]
        self.state = new.reshape(-1) / np.linalg.norm(new)
        return tuple(int(b) for b in bits)

    def measure_z(self, q: int, rng) -> int:
        return self._measure_bits([q], rng=rng)[0]

    def measure_basis(self, q: int, basis: int, rng) -> int:
        """Measure qubit q in basis 0 (Z) or 1 (X), return the bit r such that
        the outcome state is H^basis|r>."""
        b = self._measure_bits([q], basis_change={q: H} if basis else None, rng=rng)
        return b[0]

    def bell_measure(self, q1: int, q2: int, rng) -> tuple[int, int]:
        """Bell measurement of (q1,q2). Returns (s,t): the projected branch is
        (I (x) X^s Z^t)|Phi+>.  Undoing the basis change on the classical
        outcomes leaves both qubits in product states |s>|s^t>, so they are
        free for reuse."""
        self.cx(q1, q2)
        self.h(q1)
        b1, b2 = self._measure_bits([q1, q2], rng=rng)
        return b1, b1 ^ b2
