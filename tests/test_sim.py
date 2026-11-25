"""Statevector simulator: gates, measurements, Bell measurement."""
import numpy as np

from qfhe.sim import QSim, reduce_rho


def test_x_z_h_actions():
    s = QSim(1)
    s.x(0)
    assert np.allclose(np.abs(s.state[[1]]) ** 2, [1.0])
    s.z(0)  # phase only, |1> -> -|1>
    assert np.allclose(s.state, [0, -1])
    s.h(0)  # H|1> = |->
    assert np.allclose(np.abs(s.state) ** 2, [0.5, 0.5])


def test_ccx_truth_table():
    for a in (0, 1):
        for b in (0, 1):
            s = QSim(3)
            if a:
                s.x(0)
            if b:
                s.x(1)
            s.ccx(0, 1, 2)
            idx = (a << 2) | (b << 1) | (a & b)
            assert np.isclose(abs(s.state[idx]) ** 2, 1.0)


def test_measure_basis_recovers_encoded_bit():
    rng = np.random.default_rng(7)
    for trial in range(30):
        theta = [int(t) for t in rng.integers(0, 2, 4)]
        a = [int(t) for t in rng.integers(0, 2, 4)]
        s = QSim(4)
        for j in range(4):
            if a[j]:
                s.x(j)
            if theta[j]:
                s.h(j)
        got = [s.measure_basis(j, theta[j], rng) for j in range(4)]
        assert got == a  # exact: honest basis undoes honest encoding


def test_bell_measure_leaves_product_outcomes():
    """bell_measure returns (s,t); the post-measurement qubits hold |s>|s^t>,
    so re-measuring in Z reproduces them (qubits free for reuse)."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        s = QSim(4)
        s.h(0), s.h(1)  # arbitrary start
        s.cx(0, 2), s.h(2)  # entangle pairs so the Bell result is random
        st = s.bell_measure(0, 1, rng)
        b0 = s.measure_z(0, rng)
        b1 = s.measure_z(1, rng)
        assert st == (b0, b0 ^ b1)


def test_reduce_rho_maximally_mixed_single_qubit():
    """|psi> = (|0>|0> + |1>|1>)/sqrt2: qubit 1 reduced state is I/2."""
    s = QSim(2)
    s.h(0), s.cx(0, 1)
    rho = reduce_rho(s.state, [1])
    assert np.allclose(rho, np.eye(2) / 2)
