"""UP-QFHE: uncloneable-pad quantum fully homomorphic encryption.

A register of *uncloneable pads* (paper Def. 3.1) replaces the classical
secret of a one-time pad.  The payload register is always a computational-
basis state, so homomorphic evaluation is exact at any depth - no noise, no
bootstrapping, no LWE - and input confidentiality against the evaluator is
information-theoretic (uniform pad).  What an adversary cannot duplicate is
the *pad register*: two isolated parties holding halves of one ciphertext
cannot both decrypt it even after the basis key theta is revealed to both
(paper Thm. 3.4).

Scope (paper Sec. 5): classical messages, classical boolean circuits over
{NOT, AND} (the Broadbent-Jeffery setting), private key = theta; non-linear
gates are handled in transcript mode (Def. 3.3) where the evaluator releases
its padded classical transcript and the key holder recovers x, hence C(x).
For NOT-only circuits no pad access is needed: X on the payload qubit flips
the encrypted bit whatever the key basis of the pad qubit behind it (Lem. 3.6;
see tests/test_scheme.py::test_x_frame_flips_pad_bit_blindly).

Circuits are gate lists ("not", q) / ("and", c1, c2, t); "and" is w_t ^= w_c1 &
w_c2.  The target must differ from both controls so that the step is unitary
(w ^= u&w with t == a control is classically irreversible: (1,0),(1,1) collide).
"""
from __future__ import annotations

import numpy as np

from .pad import bb84, clifford, random_clifford
from .sim import QSim


# ---------------------------------------------------------------- circuits
def build_demo_circuit(n: int) -> list[tuple]:
    """Non-trivial circuit over every wire with distinct gate targets: a NOT,
    an AND/NOT ripple of depth n, closing with a final NOT."""
    if n < 3:
        raise ValueError("demo circuit needs n >= 3")
    g: list[tuple] = [("not", 0)]
    for j in range(1, n - 1):
        g.append(("and", 0, j, j + 1))
    g.append(("not", n - 1))
    return g


def eval_classical(gates: list[tuple], p: list[int]) -> list[int]:
    """The evaluator's pass over the *padded* payload.  Because the payload
    is always a Z-basis state it can be measured freely, so unbounded-depth
    homomorphic evaluation is exactly this classical simulation."""
    w = list(p)
    for g in gates:
        if g[0] == "not":
            w[g[1]] ^= 1
        elif g[0] == "and":
            _, a, b, t = g
            if t in (a, b):
                raise ValueError("and: target must differ from controls")
            w[t] ^= w[a] & w[b]
        else:
            raise ValueError(g[0])
    return w


def eval_quantum(gates: list[tuple], sim: QSim) -> None:
    """Same circuit applied physically to the payload qubits."""
    for g in gates:
        if g[0] == "not":
            sim.x(g[1])
        elif g[0] == "and":
            _, a, b, t = g
            if t in (a, b):
                raise ValueError("and: target must differ from controls")
            if a == b:
                sim.cx(a, t)             # w_t ^= w_a & w_a == w_a
            else:
                sim.ccx(a, b, t)
        else:
            raise ValueError(g[0])


# ------------------------------------------------------------------ scheme
class UPQFHE:
    """Uncloneable-pad QFHE, private-key setting; key = theta (bb84 pads).

    `simulate=True` also carries the payload and pad registers as statevectors
    (small n) so that physical evaluation can be checked against the
    transcript; `simulate=False` scales to any n and depth.
    """

    def __init__(self, n: int, pad: str = "bb84", rng=None, simulate: bool = True):
        self.n, self.pad_name, self.simulate = n, pad, simulate
        self.rng = np.random.default_rng() if rng is None else rng
        if pad == "bb84":
            self.theta = [int(b) for b in self.rng.integers(0, 2, n)]
            self.pad = bb84(self.theta)
        elif pad == "clifford":
            self.gates = random_clifford(n, self.rng)
            self.pad = clifford(self.gates, n)
        else:
            raise ValueError(pad)

    def encrypt(self, x: list[int]) -> dict:
        """Enc(x): uniform pad a; payload |x^a> (Z basis); pad register holds
        the uncloneable encoding of a."""
        a = [int(b) for b in self.rng.integers(0, 2, self.n)]
        p = [u ^ v for u, v in zip(x, a)]
        ct = {"p_in": p, "p_out": None, "a": a if not self.simulate else None,
              "out": None}
        if self.simulate:
            pay = QSim(self.n)
            for j, b in enumerate(p):
                if b:
                    pay.x(j)
            ct["payload"], ct["mask"] = pay, self.pad["encode"](a)
        return ct

    def evaluate(self, ct: dict, gates: list[tuple]) -> dict:
        """Homomorphic evaluation (Def. 3.3): physical circuit on the payload
        plus the padded transcript p_out = C(p_in).  The pad register is never
        touched, and neither theta nor x is ever available to the evaluator."""
        if self.simulate:
            eval_quantum(gates, ct["payload"])
        return {"p_in": ct["p_in"], "p_out": eval_classical(gates, ct["p_in"]),
                "a": ct["a"], "out": gates,
                **{k: ct[k] for k in ("payload", "mask") if k in ct}}

    def decrypt(self, ct: dict) -> list[int]:
        """Dec: read the pad register in the key basis to obtain a, then
        x = p_in ^ a and w = C(x)."""
        a = self.pad["decode"](ct["mask"], self.rng) if self.simulate else ct["a"]
        x = [u ^ v for u, v in zip(ct["p_in"], a)]
        return x if ct["out"] is None else eval_classical(ct["out"], x)

    def payload_bits(self, ct: dict, rng=None) -> list[int]:
        """Read the payload register (free for the evaluator: Z basis).  Equal
        to p_in before evaluation and p_out after it."""
        if not self.simulate:
            return list(ct["p_in"])
        return [int(ct["payload"].measure_z(j, rng or self.rng))
                for j in range(self.n)]


def self_check(seed: int = 0, n: int = 6) -> bool:
    """Encrypt / evaluate / decrypt round trip on a non-trivial circuit."""
    rng = np.random.default_rng(seed)
    sq = UPQFHE(n, rng=rng)
    x = [int(b) for b in rng.integers(0, 2, n)]
    gates = build_demo_circuit(n)
    ct = sq.evaluate(sq.encrypt(x), gates)
    return sq.decrypt(ct) == eval_classical(gates, x)
