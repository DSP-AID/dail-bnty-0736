"""Correctness + scaling evidence for UP-QFHE -> results/scheme.json.

Run from the repo root (src/):  python3 scripts/run_scheme.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

import numpy as np

from qfhe.scheme import (UPQFHE, build_demo_circuit, eval_classical,
                         eval_quantum)
from qfhe.sim import QSim

RNG = np.random.default_rng(2026)


def random_circuit(n, rng, depth=12):
    gates, free = [], list(range(n))
    for _ in range(depth):
        if rng.random() < 0.4:
            gates.append(("not", int(rng.integers(n))))
        elif len(free) >= 3:
            a, b, t = rng.choice(free, 3, replace=False)
            gates.append(("and", int(a), int(b), int(t)))
            free.remove(int(t))
    return gates


def main():
    out = {}

    # 1. self-check round trips on the demo circuit (simulated, both pads)
    res = []
    for seed in range(10):
        for pad in ("bb84", "clifford"):
            rng = np.random.default_rng(seed)
            sq = UPQFHE(6, pad=pad, rng=rng)
            x = [int(b) for b in rng.integers(0, 2, 6)]
            gates = build_demo_circuit(6)
            ct = sq.evaluate(sq.encrypt(x), gates)
            res.append(sq.decrypt(ct) == eval_classical(gates, x))
    out["self_check"] = {"trials": len(res), "all_pass": all(res)}

    # 2. random-circuit correctness, physical simulation vs transcript
    res = []
    for n in (3, 4, 5):
        rng = np.random.default_rng(100 + n)
        sq = UPQFHE(n, rng=rng)
        for _ in range(20):
            x = [int(b) for b in rng.integers(0, 2, n)]
            gates = random_circuit(n, rng)
            ct = sq.evaluate(sq.encrypt(x), gates)
            ok_dec = sq.decrypt(ct) == eval_classical(gates, x)
            # independent physical check: run the circuit on a bare QSim
            sim = QSim(n)
            for j, b in enumerate(x):
                if b:
                    sim.x(j)
            eval_quantum(gates, sim)
            bits = [int(sim.measure_z(j, rng)) for j in range(n)]
            res.append(ok_dec and bits == eval_classical(gates, x))
    out["random_circuit_correctness"] = {"trials": len(res), "all_pass": all(res)}

    # 3. scaling: transcript mode, depth-n ripple circuit, no quantum sim
    rows = []
    for n in (10, 100, 1000, 10000):
        rng = np.random.default_rng(7 + n)
        sq = UPQFHE(n, simulate=False, rng=rng)
        x = [int(b) for b in rng.integers(0, 2, n)]
        gates = build_demo_circuit(n)
        t0 = time.perf_counter()
        ct = sq.evaluate(sq.encrypt(x), gates)
        w = sq.decrypt(ct)
        rows.append({"n": n, "gates": len(gates),
                     "eval_decrypt_seconds": round(time.perf_counter() - t0, 6),
                     "correct": w == eval_classical(gates, x)})
    out["scaling"] = rows

    # 4. input confidentiality: p_in uniform and x-independent (n=6, 4000 draws)
    n, trials = 6, 4000
    sq = UPQFHE(n, simulate=False, rng=np.random.default_rng(9))
    freqs = {}
    for x in ([0] * n, [1] * n):
        cnt = {}
        for _ in range(trials):
            key = tuple(sq.encrypt(x)["p_in"])
            cnt[key] = cnt.get(key, 0) + 1
        freqs[str(x)] = np.array([cnt.get(t, 0) / trials
                                  for t in np.ndindex(*[2] * n)])
    out["confidentiality"] = {
        "n": n, "trials": trials,
        "max_dev_from_uniform": float(np.abs(freqs["[0, 0, 0, 0, 0, 0]"] - 1 / 64).max()),
        "max_dev_x0_vs_x1": float(np.abs(freqs["[0, 0, 0, 0, 0, 0]"]
                                         - freqs["[1, 1, 1, 1, 1, 1]"]).max())}

    dest = Path(__file__).resolve().parent.parent / "results"
    dest.mkdir(exist_ok=True)
    (dest / "scheme.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
