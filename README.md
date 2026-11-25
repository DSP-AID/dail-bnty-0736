# UP-QFHE — uncloneable-pad quantum fully homomorphic encryption

DAiL bounty `bnty_0736`: *uncloneable quantum FHE — construction or
polynomial-time break.* This repo contains the construction (ARM A), its
proof, and the code that computes every number quoted in the paper.

**Idea (one paragraph).** A one-time pad's secret bit is stored not in a
classical string but in an *uncloneable BB84 register*: a qubit prepared in a
random conjugate basis whose basis key `theta` stays private. The payload is
always a computational-basis state, so homomorphic evaluation of classical
circuits over {NOT, AND} is **exact at any depth** — no noise, no
bootstrapping, no LWE. Input confidentiality against the evaluator is
information-theoretic (uniform pad). What an adversary cannot duplicate is the
*pad register*: after `theta` is revealed, two isolated parties holding halves
of one ciphertext cannot both recover the message better than `(3/4)^n` — and
`3/4` is proven to be the exact value of the one-bit splitting game (SDP dual
certificate `Y = 3/8 * I`).

- `paper/main.tex` — the paper: definitions, security theorem, exact game
  value `p*_1 = 3/4`, `(3/4)^n` tensor-power bound, appendix with the analytic
  dual certificate.
- `qfhe/` — implementation: `scheme.py` (UP-QFHE), `pad.py` (BB84 + Clifford
  uncloneable pads), `sim.py` (dense statevector simulator), `games.py`
  (splitting-game numerics: isometry search, SDP, dual certificate).
- `scripts/run_scheme.py` -> `results/scheme.json` — correctness (20/20 round
  trips, 60/60 random circuits cross-checked against physical simulation),
  scaling to n = 10^4, transcript-confidentiality sampling.
- `scripts/run_games.py` -> `results/games.json`, `results/splitting_game.png`
  — game values: naive Z-copy 5/8, best isometry ~0.7286, optimal CPTP channel
  = 3/4 (exact), SDP dual bound 3/4, n-bit games match `(3/4)^n`.
- `tests/` — 30 pytest tests covering correctness, confidentiality, and every
  game constant quoted in the paper.

## Run it

No API keys, no network. Requires Python 3.11+ with `numpy`, `scipy`,
`sympy`, `matplotlib`, `pytest`.

```bash
python -m pytest tests/ -q          # 30 tests
python scripts/run_scheme.py        # regenerates results/scheme.json
python scripts/run_games.py         # regenerates results/games.json + png
```

## Scope (stated in the paper, Sec. "Scope")

Classical messages and classical circuits over {NOT, AND} (the
Broadbent–Jeffery setting); private key = `theta`; non-linear gates run in
transcript mode, where the evaluator releases a padded classical transcript
that is information-theoretically uniform and the key holder recovers both the
input and the output. Multi-bit security `(3/4)^n` holds for arbitrary joint
attacks on `n` pads (tensor-power of the dual certificate).
