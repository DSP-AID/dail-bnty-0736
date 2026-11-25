"""Splitting-game values behind the uncloneability bound -> results/games.json
and results/splitting_game.png.

Run from the repo root (src/):  python3 scripts/run_games.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run from anywhere

import numpy as np

from qfhe.games import (channel_splitting_value, channel_value, dual_certificate,
                        naive_values, optimal_channel, optimal_split_value,
                        scheme_splitting_game, sdp_dual_bound)


def main():
    spec_m, spec_gap = dual_certificate()
    chan_val, kraus = channel_splitting_value(restarts=20, seed=2)
    exact_kraus = optimal_channel()
    iso_val, _ = optimal_split_value(restarts=16, seed=0)
    bound, slack = sdp_dual_bound(restarts=12, seed=0)

    out = {
        "one_bit_game": {
            "naive": naive_values(),
            "best_isometry_lower_bound": iso_val,
            "channel_optimal_value": chan_val,
            "exact_channel_value": channel_value(exact_kraus),
            "exact_channel_kraus": exact_kraus.real.tolist(),
            "dual_certificate_tr_Y": 3 / 4,
            "sdp_dual_search_bound": bound,
            "sdp_dual_slack": slack,
            "objective_spectrum": [float(x) for x in spec_m],
            "gap_spectrum_3over8_minus_M": [float(x) for x in spec_gap],
            "kraus_tp_violation": float(np.abs(
                sum(k.conj().T @ k for k in kraus) - np.eye(2)).max()),
        },
        "n_bit_game": {},
    }

    ns, trials = (3, 4, 5), 4000
    for n in ns:
        r = scheme_splitting_game(n, trials, seed=1)
        out["n_bit_game"][str(n)] = r

    dest = Path(__file__).resolve().parent.parent / "results"
    dest.mkdir(exist_ok=True)
    (dest / "games.json").write_text(json.dumps(out, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nn = np.arange(1, 9)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.semilogy(nn, (5 / 8) ** nn, "b-", label=r"measure-Z attack $(5/8)^n$")
    ax.semilogy(nn, (3 / 4) ** nn, "r--",
                label=r"certificate bound $(3/4)^n$ (any attack)")
    xs = sorted(int(k) for k in out["n_bit_game"])
    pb = [out["n_bit_game"][str(n)]["p_both"] for n in xs]
    ax.semilogy(xs, pb, "ko", label="measured $p_{both}$ (full scheme)")
    ax.set_xlabel("pad bits n")
    ax.set_ylabel(r"$p_{both}$ (log scale)")
    ax.set_title("Key-revealing splitting game: both challengers decrypt")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(dest / "splitting_game.png", dpi=140)

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
