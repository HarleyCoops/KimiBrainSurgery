"""paired_eval — paired per-seed evaluation of original vs repaired weights.

Both policies play the SAME held-out games (seeds 20000-20063, greedy, never used for
mining or gating). Per-seed returns are saved so the comparison is reproducible and the
pairing is preserved. Uncertainty: bootstrap CI on the per-seed paired difference,
plus a sign test.

Writes eval/paired_heldout.json and prints a summary.
"""
import json
from pathlib import Path

import numpy as np

import failure_gym as g

ROOT = Path(__file__).parent
SEEDS = list(range(20000, 21000))  # untouched held-out set, 1000 games
BOOTSTRAP_N = 100_000
RNG_SEED = 42


def per_seed_returns(weights, seeds):
    pol = g.Policy({k: weights[k].astype(np.float32) for k in weights})
    out = {}
    for s in seeds:
        game = g.Breakout(200)
        game.reset(int(s))
        while not (game.terminated or game.truncated):
            game.step(pol.act(game.observation(), None, "greedy"))
        out[int(s)] = int(game.total)
    return out


def main():
    base = {k: v.astype(np.float64) for k, v in g.BASE_WEIGHTS.items()}
    rep = {k: v.astype(np.float64) for k, v in np.load(ROOT / "best_weights.npz").items()}

    base_scores = per_seed_returns(base, SEEDS)
    rep_scores = per_seed_returns(rep, SEEDS)

    diffs = np.array([rep_scores[s] - base_scores[s] for s in SEEDS], dtype=float)
    n = len(diffs)
    mean_diff = float(diffs.mean())
    sd_diff = float(diffs.std(ddof=1))

    rng = np.random.default_rng(RNG_SEED)
    boots = rng.choice(diffs, size=(BOOTSTRAP_N, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])

    wins = int((diffs > 0).sum())
    losses = int((diffs < 0).sum())
    ties = int((diffs == 0).sum())
    # two-sided sign test (binomial, p=0.5) over non-tied seeds
    from math import comb
    k = max(wins, losses)
    m = wins + losses
    sign_p = 2 * sum(comb(m, i) for i in range(k, m + 1)) / 2 ** m if m else 1.0

    summary = {
        "seeds": SEEDS,
        "baseline_checkpoint": "policy_weights.npz (original PPO weights, sha see git)",
        "candidate_checkpoint": "best_weights.npz (monotonic_gym lambda=100, 3 kept edits)",
        "per_seed_baseline": base_scores,
        "per_seed_repaired": rep_scores,
        "paired_diff": {int(s): int(d) for s, d in zip(SEEDS, diffs)},
        "mean_baseline": float(np.mean(list(base_scores.values()))),
        "mean_repaired": float(np.mean(list(rep_scores.values()))),
        "mean_paired_diff": mean_diff,
        "sd_paired_diff": sd_diff,
        "bootstrap_ci95_diff": [float(lo), float(hi)],
        "wins_losses_ties": [wins, losses, ties],
        "sign_test_p": float(sign_p),
        "bootstrap_seed": RNG_SEED,
    }
    out = ROOT / "eval"
    out.mkdir(exist_ok=True)
    (out / "paired_heldout.json").write_text(json.dumps(summary, indent=2))

    print(f"seeds {SEEDS[0]}-{SEEDS[-1]} ({n} games, paired, greedy)")
    print(f"baseline mean  {summary['mean_baseline']:.3f}")
    print(f"repaired mean  {summary['mean_repaired']:.3f}")
    print(f"paired diff    {mean_diff:+.3f} +/- {sd_diff:.3f} (sd)")
    print(f"bootstrap 95% CI on diff: [{lo:+.3f}, {hi:+.3f}]")
    print(f"wins/losses/ties: {wins}/{losses}/{ties}, sign test p = {sign_p:.4f}")
    print("wrote eval/paired_heldout.json")


if __name__ == "__main__":
    main()
