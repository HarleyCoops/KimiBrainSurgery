"""batch_gym — fix an entire failure CLASS with one edit instead of paying per state.

All instances of the ascent class share the same shape: policy picks no-op (or wrong move),
a right/left move saves it, the miss becomes unavoidable. Single-state edits fix one and
break others. Batch solve: one d applied to W3 that satisfies EVERY failure constraint at once.

Constraints (exact, linear in W3): for each failure i with hidden state h_i,
    h_i . (W3[:,save_i] - W3[:,wrong_i])  must increase by  gap_i + margin.
With symmetric edit (W3[:,wrong] -= d/2, W3[:,save] += d/2) that's  A d = b,  A_i = h_i.

Objective: minimize ||d||^2 + lam * ||H d||^2 / N   (H = probe hidden states)
Closed form:  d = M^-1 A^T (A M^-1 A^T)^-1 b,   M = I + lam H^T H / N.

Usage: python batch_gym.py <lambda> [margin]
"""
import json
import sys
from pathlib import Path

import numpy as np

import failure_gym as g

ROOT = Path(__file__).parent
MAX_PROBES_IN_H = 4000
EVAL_SEEDS = range(10000, 10064)

# seeds where v4 loses 13 -> 0 (from eval/paired_v4_*.json), plus general mining seeds
LOSS_SEEDS_20K = [20021, 20150, 20198, 20237, 20297, 20351, 20363, 20428, 20434, 20548,
                  20552, 20598, 20617, 20693, 20938, 20943, 20999]
LOSS_SEEDS_30K = [30024, 30030, 30043, 30113, 30319, 30348, 30354, 30359]  # partial list ok


def batch_solve(weights, failures, probes, lam, margin, soft=False, only_class=None):
    """One edit PER SAVE COLUMN: split failures by their best saving action, and for each
    class solve  min ||d||^2 + lam*||Hd||^2/N  s.t.  h_i . d = gap_i + margin  for all i,
    then apply  W3[:, save] += d.  Closed form: d = M^-1 A^T (A M^-1 A^T)^-1 b."""
    H = np.stack([g.hidden_and_logits(weights, obs)[0]
                  for obs in probes[:: max(1, len(probes) // MAX_PROBES_IN_H)]])
    M = np.eye(32) + lam * (H.T @ H) / H.shape[0]
    Minv = np.linalg.inv(M)
    new = {k: v.copy() for k, v in weights.items()}
    classes = {}
    for f in failures:
        save = max(f.saving, key=lambda a: f.logits[a])
        classes.setdefault(save, []).append(f)
    info = []
    for save, fs in classes.items():
        if only_class is not None and save != only_class:
            continue
        A = np.stack([f.h2 for f in fs])
        b = np.array([f.logits[f.wrong] - f.logits[save] + margin for f in fs])
        if soft:
            # soft: min ||A d - b||^2 + ||d||^2 + lam*||H d||^2/N  (least squares, no exact force)
            Msoft = A.T @ A + np.eye(32) + lam * (H.T @ H) / H.shape[0]
            d = np.linalg.solve(Msoft, A.T @ b)
        else:
            MinvAt = Minv @ A.T
            d = MinvAt @ np.linalg.solve(A @ MinvAt, b)
        new["W3"][:, save] += d
        info.append((save, len(fs), float(np.linalg.norm(d))))
    return new, info


def main():
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    margin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    soft = len(sys.argv) > 3 and sys.argv[3] == "soft"
    only_class = None
    if len(sys.argv) > 4:
        only_class = g.ACTIONS.index(sys.argv[4])

    W = {k: v.astype(np.float64) for k, v in np.load(ROOT / "best_weights_v4.npz").items()}

    mine_seeds = list(LOSS_SEEDS_20K) + list(LOSS_SEEDS_30K) + list(range(0, 100))
    failures, probes = g.mine(W, mine_seeds, use_flight=True)
    print(f"mined {len(failures)} failures (flight miner) under v4")
    for f in failures[:8]:
        print("  " + f.card())

    if not failures:
        print("nothing to fix"); return

    new, info = batch_solve(W, failures, probes, lam, margin, soft=soft, only_class=only_class)
    for save, n, nd in info:
        print(f"batch edit: W3[:, '{g.ACTIONS[save]}'] += d ({n} constraints, ||d||={nd:.4f})")

    # check every constraint state
    fixed_all = sum(int(g.hidden_and_logits(new, f.obs)[1].argmax()) in f.saving
                    for f in failures)
    flips = sum(int(g.hidden_and_logits(W, o)[1].argmax()
                    != g.hidden_and_logits(new, o)[1].argmax()) for o in probes)

    def mean_return(weights, seeds):
        pol = g.Policy({k: weights[k].astype(np.float32) for k in weights})
        rets = []
        for s in seeds:
            game = g.Breakout(200)
            game.reset(int(s))
            while not (game.terminated or game.truncated):
                game.step(pol.act(game.observation(), None, "greedy"))
            rets.append(game.total)
        return float(np.mean(rets))

    r_before = mean_return(W, EVAL_SEEDS)
    r_after = mean_return(new, EVAL_SEEDS)
    print(f"constraints fixed: {fixed_all}/{len(failures)} | probe flips {flips}/{len(probes)} "
          f"| gate return {r_before:.2f} -> {r_after:.2f}")

    if fixed_all == len(failures) and r_after >= r_before:
        np.savez(ROOT / "best_weights_v5.npz",
                 **{k: v.astype(np.float32) for k, v in new.items()})
        print("PASSES GATE -> wrote best_weights_v5.npz")
    else:
        print("does not pass gate; v5 not written")
    json.dump({"lambda": lam, "margin": margin, "n_failures": len(failures),
               "fixed": fixed_all, "probe_flips": flips, "probe_size": len(probes),
               "return_before": r_before, "return_after": r_after},
              open(ROOT / "results_batch.json", "w"), indent=2)


if __name__ == "__main__":
    main()
