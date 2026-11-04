"""batch_fullrank_gym — batch-constrained FULL-RANK W2 edit for a whole failure class.

Why this exists: the v4 ascent class (13 -> 0, failure to pre-position) resists every W3 edit.
The 'left' subclass needs ||d|| ~= 14-17 at W3 (nearly contradictory constraints in the 32-dim
hidden space); the exact edit destroys the policy (return 12.09 -> 2.41), and soft W3 edits fix
at most 1/5 before the same cliff. The constraints are unsatisfiable at the readout layer.

So move one layer up: edit W2 (full 32x32, Kronecker/block-diagonal structure as in
fullrank_gym.py) so the hidden representations THEMSELVES move apart, making the class
linearly separable for the existing W3.

Math (per Newton step, linearized at current weights):
    dz2_p = h1_p @ dW2                       (exact, linear in dW2)
    dh2_p ~= D_p (h1_p @ dW2),  D_p = diag(1 - h2_p^2)
    constraint i:  sum_j d_ij (h1_i . dW2[:,j]) = gap_i + margin,
        d_i = D_i * (W3[:,save_i] - W3[:,wrong_i])
    probe penalty: sum_j dW2[:,j]^T G_j dW2[:,j],  G_j = (1/N) sum_p D_p[j]^2 h1_p h1_p^T

With K constraints over 1024 unknowns and block-diagonal M = mu*I + lam*blockdiag(G_j):
    x = M^-1 A^T (A M^-1 A^T)^-1 b
M^-1 A^T needs K*32 small 32x32 solves; A M^-1 A^T is KxK. Closed form, no 1024x1024 inversion.

Gate: every constraint state's greedy action must land in its saving set AND mean return on
held-out seeds must not drop. Writes best_weights_v5.npz only if both hold.

Usage: python batch_fullrank_gym.py <lambda> [margin] [max_newton]
"""
import json
import sys
from pathlib import Path

import numpy as np

import failure_gym as g
from w2_gym import h1_h2_logits, probe_stats, MAX_PROBES_IN_H

ROOT = Path(__file__).parent
MU = 1.0
STEP_CAP = 0.5  # tighter than single-failure: batch moves are larger
EVAL_SEEDS = range(10000, 10064)
LOSS_SEEDS_20K = [20021, 20150, 20198, 20237, 20297, 20351, 20363, 20428, 20434, 20548,
                  20552, 20598, 20617, 20693, 20938, 20943, 20999]
LOSS_SEEDS_30K = [30024, 30030, 30043, 30113, 30319, 30348, 30354, 30359]


def batch_fullrank_solve(weights, failures, probes, lam, margin, max_newton):
    """Newton loop: each iteration re-linearizes all K constraints at the current weights
    and takes one trust-region-capped closed-form step."""
    new = {k: v.copy() for k, v in weights.items()}
    sub = probes[:: max(1, len(probes) // MAX_PROBES_IN_H)]
    saves = [max(f.saving, key=lambda a: f.logits[a]) for f in failures]
    total_dW2 = np.zeros_like(new["W2"])
    hist = []
    for step in range(max_newton):
        H1, D2 = probe_stats(new, sub)
        N = H1.shape[0]
        G = np.stack([((H1 * (D2[:, j] ** 2)[:, None]).T @ H1) / N for j in range(32)])

        # recompute constraint data at CURRENT weights
        K = len(failures)
        h1s, ds, targets = [], [], []
        n_fixed = 0
        for f, save in zip(failures, saves):
            h1_f, h2_f, logits = h1_h2_logits(new, f.obs)
            if int(logits.argmax()) in f.saving:
                n_fixed += 1
            gap = float(logits[f.wrong] - logits[save])
            h1s.append(h1_f)
            ds.append((1.0 - h2_f ** 2) * (new["W3"][:, save] - new["W3"][:, f.wrong]))
            targets.append(gap + margin)
        hist.append(n_fixed)
        if n_fixed == K:
            break

        # y[k][:,j] = (mu*I + lam*G_j)^-1 (d_kj h1_k);  AMinvAt[k,l] = sum_j a_kj . y_lj
        y = np.empty((K, 32, 32))
        AMinvAt = np.empty((K, K))
        Minv = [np.linalg.inv(MU * np.eye(32) + lam * G[j]) for j in range(32)]
        for k in range(K):
            for j in range(32):
                y[k][:, j] = Minv[j] @ (ds[k][j] * h1s[k])
        for k in range(K):
            for l in range(K):
                AMinvAt[k, l] = sum(float((ds[k][j] * h1s[k]) @ y[l][:, j]) for j in range(32))
        coef = np.linalg.solve(AMinvAt + 1e-9 * np.eye(K), np.array(targets))
        dW2 = np.tensordot(coef, y, axes=1)  # sum_k coef_k y[k]
        n = float(np.linalg.norm(dW2))
        if n > STEP_CAP:
            dW2 = dW2 * (STEP_CAP / n)
        new["W2"] = new["W2"] + dW2
        total_dW2 = total_dW2 + dW2
    return new, total_dW2, hist


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


def main():
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    margin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    max_newton = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    if len(sys.argv) > 4:
        global STEP_CAP
        STEP_CAP = float(sys.argv[4])

    W = {k: v.astype(np.float64) for k, v in np.load(ROOT / "best_weights_v4.npz").items()}

    mine_seeds = list(LOSS_SEEDS_20K) + list(LOSS_SEEDS_30K) + list(range(0, 100))
    failures, probes = g.mine(W, mine_seeds, use_flight=True)
    print(f"mined {len(failures)} failures (flight miner) under v4")
    for f in failures:
        print("  " + f.card())
    if not failures:
        print("nothing to fix"); return

    new, dW2, hist = batch_fullrank_solve(W, failures, probes, lam, margin, max_newton)
    print(f"newton fixed-count trajectory: {hist}  |  total ||dW2||={np.linalg.norm(dW2):.4f}")

    fixed_all = sum(int(g.hidden_and_logits(new, f.obs)[1].argmax()) in f.saving
                    for f in failures)
    flips = sum(int(g.hidden_and_logits(W, o)[1].argmax()
                    != g.hidden_and_logits(new, o)[1].argmax()) for o in probes)
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
    json.dump({"lambda": lam, "margin": margin, "layer": "W2", "rank": "full",
               "n_failures": len(failures), "fixed": fixed_all, "probe_flips": flips,
               "probe_size": len(probes), "dW2_norm": float(np.linalg.norm(dW2)),
               "newton_hist": hist, "return_before": r_before, "return_after": r_after},
              open(ROOT / "results_batch_fullrank.json", "w"), indent=2)


if __name__ == "__main__":
    main()
