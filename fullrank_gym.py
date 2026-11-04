"""fullrank_gym — full-rank constrained W2 surgery via the block-diagonal (Kronecker) Gram structure.

w2_gym.py restricted the edit to rank-1:  dW2 = h1_f g^T.  Here all 32x32 entries of dW2 are free.

Why it stays closed form: for any state with layer-1 activation h1_p,
    dz2_p = h1_p @ dW2                (EXACT, linear in dW2)
    dh2_p ~= D_p (h1_p @ dW2),  D_p = diag(1 - h2_p^2)

With vec(dW2) stacking the 32 columns, the probe penalty decomposes per output unit j:
    sum_p ||D_p (h1_p @ dW2)||^2  =  sum_j  dW2[:,j]^T G_j dW2[:,j],
    G_j = (1/N) sum_p D_p[j]^2 h1_p h1_p^T         (32x32)

So the Hessian  M = I + lam * blockdiag(G_1..G_32)  is block-diagonal: 32 small solves, no
1024x1024 inversion. The fix constraint is also per-block:
    gap flip:  sum_j d_j (h1_f . dW2[:,j]) = gap + margin,   d = D_f * (W3[:,save]-W3[:,wrong])
    => constraint vector a has blocks a_j = d_j * h1_f.

Solution:  x = target * M^-1 a / (a . M^-1 a),  dW2 = unvec(x).
Trust region per Newton step as in w2_gym.py.

Usage:  python fullrank_gym.py <lambda>     # writes results_fullrank_<lambda>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

import failure_gym as g
from w2_gym import h1_h2_logits, probe_stats, gate, MAX_PROBES_IN_H, MAX_NEWTON, MARGIN, STEP_CAP

ROOT = Path(__file__).parent
MU = 1.0


def solve_fullrank(weights, failure, best_save, probes, lam):
    """Full-rank probe-constrained W2 edit with Newton refinement.
    Returns (new_weights, total_dW2, n_newton)."""
    new = {k: v.copy() for k, v in weights.items()}
    sub = probes[:: max(1, len(probes) // MAX_PROBES_IN_H)]
    H1, D2 = probe_stats(new, sub)              # (N,32), (N,32)
    N = H1.shape[0]
    # block Gram matrices G_j = (1/N) sum_p D2[p,j]^2 h1_p h1_p^T
    G = np.stack([((H1 * (D2[:, j] ** 2)[:, None]).T @ H1) / N for j in range(32)])

    total_dW2 = np.zeros_like(new["W2"])
    n_steps = 0
    for _ in range(MAX_NEWTON):
        h1_f, h2_f, logits = h1_h2_logits(new, failure.obs)
        if int(logits.argmax()) in failure.saving:
            break
        gap = float(logits[failure.wrong] - logits[best_save])
        target = gap + MARGIN
        d = (1.0 - h2_f ** 2) * (new["W3"][:, best_save] - new["W3"][:, failure.wrong])
        # constraint vector blocks a_j = d_j * h1_f ; solve blockwise
        y = np.empty((32, 32))   # y[:,j] = (MU*I + lam*G_j)^-1 (d_j h1_f)
        ay = 0.0
        for j in range(32):
            aj = d[j] * h1_f
            yj = np.linalg.solve(MU * np.eye(32) + lam * G[j], aj)
            y[:, j] = yj
            ay += float(aj @ yj)
        dW2 = (target / ay) * y
        n = float(np.linalg.norm(dW2))
        if n > STEP_CAP:
            dW2 = dW2 * (STEP_CAP / n)
        new["W2"] = new["W2"] + dW2
        total_dW2 = total_dW2 + dW2
        n_steps += 1
    return new, total_dW2, n_steps


def run(lam, max_repairs=8, mine_seeds=range(0, 200), eval_seeds=range(10000, 10064)):
    weights = {k: v.copy() for k, v in g.BASE_WEIGHTS.items()}
    history = []
    for repair_idx in range(max_repairs):
        failures, probes = g.mine(weights, mine_seeds)
        if not failures:
            print(f"[gym] no avoidable greedy failures found after {repair_idx} repairs")
            break
        f = failures[0]
        best_save, culprits = g.diagnose(weights, f)
        new_weights, dW2, n_newton = solve_fullrank(weights, f, best_save, probes, lam)
        report = gate(weights, new_weights, f, probes, eval_seeds)
        history.append({"repair": repair_idx + 1, "failure_card": f.card(),
                        "culprits": culprits, "dW2_norm": float(np.linalg.norm(dW2)),
                        "n_newton": n_newton, **report})

        print(f"\n=== FAILURE #{repair_idx + 1} (full-rank W2, lambda={lam}) ===")
        print("  " + f.card())
        print(f"  diagnosis: top hidden units driving the wrong call: "
              + ", ".join(f"unit {j} ({v:+.2f})" for j, v in culprits))
        print(f"  edit: full-rank dW2, ||dW2||={np.linalg.norm(dW2):.4f}, "
              f"{n_newton} Newton step(s)")
        print(f"  gate: failure fixed={report['fixed']} | probe decisions flipped "
              f"{report['probe_flips']}/{report['probe_size']} | mean return "
              f"{report['return_before']:.2f} -> {report['return_after']:.2f}")
        weights = new_weights
    return history


if __name__ == "__main__":
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    hist = run(lam)
    tag = ("%g" % lam).replace(".", "p")
    out = ROOT / f"results_fullrank_{tag}.json"
    out.write_text(json.dumps({"lambda": lam, "layer": "W2", "rank": "full",
                               "history": hist}, indent=2))
    print(f"\nwrote {out.name}")
