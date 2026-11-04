"""w2_gym — null-space-constrained edits on the DEEP layer W2, not just the readout W3.

Why this is harder: h2 = tanh(h1 @ W2 + b2), so an edit to W2 passes through tanh.
Trick: use a rank-1 edit  dW2 = h1_f . g^T  (h1_f = layer-1 activation at the failure state).
Then for ANY state with layer-1 activation h1_p:

    dz2_p = (h1_p . h1_f) * g          (exact, linear in g)
    dh2_p ~= diag(1 - h2_p^2) dz2_p    (first-order in g)

Logit gap change at the failure state:  dh2_f . (W3[:,save] - W3[:,wrong]).
Probe damage:  sum_p (h1_p.h1_f)^2 * ||D_p g||^2,  D_p = diag(1 - h2_p^2).

Minimize  mu*||g||^2 + lam * mean_p (h1_p.h1_f)^2 ||D_p g||^2   s.t.  a . g = gap + margin,
with a = ||h1_f||^2 * D_f * (W3[:,save] - W3[:,wrong]).

The objective matrix is DIAGONAL:  A = mu + lam * mean_p (h1_p.h1_f)^2 D_p^2  (elementwise),
so the constrained solution is one line:  g = target * (a/A) / sum(a^2/A).

tanh makes the first-order model approximate, so after applying the edit we recompute the
exact gap and take up to MAX_NEWTON correction steps.

lam = 0 is the naive rank-1 edit (no probe protection). Sweep lam to trace the frontier.

Usage:  python w2_gym.py <lambda>     # writes results_w2_<lambda>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

import failure_gym as g

ROOT = Path(__file__).parent
MAX_PROBES_IN_H = 4000
MAX_NEWTON = 8
MARGIN = 0.1
MU = 1.0
STEP_CAP = 1.0  # trust region: max ||dW2||_F per Newton step; keeps tanh linearization honest


def h1_h2_logits(weights, obs):
    h1 = np.tanh(obs @ weights["W1"] + weights["b1"])
    z2 = h1 @ weights["W2"] + weights["b2"]
    h2 = np.tanh(z2)
    return h1, h2, h2 @ weights["W3"] + weights["b3"]


def probe_stats(weights, probes):
    """Per-probe layer-1 activations and squared tanh derivative at layer 2."""
    H1 = np.stack([h1_h2_logits(weights, obs)[0] for obs in probes])
    H2 = np.stack([h1_h2_logits(weights, obs)[1] for obs in probes])
    return H1, 1.0 - H2 ** 2  # D2 = diag entries of dh2/dz2


def solve_w2(weights, failure, best_save, probes, lam):
    """Rank-1 probe-constrained W2 edit with Newton refinement.
    Returns (new_weights, total_dW2, n_newton)."""
    new = {k: v.copy() for k, v in weights.items()}
    H1, D2 = probe_stats(new, probes[:: max(1, len(probes) // MAX_PROBES_IN_H)])
    total_dW2 = np.zeros_like(new["W2"])
    n_steps = 0
    for _ in range(MAX_NEWTON):
        h1_f, h2_f, logits = h1_h2_logits(new, failure.obs)
        if int(logits.argmax()) in failure.saving:
            break
        gap = float(logits[failure.wrong] - logits[best_save])
        target = gap + MARGIN
        D_f = 1.0 - h2_f ** 2
        w = new["W3"][:, best_save] - new["W3"][:, failure.wrong]
        a = float(h1_f @ h1_f) * D_f * w
        c = (H1 @ h1_f) ** 2                        # (N,) probe coupling to the failure state
        A = MU + lam * (c[:, None] * D2 ** 2).mean(axis=0)   # (32,) diagonal
        gdir = a / A
        g_step = target * gdir / float(a @ gdir)
        dW2 = np.outer(h1_f, g_step)
        n = float(np.linalg.norm(dW2))
        if n > STEP_CAP:
            dW2 = dW2 * (STEP_CAP / n)  # trust region; next Newton step re-aims
        new["W2"] = new["W2"] + dW2
        total_dW2 = total_dW2 + dW2
        n_steps += 1
    return new, total_dW2, n_steps


def gate(base_weights, new_weights, failure, probes, eval_seeds):
    _, _, logits = h1_h2_logits(new_weights, failure.obs)
    fixed = int(logits.argmax()) in failure.saving
    flips = 0
    for obs in probes:
        _, _, l0 = h1_h2_logits(base_weights, obs)
        _, _, l1 = h1_h2_logits(new_weights, obs)
        flips += int(l0.argmax() != l1.argmax())

    def mean_return(weights):
        pol = g.Policy({k: weights[k].astype(np.float32) for k in weights})
        rets = []
        for s in eval_seeds:
            game = g.Breakout(200)
            game.reset(int(s))
            while not (game.terminated or game.truncated):
                game.step(pol.act(game.observation(), None, "greedy"))
            rets.append(game.total)
        return float(np.mean(rets))

    return {"fixed": bool(fixed), "probe_flips": flips, "probe_size": len(probes),
            "return_before": mean_return(base_weights), "return_after": mean_return(new_weights)}


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
        new_weights, dW2, n_newton = solve_w2(weights, f, best_save, probes, lam)
        report = gate(weights, new_weights, f, probes, eval_seeds)
        history.append({"repair": repair_idx + 1, "failure_card": f.card(),
                        "culprits": culprits, "dW2_norm": float(np.linalg.norm(dW2)),
                        "n_newton": n_newton, **report})

        print(f"\n=== FAILURE #{repair_idx + 1} (W2 edit, lambda={lam}) ===")
        print("  " + f.card())
        print(f"  diagnosis: top hidden units driving the wrong call: "
              + ", ".join(f"unit {j} ({v:+.2f})" for j, v in culprits))
        print(f"  edit: rank-1 dW2, ||dW2||={np.linalg.norm(dW2):.4f}, "
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
    out = ROOT / f"results_w2_{tag}.json"
    out.write_text(json.dumps({"lambda": lam, "layer": "W2", "history": hist}, indent=2))
    print(f"\nwrote {out.name}")
