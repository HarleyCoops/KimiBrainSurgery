"""nullspace_gym — constrained weight edits in the (approximate) null space of probe activations.

Same mine -> diagnose -> solve -> gate -> iterate loop as failure_gym.py, but SOLVE is replaced:

  naive:       d = ((gap+margin) / ||h_f||^2) * h_f
               (minimal norm, ignores every other state)

  constrained: minimize  ||d||^2 + lam * ||H d||^2 / N   s.t.  h_f . d = gap + margin
               closed form:  d = alpha * (I + lam * H^T H / N)^-1 h_f,
               alpha = (gap+margin) / (h_f . (I + lam H^T H/N)^-1 h_f)

H rows are h2 hidden states of probe states (states the policy actually visits). The edit is
applied symmetrically: W3[:, wrong] -= d/2, W3[:, save] += d/2, so the failure-state logit gap
flips by exactly gap+margin while probe logits move as little as possible in the H metric.

lam = 0 reproduces failure_gym.py exactly. Sweeping lam traces the fix-vs-interference frontier.

Usage:  python nullspace_gym.py <lambda>     # writes results_nullspace_<lambda>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

import failure_gym as g

ROOT = Path(__file__).parent
ACTIONS = g.ACTIONS
MAX_PROBES_IN_H = 4000


def solve_constrained(weights, failure, best_save, H, lam, margin=0.1):
    """Closed-form ridge-constrained minimal edit. Returns (new_weights, d, alpha)."""
    gap = failure.logits[failure.wrong] - failure.logits[best_save]
    hf = failure.h2
    n_units = hf.shape[0]
    M = np.eye(n_units) + lam * (H.T @ H) / H.shape[0]
    Minv_hf = np.linalg.solve(M, hf)
    alpha = (gap + margin) / float(hf @ Minv_hf)
    d = alpha * Minv_hf
    new = {k: v.copy() for k, v in weights.items()}
    new["W3"][:, failure.wrong] -= 0.5 * d
    new["W3"][:, best_save] += 0.5 * d
    return new, d, float(alpha)


def gate(base_weights, new_weights, failure, probes, eval_seeds):
    """Same regression battery as failure_gym.gate but on the constrained edit."""
    _, logits = g.hidden_and_logits(new_weights, failure.obs)
    fixed = int(logits.argmax()) in failure.saving
    flips = 0
    for obs in probes:
        _, l0 = g.hidden_and_logits(base_weights, obs)
        _, l1 = g.hidden_and_logits(new_weights, obs)
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
        H = np.stack([g.hidden_and_logits(weights, obs)[0]
                      for obs in probes[:: max(1, len(probes) // MAX_PROBES_IN_H)]])
        new_weights, d, alpha = solve_constrained(weights, f, best_save, H, lam)
        report = gate(weights, new_weights, f, probes, eval_seeds)
        history.append({"repair": repair_idx + 1, "failure_card": f.card(),
                        "culprits": culprits, "alpha": alpha, "d_norm": float(np.linalg.norm(d)),
                        "H_size": int(H.shape[0]), **report})

        print(f"\n=== FAILURE #{repair_idx + 1} (lambda={lam}) ===")
        print("  " + f.card())
        print(f"  diagnosis: top hidden units driving the wrong call: "
              + ", ".join(f"unit {j} ({v:+.2f})" for j, v in culprits))
        print(f"  edit: constrained, ||d||={np.linalg.norm(d):.4f}, H from {H.shape[0]} probes")
        print(f"  gate: failure fixed={report['fixed']} | probe decisions flipped "
              f"{report['probe_flips']}/{report['probe_size']} | mean return "
              f"{report['return_before']:.2f} -> {report['return_after']:.2f}")
        weights = new_weights
    return history


if __name__ == "__main__":
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    hist = run(lam)
    tag = ("%g" % lam).replace(".", "p")
    out = ROOT / f"results_nullspace_{tag}.json"
    out.write_text(json.dumps({"lambda": lam, "history": hist}, indent=2))
    print(f"\nwrote {out.name}")
