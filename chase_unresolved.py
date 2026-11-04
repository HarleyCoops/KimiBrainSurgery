"""chase_unresolved — targeted attack on the one failure the monotonic loop rolled back.

Strategy: load best_weights.npz (the repaired policy), re-mine to confirm the failure is
still live, then try a GRID of solvers against exactly that state:
  - full-rank W2, lambda in {10, 100, 1000}, margin in {0.1, 0.5, 1.0, 2.0}
  - null-space W3, lambda in {10, 100, 1000}, margin in {0.1, 0.5, 1.0, 2.0}
Gate every candidate on the 64 eval seeds; a candidate wins iff the failure state's argmax
is a saving action AND mean return does not decrease. Report the full table.
"""
import numpy as np

import failure_gym as g
import nullspace_gym as ns
from fullrank_gym import solve_fullrank, MU
from w2_gym import gate, h1_h2_logits, probe_stats, MAX_PROBES_IN_H
from pathlib import Path

ROOT = Path(__file__).parent
EVAL_SEEDS = range(10000, 10064)

W_BEST = {k: v.astype(np.float64) for k, v in np.load(ROOT / "best_weights.npz").items()}


def solve_fullrank_margin(weights, failure, best_save, probes, lam, margin):
    """fullrank solve with tunable margin (monkeypatch-free reimplementation)."""
    import fullrank_gym as fr
    new = {k: v.copy() for k, v in weights.items()}
    sub = probes[:: max(1, len(probes) // MAX_PROBES_IN_H)]
    H1, D2 = probe_stats(new, sub)
    N = H1.shape[0]
    G = np.stack([((H1 * (D2[:, j] ** 2)[:, None]).T @ H1) / N for j in range(32)])
    total = np.zeros_like(new["W2"])
    for _ in range(8):
        h1_f, h2_f, logits = h1_h2_logits(new, failure.obs)
        if int(logits.argmax()) in failure.saving:
            break
        gap = float(logits[failure.wrong] - logits[best_save])
        target = gap + margin
        d = (1.0 - h2_f ** 2) * (new["W3"][:, best_save] - new["W3"][:, failure.wrong])
        y = np.empty((32, 32))
        ay = 0.0
        for j in range(32):
            aj = d[j] * h1_f
            yj = np.linalg.solve(MU * np.eye(32) + lam * G[j], aj)
            y[:, j] = yj
            ay += float(aj @ yj)
        dW2 = (target / ay) * y
        n = float(np.linalg.norm(dW2))
        if n > 1.0:
            dW2 = dW2 * (1.0 / n)
        new["W2"] = new["W2"] + dW2
        total += dW2
    return new, total


def solve_w3_margin(weights, failure, best_save, probes, lam, margin):
    """nullspace_gym solver with tunable margin."""
    H = np.stack([g.hidden_and_logits(weights, obs)[0]
                  for obs in probes[:: max(1, len(probes) // MAX_PROBES_IN_H)]])
    gap = failure.logits[failure.wrong] - failure.logits[best_save]
    hf = failure.h2
    M = np.eye(32) + lam * (H.T @ H) / H.shape[0]
    Minv_hf = np.linalg.solve(M, hf)
    alpha = (gap + margin) / float(hf @ Minv_hf)
    d = alpha * Minv_hf
    new = {k: v.copy() for k, v in weights.items()}
    new["W3"][:, failure.wrong] -= 0.5 * d
    new["W3"][:, best_save] += 0.5 * d
    return new, d


def main():
    failures, probes = g.mine(W_BEST, range(0, 200))
    print(f"mined {len(failures)} avoidable failures under best_weights")
    if not failures:
        print("nothing to chase — no avoidable failures remain")
        return
    f = failures[0]
    print("target: " + f.card())
    best_save, culprits = g.diagnose(W_BEST, f)

    rows = []
    for layer in ("W2", "W3"):
        for lam in (10.0, 100.0, 1000.0):
            for margin in (0.1, 0.5, 1.0, 2.0):
                if layer == "W2":
                    new, _ = solve_fullrank_margin(W_BEST, f, best_save, probes, lam, margin)
                else:
                    new, _ = solve_w3_margin(W_BEST, f, best_save, probes, lam, margin)
                rep = gate(W_BEST, new, f, probes, EVAL_SEEDS)
                win = rep["fixed"] and rep["return_after"] >= rep["return_before"]
                rows.append((layer, lam, margin, rep, win))
                print(f"{layer} lam={lam:>6g} margin={margin:>3g}: fixed={rep['fixed']} "
                      f"flips={rep['probe_flips']}/{rep['probe_size']} "
                      f"return {rep['return_before']:.2f} -> {rep['return_after']:.2f} "
                      f"{'*** WINNER' if win else ''}")

    winners = [r for r in rows if r[4]]
    if winners:
        best = max(winners, key=lambda r: r[3]["return_after"])
        layer, lam, margin = best[0], best[1], best[2]
        print(f"\napplying best winner: {layer} lam={lam} margin={margin} "
              f"(return {best[3]['return_after']:.2f})")
        if layer == "W2":
            new, _ = solve_fullrank_margin(W_BEST, f, best_save, probes, lam, margin)
        else:
            new, _ = solve_w3_margin(W_BEST, f, best_save, probes, lam, margin)
        np.savez(ROOT / "best_weights_v2.npz",
                 **{k: v.astype(np.float32) for k, v in new.items()})
        print("wrote best_weights_v2.npz")
    else:
        print("\nno solver config fixes this failure without hurting return — "
              "it stays honestly unresolved")


if __name__ == "__main__":
    main()
