"""monotonic_gym — the failure gym with a keep/rollback gate: performance never goes down
ON THE MEASURED GATE. A finite set of eval games can miss regressions, and repeatedly
gating against the same seeds can overfit them — so the run ends with an UNTOUCHED
held-out evaluation (seeds the gate never saw) to check whether the gains generalize.

Same surgery as fullrank_gym.py (full-rank constrained W2 edits), with four rules added:

    KEEP the edit only if  return_after >= return_before  (deterministic eval => exact rule).
    Otherwise ROLL BACK to the pre-edit weights. The failure is marked UNRESOLVED — not
    silently fixed, not forgotten: it is skipped in future mining but reported at the end.
    Previously REPAIRED failures are retested after every kept edit; regressions are
    reported, not hidden.
    A STALL means the solver stopped proposing improving edits — it does NOT prove the
    network ran out of capacity.

Tracks the best weights ever seen (by gated mean return) and writes them to
best_weights.npz at the end.

Usage:  python monotonic_gym.py [lambda] [max_repairs]
"""
import json
import sys
from pathlib import Path

import numpy as np

import failure_gym as g
from fullrank_gym import solve_fullrank
from w2_gym import gate, h1_h2_logits

ROOT = Path(__file__).parent
HELD_OUT_SEEDS = range(20000, 20064)  # never mined, never gated on — final check only


def mean_return(weights, eval_seeds):
    pol = g.Policy({k: weights[k].astype(np.float32) for k in weights})
    rets = []
    for s in eval_seeds:
        game = g.Breakout(200)
        game.reset(int(s))
        while not (game.terminated or game.truncated):
            game.step(pol.act(game.observation(), None, "greedy"))
        rets.append(game.total)
    return float(np.mean(rets))


def run(lam=100.0, max_repairs=24, mine_seeds=range(0, 200), eval_seeds=range(10000, 10064)):
    weights = {k: v.copy() for k, v in g.BASE_WEIGHTS.items()}
    history = []
    unresolved = {}           # obs bytes -> failure card, edits rolled back
    repaired = []             # (obs, saving actions, card) retested after each kept edit
    best = None               # (return, weights snapshot)
    stalls = 0

    for repair_idx in range(max_repairs):
        failures, probes = g.mine(weights, mine_seeds)
        failures = [f for f in failures if f.obs.tobytes() not in unresolved]
        if not failures:
            print(f"[gym] no new avoidable failures after {repair_idx} attempts")
            break
        f = failures[0]
        best_save, culprits = g.diagnose(weights, f)
        new_weights, dW2, n_newton = solve_fullrank(weights, f, best_save, probes, lam)
        report = gate(weights, new_weights, f, probes, eval_seeds)
        keep = report["return_after"] >= report["return_before"]

        regressions = 0
        if keep and repaired:
            for obs, saving, _card in repaired:
                _, _, lg = h1_h2_logits(new_weights, obs)
                regressions += int(int(lg.argmax()) not in saving)

        rec = {"attempt": repair_idx + 1, "failure_card": f.card(), "culprits": culprits,
               "dW2_norm": float(np.linalg.norm(dW2)), "n_newton": n_newton,
               "kept": bool(keep), "repaired_regressions": regressions, **report}
        history.append(rec)

        verdict = "KEPT" if keep else "ROLLED BACK -> failure marked unresolved"
        print(f"\n=== ATTEMPT #{repair_idx + 1} (monotonic-on-gate, lambda={lam}) -> {verdict} ===")
        print("  " + f.card())
        print(f"  gate: failure fixed={report['fixed']} | probe flips "
              f"{report['probe_flips']}/{report['probe_size']} | mean return "
              f"{report['return_before']:.2f} -> {report['return_after']:.2f}"
              + (f" | regressions in previously repaired: {regressions}/{len(repaired)}"
                 if keep and repaired else ""))

        if keep:
            if report["fixed"]:
                repaired.append((f.obs, f.saving, f.card()))
            weights = new_weights
            stalls = 0
            if best is None or report["return_after"] > best[0]:
                best = (report["return_after"], {k: v.copy() for k, v in weights.items()})
        else:
            unresolved[f.obs.tobytes()] = f.card()
            stalls += 1
            if stalls >= 8:
                print("[gym] 8 consecutive rollbacks — solver stopped proposing improving edits")
                print("[gym] NOTE: this is a solver stall, NOT proof of exhausted capacity")
                break

    print(f"\n[gym] {sum(h['kept'] for h in history)}/{len(history)} edits kept; "
          f"curve monotonic on the measured gate by construction")
    if unresolved:
        print(f"[gym] {len(unresolved)} failures remain UNRESOLVED (edits rolled back):")
        for card in unresolved.values():
            print("   - " + card)

    if best is not None:
        np.savez(ROOT / "best_weights.npz",
                 **{k: v.astype(np.float32) for k, v in best[1].items()})
        base_ho = mean_return(g.BASE_WEIGHTS, HELD_OUT_SEEDS)
        best_ho = mean_return(best[1], HELD_OUT_SEEDS)
        print(f"\n[gym] best gated return {best[0]:.2f} (baseline "
              f"{history[0]['return_before']:.2f}) -> wrote best_weights.npz")
        print(f"[gym] UNTOUCHED held-out check (seeds {HELD_OUT_SEEDS.start}-"
              f"{HELD_OUT_SEEDS.stop - 1}, never used for gating):")
        print(f"      baseline {base_ho:.2f} -> repaired {best_ho:.2f}  "
              f"({'generalizes' if best_ho >= base_ho else 'DOES NOT GENERALIZE — gate overfit'})")
        history.append({"held_out": True, "baseline": base_ho, "repaired": best_ho})
    return history, best


if __name__ == "__main__":
    lam = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
    max_repairs = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    hist, _ = run(lam, max_repairs)
    tag = ("%g" % lam).replace(".", "p")
    out = ROOT / f"results_monotonic_{tag}.json"
    out.write_text(json.dumps({"lambda": lam,
                               "rule": "keep iff return_after >= return_before (measured gate only)",
                               "history": hist}, indent=2))
    print(f"wrote {out.name}")

