"""cross_repo.py — aim KimiBrainSurgery's solver at FailureIsAllYouNeed's seed-314159 miss.

Their LLM editor improved mean return but never prevented the recorded failure.
Question: can our exact mine -> diagnose -> solve loop prevent it, and at what cost
to their audit-seed performance?

We load THEIR faulty baseline (checkpoints/baseline.safetensors), mine the failure on
seed 314159 with the flight miner, and run the full solver grid. Gate = their audit
seeds would be selection-contaminated, so we gate on our own held-out range and ALSO
report their audit range as the reference benchmark.
"""
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

THEIRS = Path.home() / "FailureIsAllYouNeed"
sys.path.insert(0, str(THEIRS / "vendor"))

import failure_gym as g
import chase_unresolved as ch

BASE = {k: v.astype(np.float64) for k, v in load_file(THEIRS / "checkpoints" / "baseline.safetensors").items()}


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


def replay_seed314159(weights):
    pol = g.Policy({k: weights[k].astype(np.float32) for k in weights})
    game = g.Breakout(200)
    game.reset(314159)
    while not (game.terminated or game.truncated):
        game.step(pol.act(game.observation(), None, "greedy"))
    return game.total, game.reason, game.steps


def main():
    ret, reason, steps = replay_seed314159(BASE)
    print(f"their baseline on seed 314159: return={ret}, reason={reason}, steps={steps}")

    failures, probes = g.mine(BASE, [314159], use_flight=True)
    print(f"mined {len(failures)} failure(s) on seed 314159 (flight miner)")
    for f in failures:
        print("  " + f.card())
    if not failures:
        return

    f = failures[0]
    best_save, culprits = g.diagnose(BASE, f)
    print("culprits:", [(j, round(v, 2)) for j, v in culprits])

    # probes from a broader rollout for the null-space constraint
    _, big_probes = g.mine(BASE, range(0, 60), use_flight=True)
    print(f"{len(big_probes)} probe states")

    AUDIT = range(4090000, 4091024)   # their audit split (reference only)
    OURS = range(10000, 10064)        # our gate (never used by their editor)
    base_audit = mean_return(BASE, AUDIT)
    base_ours = mean_return(BASE, OURS)
    print(f"their baseline: audit-seed mean {base_audit:.4f} | our-seed mean {base_ours:.4f}")

    for layer in ("W2", "W3"):
        for lam in (10.0, 100.0, 1000.0):
            if layer == "W2":
                new, _ = ch.solve_fullrank_margin(BASE, f, best_save, big_probes, lam, 0.1)
            else:
                new, _ = ch.solve_w3_margin(BASE, f, best_save, big_probes, lam, 0.1)
            _, _, lg = ch.h1_h2_logits(new, f.obs) if hasattr(ch, "h1_h2_logits") else (None, None, None)
            # fixed check via gate helper
            rep = ch.gate(BASE, new, f, big_probes, OURS)
            ret2, reason2, _ = replay_seed314159(new)
            audit_after = mean_return(new, AUDIT)
            print(f"{layer} lam={lam:>6g}: state-fixed={rep['fixed']} | "
                  f"seed314159 replay: return={ret2} reason={reason2} | "
                  f"audit {base_audit:.4f} -> {audit_after:.4f} | "
                  f"our-gate {rep['return_before']:.2f} -> {rep['return_after']:.2f}")


if __name__ == "__main__":
    main()
