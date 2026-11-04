"""cross_repo2.py — iterate the repair loop on THEIR faulty baseline until seed 314159
survives, evaluating in SAMPLED mode (their verifier's protocol) on their audit seeds.

Each iteration: replay seed 314159 greedy, mine the first failure, solve (W3 null-space,
which preserved performance in cross_repo.py), keep iff sampled audit mean doesn't drop.
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
AUDIT = range(4090000, 4091024)
AUDIT_SUB = range(4090000, 4090256)  # subset for loop speed; full range for final


def sampled_return(weights, seeds, n_games_seed=1, rng_seed=7):
    pol = g.Policy({k: weights[k].astype(np.float32) for k in weights})
    rng = np.random.default_rng(rng_seed)
    rets = []
    for s in seeds:
        game = g.Breakout(200)
        game.reset(int(s))
        while not (game.terminated or game.truncated):
            obs = game.observation()
            p = np.exp(pol.logits(obs) - pol.logits(obs).max())
            p /= p.sum()
            game.step(int(rng.choice(5, p=p)))
        rets.append(game.total)
    return float(np.mean(rets))


def replay314159(weights, greedy=True):
    pol = g.Policy({k: weights[k].astype(np.float32) for k in weights})
    game = g.Breakout(200)
    game.reset(314159)
    rng = np.random.default_rng(7)
    while not (game.terminated or game.truncated):
        obs = game.observation()
        if greedy:
            game.step(int(pol.logits(obs).argmax()))
        else:
            p = np.exp(pol.logits(obs) - pol.logits(obs).max()); p /= p.sum()
            game.step(int(rng.choice(5, p=p)))
    return game.total, game.reason, game.steps


def main():
    print(f"their baseline: sampled audit-sub mean {sampled_return(BASE, AUDIT_SUB):.4f}")
    weights = {k: v.copy() for k, v in BASE.items()}
    _, big_probes = g.mine(weights, range(0, 60), use_flight=True)

    for it in range(12):
        # replay seed 314159 and collect the trajectory states to mine on it directly
        pol = g.Policy({k: weights[k].astype(np.float32) for k in weights})
        game = g.Breakout(200)
        game.reset(314159)
        fail = None
        while not (game.terminated or game.truncated):
            obs = game.observation()
            h2, logits = g.hidden_and_logits(weights, obs)
            land = g.flight_landing(game)
            if land is not None:
                landing_col, ticks = land
                action = int(logits.argmax())
                pa = int(np.clip(game.paddle + g.PADDLE_MOVES[action], 2, 7))
                if not g.salvageable_after(pa, ticks - 1, landing_col):
                    saving = [a for a in (0, 3, 4)
                              if g.salvageable_after(int(np.clip(game.paddle + g.PADDLE_MOVES[a], 2, 7)),
                                                     ticks - 1, landing_col)]
                    if saving and action not in saving:
                        fail = g.Failure(314159, game.steps, obs, game.paddle,
                                         (game.row, game.col), landing_col, saving,
                                         action, logits, h2)
                        break
            game.step(int(logits.argmax()))

        if fail is None:
            ret, reason, steps = replay314159(weights)
            print(f"\niter {it}: seed 314159 NO FAILURE FOUND — replay: return={ret} "
                  f"reason={reason} steps={steps}")
            break

        print(f"\niter {it}: {fail.card()}")
        best_save, culprits = g.diagnose(weights, fail)
        new, _ = ch.solve_w3_margin(weights, fail, best_save, big_probes, 10.0, 0.1)
        _, _, lg = (None, None, None)
        # gate: state fixed + sampled audit-sub doesn't drop
        before = sampled_return(weights, AUDIT_SUB)
        after = sampled_return(new, AUDIT_SUB)
        _, l1 = g.hidden_and_logits(new, fail.obs)
        fixed = int(l1.argmax()) in fail.saving
        keep = fixed and after >= before
        print(f"  fixed={fixed} | sampled audit-sub {before:.4f} -> {after:.4f} | "
              f"{'KEPT' if keep else 'ROLLED BACK'}")
        if not keep:
            print("  loop stalls here — solver cannot fix without hurting the audit metric")
            break
        weights = new

    ret, reason, steps = replay314159(weights)
    print(f"\nfinal: seed 314159 greedy replay return={ret} reason={reason} steps={steps}")
    print(f"final sampled audit-sub mean: {sampled_return(weights, AUDIT_SUB):.4f}")
    np.savez(Path.cwd() / "theirs_repaired_ours.npz",
             **{k: v.astype(np.float32) for k, v in weights.items()})
    print("wrote theirs_repaired_ours.npz")


if __name__ == "__main__":
    main()
