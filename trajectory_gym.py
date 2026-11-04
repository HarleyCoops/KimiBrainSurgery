"""trajectory_gym — Experiment 10: repair a rescue, not just its entry decision.

Experiment 9 repaired individual flight-mined states, but the fresh-set return did
not move: a later defection could still waste the rescue. This loop rolls a mined
failure forward under forced saving actions until the next row-8 catch, constrains
every state on that rescue prefix, and gates on both decision preservation and an
actual greedy rescue replay.

Repair ladder (first candidate passing the hard gate wins):
  1. exact multi-constraint W3 readout edit
  2. full-rank multi-constraint W2 edit with Newton re-linearization
  3. elastic-anchored gradient hinge on all defective trajectory states

Hard gate: every constrained state chooses a saving action, the candidate catches
the target ball under greedy play, all previously protected trajectories still
pass both checks, and gate-seed mean return does not decrease.

Fresh evaluation uses seeds 50000-50499, which were not used by earlier experiments.
A v5 checkpoint requires a positive fresh paired mean difference AND zero fresh
losses versus the starting checkpoint (chain_final.npz, falling back to v4).

Usage: python trajectory_gym.py [max_attempts] [lambda]
Set TRAJECTORY_RESET=1 to discard the resumable local state.
"""
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import failure_gym as g
from chain_gym import mean_return, repair_gradient
from w2_gym import h1_h2_logits, probe_stats, MAX_PROBES_IN_H

ROOT = Path(__file__).parent
GATE_SEEDS = range(10000, 10064)
FRESH_SEEDS = range(50000, 50500)
LOSS_SEEDS_20K = [20021, 20150, 20198, 20237, 20297, 20351, 20363, 20428, 20434, 20548,
                  20552, 20598, 20617, 20693, 20938, 20943, 20999]
LOSS_SEEDS_30K = [30024, 30030, 30043, 30113, 30319, 30348, 30354, 30359]
MINE_SEEDS = list(LOSS_SEEDS_20K) + list(LOSS_SEEDS_30K) + list(range(0, 300))
DEV_SEEDS = range(0, 300)
MARGIN = 0.1
MAX_NEWTON = 12
STEP_CAP = 0.5
MU = 1.0


@dataclass
class RescueTrajectory:
    root: g.Failure
    states: list
    prefix_actions: list
    forced_actions: list


def _state_at(weights, game, seed):
    """Build a Failure-like record for the current game state."""
    obs = game.observation()
    h2, logits = g.hidden_and_logits(weights, obs)
    landing_col, ticks = g.flight_landing(game)
    saving = [a for a in (0, 3, 4)
              if g.salvageable_after(
                  int(np.clip(game.paddle + g.PADDLE_MOVES[a], 2, 7)),
                  ticks - 1, landing_col)]
    return g.Failure(seed, game.steps, obs, game.paddle, (game.row, game.col),
                     landing_col, saving, int(logits.argmax()), logits, h2), ticks


def _replay_prefix(game, trajectory):
    for action in trajectory.prefix_actions:
        game.step(int(action))
    if game.steps != trajectory.root.step:
        raise AssertionError("prefix did not reach the recorded root step")
    if not np.array_equal(game.observation(), trajectory.root.obs):
        raise AssertionError("prefix did not reproduce the recorded root observation")


def build_trajectory(weights, failure):
    """Replay to a mined root, then force saving actions until the ball is caught.

    At each node the forced action is the highest-logit currently saving action.
    This keeps the demonstration close to the policy while guaranteeing that the
    next row-8 arrival remains coverable. All nodes are retained, including ones
    the current policy already handles, so later edits cannot erode follow-through.
    """
    game = g.Breakout(200)
    game.reset(failure.seed)
    prefix_actions = []
    while game.steps < failure.step:
        _, logits = g.hidden_and_logits(weights, game.observation())
        action = int(logits.argmax())
        prefix_actions.append(action)
        game.step(action)
    if not np.array_equal(game.observation(), failure.obs):
        raise AssertionError("mined failure is not reproducible from its policy prefix")

    states, forced_actions = [], []
    for _ in range(200 - failure.step):
        state, _ticks = _state_at(weights, game, failure.seed)
        if not state.saving:
            raise AssertionError("forced rescue reached a state with no saving action")
        states.append(state)
        forced = max(state.saving, key=lambda a: (state.logits[a], -a))
        forced_actions.append(int(forced))
        game.step(forced)
        if game.terminated:
            if game.reason == "miss":
                raise AssertionError("forced saving actions missed the target ball")
            break
        if game.row == 8 and game.dy == -1:
            break
    else:
        raise AssertionError("forced rescue did not reach a row-8 arrival")

    return RescueTrajectory(failure, states, prefix_actions, forced_actions)


def defective_states(weights, states):
    return [state for state in states
            if int(g.hidden_and_logits(weights, state.obs)[1].argmax()) not in state.saving]


def replay_rescue(weights, trajectory):
    """Replay the fixed prefix, then let the candidate play greedily to row 8."""
    game = g.Breakout(200)
    game.reset(trajectory.root.seed)
    _replay_prefix(game, trajectory)
    actions = []
    for _ in range(200 - game.steps):
        _, logits = g.hidden_and_logits(weights, game.observation())
        action = int(logits.argmax())
        actions.append(action)
        game.step(action)
        if game.terminated:
            success = game.reason != "miss" and game.row == 8 and game.dy == -1
            return {"success": bool(success), "reason": game.reason,
                    "actions": actions, "steps": len(actions)}
        if game.row == 8 and game.dy == -1:
            return {"success": True, "reason": "caught",
                    "actions": actions, "steps": len(actions)}
    return {"success": False, "reason": "no_row8_arrival",
            "actions": actions, "steps": len(actions)}


def solve_w3_trajectory(weights, states, probes, lam):
    """Exact readout solve over all currently defective trajectory states.

    Unlike the earlier per-save-column batch solve, the unknown here is the whole
    W3 edit. Each constraint directly controls logit[wrong] - logit[save], while
    the ridge metric penalizes probe-logit changes across every action column.
    """
    active = defective_states(weights, states)
    if not active:
        return {k: v.copy() for k, v in weights.items()}
    H = np.stack([g.hidden_and_logits(weights, obs)[0]
                  for obs in probes[:: max(1, len(probes) // MAX_PROBES_IN_H)]])
    M = np.eye(32) + lam * (H.T @ H) / H.shape[0]
    K = len(active)
    A = np.zeros((K, 32, 5))
    Y = np.zeros_like(A)
    targets = np.empty(K)
    for k, state in enumerate(active):
        save = max(state.saving, key=lambda a: state.logits[a])
        A[k, :, save] = state.h2
        A[k, :, state.wrong] = -state.h2
        minv_h = np.linalg.solve(M, state.h2)
        Y[k, :, save] = minv_h
        Y[k, :, state.wrong] = -minv_h
        targets[k] = state.logits[state.wrong] - state.logits[save] + MARGIN
    gram = np.einsum("kia,lia->kl", A, Y)
    coef = np.linalg.solve(gram + 1e-9 * np.eye(K), targets)
    dW3 = np.tensordot(coef, Y, axes=1)
    new = {k: v.copy() for k, v in weights.items()}
    new["W3"] = new["W3"] + dW3
    return new


def solve_w2_trajectory(weights, states, probes, lam):
    """Full-rank W2 multi-constraint solve with dynamic active constraints."""
    new = {k: v.copy() for k, v in weights.items()}
    sub = probes[:: max(1, len(probes) // MAX_PROBES_IN_H)]
    total_dW2 = np.zeros_like(new["W2"])
    history = []
    for _ in range(MAX_NEWTON):
        active = defective_states(new, states)
        history.append(len(states) - len(active))
        if not active:
            break
        H1, D2 = probe_stats(new, sub)
        N = H1.shape[0]
        G = np.stack([((H1 * (D2[:, j] ** 2)[:, None]).T @ H1) / N
                      for j in range(32)])
        Minv = [np.linalg.inv(MU * np.eye(32) + lam * G[j]) for j in range(32)]

        K = len(active)
        h1s, ds, targets = [], [], []
        for state in active:
            h1, h2, logits = h1_h2_logits(new, state.obs)
            wrong = int(logits.argmax())
            save = max(state.saving, key=lambda a: logits[a])
            h1s.append(h1)
            ds.append((1.0 - h2 ** 2) * (new["W3"][:, save] - new["W3"][:, wrong]))
            targets.append(float(logits[wrong] - logits[save] + MARGIN))

        y = np.empty((K, 32, 32))
        gram = np.empty((K, K))
        for k in range(K):
            for j in range(32):
                y[k, :, j] = Minv[j] @ (ds[k][j] * h1s[k])
        for k in range(K):
            for ell in range(K):
                gram[k, ell] = sum(float((ds[k][j] * h1s[k]) @ y[ell, :, j])
                                   for j in range(32))
        coef = np.linalg.solve(gram + 1e-9 * np.eye(K), np.array(targets))
        dW2 = np.tensordot(coef, y, axes=1)
        norm = float(np.linalg.norm(dW2))
        if norm > STEP_CAP:
            dW2 *= STEP_CAP / norm
        new["W2"] = new["W2"] + dW2
        total_dW2 = total_dW2 + dW2
    return new, total_dW2, history


def gate_trajectory(weights, candidate, trajectory, protected, probes, cur_return):
    state_failures = len(defective_states(candidate, trajectory.states))
    rescue = replay_rescue(candidate, trajectory)
    regressions = 0
    regression_rescues = 0
    for old in protected:
        regressions += len(defective_states(candidate, old.states))
        regression_rescues += int(not replay_rescue(candidate, old)["success"])
    flips = sum(int(g.hidden_and_logits(weights, obs)[1].argmax()
                    != g.hidden_and_logits(candidate, obs)[1].argmax()) for obs in probes)
    return_after = mean_return(candidate, GATE_SEEDS)
    keep = (state_failures == 0 and rescue["success"] and regressions == 0
            and regression_rescues == 0 and return_after >= cur_return)
    report = {
        "trajectory_states": len(trajectory.states),
        "state_failures": state_failures,
        "rescued": int(rescue["success"]),
        "rescue_actions": rescue["actions"],
        "regressions": regressions,
        "regression_rescues": regression_rescues,
        "probe_flips": flips,
        "return_after": return_after,
    }
    return bool(keep), report


def _load_start():
    name = "chain_final.npz" if (ROOT / "chain_final.npz").exists() else "best_weights_v4.npz"
    return name, {k: v.astype(np.float64) for k, v in np.load(ROOT / name).items()}


def main():
    max_attempts = int(sys.argv[1]) if len(sys.argv) > 1 else 9
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
    start_name, start_snapshot = _load_start()
    state_path = ROOT / "trajectory_state.pkl"
    if state_path.exists() and os.environ.get("TRAJECTORY_RESET") != "1":
        state = pickle.loads(state_path.read_bytes())
        weights = state["weights"]
        protected = state["protected"]
        deferred = state["deferred"]
        history = state["history"]
        kept = state["kept"]
        cur_return = state["cur_return"]
        attempts_done = state["attempts_done"]
        print(f"[trajectory] resumed at attempt {attempts_done + 1}: {kept} kept, "
              f"gate return {cur_return:.3f}")
    else:
        weights = {k: v.copy() for k, v in start_snapshot.items()}
        protected, deferred, history = [], {}, []
        kept, attempts_done = 0, 0
        cur_return = mean_return(weights, GATE_SEEDS)
        print(f"[trajectory] start {start_name}, gate return {cur_return:.3f}")

    natural_stop = False
    for attempt in range(attempts_done + 1, max_attempts + 1):
        failures, probes = g.mine(weights, MINE_SEEDS, use_flight=True)
        failures = [f for f in failures if f.obs.tobytes() not in deferred]
        if not failures:
            print(f"[trajectory] no new avoidable failures after {attempt - 1} attempts")
            natural_stop = True
            attempts_done = attempt - 1
            break
        root = failures[0]
        trajectory = build_trajectory(weights, root)
        defective = defective_states(weights, trajectory.states)
        print(f"\n=== TRAJECTORY ATTEMPT {attempt} ===")
        print("  " + root.card())
        print(f"  rescue prefix: {len(trajectory.states)} states, "
              f"{len(defective)} defective, forced actions "
              f"{[g.ACTIONS[a] for a in trajectory.forced_actions]}")

        accepted = False
        methods = (
            ("w3-trajectory", lambda: solve_w3_trajectory(weights, trajectory.states, probes, lam)),
            ("w2-trajectory", lambda: solve_w2_trajectory(weights, trajectory.states, probes, lam)[0]),
            ("gradient-trajectory", lambda: repair_gradient(
                weights, defective, [max(f.saving, key=lambda a: f.logits[a]) for f in defective])),
        )
        for method, maker in methods:
            candidate = maker()
            keep, report = gate_trajectory(
                weights, candidate, trajectory, protected, probes, cur_return)
            print(f"  [{method}] states fixed={report['trajectory_states'] - report['state_failures']}"
                  f"/{report['trajectory_states']} rescued={report['rescued']} "
                  f"regressions={report['regressions']}+{report['regression_rescues']} replay "
                  f"flips={report['probe_flips']} return {cur_return:.3f}->"
                  f"{report['return_after']:.3f} -> {'KEPT' if keep else 'rejected'}")
            history.append({"attempt": attempt, "method": method, "kept": keep,
                            "card": root.card(), **report})
            if keep:
                weights = candidate
                protected.append(trajectory)
                cur_return = report["return_after"]
                kept += 1
                accepted = True
                break
        if not accepted:
            deferred[root.obs.tobytes()] = root.card()
        attempts_done = attempt
        state_path.write_bytes(pickle.dumps(
            {"weights": weights, "protected": protected, "deferred": deferred,
             "history": history, "kept": kept, "cur_return": cur_return,
             "attempts_done": attempts_done}))

    if not natural_stop and attempts_done < max_attempts:
        state_path.unlink(missing_ok=True)
        return
    if kept:
        np.savez(ROOT / "trajectory_final.npz",
                 **{k: v.astype(np.float32) for k, v in weights.items()})
        print("[trajectory] wrote trajectory_final.npz")

    from paired_eval import per_seed_returns
    print("\n[trajectory] fresh paired eval on untouched seeds 50000-50499")
    original = {k: v.astype(np.float64) for k, v in g.BASE_WEIGHTS.items()}
    v4 = {k: v.astype(np.float64) for k, v in np.load(ROOT / "best_weights_v4.npz").items()}
    fresh_scores = {
        "original": per_seed_returns(original, FRESH_SEEDS),
        "v4": per_seed_returns(v4, FRESH_SEEDS),
        "start": per_seed_returns(start_snapshot, FRESH_SEEDS),
        "final": per_seed_returns(weights, FRESH_SEEDS),
    }
    diff = np.array([fresh_scores["final"][s] - fresh_scores["start"][s]
                     for s in FRESH_SEEDS], dtype=float)
    dev_start = per_seed_returns(start_snapshot, DEV_SEEDS)
    dev_final = per_seed_returns(weights, DEV_SEEDS)
    dev_diff = np.array([dev_final[s] - dev_start[s] for s in DEV_SEEDS], dtype=float)
    wins, losses, ties = int((diff > 0).sum()), int((diff < 0).sum()), int((diff == 0).sum())
    v5_written = bool(kept and diff.mean() > 0 and losses == 0)
    if v5_written:
        np.savez(ROOT / "best_weights_v5.npz",
                 **{k: v.astype(np.float32) for k, v in weights.items()})
        print("[trajectory] positive fresh gain with zero losses -> wrote best_weights_v5.npz")
    else:
        print("[trajectory] v5 not written (requires kept edits, positive fresh gain, zero losses)")

    summary = {
        "start_checkpoint": start_name,
        "kept_trajectories": kept,
        "attempts": attempts_done,
        "protected_trajectories": len(protected),
        "deferred_failures": len(deferred),
        "fresh_seeds": [FRESH_SEEDS.start, FRESH_SEEDS.stop - 1],
        "fresh_mean_original": float(np.mean(list(fresh_scores["original"].values()))),
        "fresh_mean_v4": float(np.mean(list(fresh_scores["v4"].values()))),
        "fresh_mean_start": float(np.mean(list(fresh_scores["start"].values()))),
        "fresh_mean_final": float(np.mean(list(fresh_scores["final"].values()))),
        "fresh_diff_vs_start": float(diff.mean()),
        "fresh_vs_start_WLT": [wins, losses, ties],
        "dev_diff_vs_start": float(dev_diff.mean()),
        "dev_vs_start_WLT": [int((dev_diff > 0).sum()), int((dev_diff < 0).sum()),
                             int((dev_diff == 0).sum())],
        "v5_written": v5_written,
        "history": history,
    }
    (ROOT / "results_trajectory.json").write_text(json.dumps(summary, indent=2))
    eval_path = ROOT / "eval" / "paired_trajectory_fresh.json"
    eval_path.parent.mkdir(exist_ok=True)
    eval_path.write_text(json.dumps({
        "seeds": list(FRESH_SEEDS), "start_checkpoint": start_name,
        "per_seed": fresh_scores,
        "paired_diff_final_minus_start": {int(s): int(d) for s, d in zip(FRESH_SEEDS, diff)},
        "mean_diff": float(diff.mean()), "wins_losses_ties": [wins, losses, ties],
    }, indent=2))
    print(f"  start {summary['fresh_mean_start']:.3f} | final {summary['fresh_mean_final']:.3f} "
          f"| diff {diff.mean():+.3f}, W/L/T {wins}/{losses}/{ties}")
    print(f"  development diff {dev_diff.mean():+.3f}, W/L/T "
          f"{summary['dev_vs_start_WLT']}")
    print("[trajectory] wrote results_trajectory.json and eval/paired_trajectory_fresh.json")
    state_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
