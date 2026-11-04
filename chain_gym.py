"""chain_gym — the complete chain: mine -> repair -> preserve -> fresh eval -> repeat.

One continuous loop over the CURRENT ACCEPTED POLICY (v4). Each iteration:
  1. IDENTIFY   flight-mined avoidable failure in the accepted policy
  2. REPAIR     ladder of increasingly expressive edits, first candidate that
                passes the gate wins:
                  a. W3 null-space closed form      (cheapest, exact at readout)
                  b. W2 full-rank closed form       (Kronecker, moves representations)
                  c. gradient hinge on the failure  (training signal, elastic-anchored)
  3. PRESERVE   hard gate: failure fixed AND zero regressions on every previously
                repaired state AND gate-seed mean return does not drop
  4. REPEAT     re-mine under the new policy; deferred failures are not retried
  5. FRESH EVAL at the end: paired per-seed eval on seeds 40000-40499 (untouched by
                mining, gating, or development) final vs v4 vs original

If closed forms stall on 3+ deferred failures, ONE batch-gradient phase runs on all
deferred states at once (the Experiment-8 verdict: the class needs training signal,
not geometry) and is gated as a single candidate.

Monotonic on the measured gate by construction: a candidate is accepted only if the
measured return does not drop and no previously repaired failure regresses. The repair
method is a detail; the gate is the invariant.

Usage: python chain_gym.py [max_attempts] [lam]     (WANDB=1 to stream to Forge)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

import failure_gym as g
import nullspace_gym
import fullrank_gym
from w2_gym import h1_h2_logits

ROOT = Path(__file__).parent

GATE_SEEDS = range(10000, 10064)          # gate only, never mined
FRESH_SEEDS = range(40000, 40500)         # untouched until the final eval
LOSS_SEEDS_20K = [20021, 20150, 20198, 20237, 20297, 20351, 20363, 20428, 20434, 20548,
                  20552, 20598, 20617, 20693, 20938, 20943, 20999]
LOSS_SEEDS_30K = [30024, 30030, 30043, 30113, 30319, 30348, 30354, 30359]
MINE_SEEDS = list(LOSS_SEEDS_20K) + list(LOSS_SEEDS_30K) + list(range(0, 300))

MARGIN = 0.1
GRAD_STEPS = 150
GRAD_LR = 0.002
GRAD_MU = 1e-2           # elastic anchor toward current weights
STALL_BATCH_THRESHOLD = 3


# ---------------------------------------------------------------- repairs

def repair_w3(weights, f, best_save, probes, lam):
    H = np.stack([g.hidden_and_logits(weights, o)[0]
                  for o in probes[:: max(1, len(probes) // 4000)]])
    new, _d, _alpha = nullspace_gym.solve_constrained(weights, f, best_save, H, lam,
                                                      margin=MARGIN)
    return new


def repair_w2(weights, f, best_save, probes, lam):
    new, dW2, n = fullrank_gym.solve_fullrank(weights, f, best_save, probes, lam)
    return new


def _forward(weights, obs):
    h1 = np.tanh(obs @ weights["W1"] + weights["b1"])
    h2 = np.tanh(h1 @ weights["W2"] + weights["b2"])
    return h1, h2, h2 @ weights["W3"] + weights["b3"]


def repair_gradient(weights, failures, saves):
    """Hinge-loss gradient steps on a batch of failure states, W2/W3 only,
    elastic-anchored to the current weights. Returns new weights."""
    new = {k: v.copy() for k, v in weights.items()}
    base = {k: v.copy() for k, v in weights.items()}
    names = ("W2", "W3", "b2", "b3")
    m = {k: np.zeros_like(new[k]) for k in names}
    v = {k: np.zeros_like(new[k]) for k in names}
    b1r, b2r, eps = 0.9, 0.999, 1e-8

    for t in range(1, GRAD_STEPS + 1):
        grads = {k: np.zeros_like(new[k]) for k in names}
        active = 0
        for f, save in zip(failures, saves):
            h1, h2, logits = _forward(new, f.obs)
            gap = logits[save] - logits[f.wrong]
            if gap >= MARGIN:
                continue
            active += 1
            dlog = np.zeros_like(logits)
            dlog[f.wrong] = 1.0
            dlog[save] = -1.0          # dL/dlogits for relu(MARGIN - gap)
            grads["W3"] += np.outer(h2, dlog)
            grads["b3"] += dlog
            dh2 = new["W3"] @ dlog
            dz2 = dh2 * (1.0 - h2 ** 2)
            grads["W2"] += np.outer(h1, dz2)
            grads["b2"] += dz2
        if active == 0:
            break
        for k in names:
            grads[k] /= active
            grads[k] += GRAD_MU * (new[k] - base[k])
            m[k] = b1r * m[k] + (1 - b1r) * grads[k]
            v[k] = b2r * v[k] + (1 - b2r) * grads[k] * grads[k]
            mh = m[k] / (1 - b1r ** t)
            vh = v[k] / (1 - b2r ** t)
            new[k] = new[k] - GRAD_LR * mh / (np.sqrt(vh) + eps)
    return new


# ---------------------------------------------------------------- gate

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


def gate_chain(weights, cand, f, repaired, probes, cur_return):
    """Hard gate: target fixed, zero regressions on repaired states, return not worse."""
    _, lg = g.hidden_and_logits(cand, f.obs)
    fixed = int(lg.argmax()) in f.saving
    regressions = 0
    for obs, saving, _card in repaired:
        _, lg_r = g.hidden_and_logits(cand, obs)
        regressions += int(int(lg_r.argmax()) not in saving)
    flips = sum(int(g.hidden_and_logits(weights, o)[1].argmax()
                    != g.hidden_and_logits(cand, o)[1].argmax()) for o in probes)
    r_after = mean_return(cand, GATE_SEEDS)
    keep = bool(fixed) and regressions == 0 and r_after >= cur_return
    return keep, {"fixed": int(fixed), "regressions": regressions,
                  "probe_flips": flips, "return_after": r_after}


# ---------------------------------------------------------------- wandb

def wb_init(attempts, lam):
    if os.environ.get("WANDB") != "1":
        return None
    try:
        import wandb
        return wandb.init(project="kimi-brain-surgery", entity="christian-cooper-us",
                          name="chain-v4", group="failure-chain",
                          config={"start": "v4", "max_attempts": attempts, "lambda": lam,
                                  "ladder": ["w3-nullspace", "w2-fullrank", "gradient"]})
    except Exception as e:
        print(f"[chain] wandb unavailable here ({e}); logging locally, sync from WSL later")
        return None


def wb_log(run, **kw):
    if run is not None:
        run.log(kw)


# ---------------------------------------------------------------- chain

def main():
    max_attempts = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0

    import pickle
    state_path = ROOT / "chain_state.pkl"
    if state_path.exists() and os.environ.get("CHAIN_RESET") != "1":
        st = pickle.loads(state_path.read_bytes())
        weights, repaired, deferred = st["weights"], st["repaired"], st["deferred"]
        history, kept, cur_return = st["history"], st["kept"], st["cur_return"]
        attempts_done = st["attempts_done"]
        print(f"[chain] resumed: {attempts_done} attempts done, {kept} edits kept, "
              f"gate return {cur_return:.2f}")
    else:
        weights = {k: v.astype(np.float64)
                   for k, v in np.load(ROOT / "best_weights_v4.npz").items()}
        repaired, deferred, history, kept, attempts_done = [], {}, [], 0, 0
        cur_return = mean_return(weights, GATE_SEEDS)
        print(f"[chain] start: v4, gate return {cur_return:.2f}")
    v4_snapshot = {k: v.astype(np.float64)
                   for k, v in np.load(ROOT / "best_weights_v4.npz").items()}
    run = wb_init(max_attempts, lam)

    natural_stop = False
    last_batch_size = 0
    for attempt in range(attempts_done + 1, max_attempts + 1):
        failures, probes = g.mine(weights, MINE_SEEDS, use_flight=True)
        failures = [f for f in failures if f.obs.tobytes() not in deferred]
        if not failures:
            print(f"[chain] no new avoidable failures after {attempt - 1} attempts")
            natural_stop = True
            break
        f = failures[0]
        best_save, _ = g.diagnose(weights, f)
        print(f"\n=== ATTEMPT {attempt} ===")
        print("  " + f.card())

        accepted = False
        for method, maker in (("w3-nullspace", lambda: repair_w3(weights, f, best_save, probes, lam)),
                              ("w2-fullrank", lambda: repair_w2(weights, f, best_save, probes, lam)),
                              ("gradient", lambda: repair_gradient(weights, [f], [best_save]))):
            cand = maker()
            keep, rep = gate_chain(weights, cand, f, repaired, probes, cur_return)
            print(f"  [{method}] fixed={rep['fixed']} regressions={rep['regressions']} "
                  f"flips={rep['probe_flips']} return {cur_return:.2f}->{rep['return_after']:.2f} "
                  f"-> {'KEPT' if keep else 'rejected'}")
            wb_log(run, attempt=attempt, method=method, kept=int(keep), **rep)
            history.append({"attempt": attempt, "method": method, "kept": keep,
                            "card": f.card(), **rep})
            if keep:
                weights = cand
                cur_return = rep["return_after"]
                repaired.append((f.obs, f.saving, f.card()))
                kept += 1
                accepted = True
                break
        if not accepted:
            deferred[f.obs.tobytes()] = f.card()

        # batch-gradient escalation when closed forms stall (once per deferred-set state)
        if len(deferred) >= STALL_BATCH_THRESHOLD and len(deferred) != last_batch_size:
            last_batch_size = len(deferred)
            print(f"\n[chain] {len(deferred)} deferred — batch gradient phase on all of them")
            recent = [f2 for f2 in g.mine(weights, MINE_SEEDS, use_flight=True)[0]
                      if f2.obs.tobytes() in deferred]
            saves = [g.diagnose(weights, f2)[0] for f2 in recent]
            cand = repair_gradient(weights, recent, saves)
            flips = sum(int(g.hidden_and_logits(weights, o)[1].argmax()
                            != g.hidden_and_logits(cand, o)[1].argmax()) for o in probes)
            r_after = mean_return(cand, GATE_SEEDS)
            n_fixed = sum(int(g.hidden_and_logits(cand, f2.obs)[1].argmax()) in f2.saving
                          for f2 in recent)
            regressions = sum(int(g.hidden_and_logits(cand, obs)[1].argmax()) not in saving
                              for obs, saving, _c in repaired)
            keep = n_fixed == len(recent) and regressions == 0 and r_after >= cur_return
            print(f"  [batch-gradient] fixed={n_fixed}/{len(recent)} regressions={regressions} "
                  f"flips={flips} return {cur_return:.2f}->{r_after:.2f} "
                  f"-> {'KEPT' if keep else 'rejected'}")
            wb_log(run, attempt=attempt, method="batch-gradient", kept=int(keep),
                   fixed=n_fixed, regressions=regressions, probe_flips=flips,
                   return_after=r_after)
            history.append({"attempt": attempt, "method": "batch-gradient", "kept": keep,
                            "fixed": n_fixed, "of": len(recent), "regressions": regressions,
                            "probe_flips": flips, "return_after": r_after})
            if keep:
                for f2 in recent:
                    repaired.append((f2.obs, f2.saving, f2.card()))
                weights = cand
                cur_return = r_after
                kept += n_fixed
                deferred.clear()
            else:
                # leave them deferred; do not retry this run
                pass

        # checkpoint after every attempt (survives timeout-killed chunks)
        state_path.write_bytes(pickle.dumps(
            {"weights": weights, "repaired": repaired, "deferred": deferred,
             "history": history, "kept": kept, "cur_return": cur_return,
             "attempts_done": attempt}))

    finished = natural_stop or (attempt >= max_attempts)
    if not finished:
        print(f"[chain] chunk ended at attempt {attempt}; state saved, resume to continue")
        if run is not None:
            run.finish()
        return

    # ------------------------------------------------------ fresh evaluation
    print("\n[chain] fresh paired eval on seeds 40000-40499 (untouched until now)")
    from paired_eval import per_seed_returns
    orig = {k: v.astype(np.float64) for k, v in g.BASE_WEIGHTS.items()}
    s_orig = per_seed_returns(orig, FRESH_SEEDS)
    s_v4 = per_seed_returns(v4_snapshot, FRESH_SEEDS)
    s_new = per_seed_returns(weights, FRESH_SEEDS)
    d_v4 = np.array([s_new[s] - s_v4[s] for s in FRESH_SEEDS], dtype=float)
    d_orig = np.array([s_new[s] - s_orig[s] for s in FRESH_SEEDS], dtype=float)

    # development-pool eval: the seeds the miner saw; where kept repairs should show up
    DEV_SEEDS = range(0, 300)
    t_v4 = per_seed_returns(v4_snapshot, DEV_SEEDS)
    t_new = per_seed_returns(weights, DEV_SEEDS)
    d_dev = np.array([t_new[s] - t_v4[s] for s in DEV_SEEDS], dtype=float)

    if kept > 0:
        np.savez(ROOT / "chain_final.npz",
                 **{k: v.astype(np.float32) for k, v in weights.items()})
        print("[chain] wrote chain_final.npz (accepted-policy checkpoint, gated edits only)")
    summary = {
        "kept_edits": kept, "attempts": len(history),
        "repaired_states_protected": len(repaired),
        "fresh_mean_original": float(np.mean(list(s_orig.values()))),
        "fresh_mean_v4": float(np.mean(list(s_v4.values()))),
        "fresh_mean_final": float(np.mean(list(s_new.values()))),
        "fresh_diff_vs_v4": float(d_v4.mean()),
        "fresh_diff_vs_original": float(d_orig.mean()),
        "fresh_vs_v4_WLT": [int((d_v4 > 0).sum()), int((d_v4 < 0).sum()),
                            int((d_v4 == 0).sum())],
        "dev_diff_vs_v4": float(d_dev.mean()),
        "dev_vs_v4_WLT": [int((d_dev > 0).sum()), int((d_dev < 0).sum()),
                          int((d_dev == 0).sum())],
        "history": history,
    }
    print(f"  original {summary['fresh_mean_original']:.2f} | v4 {summary['fresh_mean_v4']:.2f} "
          f"| final {summary['fresh_mean_final']:.2f} "
          f"(vs v4: {summary['fresh_diff_vs_v4']:+.3f}, "
          f"W/L/T {summary['fresh_vs_v4_WLT']})")
    print(f"  development seeds 0-299 (miner's pool): final vs v4 "
          f"{summary['dev_diff_vs_v4']:+.3f}, W/L/T {summary['dev_vs_v4_WLT']}")
    wb_log(run, final_fresh_mean=summary["fresh_mean_final"],
           final_fresh_diff_vs_v4=summary["fresh_diff_vs_v4"], kept_edits=kept)

    if kept > 0 and d_v4.mean() > 0:
        np.savez(ROOT / "best_weights_v5.npz",
                 **{k: v.astype(np.float32) for k, v in weights.items()})
        print("[chain] strictly better than v4 on fresh seeds -> wrote best_weights_v5.npz")
        summary["v5_written"] = True
    else:
        print("[chain] v5 not written (no kept edits or no fresh-seed gain)")
        summary["v5_written"] = False
    (ROOT / "results_chain.json").write_text(json.dumps(summary, indent=2))
    state_path.unlink(missing_ok=True)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
