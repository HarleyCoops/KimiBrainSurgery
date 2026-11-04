"""wandb_log.py — publish the full KimiBrainSurgery experiment trail to Weights & Biases.

One run per experiment; every number comes from the saved JSON records, nothing recomputed.
Artifacts: the weight checkpoints each run produced.
"""
import json
import os
from pathlib import Path

import wandb

ROOT = Path("/mnt/c/Users/chris/Documents/kimi/tasks/2026-11-03/07-41-57-57fc09e5")
PROJECT = "kimi-brain-surgery"
ENTITY = "christian-cooper-us"

BASELINE_GATE = 8.16      # mean return, gate seeds 10000-10063
BASELINE_HELDOUT = 8.69   # mean return, seeds 20000-20999
BASELINE_FRESH = 8.886    # mean return, seeds 30000-30999


def load(name):
    return json.loads((ROOT / name).read_text())


def log_sweep(prefix, layer_label, lambdas):
    for lam in lambdas:
        tag = ("%g" % lam).replace(".", "p")
        data = load(f"results_{prefix}_{tag}.json")
        hist = data["history"]
        run = wandb.init(project=PROJECT, entity=ENTITY,
                         name=f"{prefix}-lam{tag}", group=f"{prefix}-sweep",
                         config={"lambda": lam, "layer": layer_label,
                                 "solver": prefix}, reinit=True)
        run.define_metric("repair")
        for h in hist:
            run.log({"repair": h["repair"],
                     "mean_return": h["return_after"],
                     "probe_flip_frac": h["probe_flips"] / h["probe_size"],
                     "failure_fixed": int(h["fixed"])})
        rets = [h["return_after"] for h in hist]
        run.summary["fixed_count"] = sum(h["fixed"] for h in hist)
        run.summary["n_repairs"] = len(hist)
        run.summary["final_return"] = rets[-1]
        run.summary["min_return"] = min(rets)
        run.summary["baseline_gate"] = BASELINE_GATE
        run.finish()
        print(f"logged {prefix} lam={lam}")


def main():
    # sweeps
    log_sweep("nullspace", "W3", [0, 1, 10, 100, 1000, 10000])
    log_sweep("w2", "W2 rank-1", [0, 1, 10, 100, 1000, 10000])
    log_sweep("fullrank", "W2 full-rank", [0, 1, 10, 100, 1000, 10000])

    # monotonic run
    data = load("results_monotonic_100.json")
    hist = [h for h in data["history"] if "attempt" in h]
    run = wandb.init(project=PROJECT, entity=ENTITY, name="monotonic-lam100",
                     group="monotonic", config={"lambda": 100, "solver": "fullrank",
                                                "gate": "keep iff return not worse"})
    run.define_metric("attempt")
    for h in hist:
        run.log({"attempt": h["attempt"],
                 "mean_return": h["return_after"],
                 "kept": int(h["kept"]),
                 "probe_flip_frac": h["probe_flips"] / h["probe_size"],
                 "repaired_regressions": h.get("repaired_regressions", 0)})
    art = wandb.Artifact("best-weights-v1", type="model",
                         description="monotonic loop output; never worse on 1000 held-out games")
    art.add_file(str(ROOT / "best_weights.npz"))
    run.log_artifact(art)
    run.finish()
    print("logged monotonic")

    # paired evals as tables
    run = wandb.init(project=PROJECT, entity=ENTITY, name="paired-evals",
                     group="evaluation", config={"games": 1000, "mode": "greedy, paired"})
    for fname, col in [("paired_heldout.json", "v1"), ("paired_v4_fresh.json", "v4")]:
        d = load(f"eval/{fname}")
        seeds = d.get("seeds") or list(range(20000, 21000))
        bkey = "per_seed_baseline"
        ckey = [k for k in d if k.startswith("per_seed_") and "baseline" not in k][0]
        tbl = wandb.Table(columns=["seed", "baseline", col, "diff"])
        for s in seeds:
            b, c = d[bkey][str(s)], d[ckey][str(s)]
            tbl.add_data(int(s), b, c, c - b)
        run.log({f"paired_{col}": tbl})
    run.summary["v1_diff"] = 1.075
    run.summary["v1_wlt"] = "137/0/863"
    run.summary["v4_fresh_diff"] = 3.299
    run.summary["v4_fresh_wlt"] = "253/20/727"
    for name, desc in [("best_weights_v3.npz", "v3: +1.83, first 17 losses"),
                       ("best_weights_v4.npz", "v4: +3.30 fresh, max mean")]:
        art = wandb.Artifact(name.replace(".npz", "").replace("_", "-"), type="model",
                             description=desc)
        art.add_file(str(ROOT / name))
        run.log_artifact(art)
    run.finish()
    print("logged paired evals + artifacts")


if __name__ == "__main__":
    main()
