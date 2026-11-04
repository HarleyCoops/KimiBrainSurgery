"""wandb_batch_log.py — log the Experiment 8 batch-solver sweep to W&B (Forge).

One run per solver configuration; all numbers from results_batch_sweep.json.
"""
import json
from pathlib import Path

import wandb

ROOT = Path("/mnt/c/Users/chris/Documents/kimi/tasks/2026-11-03/07-41-57-57fc09e5")
PROJECT = "kimi-brain-surgery"
ENTITY = "christian-cooper-us"


def main():
    data = json.loads((ROOT / "results_batch_sweep.json").read_text())
    for cfg in data["configs"]:
        tag = ("%g" % cfg["lambda"]).replace(".", "p")
        name = f"batch-{cfg['layer']}-{cfg['mode']}-lam{tag}"
        if cfg.get("step_cap"):
            name += f"-cap{cfg['step_cap']}"
        run = wandb.init(project=PROJECT, entity=ENTITY, name=name,
                         group="batch-solver-sweep", reinit=True,
                         config={"layer": cfg["layer"], "mode": cfg["mode"],
                                 "lambda": cfg["lambda"], "margin": cfg["margin"],
                                 "step_cap": cfg.get("step_cap"),
                                 "n_failures": cfg["n_failures"]})
        run.summary["constraints_fixed"] = cfg["fixed"]
        run.summary["n_failures"] = cfg["n_failures"]
        run.summary["fix_rate"] = cfg["fixed"] / cfg["n_failures"]
        run.summary["return_before"] = cfg["return_before"]
        run.summary["return_after"] = cfg["return_after"]
        run.summary["return_delta"] = cfg["return_after"] - cfg["return_before"]
        run.summary["probe_flips"] = cfg["probe_flips"]
        run.summary["probe_flip_frac"] = cfg["probe_flips"] / cfg["probe_size"]
        if "dW2_norm" in cfg:
            run.summary["dW2_norm"] = cfg["dW2_norm"]
        run.finish()
        print(f"logged {name}")


if __name__ == "__main__":
    main()
