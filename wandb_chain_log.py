import json
import wandb

ROOT = "/mnt/c/Users/chris/Documents/kimi/tasks/2026-11-03/07-41-57-57fc09e5"
d = json.load(open(ROOT + "/results_chain.json"))

run = wandb.init(project="kimi-brain-surgery", entity="christian-cooper-us",
                 name="chain-v4-complete", group="failure-chain",
                 config={"start": "v4", "lambda": 100,
                         "ladder": ["w3-nullspace", "w2-fullrank", "gradient"]})
for h in d["history"]:
    run.log({"attempt": h["attempt"], "method": h.get("method", "?"),
             "kept": int(h["kept"]), "fixed": h.get("fixed", 0),
             "regressions": h.get("regressions", 0),
             "probe_flips": h.get("probe_flips", 0),
             "return_after": h.get("return_after", 0)})
run.summary["kept_edits"] = d["kept_edits"]
run.summary["fresh_mean_final"] = d["fresh_mean_final"]
run.summary["fresh_diff_vs_v4"] = d["fresh_diff_vs_v4"]
run.summary["dev_diff_vs_v4"] = d["dev_diff_vs_v4"]
run.summary["v5_written"] = d["v5_written"]
run.finish()
print("logged chain-v4-complete")
