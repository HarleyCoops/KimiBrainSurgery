"""Frontier figure: naive vs null-space-constrained edits across the lambda sweep."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent
BG, FG = "#0c1014", "#e8f0e8"
LAMBDAS = [0, 1, 10, 100, 1000, 10000]
COLORS = {0: "#ff3860", 1: "#ff9f1c", 10: "#ffe14c", 100: "#39ff14",
          1000: "#4cc9f0", 10000: "#b388ff"}

runs = {}
for lam in LAMBDAS:
    tag = ("%g" % lam).replace(".", "p")
    runs[lam] = json.loads((ROOT / f"results_nullspace_{tag}.json").read_text())["history"]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=160)
fig.patch.set_facecolor(BG)
for a in (ax, ax2):
    a.set_facecolor(BG)
    a.tick_params(colors=FG)
    for s in a.spines.values():
        s.set_color(FG); s.set_alpha(0.25)

for lam in LAMBDAS:
    hist = runs[lam]
    xs = [0] + [h["repair"] for h in hist]
    rets = [hist[0]["return_before"]] + [h["return_after"] for h in hist]
    flips = [0.0] + [h["probe_flips"] / h["probe_size"] for h in hist]
    label = "naive (λ=0)" if lam == 0 else f"λ={lam:g}"
    ax.plot(xs, rets, color=COLORS[lam], lw=2 if lam in (0, 100) else 1.2,
            marker="o", ms=4.5, mfc=BG, mec=COLORS[lam], label=label,
            alpha=1.0 if lam in (0, 100) else 0.75)
    ax2.plot(xs, [100 * f for f in flips], color=COLORS[lam], lw=2 if lam in (0, 100) else 1.2,
             marker="o", ms=4.5, mfc=BG, mec=COLORS[lam], label=label,
             alpha=1.0 if lam in (0, 100) else 0.75)

ax.axhline(8.16, color=FG, alpha=0.3, ls="--", lw=1)
ax.text(7.9, 8.35, "baseline", color=FG, alpha=0.55, fontsize=8, ha="right")
ax.set_xlabel("sequential edits applied", color=FG)
ax.set_ylabel("mean return (64 held-out seeds)", color=FG)
ax.set_title("Does the policy still play well after N edits?", color=FG, fontsize=11)
ax.set_xticks(range(0, 9))
ax.set_xticklabels(["base"] + [f"#{i}" for i in range(1, 9)])

ax2.set_xlabel("sequential edits applied", color=FG)
ax2.set_ylabel("probe decisions flipped (%)", color=FG)
ax2.set_title("Collateral damage per edit", color=FG, fontsize=11)
ax2.set_xticks(range(0, 9))
ax2.set_xticklabels(["base"] + [f"#{i}" for i in range(1, 9)])
ax2.set_yscale("symlog", linthresh=1)
ax2.legend(facecolor=BG, edgecolor=FG, labelcolor=FG, fontsize=8.5, loc="upper left")

fig.suptitle("The null-space fix: constrain each edit to leave probe states unmoved",
             color=FG, fontsize=13, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(ROOT / "assets" / "nullspace.png", facecolor=BG)
print("saved assets/nullspace.png")

# console summary for the report
for lam in LAMBDAS:
    hist = runs[lam]
    rets = [h["return_after"] for h in hist]
    flips = [h["probe_flips"] / h["probe_size"] for h in hist]
    fixed = sum(h["fixed"] for h in hist)
    print(f"lam={lam:>6g}: fixed {fixed}/{len(hist)} | return after 8 edits "
          f"{rets[-1]:.2f} (min {min(rets):.2f}) | median flip frac {100*np.median(flips):.1f}%")
