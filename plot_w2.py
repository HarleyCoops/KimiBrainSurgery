"""W2 figure: composition curves for deep-layer rank-1 edits across the lambda sweep,
with the best W3 result (lambda=100) overlaid for comparison."""
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

runs, fixed_counts = {}, {}
for lam in LAMBDAS:
    tag = ("%g" % lam).replace(".", "p")
    hist = json.loads((ROOT / f"results_w2_{tag}.json").read_text())["history"]
    runs[lam] = hist
    fixed_counts[lam] = sum(h["fixed"] for h in hist)

w3_best = json.loads((ROOT / "results_nullspace_100.json").read_text())["history"]

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
    label = f"W2 λ={lam:g} ({fixed_counts[lam]}/{len(hist)} fixed)"
    ax.plot(xs, rets, color=COLORS[lam], lw=2 if lam in (0, 100) else 1.2,
            marker="o", ms=4.5, mfc=BG, mec=COLORS[lam], label=label,
            alpha=1.0 if lam in (0, 100) else 0.75)
    ax2.plot(xs, [100 * f for f in flips], color=COLORS[lam], lw=2 if lam in (0, 100) else 1.2,
             marker="o", ms=4.5, mfc=BG, mec=COLORS[lam], label=label,
             alpha=1.0 if lam in (0, 100) else 0.75)

# best W3 reference
xs = [0] + [h["repair"] for h in w3_best]
rets = [w3_best[0]["return_before"]] + [h["return_after"] for h in w3_best]
flips = [0.0] + [h["probe_flips"] / h["probe_size"] for h in w3_best]
ax.plot(xs, rets, color="#ffffff", lw=2.4, ls="--", marker="s", ms=5, mfc=BG,
        mec="#ffffff", label="W3 λ=100 (readout layer, best)")
ax2.plot(xs, [100 * f for f in flips], color="#ffffff", lw=2.4, ls="--", marker="s",
         ms=5, mfc=BG, mec="#ffffff", label="W3 λ=100")

ax.axhline(8.16, color=FG, alpha=0.3, ls=":", lw=1)
ax.text(7.9, 8.35, "baseline", color=FG, alpha=0.55, fontsize=8, ha="right")
ax.set_xlabel("sequential edits applied", color=FG)
ax.set_ylabel("mean return (64 held-out seeds)", color=FG)
ax.set_title("Deep-layer (W2) edits: performance after N repairs", color=FG, fontsize=11)
ax.set_xticks(range(0, 9))
ax.set_xticklabels(["base"] + [f"#{i}" for i in range(1, 9)])

ax2.set_xlabel("sequential edits applied", color=FG)
ax2.set_ylabel("probe decisions flipped (%)", color=FG)
ax2.set_title("Collateral damage per edit", color=FG, fontsize=11)
ax2.set_xticks(range(0, 9))
ax2.set_xticklabels(["base"] + [f"#{i}" for i in range(1, 9)])
ax2.set_yscale("symlog", linthresh=1)
ax2.legend(facecolor=BG, edgecolor=FG, labelcolor=FG, fontsize=8, loc="upper left")

fig.suptitle("Editing through the nonlinearity: rank-1 W2 surgery with probe-null-space constraint",
             color=FG, fontsize=13, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(ROOT / "assets" / "w2.png", facecolor=BG)
print("saved assets/w2.png")

for lam in LAMBDAS:
    hist = runs[lam]
    rets = [h["return_after"] for h in hist]
    flips = [h["probe_flips"] / h["probe_size"] for h in hist]
    newton = [h["n_newton"] for h in hist]
    print(f"W2 lam={lam:>6g}: fixed {fixed_counts[lam]}/{len(hist)} | return after 8 edits "
          f"{rets[-1]:.2f} (min {min(rets):.2f}) | median flip {100*np.median(flips):.1f}% "
          f"| median Newton steps {int(np.median(newton))}")
