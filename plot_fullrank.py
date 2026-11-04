"""Full-rank W2 figure: does relaxing the rank-1 restriction close the gap to W3 surgery?
Overlays: full-rank W2 sweep, best rank-1 W2, best W3."""
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
    hist = json.loads((ROOT / f"results_fullrank_{tag}.json").read_text())["history"]
    runs[lam] = hist
    fixed_counts[lam] = sum(h["fixed"] for h in hist)

w2_rank1 = json.loads((ROOT / "results_w2_0.json").read_text())["history"]  # best rank-1 by return
w3_best = json.loads((ROOT / "results_nullspace_100.json").read_text())["history"]

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), dpi=160)
fig.patch.set_facecolor(BG)
for a in (ax, ax2):
    a.set_facecolor(BG)
    a.tick_params(colors=FG)
    for s in a.spines.values():
        s.set_color(FG); s.set_alpha(0.25)


def curve(ax_, hist, style, label):
    xs = [0] + [h["repair"] for h in hist]
    rets = [hist[0]["return_before"]] + [h["return_after"] for h in hist]
    flips = [0.0] + [h["probe_flips"] / h["probe_size"] for h in hist]
    ys = rets if ax_ is ax else [100 * f for f in flips]
    ax_.plot(xs, ys, label=label, **style)


for lam in LAMBDAS:
    hl = lam in (0, 100)
    style = dict(color=COLORS[lam], lw=2 if hl else 1.2, marker="o", ms=4.5,
                 mfc=BG, mec=COLORS[lam], alpha=1.0 if hl else 0.75)
    curve(ax, runs[lam], style, f"full W2 λ={lam:g} ({fixed_counts[lam]}/{len(runs[lam])} fixed)")
    curve(ax2, runs[lam], style, f"full W2 λ={lam:g}")

curve(ax, w2_rank1, dict(color=COLORS[0], lw=1.6, ls=":", marker="^", ms=5, mfc=BG,
                         mec=COLORS[0]), "rank-1 W2 λ=0 (best rank-1)")
curve(ax2, w2_rank1, dict(color=COLORS[0], lw=1.6, ls=":", marker="^", ms=5, mfc=BG,
                          mec=COLORS[0]), "rank-1 W2 λ=0")
curve(ax, w3_best, dict(color="#ffffff", lw=2.4, ls="--", marker="s", ms=5, mfc=BG,
                        mec="#ffffff"), "W3 λ=100 (readout, best overall)")
curve(ax2, w3_best, dict(color="#ffffff", lw=2.4, ls="--", marker="s", ms=5, mfc=BG,
                         mec="#ffffff"), "W3 λ=100")

ax.axhline(8.16, color=FG, alpha=0.3, ls=":", lw=1)
ax.text(7.9, 8.35, "baseline", color=FG, alpha=0.55, fontsize=8, ha="right")
ax.set_xlabel("sequential edits applied", color=FG)
ax.set_ylabel("mean return (64 held-out seeds)", color=FG)
ax.set_title("Full-rank W2 edits: performance after N repairs", color=FG, fontsize=11)
ax.set_xticks(range(0, 9))
ax.set_xticklabels(["base"] + [f"#{i}" for i in range(1, 9)])

ax2.set_xlabel("sequential edits applied", color=FG)
ax2.set_ylabel("probe decisions flipped (%)", color=FG)
ax2.set_title("Collateral damage per edit", color=FG, fontsize=11)
ax2.set_xticks(range(0, 9))
ax2.set_xticklabels(["base"] + [f"#{i}" for i in range(1, 9)])
ax2.set_yscale("symlog", linthresh=1)
ax2.legend(facecolor=BG, edgecolor=FG, labelcolor=FG, fontsize=8, loc="upper left")

fig.suptitle("Full-rank deep surgery: freeing all 1024 entries of W2",
             color=FG, fontsize=13, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(ROOT / "assets" / "fullrank.png", facecolor=BG)
print("saved assets/fullrank.png")

for lam in LAMBDAS:
    hist = runs[lam]
    rets = [h["return_after"] for h in hist]
    flips = [h["probe_flips"] / h["probe_size"] for h in hist]
    newton = [h["n_newton"] for h in hist]
    print(f"full lam={lam:>6g}: fixed {fixed_counts[lam]}/{len(hist)} | return after 8 "
          f"{rets[-1]:.2f} (min {min(rets):.2f}) | median flip {100*np.median(flips):.1f}% "
          f"| median Newton {int(np.median(newton))}")
