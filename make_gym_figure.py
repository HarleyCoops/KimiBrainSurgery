"""Composition-curve figure for the failure gym run (data captured verbatim from stdout)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG = "#0c1014"
FG = "#e8f0e8"
ACCENT = "#39ff14"   # CRT green
WARN = "#ff3860"
SECOND = "#4cc9f0"

repairs = np.arange(0, 9)
returns = [8.16, 4.78, 1.16, 3.77, 1.06, 3.77, 1.06, 3.52, 2.38]
flip_frac = [0.0, 13708/21402, 1669/8368, 693/3240, 1208/7162,
             616/2936, 1047/7162, 582/2976, 2241/7050]
fixed = [None, True, True, True, True, True, True, True, False]

fig, ax = plt.subplots(figsize=(10, 5.6), dpi=160)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

ax.axhline(8.16, color=FG, alpha=0.35, ls="--", lw=1)
ax.text(8.35, 8.16, "baseline (trained PPO policy)", color=FG, alpha=0.6,
        va="center", fontsize=9)

ax.plot(repairs, returns, color=ACCENT, lw=2.2, marker="o", ms=7,
        mfc=BG, mec=ACCENT, mew=2, zorder=3, label="mean return (64 held-out seeds)")
for i, ok in enumerate(fixed):
    if ok is False:
        ax.plot(repairs[i], returns[i], marker="o", ms=13, mfc="none",
                mec=WARN, mew=2, zorder=4)

ax.set_xlabel("sequential weight edits applied", color=FG)
ax.set_ylabel("mean return", color=ACCENT)
ax.tick_params(colors=FG)
ax.set_xticks(repairs)
ax.set_xticklabels(["base"] + [f"#{i}" for i in range(1, 9)])
ax.set_ylim(0, 9.5)
for s in ax.spines.values():
    s.set_color(FG); s.set_alpha(0.25)

ax2 = ax.twinx()
ax2.bar(repairs, flip_frac, width=0.55, color=SECOND, alpha=0.28, zorder=1,
        label="fraction of probe decisions flipped by the edit")
ax2.set_ylabel("probe decisions flipped", color=SECOND)
ax2.tick_params(colors=SECOND)
ax2.set_ylim(0, 1.0)
for s in ax2.spines.values():
    s.set_visible(False)

# annotate the ping-pong
ax.annotate("edit #1 fixes the failure\nbut flips 64% of all probe decisions",
            xy=(1, 4.78), xytext=(1.4, 6.6), color=FG, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=FG, alpha=0.6))
ax.annotate("edits #2-#7 ping-pong between\ntwo failure modes fighting over\nthe same hidden units",
            xy=(4, 1.06), xytext=(4.2, 5.6), color=FG, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=FG, alpha=0.6))
ax.annotate("#8: edit fails the gate\n(failure not fixed)", xy=(8, 2.38), xytext=(6.1, 0.6),
            color=WARN, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=WARN, alpha=0.7))

lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
leg = ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
                facecolor=BG, edgecolor=FG, labelcolor=FG, fontsize=9)
leg.get_frame().set_alpha(0.85)

ax.set_title("The composition curve: how many one-shot weight edits can the net absorb?",
             color=FG, fontsize=12, pad=14)
fig.tight_layout()
fig.savefig("assets/gym.png", facecolor=BG)
print("saved assets/gym.png")
