# 🧠 KimiBrainSurgery

**An automated failure → diagnose → repair → verify loop that edits a neural network's weights directly — no retraining, no gradients — and measures how many repairs the network can absorb.**

The patient: [`GridPolicyMLP`](https://huggingface.co/HarleyCooper/grid-policy-mlp), a 17,253-parameter PPO-trained Breakout policy (500→32→32→5, tanh). Companion to the [live visualizer](https://harleycoops.github.io/breakout-policy-visualizer/).

---

## The loop

Like an RL gym, except the training signal is *failures*:

1. **MINE** — greedy rollouts over 200 seeds. A *failure* is the exact decision node where a descending ball's landing column (computed by exact forward simulation, wall bounces included) flips from coverable to uncoverable because of the chosen paddle action.
2. **DIAGNOSE** — the final layer is linear in the hidden state (`logits = h2 @ W3 + b3`), so the fatal logit gap decomposes *exactly* into per-hidden-unit contributions. Arithmetic, not attribution heuristics.
3. **SOLVE** — a closed-form minimal edit to `W3` that flips the decision at the failure state.
4. **GATE** — regression battery: is the failure fixed? How many of ~20k probe-state decisions flipped? What did mean return on 64 held-out seeds do?
5. **ITERATE** — apply the edit, re-mine for the *next* failure, repeat. The composition curve is the output.

```bash
python failure_gym.py          # naive solver
python nullspace_gym.py 100    # null-space-constrained solver, lambda=100
```

Numpy only. Each run reproduces its report cards in a couple of minutes.

## Experiment 1 — naive minimal-norm edits

`W3[:, wrong] −= c·h2`, `W3[:, save] += c·h2` — the smallest edit that fixes the failure, ignoring every other state.

![Composition curve](assets/gym.png)

**The net absorbs zero clean edits.** Edit #1 fixes its failure but flips 64% of all probe decisions as collateral damage; mean return halves immediately. Edits #2–#7 ping-pong between a left-edge and a right-edge miss whose culprit hidden units overlap — each repair steals back the margin the previous one spent. Edit #8 fails the gate outright.

## Experiment 2 — the null-space fix

Constrain the repair: minimize `‖d‖² + λ·‖H·d‖²/N` subject to `d·h2_failure = gap + margin`, where H stacks the hidden states of thousands of probe states. Closed form: `d ∝ (I + λ·HᵀH/N)⁻¹·h2`. λ=0 is the naive edit; large λ forces the repair into directions the rest of the policy is insensitive to.

![Null-space frontier](assets/nullspace.png)

| λ | failures fixed | return after 8 edits (baseline 8.16) | median probe damage per edit |
|---|---|---|---|
| 0 (naive) | 7/8 | **2.38** | 20.5% |
| 1 | 6/8 | 2.67 | 2.8% |
| 10 | 6/8 | 6.14 | 6.0% |
| **100** | **8/8** | **7.34** | **1.8%** |
| 1000 | 8/8 | 5.00 | 3.4% |
| 10000 | 8/8 | 5.47 | 2.3% |

At **λ=100** the loop repairs all 8 failures in sequence — including the second, third, and eighth failures the naive solver either caused or couldn't fix — with ~2% collateral damage per edit and 90% of baseline performance retained after eight sequential surgeries.

## Why it matters

Weight-space editing promises behavior as inspectable, addressable mathematics: find the numbers responsible, change exactly those, verify nothing else moved. Experiment 1 shows the naive version of that promise fails even on a toy network — knowledge is stored in shared, overlapping features, so unconstrained edits are never local. Experiment 2 shows the fix is also mathematical: the same linear structure that makes failures diagnosable makes collateral damage *compressible into a constraint*. Diagnosis and containment come from the same equation.

Honest caveats: 17k parameters, final-layer edits only, greedy-play evaluation. Whether the loop scales to deeper edits and larger models is the open question — the gym is the instrument built to ask it.

## Layout

| Piece | What it is |
|---|---|
| `failure_gym.py` | Mine → diagnose → solve → gate → iterate, naive solver |
| `nullspace_gym.py` | Same loop, ridge-constrained (null-space) solver; λ from argv |
| `plot_nullspace.py`, `make_gym_figure.py` | Figure generators |
| `assets/game.py`, `assets/policy.py` | Authoritative deterministic Breakout physics + policy wrapper |
| `assets/policy_weights.npz` | The trained GridPolicyMLP weights (17,253 params) |
| `results_nullspace_*.json` | Raw run records for the λ sweep |

*Physics engine from [weight-space-breakout](https://app.primeintellect.ai/dashboard/environments/harleycooper/weight-space-breakout) (MIT). Built with [Kimi](https://kimi.com).*
