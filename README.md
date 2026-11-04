# KimiBrainSurgery

**An automated failure → diagnose → repair → verify loop that edits a neural network's weights directly — no gradient-descent retraining; W2 solvers use analytic derivatives — and measures how many repairs the network can absorb.**

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
python nullspace_gym.py 100    # null-space-constrained W3 solver, lambda=100
python w2_gym.py 100           # rank-1 constrained W2 solver, lambda=100
python fullrank_gym.py 100     # full-rank constrained W2 solver, lambda=100
python monotonic_gym.py 100    # full-rank + keep/rollback gate; writes best_weights.npz
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

## Experiment 3 — deeper cuts: editing W2 through the nonlinearity

Readout-layer edits are easy mode: `logits = h2 @ W3 + b3` is *linear* in W3, so the fix and its containment are exact. The real question is whether surgery works on a layer whose output passes through `tanh` before reaching the logits.

`w2_gym.py` answers it with a rank-1 edit `ΔW2 = h1_failure · gᵀ`. For any state, `Δz2 = (h1·h1_failure)·g` exactly and `Δh2 ≈ diag(1−h2²)·Δz2` to first order — so the probe-protection matrix is *diagonal* and the constrained solve is still one line. A trust region (max ‖ΔW2‖ per step) plus Newton refinement handles the linearization error.

![W2 frontier](assets/w2.png)

| λ | failures fixed | return after 8 edits | median probe damage | median Newton steps |
|---|---|---|---|---|
| 0 (naive rank-1) | 5/8 | 6.38 | 5.7% | 1 |
| 1 | 4/8 | 3.38 | 14.2% | 4 |
| 10 | 5/8 | 3.22 | 7.3% | 5 |
| 100 | 4/8 | 6.50 | 9.0% | 5 |
| 1000 | 4/8 | 6.50 | 9.1% | 5 |
| 10000 | 4/8 | 6.50 | 9.1% | 5 |
| *W3, λ=100 (for contrast)* | *8/8* | *7.34* | *1.8%* | *— (exact)* |

Three honest findings:

1. **Deeper surgery is genuinely harder.** Best-case W2 fixes half the failures W3-constrained fixes, with ~5× the collateral damage. The diagnosis lives in h2 space, but a W2 edit *changes h2 itself* — you are operating on the feature you measured with.
2. **The λ sweep saturates**: λ=100, 1000, 10000 produce identical curves because the trust region, not the constraint, becomes the binding limit. Once linearization error dominates, buying more probe protection no longer helps.
3. **Naive rank-1 W2 (λ=0) beats naive W3** (5.7% vs 20.5% median damage): rank-1 edits couple to other states only through h1-similarity, which is naturally sparse. Structure can substitute for constraints — a hint that the right *parameterization* matters as much as the right penalty.

## Experiment 4 — full-rank W2: surgery that makes the policy *better than it was*

The rank-1 restriction was the prime suspect for W2's underperformance. `fullrank_gym.py` frees all 1024 entries of ΔW2. The closed form survives: with `vec(ΔW2)` stacking columns, the probe penalty is block-diagonal per output unit (`G_j = (1/N)Σ_p D_p[j]² h1_p h1_pᵀ`), so the constrained solve is 32 small linear systems, not one 1024×1024 one. Trust region + Newton refinement as before.

![Full-rank frontier](assets/fullrank.png)

| λ | failures fixed | return after 8 edits (baseline 8.16) | median probe damage |
|---|---|---|---|
| 0 | 5/8 | 6.38 | 5.7% |
| 1 | 4/8 | 4.80 | 7.7% |
| 10 | 6/8 | 7.86 | 3.1% |
| **100** | 4/8 | **8.44** (peak **11.75** after edit #6) | **2.4%** |
| 1000 | 2/8 | 8.44 | 3.3% |
| 10000 | 0/8 | 4.05 | 4.8% |
| *W3 λ=100 (contrast)* | *8/8* | *7.34* | *1.8%* |

**The composition curve inverts.** At λ=100 the sequential edits don't just avoid damage — they compound: 8.16 → 8.19 → 9.72 → 9.58 → 11.61 → 11.64 → **11.75**, +44% over the trained baseline after six surgeries, with ~2% probe damage per edit. The loop stops behaving like repair and starts behaving like *training by other means*: each diagnosed failure is a curriculum item, and each closed-form edit is a lesson that sticks.

Three honest observations:

1. **The gate and the game disagree, in both directions.** Edits #1–#3 at λ=100 are marked `fixed=False` — the exact failure state's argmax never entered the saving set (tanh saturation shrinks the constraint direction, and 8 trust-region steps can't reach it) — yet return *rose* through them. Partial fixes that move the decision boundary toward the saving action still improve play even when the binary gate says "not fixed."
2. **It still ends in interference.** Edit #8 (a re-mined variant of the seed-12 right-edge miss from #2/#3) knocks return from 10.47 back to 8.44. The ping-pong failure mode from Experiment 1 is down, not out.
3. **λ is now a capability knob, not just a safety knob.** λ=10000 fixes nothing (constraint strangles the edit) and ends at 4.05 return versus the 8.16 baseline; λ=100 fixes fewer binary gates than W3 yet produces the best-playing network this project has ever had. "How many edits can it absorb" was the wrong question for deep layers — the right one is "what does the policy become."

## Experiment 5 — the monotonic loop: nondecreasing return on the reused gate

`monotonic_gym.py` adds the keep/rollback rule to the full-rank solver: an edit is applied only if gated mean return does not decrease, otherwise it is rolled back and the failure is marked **unresolved** — reported, never silently dropped. Previously repaired failures are retested after every kept edit, and a stall is reported as a *solver* stall, not proof of exhausted capacity. Because the evaluation is deterministic (greedy play, fixed seeds), the rule is exact.

Result at λ=100: 3 of 4 proposed edits kept, one rolled back (the persistent seed-12 right-edge miss), and the run ends because the miner finds **no new avoidable failures** in 200 seeds. The composition curve is monotonic *on the measured gate* — a finite gate can miss regressions elsewhere, so the run closes with an untouched held-out evaluation on seeds never used for mining or gating:

| | gate seeds (used in loop) | held-out seeds (never seen) |
|---|---|---|
| original | 8.16 | 8.56 |
| repaired (3 kept edits) | **9.83** | **9.06** |

The gain generalizes. The repaired weights are published as
[`grid-policy-mlp-repaired`](https://huggingface.co/HarleyCooper/grid-policy-mlp-repaired),
and the [live visualizer](https://harleycoops.github.io/breakout-policy-visualizer/) has a
weights selector so you can watch the original and repaired policies play side by side.

### Paired per-seed verification (1,000 held-out games)

`paired_eval.py` replays both checkpoints on the same held-out games (seeds 20000–20999,
greedy, never mined or gated on) and stores every per-seed score in
`eval/paired_heldout.json`. Exact checkpoints, hashed for reproduction:

| checkpoint | sha256 (first 16) |
|---|---|
| `policy_weights.npz` (original) | `d3cd5623fdae730f` |
| `best_weights.npz` (repaired) | `9c338f7c08c9590a` |

Results over 1,000 paired games:

| metric | value |
|---|---|
| baseline mean return | 8.69 |
| repaired mean return | 9.77 |
| paired difference | **+1.08 ± 3.74 (sd)** |
| bootstrap 95% CI on diff | **[+0.85, +1.31]** |
| wins / losses / ties | **137 / 0 / 863** |
| sign test p | < 10⁻⁴ |

The picture is unambiguous: the repaired policy is **never worse on any of 1,000 games**
and strictly better in 137 of them — an observed paired improvement on this finite set, concentrated in the
games where the original policy had the specific failure the gym repaired. (An earlier
64-game pilot showed the same shape — 4/0/60 — but with a CI that barely excluded zero;
the larger set tightens the estimate from +0.50 [+0.02, +1.20] to +1.08 [+0.85, +1.31].)

## Experiment 6 — chasing the last failure, and finding the miner's blind spot

The monotonic loop left one failure unresolved (the seed-12 right-edge miss; its W2 edits
all tripped the gate). `chase_unresolved.py` attacks it with a solver grid — W2 and W3,
λ ∈ {10, 100, 1000}, margin ∈ {0.1, 0.5, 1, 2} — gated on return as usual. Every W2 config
fails to fix it; **every W3 null-space config fixes it** (the readout layer was the right
tool for this one). Best: W3 λ=10, return 9.83 → 9.95. One more monotonic W2 step on top
reaches 10.06 gated. Those weights are `best_weights_v3.npz`
([grid-policy-mlp-repaired-v3](https://huggingface.co/HarleyCooper/grid-policy-mlp-repaired-v3)).

But the 1,000-game paired eval surfaces what the gate missed:

| model | mean (1000 held-out) | wins / losses / ties vs original |
|---|---|---|
| original | 8.69 | — |
| v1 (published) | 9.77 | 137 / **0** / 863 |
| v3 | **10.53** | 137 / **17** / 846 |

All 17 v3 losses are the same game shape: baseline scores 13, v3 scores 0. Tracing one
(seed 20021): the original policy pre-positions the paddle while the ball is *ascending*;
v3 has unlearned that habit, so by the time the ball descends the miss is already
unavoidable. **The miner only inspects descending balls, so failure-to-pre-position is a
structural blind spot** — the failure happens at a node the failure definition never visits.
The fix for the last descending-ball failure cost an ascending-ball behavior, and no part
of the mine → diagnose → solve loop could see it. Only the paired eval could.

v1 has zero observed losses on these 1,000 games; that is not a never-worse guarantee.
v3 has higher mean return on this set but 17 regressions. Experiment 7 extends
the miner to ascent-phase positioning failures.

## Experiment 7 — closing the blind spot: the flight miner

Experiment 6 ended with a structural admission: the miner only watched *descending* balls,
so failing to pre-position the paddle during ascent was invisible. The fix is exact
simulation: since the paddle only interacts with the ball at row 8, the ball's entire
flight — ascent, brick bounces, wall bounces, brick removals — is deterministic given the
current state. `flight_landing()` in `failure_gym.py` simulates it to the next row-8
arrival, and `mine(use_flight=True)` flags ANY decision node (ascent included) where the
chosen action makes that arrival uncoverable. The 13→0 seeds that yielded zero failures
under the descent miner immediately produce clean failure cards (e.g. seed 20021 step 1:
"ball will land col 8, paddle center 2, policy chose no-op — right saves it").

A monotonic run from v3 with the flight miner keeps 6 of 12 edits and produces **v4**:

| model | mean, seeds 20000-20999 | mean, seeds 30000-30999 (fresh) | W/L/T (fresh) |
|---|---|---|---|
| original | 8.69 | 8.89 | — |
| v1 | 9.77 | — | not evaluated on fresh set here |
| v3 | 10.53 | — | not evaluated on fresh set here |
| **v4** | **12.18** (+3.49) | **12.19** (+3.30) | 253 / 20 / 727 |

The fresh set supports a positive mean gain. Agreement with the development set does not rule out gate overfitting or regressions elsewhere.

**But the blind-spot class itself resists repair.** A solver grid aimed directly at the
seed-20021 ascent failure (W2 and W3, λ ∈ {10, 100, 1000}) fixes the state every time —
and loses 1.5-2.6 gated return doing it: 460-2,300 probe decisions flip for one ascent
state. The failure is *detectable* but not *cheaply correctable* by closed-form edits.
v4 still carries the 13→0 class (~2% of games). The loop's verdict: detection is solved;
correction needs either richer edits (multi-state batch constraints) or a parameter
budget the readout/deep layers don't have. That is now the sharpest open question in the
repo.

## Experiment 8 — the batch solver: one edit for the whole class, and the price of the last 2%

Experiment 7 ended with the sharpest open question: the 13→0 ascent class is detectable but not
cheaply correctable state by state. The natural escalation is to stop paying per state and solve
for the whole class at once. `batch_gym.py` collects every flight-mined failure under v4
(7 states: 2 need `right`, 5 need `left`), stacks their linearized constraints into one system
A d = b over the readout column, and solves the regularized problem in closed form:
d = M⁻¹Aᵀ(AM⁻¹Aᵀ)⁻¹b, M = I + λHᵀH/N. One solve, one edit, every constraint satisfied — in
theory.

**The tested W3 edits have a severe interference cost.** The five left-class
constraints require large edit norms (roughly 14–17). This does not establish
linear inseparability. The implementation solves separate saving-column equality
systems; those equalities do not constrain every competing logit after all column
updates. The exact lambda-100 candidate fixes 5/7 states but drops gate return
12.09375 → 2.40625 and flips 5,561/13,650 probe decisions. A left-only soft
candidate fixes 1/7 at unchanged gate mean, with 174 probe flips.

**Moving one layer up does not rescue it.** `batch_fullrank_gym.py` frees all 1,024 entries of
W2 (block-diagonal Gram structure, K×K closed form, Newton re-linearization with a trust
region), asking the hidden representations themselves to move apart so the existing W3 can
separate them. The frontier moves but the shape does not:

| solver | config | constraints fixed | probe flips | gate return |
|---|---|---|---|---|
| — (v4 baseline) | — | 0/7 | 0 | **12.09** |
| W3 exact batch | λ=100 | 5/7 | 5,561 | 2.41 |
| W3 soft batch | λ=50 | 1/7 | 933 | 11.30 |
| W3 soft batch, left only | λ=100 | 1/7 | 174 | 12.09 |
| W2 full-rank batch | λ=1e3, cap 0.5 | 5/7 | 1,750 | 7.31 |
| W2 full-rank batch | λ=1e5, cap 0.5 | 2/7 | 834 | 9.47 |
| W2 full-rank batch | λ=1e5, cap 0.1 | 0/7 | 287 | 12.09 |

None of the 18 configurations fixes all seven states while maintaining gate
mean return, so no v5 was written. Some small trust-region W2 runs move parameters
without fixing a state. These are results for the tested objectives, layer choices,
regularization strengths, and step budgets; they are not an impossibility proof.

**Verdict: the tested batch repairs do not meet the promotion rule.** Shared
features and interference are plausible explanations. The experiment does not
prove that larger capacity is necessary or that gradient training will succeed.
The 1/7 fixed, equal-return candidate is a counterexample to an unconditional
claim that every extra repaired state costs return; it still fails the stricter
all-seven requirement. v4 remains published.

The batch miner uses eight loss seeds from the former v4 fresh set (30000 range).
Those examples are now development data for later experiments. Preserve the
frozen v4 result historically and reserve a new untouched set before testing v5.

All 18 sweep configurations are logged to W&B (project `kimi-brain-surgery`, group
`batch-solver-sweep`) and recorded in `results_batch_sweep.json`. Gate caveat: 64 seeds,
greedy play; "not fixable" means not fixable by these linearized closed-form edits at the
listed regularization strengths and solver settings.

## Experiment 9 — the complete chain, and why repairing the decision is not enough

The chain Chris specified, built as `chain_gym.py` and run continuously from v4:
identify an avoidable failure in the current accepted policy, repair it, preserve
previously protected behavior, pass a fresh evaluation, repeat. Each attempt mines a
flight-phase failure, then walks a repair ladder of increasing expressiveness —
W3 null-space closed form, W2 full-rank closed form, elastic-anchored gradient hinge —
and the FIRST candidate that passes a hard gate wins. The gate is strict: target state
fixed, ZERO regressions on every previously repaired state, gate-seed mean return not
down. Deferred failures are not retried; a batch-gradient phase fires once per stall.

Result of the first full run (9 attempts, 30 ladder evaluations):

| outcome | count |
|---|---|
| edits kept (all w2-fullrank, 91-251 flips, zero regressions) | 3 |
| candidates rejected by the gate | 27 |
| failures deferred as unfixable-within-gate | 6 |
| gate return | 12.09 -> 12.09 (never worse, by construction) |
| fresh paired eval, seeds 40000-40499 (untouched until the end) | +0.000, 500/500 ties |

The gate did its job — the chain is monotonic on the measured gate and every kept edit
is a verified decision fix. And then the fresh eval exposed something deeper: **the three
repaired games score exactly the same as before.** Not approximately — identically, 300/300
development seeds and 500/500 fresh seeds, even though the kept edits are real (max |dW2|
= 0.16) and the failure states verifiably flip to the saving action.

Replaying the repaired games frame by frame shows why. Seed 9, ball ascending, lands col 1
in 5 ticks, paddle at 7 — the rescue requires five consecutive `left` moves:

    step 17: left   <- the repaired decision (was: up)
    step 18: left
    step 19: left
    step 20: no-op  <- the follow-through defect
    step 21: left   <- too late; paddle stops at 4, ball lands at 1

One defection at step 20 wastes the entire rescue. Same story on seeds 1 and 12. The chain
repaired the entry node of each failure, but a rescue is a *trajectory*: the saving action
must be taken at every node until the ball is caught, and nothing constrains the policy
downstream of the repaired state. This also retroactively explains the Experiment 8
frontier — the residual 13->0 class is precisely the set of failures whose rescue requires
multi-step commitment. A single-node constraint does not constrain the successor
states by construction, and none of the single-node solvers tested here supplied the
follow-through. Whether a sufficiently clever single-node edit could is open; the
observed defect pattern points at trajectory constraints as the direct fix.

The fix is well-defined: trajectory-constrained repair. When a failure is mined, roll the
rescue prefix forward under forced saving actions, collect every successor state until the
catch, and add each as a constraint in the same solve — the policy must hold the commitment,
not just enter it. That is Experiment 10.

`chain_final.npz` (v4 + 3 gated decision repairs) is kept in the repo as the accepted
checkpoint; no v5 was published, because the fresh-seed gain is zero and the honest model
is the one that earned it.

## Experiment 10 — trajectory constraints fix follow-through, but the gate overfits

`trajectory_gym.py` implements the direct fix proposed above. Starting from `chain_final.npz`,
it replays each mined failure under forced saving actions until the next row-8 catch, keeps
every successor state as a constraint, and requires an actual greedy rescue replay. The repair
ladder uses an exact whole-`W3` multi-constraint solve, full-rank multi-constraint `W2`, then
an elastic-anchored gradient hinge. Previously protected trajectories are retested after every
candidate. Unlike earlier per-save-column batching, the `W3` solve directly controls each
`logit[wrong] - logit[save]` gap across the complete readout edit.

The known follow-through defect is repaired. On seed 9, the accepted starting policy scores 2;
the final trajectory candidate scores **17**. Attempt 5 fixes the step-20 defection with `W2`,
and attempt 9 repairs the re-mined step-17 entry decision with `W3`. Both actual rescue replays
contain only saving actions. Across 9 attempts, 2 of 24 candidates pass the hard gate and gate
return rises 12.09 → 13.72.

The untouched evaluation rejects the candidate anyway:

| metric, seeds 50000-50499 | start (`chain_final`) | trajectory final |
|---|---:|---:|
| mean return | **12.472** | **12.088** |
| paired difference vs start | — | **-0.384** |
| wins / losses / ties vs start | — | **159 / 144 / 197** |

Development seeds 0-299 also fall by 0.637 mean return (95 wins, 89 losses, 116 ties). Thus the
64 reused gate seeds are not a reliable proxy for whole-policy improvement: local trajectory
verification is necessary to fix follow-through, but it does not contain off-target behavior.
`trajectory_final.npz` is retained as a rejected research candidate; **no v5 was written**, under
the stricter promotion rule of positive fresh mean gain plus zero fresh losses. Raw candidate
history is in `results_trajectory.json`, and paired per-seed scores are in
`eval/paired_trajectory_fresh.json`. The next cycle should isolate the two accepted edits on a
new validation set and roll back harmful trajectory repairs before another untouched final eval.

## Why it matters

Weight-space editing promises behavior as inspectable, addressable mathematics: find the numbers responsible, change exactly those, verify nothing else moved. Experiment 1 shows the naive version of that promise fails even on a toy network — knowledge is stored in shared, overlapping features, so unconstrained edits are never local. Experiment 2 shows the fix is also mathematical: the same linear structure that makes failures diagnosable makes collateral damage *compressible into a constraint*. Diagnosis and containment come from the same equation. Experiment 3 marks the current boundary: at the readout layer the loop is exact and composes; one nonlinearity deeper, the measurement instrument and the edit start interfering with each other.

Honest caveats: 17k parameters, greedy-play evaluation, W2/W3 edits only (W1 is untouched). Whether the loop scales to larger models is the open question — the gym is the instrument built to ask it.

## Layout

| Piece | What it is |
|---|---|
| `failure_gym.py` | Mine → diagnose → solve → gate → iterate, naive solver |
| `nullspace_gym.py` | Same loop, ridge-constrained (null-space) W3 solver; λ from argv |
| `w2_gym.py` | Rank-1 probe-constrained W2 solver with trust region + Newton refinement |
| `fullrank_gym.py` | Full-rank constrained W2 solver via block-diagonal Gram structure |
| `monotonic_gym.py` | Full-rank solver + keep/rollback gate, regression retests, held-out final eval |
| `chain_gym.py` | The complete chain: mine, repair (W3/W2/gradient ladder), hard gate, fresh eval, repeat |
| `trajectory_gym.py` | Experiment 10: trajectory constraints, actual rescue replays, protected-trajectory retests |
| `test_trajectory_gym.py` | Regression tests for forced rescue construction, replay, and exact W3 constraints |
| `batch_gym.py` / `batch_fullrank_gym.py` | Batch-constrained solvers: W3 exact/soft and full-rank W2, one edit per failure class |
| `chase_unresolved.py` | Solver-grid attack on the one failure the monotonic loop rolled back |
| `paired_eval.py` | Paired per-seed held-out eval with bootstrap CI + sign test |
| `plot_nullspace.py`, `make_gym_figure.py`, `plot_w2.py`, `plot_fullrank.py` | Figure generators |
| `assets/game.py`, `assets/policy.py` | Authoritative deterministic Breakout physics + policy wrapper |
| `assets/policy_weights.npz` | The trained GridPolicyMLP weights (17,253 params) |
| `results_nullspace_*.json`, `results_w2_*.json`, `results_fullrank_*.json`, `results_monotonic_*.json` | Raw run records for all sweeps |
| `best_weights.npz` / `best_weights_v3.npz` | The repaired checkpoints (v1 = zero observed old-set losses; v3 = 17 old-set regressions; v4 available separately) |
| `eval/paired_heldout*.json` | Paired per-seed held-out scores, 1000 games |

*Physics engine from [weight-space-breakout](https://app.primeintellect.ai/dashboard/environments/harleycooper/weight-space-breakout) (MIT). Built with [Kimi](https://kimi.com).*

## Evidence reconciliation — 2026-11-03

[Consolidated status](https://github.com/HarleyCoops/FailureIsAllYouNeed/blob/codex/initial-results/docs/STATUS_2026-11-03.md) · [Dataset](https://huggingface.co/datasets/HarleyCooper/failure-is-all-you-need-evaluations) · [Full mathematics](https://github.com/HarleyCoops/FailureIsAllYouNeed/blob/codex/initial-results/docs/KIMI_METHODS.md).

The monotonic rule checks mean return on 64 reused gate seeds. Failure repair,
probe flips, and repaired-state retests are diagnostics, not additional acceptance
requirements. Only one of the three kept v1 edits reports `fixed=True`.
The 18 sweep runs apply all candidates, including regressions.

The v4 old-set file contains all 1,000 games, with 17 failures informing development.
The v1 1,000-game set includes its earlier 64-game pilot. These are not fully
independent replications. Finite evaluations cannot prove an absence of regressions.

Cross-repo solver results must preserve checkpoint identity, action mode, float
precision, and random-number protocol. The Qwen patch improves sampled return but
regresses in our local greedy check. Its three edited parameters also change argmax
decisions. A reported repaired seed-314159 free-play return of 14 has not been
reproduced under the frozen Qwen protocol. `cross_repo2.py` reuses 256 Qwen audit
seeds for selection, making them development data for that later search.
See [reconciliation and exact local evidence](https://github.com/HarleyCoops/FailureIsAllYouNeed/blob/codex/initial-results/docs/CROSS_REPO_RECONCILIATION.md).
