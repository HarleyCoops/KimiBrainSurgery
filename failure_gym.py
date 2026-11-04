"""failure_gym — an automated failure -> diagnose -> edit -> verify loop for GridPolicyMLP.

Like an RL gym, except the training signal is failures:
  1. MINE      greedy rollouts over many seeds; harvest avoidable misses (state, wrong action, saving actions)
  2. DIAGNOSE  exact logit decomposition per hidden unit (logits = h2 @ W3 + b3 is linear in h2)
  3. SOLVE     minimal-norm closed-form edit on W3 columns: +/- c * h2(failure state)
  4. GATE      regression battery: failure fixed? probe-state decisions unchanged? seed return not degraded?
  5. ITERATE   apply the edit, re-mine for the NEXT failure — measure how many repairs the net absorbs
"""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


game_mod = load("game", ROOT / "assets" / "game.py")
policy_mod = load("policy", ROOT / "assets" / "policy.py")
Breakout = game_mod.Breakout
Policy = policy_mod.Policy

ACTIONS = ("no-op", "up", "down", "left", "right")
PADDLE_MOVES = {0: 0, 1: 0, 2: 0, 3: -1, 4: +1}

W0 = np.load(ROOT / "policy_weights.npz")
BASE_WEIGHTS = {k: W0[k].astype(np.float64) for k in ("W1", "b1", "W2", "b2", "W3", "b3")}


# ---------- forward pass helpers (float64, exact) ----------
def hidden_and_logits(weights, obs):
    h1 = np.tanh(obs @ weights["W1"] + weights["b1"])
    h2 = np.tanh(h1 @ weights["W2"] + weights["b2"])
    return h2, h2 @ weights["W3"] + weights["b3"]


# ---------- 1. MINE ----------
class Failure:
    def __init__(self, seed, step, obs, paddle, ball, landing_col, saving, wrong, logits, h2):
        self.seed, self.step, self.obs = seed, step, obs
        self.paddle, self.ball, self.landing_col = paddle, ball, landing_col
        self.saving, self.wrong = saving, wrong
        self.logits, self.h2 = logits, h2

    def card(self):
        return (f"seed {self.seed} step {self.step}: ball at col {self.ball[1]} falling into col "
                f"{self.landing_col}, paddle center {self.paddle} -> policy chose '{ACTIONS[self.wrong]}' "
                f"(logits {np.round(self.logits, 2)}), any of {[ACTIONS[a] for a in self.saving]} saves it")


def saving_actions(paddle, landing_col):
    """Actions whose resulting 3-wide paddle (center +/- 1) covers the landing column."""
    return [a for a in (0, 3, 4)
            if abs(int(np.clip(paddle + PADDLE_MOVES[a], 2, 7)) - landing_col) <= 1]


def landing_column(game):
    """If the ball's current descent reaches row 8 uninterrupted, return (landing_col, ticks_to_arrival).
    A brick hit flips the ball back up, so then there is no landing to defend (returns None)."""
    if game.dy <= 0:
        return None
    r, c, dx = game.row, game.col, game.dx
    bricks = set(game.bricks)
    while True:
        if not 1 <= c + dx <= 8:
            dx = -dx
        nr, nc = r + 1, c + dx
        if (nr, nc) in bricks:
            return None  # descent interrupted; ball heads back up
        r, c = nr, nc
        if r == 8:
            return c, 8 - game.row


def flight_landing(game, cap=1000):
    """Exact ball-only forward simulation, ANY phase (ascent, brick bounces, wall bounces),
    until the ball next reaches row 8. Returns (landing_col, ticks_to_arrival).

    Mirrors Breakout.step's collision logic minus paddle/reward: the paddle only matters
    at row 8, so the landing column is fully determined by the current state. Brick
    removals are tracked because they change later bounces within the same flight.
    """
    r, c, dy, dx = game.row, game.col, game.dy, game.dx
    bricks = set(game.bricks)
    for t in range(1, cap + 1):
        if not 1 <= c + dx <= 8:
            dx *= -1
        if r + dy < 1:
            dy *= -1
        v = (r + dy, c)
        h = (r, c + dx)
        hits = set()
        hit_v, hit_h = v in bricks, h in bricks
        if hit_v:
            hits.add(v); dy *= -1
        if hit_h:
            hits.add(h); dx *= -1
        if not hit_v and not hit_h:
            diagonal = (r + dy, c + dx)
            if diagonal in bricks:
                hits.add(diagonal); dy *= -1
        bricks.difference_update(hits)
        if not 1 <= c + dx <= 8:
            dx *= -1
        if r + dy < 1:
            dy *= -1
        nr, nc = r + dy, c + dx
        if (nr, nc) in bricks:
            nr, nc = r, c
        if nr >= 8:
            return nc, t
        r, c = nr, nc
    return None


def salvageable_after(paddle_after, ticks_left, landing_col):
    """Can a paddle centered at paddle_after still cover landing_col with ticks_left moves remaining?"""
    lo = max(2, paddle_after - ticks_left)
    hi = min(7, paddle_after + ticks_left)
    return lo <= landing_col + 1 and hi >= landing_col - 1


def mine(weights, seeds, max_steps=200, use_flight=False):
    """Greedy rollouts; a FAILURE is the exact node where the ball's next row-8 arrival
    went from coverable to uncoverable because of the chosen paddle action.

    use_flight=False: descent-only (landing_column) — the original miner, which is blind
    to ascent-phase pre-positioning. use_flight=True: full flight simulation
    (flight_landing), so a failure during ASCENT is also caught — e.g. failing to move
    toward the future landing column while the ball is still bouncing off bricks."""
    failures, probes, seen = [], [], set()
    land_fn = flight_landing if use_flight else landing_column
    for seed in seeds:
        game = Breakout(max_steps)
        game.reset(seed)
        while not (game.terminated or game.truncated):
            obs = game.observation()
            h2, logits = hidden_and_logits(weights, obs)
            action = int(logits.argmax())
            probes.append(obs)
            land = land_fn(game)
            if land is not None:
                landing_col, ticks = land
                paddle_after = int(np.clip(game.paddle + PADDLE_MOVES[action], 2, 7))
                if not salvageable_after(paddle_after, ticks - 1, landing_col):
                    saving = [a for a in (0, 3, 4)
                              if salvageable_after(int(np.clip(game.paddle + PADDLE_MOVES[a], 2, 7)),
                                                   ticks - 1, landing_col)]
                    if saving and action not in saving:
                        key = obs.tobytes()
                        if key not in seen:
                            seen.add(key)
                            failures.append(Failure(seed, game.steps, obs, game.paddle,
                                                    (game.row, game.col), landing_col,
                                                    saving, action, logits, h2))
            game.step(action)
    return failures, probes


# ---------- 2. DIAGNOSE ----------
def diagnose(weights, failure):
    """Exact per-unit contribution to the fatal logit gap (wrong minus best-saving)."""
    best_save = max(failure.saving, key=lambda a: failure.logits[a])
    gap_per_unit = failure.h2 * (weights["W3"][:, failure.wrong] - weights["W3"][:, best_save])
    order = np.argsort(gap_per_unit)[::-1]
    return best_save, [(int(j), float(gap_per_unit[j])) for j in order[:3]]


# ---------- 3. SOLVE ----------
def solve(weights, failure, best_save, margin=0.1):
    """Minimal-norm W3 edit flipping the decision at the failure state:
    W3[:, wrong] -= c*h2, W3[:, save] += c*h2 with c = (gap + margin) / (2 ||h2||^2)."""
    gap = failure.logits[failure.wrong] - failure.logits[best_save]
    c = (gap + margin) / (2.0 * float(failure.h2 @ failure.h2))
    edits = {"W3": [(failure.wrong, -c), (best_save, +c)]}
    new = {k: v.copy() for k, v in weights.items()}
    new["W3"][:, failure.wrong] -= c * failure.h2
    new["W3"][:, best_save] += c * failure.h2
    return new, edits, float(c)


# ---------- 4. GATE ----------
def gate(base_weights, new_weights, failure, best_save, probes, eval_seeds):
    h2, logits = hidden_and_logits(new_weights, failure.obs)
    fixed = int(logits.argmax()) in failure.saving
    flips = 0
    for obs in probes:
        _, l0 = hidden_and_logits(base_weights, obs)
        _, l1 = hidden_and_logits(new_weights, obs)
        flips += int(l0.argmax() != l1.argmax())

    def mean_return(weights):
        pol = Policy({k: weights[k].astype(np.float32) for k in weights})
        rets = []
        for s in eval_seeds:
            g = Breakout(200)
            g.reset(int(s))
            while not (g.terminated or g.truncated):
                g.step(pol.act(g.observation(), None, "greedy"))
            rets.append(g.total)
        return float(np.mean(rets))

    return {"fixed": bool(fixed), "probe_flips": flips, "probe_size": len(probes),
            "return_before": mean_return(base_weights), "return_after": mean_return(new_weights)}


# ---------- 5. ITERATE ----------
def run(max_repairs=8, mine_seeds=range(0, 200), eval_seeds=range(10000, 10064)):
    weights = {k: v.copy() for k, v in BASE_WEIGHTS.items()}
    history = []
    for repair_idx in range(max_repairs):
        failures, probes = mine(weights, mine_seeds)
        if not failures:
            print(f"[gym] no avoidable greedy failures found after {repair_idx} repairs")
            break
        f = failures[0]
        best_save, culprits = diagnose(weights, f)
        new_weights, edits, c = solve(weights, f, best_save)
        report = gate(weights, new_weights, f, best_save, probes, eval_seeds)
        history.append({"repair": repair_idx + 1, "failure": f, "best_save": best_save,
                        "culprits": culprits, "c": c, **report})

        print(f"\n=== FAILURE #{repair_idx + 1} ===")
        print("  " + f.card())
        print(f"  diagnosis: top hidden units driving the wrong call: "
              + ", ".join(f"unit {j} ({v:+.2f})" for j, v in culprits))
        print(f"  edit: W3[:, '{ACTIONS[f.wrong]}'] -= {c:.4f}*h2, "
              f"W3[:, '{ACTIONS[best_save]}'] += {c:.4f}*h2  (minimal-norm, closed form)")
        print(f"  gate: failure fixed={report['fixed']} | probe decisions flipped "
              f"{report['probe_flips']}/{report['probe_size']} | mean return "
              f"{report['return_before']:.2f} -> {report['return_after']:.2f}")
        weights = new_weights
    return history


if __name__ == "__main__":
    run()
