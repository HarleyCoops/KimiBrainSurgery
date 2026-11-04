"""Version 1 deterministic grid physics, with randomness only at reset."""
import numpy as np

ACTIONS = ("no-op", "up", "down", "left", "right")


class Breakout:
    def __init__(self, max_steps=200):
        if not isinstance(max_steps, int) or not 1 <= max_steps <= 10000:
            raise ValueError("max_steps must be an integer in [1, 10000]")
        self.max_steps = max_steps
        self.reset()

    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        self.paddle = int(rng.integers(2, 8))
        self.row, self.col = 6, int(rng.integers(1, 9))
        self.dy, self.dx = -1, int(rng.choice([-1, 1]))
        self.bricks = {(r, c) for r in range(1, 4) for c in range(1, 9)}
        self.previous = np.zeros((10, 10), dtype=np.float32)
        self.steps, self.total = 0, 0.0
        self.terminated = self.truncated = False
        self.reason = "playing"
        return self.observation(), self.info()

    def observation(self):
        x = np.zeros((10, 10, 5), dtype=np.float32)
        x[8, self.paddle-1:self.paddle+2, 0] = 1
        x[self.row, self.col, 1] = 1
        for r, c in self.bricks:
            x[r, c, 2] = 1
        x[0, :, 3] = x[9, :, 3] = x[:, 0, 3] = x[:, 9, 3] = 1
        x[:, :, 4] = self.previous
        return x.reshape(500)

    def info(self):
        return {"step": self.steps, "return": self.total,
                "bricks_remaining": len(self.bricks), "reason": self.reason}

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError("Episode finished; call reset before step")
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)) or not 0 <= action < 5:
            raise ValueError("action must be an integer 0..4")
        x = self.observation().reshape(10, 10, 5)
        self.previous = np.maximum(x[:, :, 0], x[:, :, 1]).copy()
        self.paddle = int(np.clip(self.paddle + int(action == 4) - int(action == 3), 2, 7))
        if not 1 <= self.col + self.dx <= 8:
            self.dx *= -1
        if self.row + self.dy < 1:
            self.dy *= -1
        # Resolve vertical/horizontal faces before the diagonal cell.
        v = (self.row + self.dy, self.col)
        h = (self.row, self.col + self.dx)
        hits = set()
        hit_v, hit_h = v in self.bricks, h in self.bricks
        if hit_v:
            hits.add(v)
            self.dy *= -1
        if hit_h:
            hits.add(h)
            self.dx *= -1
        if not hit_v and not hit_h:
            diagonal = (self.row + self.dy, self.col + self.dx)
            if diagonal in self.bricks:
                hits.add(diagonal)
                self.dy *= -1
        self.bricks.difference_update(hits)
        # A brick next to an outer wall can reflect back into that wall.
        if not 1 <= self.col + self.dx <= 8:
            self.dx *= -1
        if self.row + self.dy < 1:
            self.dy *= -1
        nr, nc = self.row + self.dy, self.col + self.dx
        # If the reflected destination is occupied, remain in place this tick.
        if (nr, nc) in self.bricks:
            nr, nc = self.row, self.col
        reward = float(len(hits))
        if nr >= 8:
            nr = 8
            if abs(nc - self.paddle) <= 1:
                self.dy = -1
            else:
                reward -= 1
                self.terminated, self.reason = True, "miss"
        self.row, self.col = nr, nc
        self.steps += 1
        self.total += reward
        if not self.bricks and not self.terminated:
            self.terminated, self.reason = True, "cleared"
        self.truncated = self.steps >= self.max_steps and not self.terminated
        if self.truncated:
            self.reason = "time_limit"
        return self.observation(), reward, self.terminated, self.truncated, self.info()
