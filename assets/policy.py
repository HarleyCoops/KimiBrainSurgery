import json
from pathlib import Path
import numpy as np

SHAPES = {"W1": (500, 32), "b1": (32,), "W2": (32, 32),
          "b2": (32,), "W3": (32, 5), "b3": (5,)}


class Policy:
    def __init__(self, weights):
        if set(weights) != set(SHAPES):
            raise ValueError("Expected exactly W1, b1, W2, b2, W3, b3")
        self.weights = {}
        for name, shape in SHAPES.items():
            a = np.asarray(weights[name], dtype=np.float32)
            if a.shape != shape or not np.isfinite(a).all() or np.abs(a).max() > 100:
                raise ValueError(f"Invalid tensor {name}")
            self.weights[name] = a.copy()

    def logits(self, observation):
        x = np.asarray(observation, dtype=np.float32)
        if x.shape != (500,) or not np.isfinite(x).all():
            raise ValueError("Expected 500 finite observation values")
        w = self.weights
        h1 = np.tanh(x @ w["W1"] + w["b1"])
        h2 = np.tanh(h1 @ w["W2"] + w["b2"])
        return h2 @ w["W3"] + w["b3"]

    def probabilities(self, observation):
        z = self.logits(observation)
        p = np.exp(z - z.max())
        return p / p.sum()

    def act(self, observation, rng, mode="sample"):
        if mode not in ("sample", "greedy"):
            raise ValueError("mode must be sample or greedy")
        p = self.probabilities(observation)
        return int(rng.choice(5, p=p)) if mode == "sample" else int(p.argmax())

    def save(self, path):
        np.savez_compressed(path, **self.weights)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(dict(z))

    def sparse(self):
        return [{"tensor": name, "index": list(map(int, idx)), "value": float(a[tuple(idx)])}
                for name, a in self.weights.items() for idx in np.argwhere(a != 0)]


def tracking_policy():
    """Handwritten teaching controller, not a trained PPO checkpoint.

    One hidden path compares current ball column against paddle column.
    It does not predict future bounces or guarantee optimal play.
    """
    w = {k: np.zeros(s, dtype=np.float32) for k, s in SHAPES.items()}
    for r in range(1, 9):
        for c in range(1, 9):
            w["W1"][(r * 10 + c) * 5 + 1, 0] = c * 0.1
    for c in range(1, 9):
        w["W1"][(8 * 10 + c) * 5, 0] = -c * 0.1 / 3
    w["W2"][0, 0] = 4
    w["W3"][0, 3], w["W3"][0, 4] = -8, 8
    w["b3"][:3] = -3
    return Policy(w)


FAULTS = ("reversed_controls", "left_bias", "missing_right")


def faulty_policy(fault):
    p = tracking_policy()
    if fault == "reversed_controls":
        p.weights["W3"][0, 3:5] *= -1
    elif fault == "left_bias":
        p.weights["b3"][3] = 12
    elif fault == "missing_right":
        p.weights["W3"][0, 4] = 0
    else:
        raise ValueError("Unknown fault")
    return p


def apply_edits(base, payload, max_edits=64):
    """Data-only edits; never evaluate Python, expressions, or pickle."""
    if not isinstance(payload, dict) or set(payload) != {"edits"}:
        raise ValueError('Expected {"edits": [...]}')
    edits = payload["edits"]
    if not isinstance(edits, list) or len(edits) > max_edits:
        raise ValueError(f"At most {max_edits} scalar edits")
    p = Policy(base.weights)
    seen = set()
    for e in edits:
        if not isinstance(e, dict) or set(e) != {"tensor", "index", "value"}:
            raise ValueError("Each edit needs tensor, index, value")
        name, idx, value = e["tensor"], e["index"], e["value"]
        if not isinstance(name, str) or name not in SHAPES or not isinstance(idx, list):
            raise ValueError("Unknown tensor or invalid index")
        shape = SHAPES[name]
        if len(idx) != len(shape) or any(type(i) is not int or not 0 <= i < n for i, n in zip(idx, shape)):
            raise ValueError("Out-of-bounds index")
        if type(value) not in (int, float) or not np.isfinite(value) or abs(value) > 100:
            raise ValueError("Values must be finite numbers in [-100, 100]")
        key = (name, tuple(idx))
        if key in seen:
            raise ValueError("Duplicate edit")
        seen.add(key)
        p.weights[name][tuple(idx)] = value
    return p


def parse_submission(text):
    if not isinstance(text, str) or len(text) > 20000:
        raise ValueError("Submission exceeds 20,000 characters")
    text = text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d:
                raise ValueError("Duplicate JSON key")
            d[k] = v
        return d
    return json.loads(text, object_pairs_hook=pairs)


def repair_edits(fault):
    base, target = faulty_policy(fault), tracking_policy()
    return {"edits": [{"tensor": k, "index": list(map(int, i)), "value": float(target.weights[k][tuple(i)])}
                      for k in SHAPES for i in np.argwhere(base.weights[k] != target.weights[k])]}
