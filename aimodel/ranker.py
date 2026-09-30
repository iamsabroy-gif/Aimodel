"""A small neural network that learns from your feedback which evidence is good.

Each candidate sentence is described by a few numbers (how well it matches,
whether it's a definition, whether it came from your own data...). The
network starts neutral, and every /good or /bad trains it, so over time it
prefers the kind of evidence you like.
"""

from __future__ import annotations

import numpy as np

FEATURES = ("tfidf", "neural", "coverage", "lead", "definition", "length", "your_data", "web")


class FeedbackRanker:
    def __init__(self, hidden: int = 8, lr: float = 0.5, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.w1 = rng.normal(0, 0.5, (len(FEATURES), hidden))
        self.b1 = np.zeros(hidden)
        self.w2 = np.zeros(hidden)  # starts neutral: no adjustment until you give feedback
        self.b2 = 0.0
        self.examples = 0

    def _forward(self, x: np.ndarray):
        h = np.tanh(x @ self.w1 + self.b1)
        return h, h @ self.w2 + self.b2

    def adjust(self, features) -> np.ndarray:
        """Score adjustment in [-0.3, 0.3] for each row of features."""
        x = np.atleast_2d(np.asarray(features, dtype=float))
        return 0.3 * np.tanh(self._forward(x)[1])

    def train(self, features, labels, steps: int = 30) -> float:
        """Learn that rows labelled 1 are good evidence and 0 are bad. Returns loss."""
        x = np.atleast_2d(np.asarray(features, dtype=float))
        y = np.asarray(labels, dtype=float)
        if not len(x):
            return 0.0
        for _ in range(steps):
            h, z = self._forward(x)
            p = 1 / (1 + np.exp(-z))
            dz = (p - y) / len(x)
            dh = np.outer(dz, self.w2) * (1 - h ** 2)
            self.w2 -= self.lr * h.T @ dz
            self.b2 -= self.lr * dz.sum()
            self.w1 -= self.lr * x.T @ dh
            self.b1 -= self.lr * dh.sum(0)
        self.examples += len(x)
        p = 1 / (1 + np.exp(-self._forward(x)[1]))
        return float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)))

    def to_dict(self) -> dict:
        return {"w1": self.w1.tolist(), "b1": self.b1.tolist(), "w2": self.w2.tolist(),
                "b2": self.b2, "examples": self.examples}

    @classmethod
    def from_dict(cls, data: dict) -> "FeedbackRanker":
        r = cls()
        if data:
            r.w1, r.b1 = np.array(data["w1"]), np.array(data["b1"])
            r.w2, r.b2 = np.array(data["w2"]), float(data["b2"])
            r.examples = data.get("examples", 0)
        return r
