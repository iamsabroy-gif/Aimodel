"""Core learning model.

The model learns in two ways, both from what you type:

1. **Memory (retrieval)** - you teach it prompt -> response pairs. When you
   say something, it finds the most similar thing it has learned (TF-IDF
   cosine similarity) and replies with that response. Your feedback
   (good/bad) changes how much it trusts each memory.
2. **Style (generation)** - every message you write trains a small word-level
   Markov chain, so it can generate new text that sounds like you.

Everything is saved to a JSON file so the model keeps learning across sessions.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter, defaultdict

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_WORD_RE = re.compile(r"\S+")

START, END = "<s>", "</s>"


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens used for matching."""
    return _TOKEN_RE.findall(text.lower())


class LearningModel:
    def __init__(self, threshold: float = 0.35, order: int = 2, seed: int | None = None):
        self.threshold = threshold
        self.order = order
        self.memories: list[dict] = []  # {"prompt", "response", "weight"}
        # context tuple (as "a b" string key) -> Counter of next words
        self.chain: dict[str, Counter] = defaultdict(Counter)
        self.last_match: int | None = None
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------ memory
    def learn(self, prompt: str, response: str) -> None:
        """Teach the model to reply to `prompt` with `response`."""
        prompt, response = prompt.strip(), response.strip()
        if not prompt or not response:
            raise ValueError("prompt and response must be non-empty")
        for mem in self.memories:
            if mem["prompt"].lower() == prompt.lower() and mem["response"] == response:
                mem["weight"] += 0.5  # taught again: reinforce
                break
        else:
            self.memories.append({"prompt": prompt, "response": response, "weight": 1.0})
        self.observe(prompt)
        self.observe(response)

    def _idf(self) -> dict[str, float]:
        n = len(self.memories)
        df = Counter()
        for mem in self.memories:
            df.update(set(tokenize(mem["prompt"])))
        return {t: math.log((1 + n) / (1 + c)) + 1 for t, c in df.items()}

    @staticmethod
    def _vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
        tf = Counter(tokens)
        vec = {t: c * idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def rank(self, text: str) -> list[tuple[float, int]]:
        """Return (score, memory_index) pairs, best first."""
        if not self.memories:
            return []
        idf = self._idf()
        query = self._vector(tokenize(text), idf)
        scored = []
        for i, mem in enumerate(self.memories):
            vec = self._vector(tokenize(mem["prompt"]), idf)
            sim = sum(w * vec.get(t, 0.0) for t, w in query.items())
            # Feedback nudges the score up or down without overriding relevance.
            scored.append((sim * (0.5 + 0.5 * math.tanh(mem["weight"])), i))
        scored.sort(reverse=True)
        return scored

    def respond(self, text: str) -> tuple[str | None, float]:
        """Reply to `text`. Returns (response or None if unsure, confidence)."""
        self.observe(text)
        ranked = self.rank(text)
        if not ranked or ranked[0][0] < self.threshold:
            self.last_match = None
            return None, ranked[0][0] if ranked else 0.0
        score, idx = ranked[0]
        self.last_match = idx
        return self.memories[idx]["response"], score

    def feedback(self, good: bool) -> bool:
        """Reward or penalise the last response. Returns False if nothing to rate."""
        if self.last_match is None:
            return False
        self.memories[self.last_match]["weight"] += 0.5 if good else -1.0
        return True

    def correct(self, prompt: str, better_response: str) -> None:
        """Penalise the last answer and learn a better one."""
        self.feedback(good=False)
        self.learn(prompt, better_response)

    def forget(self, text: str) -> int:
        """Forget memories whose prompt contains `text`. Returns count removed."""
        needle = text.lower()
        before = len(self.memories)
        self.memories = [m for m in self.memories if needle not in m["prompt"].lower()]
        self.last_match = None
        return before - len(self.memories)

    # ------------------------------------------------------------------- style
    def observe(self, text: str) -> None:
        """Train the style model on any text you write."""
        words = _WORD_RE.findall(text)
        if not words:
            return
        seq = [START] * self.order + words + [END]
        for i in range(self.order, len(seq)):
            for k in range(1, self.order + 1):  # store every order for backoff
                ctx = " ".join(seq[i - k:i])
                self.chain[ctx][seq[i]] += 1

    def generate(self, seed: str = "", max_words: int = 30) -> str:
        """Generate text in your style, optionally starting from `seed`."""
        if not self.chain:
            return ""
        out = _WORD_RE.findall(seed)
        history = [START] * self.order + out
        for _ in range(max_words):
            nxt = None
            for k in range(self.order, 0, -1):  # back off to shorter contexts
                options = self.chain.get(" ".join(history[-k:]))
                if options:
                    words, counts = zip(*options.items())
                    nxt = self._rng.choices(words, weights=counts)[0]
                    break
            if nxt is None or nxt == END:
                break
            out.append(nxt)
            history.append(nxt)
        return " ".join(out)

    # ------------------------------------------------------------- persistence
    def stats(self) -> dict:
        return {
            "memories": len(self.memories),
            "vocabulary": len({w for c in self.chain.values() for w in c} - {END}),
            "contexts": len(self.chain),
        }

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "order": self.order,
            "memories": self.memories,
            "chain": {k: dict(v) for k, v in self.chain.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LearningModel":
        model = cls(threshold=data.get("threshold", 0.35), order=data.get("order", 2))
        model.memories = data.get("memories", [])
        for k, v in data.get("chain", {}).items():
            model.chain[k] = Counter(v)
        return model

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)
        os.replace(tmp, path)  # atomic: never leaves a half-written brain

    @classmethod
    def load(cls, path: str) -> "LearningModel":
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
