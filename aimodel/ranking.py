"""Learn how to rank evidence sentences from question sets: `python -m aimodel.ranking`.

For every answerable question in a set (see devset.py) the model's candidate sentences are collected with their
features; a sentence is "right" if it holds every word of the answer key. A linear scorer is then fitted so that
right sentences rank above wrong ones (a softmax over each question's candidates). The weights are saved to
aimodel/data/rank_weights.json and used by LearningModel.answer_from_knowledge.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from .model import RANK_WEIGHTS_PATH, LearningModel
from .reasoning import stem
from .text import tokenize

FEATURES = ["base", "tfidf", "meaning", "coverage", "content", "key", "pos0", "definition", "about", "intro",
            "where", "how", "length", "web", "bigram"]


MAX_WEIGHT = 3.0


def collect(folder: str, vectors: bool = True) -> list[dict]:
    """Per answerable question: its candidate sentences as (features, is_right)."""
    model = LearningModel(seed=0)
    if vectors:
        model.load_builtin_vectors()
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            model.add_document(f.read(), os.path.basename(path))
    model.rank_weights = dict(model.rank_weights)  # ranking is measured with whatever weights are in use
    model._candidate_log = []
    rows = []
    with open(os.path.join(folder, "questions.json"), encoding="utf-8") as f:
        items = json.load(f)
    for skill, question, gold, options in items:
        if skill in ("abstain", "yesno") or not gold:
            continue
        wanted = [stem(w) for w in gold]
        banned = [stem(w) for w in options.get("no", [])]

        def right(text: str) -> bool:
            have = {stem(t) for t in tokenize(text)}
            return all(w in have for w in wanted) and not any(w in have for w in banned)

        exists = any(right(k["text"]) for k in model.knowledge)
        model._candidate_log.clear()
        model.answer_from_knowledge(question)
        cands = model._candidate_log[-1]["candidates"] if model._candidate_log else []
        rows.append({"question": question, "exists": exists,
                     "cands": [({n: c["named"][n] for n in FEATURES}, right(model.knowledge[c["i"]]["text"]))
                               for c in cands]})
    return rows


def _matrix(row: dict):
    x = np.array([[f[n] for n in FEATURES] for f, _ in row["cands"]])
    return x, np.array([ok for _, ok in row["cands"]], dtype=float)


def train(rows: list[dict], l2: float = 0.05, epochs: int = 400, lr: float = 0.5) -> dict[str, float]:
    """Weights per feature (on the raw features) from a softmax over each question's candidates."""
    data = [_matrix(r) for r in rows if r["exists"] and r["cands"] and any(ok for _, ok in r["cands"])]
    if not data:
        raise ValueError("no questions with a right sentence to learn from")
    every = np.concatenate([x for x, _ in data])
    mu, sd = every.mean(0), every.std(0) + 1e-6
    w = np.zeros(len(FEATURES))
    for _ in range(epochs):
        grad = np.zeros_like(w)
        for x, y in data:
            z = (x - mu) / sd
            logits = z @ w
            p = np.exp(logits - logits.max())
            p /= p.sum()
            grad += z.T @ (p - y / y.sum())
        w -= lr * (grad / len(data) + l2 * w)
    # a rarely-on feature (like "how") gets a big raw weight from a small standard deviation; cap it
    return {n: round(float(np.clip(v, -MAX_WEIGHT, MAX_WEIGHT)), 4) for n, v in zip(FEATURES, w / sd)}


def metrics(rows: list[dict], weights: dict[str, float] | None = None) -> dict:
    """How often the right sentence is first (and its mean reciprocal rank) among questions that have one."""
    n = hit1 = pool = 0
    mrr = 0.0
    for row in rows:
        if not row["exists"]:
            continue
        n += 1
        pool += any(ok for _, ok in row["cands"])
        score = [sum(weights[k] * f[k] for k in weights if k in f) if weights else 0 for f, _ in row["cands"]]
        order = sorted(range(len(score)), key=lambda i: -score[i])
        for rank, i in enumerate(order, 1):
            if row["cands"][i][1]:
                hit1 += rank == 1
                mrr += 1 / rank
                break
    return {"questions": n, "in_pool": pool, "first": hit1, "mrr": round(mrr / max(n, 1), 3)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("folders", nargs="+", help="question sets to learn from")
    ap.add_argument("--check", nargs="*", default=[], help="question sets to report on (not learned from)")
    ap.add_argument("--l2", type=float, default=0.05)
    ap.add_argument("--out", default=RANK_WEIGHTS_PATH)
    ap.add_argument("--dry-run", action="store_true", help="do not write the weights file")
    args = ap.parse_args(argv)
    rows = [r for folder in args.folders for r in collect(folder)]
    weights = train(rows, args.l2)
    print("weights:", weights)
    print("learned from:", metrics(rows, weights))
    for folder in args.check:
        print(f"{folder}:", metrics(collect(folder), weights))
    if not args.dry_run:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(weights, f, indent=1)
        print("saved", args.out)


if __name__ == "__main__":
    main()
