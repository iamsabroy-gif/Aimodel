"""Learn when to answer and when to say "I don't know".

`answer_from_knowledge` used to decline by hand-set rules (is the key word covered? is the confidence
high enough?). Those rules trade off two mistakes: declining a question the evidence does answer, and
answering one it does not. Here a small logistic model learns that trade-off from labelled questions
(dev sets): each question is answered even when the rules would decline, the answer is marked right or
wrong, and the model learns which decision features go with a right answer.

    python -m aimodel.calibrate                  # leave-one-set-out check on the dev sets
    python -m aimodel.calibrate --save           # fit on all of them and write data/accept_weights.json

Result when this was tried (12 dev sets, 527 labelled questions, trained on 11 and tested on the 12th):
for the evidence route alone it gave 32% fewer wrong answers (60 -> 41) for 5% fewer right ones (342 -> 324).
In the whole pipeline the effect nearly vanished (27 -> 26 wrong answers, 483 -> 478 right ones), because
other routes answer many questions too. So it is NOT switched on: the hand-set rules stay. `load_calibrator()`
exists to try it again with more data or better features.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "accept_weights.json")
ROOT = os.path.join(os.path.dirname(HERE), "sample_data")
SETS = ["devset", "devset2", "devset3", "devset4", "devset5", "devset6", "devset7", "devset8",
        "human_body", "reasoning", "reasoning2", "reasoning3"]
FEATURES = ["confidence", "coverage", "top", "content_n", "content_hit", "key_in", "together", "one_source",
            "n_chosen", "margin", "order", "fit", "asked", "q_len", "n_cands"]


class Calibrator:
    def __init__(self, weights, bias, mean, scale, threshold, features=FEATURES):
        self.w, self.b = np.array(weights, float), float(bias)
        self.mean, self.scale = np.array(mean, float), np.array(scale, float)
        self.threshold, self.features = float(threshold), list(features)

    def probability(self, decision: dict) -> float:
        x = (np.array([decision.get(f, 0.0) for f in self.features]) - self.mean) / self.scale
        return float(1 / (1 + np.exp(-(x @ self.w + self.b))))

    def accepts(self, decision: dict | None):
        return None if decision is None else self.probability(decision) >= self.threshold

    def save(self, path: str = DATA) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"features": self.features, "weights": self.w.tolist(), "bias": self.b,
                       "mean": self.mean.tolist(), "scale": self.scale.tolist(), "threshold": self.threshold},
                      f, indent=1)

    @classmethod
    def load(cls, path: str = DATA) -> "Calibrator | None":
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(d["weights"], d["bias"], d["mean"], d["scale"], d["threshold"], d["features"])


def collect(sets=SETS, root: str = ROOT) -> list[dict]:
    """One row per question the evidence route can say something about."""
    from .evaluation import judge
    from .model import LearningModel
    rows = []
    for name in sets:
        folder = os.path.join(root, name)
        model = LearningModel(seed=0)
        model.load_builtin_vectors()
        model._calibrator = None
        for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
            with open(path, encoding="utf-8") as f:
                model.add_document(f.read(), os.path.basename(path))
        with open(os.path.join(folder, "questions.json"), encoding="utf-8") as f:
            items = json.load(f)
        for skill, question, gold, options in items:
            if skill == "yesno" or "verdict" in options:
                continue
            model.last_decision, model._force_evidence = None, False
            plain = model.answer_from_knowledge(model.understand_question(question))
            model._force_evidence = True
            try:
                forced = model.answer_from_knowledge(model.understand_question(question))
            finally:
                model._force_evidence = False
            if forced is None or model.last_decision is None:
                continue
            right = skill != "abstain" and judge(forced[0], skill, gold, options)
            rows.append({"set": name, "q": question, "x": dict(model.last_decision),
                         "label": 1.0 if right else 0.0, "rules_accept": plain is not None})
    return rows


def _matrix(rows):
    return np.array([[r["x"].get(f, 0.0) for f in FEATURES] for r in rows], float)


def fit(rows, l2: float = 10.0, steps: int = 400, lr: float = 0.3, threshold: float | None = None,
        cost: float | None = None) -> Calibrator:
    X, y = _matrix(rows), np.array([r["label"] for r in rows])
    mean, scale = X.mean(0), X.std(0) + 1e-6
    Z = (X - mean) / scale
    # weight the two outcomes equally, so a few right answers are not drowned by many declines
    sw = np.where(y > 0, 0.5 / max(1.0, y.sum()), 0.5 / max(1.0, (1 - y).sum())) * len(y)
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = (p - y) * sw
        w -= lr * (Z.T @ g / len(y) + l2 * w / len(y))
        b -= lr * g.mean()
    if threshold is None:
        threshold = best_threshold(1 / (1 + np.exp(-(Z @ w + b))), y, cost)
    return Calibrator(w, b, mean, scale, threshold)


WRONG_ANSWER_COST = 2.0  # a wrong answer is worse than "I don't know": it counts this many missed answers


def best_threshold(p, y, cost: float | None = None) -> float:
    """The probability above which to answer: most right answers and right declines, where giving a
    wrong answer costs `cost` times as much as declining a question that had an answer."""
    cost = WRONG_ANSWER_COST if cost is None else cost
    best, at = -1e9, 0.5
    for t in np.linspace(0.2, 0.8, 61):
        s = float(((p >= t) & (y > 0)).sum() + ((p < t) & (y == 0)).sum() - (cost - 1) * ((p >= t) & (y == 0)).sum())
        if s > best:
            best, at = s, float(t)
    return at


def score(accept, rows) -> tuple[int, int, int, int]:
    """(right answers given, wrong answers given, right declines, answers wrongly declined)."""
    a = np.array(accept, bool)
    y = np.array([r["label"] for r in rows]) > 0
    return int((a & y).sum()), int((a & ~y).sum()), int((~a & ~y).sum()), int((~a & y).sum())


def cross_validate(rows, **kw) -> dict:
    out, total_rules, total_model = {}, np.zeros(4, int), np.zeros(4, int)
    for held in sorted({r["set"] for r in rows}):
        train, test = [r for r in rows if r["set"] != held], [r for r in rows if r["set"] == held]
        cal = fit(train, **kw)
        mine = score([cal.probability(r["x"]) >= cal.threshold for r in test], test)
        rules = score([r["rules_accept"] for r in test], test)
        out[held] = {"rules": rules, "learned": mine, "n": len(test)}
        total_rules += rules
        total_model += mine
    out["total"] = {"rules": tuple(map(int, total_rules)), "learned": tuple(map(int, total_model)), "n": len(rows)}
    return out


def main(argv=None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    rows = collect()
    print(f"{len(rows)} labelled questions ({int(sum(r['label'] for r in rows))} answerable correctly)")
    result = cross_validate(rows)
    print("set          rules(right,wrong,declined-right,declined-wrong)   learned(...)")
    for name, v in result.items():
        print(f"{name:11s} {v['rules']}   {v['learned']}")
    if "--save" in argv:
        fit(rows).save()
        print("wrote", DATA)


if __name__ == "__main__":
    main()
