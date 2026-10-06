"""Score the model on real text it was not built around: `python -m aimodel.devset [folder]`.

The folder holds .txt articles and a questions.json of [skill, question, wanted words, options] rows
(the same shape as the built-in benchmark). Questions with wanted=null should be declined.
"""

from __future__ import annotations

import glob
import json
import os
import sys

from .evaluation import judge
from .model import LearningModel

DEFAULT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data", "devset")


def run(folder: str = DEFAULT) -> dict:
    model = LearningModel(seed=0)
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            model.add_document(f.read(), os.path.basename(path))
    with open(os.path.join(folder, "questions.json"), encoding="utf-8") as f:
        items = json.load(f)
    rows, by_skill = [], {}
    for skill, question, gold, options in items:
        reply, _ = model.respond(question, learn=False)
        ok = judge(reply, "x" if skill == "yesno" else skill, gold, options)
        rows.append({"skill": skill, "question": question, "reply": reply, "ok": ok})
        by_skill.setdefault(skill, []).append(ok)
    answerable = [r for r in rows if r["skill"] != "abstain"]
    declined = [r for r in rows if r["skill"] == "abstain"]
    return {
        "by_skill": {k: (sum(v), len(v)) for k, v in by_skill.items()},
        "answerable": (sum(r["ok"] for r in answerable), len(answerable)),
        "wrong": sum(1 for r in answerable if r["reply"] is not None and not r["ok"]),
        "declined_correctly": (sum(r["ok"] for r in declined), len(declined)),
        "misses": [r for r in rows if not r["ok"]],
    }


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    folder = next((a for a in argv if not a.startswith("-")), DEFAULT)
    result = run(folder)
    for skill, (ok, total) in result["by_skill"].items():
        print(f"{skill:11s} {ok}/{total}")
    a, d = result["answerable"], result["declined_correctly"]
    print(f"answerable correct {a[0]}/{a[1]} ({a[0] / a[1]:.0%}); answered wrongly {result['wrong']}; "
          f"declined correctly {d[0]}/{d[1]}")
    if "--misses" in argv:
        for r in result["misses"]:
            print(f"MISS {r['skill']} | {r['question']} -> {(r['reply'] or 'None')[:160]}")


if __name__ == "__main__":
    main()
