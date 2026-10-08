"""Short story problems: "Tom has 5 apples and buys 3 more. How many apples does he have now?"

The numbers and what happens to them are read from the story, then worked out exactly by the calculator
(see mathsolver.py). It only answers when every number in the story has a clear role and the question asks
for a total, what is left, each share or a difference; anything it is unsure of is left to the other ways of
answering, so a wrong guess is never presented as a calculation.
"""

from __future__ import annotations

import re
from fractions import Fraction

from .mathsolver import _format, words_to_numbers

_ADD = r"buys?|bought|gets?|got|finds?|found|receives?|received|gains?|gained|picks? up|picked up|adds?|added|earns?|earned|collects?|collected|wins?|won|is given|are given|was given|were given|more arrive|joins?|joined|arrives?|arrived|makes?|made|grows?|grew|plants?|planted"
_SUB = r"gives? away|gave away|gives?|gave|loses?|lost|eats?|ate|sells?|sold|spends?|spent|breaks?|broke|uses?|used|takes? away|took away|throws? away|threw away|drops?|dropped|leaves?|left|donates?|donated|removes?|removed|pays?|paid|drinks?|drank|wastes?|wasted|fly away|flew away|runs? away|ran away|goes? away|went away|walks? away|walked away|swims? away|swam away|disappears?|disappeared|leaves|moves? away|moved away"
_TIMES = r"each|every|per|apiece"
_ASK_TOTAL = re.compile(r"\bhow (?:many|much)\b.*\b(?:now|in total|altogether|in all|total|together|are there|do they have|does \w+ have|remain|are left|is left|left|remaining)\b", re.I)
_ASK_EACH = re.compile(r"\bhow (?:many|much)\b.*\b(?:each|per|every|apiece)\b", re.I)
_ASK_DIFF = re.compile(r"\bhow (?:many|much)\b.*\b(?:more|fewer|less)\b.*\bthan\b", re.I)
_SHARE = re.compile(r"\b(?:shared?|split|divided?|distributed?|shares?)\b.*\b(?:equally|evenly|among|between|into)\b", re.I)
_NUM = r"\d[\d,]*(?:\.\d+)?"


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", words_to_numbers(text.strip().lower())) if s.strip()]


def _numbers(sentence: str) -> list[Fraction]:
    return [Fraction(n.replace(",", "")) for n in re.findall(_NUM, sentence)]


def solve_story(text: str):
    """{"answer", "expression", "steps"} for a simple story problem, or None."""
    parts = _sentences(text)
    if len(parts) < 2 or not parts[-1].rstrip().endswith("?") and not re.search(r"\bhow (?:many|much)\b", parts[-1]):
        return None
    ask, story = parts[-1], parts[:-1]
    if not re.search(r"\bhow (?:many|much)\b", ask):
        return None
    facts = [(s, _numbers(s)) for s in story]
    if not any(nums for _, nums in facts):
        return None
    if any(len(nums) > 2 for _, nums in facts):
        return None  # too tangled to read safely

    steps: list[str] = []

    def record(expr: str, value: Fraction) -> Fraction:
        steps.append(f"{expr} = {_format(value)}")
        return value

    # a difference: "Ann has 12 books. Bob has 8 books. How many more books does Ann have than Bob?"
    if _ASK_DIFF.search(ask):
        flat = [n for _, nums in facts for n in nums]
        if len(flat) != 2:
            return None
        big, small = max(flat), min(flat)
        v = record(f"{_format(big)} - {_format(small)}", big - small)
        return {"answer": _format(v), "expression": f"{_format(big)} - {_format(small)}", "steps": steps, "story": True}

    # sharing: "24 sweets are shared equally among 6 children. How many does each child get?"
    if _ASK_EACH.search(ask) or any(_SHARE.search(s) for s, _ in facts):
        flat = [n for _, nums in facts for n in nums]
        if len(flat) == 2 and (_SHARE.search(" ".join(story)) or re.search(r"\bsplit|shared|divided\b", " ".join(story))):
            total, groups = flat
            if groups == 0:
                return None
            v = record(f"{_format(total)} / {_format(groups)}", total / groups)
            return {"answer": _format(v), "expression": f"{_format(total)} / {_format(groups)}", "steps": steps, "story": True}

    # groups of the same size: "There are 4 boxes with 6 pencils each. How many pencils in total?"
    flat_story = " ".join(story)
    if re.search(r"\b(?:" + _TIMES + r")\b", flat_story) and len([n for _, nums in facts for n in nums]) == 2 \
            and _ASK_TOTAL.search(ask):
        a, b = [n for _, nums in facts for n in nums]
        v = record(f"{_format(a)} * {_format(b)}", a * b)
        return {"answer": _format(v), "expression": f"{_format(a)} * {_format(b)}", "steps": steps, "story": True}

    # a running total: start with the first number, then add or take away for each later one. Each clause
    # ("sold 20", "received 30 more") is read on its own so that its own action word decides.
    if not re.search(r"\bhow (?:many|much)\b", ask):
        return None
    clauses = [c.strip() for s in story for c in re.split(r"\band then\b|\bthen\b|\bbut\b|,|\band\b", s) if c.strip()]
    total: Fraction | None = None
    for clause in clauses:
        nums = _numbers(clause)
        if not nums:
            continue
        if len(nums) != 1:
            return None
        n = nums[0]
        if total is None:
            total = n
            continue
        adds = re.search(r"\b(?:" + _ADD + r")\b", clause)
        subs = re.search(r"\b(?:" + _SUB + r")\b", clause)
        if bool(adds) == bool(subs):
            return None  # no clear action, or both: do not guess
        total = record(f"{_format(total)} {'+' if adds else '-'} {_format(n)}", total + n if adds else total - n)
    if total is None or not steps or total < 0:
        return None
    return {"answer": _format(total), "expression": steps[-1].split(" = ")[0], "steps": steps, "story": True}
