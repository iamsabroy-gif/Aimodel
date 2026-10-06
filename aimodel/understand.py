"""Understanding a question before answering it: typos, synonyms and "which X are Y?" questions."""

from __future__ import annotations

import re

from . import reasoning as rsn
from .reasoning import stem
from .text import STOPWORDS, tokenize

# Everyday words for the things facts are usually about, mapped to the word the facts use.
SYNONYMS = {
    "dwell": "live", "dwells": "live", "reside": "live", "resides": "live", "inhabit": "live",
    "inhabits": "live", "habitat": "live",
    "consume": "eat", "consumes": "eat", "devour": "eat", "devours": "eat", "feed": "eat",
    "feeds": "eat", "diet": "eat",
    "construct": "build", "constructs": "build", "manufacture": "make", "manufactures": "make",
    "produce": "make", "produces": "make",
    "purpose": "use", "utilize": "use", "utilizes": "use",
}
_WORDS = re.compile(r"[A-Za-z']+")
_GENERIC = {"thing", "things", "one", "ones", "kind", "kinds", "type", "types", "item", "items"}


def _edits_one(a: str, b: str) -> bool:
    """Are a and b one insertion, deletion, substitution or swap of neighbours apart?"""
    if a == b:
        return False
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False
    if la == lb:
        diff = [i for i in range(la) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                  and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]])
    short, long_ = (a, b) if la < lb else (b, a)
    i = 0
    while i < len(short) and short[i] == long_[i]:
        i += 1
    return short[i:] == long_[i + 1:]


def _subsequence(small: str, big: str) -> bool:
    it = iter(big)
    return all(c in it for c in small)


def fix_typos(text: str, vocab: set[str]) -> str:
    """Replace words I've never seen with the one known word that is a single slip away.

    Only long-enough words are touched, only when exactly one known word fits,
    and the first letter must agree, so a new name is not turned into another word.
    """
    by_len: dict[int, list[str]] = {}
    for w in vocab:
        by_len.setdefault(len(w), []).append(w)

    def fix(match: re.Match) -> str:
        word = match.group(0)
        low = word.lower()
        first = match.start() == 0
        if len(low) < (3 if first else 5) or low in vocab or stem(low) in vocab or "'" in low or word[0].isupper() and match.start() > 0:
            return word
        # "speed" is a word, not a slip of "seed": dropping a letter needs a longer word to be sure.
        sizes = (len(low), len(low) + 1) + ((len(low) - 1,) if len(low) >= 6 or first else ())
        close = {w for n in sizes for w in by_len.get(n, ())
                 if w[0] == low[0] and _edits_one(low, w)}
        if len(close) > 1 and first:  # a slip in the first word: a question word
            close &= _QUESTION_VOCAB
            if len(close) > 1:  # a dropped letter ("whre" -> "where") beats a changed one ("were")
                close = {w for w in close if _subsequence(low, w)} or close
        if len(close) != 1:
            return word
        fixed = close.pop()
        return fixed.capitalize() if word[0].isupper() else fixed

    return _WORDS.sub(fix, text)


def use_common_words(text: str) -> str:
    """"Where does the lorpan dwell?" -> "Where does the lorpan live?"."""
    return _WORDS.sub(lambda m: SYNONYMS.get(m.group(0).lower(), m.group(0)), text)


def understand(text: str, vocab: set[str]) -> str:
    return use_common_words(fix_typos(text, vocab | STOPWORDS | _QUESTION_VOCAB))


_QUESTION_VOCAB = set("what who whom whose which when where why how does did are were was is "
                      "tell about explain describe name list".split())


def members_question(text: str):
    """Parse "Which animals are birds?" into (category, class) = ("animal", "bird"), else None."""
    words = tokenize(text)
    if len(words) >= 2 and words[0] in ("name", "list", "give"):  # "Name all the mammals."
        rest = [w for w in words[1:] if w not in ("all", "the", "every", "me", "of", "some", "kinds", "types")]
        return ("", rsn.head(" ".join(rest))) if rest and not any(w in rsn.BE for w in rest) else None
    if len(words) < 3 or words[0] not in ("which", "what"):
        return None
    be = next((i for i, w in enumerate(words) if w in rsn.BE), None)
    if be is None or be + 1 >= len(words) or be < 2:  # a category noun must come first
        return None
    middle = [w for w in words[1:be] if w not in ("all", "the", "of", "kinds", "types", "are")]
    cls = rsn.head(" ".join(words[be + 1:]))
    if not cls or (middle and middle[0] in ("is", "are")):
        return None
    if words[0] == "what" and not middle:
        return None  # "What is a bird?" asks for a definition
    return (stem(middle[-1]) if middle else ""), cls


def members(facts: list[dict], category: str, cls: str) -> list[dict]:
    """The is-a facts that put something in `cls` (and, if given, in `category` too)."""
    found, seen = [], set()
    for f in facts:
        if f["rel"] not in ("be", "be called") or f["neg"]:
            continue
        who = rsn.head(f["subj"])
        if not who or who == cls or rsn.head(f["obj"]) != cls or who in seen:
            continue
        seen.add(who)
        found.append(f)
    if category and category not in _GENERIC and stem(category) != cls:
        kept = [f for f in found if rsn.isa_chain(facts, rsn.head(f["subj"]), category)]
        found = kept or found
    return found
