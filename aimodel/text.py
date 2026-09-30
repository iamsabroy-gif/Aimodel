"""Text utilities shared by the model: tokenizing, sentence splitting, TF-IDF."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Za-z0-9])")
_ABBREV = re.compile(r"(\b(e\.g|i\.e|etc|vs|mr|mrs|ms|dr|prof|st|no|fig|approx|inc|ltd|jr|sr)|\b[A-Za-z])\.$", re.I)

STOPWORDS = frozenset("""
a an the is are was were be been being am do does did done of to in on at for by
with from and or but not no nor what who whom whose which when where why how this
that these those it its i you he she they we me my mine your yours our their his
her them us as if then than so such can could would should will shall may might
must there here about into over under up down out just also very tell please know
explain define describe give some any all more most much many s t don't what's
who's it's i'm
""".split())

_QUESTION_WORDS = ("what", "who", "whom", "whose", "which", "when", "where", "why",
                   "how", "is", "are", "was", "were", "do", "does", "did", "can",
                   "could", "should", "tell", "explain", "define", "describe")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens used for matching."""
    return _TOKEN_RE.findall(text.lower())


def keywords(text: str) -> list[str]:
    """Tokens that carry meaning (stopwords removed)."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


def looks_like_question(text: str) -> bool:
    text = text.strip().lower()
    return text.endswith("?") or text.startswith(_QUESTION_WORDS)


_DANGLING = {"the", "a", "an", "of", "in", "to", "and", "or", "by", "with", "for", "from", "as", "at", "on"}
_CODE = re.compile(r">>>|==|\s=\s|\w\(|[{}]|\.\.\.|;\s*$")
_PROSE_CHARS = re.compile(r"[\w\s,.'’%!?()\-\"]")


def is_prose(sentence: str) -> bool:
    """False for code snippets and symbol soup that got mixed into text."""
    if _CODE.search(sentence):
        return False
    words = sentence.split()
    if words and words[-1].lower().rstrip(".!?") in _DANGLING:
        return False  # cut off mid-phrase ("... directly in the.")
    if not sentence.endswith((".", "!", "?", '"', "'")) and len(words) <= 8:
        later = [w for w in words[1:] if w[0].isalpha()]
        if later and sum(w[0].isupper() for w in later) >= 0.6 * len(later):
            return False  # a heading like "Unpacking Argument Lists"
    return len(_PROSE_CHARS.findall(sentence)) >= 0.9 * len(sentence)


def split_sentences(text: str, min_words: int = 3, max_chars: int = 400) -> list[str]:
    """Split raw text (a document, web page...) into clean sentences."""
    out = []
    for line in text.splitlines():
        line = " ".join(line.replace("¶", " ").split())
        if not line or line.startswith("="):  # skip blank lines and wiki headings
            continue
        pieces = []
        for piece in _SENT_RE.split(line):
            if pieces and _ABBREV.search(pieces[-1]):  # "e.g. this" isn't a new sentence
                pieces[-1] += " " + piece
            else:
                pieces.append(piece)
        for sent in pieces:
            sent = sent.strip()
            if len(sent.split()) < min_words or not is_prose(sent):
                continue
            if len(sent) > max_chars:
                sent = sent[:max_chars].rsplit(" ", 1)[0] + " ..."
            out.append(sent)
    return out


class TfidfIndex:
    """A small TF-IDF index with cosine-similarity search."""

    def __init__(self, docs: list[list[str]]):
        self.size = len(docs)
        df = Counter()
        for doc in docs:
            df.update(set(doc))
        self.idf = {t: math.log((1 + self.size) / (1 + c)) + 1 for t, c in df.items()}
        self.postings: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for i, doc in enumerate(docs):
            for t, w in self.vector(doc).items():
                self.postings[t].append((i, w))

    def vector(self, tokens: list[str]) -> dict[str, float]:
        tf = Counter(tokens)  # sublinear: saying a word 3 times isn't 3x as relevant
        vec = {t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items() if v}

    def scores(self, tokens: list[str]) -> list[float]:
        """Cosine similarity of `tokens` against every document."""
        out = [0.0] * self.size
        for t, w in self.vector(tokens).items():
            for i, dw in self.postings.get(t, ()):
                out[i] += w * dw
        return out
