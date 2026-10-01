"""Study mode: learning the way a child does.

A child doesn't just store everything. This module adds the missing steps:

* **Memory strength** - everything learned has a strength that rises when it is
  used, confirmed or reviewed and fades with time (a forgetting curve with
  spaced repetition: each successful recall doubles the wait before the next).
* **Importance** - when reading a long web page, only the sentences that
  matter (they recur, define things, match your interests or your question)
  are studied. The rest go on a *shelf*, like a book you can look things up in.
* **Dictionary** - unknown words are looked up and learned.
* **Exams** (`quiz`) - it writes questions from what it knows, answers them from
  memory and grades itself, which strengthens what it recalled and exposes
  what it forgot. It can also quiz you.
* **Consolidation** (`sleep`) - replays, forgets what was never used (to the
  shelf, not deleted), merges duplicates, settles contradictions and retrains.
* **Curiosity** - questions it couldn't answer and exam misses become study
  goals it goes and reads about.
"""

from __future__ import annotations

import math
import re
import time
import urllib.error
from collections import Counter

from . import reasoning as rsn
from . import web
from .reasoning import stem
from .text import keywords, tokenize

DECAY = 0.93           # strength left after one interval without use
FADE_BELOW = 0.2       # weaker than this: moves to the shelf
MAX_STRENGTH = 3.0
MAX_INTERVAL = 64      # days
KEEP_ALL_BELOW = 60    # short texts are studied in full
KEEP_FRACTION = 0.5    # of a long web page
MIN_KEPT = 40
DAY = 86400.0
MAX_GAPS = 100
MAX_TRIES = 3

# Words too general to build a link or a question around ("values", "things"...).
GENERIC = {"value", "thing", "way", "number", "part", "type", "kind", "example", "item", "time",
           "use", "lot", "set", "case", "point", "form", "end", "result", "other", "same",
           "system", "process", "method", "area", "level", "group", "field"}
# Everyday verbs and question words that say little about *what* a question is about.
WEAK_WORDS = {"live", "work", "make", "use", "do", "go", "get", "need", "want", "call", "mean",
              "come", "take", "give", "find", "know", "happen", "start", "begin", "name", "tell",
              "work", "long", "many", "much", "old", "big", "small", "good", "bad"}
# Verbs that read naturally as "What do I <verb>?".
TRANSITIVE = {"like", "love", "hate", "prefer", "want", "need", "own", "enjoy", "study", "play",
              "eat", "drive", "speak", "teach", "learn", "know", "build", "write", "use", "have",
              "make", "visit"}
_START = {"you said": 1.0, "file": 0.8, "dictionary": 0.7, "web": 0.5}
_NEGATED = re.compile(r"\b(not|no|never|none|cannot|neither|nor)\b|n't", re.I)
_DEFINES = re.compile(r"^[\w\s'’-]{1,50}?\b(is|are|was|were|refers to|means)\b", re.I)
_SINGLE_TOPICS = {"name", "age", "job", "address", "city", "birthday"}
_SINGLE_RELS = {"be called", "live", "work"}
_COMPUTE = object()
NET_ERRORS = (urllib.error.URLError, TimeoutError, OSError, ValueError)


def kind_of(source: str) -> str:
    """Where knowledge came from: 'you said', 'file', 'dictionary' or 'web'."""
    if source == "you said":
        return "you said"
    if source.startswith("dictionary"):
        return "dictionary"
    return "web" if source.startswith("http") else "file"


def bare(phrase: str) -> str:
    """'the Rex' -> 'rex': a phrase without its determiners, for exact matching."""
    return " ".join(w for w in tokenize(phrase) if w not in {"a", "an", "the"})


def _article(word: str) -> str:
    return "an" if word[:1] in "aeiou" else "a"


class StudyMixin:
    """Study behaviour for LearningModel (which supplies knowledge, facts, ...)."""

    def _init_study(self) -> None:
        self.day = 0                    # study days (advanced by sleep)
        self.last_sleep = None          # wall-clock time of the last sleep
        self.clock = time.time          # swapped out in tests
        self.shelf: list[dict] = []     # sentences read but not studied
        self.gaps: list[dict] = []      # things it wanted to answer but couldn't
        self.interests: Counter = Counter()
        self.exam_log: list[dict] = []
        self.no_definition: set[str] = set()
        self.last_read = {"kept": 0, "shelved": 0}
        self._shelf_cache = None
        self._updated: list[dict] = []  # facts replaced by the last thing you told me

    # ------------------------------------------------------------ memory strength
    def _init_item(self, entry: dict) -> dict:
        start = 1.0 if "prompt" in entry else _START[kind_of(entry.get("source", ""))]
        entry.setdefault("s", start)
        entry.setdefault("iv", 1)
        entry.setdefault("due", self.day)
        entry.setdefault("n", 0)
        entry.setdefault("ok", 0)
        entry.setdefault("bad", 0)
        return entry

    def _protected(self, entry: dict) -> bool:
        """What you taught or gave me is never forgotten (only web text fades)."""
        return "prompt" in entry or kind_of(entry.get("source", "")) in ("you said", "file")

    def reinforce(self, entry: dict, amount: float = 0.15, recalled: bool = False) -> None:
        """Strengthen a memory; a successful recall also doubles its review interval."""
        self._init_item(entry)
        entry["s"] = min(MAX_STRENGTH, entry["s"] + amount)
        entry["n"] += 1
        if recalled:
            entry["ok"] += 1
            entry["iv"] = min(MAX_INTERVAL, entry["iv"] * 2)
            entry["due"] = self.day + entry["iv"]

    def weaken(self, entry: dict, amount: float = 0.3) -> None:
        self._init_item(entry)
        entry["s"] = max(0.0, entry["s"] - amount)
        entry["bad"] += 1
        entry["iv"] = 1
        entry["due"] = self.day

    def _trace_entries(self, trace: list[dict]) -> list[dict]:
        out = []
        for step in trace:
            entry = self._known.get(step.get("sentence") or step.get("text", ""))
            if entry is not None and not entry.get("shelved") and entry not in out:
                out.append(entry)
        return out

    def _mark_used(self, trace: list[dict]) -> None:
        for entry in self._trace_entries(trace):
            self.reinforce(entry, 0.1)

    def _grade_trace(self, good: bool) -> None:
        """/good and /bad: your feedback strengthens or weakens what the answer used."""
        for entry in self._trace_entries(self.last_trace):
            if good:
                self.reinforce(entry, 0.3, recalled=True)
            else:
                self.weaken(entry, 0.4)

    def _note_interest(self, text: str) -> None:
        for w in keywords(text):
            self.interests[stem(w)] += 1

    def _invalidate(self) -> None:
        self._tfidf_cache = self._vector_cache = self._pretrained_cache = None
        self._names_cache = (-1, set())
        self._shelf_cache = None

    def _remove(self, entries: list[dict], shelve: bool) -> None:
        """Take sentences (and the facts built from them) out of what I know."""
        if not entries:
            return
        ids = {id(e) for e in entries}
        texts = {e["text"] for e in entries}
        self.knowledge = [k for k in self.knowledge if id(k) not in ids]
        self.facts = [f for f in self.facts if f["text"] not in texts]
        for e in entries:
            if shelve:
                e["shelved"] = True
                self.shelf.append(e)
            elif self._known.get(e["text"]) is e:
                del self._known[e["text"]]
        self._invalidate()

    def _remember(self, entry: dict, fact=_COMPUTE) -> None:
        """Study a sentence: add it to knowledge and extract its fact."""
        for key in ("s", "iv", "due", "n", "ok", "bad", "shelved", "again"):
            entry.pop(key, None)
        self._init_item(entry)
        self._known[entry["text"]] = entry
        self.knowledge.append(entry)
        if fact is _COMPUTE:
            fact = rsn.extract_fact(entry["text"], topic=entry.get("hint"))
        if fact:
            self.facts.append({**fact, "source": entry["source"], "text": entry["text"]})

    # ------------------------------------------------------------------ importance
    def _importance(self, sentences: list[str], focus: str | None = None) -> list[float]:
        """Score how worth studying each sentence of a document is."""
        stems = [{stem(w) for w in keywords(s)} for s in sentences]
        df = Counter(w for st in stems for w in st)
        focus_stems = {stem(w) for w in keywords(focus)} if focus else set()
        interests = set(self.interests)
        scores = []
        for i, (s, st) in enumerate(zip(sentences, stems)):
            if not st:
                scores.append(0.0)
                continue
            recurring = sum(math.log1p(df[w] - 1) for w in st) / math.sqrt(len(st))
            interest = len(st & interests) / len(st)
            about = len(st & focus_stems) / len(focus_stems) if focus_stems else 0.0
            scores.append(recurring + 4 * interest + 10 * about
                          + 6.0 * (i < 3) + 4.0 * bool(_DEFINES.search(s)))
        return scores

    def _choose_keep(self, entries: list[dict], source: str, focus: str | None) -> set[int]:
        """Indexes of the sentences worth studying. Your own data is always kept."""
        n = len(entries)
        if kind_of(source) != "web" or n <= KEEP_ALL_BELOW:
            return set(range(n))
        scores = self._importance([e["text"] for e in entries], focus)
        k = max(MIN_KEPT, int(n * KEEP_FRACTION))
        return set(sorted(range(n), key=lambda i: -scores[i])[:k])

    # ----------------------------------------------------------------------- shelf
    def _shelf_index(self):
        if self._shelf_cache is None or self._shelf_cache[0] != len(self.shelf):
            stems = [{stem(t) for t in tokenize(e["text"])} for e in self.shelf]
            self._shelf_cache = (len(self.shelf), stems, Counter(w for st in stems for w in st))
        return self._shelf_cache[1], self._shelf_cache[2]

    def recall(self, text: str, limit: int = 4) -> int:
        """Look a question up in the books on the shelf and study what matches.

        Returns the number of sentences moved from the shelf into knowledge.
        """
        if not self.shelf:
            return 0
        wanted = self._wanted(text)
        if not wanted:
            return 0
        stems, df = self._shelf_index()
        n = len(self.shelf)

        def rarity(word: str) -> float:
            return max((math.log((1 + n) / (1 + df.get(v, 0))) for v in wanted[word]), default=0.0)

        key = max([w for w in wanted if w not in WEAK_WORDS] or wanted, key=rarity)
        cands = []
        for i, st in enumerate(stems):
            covered = {w for w, alts in wanted.items() if alts & st}
            if key in covered and len(covered) / len(wanted) >= 0.5:
                cands.append((len(covered), sum(rarity(w) for w in covered), -i, i))
        if not cands:
            return 0
        chosen = {c[3] for c in sorted(cands, reverse=True)[:limit]}
        for i in list(chosen):  # the next steps of a procedure come along
            base = self.shelf[i]
            for j, e in enumerate(self.shelf):
                if (e["source"] == base["source"] and 0 < e.get("pos", 0) - base.get("pos", 0) <= 4
                        and rsn.SEQUENCE_MARKER.search(e["text"])):
                    chosen.add(j)
        picked = [self.shelf[i] for i in sorted(chosen)]
        self.shelf = [e for i, e in enumerate(self.shelf) if i not in chosen]
        for e in picked:
            self._remember(e)
        self._invalidate()
        self.neural.train([tokenize(e["text"]) for e in picked], epochs=3, min_pairs=1000)
        return len(picked)

    # ------------------------------------------------------------------ dictionary
    @staticmethod
    def _definition_sentence(word: str, definition: str) -> str:
        text = definition.strip().rstrip(".")
        text = text[:1].lower() + text[1:]
        verb = "is" if text.split(" ", 1)[0] in ("a", "an", "the") else "means"
        return f"{word.capitalize()} {verb} {text}."

    def define(self, word: str) -> int:
        """Look a word up in the dictionary and learn its meaning."""
        word = word.strip().lower()
        if not word or word in self.no_definition:
            return 0
        defs = web.dictionary(word, get=self.fetch)
        if not defs:
            self.no_definition.add(word)
            return 0
        text = "\n".join(self._definition_sentence(word, d) for _pos, d in defs)
        return self.add_document(text, f"dictionary:{word}", topic=word.capitalize())

    def define_unknown(self, text: str, limit: int = 2) -> int:
        """Look up the words in `text` that I have never seen before."""
        words = [w for w in keywords(text)
                 if w.isalpha() and len(w) >= 4 and w not in self.neural.vocab
                 and stem(w) not in self.neural.vocab and w not in self.no_definition]
        learned = 0
        for word in words[:limit]:
            if self.notify:
                self.notify(f"Looking up '{word}' in the dictionary...")
            learned += self.define(word)
        return learned

    # ------------------------------------------------------------------------ gaps
    def _add_gap(self, text: str, why: str, check: dict | None = None) -> None:
        topic = " ".join(keywords(text))
        if not topic:
            return
        for gap in self.gaps:
            if gap["topic"] == topic:
                gap["count"] += 1
                return
        self.gaps.append({"topic": topic, "q": text, "why": why, "count": 1, "tries": 0,
                          "check": check})
        if len(self.gaps) > MAX_GAPS:
            self.gaps.sort(key=lambda g: -g["count"])
            del self.gaps[MAX_GAPS:]

    def _close_gap(self, text: str) -> None:
        topic = " ".join(keywords(text))
        self.gaps = [g for g in self.gaps if g["topic"] != topic]

    def study_goals(self, limit: int = 5) -> list[dict]:
        """What I most want to learn about: the questions I miss most often."""
        return sorted(self.gaps, key=lambda g: (-g["count"], g["tries"]))[:limit]

    def _gap_resolved(self, gap: dict) -> bool:
        answer = self.explain(gap["q"])
        if answer is None:
            return False
        return self.grade(gap["check"], answer[0]) if gap.get("check") else True

    def be_curious(self, use_web: bool = False, limit: int = 3) -> list[dict]:
        """Go and study my biggest gaps: the shelf first, then the dictionary and web."""
        report = []
        for gap in self.study_goals(limit):
            learned, error = self.recall(gap["topic"]), None
            try:
                if use_web and not self._gap_resolved(gap):
                    learned += self.define_unknown(gap["topic"])
                    if not self._gap_resolved(gap):
                        learned += self.search_web(gap["topic"])
                        learned += self.recall(gap["topic"])
            except NET_ERRORS as e:
                error = str(e)
            resolved = self._gap_resolved(gap)
            gap["tries"] += 1
            if resolved or gap["tries"] >= MAX_TRIES:
                self.gaps = [g for g in self.gaps if g is not gap]
            report.append({"topic": gap["topic"], "question": gap["q"], "learned": learned,
                           "resolved": resolved, "error": error})
        return report

    # ------------------------------------------------------------------------ exams
    def _answer_key(self, text: str, exclude=frozenset()) -> str | None:
        """The most specific word of an answer: what a correct reply must contain."""
        words = [w for w in keywords(text)
                 if w.isalpha() and len(w) > 2 and stem(w) not in exclude]
        if not words:
            return None
        idf = self._knowledge_index()[0].idf if self.knowledge else {}
        return max(words, key=lambda w: (idf.get(w, 0.0), len(w)))

    def _accept_words(self, text: str, exclude=frozenset()) -> list[str]:
        """Stems that would show someone knows the answer (used to grade people)."""
        return sorted({stem(w) for w in keywords(text) if w.isalpha() and len(w) > 2
                       and stem(w) not in exclude and w not in WEAK_WORDS
                       and stem(w) not in GENERIC})

    def _fact_question(self, f: dict) -> dict | None:
        subj = f["subj"]
        exclude = {stem(w) for w in tokenize(subj)}
        key = self._answer_key(f["obj"], exclude)
        entry = self._known.get(f["text"])
        if key is None or entry is None:
            return None
        topic = " ".join(keywords(subj)) or subj.lower()
        be = f["rel"] in ("be", "be called")
        if subj.lower() == "i":
            if be or f["rel"] not in TRANSITIVE:
                return None
            q, kind = f"What do I {f['rel']}?", "personal"
        elif rsn.is_personal(f) or f["source"] == "you said":
            q, kind = f"Tell me about {self._lower(subj)}", "personal"
        elif be:
            verb = f["verb"].split(" ", 1)[0].lower()
            q, kind = f"What {verb if verb in rsn.BE else 'is'} {self._lower(subj)}?", "define"
        else:
            q, kind = f"What do you know about {self._lower(subj)} and {f['rel']}?", "recall"
        return {"q": q, "kind": kind, "expect": key, "items": [entry],
                "answer": rsn.state(f), "topic": topic,
                "accept": self._accept_words(f["obj"], exclude)}

    def _inference_questions(self, live: list[dict]) -> list[dict]:
        """'Is a cat an animal?' from 'cats are mammals' + 'mammals are animals'."""
        edges: dict[str, list[tuple[str, dict]]] = {}
        for f in live:
            if f["rel"] in ("be", "be called") and not f["neg"]:
                a, b = rsn.head(f["subj"]), rsn.head(f["obj"])
                if a and b and a != b and a not in GENERIC and b not in GENERIC:
                    edges.setdefault(a, []).append((b, f))
        out, seen = [], set()
        for a, firsts in edges.items():
            direct = {b for b, _ in firsts}
            for b, f1 in firsts:
                for c, f2 in edges.get(b, ()):
                    if c != a and c not in direct and (a, c) not in seen:
                        seen.add((a, c))
                        items = [e for e in (self._known.get(f1["text"]), self._known.get(f2["text"])) if e]
                        out.append({"q": f"Is a {a} {_article(c)} {c}?", "kind": "infer", "expect": None,
                                    "items": items, "topic": a,
                                    "answer": f"Yes. {rsn.state(f1).rstrip('.')}, and "
                                              f"{self._lower(rsn.state(f2).rstrip('.'))}."})
        return out

    def make_questions(self, n: int = 5) -> list[dict]:
        """Write exam questions from what I know, weakest and most overdue first.

        Only questions with a single fair answer are asked, and never one that
        gives the answer away.
        """
        live = [f for f in self.facts if not f["neg"] and f["text"] in self._known]
        group = lambda f: (bare(f["subj"]), "be" if f["rel"] in ("be", "be called") else f["rel"])
        counts = Counter(group(f) for f in live)
        questions = [q for f in live if counts[group(f)] == 1 for q in [self._fact_question(f)] if q]
        for m in self.memories:
            key = self._answer_key(m["response"])
            questions.append({"q": m["prompt"], "kind": "memory", "expect": key, "items": [m],
                              "answer": m["response"], "topic": " ".join(keywords(m["prompt"])),
                              "accept": self._accept_words(m["response"])})
        questions += self._inference_questions(live)

        def priority(q):
            due = any(e.get("due", self.day) <= self.day for e in q["items"])
            strength = min((e.get("s", 0.8) for e in q["items"]), default=1.0)
            return (0 if due else 1, strength, self._rng.random())

        questions.sort(key=priority)
        out, seen = [], set()
        for q in questions:
            if q["q"] not in seen:
                seen.add(q["q"])
                out.append(q)
            if len(out) >= n:
                break
        return out

    def grade(self, question: dict, answer: str, lenient: bool = False) -> bool:
        """Is `answer` right?

        My own replies are sentences, so they must contain the answer's key word
        (or a synonym). With `lenient=True` (grading a person's short answer),
        any meaningful word of the correct answer is enough.
        """
        if question["kind"] == "infer":
            return answer.strip().lower().startswith(("yes", "probably yes"))
        heard = {stem(t) for t in tokenize(answer)}
        if lenient and question.get("accept") and set(question["accept"]) & heard:
            return True
        if question["expect"] is None:
            return tokenize(answer) == tokenize(question["answer"])
        return bool(self._expand(question["expect"]) & heard)

    def record(self, question: dict, correct: bool) -> None:
        """Learn from an exam answer: recalled things get stronger, misses get reviewed."""
        for entry in question["items"]:
            if correct:
                self.reinforce(entry, 0.3, recalled=True)
            else:
                self.weaken(entry, 0.3)
        if not correct:
            check = {k: question[k] for k in ("kind", "expect", "answer")}
            self._add_gap(question["topic"] or question["q"], "forgot", check)
            for gap in self.gaps:
                if gap["topic"] == " ".join(keywords(question["topic"] or question["q"])):
                    gap["q"], gap["check"] = question["q"], check

    def log_exam(self, score: int, total: int, who: str) -> None:
        self.exam_log.append({"day": self.day, "score": score, "total": total, "who": who})
        del self.exam_log[:-50]

    def quiz(self, n: int = 5) -> dict:
        """Take an exam: answer my own questions from memory and grade myself."""
        saved = (self.last_match, self.last_query, self.last_reply, self.last_source, self.last_trace)
        results = []
        try:
            for q in self.make_questions(n):
                reply, _ = self.respond(q["q"], learn=False)
                correct = self.grade(q, reply or "")
                results.append({**q, "reply": reply, "correct": correct})
                self.record(q, correct)
        finally:
            (self.last_match, self.last_query, self.last_reply,
             self.last_source, self.last_trace) = saved
        score = sum(r["correct"] for r in results)
        if results:
            self.log_exam(score, len(results), "me")
        return {"results": results, "score": score, "total": len(results)}

    # --------------------------------------------------------------- consolidation
    def _supersede(self, fact: dict) -> None:
        """You moved house: the new 'My sister lives in X' replaces the old one."""
        if fact["neg"] or fact["source"] != "you said":
            return
        single = fact["rel"] in _SINGLE_RELS or rsn.head(fact["subj"]) in _SINGLE_TOPICS
        if not single:
            return
        old = [g for g in self.facts if g is not fact and g["source"] == "you said" and not g["neg"]
               and g["rel"] == fact["rel"] and bare(g["subj"]) == bare(fact["subj"])
               and bare(g["obj"]) != bare(fact["obj"])]
        entries = [self._known[g["text"]] for g in old if g["text"] in self._known]
        self._updated.extend(old)
        self._remove(entries, shelve=False)

    def _merge_duplicates(self) -> int:
        groups: dict[frozenset, list[dict]] = {}
        for e in self.knowledge:
            words = frozenset(stem(w) for w in keywords(e["text"]))
            if len(words) >= 2:  # "X is Y" and "X is not Y" are opposites, not repeats
                groups.setdefault((words, bool(_NEGATED.search(e["text"]))), []).append(e)
        drop = []
        for entries in groups.values():
            if len(entries) > 1:
                best = max(entries, key=lambda e: (self._protected(e), self._init_item(e)["s"]))
                best["s"] = min(MAX_STRENGTH, best["s"] + 0.1 * (len(entries) - 1))
                drop += [e for e in entries if e is not best]
        self._remove(drop, shelve=False)
        seen, dup_facts = set(), []
        for f in self.facts:
            key = (rsn.head(f["subj"]), f["rel"], rsn.head(f["obj"]), f["neg"])
            if key in seen and key[0] and key[2]:
                dup_facts.append(f)
            seen.add(key)
        self.facts = [f for f in self.facts if not any(f is d for d in dup_facts)]
        return len(drop) + len(dup_facts)

    def _settle_contradictions(self) -> list[dict]:
        """'X is Y' and 'X is not Y': keep the one I trust more, or ask you."""
        pos: dict[tuple, dict] = {}
        neg: dict[tuple, dict] = {}
        for f in self.facts:
            if f["rel"] in ("be", "be called"):
                key = (rsn.head(f["subj"]), rsn.head(f["obj"]))
                if key[0] and key[1]:
                    (neg if f["neg"] else pos).setdefault(key, f)
        found, losers = [], []
        for key in pos.keys() & neg.keys():
            p, n = pos[key], neg[key]
            ep, en = self._known.get(p["text"]), self._known.get(n["text"])
            if not ep or not en:
                continue
            rank = lambda e: (self._protected(e), self._init_item(e)["s"])
            if rank(ep) == rank(en):
                found.append({"a": rsn.state(p), "b": rsn.state(n), "kept": None})
            else:
                win, lose = (p, en) if rank(ep) > rank(en) else (n, ep)
                losers.append(lose)
                found.append({"a": rsn.state(p), "b": rsn.state(n), "kept": rsn.state(win)})
        self._remove(losers, shelve=True)
        return found

    def sleep(self, days: int = 1) -> dict:
        """Consolidate: forget what was never used, merge repeats, settle conflicts, replay."""
        days = max(1, int(days))
        self.day += days
        self.last_sleep = self.clock()
        for e in self.knowledge:
            if not self._protected(e):
                self._init_item(e)
                e["s"] *= DECAY ** (days / e["iv"])
        contradictions = self._settle_contradictions()
        faded = [e for e in self.knowledge if not self._protected(e) and e["s"] < FADE_BELOW]
        self._remove(faded, shelve=True)
        merged = self._merge_duplicates()
        loss = self.retrain(epochs=2, limit=2000) if self.knowledge or self.memories else None
        return {"day": self.day, "faded": [e["text"] for e in faded], "merged": merged,
                "contradictions": contradictions, "retrained": loss,
                "strong": sum(1 for e in self.knowledge if e.get("s", 0.8) >= 1.0),
                "kept": len(self.knowledge), "shelf": len(self.shelf)}

    def sleep_if_due(self) -> dict | None:
        """Sleep automatically once a day has passed since the last one."""
        now = self.clock()
        if self.last_sleep is None:
            self.last_sleep = now
            return None
        days = int((now - self.last_sleep) // DAY)
        return self.sleep(days=min(days, 30)) if days >= 1 else None

    # ----------------------------------------------------------------- study session
    def study_session(self, use_web: bool = False, quiz_size: int = 5) -> dict:
        """A day at school: be curious, take an exam, then sleep on it."""
        return {"curiosity": self.be_curious(use_web=use_web),
                "exam": self.quiz(quiz_size),
                "sleep": self.sleep()}

    def progress(self) -> dict:
        items = self.knowledge + self.memories
        return {
            "day": self.day,
            "studied": len(self.knowledge),
            "on the shelf": len(self.shelf),
            "strong (>=1.0)": sum(1 for e in items if e.get("s", 0.8) >= 1.0),
            "fading (<0.4)": sum(1 for e in self.knowledge
                                 if not self._protected(e) and e.get("s", 0.8) < 0.4),
            "due for review": sum(1 for e in items if e.get("due", self.day) <= self.day),
            "open questions": len(self.gaps),
            "exams": self.exam_log[-5:],
        }
