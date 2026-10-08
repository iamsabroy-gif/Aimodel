"""Core learning model.

The model learns from you and from what it reads:

1. **Memory** - prompt -> response pairs you teach it. Your feedback
   (good/bad) changes how much it trusts each one.
2. **Neural network** - a small word-embedding network (see `neural.py`) that
   trains on everything it sees and learns which words mean similar things,
   so matching works on meaning and not only on exact words.
3. **Knowledge** - sentences from files and web pages you give it, plus
   Wikipedia articles it looks up itself when you ask something it doesn't know.
   To answer, it gathers the sentences that best cover your question and
   combines them, keeping track of where each one came from (see `/why`).
4. **Style** - every message you write trains a small Markov chain, so it can
   generate new text that sounds like you.

5. **Reasoning** - sentences become facts ("cats -> are -> mammals") that
   can be chained to prove new things, and answers are written from those
   facts in the model's own words (see `reasoning.py`).
6. **Feedback network** - a second small neural network that learns from
   /good and /bad which evidence makes a good answer (see `ranker.py`).
7. **Study mode** - memory strength and forgetting, importance sorting, a
   dictionary, self-exams, sleep-like consolidation and curiosity
   (see `study.py`).
8. **Writer** - a tiny transformer you train on Kaggle words the answers,
   checked against the evidence (see `writer.py`, `transformer.py`).
9. **Greetings** - hello, how are you, thanks, bye: built in, in your name,
   and trainable in your own words (see `smalltalk.py`).

Everything is saved (a JSON "brain" plus the networks' weights in .npz).
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np

from . import reasoning as rsn
from . import web
from .neural import WordEmbeddings
from .ranker import FeedbackRanker
from .reasoning import stem
from . import mathsolver
from .smalltalk import SmallTalkMixin
from .study import GENERIC as _GENERIC, WEAK_WORDS, StudyMixin, bare as _bare
from .text import STOPWORDS, TfidfIndex, keywords, looks_like_question, split_sentences, tokenize
from .documents import extract
from .understand import (check_claim, common_question, family, in_common, members, members_question,
                         understand)
from .vectors import PretrainedVectors
from .writer import WriterMixin

_WORD_RE = re.compile(r"\S+")

START, END = "<s>", "</s>"


def neural_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".neural.npz"


def vectors_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".vectors.npz"


def writer_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".writer.npz"


# "Here is an example:" - introduces something that isn't there (code, a table).
_INTRO = re.compile(r"\b(example|following|below|follows|like this|shown)\b[^.!?]*:\s*$", re.I)

_CONNECTORS = ("Also, ", "On top of that, ", "In addition, ")
_STEPS = ("First, ", "Then, ", "After that, ", "Finally, ")


_QUANTITY = re.compile(r"^\W*(when|what year|in what year)\b|^\W*how\s+(many|much|old|long|tall|big|far|deep|wide|fast|heavy|high|large|often|hot|cold)\b"
                       r"|\b(percentage|percent)\b", re.I)
_NUMBER = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|hundred|thousand|"
                     r"million|billion|half|double|dozen)\b", re.I)
_UNITS = (r"(?:m|km|cm|mm|ft|mi|kg|g|lb|t|tonnes?|tons?|metres?|meters?|kilomet(?:re|er)s?|miles?|feet|foot|"
          r"inch(?:es)?|kilograms?|pounds?|years?|mya|days?|hours?|minutes?|degrees?|%|percent)")
_MEASURED = re.compile(r"\d[\d,.]*\s*" + _UNITS + r"(?![a-z])|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|"
                       r"hundred|thousand|million|billion)\b[- ]" + _UNITS + r"\b", re.I)
_AGE_UNITS = re.compile(r"(\d[\d,.]*|\b(?:one|two|three|four|five|six|seven|eight|nine|ten|twelve|hundred|thousand|"
                        r"million|billion)\b)[- ]*(years?|centuries|century|decades?|mya|months|days)\b", re.I)
# "How old is X?" wants an age ("4.5 billion years ago", "aged 30"), not any amount of years ("an orbital period of 29 years")
_OLD_AGE = re.compile(r"\b(years?|centuries|decades|mya|billion|million)\b[^.;]{0,15}\b(old|ago)\b|\bage[d]?\b|"
                      r"\b(formed|founded|built|born|established|created|began|started|dates? (back )?(to|from)|dated)\b"
                      r"[^.;]{0,40}\d|\bfor\s+(?:about|around|nearly|over|more than|almost)?\s*\d[\d,.]*\s*(?:years|centuries|millennia)\b", re.I)
_MANNER = re.compile(r"\b(by|through|via|using)\s+\w+ing\b", re.I)  # "by gathering", "through regurgitation"
_WHEN = re.compile(r"^\W*(when|what year|in what year)\b", re.I)
_DATE = re.compile(r"\b\d{3,4}\b|\b(january|february|march|april|may|june|july|august|september|october|november|"
                   r"december|monday|tuesday|wednesday|thursday|friday|saturday|sunday|morning|afternoon|evening|night|noon|"
                   r"today|tomorrow|yesterday|spring|summer|autumn|fall|winter)\b|\b\d{1,2}(st|nd|rd|th)\s+century\b|"
                   r"\bcentur(y|ies)\b|\b\d{1,2}\s*(am|pm)\b", re.I)
_WEIGHT_UNITS = re.compile(r"\d[\d,.]*\s*(kg|g|lb|t|tonnes?|tons?|kilograms?|pounds?)\b", re.I)
_LENGTH_UNITS = re.compile(r"\d[\d,.]*\s*(?:square\s+|cubic\s+|sq\s+)?(m|km|cm|mm|ft|mi|metres?|meters?|kilomet(?:re|er)s?|miles?|feet|foot|inch(?:es)?)\b",
                           re.I)
_WHERE_HINT = re.compile(r"\b(lies|lie|located|situated|found|lives|live|inhabit\w*|stands|flows|borders|between|"
                         r"north|south|east|west|scattered|distributed|throughout|native|across|spread|occurs?|"
                         r"widespread|range|indigenous|endemic|inhabit\w*|where)\b", re.I)


_DENIES = re.compile(r"\b(not|no|never|none|cannot|neither|nor|only|except|unlike|without|rarely)\b|n't", re.I)
def _speaks_as_i(sentence: str) -> bool:
    """First person ("I like it") but not a name with a numeral ("Francis I")."""
    return bool(_FIRST_PERSON.search(re.sub(r"\b[A-Z][a-z]+\s+(I{1,3}|IV|V|VI{0,3})\b", "", sentence)))


_LIFE_WORDS = {"born", "birth", "die", "died", "death", "dead", "bear"}
_PERSON = re.compile(r"\(\s*(?:born\s+)?(?:\d{1,2}\s+\w+\s+)?\d{3,4}\s*[–-]|\bborn\b")
_FIRST_PERSON = re.compile(r"\b(I|me|my|mine|we|our|you|your)\b")
_EXPLETIVE = re.compile(r"\b(makes?|made|find|finds|found|think|thinks)\s+it\b|\bit\s+(is|was|seems|appears|has been)\s+\w+\s+(to|that)\b"
                        r"|\bit\s+(is|was)\s+(not\s+)?(known|said|believed|thought|estimated|reported|clear)\b", re.I)
_NOT_A_SUBJECT = {"to", "how", "if", "when", "in", "on", "at", "by", "for", "with", "after", "before", "first",
                  "step", "then", "next", "there", "here", "this", "that", "these", "those", "what", "why", "where"}
_DIMENSION = re.compile(r"^\W*how\s+(tall|deep|wide|high|long|big|large|heavy|fast)\b", re.I)
_SEVERAL = re.compile(r"\b(and|also|both|as well)\b|,", re.I)


def _has_dimension(text: str, stems: set[str]) -> bool:
    """"How deep...?" wants a sentence that talks about depth, not any sentence with a length in it."""
    m = _DIMENSION.match(text)
    if m and m.group(1).lower() == "long" and re.search(r"\b(live|lives|last|lasts|survive|lifespan|take|takes)\b", text, re.I):
        return True  # "How long do they live?" asks for a time, not a length
    return not m or bool({stem(w) for w in family(m.group(1))} & stems)


def _amount_pattern(text: str):
    """What an answer to "How old/tall/heavy/many...?" has to contain."""
    if _WHEN.match(text):
        return _DATE
    m = re.match(r"^\W*how\s+(\w+)", text, re.I)
    word = m.group(1).lower() if m else ""
    if word == "many" or word == "much" or word == "often":
        return _NUMBER
    if word == "old":
        return _OLD_AGE
    if word == "long" and re.search(r"\b(live|lives|last|lasts|survive|lifespan|take|takes)\b", text, re.I):
        return _AGE_UNITS
    if word == "heavy":
        return _WEIGHT_UNITS
    if word in ("tall", "long", "high", "deep", "wide", "far", "big", "large"):
        return _LENGTH_UNITS
    return _MEASURED


# The first set gives the old hand-made score: it still decides how sure an answer is. The learned weights
# (aimodel/ranking.py, saved in aimodel/data/rank_weights.json) decide the order of the sentences.
DEFAULT_RANK_WEIGHTS = {"base": 1.0, "pos0": 0.1, "definition": 0.1, "about": 0.1, "intro": -0.3,
                        "where": 0.15, "how": 0.15, "bonus": 1.0}
RANK_WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "rank_weights.json")


def _learned_weights() -> dict[str, float] | None:
    try:
        with open(RANK_WEIGHTS_PATH, encoding="utf-8") as f:
            return {k: float(v) for k, v in json.load(f).items()}
    except (OSError, ValueError):
        return None


RANK_SCALE = 4.0  # a learned score's units are about this many times the hand-made score's


class LearningModel(StudyMixin, WriterMixin, SmallTalkMixin):
    # When the network knows this many words, it is rebuilt with bigger vectors.
    GROWTH = ((8000, 96), (30000, 160))

    def __init__(self, threshold: float = 0.35, knowledge_threshold: float = 0.15,
                 order: int = 2, seed: int | None = None):
        self.threshold = threshold
        self.knowledge_threshold = knowledge_threshold
        self.order = order
        self.memories: list[dict] = []   # {"prompt", "response", "weight"}
        self.knowledge: list[dict] = []  # {"text", "source", "pos"}
        self.facts: list[dict] = []      # {"subj", "rel", "verb", "obj", "neg", "source", "text"}
        self._known: dict[str, dict] = {}  # sentence -> its knowledge/shelf entry
        # context tuple (as "a b" string key) -> Counter of next words
        self.chain: dict[str, Counter] = defaultdict(Counter)
        self.neural = WordEmbeddings(seed=seed or 0)
        self.ranker = FeedbackRanker(seed=seed or 0)
        self.pretrained: PretrainedVectors | None = None
        # Which word vectors a saved brain uses: "builtin" (ships with Aimodel, not stored in the brain),
        # "custom" (your own file, stored beside the brain) or "off". Only load() acts on it.
        self.vectors_mode = "builtin"
        self.fetch = web.http_get  # swap out in tests to avoid the network
        self.notify = None  # optional callback for progress messages
        self._rng = random.Random(seed)
        self._tfidf_cache = None   # (size, TfidfIndex, token lists)
        self._vector_cache = None  # (size, neural version, matrix)
        self._pretrained_cache = None
        self._vectors_changed = False
        self._expand_cache: dict = {}
        self._names_cache = (-1, set())
        self._vocab_cache = (None, set())
        self._candidate_log = None
        # how much each ranking feature counts (the first six reproduce the hand-set scoring)
        self.rank_weights = dict(DEFAULT_RANK_WEIGHTS)
        self.evidence_first = False
        self.learned_weights = _learned_weights()  # None: fall back to the hand-made order
        # how pretrained word vectors are used: to widen question words, and/or to compare whole sentences
        # (widening question words with similar words let wrong sentences in; comparing whole sentences did not)
        self.vector_settings = {"expand": True, "sentence": True, "k": 3, "min": 0.85}
        self._df_cache = (0, Counter())
        self._topic_cache = (-1, {})
        self.last_match: int | None = None
        self.last_query: str | None = None
        self.last_reply: str | None = None
        self.last_source: str | None = None  # "memory" | "knowledge" | "noted" | None
        self.last_trace: list[dict] = []
        self._init_study()
        self._init_writer()
        self._init_smalltalk()

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
        self._note_interest(prompt + " " + response)
        self.neural.train([tokenize(prompt) + tokenize(response)], epochs=10, min_pairs=1000)

    def _similarity(self, tfidf: np.ndarray, neural: np.ndarray) -> np.ndarray:
        # The network can raise a match (different words, same meaning) but
        # never lowers a match on shared words.
        return np.maximum(tfidf, 0.5 * tfidf + 0.5 * np.clip(neural, 0, 1))

    def rank(self, text: str) -> list[tuple[float, int]]:
        """Return (score, memory_index) pairs for taught memories, best first."""
        if not self.memories:
            return []
        # Match on meaningful words only: otherwise "what is ..." matches every question.
        prompts = [keywords(m["prompt"]) or tokenize(m["prompt"]) for m in self.memories]
        query = keywords(text) or tokenize(text)
        tfidf = np.array(TfidfIndex(prompts).scores(query))
        vecs = self.neural.sentence_vectors(prompts + [query])
        meaning = (vecs[:-1] @ vecs[-1]) * self.neural.maturity
        if self.pretrained is not None:
            pre = self.pretrained.sentence_vectors(prompts + [query])
            meaning = np.maximum(meaning, pre[:-1] @ pre[-1])
        sims = self._similarity(tfidf, meaning)
        trust = [0.5 + 0.5 * math.tanh(m["weight"]) for m in self.memories]
        scored = [(float(s * w), i) for i, (s, w) in enumerate(zip(sims, trust))]
        scored.sort(reverse=True)
        return scored

    def respond(self, text: str, use_web: bool = False, notify=None,
                learn: bool = True) -> tuple[str | None, float]:
        """Reply to `text`. Returns (response or None if unsure, confidence).

        Tries taught memories first. Statements about facts are remembered.
        Questions are answered by reasoning over what it has studied; if that
        fails it looks on its shelf, then (if `use_web`) in the dictionary and
        on Wikipedia. With `learn=False` (used for exams) nothing is learned
        from the exchange and no outside help is used.
        """
        notify = notify or self.notify
        if learn:
            self.observe(text)
            self._note_interest(text)
        self.last_query, self.last_reply, self.last_source = text, None, None
        self.last_match, self.last_trace = None, []

        ranked = self.rank(text)
        best = ranked[0][0] if ranked else 0.0
        # Greetings ("Hi, how are you?"). Replies you taught me always win over built-in ones.
        talk = None if (ranked and best >= self.threshold) else self._small_talk(text)
        if talk and talk["rest"]:  # "Hi, what is Rex?": greet, then answer the rest
            reply, confidence = self.respond(talk["rest"], use_web=use_web, notify=notify, learn=learn)
            if reply is not None:
                self.last_reply = f"{talk['reply']} {reply}"
                return self.last_reply, confidence
            talk = None  # more than a greeting that I can't answer: handle the whole message
        if (calc := mathsolver.solve(text)) is not None:  # sums are calculated, never guessed or recalled
            self.last_reply = calc["error"] if "error" in calc else (
                f"{calc['expression']} = {calc['answer']}"
                + (f"  (steps: {'; '.join(calc['steps'])})" if len(calc["steps"]) > 1 else ""))
            self.last_source, best = "calculation", 1.0
            self.last_trace = [{"kind": "calculation", "text": step, "source": "my calculator"}
                               for step in calc["steps"]]
        elif ranked and best >= self.threshold:
            idx = ranked[0][1]
            mem = self.memories[idx]
            self.last_match, self.last_source = idx, "memory"
            self.last_reply = mem["response"]
            self.last_trace = [{"kind": "memory", "text": f"{mem['prompt']} -> {mem['response']}",
                                "source": "taught by you", "score": best,
                                "matched": sorted(set(keywords(text)) & set(keywords(mem["prompt"])))}]
            if learn:
                self.reinforce(mem, 0.1)
        elif talk is not None:
            self.last_reply, self.last_source, best = talk["reply"], "smalltalk", 1.0
            self._last_talk = talk["templates"]
            self.last_trace = [{"kind": "smalltalk", "source": "my greetings",
                                "text": "Recognised a greeting: " + ", ".join(talk["intents"])}]
        elif not looks_like_question(text):
            self._updated = []
            noted = (self._learn_name(text) or self.note(text)) if learn else []
            if noted:
                self.last_reply = "Got it: " + " ".join(rsn.state(f) for f in noted)
                if any(" ".join(f["subj"].lower().split()) == "my name" for f in noted):
                    self.last_reply += " Nice to meet you!"
                if self._updated:
                    self.last_reply += (" (Updated - before, I had: "
                                        + " ".join(rsn.state(f) for f in self._updated) + ")")
                self.last_source, best = "noted", 1.0
                self.last_trace = [{"kind": "fact", "text": rsn.state(f), "source": "you said"}
                                   for f in noted]
        else:
            answer = self.explain(text)
            if answer is None and learn and self.recall(text):  # look in the books on the shelf
                answer = self.explain(text)
            if answer is None and learn and use_web:
                if self.define_unknown(text):  # words I've never seen: ask the dictionary
                    answer = self.explain(text)
                if answer is None:
                    if notify:
                        notify("Searching the web...")
                    if self.search_web(text):
                        answer = self.explain(text)
                        if answer is None and self.recall(text):
                            answer = self.explain(text)
            if answer is not None:
                reply, best, trace = answer
                self.last_reply, self.last_trace = self._write(text, reply, trace, learn)
                self.last_source = "knowledge"
                if learn:
                    self._mark_used(self.last_trace)
                    self._close_gap(text)
            elif learn:
                self._add_gap(text, "unanswered")

        if learn:
            self.neural.train([tokenize(text)], epochs=1)
        return self.last_reply, best

    def feedback(self, good: bool) -> bool:
        """Reward or penalise the last response. Returns False if nothing to rate."""
        if self.last_source == "smalltalk":
            self._rate_talk(good)  # say it more (or less) often
            return True
        if self.last_source == "memory" and self.last_match is not None:
            mem = self.memories[self.last_match]
            mem["weight"] += 0.5 if good else -1.0
            if good:
                self.reinforce(mem, 0.3, recalled=True)
            else:
                self.weaken(mem, 0.4)
            return True
        if self.last_source == "knowledge":
            evidence = [t["features"] for t in self.last_trace if "features" in t]
            if evidence:  # teach the feedback network what good evidence looks like
                self.ranker.train(evidence, [1.0 if good else 0.0] * len(evidence))
            self._grade_trace(good)  # and strengthen/weaken what the answer was built from
            self._rate_last(good)    # and remember the rating to train the writer
            if good:  # a good researched answer becomes a trusted memory
                self.learn(self.last_query, self.last_reply)
                self.last_match, self.last_source = len(self.memories) - 1, "memory"
            return True
        return False

    def correct(self, prompt: str, better_response: str) -> None:
        """Penalise the last answer and learn a better one."""
        self.feedback(good=False)
        self._correct_last(prompt, better_response)
        self.learn(prompt, better_response)

    def forget(self, text: str) -> int:
        """Forget memories whose prompt contains `text`. Returns count removed."""
        needle = text.lower()
        before = len(self.memories)
        self.memories = [m for m in self.memories if needle not in m["prompt"].lower()]
        self.last_match = None
        return before - len(self.memories)

    # --------------------------------------------------------------- knowledge
    def add_document(self, text: str, source: str, topic: str | None = None,
                     focus: str | None = None) -> int:
        """Learn from a block of text. Returns the number of new sentences studied.

        Short texts and your own files are studied in full. Long web pages are
        sorted by importance: the best sentences are studied and the rest go on
        the shelf for later (see `last_read`). `topic` (e.g. an article title)
        lets "It is ..." sentences become facts; `focus` is the question the
        text is being read for.
        """
        fresh, parsed, promote, last_subject = [], {}, [], topic
        about, person_name = topic, None
        if topic is None:
            first = next(iter(split_sentences(text, min_words=3)), "")
            lead = re.match(r"^(?:(?:The|A|An)\s+)?((?:[\w'-]+\s+){0,3}?[\w'-]+?)\s*(?:\(|\b(?:is|are|was|were)\b)", first)
            if lead and lead.group(1).split()[0].lower() in _NOT_A_SUBJECT:
                lead = None  # "To plant a seed is easy" / "In 1889 it was..." name nothing
            last_subject = lead.group(1) if lead and len(lead.group(1).split()) <= 4 else None
            if _PERSON.search(first):  # a biography: "he/she" can be named too
                person_name = (os.path.splitext(os.path.basename(source))[0].replace("_", " ")
                               if not source.startswith("http") else None) or last_subject
            article = re.match(r"^(The|A|An)\s", first)
            about = (f"{article.group(1)} {last_subject}" if article and last_subject else last_subject)
        for pos, s in enumerate(split_sentences(text, min_words=2 if source == "you said" else 3)):
            s = self._name_the_subject(s, last_subject, about if pos and source != "you said" else None,
                                       person_name if pos and source != "you said" else None)
            hint = last_subject
            fact = rsn.extract_fact(s, topic=hint)
            if fact:
                last_subject = topic or fact["subj"]
            known = self._known.get(s)
            if known is not None:
                if known.get("shelved"):  # read again: it recurs, so it matters
                    known["again"] = known.get("again", 0) + 1
                    promote.append(known)
                else:
                    self.reinforce(known, 0.1)
                continue
            entry = {"text": s, "source": source, "pos": pos}
            if hint:
                entry["hint"] = hint
            fresh.append(entry)
            parsed[s] = fact

        keep = self._choose_keep(fresh, source, focus)
        studied = [e for i, e in enumerate(fresh) if i in keep]
        for i, e in enumerate(fresh):
            if i in keep:
                self._remember(e, parsed[e["text"]])
            else:
                e["shelved"] = True
                self._known[e["text"]] = e
                self.shelf.append(e)
        if promote:
            self.shelf = [e for e in self.shelf if not any(e is p for p in promote)]
            for e in promote:
                self._remember(e)
            studied += promote
        self.last_read = {"kept": len(studied), "shelved": len(fresh) - len(keep)}
        if studied:
            self.neural.train([tokenize(e["text"]) for e in studied], epochs=5, min_pairs=3000)
            self._maybe_grow()
        return len(studied)

    @staticmethod
    def _name_the_subject(sentence: str, subject: str | None, about: str | None = None,
                          person: str | None = None) -> str:
        """Say who "it" is: "It lies in the Himalayas." -> "Mount Everest lies in the Himalayas."

        Answers are looked up sentence by sentence, so a sentence that only says "it" can never be found
        by asking about the thing it is about. A sentence that *starts* with It/They takes the subject of
        the sentence before it. `about` (what the whole text is about) also fills in a later "it":
        "At 7,088 km long, it is the longest river" -> "..., the Nile is the longest river".
        """
        if person and not _speaks_as_i(sentence):
            named = " ".join(person.split())
            if re.match(r"^(He|She)\s", sentence):
                return f"{named} " + sentence.split(" ", 1)[1]
            if re.match(r"^His\s", sentence):
                return f"{named}'s " + sentence.split(" ", 1)[1]
            words = {w.lower() for w in re.findall(r"[A-Za-z']+", sentence)}
            if (words & {"he", "she", "him", "his"} and not words & {"it", "they", "them"}
                    and named.split()[0].lower() not in words
                    and re.search(r",\s+(he|she)\b", sentence, re.I)):  # "Upon his invitation, he spent..."
                return re.sub(r"\b(he|she|him)\b", named, sentence, count=1, flags=re.I)
        if subject and len(subject.split()) <= 4 and not _EXPLETIVE.search(sentence):
            name = " ".join(subject.split())
            if re.match(r"^(Its|Their)\s", sentence):
                return f"{name}{chr(39) if name.endswith('s') else chr(39) + 's'} " + sentence.split(" ", 1)[1]
            if re.match(r"^(It|They)\s", sentence):
                return f"{name} " + sentence.split(" ", 1)[1]
        same = bool(subject and about and subject.split()[-1].lower() == about.split()[-1].lower())
        fronted = bool(re.search(r",\s+(it|they)\b", sentence, re.I))  # "At 7,088 km long, it is..."
        if about and len(about.split()) <= 4 and (same or fronted) and not _speaks_as_i(sentence):
            name = " ".join(about.split())
            if re.match(r"(?i)(a|an)\s", name):
                name = "the " + name.split(" ", 1)[1]
            plural = name.split()[-1].lower().endswith("s") and not name.lower().endswith(("ss", "us", "is"))
            pronouns = ("they", "their") if plural else ("it", "its")
            other = ("it", "its") if plural else ("they", "their")
            words = {w.lower() for w in re.findall(r"[A-Za-z']+", sentence)}
            if (words & set(pronouns) and not words & set(other)
                    and name.split()[-1].lower() not in words and not _EXPLETIVE.search(sentence)):
                def fill(m: re.Match) -> str:
                    word = m.group(0).lower()
                    text = name[0].upper() + name[1:] if m.start() == 0 else name[0].lower() + name[1:] \
                        if re.match(r"(?i)the\s", name) else name
                    return text + (("'" if text.endswith("s") else "'s") if word in ("its", "their") else "")
                return re.sub(r"\b(it|its|they|their)\b", fill, sentence, flags=re.I)
        return sentence

    def note(self, text: str) -> list[dict]:
        """Remember facts you state in chat ("My sister lives in Delhi")."""
        before = len(self.facts)
        if not any(rsn.extract_fact(s) for s in split_sentences(text, min_words=2)):
            return []
        self.add_document(text, "you said")
        new = self.facts[before:]
        for fact in new:
            self._supersede(fact)  # "lives in Mumbai" replaces "lives in Delhi"
        return new

    def _maybe_grow(self) -> bool:
        """Rebuild the network with bigger word vectors once it knows enough words."""
        for words, dim in self.GROWTH:
            if len(self.neural) >= words and self.neural.dim < dim:
                if self.notify:
                    self.notify(f"I know {len(self.neural)} words now - growing my neural "
                                f"network to {dim} dimensions and retraining...")
                self.neural = WordEmbeddings(dim=dim, seed=self.neural.dim)
                self.neural.train(self._corpus(), epochs=3, min_pairs=20000)
                return True
        return False

    def load_vectors(self, path: str, max_words: int = 50_000) -> int:
        """Load pretrained word vectors (GloVe/fastText text files)."""
        self.pretrained = PretrainedVectors.from_text(path, max_words)
        self.vectors_mode = "custom"
        self._pretrained_cache, self._expand_cache = None, {}
        self._vectors_changed = True
        return len(self.pretrained)

    def load_builtin_vectors(self) -> int:
        """Use the word vectors that ship with Aimodel. Returns how many words, or 0 if the file is missing."""
        vectors = PretrainedVectors.builtin()
        if vectors is None:
            return 0
        self.pretrained, self.vectors_mode = vectors, "builtin"
        self._pretrained_cache, self._expand_cache = None, {}
        return len(vectors)

    def vectors_off(self) -> None:
        """Stop using word vectors (a custom file stays on disk until you load vectors again)."""
        self.pretrained, self.vectors_mode = None, "off"
        self._pretrained_cache, self._expand_cache = None, {}

    def read(self, path_or_url: str) -> int:
        """Learn from a local file or a web page."""
        if re.match(r"https?://", path_or_url):
            return self.add_document(web.fetch_page(path_or_url, get=self.fetch), path_or_url)
        with open(os.path.expanduser(path_or_url), "rb") as f:  # text, markdown or PDF
            data = f.read()
        return self.add_document(extract(path_or_url, data), os.path.basename(path_or_url))

    def search_web(self, query: str) -> int:
        """Look `query` up on Wikipedia and learn from the results."""
        terms = " ".join(keywords(query)) or query
        added = 0
        for title, text, url in web.wikipedia(terms, get=self.fetch):
            added += self.add_document(text, url, topic=title, focus=query)
        return added

    def _topic_of(self, source: str) -> set[str]:
        """What a text is about: the words of its name and of its first sentence ("Octopus.txt")."""
        size = len(self.knowledge)
        if self._topic_cache[0] != size:
            self._topic_cache = (size, {})
        cache = self._topic_cache[1]
        if source not in cache:
            first = next((k["text"] for k in self.knowledge if k["source"] == source), "")
            name = os.path.splitext(os.path.basename(source))[0].replace("_", " ")
            lead = " ".join(first.split()[:12])
            words = [t for t in tokenize(f"{name} {lead}") if t not in STOPWORDS]
            cache[source] = {stem(f) for t in words for f in (t, t + "s", t + "es")}  # "octopus" ~ "octopuses"
        return cache[source]

    def _knowledge_index(self):
        size = len(self.knowledge)
        if self._tfidf_cache is None or self._tfidf_cache[0] != size:
            toks = [tokenize(k["text"]) for k in self.knowledge]
            stems = [{stem(t) for t in tk} | {stem(t[:-2]) for t in tk if len(t) > 6 and t.endswith("ed")}
                     for tk in toks]  # "eight-limbed" also counts as "limb"
            self._tfidf_cache = (size, TfidfIndex([[stem(t) for t in tk] for tk in toks]), toks, stems)
        _, index, toks, stems = self._tfidf_cache
        key = (size, self.neural.version)
        if self._vector_cache is None or self._vector_cache[0] != key:
            self._vector_cache = (key, self.neural.sentence_vectors(toks))
        pre = None
        if self.pretrained is not None:
            if self._pretrained_cache is None or self._pretrained_cache[0] != size:
                self._pretrained_cache = (size, self.pretrained.sentence_vectors(toks))
            pre = self._pretrained_cache[1]
        return index, toks, stems, self._vector_cache[1], pre

    def _meaning(self, query: list[str], own: np.ndarray, pre: np.ndarray | None) -> np.ndarray:
        scores = (own @ self.neural.sentence_vectors([query])[0]) * self.neural.maturity
        if pre is not None and self.vector_settings["sentence"]:
            scores = np.maximum(scores, pre @ self.pretrained.sentence_vectors([query])[0])
        return scores

    def _expand(self, word: str) -> set[str]:
        """The word plus words that mean nearly the same (from the networks)."""
        if word not in self._expand_cache:
            out = {stem(word)} | {stem(w) for w in family(word)}
            if len(family(word)) > 1:  # a word with stand-ins: "die" also means "died", "dies"
                out |= {stem(f) for w in family(word) for f in (w + "d", w + "ed", w + "s", w + "ing")}
            if self.pretrained is not None and self.vector_settings["expand"]:
                cfg = self.vector_settings
                out |= {stem(w) for w, s in self.pretrained.similar(word, cfg["k"]) if s >= cfg["min"]}
            if len(self.neural) >= 5000:
                out |= {stem(w) for w, s in self.neural.similar(word, 10) if s >= 0.8}
            self._expand_cache[word] = out
        return self._expand_cache[word]

    def _wanted(self, text: str) -> dict[str, set[str]]:
        words = keywords(text) or tokenize(text)
        return {stem(w): self._expand(w) for w in words}

    def _key_word(self, wanted: dict[str, set[str]]) -> str:
        """The most specific word of a question (rarest in what I've read).

        An answer that doesn't cover it isn't about the question at all, e.g.
        photosynthesis text mentioning "work" for "How do vaccines work?".
        """
        n = len(self.knowledge)
        if n and self._df_cache[0] != n:
            self._df_cache = (n, Counter(w for doc in self._knowledge_index()[2] for w in doc))
        df = self._df_cache[1] if n else Counter()

        def rarity(w):  # how few sentences have the word (or a near synonym); never-seen words are rarest
            # a word is only as rare as its commonest stand-in ("summit" is not rare because a vector says
            # it is like "meeting", which I never read)
            return min((math.log((n + 1) / (1 + df.get(v, 0))) for v in wanted[w]), default=99.0)
        return max([w for w in wanted if w not in WEAK_WORDS] or wanted, key=rarity)

    def answer_from_knowledge(self, text: str, max_sentences: int = 3, bonus=None):
        """Gather evidence sentences for a question.

        Picks the best-matching sentence, then adds sentences that cover parts
        of the question the evidence doesn't cover yet. The feedback network
        adjusts the ranking. Returns (answer, confidence, trace) or None.
        """
        if not self.knowledge:
            return None
        wanted = self._wanted(text)
        if not wanted:
            return None
        query = keywords(text) or tokenize(text)
        index, toks, stems, own, pre = self._knowledge_index()
        tfidf = np.array(index.scores([stem(w) for w in query]))  # the index counts word stems
        meaning = self._meaning(query, own, pre)
        base = self._similarity(tfidf, meaning)
        topic = "|".join(map(re.escape, wanted))
        definition = re.compile(r"\b(%s)\w*\s+(is|are|was|were|refers|means|consists)\b" % topic, re.I)
        about = re.compile(r"^(the |a |an )?(\w+ )?(%s)\w*\b" % topic, re.I)  # topic is the subject

        quantity = bool(_QUANTITY.search(text))
        amount = _amount_pattern(text)
        content = {w for w in wanted if w not in WEAK_WORDS}
        where = bool(re.match(r"^\W*where\b", text, re.I))
        how = bool(re.match(r"^\W*how\s+(do|does|did|can|could|is|are|to)\b", text, re.I))
        key = self._key_word(wanted)
        words = [stem(t) for t in tokenize(text) if t not in STOPWORDS]
        pairs = [words[n:n + 2] for n in range(len(words) - 1)]  # neighbouring words of the question
        cands = []
        for i in (int(i) for i in np.argsort(-base)[:60]):
            covered = {w for w, alts in wanted.items() if alts & stems[i]}
            if not covered:
                continue
            k = self.knowledge[i]
            context = {w for w, alts in wanted.items() if alts & self._topic_of(k["source"])}
            if (_WHEN.match(text) or re.search(r"\b(born|birth)\b", text, re.I)) and _PERSON.search(k["text"]):
                # "Name (15 April 1452 - 2 May 1519) was..." says when they were born and died
                covered |= {w for w, alts in wanted.items() if alts & _LIFE_WORDS}
            if content and not (covered | context) & content:
                continue  # it must be about something the question asks about, not just a filler word
            if quantity and not (amount.search(k["text"]) and _has_dimension(text, stems[i])):
                continue  # "How tall is...?" needs a sentence that gives an amount
            web_source = k["source"].startswith("http")
            feats = [tfidf[i], meaning[i], len(covered) / len(wanted),
                     float(k.get("pos", 99) < 2), float(bool(definition.search(rsn.clean(k["text"])))),
                     min(1.0, len(toks[i]) / 40), float(not web_source), float(web_source)]
            sentence_stems = [stem(t) for t in toks[i]]
            named = {
                "base": float(base[i]), "tfidf": float(tfidf[i]), "meaning": float(meaning[i]),
                "coverage": len(covered) / len(wanted),
                "content": len((covered | context) & content) / len(content) if content else 1.0,
                "key": float(key in (covered | context)),
                "pos0": feats[3], "definition": feats[4],
                "about": float(bool(about.search(rsn.clean(k["text"])))),
                "intro": float(bool(_INTRO.search(k["text"]))),
                "where": float(where and bool(_WHERE_HINT.search(k["text"]))),
                "how": float(how and bool(_MANNER.search(k["text"]))),
                "length": feats[5], "web": feats[7],
                "bigram": float(any(sentence_stems[n:n + 2] == pair for pair in pairs
                                    for n in range(len(sentence_stems) - 1))),
                "bonus": float(bonus(i, feats[2])) if bonus else 0.0,
            }
            score = sum(self.rank_weights.get(name, 0.0) * value for name, value in named.items())
            order = (sum(self.learned_weights.get(name, 0.0) * value for name, value in named.items())
                     if self.learned_weights else score * RANK_SCALE)
            cands.append({"i": i, "score": float(score), "order": float(order), "covered": covered,
                          "context": context, "features": feats, "named": named})
        if not cands:
            return None
        for c, adj in zip(cands, self.ranker.adjust([c["features"] for c in cands])):
            c["score"] += float(adj)
            c["order"] += float(adj) * RANK_SCALE
        if self._candidate_log is not None:  # for measuring the ranking (see aimodel/ranking_test.py)
            self._candidate_log.append({"question": text, "candidates": list(cands)})

        cands.sort(key=lambda c: -c["order"])
        top = max(c["score"] for c in cands)  # how sure I am still comes from the hand-made score
        chosen, covered = [], set()
        while cands and len(chosen) < max_sentences:
            c = max(cands, key=lambda c: c["order"] + 0.15 * RANK_SCALE * len(c["covered"] - covered))
            cands.remove(c)
            new_terms = (c["covered"] - covered) & (content or c["covered"])
            if chosen and (not new_terms or c["score"] < 0.5 * top):
                continue
            chosen.append(c)
            covered |= new_terms

        about = set().union(*(c["context"] for c in chosen))  # what the whole text is about, e.g. "Octopus"
        coverage = len(covered | about) / len(wanted)
        confidence = float(min(1.0, top)) * (0.4 + 0.6 * coverage)
        # The evidence has to mention the question's key word and at least
        # half of what you asked about. A short question must be covered in full: "the population of
        # Nepal" is not answered by a sentence that only mentions Nepal.
        together = any(content <= (c["covered"] | c["context"]) for c in chosen)  # one sentence says it all
        one_source = len({self.knowledge[c["i"]]["source"] for c in chosen}) == 1
        if (coverage < 0.5 or self._key_word(wanted) not in (covered | about)
                or confidence < self.knowledge_threshold
                or (0 < len(content) <= 3 and not content <= (covered | about))
                or (1 < len(content) <= 3 and not together and not one_source and not _SEVERAL.search(text))):
            return None
        trace = [{"kind": "evidence", "text": self.knowledge[c["i"]]["text"],
                  "source": self.knowledge[c["i"]]["source"], "pos": self.knowledge[c["i"]].get("pos", 0),
                  "score": c["score"], "matched": sorted(c["covered"]), "features": c["features"]}
                 for c in chosen]
        return " ".join(t["text"] for t in trace), confidence, trace

    # --------------------------------------------------------------- reasoning
    def _relevant_facts(self, text: str) -> tuple[list[dict], float]:
        """Facts that together cover the question. Returns (facts, coverage)."""
        wanted = self._wanted(text)
        if not wanted or not self.facts:
            return [], 0.0
        personal = bool(rsn.PERSONAL & set(tokenize(text)))
        quantity = bool(_QUANTITY.search(text))
        amount = _amount_pattern(text)
        cands = []
        for f in self.facts:
            if quantity and not (amount.search(f["text"])
                                 and _has_dimension(text, {stem(t) for t in tokenize(f["text"])})):
                continue
            stems = {stem(t) for t in tokenize(f"{f['subj']} {f['verb']} {f['obj']}")}
            covered = {w for w, alts in wanted.items() if alts & stems}
            if not covered:
                continue
            subj = {stem(t) for t in tokenize(f["subj"])}
            score = len(covered) / len(wanted) + 0.3 * bool(set(wanted) & subj)
            if personal:
                score += 0.3 if rsn.is_personal(f) or f["source"] == "you said" else -0.2
            cands.append((score, covered, f))
        cands.sort(key=lambda c: -c[0])
        chosen, covered, each = [], set(), []
        for score, cov, f in cands:
            if len(chosen) >= 4:
                break
            if not chosen or (cov - covered and score >= 0.5 * cands[0][0]):
                chosen.append(f)
                each.append(cov)
                covered |= cov
        if chosen:
            top = chosen[0]
            # "Tell me about the gorzu" asks about nothing but its subject: say all I know.
            broad = set(wanted) <= {stem(t) for t in tokenize(top["subj"])}
            for score, cov, f in cands:
                if len(chosen) < 4 and f not in chosen and _bare(f["subj"]) == _bare(top["subj"]) and (
                        broad or (score >= 0.9 * cands[0][0] and f["rel"] == top["rel"])):
                    chosen.append(f)  # also lists: "I like tea" and "I like coffee"
        coverage = len(covered) / len(wanted)
        content = {w for w in wanted if w not in WEAK_WORDS}
        if coverage < 0.5 or self._key_word(wanted) not in covered:
            return [], 0.0
        if 0 < len(content) <= 3 and not content <= covered and not _SEVERAL.search(text):
            return [], 0.0  # "the population of Nepal" is not answered by a fact about a population
        if (1 < len(content) <= 3 and not any(content <= cov for cov in each)
                and len({f["source"] for f in chosen}) > 1 and not _SEVERAL.search(text)):
            return [], 0.0  # "What do camels eat?" is not answered by a camel fact plus an unrelated eating fact
        return chosen, coverage

    def _prove(self, text: str):
        """Answer "Is X a Y?" by chaining is-a facts."""
        words = tokenize(text)
        if len(words) < 3 or words[0] not in rsn.BE:
            return None
        rest = words[1:]
        for k in range(1, len(rest)):
            subj, obj = " ".join(rest[:k]), " ".join(rest[k:])
            a, b = rsn.head(subj), rsn.head(obj)
            if not a or not b or a == b:
                continue
            for f in self.facts:
                if f["neg"] and rsn.head(f["subj"]) == a and rsn.head(f["obj"]) == b:
                    return (f"No. {rsn.state(f)}", 0.9,
                            [{"kind": "fact", "text": rsn.state(f), "source": f["source"],
                              "sentence": f["text"]}])
            path = rsn.isa_chain(self.facts, a, b)
            if path:
                links = [rsn.state(f).rstrip(".") for f in path]
                steps = [{"kind": "fact", "text": rsn.state(f), "source": f["source"],
                          "sentence": f["text"]} for f in path]
                # "The sperm whale is a mammal" says something about one kind of whale,
                # so for "a whale" it's a good guess, not a proof.
                narrower = {stem(w) for w in tokenize(_bare(path[0]["subj"]))} - {stem(w) for w in tokenize(_bare(subj))}
                verdict, conf = ("Probably yes", 0.6) if narrower else ("Yes", 0.9 if len(path) == 1 else 0.8)
                if narrower:
                    steps.append({"kind": "inference", "source": "my reasoning",
                                  "text": f"'{path[0]['subj']}' is one kind of {a}, so this is a generalization"})
                if len(path) == 1:
                    return f"{verdict}. {links[0]}.", conf, steps
                chain = ", and ".join([links[0]] + [self._lower(l) for l in links[1:]])
                steps.append({"kind": "inference", "source": "my reasoning",
                              "text": f"Chained {len(path)} facts: {' -> '.join(rsn.head(f['subj']) for f in path)} -> {b}"})
                return f"{verdict}. {chain}, so {subj} {words[0]} {obj}.", conf, steps
        return None

    def explain(self, text: str):
        """Answer a question in the model's own words, with the reasoning steps.

        Returns (answer, confidence, steps) or None.
        """
        text = self.understand_question(text)
        kind = rsn.question_kind(text)
        asked = members_question(text)
        if asked:
            found = members(self.facts, *asked)
            if found:
                return self._list_members(found, asked[1])
        pair = common_question(text)
        if pair:
            shared = self._in_common(*pair)
            if shared:
                return shared
        if kind == "yesno":
            proof = (self._prove_either(text) or self._prove(text) or self._check_claim(text)
                     or self._said_so(text))
            if proof:
                return proof
        if kind == "why":
            return self._explain_why(text)
        if kind == "how":
            return self._explain_how(text)

        facts, coverage = self._relevant_facts(text)
        evidence = self.answer_from_knowledge(text)
        if evidence is not None:
            matched = set().union(*(set(t["matched"]) for t in evidence[2]))
            if coverage < len(matched) / max(1, len(self._wanted(text))):
                facts, coverage = [], 0.0  # the evidence answers more of the question
            elif (self.evidence_first and facts and evidence[2]
                  and evidence[2][0]["text"] not in {f["text"] for f in facts}):
                facts, coverage = [], 0.0  # the best-ranked sentence is not one of the facts: lead with it
        if not facts and evidence is None:
            return None
        parts, steps, used = [], [], set()
        if facts:
            first = facts[0]
            parts.append(self._say(first))
            steps.append({"kind": "fact", "text": rsn.state(first), "source": first["source"],
                          "sentence": first["text"]})
            used.add(first["text"])
            subject = rsn.head(first["subj"])
            for f in facts[1:]:
                if f["text"] in used:
                    continue
                same = rsn.head(f["subj"]) == subject and not self._trimmed(f)
                parts.append(rsn.state(f, subject=rsn.pronoun(first), also=True) if same
                             else _CONNECTORS[len(parts) % 3] + self._lower(self._say(f)))
                steps.append({"kind": "fact", "text": rsn.state(f), "source": f["source"],
                              "sentence": f["text"]})
                used.add(f["text"])
            link = self._linked_fact(first, used)
            if link and _bare(first["obj"]) == _bare(link["subj"]):
                # "Your dog is called Rex" + "Rex is a retriever" -> one sentence
                parts[0] = parts[0].rstrip(".") + f", which {link['verb']} {link['obj']}."
            elif link:
                parts.append(f"That involves {self._lower(link['subj'])}, and "
                             f"{self._lower(rsn.state(link))}")
            if link:
                steps.append({"kind": "inference", "source": "my reasoning",
                              "text": f"'{first['obj']}' mentions '{link['subj']}', which I know more about"})
                steps.append({"kind": "fact", "text": rsn.state(link), "source": link["source"],
                              "sentence": link["text"]})
                used.add(link["text"])
        confidence = 0.8 * coverage
        if evidence is not None:
            _, ev_conf, trace = evidence
            confidence = max(confidence, ev_conf)
            extra = [t for t in trace if t["text"] not in used][:max(1 if ev_conf >= 0.5 else 0, 3 - len(parts))]
            for t in extra:
                prefix = _CONNECTORS[len(parts) % 3] if parts else ""
                parts.append(self._rephrase(t, prefix))
            steps += [t for t in trace if t["text"] not in used]
        if not parts:
            return None
        if kind == "yesno":
            parts.insert(0, "I can't prove a yes or no from what I know, but here's what I found:")
            confidence *= 0.8
        if evidence is not None:  # keep what the feedback network needs to learn from
            features = {t["text"]: t["features"] for t in evidence[2]}
            for step in steps:
                if step.get("sentence") in features:
                    step["features"] = features[step["sentence"]]
        return " ".join(parts), confidence, steps

    def _said_so(self, text: str):
        """"Are octopuses venomous?" when a studied sentence says exactly that (and nothing denies it)."""
        words = tokenize(text)
        if len(words) < 3 or words[0] not in rsn.BE:
            return None
        wanted = {stem(w) for w in keywords(text)}
        if len(wanted) < 2:
            return None
        for entry in self.knowledge:
            for clause in re.split(r";|,\s+(?:and|but|though|although|while|whereas)\s+", rsn.clean(entry["text"])):
                heard = {stem(t) for t in tokenize(clause)}
                if (wanted <= heard and not _DENIES.search(clause)
                        and len(clause.split()) <= 30):
                    step = {"kind": "evidence", "text": entry["text"], "source": entry["source"],
                            "sentence": entry["text"]}
                    return (f"Probably yes. {rsn.sentence(clause.strip())}", 0.6, [step])
        return None

    def _check_claim(self, text: str):
        """"Does the lorpan eat meat?" from what I know about what the lorpan eats."""
        claim = check_claim(self.facts, text)
        if claim is None:
            return None
        said, used = claim
        steps = [{"kind": "fact", "text": rsn.state(f), "source": f["source"], "sentence": f["text"]}
                 for f in used]
        know = " ".join(rsn.state(f) for f in used)
        if said == "inherited":
            links = [rsn.state(f).rstrip(".") for f in used]
            steps.append({"kind": "inference", "source": "my reasoning",
                          "text": f"Kinds pass on what they are like, but there can be exceptions"})
            return "Probably yes. " + ", and ".join([links[0]] + [self._lower(l) for l in links[1:]]) + ".", 0.7, steps
        if said == "yes":
            return f"Yes. {know}", 0.9, steps
        if said == "no":
            return f"No. {know}", 0.9, steps
        steps.append({"kind": "inference", "source": "my reasoning",
                      "text": "What I know about this doesn't include it, so I answer 'not as far as I know'"})
        return f"No, not as far as I know. {know}", 0.55, steps

    def _in_common(self, a: str, b: str):
        """"What do mammals and birds have in common?": the nearest kind they both are."""
        found = in_common(self.facts, a, b)
        if not found:
            return None
        kind, path_a, path_b = found
        names = [" ".join(tokenize(x)) for x in (a, b)]
        plural = kind if kind.endswith("s") else kind + ("es" if kind.endswith(("sh", "ch", "x")) else "s")
        last = (path_a or path_b)[-1]["obj"].lower()
        if kind == "thing" and "living" in last:
            plural = "living things"
        reply = f"{names[0].capitalize()} and {names[1]} are both {plural}."
        steps, seen = [], set()
        for f in path_a + path_b:
            if f["text"] not in seen:
                seen.add(f["text"])
                steps.append({"kind": "fact", "text": rsn.state(f), "source": f["source"], "sentence": f["text"]})
        steps.append({"kind": "inference", "source": "my reasoning",
                      "text": f"Both lead to '{kind}' through is-a facts"})
        return reply, 0.8, steps

    def _prove_either(self, text: str):
        """"Is the mimbat a bird or a mammal?": say which one I can prove."""
        words = tokenize(text)
        if "or" not in words[2:-1] or words[0] not in rsn.BE:
            return None
        cut = words.index("or", 2)
        left, right = words[1:cut], words[cut + 1:]
        for k in range(1, len(left)):
            subj, first = " ".join(left[:k]), " ".join(left[k:])
            proofs = [(opt, self._prove(f"{words[0]} {subj} {opt}")) for opt in (first, " ".join(right))]
            yes = [(opt, p) for opt, p in proofs if p and p[0].startswith(("Yes", "Probably yes"))]
            if yes:
                reply = " ".join(p[0] for _, p in yes)
                return reply, min(p[1] for _, p in yes), [step for _, p in yes for step in p[2]]
        return None

    def _vocabulary(self) -> set[str]:
        """Every word I have read, for spotting typos (rebuilt only when I learn something)."""
        key = (len(self.knowledge), len(self.facts), len(self.memories), len(self.shelf))
        if self._vocab_cache[0] != key:
            words = set()
            for entry in self.knowledge + self.shelf:
                words.update(tokenize(entry["text"]))
            for m in self.memories:
                words.update(tokenize(m["prompt"]))
            self._vocab_cache = (key, words | {stem(w) for w in words})
        return self._vocab_cache[1]

    def understand_question(self, text: str) -> str:
        return understand(text, self._vocabulary())

    def _list_members(self, found: list[dict], cls: str):
        names = [_bare(f["subj"]) for f in found[:8]]
        if len(names) == 1:
            joined, verb = f"the {names[0]}", "is"
        else:
            joined, verb = "the " + ", the ".join(names[:-1]) + " and the " + names[-1], "are"
        plural = cls if cls.endswith("s") else cls + ("es" if cls.endswith(("sh", "ch", "x")) else "s")
        what = f"a {cls}" if verb == "is" else plural
        reply = (joined[0].upper() + joined[1:]) + f" {verb} {what}."
        steps = [{"kind": "fact", "text": rsn.state(f), "source": f["source"], "sentence": f["text"]}
                 for f in found[:8]]
        steps.append({"kind": "inference", "source": "my reasoning",
                      "text": f"Collected everything I know that is a {cls}"})
        return reply, 0.85, steps

    @staticmethod
    def _trimmed(fact: dict) -> bool:
        """Did the fact lose part of its sentence (so the whole sentence should be said)?"""
        plain = rsn.clean(fact["text"]).rstrip(".!?: ")
        return len(plain) > len(rsn.state(fact).rstrip(".")) + 8

    def _say(self, fact: dict) -> str:
        """A fact as a sentence. If the fact lost part of its sentence (a trailing
        'who loves swimming'), say the whole sentence instead of a cut-off one."""
        text = rsn.state(fact)
        plain = rsn.clean(fact["text"]).rstrip(".!?: ")
        if len(plain) <= len(text.rstrip(".")) + 8:
            return text
        if rsn.is_personal(fact) or fact["source"] == "you said":
            plain = rsn.flip_person(plain)
        return rsn.sentence(plain)

    def _linked_fact(self, fact: dict, used: set) -> dict | None:
        """A fact about something mentioned in `fact`'s object (one reasoning hop)."""
        mentioned = {stem(w) for w in keywords(fact["obj"])} - {rsn.head(fact["subj"])} - _GENERIC
        for f in self.facts:
            if (f["text"] not in used and f["source"] == fact["source"] and not f["neg"]
                    and rsn.head(f["subj"]) in mentioned):
                return f
        return None

    def _explain_how(self, text: str):
        """Explain a process: the best matching sentence and the steps after it."""
        evidence = self.answer_from_knowledge(text, max_sentences=2)
        if evidence is None:
            return None
        _, conf, trace = evidence
        start = trace[0]
        later = sorted((k for k in self.knowledge
                        if k["source"] == start["source"] and 0 < k.get("pos", 0) - start["pos"] <= 6),
                       key=lambda k: k["pos"])
        following = []
        for k in later:
            if re.match(r"^to\b", k["text"], re.I):  # "To wash your hands, ..." starts the next how-to
                break
            following.append({"kind": "evidence", "text": k["text"], "source": k["source"],
                              "pos": k.get("pos", 0)})
            if re.match(r"^(finally|lastly)\b", k["text"], re.I):  # the last step
                break
        if not any(rsn.SEQUENCE_MARKER.search(t["text"]) for t in following):
            following = []  # the next sentences aren't steps of a process
        if not following:
            parts = [self._rephrase(t, _CONNECTORS[n % 3] if n else "") for n, t in enumerate(trace)]
            return " ".join(parts), conf, trace

        # "To make tea, boil water." -> "Here's how to make tea: First, boil water."
        first, header = dict(start), "Here's how:"
        m = re.match(r"^to\s+(.+?),\s+(\S.*)$", rsn.clean(start["text"]), re.I)
        if m:
            header, first["text"] = f"Here's how to {m.group(1)}:", m.group(2)
        steps = [first] + following
        middle = ("Then, ", "After that, ", "Next, ")
        parts = []
        for n, step in enumerate(steps):
            lead = ("First, " if n == 0 else "Finally, " if n == len(steps) - 1
                    else middle[(n - 1) % 3])
            parts.append(self._rephrase(step, lead))
        return header + " " + " ".join(parts), conf, [start] + following

    def _explain_why(self, text: str):
        def causal(i, coverage):  # a stated cause, as long as it's about the question
            return 0.3 if coverage >= 0.5 and rsn.CAUSAL.search(self.knowledge[i]["text"]) else 0.0
        evidence = self.answer_from_knowledge(text, bonus=causal)
        if evidence is None:
            return None
        _, conf, trace = evidence
        wanted = len(self._wanted(text))
        for t in trace:
            m = rsn.CAUSAL.search(rsn.clean(t["text"]))
            if m and len(t["matched"]) >= 0.5 * wanted:
                marker, reason = m.group(1).lower(), m.group(2).rstrip(".")
                lead = ("The reason is that " if marker in ("because", "since", "so that")
                        else f"It happens {marker} ")
                if marker == "so that":
                    lead = "It's so that "
                parts = [lead + self._lower(reason) + "."]
                parts += [self._rephrase(o, _CONNECTORS[n % 3]) for n, o in enumerate(trace) if o is not t][:2]
                steps = [{"kind": "inference", "source": "my reasoning",
                          "text": f"Found a cause ('{marker}') in the evidence below"}] + trace
                return " ".join(parts), conf, steps
        parts = [self._rephrase(t, _CONNECTORS[n % 3] if n else "") for n, t in enumerate(trace)]
        return "I'm not sure of the exact reason, but here's what I know: " + " ".join(parts), conf * 0.8, trace

    def _rephrase(self, step: dict, prefix: str = "") -> str:
        """Tidy an evidence sentence and say it from the model's point of view."""
        text = rsn.clean(step["text"], keep_short=True)
        if not step["source"].startswith("http"):
            text = rsn.flip_person(text)
        if prefix:
            text = rsn.drop_sequence_word(text)
        return rsn.sentence(prefix + (self._lower(text) if prefix else text))

    def _names(self) -> set[str]:
        if self._names_cache[0] != len(self.knowledge):
            self._names_cache = (len(self.knowledge), rsn.find_names(k["text"] for k in self.knowledge))
        return self._names_cache[1]

    def _lower(self, text: str) -> str:
        return rsn.lower_first(text, self._names())

    def _corpus(self, limit: int | None = None) -> list[list[str]]:
        """Everything I've studied as token lists (the `limit` strongest sentences if given)."""
        corpus = [tokenize(m["prompt"]) + tokenize(m["response"]) for m in self.memories]
        entries = self.knowledge
        if limit and len(entries) > limit:
            entries = sorted(entries, key=lambda k: -k.get("s", 0.8))[:limit]
        return corpus + [tokenize(k["text"]) for k in entries]

    def retrain(self, epochs: int = 5, limit: int | None = None) -> float | None:
        """Give the neural network extra practice on what it knows."""
        return self.neural.train(self._corpus(limit), epochs=epochs, new=False, min_pairs=5000)

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
    def parameter_counts(self) -> dict:
        """How many learned numbers each neural part has (the usual 'parameters')."""
        word = int(self.neural.w_in.size + self.neural.w_out.size)
        feedback = int(self.ranker.w1.size + self.ranker.b1.size + self.ranker.w2.size + 1)
        writer = int(self.writer.n_params) if self.writer else 0
        return {"word network": word, "feedback network": feedback,
                "transformer writer": writer, "total": word + feedback + writer}

    def stats(self) -> dict:
        counts = self.parameter_counts()
        return {
            "parameters (total)": (f"{counts['total']:,} = word network {counts['word network']:,}"
                                   f" + feedback network {counts['feedback network']:,}"
                                   f" + transformer writer {counts['transformer writer']:,}"),
            **self._stats()}

    def _stats(self) -> dict:
        return {
            "memories": len(self.memories),
            "knowledge sentences": len(self.knowledge),
            "knowledge sources": len({k["source"] for k in self.knowledge}),
            "facts": len(self.facts),
            "shelf sentences": len(self.shelf),
            "open questions": len(self.gaps),
            "neural vocabulary": len(self.neural),
            "neural size (dimensions)": self.neural.dim,
            "feedback examples": self.ranker.examples,
            "pretrained words": len(self.pretrained) if self.pretrained is not None else 0,
            "writer": (f"transformer, {self.writer.n_params:,} parameters"
                       + ("" if self.writer_enabled else " (off)")) if self.writer else "templates",
            "conversations logged for training": len(self.answer_log),
            "greetings you taught": len(self.smalltalk_custom),
            "your name": self.user_name() or "(not told yet)",
            "style vocabulary": len({w for c in self.chain.values() for w in c} - {END}),
        }

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "knowledge_threshold": self.knowledge_threshold,
            "order": self.order,
            "memories": self.memories,
            "knowledge": self.knowledge,
            "facts": self.facts,
            "ranker": self.ranker.to_dict(),
            "study": {"day": self.day, "last_sleep": self.last_sleep, "shelf": self.shelf,
                      "gaps": self.gaps, "exam_log": self.exam_log,
                      "interests": dict(self.interests.most_common(300))},
            "writer": {"log": self.answer_log, "enabled": self.writer_enabled},
            "smalltalk": {"custom": self.smalltalk_custom, "weights": self.smalltalk_weights},
            "chain": {k: dict(v) for k, v in self.chain.items()},
            "vectors": {"mode": self.vectors_mode},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LearningModel":
        model = cls(threshold=data.get("threshold", 0.35),
                    knowledge_threshold=data.get("knowledge_threshold", 0.15),
                    order=data.get("order", 2))
        model.memories = data.get("memories", [])
        model.knowledge = data.get("knowledge", [])
        study = data.get("study", {})
        model.shelf = study.get("shelf", [])
        model._known = {k["text"]: k for k in model.knowledge + model.shelf}
        model.day, model.last_sleep = study.get("day", 0), study.get("last_sleep")
        model.gaps, model.exam_log = study.get("gaps", []), study.get("exam_log", [])
        model.interests = Counter(study.get("interests", {}))
        writer = data.get("writer", {})
        model.answer_log = writer.get("log", [])
        model.writer_enabled = writer.get("enabled", True)
        talk = data.get("smalltalk", {})
        model.smalltalk_custom = talk.get("custom", [])
        model.smalltalk_weights = talk.get("weights", {})
        model.vectors_mode = data.get("vectors", {}).get("mode")  # None: decided by load()
        model.facts = data.get("facts", [])
        model.ranker = FeedbackRanker.from_dict(data.get("ranker"))
        if not model.facts and model.knowledge:  # brain from before facts existed
            for k in model.knowledge:
                fact = rsn.extract_fact(k["text"])
                if fact:
                    model.facts.append({**fact, "source": k["source"], "text": k["text"]})
        for k, v in data.get("chain", {}).items():
            model.chain[k] = Counter(v)
        return model

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)
        os.replace(tmp, path)  # atomic: never leaves a half-written brain
        self.neural.save(neural_path(path))
        if (self.pretrained is not None and self.vectors_mode == "custom"
                and (self._vectors_changed or not os.path.exists(vectors_path(path)))):
            self.pretrained.save(vectors_path(path))
            self._vectors_changed = False
        if self.writer is not None and (self._writer_changed or not os.path.exists(writer_path(path))):
            self.writer.save(writer_path(path))
            self._writer_changed = False

    @classmethod
    def load(cls, path: str) -> "LearningModel":
        """Open a saved brain (or start a new one). Word vectors are on unless you turned them off."""
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                model = cls.from_dict(json.load(f))
            model.neural = WordEmbeddings.load(neural_path(path))
            if os.path.exists(writer_path(path)):
                from .transformer import TinyTransformer
                model.writer = TinyTransformer.load(writer_path(path))
            if not len(model.neural):  # brain from before the network existed
                model.neural.train([tokenize(m["prompt"]) + tokenize(m["response"])
                                    for m in model.memories]
                                   + [tokenize(k["text"]) for k in model.knowledge], epochs=3)
        else:
            model = cls()
        mine = PretrainedVectors.load(vectors_path(path))
        if model.vectors_mode is None:  # a brain saved before this setting existed
            model.vectors_mode = "custom" if mine is not None else "builtin"
        if model.vectors_mode == "custom" and mine is not None:
            model.pretrained = mine
        elif model.vectors_mode != "off":
            model.load_builtin_vectors()
        return model
