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

Everything is saved (a JSON "brain" plus the network's weights in .npz).
"""

from __future__ import annotations

import json
import math
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np

from . import web
from .neural import WordEmbeddings
from .text import TfidfIndex, keywords, looks_like_question, split_sentences, tokenize

_WORD_RE = re.compile(r"\S+")

START, END = "<s>", "</s>"


def neural_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".neural.npz"


class LearningModel:
    def __init__(self, threshold: float = 0.35, knowledge_threshold: float = 0.15,
                 order: int = 2, seed: int | None = None):
        self.threshold = threshold
        self.knowledge_threshold = knowledge_threshold
        self.order = order
        self.memories: list[dict] = []   # {"prompt", "response", "weight"}
        self.knowledge: list[dict] = []  # {"text", "source"}
        self._known = set()
        # context tuple (as "a b" string key) -> Counter of next words
        self.chain: dict[str, Counter] = defaultdict(Counter)
        self.neural = WordEmbeddings(seed=seed or 0)
        self.fetch = web.http_get  # swap out in tests to avoid the network
        self._rng = random.Random(seed)
        self._tfidf_cache = None   # (size, TfidfIndex, token lists)
        self._vector_cache = None  # (size, neural version, matrix)
        self.last_match: int | None = None
        self.last_query: str | None = None
        self.last_reply: str | None = None
        self.last_source: str | None = None  # "memory" | "knowledge" | None
        self.last_trace: list[dict] = []

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
        self.neural.train([tokenize(prompt) + tokenize(response)], epochs=10, min_pairs=1000)

    def _similarity(self, tfidf: np.ndarray, neural: np.ndarray) -> np.ndarray:
        # The network can raise a match (different words, same meaning) but
        # never lowers a match on shared words.
        return np.maximum(tfidf, 0.5 * tfidf + 0.5 * np.clip(neural, 0, 1))

    def rank(self, text: str) -> list[tuple[float, int]]:
        """Return (score, memory_index) pairs for taught memories, best first."""
        if not self.memories:
            return []
        prompts = [tokenize(m["prompt"]) for m in self.memories]
        query = tokenize(text)
        tfidf = np.array(TfidfIndex(prompts).scores(query))
        vecs = self.neural.sentence_vectors(prompts + [query])
        sims = self._similarity(tfidf, vecs[:-1] @ vecs[-1])
        trust = [0.5 + 0.5 * math.tanh(m["weight"]) for m in self.memories]
        scored = [(float(s * w), i) for i, (s, w) in enumerate(zip(sims, trust))]
        scored.sort(reverse=True)
        return scored

    def respond(self, text: str, use_web: bool = False, notify=None) -> tuple[str | None, float]:
        """Reply to `text`. Returns (response or None if unsure, confidence).

        Tries taught memories first, then your data and anything it has read,
        then (if `use_web`) looks the question up on Wikipedia.
        """
        self.observe(text)
        self.last_query, self.last_reply, self.last_source = text, None, None
        self.last_match, self.last_trace = None, []

        ranked = self.rank(text)
        best = ranked[0][0] if ranked else 0.0
        if ranked and best >= self.threshold:
            idx = ranked[0][1]
            mem = self.memories[idx]
            self.last_match, self.last_source = idx, "memory"
            self.last_reply = mem["response"]
            self.last_trace = [{"text": f"{mem['prompt']} -> {mem['response']}",
                                "source": "taught by you", "score": best,
                                "matched": sorted(set(keywords(text)) & set(keywords(mem["prompt"])))}]
        else:
            answer = self.answer_from_knowledge(text)
            if answer is None and use_web and looks_like_question(text):
                if notify:
                    notify("Searching the web...")
                if self.search_web(text):
                    answer = self.answer_from_knowledge(text)
            if answer is not None:
                self.last_reply, best, self.last_trace = answer
                self.last_source = "knowledge"

        self.neural.train([tokenize(text)], epochs=1)
        return self.last_reply, best

    def feedback(self, good: bool) -> bool:
        """Reward or penalise the last response. Returns False if nothing to rate."""
        if self.last_source == "memory" and self.last_match is not None:
            self.memories[self.last_match]["weight"] += 0.5 if good else -1.0
            return True
        if self.last_source == "knowledge":
            if good:  # a good researched answer becomes a trusted memory
                self.learn(self.last_query, self.last_reply)
                self.last_match, self.last_source = len(self.memories) - 1, "memory"
            return True
        return False

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

    # --------------------------------------------------------------- knowledge
    def add_document(self, text: str, source: str) -> int:
        """Learn from a block of text. Returns the number of new sentences."""
        new = []
        for pos, s in enumerate(split_sentences(text)):
            if s not in self._known:
                self._known.add(s)
                self.knowledge.append({"text": s, "source": source, "pos": pos})
                new.append(s)
        if new:
            self.neural.train([tokenize(s) for s in new], epochs=5, min_pairs=3000)
        return len(new)

    def read(self, path_or_url: str) -> int:
        """Learn from a local file or a web page."""
        if re.match(r"https?://", path_or_url):
            return self.add_document(web.fetch_page(path_or_url, get=self.fetch), path_or_url)
        with open(os.path.expanduser(path_or_url), encoding="utf-8", errors="replace") as f:
            return self.add_document(f.read(), os.path.basename(path_or_url))

    def search_web(self, query: str) -> int:
        """Look `query` up on Wikipedia and learn from the results."""
        terms = " ".join(keywords(query)) or query
        added = 0
        for _title, text, url in web.wikipedia(terms, get=self.fetch):
            added += self.add_document(text, url)
        return added

    def _knowledge_index(self):
        size = len(self.knowledge)
        if self._tfidf_cache is None or self._tfidf_cache[0] != size:
            toks = [tokenize(k["text"]) for k in self.knowledge]
            self._tfidf_cache = (size, TfidfIndex(toks), toks)
        _, index, toks = self._tfidf_cache
        key = (size, self.neural.version)
        if self._vector_cache is None or self._vector_cache[0] != key:
            self._vector_cache = (key, self.neural.sentence_vectors(toks))
        return index, toks, self._vector_cache[1]

    def answer_from_knowledge(self, text: str, max_sentences: int = 3):
        """Build an answer from known sentences.

        Picks the best-matching sentence, then adds sentences that cover parts
        of the question the answer doesn't cover yet (simple multi-step
        reasoning over evidence). Returns (answer, confidence, trace) or None.
        """
        if not self.knowledge:
            return None
        query = keywords(text) or tokenize(text)
        wanted = set(query)
        if not wanted:
            return None
        index, toks, vecs = self._knowledge_index()
        qvec = self.neural.sentence_vectors([query])[0]
        scores = self._similarity(np.array(index.scores(query)), vecs @ qvec)
        candidates = [int(i) for i in np.argsort(-scores)[:30] if wanted & set(toks[int(i)])]
        if not candidates:
            return None
        # Prefer sentences that define things: a document's opening lines and
        # "<keyword> is/are/was ..." statements.
        for i in candidates:
            if self.knowledge[i].get("pos", 99) < 2:
                scores[i] += 0.1
            if re.search(r"\b(%s)\w*\s+(is|are|was|were|refers)\b" % "|".join(map(re.escape, wanted)),
                         self.knowledge[i]["text"], re.I):
                scores[i] += 0.1

        candidates.sort(key=lambda i: -scores[i])
        top = scores[candidates[0]]
        chosen, covered = [], set()
        while candidates and len(chosen) < max_sentences:
            gain = lambda i: scores[i] + 0.15 * len((wanted & set(toks[i])) - covered)
            i = max(candidates, key=gain)
            candidates.remove(i)
            new_terms = (wanted & set(toks[i])) - covered
            if chosen and (not new_terms or scores[i] < 0.5 * top):
                continue
            chosen.append(i)
            covered |= new_terms

        coverage = len(covered) / len(wanted)
        confidence = float(min(1.0, top)) * (0.4 + 0.6 * coverage)
        # The evidence has to mention at least half of what you asked about.
        if coverage < 0.5 or confidence < self.knowledge_threshold:
            return None
        trace = [{"text": self.knowledge[i]["text"], "source": self.knowledge[i]["source"],
                  "score": float(scores[i]), "matched": sorted(wanted & set(toks[i]))}
                 for i in chosen]
        return " ".join(t["text"] for t in trace), confidence, trace

    def retrain(self, epochs: int = 5) -> float | None:
        """Give the neural network extra practice on everything it knows."""
        corpus = [tokenize(m["prompt"]) + tokenize(m["response"]) for m in self.memories]
        corpus += [tokenize(k["text"]) for k in self.knowledge]
        return self.neural.train(corpus, epochs=epochs, new=False, min_pairs=5000)

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
            "knowledge sentences": len(self.knowledge),
            "knowledge sources": len({k["source"] for k in self.knowledge}),
            "neural vocabulary": len(self.neural),
            "style vocabulary": len({w for c in self.chain.values() for w in c} - {END}),
        }

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "knowledge_threshold": self.knowledge_threshold,
            "order": self.order,
            "memories": self.memories,
            "knowledge": self.knowledge,
            "chain": {k: dict(v) for k, v in self.chain.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LearningModel":
        model = cls(threshold=data.get("threshold", 0.35),
                    knowledge_threshold=data.get("knowledge_threshold", 0.15),
                    order=data.get("order", 2))
        model.memories = data.get("memories", [])
        model.knowledge = data.get("knowledge", [])
        model._known = {k["text"] for k in model.knowledge}
        for k, v in data.get("chain", {}).items():
            model.chain[k] = Counter(v)
        return model

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=1)
        os.replace(tmp, path)  # atomic: never leaves a half-written brain
        self.neural.save(neural_path(path))

    @classmethod
    def load(cls, path: str) -> "LearningModel":
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as f:
            model = cls.from_dict(json.load(f))
        model.neural = WordEmbeddings.load(neural_path(path))
        if not len(model.neural):  # brain from before the network existed
            model.neural.train([tokenize(m["prompt"]) + tokenize(m["response"])
                                for m in model.memories]
                               + [tokenize(k["text"]) for k in model.knowledge], epochs=3)
        return model
