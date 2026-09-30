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
from .text import TfidfIndex, keywords, looks_like_question, split_sentences, tokenize
from .vectors import PretrainedVectors

_WORD_RE = re.compile(r"\S+")

START, END = "<s>", "</s>"


def neural_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".neural.npz"


def vectors_path(path: str) -> str:
    return os.path.splitext(path)[0] + ".vectors.npz"


# Words too general to connect two facts ("values", "things"...).
_GENERIC = {"value", "thing", "way", "number", "part", "type", "kind", "example", "item", "time",
            "use", "lot", "set", "case", "point", "form", "end", "result", "other", "same",
            "system", "process", "method", "area", "level", "group", "field"}
# "Here is an example:" - introduces something that isn't there (code, a table).
_INTRO = re.compile(r"\b(example|following|below|follows|like this|shown)\b[^.!?]*:\s*$", re.I)
def _bare(phrase: str) -> str:
    """'the Rex' -> 'rex': a phrase without its determiners, for exact matching."""
    return " ".join(w for w in tokenize(phrase) if w not in {"a", "an", "the"})


_CONNECTORS = ("Also, ", "On top of that, ", "In addition, ")
_STEPS = ("First, ", "Then, ", "After that, ", "Finally, ")


class LearningModel:
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
        self._known = set()
        # context tuple (as "a b" string key) -> Counter of next words
        self.chain: dict[str, Counter] = defaultdict(Counter)
        self.neural = WordEmbeddings(seed=seed or 0)
        self.ranker = FeedbackRanker(seed=seed or 0)
        self.pretrained: PretrainedVectors | None = None
        self.fetch = web.http_get  # swap out in tests to avoid the network
        self.notify = None  # optional callback for progress messages
        self._rng = random.Random(seed)
        self._tfidf_cache = None   # (size, TfidfIndex, token lists)
        self._vector_cache = None  # (size, neural version, matrix)
        self._pretrained_cache = None
        self._vectors_changed = False
        self._expand_cache: dict = {}
        self._names_cache = (-1, set())
        self.last_match: int | None = None
        self.last_query: str | None = None
        self.last_reply: str | None = None
        self.last_source: str | None = None  # "memory" | "knowledge" | "noted" | None
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
        meaning = (vecs[:-1] @ vecs[-1]) * self.neural.maturity
        if self.pretrained is not None:
            pre = self.pretrained.sentence_vectors(prompts + [query])
            meaning = np.maximum(meaning, pre[:-1] @ pre[-1])
        sims = self._similarity(tfidf, meaning)
        trust = [0.5 + 0.5 * math.tanh(m["weight"]) for m in self.memories]
        scored = [(float(s * w), i) for i, (s, w) in enumerate(zip(sims, trust))]
        scored.sort(reverse=True)
        return scored

    def respond(self, text: str, use_web: bool = False, notify=None) -> tuple[str | None, float]:
        """Reply to `text`. Returns (response or None if unsure, confidence).

        Tries taught memories first. Statements about facts are remembered.
        Questions are answered by reasoning over your data and anything it
        has read, and (if `use_web`) by looking the question up on Wikipedia.
        """
        notify = notify or self.notify
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
            self.last_trace = [{"kind": "memory", "text": f"{mem['prompt']} -> {mem['response']}",
                                "source": "taught by you", "score": best,
                                "matched": sorted(set(keywords(text)) & set(keywords(mem["prompt"])))}]
        elif not looks_like_question(text):
            noted = self.note(text)
            if noted:
                self.last_reply = "Got it: " + " ".join(rsn.state(f) for f in noted)
                self.last_source, best = "noted", 1.0
                self.last_trace = [{"kind": "fact", "text": rsn.state(f), "source": "you said"}
                                   for f in noted]
        else:
            answer = self.explain(text)
            if answer is None and use_web:
                if notify:
                    notify("Searching the web...")
                if self.search_web(text):
                    answer = self.explain(text)
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
            evidence = [t["features"] for t in self.last_trace if "features" in t]
            if evidence:  # teach the feedback network what good evidence looks like
                self.ranker.train(evidence, [1.0 if good else 0.0] * len(evidence))
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
    def add_document(self, text: str, source: str, topic: str | None = None) -> int:
        """Learn from a block of text. Returns the number of new sentences.

        `topic` (e.g. an article title) lets "It is ..." sentences become facts.
        """
        new, last_subject = [], topic
        for pos, s in enumerate(split_sentences(text, min_words=2 if source == "you said" else 3)):
            fact = rsn.extract_fact(s, topic=last_subject)
            if fact:
                last_subject = topic or fact["subj"]
            if s in self._known:
                continue
            self._known.add(s)
            self.knowledge.append({"text": s, "source": source, "pos": pos})
            if fact:
                self.facts.append({**fact, "source": source, "text": s})
            new.append(s)
        if new:
            self.neural.train([tokenize(s) for s in new], epochs=5, min_pairs=3000)
            self._maybe_grow()
        return len(new)

    def note(self, text: str) -> list[dict]:
        """Remember facts you state in chat ("My sister lives in Delhi")."""
        before = len(self.facts)
        if not any(rsn.extract_fact(s) for s in split_sentences(text, min_words=2)):
            return []
        self.add_document(text, "you said")
        return self.facts[before:]

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
        self._pretrained_cache, self._expand_cache = None, {}
        self._vectors_changed = True
        return len(self.pretrained)

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
        for title, text, url in web.wikipedia(terms, get=self.fetch):
            added += self.add_document(text, url, topic=title)
        return added

    def _knowledge_index(self):
        size = len(self.knowledge)
        if self._tfidf_cache is None or self._tfidf_cache[0] != size:
            toks = [tokenize(k["text"]) for k in self.knowledge]
            stems = [{stem(t) for t in tk} for tk in toks]
            self._tfidf_cache = (size, TfidfIndex(toks), toks, stems)
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
        if pre is not None:
            scores = np.maximum(scores, pre @ self.pretrained.sentence_vectors([query])[0])
        return scores

    def _expand(self, word: str) -> set[str]:
        """The word plus words that mean nearly the same (from the networks)."""
        if word not in self._expand_cache:
            out = {stem(word)}
            if self.pretrained is not None:
                out |= {stem(w) for w, s in self.pretrained.similar(word, 15) if s >= 0.6}
            if len(self.neural) >= 5000:
                out |= {stem(w) for w, s in self.neural.similar(word, 10) if s >= 0.8}
            self._expand_cache[word] = out
        return self._expand_cache[word]

    def _wanted(self, text: str) -> dict[str, set[str]]:
        words = keywords(text) or tokenize(text)
        return {stem(w): self._expand(w) for w in words}

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
        tfidf = np.array(index.scores(query))
        meaning = self._meaning(query, own, pre)
        base = self._similarity(tfidf, meaning)
        topic = "|".join(map(re.escape, wanted))
        definition = re.compile(r"\b(%s)\w*\s+(is|are|was|were|refers|means|consists)\b" % topic, re.I)
        about = re.compile(r"^(the |a |an )?(\w+ )?(%s)\w*\b" % topic, re.I)  # topic is the subject

        cands = []
        for i in (int(i) for i in np.argsort(-base)[:60]):
            covered = {w for w, alts in wanted.items() if alts & stems[i]}
            if not covered:
                continue
            k = self.knowledge[i]
            web_source = k["source"].startswith("http")
            feats = [tfidf[i], meaning[i], len(covered) / len(wanted),
                     float(k.get("pos", 99) < 2), float(bool(definition.search(k["text"]))),
                     min(1.0, len(toks[i]) / 40), float(not web_source), float(web_source)]
            score = (base[i] + 0.1 * feats[3] + 0.1 * feats[4] + (bonus(i, feats[2]) if bonus else 0.0)
                     + 0.1 * bool(about.search(k["text"])) - 0.3 * bool(_INTRO.search(k["text"])))
            cands.append({"i": i, "score": float(score), "covered": covered, "features": feats})
        if not cands:
            return None
        for c, adj in zip(cands, self.ranker.adjust([c["features"] for c in cands])):
            c["score"] += float(adj)

        cands.sort(key=lambda c: -c["score"])
        top = cands[0]["score"]
        chosen, covered = [], set()
        while cands and len(chosen) < max_sentences:
            c = max(cands, key=lambda c: c["score"] + 0.15 * len(c["covered"] - covered))
            cands.remove(c)
            new_terms = c["covered"] - covered
            if chosen and (not new_terms or c["score"] < 0.5 * top):
                continue
            chosen.append(c)
            covered |= new_terms

        coverage = len(covered) / len(wanted)
        confidence = float(min(1.0, top)) * (0.4 + 0.6 * coverage)
        # The evidence has to mention at least half of what you asked about.
        if coverage < 0.5 or confidence < self.knowledge_threshold:
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
        cands = []
        for f in self.facts:
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
        chosen, covered = [], set()
        for score, cov, f in cands:
            if len(chosen) >= 4:
                break
            if not chosen or (cov - covered and score >= 0.5 * cands[0][0]):
                chosen.append(f)
                covered |= cov
        coverage = len(covered) / len(wanted)
        return (chosen, coverage) if coverage >= 0.5 else ([], 0.0)

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
                            [{"kind": "fact", "text": rsn.state(f), "source": f["source"]}])
            path = rsn.isa_chain(self.facts, a, b)
            if path:
                links = [rsn.state(f).rstrip(".") for f in path]
                steps = [{"kind": "fact", "text": rsn.state(f), "source": f["source"]} for f in path]
                if len(path) == 1:
                    return f"Yes. {links[0]}.", 0.9, steps
                chain = ", and ".join([links[0]] + [self._lower(l) for l in links[1:]])
                steps.append({"kind": "inference", "source": "my reasoning",
                              "text": f"Chained {len(path)} facts: {' -> '.join(rsn.head(f['subj']) for f in path)} -> {b}"})
                return f"Yes. {chain}, so {subj} {words[0]} {obj}.", 0.8, steps
        return None

    def explain(self, text: str):
        """Answer a question in the model's own words, with the reasoning steps.

        Returns (answer, confidence, steps) or None.
        """
        kind = rsn.question_kind(text)
        if kind == "yesno":
            proof = self._prove(text)
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
        if not facts and evidence is None:
            return None
        parts, steps, used = [], [], set()
        if facts:
            first = facts[0]
            parts.append(rsn.state(first))
            steps.append({"kind": "fact", "text": rsn.state(first), "source": first["source"],
                          "sentence": first["text"]})
            used.add(first["text"])
            subject = rsn.head(first["subj"])
            for f in facts[1:]:
                if f["text"] in used:
                    continue
                same = rsn.head(f["subj"]) == subject
                parts.append(rsn.state(f, subject=rsn.pronoun(first), also=True) if same
                             else _CONNECTORS[len(parts) % 3] + self._lower(rsn.state(f)))
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
            extra = [t for t in trace if t["text"] not in used][:max(0, 3 - len(parts))]
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
        following = sorted(({"kind": "evidence", "text": k["text"], "source": k["source"],
                             "pos": k.get("pos", 0)}
                            for k in self.knowledge
                            if k["source"] == start["source"] and 0 < k.get("pos", 0) - start["pos"] <= 4),
                           key=lambda t: t["pos"])
        if not any(rsn.SEQUENCE_MARKER.search(t["text"]) for t in following):
            following = []  # the next sentences aren't steps of a process
        steps = [start] + following
        if len(steps) == 1:
            parts = [self._rephrase(t, _CONNECTORS[n % 3] if n else "") for n, t in enumerate(trace)]
            return " ".join(parts), conf, trace
        parts = [self._rephrase(t, "Finally, " if n == len(steps) - 1 else _STEPS[min(n, 2)])
                 for n, t in enumerate(steps)]
        return "Here's how: " + " ".join(parts), conf, trace + following

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
        text = rsn.clean(step["text"])
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

    def _corpus(self) -> list[list[str]]:
        corpus = [tokenize(m["prompt"]) + tokenize(m["response"]) for m in self.memories]
        return corpus + [tokenize(k["text"]) for k in self.knowledge]

    def retrain(self, epochs: int = 5) -> float | None:
        """Give the neural network extra practice on everything it knows."""
        return self.neural.train(self._corpus(), epochs=epochs, new=False, min_pairs=5000)

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
            "facts": len(self.facts),
            "neural vocabulary": len(self.neural),
            "neural size (dimensions)": self.neural.dim,
            "feedback examples": self.ranker.examples,
            "pretrained words": len(self.pretrained) if self.pretrained is not None else 0,
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
        if self.pretrained is not None and (self._vectors_changed or not os.path.exists(vectors_path(path))):
            self.pretrained.save(vectors_path(path))
            self._vectors_changed = False

    @classmethod
    def load(cls, path: str) -> "LearningModel":
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as f:
            model = cls.from_dict(json.load(f))
        model.neural = WordEmbeddings.load(neural_path(path))
        model.pretrained = PretrainedVectors.load(vectors_path(path))
        if not len(model.neural):  # brain from before the network existed
            model.neural.train([tokenize(m["prompt"]) + tokenize(m["response"])
                                for m in model.memories]
                               + [tokenize(k["text"]) for k in model.knowledge], epochs=3)
        return model
