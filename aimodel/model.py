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
from .smalltalk import SmallTalkMixin
from .study import GENERIC as _GENERIC, WEAK_WORDS, StudyMixin, bare as _bare
from .text import TfidfIndex, keywords, looks_like_question, split_sentences, tokenize
from .understand import (check_claim, common_question, in_common, members, members_question,
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
        if ranked and best >= self.threshold:
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
        for pos, s in enumerate(split_sentences(text, min_words=2 if source == "you said" else 3)):
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
            added += self.add_document(text, url, topic=title, focus=query)
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

    def _key_word(self, wanted: dict[str, set[str]]) -> str:
        """The most specific word of a question (rarest in what I've read).

        An answer that doesn't cover it isn't about the question at all, e.g.
        photosynthesis text mentioning "work" for "How do vaccines work?".
        """
        index = self._knowledge_index()[0] if self.knowledge else None
        def rarity(w):
            if index is None:
                return 0.0
            return max((index.idf.get(v, 99.0) for v in wanted[w]), default=99.0)
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
        # The evidence has to mention the question's key word and at least
        # half of what you asked about.
        if (coverage < 0.5 or self._key_word(wanted) not in covered
                or confidence < self.knowledge_threshold):
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
        if chosen:
            top = chosen[0]
            # "Tell me about the gorzu" asks about nothing but its subject: say all I know.
            broad = set(wanted) <= {stem(t) for t in tokenize(top["subj"])}
            for score, cov, f in cands:
                if len(chosen) < 4 and f not in chosen and _bare(f["subj"]) == _bare(top["subj"]) and (
                        broad or (score >= 0.9 * cands[0][0] and f["rel"] == top["rel"])):
                    chosen.append(f)  # also lists: "I like tea" and "I like coffee"
        coverage = len(covered) / len(wanted)
        if coverage < 0.5 or self._key_word(wanted) not in covered:
            return [], 0.0
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
            proof = self._prove_either(text) or self._prove(text) or self._check_claim(text)
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
        if self.writer is not None and (self._writer_changed or not os.path.exists(writer_path(path))):
            self.writer.save(writer_path(path))
            self._writer_changed = False

    @classmethod
    def load(cls, path: str) -> "LearningModel":
        if not os.path.exists(path):
            return cls()
        with open(path, encoding="utf-8") as f:
            model = cls.from_dict(json.load(f))
        model.neural = WordEmbeddings.load(neural_path(path))
        model.pretrained = PretrainedVectors.load(vectors_path(path))
        if os.path.exists(writer_path(path)):
            from .transformer import TinyTransformer
            model.writer = TinyTransformer.load(writer_path(path))
        if not len(model.neural):  # brain from before the network existed
            model.neural.train([tokenize(m["prompt"]) + tokenize(m["response"])
                                for m in model.memories]
                               + [tokenize(k["text"]) for k in model.knowledge], epochs=3)
        return model
