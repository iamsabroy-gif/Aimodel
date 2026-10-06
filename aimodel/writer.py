"""The writer: turning the evidence behind an answer into words.

The reasoning (memory, facts, study) decides *what* is true; the writer only
decides *how to say it*. Answers start out written with built-in sentence
templates. Once you've trained the tiny transformer on Kaggle (train_writer.py)
it writes them instead, but every draft is checked first: each meaningful word
must appear in the question or the evidence, and it must say (almost) all that
the template answer says. Otherwise the template answer is used. A small model
can't quietly make things up, or leave things out, this way.

Training data comes from the model itself (see `build_examples`):
your real questions and its answers, your /good ratings and /bad corrections
(which count the most), practice questions written from everything it knows,
and the replies you taught it.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

from . import reasoning as rsn
from .reasoning import stem
from .text import keywords, split_sentences, tokenize

# Words the writer may use to connect sentences without them being in the evidence.
GLUE = {"also", "addition", "top", "first", "then", "after", "finally", "next", "reason", "because",
        "yes", "probably", "here", "here's", "involves", "sure", "exact", "found", "prove",
        "can't", "cannot", "short", "means", "that's", "which", "so", "while", "both",
        "together", "other", "words", "summary", "overall", "however", "too", "well", "far"}
MAX_LOG = 2000


def writer_evidence(trace: list[dict]) -> list[str]:
    """The sentences an answer was built from, as the writer sees them."""
    out = []
    for step in trace:
        if step.get("kind") not in ("fact", "evidence"):
            continue
        # A fact is a trimmed copy of a sentence; the writer should see the whole sentence.
        text = rsn.clean(step.get("sentence") or step["text"])
        if not step["source"].startswith("http"):
            text = rsn.flip_person(text)  # your notes say "I", answers to you say "you"
        text = rsn.sentence(text)
        if text and text not in out:
            out.append(text)
    return out


def grounded(answer: str, sources: list[str]) -> tuple[bool, list[str]]:
    """Is every meaningful word of `answer` found in `sources`? Returns (ok, unsupported words)."""
    words = keywords(answer)
    if len(answer.strip()) < 3 or not words:
        return False, ["(nothing said)"]
    allowed = {stem(t) for s in sources for t in tokenize(s)}
    missing = list(dict.fromkeys(w for w in words if stem(w) not in allowed and w not in GLUE))
    if not any(stem(w) in allowed for w in words):
        return False, missing or ["(nothing from the evidence)"]
    toks = tokenize(answer)
    if any(toks[i] == toks[i + 1] == toks[i + 2] for i in range(len(toks) - 2)):
        return False, ["(repeats itself)"]
    return not missing, missing


_VERDICT = re.compile(r"^\W*(yes|no|probably yes)\b", re.I)
_MAIN_CLAUSE = re.compile(r",?\s+(?:so|which|who|because|since|while)\s+", re.I)


def verdict(text: str) -> str | None:
    """'Yes', 'No' or 'Probably yes' if the answer opens with one."""
    m = _VERDICT.match(text)
    return m.group(1).lower() if m else None


def repeats(text: str) -> bool:
    """A repeated phrase ("in Africa in Africa") or a repeated sentence."""
    toks = tokenize(text)
    for n in range(2, 5):
        if any(toks[i:i + n] == toks[i + n:i + 2 * n] for i in range(len(toks) - 2 * n + 1)):
            return True
    # the same sentence again, even reworded ("You like it. You also like it.")
    sentences = [frozenset(_words(x)) or frozenset(tokenize(x)) for x in split_sentences(text, min_words=2)]
    return len(sentences) != len(set(sentences))


def _clauses(draft: str):
    """The statements to verify: each sentence's main clause and the clauses joined by 'and'.
    What follows 'so' or 'which' is a conclusion or an aside, not a claim to check."""
    for sentence in split_sentences(draft, min_words=2):
        main = _MAIN_CLAUSE.split(rsn.clean(sentence), 1)[0]
        yield from re.split(r",\s+(?:and|but)\s+", main)


def _words(text: str) -> set[str]:
    return {stem(w) for w in keywords(text)}


def _same_fact(a: dict, b: dict) -> bool:
    be = ("be", "be called")
    return (rsn.head(a["subj"]) == rsn.head(b["subj"]) and a["neg"] == b["neg"]
            and (a["rel"] == b["rel"] or (a["rel"] in be and b["rel"] in be))
            and _words(a["obj"]) <= _words(b["obj"]))


def supported(draft: str, evidence: list[str]) -> tuple[bool, list[str]]:
    """Is every statement in `draft` backed by a statement in the evidence?

    Statements are read as facts (subject, verb, object, yes/no), so "the sheep eats
    grass" does not pass for "the grass eats sheep" and "whales are not fish" does not
    pass for "whales are fish", even though they use the same words. A statement about
    something the evidence never states as a fact is left to the word check.
    """
    known = [f for e in evidence for part in split_sentences(e, min_words=2)
             for f in [rsn.extract_fact(rsn.clean(part))] if f]
    wrong, subject = [], None
    for part in _clauses(draft):
        fact = rsn.extract_fact(part, topic=subject)
        if not fact:
            continue
        subject = fact["subj"]
        who = rsn.head(fact["subj"])
        about = [k for k in known if rsn.head(k["subj"]) == who]
        # Backed if the evidence says it. Not backed if the evidence says something else about
        # this subject, or only mentions it as the thing acted on ("the sheep eats GRASS" is
        # no evidence that grass eats anything).
        acted_on = who in {w for k in known for w in _words(k["obj"])}
        if (about and not any(_same_fact(fact, k) for k in about)) or (not about and acted_on):
            wrong.append(part.strip())
    return not wrong, wrong


def check_draft(draft: str, evidence: list[str], question: str = "",
                reference: str | None = None) -> tuple[bool, list[str]]:
    """Should this draft be shown? Returns (ok, problems). Every check must pass:
    only evidence words, no repeats, every statement backed by the evidence, and (given
    the template answer as `reference`) the same yes/no and nothing important left out."""
    problems = []
    ok, missing = grounded(draft, evidence + [question])
    if not ok:
        problems.append("words not in the evidence: " + ", ".join(missing[:5]))
    if repeats(draft):
        problems.append("it repeats itself")
    claims_ok, wrong = supported(draft, evidence)
    if not claims_ok:
        problems.append("a statement the evidence doesn't support: " + "; ".join(wrong[:2]))
    if reference is not None:
        if verdict(draft) != verdict(reference):
            problems.append(f"it says {verdict(draft) or 'nothing'} where the answer is "
                            f"{verdict(reference) or 'not a yes/no'}")
        whole, left_out = complete(draft, reference)
        if not whole:
            problems.append("it left things out: " + ", ".join(left_out[:5]))
    return not problems, problems


def complete(draft: str, reference: str, share: float = 0.8) -> tuple[bool, list[str]]:
    """Does `draft` say (almost) everything `reference` says? Returns (ok, left out words)."""
    said = {stem(w) for w in keywords(draft)}
    needed = list(dict.fromkeys(w for w in keywords(reference) if w not in GLUE))
    left_out = [w for w in needed if stem(w) not in said]
    return len(left_out) <= (1 - share) * len(needed), left_out


# ----------------------------------------------------------------- training data
def build_examples(model, limit: int = 5000, extra_questions=(), exam_questions: bool = True) -> list[dict]:
    """Question -> evidence -> answer examples to train the writer on.

    `extra_questions` are more practice questions to answer from what it knows
    (used by the starter dataset to cover many ways of asking). With
    `exam_questions=False` only those are used, not the model's own exam questions.
    """
    examples, seen = [], set()

    def add(question, evidence, answer, weight, kind):
        key = (question.lower(), answer)
        if question and answer and key not in seen:
            seen.add(key)
            examples.append({"question": question, "evidence": evidence, "answer": answer,
                             "weight": weight, "kind": kind})

    # 1. Real conversations. Your corrections and good ratings teach the most.
    for e in model.answer_log:
        if e.get("correction"):
            add(e["q"], e["evidence"], e["correction"], 3.0, "your correction")
        elif e.get("rating") == "good":
            add(e["q"], e["evidence"], e["final"], 2.0, "rated good")
        elif e.get("rating") != "bad":
            add(e["q"], e["evidence"], e["template"], 1.0, "asked")

    # 2. Practice questions written from everything it knows (like exam questions),
    #    plus "Tell me about X" for things it knows several facts about.
    questions = []
    if exam_questions:
        questions = [q["q"] for q in model.make_questions(limit) if q["kind"] != "memory"]
        subjects = Counter(rsn.head(f["subj"]) for f in model.facts if not rsn.is_personal(f))
        questions += [f"Tell me about {s}" for s, n in subjects.items() if n >= 2 and s]
    questions += list(extra_questions)
    for q in questions[:limit + len(extra_questions)]:
        answer = model.explain(q)
        if answer is not None:
            add(q, writer_evidence(answer[2]), answer[0], 1.0, "practice")

    # 3. Replies you taught it (no evidence: these are learned by heart).
    for m in model.memories:
        add(m["prompt"], [], m["response"], 2.0, "taught")
    return examples


def write_training_files(folder: str, examples: list[dict], lines: list[str]) -> dict:
    """Write writer_data.jsonl (examples), corpus.txt (text to read) and ABOUT.txt."""
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "writer_data.jsonl"), "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(os.path.join(folder, "corpus.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(folder, "ABOUT.txt"), "w", encoding="utf-8") as f:
        f.write("Training data for the Aimodel writer (a tiny transformer).\n"
                "writer_data.jsonl: question / evidence / answer examples.\n"
                "corpus.txt: everything the model has read, to learn the language.\n"
                "This contains your private notes. On Kaggle, upload it as a PRIVATE dataset.\n"
                "Train with: python -m aimodel.train_writer --data <this folder> --out writer.npz\n")
    return {"folder": folder, "examples": len(examples), "corpus lines": len(lines),
            "by kind": dict(Counter(e["kind"] for e in examples))}


def export_training_data(model, folder: str, extra_questions=(), exam_questions: bool = True) -> dict:
    """Write everything the writer can learn from this model (see `write_training_files`)."""
    examples = build_examples(model, extra_questions=extra_questions, exam_questions=exam_questions)
    lines = [k["text"] for k in model.knowledge + model.shelf]
    lines += [f"{m['prompt']}\n{m['response']}" for m in model.memories]
    return write_training_files(folder, examples, lines)


# ------------------------------------------------------------------- the model side
class WriterMixin:
    """Lets LearningModel use a trained writer and keep a log to train it with."""

    def _init_writer(self) -> None:
        self.writer = None             # a TinyTransformer once you've trained one
        self.writer_enabled = True
        self.answer_log: list[dict] = []
        self._writer_changed = False

    def load_writer(self, path: str):
        from .transformer import TinyTransformer
        self.writer = TinyTransformer.load(path)
        self.writer_enabled = True
        self._writer_changed = True
        return self.writer

    def _write(self, question: str, answer: str, trace: list[dict], learn: bool):
        """Let the writer word the answer, if it stays true to the evidence."""
        evidence = writer_evidence(trace)
        final, by, rejected = answer, "templates", []
        if self.writer is not None and self.writer_enabled and evidence:
            draft = self.writer.write(question, evidence)
            ok, problems = check_draft(draft, evidence, question, reference=answer)
            if ok:
                final, by = rsn.sentence(draft), "transformer"
                note = "Worded by my transformer; every statement checked against the evidence"
            else:
                rejected = problems
                note = (f"My transformer's draft was rejected ({'; '.join(problems)}), "
                        "so I used my template answer")
            trace = trace + [{"kind": "writer", "text": note, "source": "my transformer"}]
        if learn:
            self.answer_log.append({"q": question, "evidence": evidence, "template": answer,
                                    "final": final, "by": by, "rating": None, "rejected": rejected})
            del self.answer_log[:-MAX_LOG]
        return final, trace

    def _rate_last(self, good: bool) -> None:
        if self.answer_log and self.answer_log[-1]["q"] == self.last_query:
            self.answer_log[-1]["rating"] = "good" if good else "bad"

    def _correct_last(self, prompt: str, better: str) -> None:
        if self.answer_log and self.answer_log[-1]["q"] == prompt:
            self.answer_log[-1]["correction"] = better

    def export_training_data(self, folder: str, extra_questions=(), exam_questions: bool = True) -> dict:
        return export_training_data(self, folder, extra_questions, exam_questions)
