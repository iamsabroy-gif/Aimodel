"""The standard scorecard: the same fixed tests, run from time to time.

    python -m aimodel.evaluation --brain brain.json        (or /evaluate in the chat)

Two kinds of metric, and the difference matters:

* **Fixed tests (A-C)** use a small made-up world, so they don't depend on your data and
  mean the same thing on every run. They guard against regressions: after updating the code
  or retraining, none of these should get worse.
    A  Reasoning benchmark  - is it right when it answers, and does it say "I don't know"
                              when it should?
    B  Safety check         - does the check on the writer's drafts catch corrupted answers
                              without rejecting good ones?
    C  Greetings            - does it recognise greetings without mistaking questions for them?
* **Your brain (D-F)** measure the model you have built, so they change as you teach it.
  They show progress: recall of what it knows, memory health, how often you approve of its
  answers, and how the transformer writer is doing.
    D  Knowledge & recall   E  Feedback & memory health   F  Writer   (and speed)

Every run is appended to a history file next to the brain, so you can see the trend.
Nothing here changes the brain: no learning, no strengthening, no web access.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import time
from collections import Counter

from .model import LearningModel
from .reasoning import stem
from .text import tokenize
from .writer import check_draft, grounded, verdict, writer_evidence

# ---------------------------------------------------------------------- A: benchmark
# A made-up world, so nothing can be answered from memory or from the web.
WORLD = {
    "bench_kinds.txt": "Mammals are warm-blooded animals. Birds are animals that have feathers. "
                       "Reptiles are cold-blooded animals. Animals are living things. "
                       "Plants are living things.",
    "bench_animals.txt": "The mimbat is a mammal. The mimbat lives in caves. The mimbat eats beetles. "
                         "The lorpan is a mammal. The lorpan lives in mountains. The lorpan eats grass. "
                         "The quillet is a bird. The quillet lives in marshes. The quillet eats seeds and berries. "
                         "The druxle is a bird. The druxle lives in cliffs. The druxle eats fish. "
                         "The zarnak is a reptile. The zarnak lives in deserts. The zarnak eats lizards. "
                         "The zarnak is not a mammal.",
    "bench_places.txt": "Brelm is the capital of Vostria. Vostria is a country in Norvia. "
                        "Tamsk is the capital of Ulderan. Ulderan is a country in Norvia. "
                        "Pellow is the capital of Kesmar. Kesmar is a country in Zaphiria.",
    "bench_science.txt": "The marsh floods because the river overflows in spring. "
                         "The lantern glows because it burns oil.",
    "bench_howto.txt": "To bake flatbread, mix flour and water. Then knead the dough. After that, roll it "
                       "flat. Finally, cook it on a hot pan.\n"
                       "To tie a knot, make a loop with the rope. Then pass the end through the loop. "
                       "Finally, pull it tight.",
    "bench_me.txt": "My cat is called Pixel. My brother lives in Dunmore. I like origami.",
}

# skill, question, words the answer must contain, and options:
#   no=words it must not contain (a different animal's facts), ordered=in this order,
#   verdict="yes"/"no" for yes/no questions. gold None means: it should decline to answer.
ITEMS = [
    ("lookup", "What is the mimbat?", ["mammal"], {"no": ["bird", "reptile"]}),
    ("lookup", "Where does the lorpan live?", ["mountains"], {"no": ["caves", "deserts"]}),
    ("lookup", "What does the druxle eat?", ["fish"], {"no": ["seeds", "grass"]}),
    ("lookup", "What is the capital of Vostria?", ["brelm"], {"no": ["tamsk", "pellow"]}),
    ("lookup", "Where is Ulderan?", ["norvia"], {"no": ["zaphiria"]}),
    ("lookup", "What is Pellow?", ["capital", "kesmar"], {}),
    ("lookup", "Where does the quillet live?", ["marshes"], {"no": ["cliffs", "caves"]}),
    ("paraphrase", "What is a quillet?", ["bird"], {"no": ["mammal"]}),
    ("paraphrase", "What does a zarnak eat?", ["lizards"], {"no": ["beetles", "fish"]}),
    ("paraphrase", "Tell me where the mimbat lives.", ["caves"], {"no": ["marshes"]}),
    ("list", "What does the quillet eat?", ["seeds", "berries"], {"no": ["fish", "grass"]}),
    ("multi-fact", "Tell me about the mimbat", ["mammal", "caves", "beetles"], {"no": ["quillet"]}),
    ("multi-fact", "Tell me about the quillet", ["bird", "marshes", "seeds"], {"no": ["mimbat"]}),
    ("chain", "Is a mimbat an animal?", None, {"verdict": "yes"}),
    ("chain", "Is a quillet a living thing?", None, {"verdict": "yes"}),
    ("chain", "Is the druxle an animal?", None, {"verdict": "yes"}),
    ("direct yes/no", "Is the quillet a bird?", None, {"verdict": "yes"}),
    ("negation", "Is a zarnak a mammal?", None, {"verdict": "no"}),
    ("cause", "Why does the marsh flood?", ["river", "overflows"], {}),
    ("cause", "Why does the lantern glow?", ["burns", "oil"], {}),
    ("steps", "How do I bake flatbread?", ["mix", "knead", "roll", "cook"], {"ordered": True, "no": ["knot"]}),
    ("steps", "How do I tie a knot?", ["loop", "pass", "pull"], {"ordered": True, "no": ["flatbread"]}),
    ("personal", "What is my cat called?", ["pixel"], {}),
    ("personal", "Where does my brother live?", ["dunmore"], {}),
    ("personal", "What do I like?", ["origami"], {}),
    # harder than it is built for today: these show headroom, so improvement can show up over time
    ("stretch", "Which animal lives in marshes?", ["quillet"], {"no": ["mimbat", "druxle"]}),
    ("stretch", "What does the mimbat eat and where does it live?", ["beetles", "caves"], {}),
    ("stretch", "Where does the lorpan dwell?", ["mountains"], {}),
    ("stretch", "Whre does the lorpan live?", ["mountains"], {}),
    ("stretch", "Which animals are birds?", ["quillet", "druxle"], {"no": ["mimbat"]}),
    ("stretch", "Is the quillet an animal and does it live in marshes?", ["yes", "marshes"], {}),
    ("stretch", "Name all the birds.", ["quillet", "druxle"], {"no": ["mimbat"]}),
    ("stretch", "Is the mimbat a bird or a mammal?", ["mammal"], {"no": ["quillet"]}),
    ("stretch", "Does the lorpan eat meat?", None, {"verdict": "no"}),
    ("stretch", "What do mammals and birds have in common?", ["animals"], {}),
    ("stretch", "How are the mimbat and the quillet alike?", ["animals"], {"no": ["lorpan"]}),
    ("stretch", "Does a quillet have feathers?", None, {"verdict": "yes"}),
    ("stretch", "What is the difference between a mimbat and a quillet?", ["caves", "marshes"], {}),
    # things it was never told: the right answer is "I don't know"
    ("abstain", "What does the glorb eat?", None, {}),
    ("abstain", "Is a glorb an animal?", None, {}),
    ("abstain", "Who invented the telephone?", None, {}),
    ("abstain", "What is the capital of Wexonia?", None, {}),
    ("abstain", "Why is grass purple?", None, {}),
    ("abstain", "How do I fix a bicycle?", None, {}),
    ("abstain", "How do I knit a scarf?", None, {}),
    ("abstain", "What is my favourite colour?", None, {}),
    ("abstain", "Where does my sister live?", None, {}),
    ("abstain", "What is the speed of light?", None, {}),
]


def benchmark_model() -> LearningModel:
    m = LearningModel(seed=0)
    for source, text in WORLD.items():
        m.add_document(text, source)
    return m


def judge(reply: str | None, skill: str, gold, options: dict) -> bool:
    """Right if it declined (for abstain items), or it has the right yes/no, or it contains every
    wanted word (in order if asked) and none of the forbidden ones."""
    if skill == "abstain":
        return reply is None
    if reply is None:
        return False
    if "verdict" in options:
        say = verdict(reply)
        return say in ("yes", "probably yes") if options["verdict"] == "yes" else say == "no"
    heard = [stem(t) for t in tokenize(reply)]
    if any(stem(w) in heard for w in options.get("no", [])):
        return False
    wanted = [stem(w) for w in gold]
    if not all(w in heard for w in wanted):
        return False
    if options.get("ordered"):
        spots = [heard.index(w) for w in wanted]
        return spots == sorted(spots)
    return True


def run_benchmark(model: LearningModel | None = None) -> dict:
    """A: ask the fixed questions. Returns metrics, per-skill detail and the failures."""
    m = model or benchmark_model()
    kb = [s for text in WORLD.values() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))]
    rows, times = [], []
    for skill, question, gold, options in ITEMS:
        start = time.perf_counter()
        reply, _ = m.respond(question, learn=False)
        times.append((time.perf_counter() - start) * 1000)
        rows.append((skill, question, reply, gold, options, judge(reply, skill, gold, options)))
    core = [r for r in rows if r[0] not in ("abstain", "stretch")]
    stretch = [r for r in rows if r[0] == "stretch"]
    unanswerable = [r for r in rows if r[0] == "abstain"]
    answered = [r for r in rows if r[2] is not None]
    answered_wrongly = [r for r in rows if r[0] != "abstain" and r[2] is not None and not r[5]]
    unsupported = sum(1 for r in answered if not grounded(r[2], kb + [r[1]])[0])
    by_skill: dict[str, list[bool]] = {}
    for r in rows:
        by_skill.setdefault(r[0], []).append(r[5])
    return {
        "metrics": {
            "bench.accuracy": sum(r[5] for r in core) / len(core),
            "bench.stretch": sum(r[5] for r in stretch) / len(stretch) if stretch else None,
            "bench.abstain": sum(r[5] for r in unanswerable) / len(unanswerable),
            "bench.wrong_rate": len(answered_wrongly) / max(1, len(answered)),
            "bench.unsupported": unsupported / max(1, len(answered)),
            "speed.median_ms": statistics.median(times),
            "speed.p95_ms": sorted(times)[int(0.95 * (len(times) - 1))],
        },
        "by_skill": {k: (sum(v), len(v)) for k, v in by_skill.items()},
        "failures": [(r[0], r[1], r[2], r[3] or r[4].get("verdict")) for r in rows if not r[5]],
        "questions": len(rows),
    }


# ------------------------------------------------------------------- B: safety check
def _flip_verdict(text: str) -> str | None:
    m = re.match(r"^(\W*)(yes|no)\b", text, re.I)
    if not m:
        return None
    return m.group(1) + ("No" if m.group(2).lower() == "yes" else "Yes") + text[m.end():]


def _drop_not(text: str) -> str | None:
    return re.sub(r"\bnot\s+", "", text, count=1) if re.search(r"\bnot\b", text) else None


def _swap_roles(text: str) -> str | None:
    m = re.search(r"\bThe (\w+) eats (\w+)\b", text)
    return text[:m.start()] + f"The {m.group(2)} eats the {m.group(1)}" + text[m.end():] if m else None


def _swap_entity(text: str) -> str | None:
    pairs = [("mimbat", "lorpan"), ("quillet", "druxle"), ("zarnak", "mimbat"), ("Vostria", "Ulderan"),
             ("Brelm", "Tamsk"), ("Pixel", "Whiskers"), ("Dunmore", "Carrow")]
    for old, new in pairs:
        if old in text:
            return text.replace(old, new)
    return None


def _invent(text: str) -> str:
    return text.rstrip() + " It also sparkles in the moonlight."


def _repeat(text: str) -> str:
    longest = max(re.split(r"(?<=[.!?])\s+", text.strip()), key=len)
    return text.rstrip() + " " + longest


def _truncate(text: str) -> str | None:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[:max(1, len(parts) // 2)]) if len(parts) >= 3 else None


CORRUPTIONS = {"wrong yes/no": _flip_verdict, "dropped 'not'": _drop_not, "roles swapped": _swap_roles,
               "wrong entity": _swap_entity, "invented fact": _invent, "repeated phrase": _repeat,
               "left things out": _truncate}


def run_safety_check(model: LearningModel | None = None) -> dict:
    """B: take the model's own correct answers; corrupt them; the check must reject every
    corrupted copy and accept every original."""
    m = model or benchmark_model()
    originals = rejected_ok = caught = made = 0
    missed, by_kind = [], {}
    for skill, question, gold, options in ITEMS:
        if skill == "abstain":
            continue
        answer = m.explain(question)
        if answer is None or not judge(answer[0], skill, gold, options):
            continue
        reply, _, trace = answer
        evidence = writer_evidence(trace)
        originals += 1
        rejected_ok += not check_draft(reply, evidence, question, reference=reply)[0]
        for name, corrupt in CORRUPTIONS.items():
            bad = corrupt(reply)
            if bad is None or bad == reply:
                continue
            made += 1
            stopped = not check_draft(bad, evidence, question, reference=reply)[0]
            caught += stopped
            row = by_kind.setdefault(name, [0, 0])
            row[0] += stopped
            row[1] += 1
            if not stopped:
                missed.append((name, question, bad))
    return {"metrics": {"check.catch_rate": caught / made if made else None,
                        "check.false_reject": rejected_ok / originals if originals else None},
            "by_kind": {k: tuple(v) for k, v in by_kind.items()}, "missed": missed,
            "corrupted": made, "originals": originals}


# ------------------------------------------------------------------------ C: greetings
GREETINGS = [
    "hello", "hi", "hey", "Hi there!", "Hello everyone", "good morning", "good afternoon", "good evening",
    "good night", "how are you?", "How are you doing today?", "what's up", "sup", "wassup", "heyyy", "hii",
    "hellooo", "thanks", "thank you so much", "thx", "many thanks", "bye", "goodbye", "see you later",
    "take care", "nice to meet you", "pleased to meet you", "sorry", "my bad", "ok", "okay", "cool",
    "who are you?", "what's your name?", "what can you do?", "help me", "namaste", "hola", "bonjour",
    "I'm fine", "I am good", "not bad", "I'm tired", "lol", "gm", "see ya",
]
NOT_GREETINGS = [
    "What is the name of the capital?", "hello world program in python", "tell me about the good things in life",
    "who made the capital of France", "What are you made of?", "Thanks to the river, the marsh floods.",
    "How are you supposed to boil an egg?", "Is it a good morning for a walk?", "bye the way what is Rex",
    "I like the hi-fi system", "Good grief, that is wrong", "what is the time", "tell me the name of the dog",
    "Help me understand photosynthesis", "I am learning Python", "I'm Sam and I like tea",
    "see you at the station tomorrow at five", "ok so what is gravity", "sorry to bother you but what is Mars",
    "Thank goodness it's Friday", "hello kitty is a cartoon character", "my night shift starts at nine",
]


def run_greetings() -> dict:
    """C: recognised greetings, and questions wrongly treated as greetings."""
    m = LearningModel(seed=0)
    missed, false_alarms = [], []
    for text in GREETINGS:
        talk = m._small_talk(text)
        if talk is None or talk["rest"]:
            missed.append(text)
    for text in NOT_GREETINGS:
        if m._small_talk(text) is not None:
            false_alarms.append(text)
    return {"metrics": {"greet.recall": 1 - len(missed) / len(GREETINGS),
                        "greet.false_positive": len(false_alarms) / len(NOT_GREETINGS)},
            "missed": missed, "false_alarms": false_alarms}


# --------------------------------------------------------------- D, E, F: your brain
def _quietly(model: LearningModel, work):
    """Run `work()` without leaving a trace on what the chat shows or learns."""
    saved = (model.last_match, model.last_query, model.last_reply, model.last_source, model.last_trace,
             model._last_talk, model.writer_enabled, model._rng)
    model.writer_enabled = False       # this measures what it knows, not how the writer words it
    model._rng = random.Random(0)      # the same questions in the same order on every run
    try:
        return work()
    finally:
        (model.last_match, model.last_query, model.last_reply, model.last_source, model.last_trace,
         model._last_talk, model.writer_enabled, model._rng) = saved


def run_recall(model: LearningModel, sample: int = 200) -> dict:
    """D: ask the model its own questions (every fact, taught reply and chain) and grade them."""
    by_kind: dict[str, list[int]] = {}
    times = []

    def ask():
        questions = sorted(model.make_questions(100_000), key=lambda q: q["q"])
        random.Random(0).shuffle(questions)
        for q in questions[:sample]:
            start = time.perf_counter()
            reply, _ = model.respond(q["q"], learn=False)
            times.append((time.perf_counter() - start) * 1000)
            row = by_kind.setdefault(q["kind"], [0, 0])
            row[0] += model.grade(q, reply or "")
            row[1] += 1
    _quietly(model, ask)
    total = sum(r[1] for r in by_kind.values())
    return {"metrics": {"brain.recall": sum(r[0] for r in by_kind.values()) / total if total else None,
                        "brain.p95_ms": sorted(times)[int(0.95 * (len(times) - 1))] if times else None},
            "by_kind": {k: tuple(v) for k, v in by_kind.items()}, "asked": total}


def run_health(model: LearningModel, path: str | None = None) -> dict:
    """D/E: size, memory health and your feedback."""
    progress = model.progress()
    studied = max(1, progress["studied"])
    last = model.answer_log[-200:]
    good = sum(1 for e in last if e.get("rating") == "good")
    bad = sum(1 for e in last if e.get("rating") == "bad")
    corrections = sum(1 for e in last if e.get("correction"))
    size = None
    if path and os.path.exists(path):
        base = os.path.splitext(path)[0]
        size = sum(os.path.getsize(f) for f in (path, base + ".neural.npz", base + ".writer.npz",
                                                base + ".vectors.npz") if os.path.exists(f)) / 1e6
    counts = model.parameter_counts()
    return {"metrics": {
        "brain.facts": len(model.facts), "brain.sentences": len(model.knowledge),
        "brain.memories": len(model.memories), "brain.parameters": counts["total"],
        "brain.disk_mb": size,
        "health.strong_share": progress["strong (>=1.0)"] / (studied + len(model.memories)),
        "health.fading_share": progress["fading (<0.4)"] / studied,
        "health.open_questions": progress["open questions"],
        "feedback.good_rate": good / (good + bad) if good + bad else None,
        "feedback.corrections": corrections},
        "rated": good + bad, "exams": progress["exams"]}


def run_writer(model: LearningModel, sample: int = 40) -> dict:
    """F: how the transformer writer does on your data, and how it behaves in real use."""
    if model.writer is None:
        return {"metrics": {}, "note": "no transformer writer loaded"}
    from .train_writer import evaluate, name_stats
    from .writer import build_examples
    examples = _quietly(model, lambda: build_examples(model, limit=sample))
    examples = sorted((e for e in examples if e["evidence"]), key=lambda e: (e["question"], e["answer"]))
    out = {"metrics": {}, "examples": len(examples)}
    if examples:
        random.Random(0).shuffle(examples)
        common, rare = name_stats([e["answer"] for e in examples] + [k["text"] for k in model.knowledge])
        start = time.perf_counter()
        report = evaluate(model.writer, examples, limit=sample, common=common, rare=rare)
        out["metrics"]["writer.pass_seen"] = report["seen"]["grounded"]
        out["metrics"]["writer.pass_new"] = report["new"]["grounded"]
        out["metrics"]["writer.ms"] = (time.perf_counter() - start) * 1000 / (2 * report["seen"]["checked"])
    recent = model.answer_log[-100:]
    if recent:
        out["metrics"]["writer.adoption"] = sum(1 for e in recent if e.get("by") == "transformer") / len(recent)
        reasons = Counter(re.split(r":| \(", r)[0] for e in recent for r in e.get("rejected", []))
        out["rejections"] = dict(reasons.most_common(4))
    return out


# --------------------------------------------------------------------------- targets
# key: (label, group, direction, target, warn, kind). "info" metrics have no target.
METRICS = {
    "bench.accuracy": ("Answers correct", "A  Reasoning benchmark (fixed)", "higher", 0.90, 0.75, "pct"),
    "bench.stretch": ("Harder questions answered (room to grow)", "A  Reasoning benchmark (fixed)", "info", None, None, "pct"),
    "bench.abstain": ("Says 'I don't know' when it should", "A  Reasoning benchmark (fixed)", "higher", 0.90, 0.70, "pct"),
    "bench.wrong_rate": ("Wrong among its answers", "A  Reasoning benchmark (fixed)", "lower", 0.05, 0.15, "pct"),
    "bench.unsupported": ("Answers with words not in its sources", "A  Reasoning benchmark (fixed)", "lower", 0.0, 0.05, "pct"),
    "check.catch_rate": ("Corrupted drafts caught", "B  Writer safety check (fixed)", "higher", 0.95, 0.85, "pct"),
    "check.false_reject": ("Good drafts wrongly rejected", "B  Writer safety check (fixed)", "lower", 0.10, 0.25, "pct"),
    "greet.recall": ("Greetings recognised", "C  Greetings (fixed)", "higher", 0.95, 0.85, "pct"),
    "greet.false_positive": ("Questions mistaken for greetings", "C  Greetings (fixed)", "lower", 0.0, 0.05, "pct"),
    "brain.recall": ("Its own questions answered right", "D  Your brain: knowledge", "higher", 0.85, 0.60, "pct"),
    "brain.facts": ("Facts", "D  Your brain: knowledge", "info", None, None, "int"),
    "brain.sentences": ("Sentences studied", "D  Your brain: knowledge", "info", None, None, "int"),
    "brain.memories": ("Replies you taught", "D  Your brain: knowledge", "info", None, None, "int"),
    "brain.parameters": ("Parameters", "D  Your brain: knowledge", "info", None, None, "int"),
    "brain.disk_mb": ("Size on disk (MB)", "D  Your brain: knowledge", "info", None, None, "mb"),
    "health.strong_share": ("Strongly remembered (grows as you review)", "E  Your brain: memory & feedback", "info", None, None, "pct"),
    "health.fading_share": ("Fading (about to be forgotten)", "E  Your brain: memory & feedback", "lower", 0.20, 0.40, "pct"),
    "health.open_questions": ("Open questions (study goals)", "E  Your brain: memory & feedback", "info", None, None, "int"),
    "feedback.good_rate": ("Rated answers you approved (last 200)", "E  Your brain: memory & feedback", "higher", 0.80, 0.60, "pct"),
    "feedback.corrections": ("Corrections you typed (last 200)", "E  Your brain: memory & feedback", "info", None, None, "int"),
    "writer.pass_seen": ("Writer drafts passing the check", "F  Transformer writer", "higher", 0.90, 0.75, "pct"),
    "writer.pass_new": ("... with names it has never seen", "F  Transformer writer", "higher", 0.80, 0.60, "pct"),
    "writer.adoption": ("Recent answers it worded (last 100)", "F  Transformer writer", "info", None, None, "pct"),
    "writer.ms": ("Writer time per answer (ms)", "F  Transformer writer", "info", None, None, "ms"),
    "speed.median_ms": ("Median answer time (ms)", "Speed (depends on your device)", "info", None, None, "ms"),
    "speed.p95_ms": ("Slowest 5% answer time (ms)", "Speed (depends on your device)", "lower", 500, 2000, "ms"),
    "brain.p95_ms": ("Slowest 5% on your brain (ms)", "Speed (depends on your device)", "info", None, None, "ms"),
}
FIXED = ("bench.accuracy", "bench.abstain", "check.catch_rate", "greet.recall")
FIXED_INVERSE = ("check.false_reject", "greet.false_positive", "bench.wrong_rate")


def regression_index(metrics: dict) -> float | None:
    """One number (0-100) for the fixed tests, to chart over time. It should never fall after a
    code change. It leaves out your own data on purpose."""
    parts = [metrics.get(k) for k in FIXED] + [None if metrics.get(k) is None else 1 - metrics[k]
                                              for k in FIXED_INVERSE]
    parts = [p for p in parts if p is not None]
    return 100 * sum(parts) / len(parts) if parts else None


def status(key: str, value) -> str:
    _label, _group, direction, target, warn, _kind = METRICS[key]
    if value is None or direction == "info":
        return ""
    better = (lambda v, t: v >= t) if direction == "higher" else (lambda v, t: v <= t)
    return "PASS" if better(value, target) else "WARN" if better(value, warn) else "FAIL"


def show(key: str, value) -> str:
    if value is None:
        return "-"
    kind = METRICS[key][5]
    return {"pct": f"{value:.0%}", "int": f"{int(value):,}", "mb": f"{value:.1f}", "ms": f"{value:.0f}"}[kind]


# ------------------------------------------------------------------------- the run
def evaluate_all(model: LearningModel | None = None, path: str | None = None, quick: bool = False) -> dict:
    """Run the standard tests. `quick` = only the fixed tests (A-C), which need no brain."""
    started = time.time()
    parts = {"benchmark": run_benchmark(), "safety": run_safety_check(), "greetings": run_greetings()}
    if model is not None and not quick:
        parts["recall"] = run_recall(model) if (model.facts or model.memories) else {"metrics": {}}
        parts["health"] = run_health(model, path)
        parts["writer"] = run_writer(model)
    metrics = {}
    for part in parts.values():
        metrics.update(part["metrics"])
    metrics["index"] = regression_index(metrics)
    return {"when": time.strftime("%Y-%m-%d %H:%M"), "quick": quick, "seconds": round(time.time() - started, 1),
            "metrics": metrics, "parts": parts}


def history_path(brain: str) -> str:
    return os.path.splitext(brain)[0] + ".eval.json"


def load_history(path: str) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)["runs"]
    except (OSError, ValueError, KeyError):
        return []


def save_run(path: str, report: dict, label: str = "") -> None:
    runs = load_history(path)
    runs.append({"when": report["when"], "label": label, "quick": report["quick"],
                 "metrics": {k: v for k, v in report["metrics"].items() if v is not None}})
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"runs": runs[-100:]}, f, indent=1)
    os.replace(tmp, path)


def change(key: str, now, before) -> str:
    """'+3 pts better' style change versus the previous run."""
    if now is None or before is None or key not in METRICS:
        return ""
    direction, kind = METRICS[key][2], METRICS[key][5]
    delta = now - before
    if abs(delta) < (0.005 if kind == "pct" else 1e-9 if kind == "int" else 0.5):
        return "same"
    text = f"{delta * 100:+.0f} pts" if kind == "pct" else f"{delta:+,.0f}" if kind != "mb" else f"{delta:+.1f}"
    if direction == "info":
        return text
    good = (delta > 0) == (direction == "higher")
    return f"{text} {'better' if good else 'WORSE'}"


def format_report(report: dict, previous: dict | None = None, details: bool = True) -> str:
    lines, metrics = [], report["metrics"]
    before = (previous or {}).get("metrics", {})
    lines.append(f"Scorecard {report['when']}  ({'fixed tests only' if report['quick'] else 'full'}, "
                 f"{report['seconds']}s)" + (f"   compared with the run of {previous['when']}" if previous else ""))
    group = None
    for key, (label, grp, direction, target, _warn, _kind) in METRICS.items():
        if metrics.get(key) is None:
            continue
        if grp != group:
            lines.append(f"\n{grp}")
            group = grp
        goal = "" if target is None else f"target {'>=' if direction == 'higher' else '<='} {show(key, target)}"
        lines.append(f"  {label:<44}{show(key, metrics[key]):>9}  {status(key, metrics[key]):<5} "
                     f"{change(key, metrics[key], before.get(key)):<15}{goal}")
    if metrics.get("index") is not None:
        lines.append(f"\nRegression index (fixed tests only): {metrics['index']:.1f} / 100"
                     + (f"   (last time {before['index']:.1f})" if before.get("index") is not None else ""))
    below = [f"{METRICS[k][0]} ({'FAIL' if status(k, metrics[k]) == 'FAIL' else 'below target'})"
             for k in METRICS if metrics.get(k) is not None and status(k, metrics[k]) in ("FAIL", "WARN")]
    worse = [k for k, v in metrics.items() if k in METRICS and "WORSE" in change(k, v, before.get(k))]
    lines.append("\nNeeds attention: " + (", ".join(below) if below else "nothing below target"))
    if worse:
        lines.append("Got worse since last time: " + ", ".join(METRICS[k][0] for k in worse))
    if details:
        lines.append(_details(report))
    return "\n".join(lines)


def _details(report: dict) -> str:
    out = []
    p = report["parts"]
    bench = p["benchmark"]
    skills = ", ".join(f"{k} {a}/{b}" for k, (a, b) in bench["by_skill"].items())
    out.append(f"\nBenchmark by skill: {skills}")
    for skill, question, reply, gold in bench["failures"][:6]:
        out.append(f"  missed [{skill}] {question} -> {str(reply)[:90]!r} (wanted {gold})")
    safety = p["safety"]
    kinds = ", ".join(f"{k} {a}/{b}" for k, (a, b) in safety["by_kind"].items())
    out.append(f"Safety check caught, by kind of corruption: {kinds}")
    for kind, question, bad in safety["missed"][:4]:
        out.append(f"  NOT caught [{kind}] {question} -> {bad[:90]!r}")
    greet = p["greetings"]
    if greet["missed"]:
        out.append("Greetings not recognised: " + "; ".join(greet["missed"][:6]))
    if greet["false_alarms"]:
        out.append("Treated as greetings by mistake: " + "; ".join(greet["false_alarms"][:6]))
    if "recall" in p and p["recall"].get("by_kind"):
        out.append("Recall by question type: " + ", ".join(
            f"{k} {a}/{b}" for k, (a, b) in p["recall"]["by_kind"].items()))
    if p.get("writer", {}).get("rejections"):
        out.append("Why recent writer drafts were rejected: " + ", ".join(
            f"{k} ({n})" for k, n in p["writer"]["rejections"].items()))
    return "\n".join(out)


def format_history(runs: list[dict], last: int = 8) -> str:
    if not runs:
        return "No evaluations saved yet. Run /evaluate."
    keys = ["index", "bench.accuracy", "bench.abstain", "check.catch_rate", "greet.recall",
            "brain.recall", "feedback.good_rate", "writer.pass_new"]
    heads = ["index", "answers", "abstain", "check", "greet", "recall", "approved", "writer*"]
    lines = ["When              " + "".join(f"{h:>9}" for h in heads) + "   (writer* = new names)"]
    for run in runs[-last:]:
        cells = []
        for k in keys:
            v = run["metrics"].get(k)
            cells.append("-" if v is None else f"{v:.1f}" if k == "index" else f"{v:.0%}")
        lines.append(f"{run['when']:<18}" + "".join(f"{c:>9}" for c in cells)
                     + ("  quick" if run.get("quick") else "") + (f"  {run['label']}" if run.get("label") else ""))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict | None:
    ap = argparse.ArgumentParser(description="Run the standard evaluation scorecard.")
    ap.add_argument("--brain", default="brain.json")
    ap.add_argument("--quick", action="store_true", help="only the fixed tests (A-C)")
    ap.add_argument("--history", action="store_true", help="show past runs and exit")
    ap.add_argument("--label", default="", help="a note to save with this run, e.g. 'after retraining'")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the raw numbers")
    args = ap.parse_args(argv)
    hist = history_path(args.brain)
    if args.history:
        print(format_history(load_history(hist), last=20))
        return None
    model = LearningModel.load(args.brain)
    sibling = os.path.join(os.path.dirname(os.path.abspath(args.brain)), "writer.npz")
    if model.writer is None and os.path.exists(sibling):
        model.load_writer(sibling)
    report = evaluate_all(model, args.brain, quick=args.quick)
    runs = load_history(hist)
    previous = runs[-1] if runs else None
    print(json.dumps(report["metrics"], indent=1) if args.json else format_report(report, previous))
    if not args.no_save:
        save_run(hist, report, args.label)
        print(f"\nSaved to {hist}")
    return report


if __name__ == "__main__":
    main()
