"""Turn a dataset file (text, CSV, JSON or JSONL) into things the model can learn from."""

from __future__ import annotations

import csv
import io
import json
from typing import Callable, Iterable

from .documents import MARKDOWN, extract, markdown_to_text

# Column / key names that mark a question-and-answer dataset, in order of preference.
PROMPT_KEYS = ("prompt", "question", "instruction", "input", "query", "q", "user", "human")
REPLY_KEYS = ("response", "answer", "output", "reply", "completion", "a", "assistant", "bot")
TEXT_KEYS = ("text", "content", "body", "article", "sentence", "document")

MAX_PAIRS = 5000  # keeps a phone responsive; bigger files are cut here


def _pick(keys: Iterable[str], wanted: tuple[str, ...]) -> str | None:
    lower = {k.strip().lower(): k for k in keys if isinstance(k, str)}
    return next((lower[w] for w in wanted if w in lower), None)


def _from_records(records: list) -> dict:
    pairs, texts = [], []
    for rec in records:
        if isinstance(rec, str):
            texts.append(rec)
        elif isinstance(rec, dict):
            p, r, t = (_pick(rec, PROMPT_KEYS), _pick(rec, REPLY_KEYS), _pick(rec, TEXT_KEYS))
            if p and r and str(rec[p]).strip() and str(rec[r]).strip():
                pairs.append((str(rec[p]).strip(), str(rec[r]).strip()))
            elif t and str(rec[t]).strip():
                texts.append(str(rec[t]))
    return {"pairs": pairs, "text": "\n".join(texts)}


def parse(name: str, content: str) -> dict:
    """Return {"pairs": [(prompt, reply)...], "text": str, "format": str} for a dataset file."""
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    content = content.lstrip("﻿")
    result = None
    if ext in ("jsonl", "ndjson") or (ext == "json" and content.lstrip()[:1] != "["):
        recs = []
        for line in content.splitlines():
            line = line.strip()
            if line:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
        result = (_from_records(recs), "JSON lines") if recs else None
        if result is None and ext == "json":  # one big object after all
            try:
                obj = json.loads(content)
                result = (_from_records(obj if isinstance(obj, list) else obj.get("data", [obj])), "JSON")
            except (ValueError, AttributeError):
                result = None
    elif ext == "json":
        try:
            result = (_from_records(json.loads(content)), "JSON")
        except ValueError:
            result = None
    elif ext in ("csv", "tsv"):
        delimiter = "\t" if ext == "tsv" else ","
        rows = list(csv.DictReader(io.StringIO(content), delimiter=delimiter))
        if rows and rows[0] and (_pick(rows[0], PROMPT_KEYS) and _pick(rows[0], REPLY_KEYS)
                                 or _pick(rows[0], TEXT_KEYS)):
            result = (_from_records(rows), "CSV" if ext == "csv" else "TSV")
    if result is None:
        if ext in ("md", "markdown", "mdown", "mkd"):
            return {"pairs": [], "text": markdown_to_text(content), "format": "markdown"}
        return {"pairs": [], "text": content, "format": "plain text"}
    parsed, label = result
    parsed["format"] = label
    return parsed


def import_dataset(model, name: str, content: str | bytes,
                   progress: Callable[[int, int, str], None] | None = None) -> dict:
    """Teach `model` from a dataset. Returns what it learned.

    Question/answer rows become taught replies; prose is read like a document.
    `progress(done, total, message)` is called as it goes.
    """
    is_pdf = name.lower().endswith(".pdf")
    if is_pdf:
        content = extract(name, content)  # raises DocumentError with advice if it can't be read
    elif isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")
    data = parse(name, content)
    if is_pdf:
        data["format"] = "PDF"
    pairs = data["pairs"][:MAX_PAIRS]
    total = len(pairs) + (1 if data["text"].strip() else 0)
    learned = 0
    for i, (p, r) in enumerate(pairs, 1):
        try:
            model.learn(p, r)
            learned += 1
        except ValueError:
            pass
        if progress and (i % 25 == 0 or i == len(pairs)):
            progress(i, total, f"Learning answers ({i}/{len(pairs)})")
    sentences = 0
    if data["text"].strip():
        if progress:
            progress(len(pairs), total, "Reading the text")
        sentences = model.add_document(data["text"], name)
        if progress:
            progress(total, total, "Done")
    return {"name": name, "format": data["format"], "answers": learned,
            "sentences": sentences, "cut": max(0, len(data["pairs"]) - MAX_PAIRS)}
