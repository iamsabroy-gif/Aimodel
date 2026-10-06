"""Turn files people actually have (markdown, PDF) into plain text the model can read.

Markdown needs nothing extra. PDF needs the optional, pure-Python `pypdf` package (pip install pypdf).
"""

from __future__ import annotations

import io
import re
from collections import Counter

MARKDOWN = (".md", ".markdown", ".mdown", ".mkd")


class DocumentError(ValueError):
    """A file that can't be read, with a message that says what to do."""


_FENCE = re.compile(r"^\s*(```|~~~)")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_RULE = re.compile(r"^\s{0,3}([-*_]\s*){3,}$|^\s*(=+|-+)\s*$")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d{1,3}[.)])\s+(?:\[[ xX]\]\s+)?")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)|\[([^\]]+)\]\[[^\]]*\]")
_FOOTNOTE = re.compile(r"\[\^[^\]]+\]")
_URL = re.compile(r"<?https?://\S+>?")
_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~)(?=\S)(.+?)(?<=\S)\1")
_TAG = re.compile(r"</?[A-Za-z][^>]*>|<!--.*?-->")
_END = (".", "!", "?", ":", ";", '"', "'", ")")


def _inline(text: str) -> str:
    """Remove inline markup, keeping the words."""
    text = _IMAGE.sub(lambda m: m.group(1), text)
    text = _LINK.sub(lambda m: m.group(1) or m.group(2), text)
    text = _FOOTNOTE.sub("", text)
    text = _URL.sub("", text)
    text = _TAG.sub("", text)
    for _ in range(2):  # **bold _and italic_**
        text = _EMPHASIS.sub(lambda m: m.group(2), text)
    text = text.replace("`", "")
    text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|>~])", r"\1", text)
    return " ".join(text.split())


def _cells(row: str) -> list[str]:
    return [_inline(c) for c in row.strip().strip("|").split("|")]


def markdown_to_text(md: str) -> str:
    """Plain paragraphs from markdown: no symbols, bullets become sentences, tables become rows of facts."""
    lines = md.replace("\r\n", "\n").lstrip("﻿").split("\n")
    if lines and lines[0].strip() == "---":  # front matter
        for i in range(1, len(lines)):
            if lines[i].strip() in ("---", "..."):
                lines = lines[i + 1:]
                break
    out: list[str] = []
    para: list[str] = []
    item = False  # is `para` a bullet item (it may wrap onto indented lines)
    in_code = False

    def flush() -> None:
        nonlocal item
        if para:
            text = _inline(" ".join(para))
            if item:
                if len(text.split()) >= 3:
                    out.append(text if text.endswith(_END) else text + ".")
            elif text:
                out.append(text)
            para.clear()
        item = False

    i = 0
    while i < len(lines):
        raw = lines[i]
        i += 1
        if _FENCE.match(raw):
            flush()
            in_code = not in_code
            continue
        if in_code:
            continue
        line = raw.strip()
        if not line:
            flush()
        elif _HEADING.match(raw) or _RULE.match(raw):
            flush()
        elif line.startswith("|"):  # a table: header row, separator, then rows
            flush()
            header = _cells(line)
            if i < len(lines) and _TABLE_SEP.match(lines[i]):
                i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                row = _cells(lines[i])
                i += 1
                if row and row[0]:
                    pairs = [f"{h} {c}" for h, c in zip(header[1:], row[1:]) if c and h]
                    out.append(f"{row[0]}: " + ", ".join(pairs) + "." if pairs else row[0] + ".")
        elif _BULLET.match(raw):
            flush()
            item = True
            para.append(_BULLET.sub("", raw, count=1).strip())
        elif raw.startswith(("    ", "\t")) and not para:
            continue  # an indented code block
        else:
            if line.startswith(">"):
                line = line.lstrip("> ").strip()
            if line:
                para.append(line)
    flush()
    return "\n".join(out)


def unwrap(text: str) -> str:
    """Join lines broken at the page margin and undo hyphenation: "exam-\\nple" -> "example"."""
    paragraphs, current = [], []
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            if current:
                paragraphs.append(current)
                current = []
            continue
        current.append(line)
    if current:
        paragraphs.append(current)
    out = []
    for lines in paragraphs:
        merged = ""
        for line in lines:
            if merged.endswith("-") and merged[-2:-1].isalpha() and line[:1].islower():
                merged = merged[:-1] + line
            else:
                merged = f"{merged} {line}" if merged else line
        out.append(merged)
    return "\n".join(out)


def pdf_to_text(data: bytes) -> str:
    """The text of a PDF (needs pypdf). Page numbers and headers repeated on most pages are dropped."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise DocumentError("Reading PDFs needs the pypdf package: run  pip install pypdf  (in Termux: "
                            "pip install pypdf), then try again.") from None
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise DocumentError("That PDF is password-protected.")
        pages = [(page.extract_text() or "") for page in reader.pages]
    except DocumentError:
        raise
    except Exception as e:  # pypdf raises many kinds of errors for damaged files
        raise DocumentError(f"I couldn't open that PDF ({type(e).__name__}).") from None
    if not any(p.strip() for p in pages):
        raise DocumentError("That PDF has no text I can read (it may be a scan; that needs OCR).")
    # lines repeated on at least a third of the pages (and 3+ pages) are headers, footers or page numbers
    seen = Counter(line.strip() for p in pages for line in set(p.split("\n")) if line.strip())
    common = {line for line, n in seen.items() if len(pages) >= 3 and n >= max(3, len(pages) / 3)}
    kept = []
    for p in pages:
        lines = [ln for ln in p.split("\n") if ln.strip() not in common and not re.fullmatch(r"\s*(page\s+)?\d{1,4}\s*", ln, re.I)]
        kept.append("\n".join(lines))
    return unwrap("\n\n".join(kept))


def extract(name: str, data: bytes | str) -> str:
    """Plain text for a file of any kind we understand. Text and markdown come as text or bytes; PDF as bytes."""
    lower = name.lower()
    if lower.endswith(".pdf"):
        if isinstance(data, str):
            raise DocumentError("A PDF has to be read from the file itself, not from text.")
        return pdf_to_text(data)
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    if lower.endswith(MARKDOWN):
        return markdown_to_text(text)
    return text
