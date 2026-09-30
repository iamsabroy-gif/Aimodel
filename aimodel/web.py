"""Reading from the internet: web pages and Wikipedia search (stdlib only)."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

USER_AGENT = "Aimodel/0.2 (personal learning assistant; +https://github.com/iamsabroy-gif/Aimodel)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
MAX_BYTES = 3_000_000


def http_get(url: str, timeout: float = 15.0, retries: int = 2) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                return resp.read(MAX_BYTES).decode(charset, errors="replace")
        except urllib.error.HTTPError as e:
            if e.code not in (429, 503) or attempt == retries:
                raise
            try:  # rate limited: wait as asked (capped), then try again
                wait = float(e.headers.get("Retry-After", ""))
            except ValueError:
                wait = 2.0 * (attempt + 1)
            time.sleep(min(wait, 10.0))
    raise AssertionError("unreachable")


_SKIP = {"script", "style", "noscript", "nav", "header", "footer", "svg", "form", "aside"}
_BLOCK = {"p", "div", "br", "li", "tr", "td", "section", "article", "blockquote", "pre",
          "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "table"}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title = ""
        self._skip = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self._skip += 1
        elif tag == "title":
            self._in_title = True
        if tag in _BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP and self._skip:
            self._skip -= 1
        elif tag == "title":
            self._in_title = False
        if tag in _BLOCK:
            self.parts.append("\n")

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self._skip:  # line breaks in HTML source aren't real breaks
            self.parts.append(data.replace("\r", " ").replace("\n", " "))


def html_to_text(html: str) -> tuple[str, str]:
    """Return (title, visible text) of an HTML page."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return " ".join(parser.title.split()), "".join(parser.parts)


def fetch_page(url: str, get=http_get) -> str:
    body = get(url)
    head = body[:2000].lower()
    if "<html" in head or "<body" in head or "<!doctype html" in head:
        return html_to_text(body)[1]
    return body


_WIKI_TAIL = re.compile(r"\n==\s*(See also|References|Notes|External links|Further reading|"
                        r"Bibliography|Sources|Citations)\s*==")


def wikipedia(query: str, limit: int = 2, get=http_get,
              max_chars: int = 20000) -> list[tuple[str, str, str]]:
    """Search Wikipedia. Returns [(title, plain text, url), ...]."""
    params = {"action": "query", "list": "search", "srsearch": query,
              "srlimit": limit, "format": "json"}
    data = json.loads(get(WIKI_API + "?" + urllib.parse.urlencode(params)))
    titles = [hit["title"] for hit in data.get("query", {}).get("search", [])]
    results = []
    for title in titles:  # full-article extracts are limited to one title per request
        params = {"action": "query", "prop": "extracts", "explaintext": 1,
                  "redirects": 1, "titles": title, "format": "json"}
        data = json.loads(get(WIKI_API + "?" + urllib.parse.urlencode(params)))
        for page in data.get("query", {}).get("pages", {}).values():
            text = page.get("extract") or ""
            tail = _WIKI_TAIL.search(text)
            text = text[:tail.start()] if tail else text
            if text:
                url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(page["title"].replace(" ", "_"))
                results.append((page["title"], text[:max_chars], url))
    return results
