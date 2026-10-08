"""Chains of relations: "Ben's daughter", "Carla lives in Paris, Paris is in France, France is in Europe".

Simple sentences are read as links between things (X lives in Y, X is in Y, X is the capital of Y,
X is the mother of Y, X is a Y, X is older than Y). Questions that need several links are answered
by following them: "In which country does Ben's daughter live?", "Does Carla live in Europe?",
"Who is the youngest of the Johnson children?". "The school with 620 students" and "the school in
Bath" are first turned into the school's name. When the links do not settle the question nothing is
returned, so the caller can carry on with its other ways of answering.
"""

from __future__ import annotations

import re

from .quantities import cap, find_names

_ART = r"(?:the |a |an )?"
_NAME = r"([A-Z][\w'-]*(?: [\w'-]+){0,3}?)"
_LOC = re.compile(r"^" + _ART + _NAME + r"\s+(?:lives in|live in|is located in|is in|are in|is a (?:city|town|village) in)\s+" + _ART + _NAME + r"\.?$", re.I)
_CAPITAL = re.compile(r"^" + _ART + _NAME + r"\s+is the capital(?: city)? of\s+" + _ART + _NAME + r"\.?$", re.I)
_FAMILY = re.compile(r"^" + _ART + _NAME + r"\s+is the (mother|father|brother|sister|son|daughter|husband|wife|uncle|aunt|cousin) of\s+" + _NAME + r"\.?$", re.I)
_ISA = re.compile(r"^" + _ART + r"([A-Za-z][\w -]*?)\s+(?:is|are)\s+(?:a|an)\s+([a-z][\w -]*?)\.?$", re.I)
_ORDER = re.compile(r"^" + _ART + _NAME + r"\s+is\s+(older|taller|younger|shorter|faster|slower|heavier|lighter|bigger|smaller)\s+than\s+" + _ART + _NAME + r"\.?$", re.I)
_HAS = re.compile(r"^" + _ART + _NAME + r"\s+has\s+(\d[\d,]*)\s+([a-z]+)\.?$", re.I)
_OPPOSITE = {"younger": "older", "shorter": "taller", "slower": "faster", "lighter": "heavier", "smaller": "bigger"}
_CHILD = {"son", "daughter"}
CONTINENTS = {"africa", "asia", "europe", "north america", "south america", "antarctica", "oceania", "australia"}
_LISTED = re.compile(r"^" + _ART + _NAME + r"\s+has\s+\w+\s+(\w+)\s*:\s*(.+?)\.?$", re.I)


def _key(s: str) -> str:
    return re.sub(r"^(?:the|a|an)\s+", "", s.strip().lower())


class Relations:
    def __init__(self, sentences):
        self.loc: dict[str, tuple[str, str, str]] = {}   # thing -> (place, sentence, source)
        self.capitals: set[str] = set()                    # things that are the capital of some country
        self.countries: set[str] = set()
        self.family: list[tuple[str, str, str, str, str]] = []  # (a, role, b, sentence, source)
        self.isa: dict[str, list[tuple[str, str, str]]] = {}
        self.order: list[tuple[str, str, str, str, str]] = []   # (bigger, adj, smaller, sentence, source)
        self.has: list[tuple[str, str, float, str, str]] = []   # (thing, unit, value, sentence, source)
        self.groups: dict[str, list[str]] = {}                  # "johnson" -> [Ann, Bob, Cleo]
        self.display: dict[str, str] = {}
        seen = set()
        for s in sentences:
            text, src = s["text"].strip(), s["source"]
            if text.lower() in seen:
                continue
            seen.add(text.lower())
            m = _CAPITAL.match(text)
            if m:
                a, b = m.group(1), m.group(2)
                self._see(a, b)
                self.loc.setdefault(_key(a), (b, text, src))
                self.capitals.add(_key(a))
                self.countries.add(_key(b))
                continue
            m = _LOC.match(text)
            if m:
                self._see(m.group(1), m.group(2))
                self.loc.setdefault(_key(m.group(1)), (m.group(2), text, src))
                continue
            m = _FAMILY.match(text)
            if m:
                self._see(m.group(1), m.group(3))
                self.family.append((m.group(1), m.group(2).lower(), m.group(3), text, src))
                continue
            m = _ORDER.match(text)
            if m:
                self._see(m.group(1), m.group(3))
                adj = m.group(2).lower()
                a, b = (m.group(3), m.group(1)) if adj in _OPPOSITE else (m.group(1), m.group(3))
                self.order.append((a, _OPPOSITE.get(adj, adj), b, text, src))
                continue
            m = _HAS.match(text)
            if m:
                self._see(m.group(1), m.group(1))
                self.has.append((m.group(1), m.group(3).lower(), float(m.group(2).replace(",", "")), text, src))
            m = _LISTED.match(text)
            if m:
                items = [i.strip() for i in re.split(r",\s*(?:and\s+)?|\s+and\s+", m.group(3)) if i.strip()]
                self.groups[_key(m.group(1)).split()[0]] = items
                self.groups[_key(m.group(1))] = items
                continue
            m = _ISA.match(text)
            if m and len(m.group(1).split()) <= 3:
                self.isa.setdefault(_key(m.group(1)), []).append((_key(m.group(2)), text, src))

    def _see(self, *names):
        for n in names:
            self.display.setdefault(_key(n), re.sub(r"^(?:the|a|an)\s+", "", n.strip(), flags=re.I))

    def _show(self, key: str) -> str:
        return self.display.get(key, key)

    # -- places
    def chain(self, thing: str):
        """thing -> [(place, sentence, source), ...] following 'in' links upward."""
        out, seen, cur = [], set(), _key(thing)
        while cur in self.loc and cur not in seen:
            seen.add(cur)
            place, sent, src = self.loc[cur]
            out.append((_key(place), sent, src))
            cur = _key(place)
        return out

    def _person(self, phrase: str):
        """'Ben', 'Carla', "Ben's daughter" -> (name key, steps) or None."""
        phrase = phrase.strip()
        m = re.match(r"^(.+?)'s\s+(\w+)$", phrase)
        if not m:
            found = find_names(phrase, list(self.display.values()))
            if len(found) != 1 or _key(phrase) != _key(found[0]):
                return None  # one named thing, nothing else ("the owl and the fox" is two)
            return _key(found[0]), []
        base = self._person(m.group(1))
        if base is None:
            return None
        who, steps = base
        role = m.group(2).lower()
        matches = []
        for a, r, b, sent, src in self.family:
            if _key(b) == who and (r == role or (role in ("child", "kid") and r in _CHILD)
                                   or (role == "parent" and r in ("mother", "father"))):
                matches.append((_key(a), sent, src))
            elif _key(a) == who and r in ("mother", "father") and role in ("son", "daughter", "child"):
                matches.append((_key(b), sent, src))
            elif _key(a) == who and r == role and r in ("brother", "sister", "cousin", "husband", "wife"):
                matches.append((_key(b), sent, src))
        if role in ("daughter", "son", "child"):
            matches = [(_key(b), sent, src) for a, r, b, sent, src in self.family
                       if _key(a) == who and r in ("mother", "father")]
        if len({k for k, _, _ in matches}) != 1:
            return None
        k, sent, src = matches[0]
        return k, steps + [{"kind": "inference", "text": sent, "source": src}]

    def _place_question(self, q: str):
        m = re.match(r"^(?:in |on |at )?(?:which|what) (city|town|country|continent|place|region)\s+(?:is|does|do)\s+(.+?)(?:\s+(?:live|located|situated))?$", q, re.I) \
            or re.match(r"^where\s+(?:is|does|do)\s+(.+?)(?:\s+(?:live|located|situated))?$", q, re.I)
        if not m:
            return None
        level = m.group(1).lower() if m.re.groups == 2 else "place"
        phrase = m.group(2) if m.re.groups == 2 else m.group(1)
        if "'" not in phrase and level not in ("country", "continent"):
            return None  # "Where does my sister live?" is a plain lookup: not a chain
        who = self._person(phrase)
        if who is None:
            return None
        key, steps = who
        chain = self.chain(key)
        if not chain:
            return None
        pick = chain[0]
        if level == "country":
            pick = next((c for c in chain if c[0] in self.countries), None)
            if pick is None:  # no capital told me which place is a country: the one below the continent
                below = [c for c in chain if c[0] not in CONTINENTS]
                pick = below[-1] if below else None
        elif level == "continent":
            pick = next((c for c in chain if c[0] in CONTINENTS), None)
        if pick is None:
            return None
        used = chain[:chain.index(pick) + 1]
        name = f"{cap(phrase)}" if "'" in phrase else cap(self._show(key))
        reply = f"{name} is in {self._show(pick[0])}."
        if "'" in phrase:
            reply = f"{cap(phrase)}, {self._show(key)}, is in {self._show(pick[0])}."
        return reply, steps + [{"kind": "inference", "text": c[1], "source": c[2]} for c in used]

    def _in_question(self, q: str, negative: bool = False):
        m = re.match(r"^(?:does|is|are|do)\s+(.+?)\s+(?:live in|located in|in|situated in)\s+(.+)$", q, re.I)
        if not m:
            return None
        who = self._person(m.group(1))
        place = _key(re.sub(r"^(?:the|a|an)\s+", "", m.group(2).strip(), flags=re.I))
        if who is None:
            return None
        key, steps = who
        chain = self.chain(key)
        known = {_key(n) for n in self.display}
        if not chain or place not in known:
            return None
        if chain[0][0] == place:
            return None  # said outright: a plain lookup can answer that
        if negative != (not any(c[0] == place for c in chain)):
            return None
        if any(c[0] == place for c in chain):
            used = chain[:[c[0] for c in chain].index(place) + 1]
            return (f"Yes. {cap(self._show(key))} is in {self._show(place)}.",
                    steps + [{"kind": "inference", "text": c[1], "source": c[2]} for c in used])
        return (f"No, not as far as I know. {cap(self._show(key))} is in {self._show(chain[0][0])}.",
                steps + [{"kind": "inference", "text": chain[0][1], "source": chain[0][2]}])

    # -- kinds of thing
    def _class_question(self, q: str):
        m = re.match(r"^(?:is|are)\s+(?:the |a |an )?(.+?)\s+(?:a|an)\s+(.+)$", q, re.I)
        if not m:
            return None
        who, asked = _key(m.group(1)), _key(m.group(2))
        if who not in self.isa:
            return None
        todo, seen, found = [who], set(), []
        while todo:
            cur = todo.pop()
            for cls, sent, src in self.isa.get(cur, []):
                if cls == asked:
                    return None  # a yes is proved by the ordinary reasoning
                if cls not in seen:
                    seen.add(cls)
                    todo.append(cls)
                found.append((cls, sent, src))
        classes = {c for lst in self.isa.values() for c, _, _ in lst}
        if asked in classes:  # a kind I know about, and not one of this thing's kinds
            cls, sent, src = self.isa[who][0]
            return (f"No, not as far as I know. {cap(m.group(1))} is {_article(cls)} {cls}.",
                    [{"kind": "inference", "text": sent, "source": src}])
        return None

    # -- who is the oldest / youngest of ...
    def _order_question(self, q: str):
        m = re.match(r"^(?:who|which|what) is the (oldest|youngest|tallest|shortest|fastest|slowest|heaviest|lightest|biggest|smallest) (?:one )?(?:of|in|among) (?:the )?(.+)$", q, re.I)
        if not m:
            return None
        sup = m.group(1).lower()
        base = {"oldest": "older", "youngest": "older", "tallest": "taller", "shortest": "taller", "fastest": "faster",
                "slowest": "faster", "heaviest": "heavier", "lightest": "heavier", "biggest": "bigger", "smallest": "bigger"}[sup]
        top = sup in ("oldest", "tallest", "fastest", "heaviest", "biggest")
        words = re.findall(r"[a-z]+", m.group(2).lower())
        group = next((self.groups[w] for w in words if w in self.groups), None)
        if not group:
            return None
        names = {_key(g) for g in group}
        edges = [(_key(a), _key(b), sent, src) for a, adj, b, sent, src in self.order if adj == base]
        beats = {n: set() for n in names}  # n -> those it is bigger than (transitively)
        for a, b, _, _ in edges:
            if a in names and b in names:
                beats[a].add(b)
        for _ in names:
            for a in names:
                for b in list(beats[a]):
                    beats[a] |= beats[b]
        if top:
            winners = [n for n in names if len(beats[n]) == len(names) - 1]
        else:
            winners = [n for n in names if all(n in beats[o] for o in names if o != n)]
        if len(winners) != 1:
            return None
        used = [{"kind": "inference", "text": sent, "source": src} for a, b, sent, src in edges if a in names and b in names]
        return f"{cap(self._show(winners[0]))}.", used

    # -- "the school with 620 students" / "the school in Bath"
    def rewrite(self, question: str):
        """Name the thing a question only describes, or None if it describes nothing."""
        m = re.search(r"\bthe (\w+) with (\d[\d,]*) (\w+)", question, re.I)
        if m:
            value, unit, noun = float(m.group(2).replace(",", "")), m.group(3).lower(), m.group(1).lower()
            hits = [h for h in self.has if h[1] == unit and h[2] == value and noun in h[0].lower()]
            if len(hits) == 1:
                return question[:m.start()] + hits[0][0] + question[m.end():]
        m = re.search(r"\bthe (\w+) (?:in|at) ([A-Z]\w*(?: \d+)?)", question)
        if m:
            noun, place = m.group(1).lower(), _key(m.group(2))
            hits = [k for k, (p, _, _) in self.loc.items() if _key(p) == place and noun in k]
            if len(hits) == 1:
                return question[:m.start()] + self._show(hits[0]) + question[m.end():]
        return None

    def answer(self, question: str):
        q = question.strip().rstrip("?").strip()
        for step in (self._place_question, self._in_question, self._order_question):
            got = step(q)
            if got:
                return got
        return None

    def closed_world(self, question: str):
        """A hedged 'No' when everything I know about a thing points somewhere else: use only after
        the ordinary reasoning could not prove the answer."""
        q = question.strip().rstrip("?").strip()
        return self._in_question(q, negative=True) or self._class_question(q)


def _article(word: str) -> str:
    return "an" if word[:1] in "aeiou" else "a"
