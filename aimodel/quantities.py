"""Numbers and lists in the model's own sentences: comparing, ranking and counting.

"Mount Everest is 8849 meters high." and "K2 is 8611 meters high." are read as measurements, so
"Which is higher, Everest or K2?", "Is K2 higher than Mont Blanc?", "What is the highest
mountain?" and "How many mountains are higher than 8000 meters?" are answered by comparing the
numbers, not by guessing which sentence looks most like the question. Lists work the same way:
"Maple School teaches French, Spanish and Latin." answers "How many languages does Maple School
teach?" and "Does Maple School teach Latin?". If the question is not clear or the facts do not
settle it, nothing is returned, so the caller can carry on with its other ways of answering.
"""

from __future__ import annotations

import re

NUMBER_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
                "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
                "eighteen", "nineteen", "twenty"]

_MEASURE = re.compile(
    r"^(?:the |a |an )?(?P<subj>[A-Z0-9][\w'.-]*(?: [\w'.-]+){0,3}?)\s+"
    r"(?:is|are|has|have|runs at|travels at|flows at|weighs|measures|costs|grows to|reaches|lives for|lasts|stands)\s+"
    r"(?:about |around |roughly )?(?P<num>\d[\d,]*(?:\.\d+)?|no|" + "|".join(NUMBER_WORDS) + r")\s+(?P<rest>[a-z][a-z ]*?)\s*\.?$", re.I)
_OF = re.compile(  # "Jupiter has a diameter of 139820 kilometers."
    r"^(?:the |a |an )?(?P<subj>[A-Z0-9][\w'.-]*(?: [\w'.-]+){0,3}?)\s+(?:has|have|is|are)\s+(?:a |an |the )?"
    r"(?P<attr>[a-z]+)\s+of\s+(?:about |around |roughly )?(?P<num>\d[\d,]*(?:\.\d+)?)\s+(?P<unit>[a-z]+)\s*\.?$", re.I)
_LIST = re.compile(
    r"^(?:the |a |an )?(?P<subj>[A-Z][\w'.-]*(?: [\w'.-]+){0,3}?)\s+"
    r"(?P<verb>teaches|speaks|plays|owns|contains|includes|sells|offers|grows|studies|eats|has|have|likes)\s+"
    r"(?:only |also )?(?:(?P<count>\w+)\s+(?P<noun>\w+)\s*:\s*)?(?P<items>[^.:]+?)\s*\.?$", re.I)

# adjective -> (what it measures, +1 bigger / -1 smaller). "shorter" may be length or height.
_ADJ = {
    "higher": ("height", 1), "highest": ("height", 1), "taller": ("height", 1), "tallest": ("height", 1),
    "lower": ("height", -1), "lowest": ("height", -1),
    "longer": ("length", 1), "longest": ("length", 1),
    "shorter": ("length|height", -1), "shortest": ("length|height", -1),
    "faster": ("speed", 1), "fastest": ("speed", 1), "slower": ("speed", -1), "slowest": ("speed", -1),
    "deeper": ("depth", 1), "deepest": ("depth", 1), "shallower": ("depth", -1), "shallowest": ("depth", -1),
    "wider": ("width", 1), "widest": ("width", 1), "narrower": ("width", -1), "narrowest": ("width", -1),
    "heavier": ("weight", 1), "heaviest": ("weight", 1), "lighter": ("weight", -1), "lightest": ("weight", -1),
    "larger": (None, 1), "largest": (None, 1), "bigger": (None, 1), "biggest": (None, 1),
    "smaller": (None, -1), "smallest": (None, -1),
    "more": ("noun", 1), "most": ("noun", 1), "fewer": ("noun", -1), "fewest": ("noun", -1),
    "less": ("noun", -1), "least": ("noun", -1),
}
_SYN = {"population": "people", "inhabitants": "people", "residents": "people", "tall": "tall", "age": "years"}
_SIZES = {"height", "length", "width", "depth", "weight", "diameter", "area", "size", "radius", "mass", "volume"}
_SUPERLATIVE = {a for a in _ADJ if a.endswith("est") or a in ("most", "fewest", "least")}


def _stem(word: str) -> str:
    """'cities' and 'city', 'rivers' and 'river' compare equal."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


def _kind(rest: str) -> str:
    """What a measurement is about: 'meters high' is a height, 'kilometers per hour' a speed."""
    r = rest.lower()
    for word, kind in (("years", "years"), ("high", "height"), ("tall", "height"), ("long", "length"), ("per hour", "speed"),
                       ("an hour", "speed"), ("wide", "width"), ("heavy", "weight"), ("deep", "depth")):
        if word in r:
            return kind
    return r.split()[-1]  # "students", "books", ...


class Measure:
    def __init__(self, subj: str, value: float, rest: str, sentence: str, source: str):
        self.subj, self.value, self.rest, self.sentence, self.source = subj, value, rest.lower(), sentence, source
        self.kind = _kind(rest)

    @property
    def name(self) -> str:
        return re.sub(r"^(?:the|a|an)\s+", "", self.subj, flags=re.I)

    @property
    def shown(self) -> str:
        """As it reads in a sentence: 'the tortoise', 'Mount Everest'."""
        return ("the " + self.name) if re.match(r"(?:the|a|an)\s", self.sentence, re.I) and self.name[0].islower() \
            or self.sentence.lower().startswith("the ") else self.name


def cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _forms(name: str, tails: dict | None = None) -> list[str]:
    """The ways a question may name something: 'Mount Everest' is also 'Everest' (if no other
    name ends the same way, so 'school' alone never means one particular school)."""
    low = re.sub(r"^(?:the|a|an)\s+", "", name.lower())
    parts = low.split()
    proper = name.split()[-1][:1].isupper()  # 'Everest' is a name; 'tree' in 'apple tree' is just a noun
    short = len(parts) > 1 and len(parts[-1]) >= 4 and proper and (tails or {}).get(parts[-1], 1) == 1
    return [low] + ([parts[-1]] if short else [])


def find_names(question: str, names) -> list[str]:
    """Which of `names` the question mentions, in the order they appear (longest match wins)."""
    q, taken, found = question.lower(), [], []
    spots, tails = [], {}
    for name in set(names):
        tail = re.sub(r"^(?:the|a|an)\s+", "", name.lower()).split()[-1]
        tails[tail] = tails.get(tail, 0) + 1
    for name in set(names):
        for form in _forms(name, tails):
            for m in re.finditer(r"(?<![\w-])" + re.escape(form) + r"(?![\w-])", q):
                spots.append((m.start(), m.end(), name))
    for start, end, name in sorted(spots, key=lambda s: (-(s[1] - s[0]), s[0])):
        if name not in found and not any(start < e and s < end for s, e in taken):
            taken.append((start, end))
            found.append((start, name))
    return [n for _, n in sorted(found)]


def number_word(n: int) -> str:
    return NUMBER_WORDS[n] if 0 <= n <= 20 else str(n)


def _num(text: str):
    t = text.lower()
    if t in NUMBER_WORDS:
        return NUMBER_WORDS.index(t)
    return int(t) if t.isdigit() else None


def _fmt(v: float) -> str:
    return f"{v:g}" if abs(v) < 1e15 else str(v)


_CLAUSE = re.compile(r"\b(?:is|are|was|were|has|have|had|can|will|which|that|who|they|it|them|because|where|when)\b", re.I)


def _is_list(items: list[str], verb: str, colon: bool) -> bool:
    """Items of a list are short names ("French, Spanish and Latin"), not pieces of a longer sentence.
    With 'has'/'have' only a counted list ("has three children: Ann, Bob and Cleo") is trusted."""
    if verb in ("has", "have", "likes") and not colon:
        return False
    return all(1 <= len(i.split()) <= 3 and not _CLAUSE.search(i) and not re.search(r"\d", i) for i in items)


def _items(text: str) -> list[str]:
    return [i.strip() for i in re.split(r",\s*(?:and\s+)?|\s+and\s+", text) if i.strip()]


class Quantities:
    def __init__(self, sentences):
        """sentences: dicts with "text" and "source"."""
        self.measures: list[Measure] = []
        self.lists: list[dict] = []
        seen = set()
        for s in sentences:
            text = s["text"].strip()
            if text.lower() in seen:  # the same sentence kept as knowledge and as a fact
                continue
            seen.add(text.lower())
            m = _MEASURE.match(text)
            if m:
                num = m.group("num").lower().replace(",", "")
                value = 0.0 if num == "no" else float(NUMBER_WORDS.index(num)) if num in NUMBER_WORDS else float(num)
                self.measures.append(Measure(m.group("subj"), value, m.group("rest"), text, s["source"]))
                continue
            m = _OF.match(text)
            if m:
                self.measures.append(Measure(m.group("subj"), float(m.group("num").replace(",", "")),
                                             f"{m.group('unit')} {m.group('attr')}", text, s["source"]))
                continue
            m = _LIST.match(text)
            if m:
                items = _items(m.group("items"))
                counted = _num(m.group("count") or "") if m.group("count") else None
                if not _is_list(items, m.group("verb").lower(), colon=bool(m.group("noun"))):
                    items = []
                if len(items) >= 2 or (counted is not None and items):
                    self.lists.append({"subj": m.group("subj"), "verb": m.group("verb").lower(),
                                       "items": items, "count": counted if counted is not None else len(items),
                                       "noun": (m.group("noun") or "").lower(), "sentence": text,
                                       "source": s["source"]})

    # -- comparing and ranking
    def _groups(self):
        groups: dict[str, list[Measure]] = {}
        for m in self.measures:  # "one moon" and "two moons" measure the same thing
            groups.setdefault(" ".join(_stem(w) for w in m.rest.split()), []).append(m)
        return {k: v for k, v in groups.items() if len(v) >= 2}

    def _hint(self, group: list[Measure], question: str) -> int:
        """How many of the question's words name the things in this group (their names, the
        document they came from, or what is measured): 'river' for Rivers.txt, 'students'."""
        words = {_stem(_SYN.get(w, w)) for w in re.findall(r"[a-z]+", question.lower()) if len(w) > 2}
        names = set()
        for m in group:
            names.update(_stem(w) for w in re.findall(r"[a-z]+", m.name.lower()))
            names.update(_stem(w) for w in re.findall(r"[a-z]+", m.source.lower().rsplit(".", 1)[0]))
            names.update(_stem(w) for w in m.rest.split())
        return len(words & names)

    def _pick_group(self, adj: str, question: str, entities: list[str]):
        """The set of measurements the question is about, or None if that is unclear."""
        want, _ = _ADJ[adj]
        if adj in ("longer", "longest") and re.search(r"\blive[sd]?\b", question.lower()):
            want = "years"  # "which lives longer" is about age, not length
        groups = self._groups()
        if entities:
            groups = {k: v for k, v in groups.items() if all(any(m.name == e for m in v) for e in entities)}
        if want == "noun":
            words = set(re.findall(r"[a-z]+", question.lower()))
            groups = {k: v for k, v in groups.items() if v[0].kind in words or v[0].kind.rstrip("s") in words}
        elif want:
            kinds = want.split("|")  # "shorter" may be about length or height: the nouns decide
            groups = {k: v for k, v in groups.items() if v[0].kind in kinds}
        if len(groups) > 1 or (len(groups) == 1 and not entities):
            scored = sorted(((self._hint(v, question), k) for k, v in groups.items()), reverse=True)
            top = [k for score, k in scored if score == scored[0][0]]
            if scored[0][0] == 0:
                return None  # the question does not say which kind of thing it is about
            if len(top) > 1 and want is None:  # "the largest planet": a size beats a count
                top = [k for k in top if groups[k][0].kind in _SIZES] or top
            return groups[top[0]] if len(top) == 1 else None
        return next(iter(groups.values())) if len(groups) == 1 else None

    def compare(self, question: str):
        """(reply, steps) for 'Which is higher, A or B?', 'Is A higher than B?', 'What is the
        longest river?' ...; None if this is not such a question or the facts do not settle it."""
        q = question.strip().rstrip("?").strip()
        words = re.findall(r"[a-z]+", q.lower())
        adjs = [w for w in words if w in _ADJ]
        if len(adjs) != 1:
            return None
        adj = adjs[0]
        sign = _ADJ[adj][1]
        names = [m.name for m in self.measures]
        entities = find_names(q, names)
        yesno = bool(re.match(r"^(is|are|does|do|did|has|have|can)\b", q, re.I))
        if yesno and len(entities) == 2 and re.search(r"\b" + adj + r"\s+than\b", q, re.I) is None \
                and adj not in ("more", "fewer", "less"):
            return None
        group = self._pick_group(adj, q, entities)
        if group is None:
            return None
        by_name = {m.name: m for m in group}
        steps = lambda ms: [{"kind": "inference", "text": m.sentence, "source": m.source} for m in ms]
        if yesno and len(entities) == 2:
            a, b = by_name[entities[0]], by_name[entities[1]]
            if a.value == b.value:
                return None
            yes = (a.value > b.value) == (sign > 0)
            unit = a.rest
            reply = (f"{'Yes' if yes else 'No'}. {cap(a.shown)} is {_fmt(a.value)} {unit}, "
                     f"and {b.shown} is {_fmt(b.value)} {unit}.")
            return reply, steps([a, b]) + [{"kind": "inference", "source": "my reasoning",
                                           "text": f"{_fmt(a.value)} {'>' if a.value > b.value else '<'} {_fmt(b.value)}"}]
        if len(entities) >= 2:
            pool = [by_name[e] for e in entities if e in by_name]
        elif not entities and adj in _SUPERLATIVE:
            pool = group
        else:
            return None
        if len(pool) < 2:
            return None
        ranked = sorted(pool, key=lambda m: m.value, reverse=sign > 0)
        if ranked[0].value == ranked[1].value:
            return None  # a tie: do not pick one
        best = ranked[0]
        if len(entities) >= 2:
            others = ", ".join(_fmt(m.value) for m in ranked[1:])
            reply = f"{cap(best.shown)}, at {_fmt(best.value)} {best.rest} (the other {'is' if len(ranked) == 2 else 'are'} {others})."
        else:
            reply = f"{cap(best.shown)}, at {_fmt(best.value)} {best.rest}."
        return reply, steps(ranked) + [{"kind": "inference", "source": "my reasoning",
                                        "text": f"{_fmt(best.value)} is the {'largest' if sign > 0 else 'smallest'} of "
                                                + ", ".join(_fmt(m.value) for m in ranked)}]

    def threshold(self, question: str):
        """'How many mountains are higher than 8000 meters?'"""
        m = re.search(r"\bhow many\b.*?\b(\w+er|more|fewer|less)\s+than\s+(\d[\d,]*(?:\.\d+)?)", question, re.I)
        if not m or m.group(1).lower() not in _ADJ:
            return None
        adj, limit = m.group(1).lower(), float(m.group(2).replace(",", ""))
        group = self._pick_group(adj, question, [])
        if group is None:
            return None
        sign = _ADJ[adj][1]
        hits = [x for x in group if (x.value > limit if sign > 0 else x.value < limit)]
        names = [x.name for x in hits]
        noun = re.search(r"how many\s+(\w+)", question, re.I).group(1)
        n = len(hits)
        reply = f"{number_word(n).capitalize()} {noun}" + (": " + _join(names) + "." if names else ".")
        return reply, [{"kind": "inference", "text": x.sentence, "source": x.source} for x in hits]

    # -- counting and membership
    def count(self, question: str):
        """'How many languages does Maple School teach?' and 'Does Oak School teach French?'"""
        q = question.strip().rstrip("?").strip()
        entities = find_names(q, [l["subj"] for l in self.lists])
        if len(entities) != 1:
            return None
        subj = entities[0]
        cands = [l for l in self.lists if l["subj"] == subj]
        how_many = re.match(r"^how many\s+(\w+)", q, re.I)
        if how_many:
            noun = how_many.group(1).lower()
            verb_stem = None
            vm = re.search(r"\b(?:does|do|did)\s+.*?\b(\w+)$|\b(?:has|have)\b", q, re.I)
            if vm and vm.group(1):
                verb_stem = vm.group(1).lower()
            fits = [l for l in cands if (l["noun"] and l["noun"].rstrip("s") == noun.rstrip("s"))
                    or (verb_stem and l["verb"].startswith(verb_stem[:4]))
                    or (re.search(r"\b(has|have)\b", q) and l["verb"] in ("has", "have"))]
            if len(fits) != 1:
                return None
            lst = fits[0]
            listed = _join(lst["items"])
            reply = f"{lst['subj']} {lst['verb']} {number_word(lst['count'])} {noun}: {listed}."
            return reply, [{"kind": "inference", "text": lst["sentence"], "source": lst["source"]}]
        masked = q
        for form in sorted(_forms(subj), key=len, reverse=True):
            masked = re.sub(r"(?<![\w-])(?:the\s+)?" + re.escape(form) + r"(?![\w-])", "X", masked, flags=re.I)
        member = re.match(r"^(?:does|do|did)\s+X\s+(\w+)\s+(.+)$", masked, re.I)
        if member:
            verb_stem = member.group(1).lower()
            asked = member.group(2).strip().lower()
            asked = re.sub(r"^(?:the|a|an)\s+", "", asked)
            fits = [l for l in cands if l["verb"].startswith(verb_stem[:4])]
            if len(fits) != 1:
                return None
            lst = fits[0]
            yes = asked in [i.lower() for i in lst["items"]]
            if not yes and not any(re.fullmatch(r"[A-Za-z][\w -]*", i) for i in lst["items"]):
                return None
            reply = f"{'Yes' if yes else 'No'}. {lst['sentence'].rstrip('.')}."
            return reply, [{"kind": "inference", "text": lst["sentence"], "source": lst["source"]}]
        return None

    def answer(self, question: str):
        for step in (self.compare, self.threshold, self.count):
            got = step(question)
            if got:
                return got
        return None


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
