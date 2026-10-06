"""Reasoning: facts, chains of facts, and explanations in the model's own words.

Sentences are turned into simple facts (subject, verb, object):
"Cats are mammals." -> ("Cats", "are", "mammals"). Facts can be chained:
"cats are mammals" + "mammals are animals" -> "a cat is an animal".
Answers are then written from facts with the model's own sentence patterns
instead of copying the source text word for word.
"""

from __future__ import annotations

import re
from collections import defaultdict, deque

from .text import STOPWORDS, tokenize

BE = {"is", "are", "was", "were", "am"}
_AUX = {"do", "does", "did"}
_ADVERBS = {"also", "often", "usually", "mainly", "primarily", "mostly", "generally",
            "typically", "always", "sometimes", "only", "still", "really", "even", "just",
            "can", "could", "will", "would", "may", "might", "must", "should"}
_NEG = {"not", "never", "n't"}
_BASE_VERBS = """use produce convert cause lead_to result_in make create need require allow
help depend_on live eat work like love prefer want own study build drive play absorb release
store provide protect form grow feed orbit support run connect carry contain include consist_of
mean refer_to have enable prevent reduce increase control generate transform break_down go meet
speak write teach learn know enjoy hate visit move become remain belong_to come_from""".split()
_IRREGULAR = {
    "have": ["have", "has", "had"], "go": ["go", "goes", "went"], "make": ["make", "makes", "made"],
    "eat": ["eat", "eats", "ate"], "run": ["run", "runs", "ran"], "drive": ["drive", "drives", "drove"],
    "grow": ["grow", "grows", "grew"], "write": ["write", "writes", "wrote"],
    "speak": ["speak", "speaks", "spoke"], "teach": ["teach", "teaches", "taught"],
    "know": ["know", "knows", "knew"], "become": ["become", "becomes", "became"],
    "come": ["come", "comes", "came"], "feed": ["feed", "feeds", "fed"],
    "build": ["build", "builds", "built"], "lead": ["lead", "leads", "led"],
    "mean": ["mean", "means", "meant"], "learn": ["learn", "learns", "learned", "learnt"],
    "break": ["break", "breaks", "broke"], "meet": ["meet", "meets", "met"],
}


def _forms(base: str) -> list[tuple[str, ...]]:
    head, *rest = base.split("_")
    if head in _IRREGULAR:
        forms = _IRREGULAR[head]
    elif head.endswith("y") and head[-2] not in "aeiou":
        forms = [head, head[:-1] + "ies", head[:-1] + "ied"]
    elif head.endswith(("s", "sh", "ch", "x", "o")):
        forms = [head, head + "es", head + "ed"]
    elif head.endswith("e"):
        forms = [head, head + "s", head + "d"]
    else:
        forms = [head, head + "s", head + "ed"]
    return [tuple([f, *rest]) for f in forms]


VERB_FORMS: dict[tuple[str, ...], str] = {}
for _base in _BASE_VERBS:
    for _form in _forms(_base):
        VERB_FORMS[_form] = _base.replace("_", " ")
for _be in BE:
    VERB_FORMS[(_be,)] = "be"
    VERB_FORMS[(_be, "called")] = "be called"
    VERB_FORMS[(_be, "known", "as")] = "be called"
_MAX_VERB = max(len(k) for k in VERB_FORMS)

_BAD_START = {"in", "on", "at", "for", "during", "after", "before", "when", "if", "although",
              "though", "while", "because", "since", "as", "by", "from", "with", "however",
              "but", "and", "or", "so", "there", "to", "of", "unlike", "like", "despite",
              "once", "until", "what", "who", "why", "how", "where", "which"}
_BAD_INSIDE = {"that", "which", "who", "whom", "whose", "when", "where", "if", "because",
               "although", "while"}
_DETERMINERS = {"a", "an", "the", "my", "your", "his", "her", "our", "their", "its", "this",
                "that", "these", "those", "some", "any", "one"}
_BAD_END = _DETERMINERS | {"of", "in", "on", "at", "for", "to", "with", "by", "and", "or", "not",
                           "through", "from", "into", "about", "over", "under", "between", "during",
                           "than", "as", "like", "per", "via", "within", "without", "across", "onto",
                           "upon", "against", "among", "around", "after", "before", "since", "until"}
_PRONOUNS = {"it", "they", "this", "these", "he", "she"}
_STOP_OBJ = {"which", "who", "whom", "because", "while", "whereas", "although", "but",
             "since", "whose", "though", "however"}
_KIND_OF = {"type", "kind", "sort", "form", "member", "species", "group", "part", "example"}
_CHUNK_END = {"that", "which", "who", "used", "with", "of", "in", "for", "by", "to", "from",
              "and", "or", "on", "at", "as", "into", "using", "when", "where", "called"}
_TOKEN = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?[\w'’\-]*|[A-Za-z0-9][\w'’\-]*|[,;:]")  # "8,848" is one number
PERSONAL = {"i", "me", "my", "mine", "myself", "i'm"}


_KEEPABLE = re.compile(r"\(([a-z][a-z ,'-]{2,40})\)")  # "(primarily floral nectar)", not "(Apis mellifera)" or "(5 ft)"


def clean(sentence: str, keep_short: bool = False) -> str:
    """Drop parenthetical asides and citation marks like [1].

    With `keep_short`, a short plain-words aside such as "(camel milk and meat)" stays, because the answer
    to a question is sometimes in it.
    """
    kept = []
    if keep_short:
        def hold(m: re.Match) -> str:
            if len(m.group(1).split()) > 5:
                return m.group(0)
            kept.append(m.group(0))
            return f"\x00{len(kept) - 1}\x00"
        sentence = _KEEPABLE.sub(hold, sentence)
    previous = None
    while previous != sentence:  # nested asides: "(from Latin: x (y) z)"
        previous = sentence
        sentence = re.sub(r"\s*[\(\[][^\(\)\[\]]*[\)\]]", "", sentence)
    sentence = " ".join(sentence.split())
    for n, aside in enumerate(kept):
        sentence = sentence.replace(f"\x00{n}\x00", aside)
    return sentence


def stem(word: str) -> str:
    """Very small stemmer so 'cats' matches 'cat'."""
    w = word.lower().strip("'’")
    if w.endswith(("'s", "’s")):
        w = w[:-2]
    if len(w) > 4 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 4 and w.endswith(("sses", "shes", "ches", "xes")):
        return w[:-2]
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def head(phrase: str) -> str:
    """The main noun of a phrase: 'a small domesticated mammal' -> 'mammal'."""
    words, chunk, i = tokenize(phrase), [], 0
    while i < len(words):
        w = words[i]
        if w in _KIND_OF and i + 1 < len(words) and words[i + 1] == "of":
            chunk, i = [], i + 2
            continue
        if w in _CHUNK_END and chunk:
            break
        if w not in _DETERMINERS:
            chunk.append(w)
        i += 1
    return stem(chunk[-1]) if chunk else ""


_VAGUE_START = {"this", "that", "these", "those", "such", "we", "you", "one", "each", "both",
                "all", "another", "other", "some", "many", "most", "it's", "there's"}


def _good_subject(words: list[str]) -> bool:
    low = [w.lower().replace("’", "'") for w in words]
    if len(low) > 1 and low[0] in _VAGUE_START:  # "this value" - unclear what it refers to
        return False
    if any(w.endswith(("'ll", "'re", "'ve", "'d")) for w in low):
        return False
    return (0 < len(low) <= 6 and low[0] not in _BAD_START and low[-1] not in _BAD_END
            and not _BAD_INSIDE & set(low[1:]) and not any(w in ",;:" for w in low))


_CLAUSE_STARTERS = {"and", "or", "but", "so", "then", "which", "who", "whom", "whose", "because",
                    "while", "where", "when", "although", "though", "since", "whereas", "as", "if",
                    "however", "that", "yet", "while", "thus", "therefore"}


def _list_comma(words: list[str], i: int) -> bool:
    """Is the comma at words[i] part of a list ("seeds, insects and small animals")?

    A list comma is followed by a short item and the list ends with "and"/"or";
    a comma that opens a new clause ("..., and it flies", "..., which is") is not.
    """
    after = words[i + 1:]
    if not after or after[0].lower() in _CLAUSE_STARTERS or after[0] in ",;:":
        return False
    tail = []
    for w in after:
        if w in ";:":
            break
        tail.append(w.lower())
    return len(tail) <= 7 and ("and" in tail or "or" in tail)


def extract_fact(sentence: str, topic: str | None = None) -> dict | None:
    """Turn a sentence into {"subj", "rel", "verb", "obj", "neg"} (or None).

    `topic` replaces a leading pronoun ("It is ..." in an article about X).
    """
    words = _TOKEN.findall(clean(sentence))
    low = [w.lower() for w in words]
    for i in range(1, min(len(words), 8)):
        if low[i] in ",;:":
            return None
        j, neg, verb_words = i, False, []
        while j < len(words) and (low[j] in _ADVERBS or low[j] in _AUX or low[j] in _NEG):
            neg |= low[j] in _NEG
            verb_words.append(words[j])
            j += 1
        for k in range(_MAX_VERB, 0, -1):
            rel = VERB_FORMS.get(tuple(low[j:j + k]))
            if rel:
                break
        else:
            continue
        if not _good_subject(words[:i]):
            continue
        verb_words += words[j:j + k]
        m = j + k
        while m < len(words) and (low[m] in _NEG or low[m] in _ADVERBS):
            neg |= low[m] in _NEG
            verb_words.append(words[m])
            m += 1
        obj = []
        rest = words[m:]
        for n, w in enumerate(rest):
            if w.lower() in _STOP_OBJ or len(obj) >= 20 or w in ";:":
                break
            date_comma = (w == "," and n > 0 and n + 1 < len(rest)
                          and re.fullmatch(r"\d{1,2}", rest[n - 1]) and re.fullmatch(r"\d{4}", rest[n + 1]))
            if w == "," and not date_comma and not _list_comma(rest, n):  # "October 28, 1886" is one date
                break
            obj.append(w)
        while obj and obj[-1].lower() in _BAD_END:  # "processes by (which...)" -> "processes"
            obj.pop()
        if not obj:
            return None
        subj = " ".join(words[:i])
        if subj.lower() in _PRONOUNS:
            if not topic:
                return None
            subj = topic
        return {"subj": subj, "rel": rel, "verb": " ".join(verb_words),
                "obj": re.sub(r"\s+,", ",", " ".join(obj)), "neg": neg}
    return None


_FLIP = {"i": "you", "me": "you", "my": "your", "mine": "yours", "myself": "yourself",
         "am": "are", "i'm": "you're"}


def flip_person(text: str) -> str:
    """Rewrite something you said about yourself: 'I like my dog' -> 'you like your dog'."""
    def swap(m):
        word = m.group(0)
        new = _FLIP.get(word.lower())
        if new is None:
            return word
        return new.capitalize() if word[0].isupper() and word != "I" else new
    out = re.sub(r"[A-Za-z']+", swap, text)
    return re.sub(r"\b([Yy]ou) was\b", r"\1 were", out)


def is_personal(fact_or_text) -> bool:
    text = fact_or_text["subj"] if isinstance(fact_or_text, dict) else fact_or_text
    return bool(PERSONAL & set(tokenize(text)))


def sentence(text: str) -> str:
    text = text.strip().rstrip(":").rstrip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    return text if text.endswith((".", "!", "?")) else text + "."


def lower_first(text: str, names: set[str] = frozenset()) -> str:
    """Lowercase the first word unless it's a name ('Rex') or 'I' or an acronym.

    `names` holds words only ever seen capitalized in the middle of sentences.
    """
    first = re.sub(r"\W+$", "", text.split(" ", 1)[0])
    if not first or first == "I" or first in names or (len(first) > 1 and first.isupper()):
        return text
    return text[0].lower() + text[1:]


def find_names(sentences) -> set[str]:
    """Words capitalized mid-sentence and never written in lowercase: names."""
    capital, lower = set(), set()
    for text in sentences:
        words = re.findall(r"[A-Za-z][\w'-]*", text)
        for n, w in enumerate(words):
            if w[0].isupper():
                if n:
                    capital.add(w)
            else:
                lower.add(w)
    return {w for w in capital if w.lower() not in lower}


_SEQUENCE = re.compile(r"^(first|firstly|then|next|finally|lastly|after that|afterwards|now)\b,?\s*", re.I)


def drop_sequence_word(text: str) -> str:
    """'Then put a bag in' -> 'put a bag in' (the model adds its own 'Then')."""
    return _SEQUENCE.sub("", text)


def state(fact: dict, subject: str | None = None, also: bool = False) -> str:
    """Say a fact as a sentence, in second person if it's about you."""
    verb = fact["verb"]
    if also:
        parts = verb.split(" ")
        if parts[0].lower() in BE or parts[0].lower() in _ADVERBS | _AUX:
            verb = " ".join([parts[0], "also", *parts[1:]])
        else:
            verb = "also " + verb
    text = f"{subject or fact['subj']} {verb} {fact['obj']}"
    if is_personal(fact) or fact.get("source") == "you said":
        text = flip_person(text)
    return sentence(text)


def pronoun(fact: dict) -> str:
    subj = tokenize(fact["subj"])
    if subj == ["i"]:
        return "you"
    if is_personal(fact):
        return flip_person(fact["subj"])
    word = subj[-1] if subj else ""
    return "they" if stem(word) != word and len(subj) == 1 else "it"


def isa_chain(facts: list[dict], start: str, goal: str, max_depth: int = 4) -> list[dict] | None:
    """Find 'start is ... is goal' through is-a facts (breadth-first search)."""
    edges = defaultdict(list)
    for f in facts:
        if f["rel"] in ("be", "be called") and not f["neg"]:
            a, b = head(f["subj"]), head(f["obj"])
            if a and b and a != b:
                edges[a].append((b, f))
    queue, seen = deque([(start, [])]), {start}
    while queue:
        node, path = queue.popleft()
        if node == goal and path:
            return path
        if len(path) >= max_depth:
            continue
        for nxt, f in edges.get(node, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append((nxt, path + [f]))
    return None


def question_kind(text: str) -> str:
    words = tokenize(text)
    if not words:
        return "what"
    if words[0] == "why":
        return "why"
    if words[0] == "how" and (len(words) < 2 or words[1] not in {"many", "much", "old", "long", "far", "big", "tall", "high", "deep", "wide", "fast",
                                                         "heavy", "large", "often", "hot", "cold", "small", "short"}):
        return "how"
    if words[0] in BE | {"do", "does", "did", "can", "could", "has", "have"}:
        return "yesno"
    return "what"


SEQUENCE_MARKER = re.compile(r"^(first|firstly|second|secondly|then|next|finally|lastly|"
                             r"after that|afterwards|now|step \d+)\b", re.I)

CAUSAL = re.compile(r"\b(because|due to|since|as a result of|caused by|so that|thanks to)\b\s+(.*)", re.I)
