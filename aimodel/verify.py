"""Does a sentence have the right *kind* of content to answer the question?

Ranking by word overlap cannot tell "When was it built?" from "Where was it built?": both share the
same words. This looks at what the question asks for (a time, a number, a person, a place, a reason,
a name, a colour) and checks that the sentence contains that kind of thing. `fit` is +1 when it does,
-1 when it plainly does not, and 0 for questions that do not ask for one particular kind of thing.
"""

from __future__ import annotations

import re

_MONTHS = r"january|february|march|april|may|june|july|august|september|october|november|december"
_TIME = re.compile(r"\b\d{3,4}s?\b|\b(" + _MONTHS + r")\b|\bcentur(y|ies)\b|\b\d{1,2}(st|nd|rd|th)\b|"
                   r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|morning|afternoon|evening|"
                   r"night|noon|spring|summer|autumn|winter|ancient|medieval|decade|era|age)\b|\b\d{1,2}\s*(am|pm)\b", re.I)
_NUMBER = re.compile(r"\d|\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|"
                     r"fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|hundred|thousand|"
                     r"million|billion|dozen|half|double|single|pair)\b", re.I)
_PERSON = re.compile(r"\b(inventor|invented|discovered|founder|founded|king|queen|emperor|architect|designer|designed|"
                     r"author|wrote|written|painter|painted|scientist|physicist|astronaut|first person|vizier|"
                     r"president|leader|explorer|artist|composer|born|named after|built by|created by|"
                     r"developed by|legend|according to)\b", re.I)
_PLACE = re.compile(r"\b(in|at|on|across|throughout|from|near|between|within|around)\s+(the\s+)?[A-Z][a-z]+|"
                    r"\b(lies|lie|located|situated|found|lives|live|inhabit\w*|stands|flows|borders|north|south|east|west|"
                    r"native|scattered|distributed|range|continent|countr(y|ies)|region|hill\w*|mountain\w*|forest\w*|"
                    r"ocean\w*|sea|river\w*|island\w*|coast\w*|climate\w*|city|cities)\b", re.I)
_REASON = re.compile(r"\b(because|due to|so that|in order to|since|as a result|thanks to|which is why|reason|"
                     r"helps?|allows?|makes? it|lets?|enables?|keeps?|protects?|strengthens?|prevents?|to \w+)\b", re.I)
_COLOUR = re.compile(r"\b(red|orange|yellow|green|blue|purple|violet|pink|brown|black|white|grey|gray|golden|gold|"
                     r"silver|striped|spotted|pale|dark|colou?r\w*)\b", re.I)
_CALLED = re.compile(r"\b(called|known as|named|termed|nicknamed|referred to|knows? as|means?)\b", re.I)
_CLAUSE_START = re.compile(r"^\W*(who|whom|whose)\b", re.I)


def kind(question: str) -> str | None:
    q = question.strip().lower()
    if re.match(r"^\W*(when|what year|in what year|what time|which year|what day|which day)\b", q):
        return "time"
    if re.match(r"^\W*how\s+(many|much)\b", q):
        return "count"
    if re.match(r"^\W*how\s+(old|long|tall|big|far|deep|wide|fast|heavy|high|large|often|hot|cold|strong)\b", q):
        return "measure"
    if _CLAUSE_START.match(q):
        return "person"
    if re.match(r"^\W*(where|in which|on which|which (country|city|place|continent|region))\b", q):
        return "place"
    if re.match(r"^\W*why\b", q):
        return "reason"
    if re.match(r"^\W*what\s+colou?r\b", q):
        return "colour"
    if re.search(r"\b(called|known as|named)\b\s*\??$", q):
        return "name"
    return None


def fit(question: str, sentence: str, how: str | None = None) -> float:
    """+1 the sentence has what the question asks for, -1 it plainly lacks it, 0 not applicable."""
    k = how or kind(question)
    if k is None:
        return 0.0
    if k == "time":
        return 1.0 if _TIME.search(sentence) else -1.0
    if k in ("count", "measure"):
        return 1.0 if _NUMBER.search(sentence) else -1.0
    if k == "person":
        names = re.findall(r"(?<![.!?]\s)(?<!^)\b[A-Z][a-z]+", sentence)
        return 1.0 if (_PERSON.search(sentence) or names) else -0.5
    if k == "place":
        return 1.0 if _PLACE.search(sentence) else -0.5
    if k == "reason":
        return 1.0 if _REASON.search(sentence) else -0.5
    if k == "colour":
        return 1.0 if _COLOUR.search(sentence) else -1.0
    if k == "name":
        return 1.0 if _CALLED.search(sentence) else -0.5
    return 0.0
