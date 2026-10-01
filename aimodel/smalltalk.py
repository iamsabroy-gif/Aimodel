"""Greetings and small talk: hello, good morning, how are you, thanks, bye...

A greeting needs no knowledge, only manners, so it is handled before the
reasoning starts. Matching is strict on purpose: after dropping filler words
("there", "so much", the name you gave me), the whole message has to equal one
known phrase, so a real question such as "What is the name of the capital?"
is never mistaken for small talk.

What it learns from you:
* **Your name**: say "My name is Sam" and greetings use it ("Hello Sam!").
* **Your replies**: `/greet namaste => Namaste {name}!` teaches a new phrase or
  one more reply for a phrase (any language), stored in the brain.
* **Your taste**: each reply has a weight. /good makes it more likely, /bad
  less likely, so over time it says what you like to hear.
Replies you taught with /teach always win over the built-in ones.
"""

from __future__ import annotations

import re

from . import reasoning as rsn

# --------------------------------------------------------------------------- data
# name: (phrases people say, replies it can give). "{name}" is your name if known.
INTENTS: dict[str, tuple[list[str], list[str]]] = {
    "hello": (
        ["hello", "hi", "hey", "hiya", "heya", "howdy", "greetings", "yo", "good day", "hullo"],
        ["Hello {name}! How can I help?",
         "Hi {name}! What would you like to do today?",
         "Hey {name}! Good to see you.",
         "Hello {name}! Ask me something, teach me something, or just chat."]),
    "good morning": (
        ["good morning", "morning", "gm", "good mornin"],
        ["Good morning, {name}! Ready to learn something new today?",
         "Good morning! I hope your day is off to a good start, {name}."]),
    "good afternoon": (
        ["good afternoon", "afternoon"],
        ["Good afternoon, {name}! What shall we work on?",
         "Good afternoon! I hope your day is going well, {name}."]),
    "good evening": (
        ["good evening", "evening"],
        ["Good evening, {name}! How was your day?",
         "Good evening! Anything you'd like to teach me tonight, {name}?"]),
    "good night": (
        ["good night", "gn", "night night", "nighty night", "night", "sweet dreams"],
        ["Good night, {name}! Sleep well. I'll sleep on what I learned too.",
         "Good night! See you tomorrow, {name}."]),
    "how are you": (
        ["how are you", "how are you doing", "how do you do", "how are you today", "hows it going",
         "how is it going", "hows everything", "how have you been", "hows life", "how r u", "hru",
         "how are things", "how you doing", "how is your day", "hows your day", "are you ok",
         "are you well", "how are you feeling"],
        ["I'm doing well, thank you! How are you, {name}?",
         "Doing great, and learning every day. How are you?",
         "All good here, {name}. How about you?"]),
    "whats up": (
        ["whats up", "sup", "wassup", "wazzup", "whats new", "whats going on", "whats happening",
         "what are you up to", "what is up", "what is new"],
        ["Not much, {name}, just learning. What's up with you?",
         "Studying and waiting for your next question. What's up?"]),
    "feeling good": (
        ["i am fine", "im fine", "i am good", "im good", "i am great", "im great", "i am well",
         "im well", "fine", "good", "great", "not bad", "doing well", "im doing well",
         "i am doing well", "all good", "im okay", "i am okay", "im ok", "i am ok", "fine thanks",
         "good thanks", "great thanks", "pretty good", "very good", "feeling good", "feeling great"],
        ["Glad to hear it, {name}!",
         "That's good to hear. What shall we do?",
         "Great! Ask me anything, or teach me something new."]),
    "feeling bad": (
        ["im sad", "i am sad", "im tired", "i am tired", "im not good", "i am not good", "not good",
         "not great", "im not fine", "feeling down", "im bored", "i am bored", "im stressed",
         "i am stressed", "im unwell", "feeling sad", "feeling tired", "bad", "terrible"],
        ["I'm sorry to hear that, {name}. Want to tell me about it, or shall I help with something?",
         "That sounds tough. I'm here if you want to talk, or I can help take your mind off it."]),
    "who are you": (
        ["who are you", "what are you", "whats your name", "what is your name", "your name",
         "tell me about yourself", "introduce yourself", "who am i talking to", "who is this",
         "what should i call you", "what do i call you", "who made you"],
        ["I'm Aimodel, a small AI that learns from you, from the files you give me and from the web. "
         "I explain my reasoning, and like a student I forget what I never use.",
         "I'm Aimodel. Teach me things, ask me questions, or type /help to see what I can do."]),
    "what can you do": (
        ["what can you do", "what do you do", "help", "help me", "how can you help",
         "how can you help me", "what can i ask you", "what can i ask", "how do you work",
         "what are your features", "what can you do for me", "can you help me"],
        ["I can answer from what I've learned, remember facts you tell me, read your files and web "
         "pages (/read), quiz myself or you (/quiz), and show my reasoning (/why). "
         "Type /help for every command."]),
    "thanks": (
        ["thanks", "thank you", "thx", "thnx", "tnx", "ty", "cheers", "many thanks", "appreciate it",
         "i appreciate it", "thanks for your help", "thank you for your help", "thanks for the help",
         "thank u", "thanku", "appreciated", "thanks a ton"],
        ["You're welcome, {name}!",
         "Happy to help!",
         "Anytime! If I get something wrong, tell me with /bad and teach me the right answer."]),
    "youre welcome": (
        ["youre welcome", "you are welcome", "no problem", "no worries", "np", "my pleasure",
         "anytime", "dont mention it"],
        ["Great!", "Glad we're on the same page.", "Okay!"]),
    "bye": (
        ["bye", "goodbye", "good bye", "see you", "see you later", "see ya", "cya", "take care",
         "talk to you later", "ttyl", "later", "gtg", "i have to go", "i must go", "i need to go",
         "bye bye", "farewell", "catch you later", "see you soon", "see you tomorrow",
         "have a nice day", "have a good day"],
        ["Goodbye, {name}! I'll keep what I learned. Type /quit to exit, or come back anytime.",
         "See you later, {name}! Everything is saved as we go.",
         "Take care, {name}! I'll sleep on what we covered."]),
    "nice to meet you": (
        ["nice to meet you", "pleased to meet you", "glad to meet you", "good to meet you",
         "nice meeting you"],
        ["Nice to meet you too, {name}!",
         "Pleased to meet you! Tell me about yourself and I'll remember."]),
    "sorry": (
        ["sorry", "my bad", "apologies", "i am sorry", "im sorry", "excuse me", "pardon",
         "pardon me"],
        ["No problem at all, {name}!", "That's okay!"]),
    "ok": (
        ["ok", "okay", "k", "kk", "alright", "all right", "cool", "nice", "got it", "i see",
         "understood", "sure", "okey", "right"],
        ["Okay!", "Alright.", "Got it."]),
    "laugh": (
        ["haha", "hahaha", "hahahaha", "lol", "lmao", "hehe", "hehehe"],
        ["Glad that made you smile!", "Ha! Happy to amuse."]),
    # A few greetings from other languages, answered in kind. /greet adds yours.
    "hola": (["hola", "buenos dias", "buenas"], ["¡Hola {name}!"]),
    "bonjour": (["bonjour", "salut", "bonsoir"], ["Bonjour {name} !"]),
    "hallo": (["hallo", "guten tag", "guten morgen"], ["Hallo {name}!"]),
    "ciao": (["ciao", "buongiorno"], ["Ciao {name}!"]),
    "namaste": (["namaste", "namaskar", "namaskaram"], ["Namaste {name}!"]),
    "salaam": (["salaam", "salam", "assalamu alaikum", "assalamualaikum", "as salaam alaikum"],
               ["Wa alaikum assalam {name}!"]),
    "konnichiwa": (["konnichiwa", "ohayo", "ohayou"], ["Konnichiwa {name}!"]),
}

# Words that don't change what a greeting means ("hey there", "thank you so much", "hi buddy").
FILLER = {"there", "again", "so", "very", "much", "a", "lot", "really", "just", "please", "kindly",
          "dear", "friend", "buddy", "mate", "pal", "sir", "madam", "everyone", "everybody", "folks",
          "guys", "all", "oh", "well", "um", "uh", "ai", "aimodel", "bot", "assistant"}

_NOT_NAMES = {"fine", "good", "great", "happy", "sad", "tired", "okay", "ok", "ready", "back", "here",
              "sorry", "new", "bored", "hungry", "busy", "well", "stressed", "unwell", "late"}
_NAME_PATTERNS = [
    re.compile(r"\b(?:my name is|my name's|my name\s+is)\s+([A-Za-z][A-Za-z'’-]{1,29})\b", re.I),
    re.compile(r"\b(?:you can call me|call me|people call me)\s+([A-Z][A-Za-z'’-]{1,29})\b"),
    re.compile(r"\b(?:I am|I'm|Im)\s+([A-Z][a-z'’-]{1,29})\b"),
]
_CLAUSE = re.compile(r"[.!?,;:]+|\s+(?:and|&)\s+")


def _tokens(text: str, extra=frozenset()) -> list[str]:
    toks = re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))
    toks = [t.replace("'", "") for t in toks]
    return [t for t in toks if t and t not in FILLER and t not in extra]


def _variants(tokens: list[str]) -> list[str]:
    """The message as typed, and with stretched letters undone ("heyyy", "hii", "byeee")."""
    squeeze = [re.sub(r"(.)\1{2,}", r"\1", t) for t in tokens]
    trailing = [re.sub(r"(.)\1+$", r"\1", t) for t in tokens]
    both = [re.sub(r"(.)\1+$", r"\1", t) for t in squeeze]
    return list(dict.fromkeys(" ".join(v) for v in (tokens, squeeze, trailing, both)))


def _build_index() -> dict[str, str]:
    index = {}
    for intent, (phrases, _replies) in INTENTS.items():
        for phrase in phrases:
            index.setdefault(" ".join(_tokens(phrase)), intent)
    return index


PHRASES = _build_index()


def render(template: str, name: str | None) -> str:
    """Fill in your name, or leave it (and its comma or space) out if I don't know it."""
    if name:
        return template.replace("{name}", name)
    return re.sub(r"(?:,\s*| )\{name\}", "", template).replace("{name}", "").strip()


class SmallTalkMixin:
    """Greetings for LearningModel."""

    def _init_smalltalk(self) -> None:
        self.smalltalk_custom: list[dict] = []   # {"phrase", "reply"} you taught with /greet
        self.smalltalk_weights: dict[str, float] = {}
        self._last_talk: list[str] = []          # reply templates of the last greeting

    # ---------------------------------------------------------------- your name
    def user_name(self) -> str | None:
        """The name you told me, if any."""
        for fact in reversed(self.facts):
            if (fact["source"] == "you said" and not fact["neg"]
                    and fact["rel"] in ("be", "be called")
                    and " ".join(w for w in fact["subj"].lower().split() if w != "the") == "my name"):
                words = re.findall(r"[A-Za-z][A-Za-z'’-]*", fact["obj"])
                if words:
                    return words[0]
        return None

    def _learn_name(self, text: str) -> list[dict]:
        """"My name is Sam", "call me Sam" or "I'm Sam" -> remember the name."""
        for pattern in _NAME_PATTERNS:
            m = pattern.search(text)
            if m and m.group(1).lower() not in _NOT_NAMES:
                name = m.group(1)
                return self.note(f"My name is {name[0].upper() + name[1:]}.")
        return []

    # --------------------------------------------------------------- matching
    def _custom_index(self) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for item in self.smalltalk_custom:
            for key in _variants(_tokens(item["phrase"])):
                index.setdefault(key, []).append(item["reply"])
        return index

    def _match_greeting(self, clause: str, name: str | None):
        """(intent, reply templates) if the whole clause is a known greeting, else None."""
        extra = {name.lower()} if name else set()
        tokens = _tokens(clause, extra)
        if not tokens or len(tokens) > 6:
            return None
        custom = self._custom_index()
        for key in _variants(tokens):
            if key in custom:
                return "your greeting", custom[key]
        for key in _variants(tokens):
            if key in PHRASES:
                return PHRASES[key], INTENTS[PHRASES[key]][1]
        return None

    def _pick(self, templates: list[str]) -> str:
        weights = [max(0.05, self.smalltalk_weights.get(t, 1.0)) for t in templates]
        return self._rng.choices(templates, weights=weights)[0]

    def _small_talk(self, text: str) -> dict | None:
        """Peel greetings off the front of `text`.

        Returns {"reply", "rest", "intents", "templates"} or None if it doesn't
        start with a greeting. `rest` is whatever follows ("Hi, what is Rex?").
        """
        name = self.user_name()
        pieces, pos = [], 0  # (clause, where it ends)
        for m in _CLAUSE.finditer(text):
            pieces.append((text[pos:m.start()], m.start()))
            pos = m.end()
        pieces.append((text[pos:], len(text)))

        intents, templates, replies, cursor = [], [], [], 0
        for clause, end in pieces:
            if not clause.strip():
                continue
            hit = self._match_greeting(clause, name)
            if not hit:
                break
            intent, options = hit
            template = self._pick(options)
            intents.append(intent)
            templates.append(template)
            replies.append(render(template, name))
            cursor = end
        if not replies:
            return None
        rest = text[cursor:].lstrip(" \t\n.,;:!?&-")
        rest = re.sub(r"^and\s+", "", rest, flags=re.I)
        return {"reply": " ".join(replies), "rest": rest.strip(), "intents": intents,
                "templates": templates}

    # ----------------------------------------------------------------- feedback
    def _rate_talk(self, good: bool) -> None:
        for template in self._last_talk:
            weight = self.smalltalk_weights.get(template, 1.0)
            self.smalltalk_weights[template] = weight + 1.0 if good else max(0.05, weight * 0.3)

    # ------------------------------------------------------------ teaching
    def add_greeting(self, phrase: str, reply: str) -> None:
        """Teach a greeting phrase and a reply (use {name} for the user's name)."""
        phrase, reply = phrase.strip(), reply.strip()
        if not phrase or not reply:
            raise ValueError("phrase and reply must be non-empty")
        if not _tokens(phrase):
            raise ValueError("that phrase has no real words in it")
        if not any(i["phrase"].lower() == phrase.lower() and i["reply"] == reply
                   for i in self.smalltalk_custom):
            self.smalltalk_custom.append({"phrase": phrase, "reply": reply})

    def forget_greeting(self, phrase: str) -> int:
        """Forget the greetings you taught for `phrase`. Returns how many were removed."""
        keys = set(_variants(_tokens(phrase)))
        keep = [i for i in self.smalltalk_custom if not keys & set(_variants(_tokens(i["phrase"])))]
        removed = len(self.smalltalk_custom) - len(keep)
        self.smalltalk_custom = keep
        return removed

    def greeting_summary(self) -> dict:
        return {"built-in": {k: len(v[0]) for k, v in INTENTS.items()},
                "custom": list(self.smalltalk_custom), "name": self.user_name()}
