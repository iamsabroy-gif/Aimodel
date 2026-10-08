"""A tiny GPT-style transformer that runs with numpy only.

It is trained on Kaggle with PyTorch (see `train_writer.py`); the weights are
saved to an .npz file, and this module runs them on any device that has numpy
(a phone included). Nothing is sent anywhere.

How it works (a decoder-only transformer, like GPT, just very small):

1. Each character becomes a vector (token embedding) plus a vector for its
   position in the text (position embedding).
2. Each layer lets every character *attend* to the characters before it
   (multi-head causal self-attention), then passes the result through a small
   feed-forward network. Layer norms and residual connections keep it stable.
3. The last layer's output is turned into a score for every possible next
   character; the model writes by repeatedly picking the most likely one.

The text it sees is:  <q> question <e> evidence <a> answer <end>
so it learns to write an answer from the evidence it is given. Text is split
into tokens with byte-pair encoding (BPE), like GPT, or character by character.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

import numpy as np

SPECIALS = ["<pad>", "<q>", "<e>", "<a>", "<end>", "<unk>"]
# Every printable ASCII character (and newline) is always known, seen in training or not.
BASE_CHARS = [chr(c) for c in range(32, 127)] + ["\n"]
PAD, Q, E, A, END, UNK = range(len(SPECIALS))


def _base_chars(texts, max_chars: int) -> list[str]:
    """Printable ASCII plus the most common other characters in the texts (é, ₹, ...)."""
    counts = Counter(c for t in texts for c in t if c not in BASE_CHARS)
    extra = [c for c, _ in counts.most_common(max(0, max_chars - len(BASE_CHARS)))]
    return sorted(set(BASE_CHARS) | set(extra))


class _Tokenizer:
    """Shared prompt layout for both tokenizers."""

    kind = "?"

    def prompt(self, question: str, evidence: list[str], budget: int) -> list[int]:
        """<q> question <e> evidence <a>, cut to fit in `budget` tokens."""
        q = self.encode(question)[: max(0, min(200, budget // 2))]
        room = max(0, budget - len(q) - 3)
        ev = self.encode("\n".join(evidence))[:room]
        return [Q] + q + [E] + ev + [A]

    def example(self, question: str, evidence: list[str], answer: str,
                block: int) -> tuple[list[int], int]:
        """A training sequence and the index where the answer starts."""
        ans = self.encode(answer)[:block // 2] + [END]
        ids = self.prompt(question, evidence, block + 1 - len(ans))
        return ids + ans, len(ids)


class CharTokenizer(_Tokenizer):
    """Turns text into numbers one character at a time."""

    kind = "char"

    def __init__(self, chars: list[str]):
        self.chars = list(chars)
        self.stoi = {c: i + len(SPECIALS) for i, c in enumerate(self.chars)}

    @classmethod
    def build(cls, texts, max_chars: int = 300) -> "CharTokenizer":
        return cls(_base_chars(texts, max_chars))

    def __len__(self) -> int:
        return len(SPECIALS) + len(self.chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, UNK) for c in text]

    def decode(self, ids) -> str:
        n = len(SPECIALS)
        return "".join(self.chars[i - n] for i in ids if i >= n)

    def to_json(self) -> dict:
        return {"kind": self.kind, "chars": self.chars}


_CHUNK = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+")


class BPETokenizer(_Tokenizer):
    """Byte-pair encoding, like GPT: frequent chunks ("the", " mammal") become one token.

    It starts from single characters and repeatedly merges the most common
    neighbouring pair into a new token. Text gets about 4x shorter, so the
    model trains and writes faster and copies whole words more easily.
    """

    kind = "bpe"

    def __init__(self, chars: list[str], merges: list[tuple[str, str]]):
        self.chars = list(chars)
        self.merges = [tuple(m) for m in merges]
        self.tokens = self.chars + [a + b for a, b in self.merges]
        self.stoi = {t: i + len(SPECIALS) for i, t in enumerate(self.tokens)}
        self.ranks = {m: r for r, m in enumerate(self.merges)}
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def build(cls, texts, vocab_size: int = 1024, max_chars: int = 300) -> "BPETokenizer":
        chars = _base_chars(texts, max_chars)
        words = Counter(w for t in texts for w in _CHUNK.findall(t))
        splits = {w: [c for c in w] for w in words}
        merges = []
        while len(SPECIALS) + len(chars) + len(merges) < vocab_size:
            pairs = Counter()
            for w, parts in splits.items():
                for a, b in zip(parts, parts[1:]):
                    pairs[a, b] += words[w]
            if not pairs:
                break
            (a, b), n = pairs.most_common(1)[0]
            if n < 2:
                break
            merges.append((a, b))
            for w, parts in splits.items():
                i, out = 0, []
                while i < len(parts):
                    if i + 1 < len(parts) and parts[i] == a and parts[i + 1] == b:
                        out.append(a + b)
                        i += 2
                    else:
                        out.append(parts[i])
                        i += 1
                splits[w] = out
        return cls(chars, merges)

    def __len__(self) -> int:
        return len(SPECIALS) + len(self.tokens)

    def _encode_chunk(self, chunk: str) -> list[int]:
        if chunk not in self._cache:
            parts = list(chunk)
            while len(parts) > 1:
                pair = min(zip(parts, parts[1:]), key=lambda p: self.ranks.get(p, 1 << 30))
                if pair not in self.ranks:
                    break
                i, out = 0, []
                while i < len(parts):
                    if i + 1 < len(parts) and (parts[i], parts[i + 1]) == pair:
                        out.append(parts[i] + parts[i + 1])
                        i += 2
                    else:
                        out.append(parts[i])
                        i += 1
                parts = out
            if len(self._cache) > 50000:
                self._cache.clear()
            self._cache[chunk] = [self.stoi.get(p, UNK) for p in parts]
        return self._cache[chunk]

    def encode(self, text: str) -> list[int]:
        return [i for chunk in _CHUNK.findall(text) for i in self._encode_chunk(chunk)]

    def decode(self, ids) -> str:
        n = len(SPECIALS)
        return "".join(self.tokens[i - n] for i in ids if i >= n)

    def to_json(self) -> dict:
        return {"kind": self.kind, "chars": self.chars, "merges": self.merges}


def tokenizer_from_json(data) -> _Tokenizer:
    if isinstance(data, list):  # writers saved before BPE: a plain character list
        return CharTokenizer(data)
    if data.get("kind") == "bpe":
        return BPETokenizer(data["chars"], data["merges"])
    return CharTokenizer(data["chars"])


def _layer_norm(x, g, b, eps: float = 1e-5):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def _gelu(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def _softmax(x, axis: int = -1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class TinyTransformer:
    """Runs a trained writer. `weights` uses the names written by train_writer.py."""

    def __init__(self, config: dict, weights: dict, tokenizer: _Tokenizer, meta: dict | None = None):
        self.cfg = config
        self.w = {k: np.asarray(v, dtype=np.float32) for k, v in weights.items()}
        self.tokenizer = tokenizer
        self.meta = meta or {}
        self.n_layer, self.n_head = config["n_layer"], config["n_head"]
        self.d = config["d_model"]
        self.block = config["block_size"]
        self.hd = self.d // self.n_head

    @property
    def n_params(self) -> int:
        return int(sum(v.size for k, v in self.w.items()))

    # ------------------------------------------------------------- one layer
    def _qkv(self, l: int, h: np.ndarray):
        qkv = h @ self.w[f"h{l}.attn.w"] + self.w[f"h{l}.attn.b"]
        q, k, v = np.split(qkv, 3, axis=-1)
        shape = q.shape[:-1] + (self.n_head, self.hd)
        return q.reshape(shape), k.reshape(shape), v.reshape(shape)

    def _mlp(self, l: int, x: np.ndarray) -> np.ndarray:
        h = _layer_norm(x, self.w[f"h{l}.ln2.g"], self.w[f"h{l}.ln2.b"])
        h = _gelu(h @ self.w[f"h{l}.mlp.fc.w"] + self.w[f"h{l}.mlp.fc.b"])
        return x + h @ self.w[f"h{l}.mlp.proj.w"] + self.w[f"h{l}.mlp.proj.b"]

    def _logits(self, x: np.ndarray) -> np.ndarray:
        return _layer_norm(x, self.w["ln_f.g"], self.w["ln_f.b"]) @ self.w["tok_emb"].T

    # --------------------------------------------------- whole text at once
    def forward(self, ids: list[int], cache: list | None = None) -> np.ndarray:
        """Next-character scores at every position (T, vocab). Fills `cache` if given."""
        ids = list(ids)[: self.block]
        t = len(ids)
        x = self.w["tok_emb"][ids] + self.w["pos_emb"][:t]
        mask = np.triu(np.full((t, t), -np.inf, dtype=np.float32), k=1)
        for l in range(self.n_layer):
            h = _layer_norm(x, self.w[f"h{l}.ln1.g"], self.w[f"h{l}.ln1.b"])
            q, k, v = (z.transpose(1, 0, 2) for z in self._qkv(l, h))  # (H, T, hd)
            att = q @ k.transpose(0, 2, 1) / np.sqrt(self.hd) + mask
            y = (_softmax(att) @ v).transpose(1, 0, 2).reshape(t, self.d)
            x = x + y @ self.w[f"h{l}.attn.proj.w"] + self.w[f"h{l}.attn.proj.b"]
            x = self._mlp(l, x)
            if cache is not None:  # keys/values kept for writing the next characters
                kc = np.zeros((self.n_head, self.block, self.hd), np.float32)
                vc = np.zeros_like(kc)
                kc[:, :t], vc[:, :t] = k, v
                cache.append([kc, vc])
        return self._logits(x)

    # ------------------------------------------------ one new character
    def _step(self, token: int, pos: int, cache: list) -> np.ndarray:
        """Scores for the next character, reusing the keys/values already computed."""
        x = self.w["tok_emb"][token] + self.w["pos_emb"][pos]
        for l in range(self.n_layer):
            h = _layer_norm(x, self.w[f"h{l}.ln1.g"], self.w[f"h{l}.ln1.b"])
            q, k, v = self._qkv(l, h)                        # (H, hd)
            kc, vc = cache[l]
            kc[:, pos], vc[:, pos] = k, v
            att = _softmax((kc[:, : pos + 1] @ q[:, :, None])[:, :, 0] / np.sqrt(self.hd))
            y = (att[:, None, :] @ vc[:, : pos + 1])[:, 0, :].reshape(self.d)
            x = x + y @ self.w[f"h{l}.attn.proj.w"] + self.w[f"h{l}.attn.proj.b"]
            x = self._mlp(l, x)
        return self._logits(x)

    def generate(self, prompt: list[int], max_new: int = 300, temperature: float = 0.0,
                 rng: np.random.Generator | None = None) -> list[int]:
        """Write characters after `prompt` until <end> (greedy unless temperature > 0)."""
        prompt = list(prompt)[: self.block - 1]
        max_new = min(max_new, self.block - len(prompt))
        cache: list = []
        logits = self.forward(prompt, cache)[-1]
        out: list[int] = []
        for _ in range(max_new):
            logits = logits.copy()
            logits[[PAD, Q, E, A]] = -np.inf  # structure tokens are never written
            if temperature > 0:
                p = _softmax(logits / temperature)
                token = int((rng or np.random.default_rng()).choice(len(p), p=p))
            else:
                token = int(np.argmax(logits))
            if token == END:
                break
            out.append(token)
            pos = len(prompt) + len(out) - 1
            if pos >= self.block:
                break
            logits = self._step(token, pos, cache)
        return out

    def write(self, question: str, evidence: list[str], max_new: int = 300) -> str:
        max_new = min(max_new, self.block // 2)
        ids = self.tokenizer.prompt(question, evidence, self.block - max_new)
        return self.tokenizer.decode(self.generate(ids, max_new)).strip()

    # --------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        extra = {"config": json.dumps(self.cfg), "tokenizer": json.dumps(self.tokenizer.to_json()),
                 "meta": json.dumps(self.meta)}
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **self.w, **{f"__{k}__": np.array(v) for k, v in extra.items()})
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "TinyTransformer":
        data = np.load(os.path.expanduser(path))
        config = json.loads(str(data["__config__"]))
        tok = json.loads(str(data["__tokenizer__"] if "__tokenizer__" in data else data["__chars__"]))
        meta = json.loads(str(data["__meta__"])) if "__meta__" in data else {}
        weights = {k: data[k] for k in data.files if not k.startswith("__")}
        return cls(config, weights, tokenizer_from_json(tok), meta)

    @classmethod
    def random(cls, tokenizer: _Tokenizer, n_layer: int = 2, n_head: int = 2,
               d_model: int = 32, block_size: int = 128, seed: int = 0) -> "TinyTransformer":
        """An untrained model (for tests): same shapes as a trained one."""
        rng = np.random.default_rng(seed)
        d, v = d_model, len(tokenizer)
        r = lambda *s: rng.normal(0, 0.02, s).astype(np.float32)
        w = {"tok_emb": r(v, d), "pos_emb": r(block_size, d),
             "ln_f.g": np.ones(d, np.float32), "ln_f.b": np.zeros(d, np.float32)}
        for l in range(n_layer):
            w.update({f"h{l}.ln1.g": np.ones(d, np.float32), f"h{l}.ln1.b": np.zeros(d, np.float32),
                      f"h{l}.attn.w": r(d, 3 * d), f"h{l}.attn.b": r(3 * d),
                      f"h{l}.attn.proj.w": r(d, d), f"h{l}.attn.proj.b": r(d),
                      f"h{l}.ln2.g": np.ones(d, np.float32), f"h{l}.ln2.b": np.zeros(d, np.float32),
                      f"h{l}.mlp.fc.w": r(d, 4 * d), f"h{l}.mlp.fc.b": r(4 * d),
                      f"h{l}.mlp.proj.w": r(4 * d, d), f"h{l}.mlp.proj.b": r(d)})
        cfg = {"n_layer": n_layer, "n_head": n_head, "d_model": d, "block_size": block_size,
               "vocab_size": v}
        return cls(cfg, w, tokenizer)
