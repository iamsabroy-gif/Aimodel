"""A small neural network that learns word meanings from your text.

This is a skip-gram word2vec model with negative sampling, written in numpy.
It reads every sentence the model sees (your chats, your documents, web pages)
and learns a vector for each word so that words used in similar contexts end
up close together. Averaging those vectors gives a "meaning" vector for a whole
sentence, which lets the model match questions to answers even when they use
different words.
"""

from __future__ import annotations

import os

import numpy as np


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _apply(matrix: np.ndarray, rows: np.ndarray, updates: np.ndarray, max_norm: float = 1.0) -> None:
    """Add a batch of row updates to `matrix`.

    A common word can appear many times in one batch; summing all of its
    updates makes training blow up, so they are damped by sqrt(occurrences)
    and each row's step is capped.
    """
    # One sort groups the rows; reduceat then sums each group (np.add.at does the same far more slowly).
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    starts = np.flatnonzero(np.concatenate(([True], sorted_rows[1:] != sorted_rows[:-1])))
    uniq = sorted_rows[starts]
    total = np.add.reduceat(updates[order], starts, axis=0).astype(matrix.dtype, copy=False)
    total /= np.sqrt(np.diff(np.append(starts, len(rows))))[:, None]
    norms = np.sqrt(np.einsum("ij,ij->i", total, total))[:, None]
    total *= np.minimum(1.0, max_norm / np.maximum(norms, 1e-12))
    matrix[uniq] += total


class WordEmbeddings:
    def __init__(self, dim: int = 48, window: int = 3, negatives: int = 5,
                 lr: float = 0.3, seed: int = 0):
        self.dim, self.window, self.negatives, self.lr = dim, window, negatives, lr
        self.vocab: dict[str, int] = {}
        self.words: list[str] = []
        self.counts = np.zeros(0)
        self.w_in = np.zeros((0, dim), dtype=np.float32)   # the word vectors
        self.w_out = np.zeros((0, dim), dtype=np.float32)  # context vectors
        self.version = 0  # bumped after every training run
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.words)

    # Word vectors need a lot of text before their similarities can be trusted.
    MATURE_WORDS = 50_000

    @property
    def maturity(self) -> float:
        """0..1: how much to trust this network, growing with the text it has read."""
        return float(min(1.0, self.counts.sum() / self.MATURE_WORDS)) if len(self.counts) else 0.0

    def _add_words(self, tokens: list[str]) -> None:
        new = [t for t in dict.fromkeys(tokens) if t not in self.vocab]
        if new:
            for t in new:
                self.vocab[t] = len(self.words)
                self.words.append(t)
            n = len(new)
            init = (self._rng.random((n, self.dim), dtype=np.float32) - 0.5) / self.dim
            self.w_in = np.vstack([self.w_in, init])
            self.w_out = np.vstack([self.w_out, np.zeros((n, self.dim), np.float32)])
            self.counts = np.concatenate([self.counts, np.zeros(n)])
        np.add.at(self.counts, [self.vocab[t] for t in tokens], 1)

    def _pairs(self, sentences: list[np.ndarray], keep: np.ndarray):
        centers, contexts = [], []
        for ids in sentences:
            ids = ids[self._rng.random(len(ids)) < keep[ids]]  # drop very common words
            for off in range(1, self.window + 1):
                if len(ids) > off:
                    centers += [ids[:-off], ids[off:]]
                    contexts += [ids[off:], ids[:-off]]
        if not centers:
            return np.zeros(0, int), np.zeros(0, int)
        return np.concatenate(centers), np.concatenate(contexts)

    def train(self, sentences: list[list[str]], epochs: int = 2, new: bool = True,
              batch: int = 256, min_pairs: int = 0) -> float | None:
        """Train on tokenized sentences. Returns the average loss (None if nothing to learn).

        `new=True` means this text hasn't been seen before, so its word counts
        are added to the vocabulary statistics. `min_pairs` repeats small
        inputs so that a few sentences still get enough practice to learn from.
        """
        sentences = [s for s in sentences if s]
        if new:
            for s in sentences:
                self._add_words(s)
        sentences = [np.array([self.vocab[t] for t in s if t in self.vocab]) for s in sentences]
        sentences = [s for s in sentences if len(s) >= 2]
        if not sentences:
            return None

        freq = self.counts / self.counts.sum()
        t = 1e-3
        keep = np.minimum(1.0, (np.sqrt(freq / t) + 1) * t / np.maximum(freq, 1e-12))
        noise = self.counts ** 0.75
        noise /= noise.sum()
        cdf = np.cumsum(noise)

        pairs_per_epoch = sum(min(len(ids), 2 * self.window) * len(ids) for ids in sentences) // 2
        epochs = min(200, max(epochs, -(-min_pairs // max(1, pairs_per_epoch))))
        total_loss, total_pairs = 0.0, 0
        for epoch in range(epochs):
            lr = self.lr * (1 - 0.9 * epoch / max(1, epochs))
            centers, contexts = self._pairs(sentences, keep)
            order = self._rng.permutation(len(centers))
            centers, contexts = centers[order], contexts[order]
            for start in range(0, len(centers), batch):
                c = centers[start:start + batch]
                o = contexts[start:start + batch]
                neg = np.minimum(np.searchsorted(cdf, self._rng.random((len(c), self.negatives))),
                                 len(self.words) - 1)  # draws from `noise`; the table is built once, not per batch
                targets = np.concatenate([o[:, None], neg], axis=1)        # (B, 1+K)
                vc = self.w_in[c]                                          # (B, d)
                vo = self.w_out[targets]                                   # (B, 1+K, d)
                prob = _sigmoid(np.einsum("bd,bkd->bk", vc, vo))
                labels = np.zeros_like(prob)
                labels[:, 0] = 1.0
                grad = (prob - labels) * lr                                # (B, 1+K)
                _apply(self.w_in, c, -np.einsum("bk,bkd->bd", grad, vo))
                _apply(self.w_out, targets.ravel(),
                       -(grad[:, :, None] * vc[:, None, :]).reshape(-1, self.dim))
                eps = 1e-7
                total_loss -= float(np.log(prob[:, 0] + eps).sum()
                                    + np.log(1 - prob[:, 1:] + eps).sum())
                total_pairs += len(c)
        self.version += 1
        return total_loss / total_pairs if total_pairs else None

    # ----------------------------------------------------------------- using it
    def _centered(self) -> np.ndarray:
        # All word vectors share a common direction (they are all "words");
        # removing it makes similarities reflect meaning.
        return self.w_in - self.w_in.mean(axis=0)

    def sentence_vectors(self, token_lists: list[list[str]]) -> np.ndarray:
        """Unit-length meaning vectors for each sentence (zeros if no known words)."""
        out = np.zeros((len(token_lists), self.dim), dtype=np.float32)
        if not self.words:
            return out
        p = self.counts / self.counts.sum()
        weight = 1e-3 / (1e-3 + p)  # rare words matter more (SIF weighting)
        vecs = self._centered()
        for i, toks in enumerate(token_lists):
            ids = [self.vocab[t] for t in toks if t in self.vocab]
            if ids:
                out[i] = (weight[ids, None] * vecs[ids]).sum(0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)

    def similar(self, word: str, k: int = 5) -> list[tuple[str, float]]:
        """Words the network thinks are closest in meaning to `word`."""
        if word not in self.vocab:
            return []
        vecs = self._centered()
        vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
        sims = vecs @ vecs[self.vocab[word]]
        best = [i for i in np.argsort(-sims) if self.words[i] != word][:k]
        return [(self.words[i], float(sims[i])) for i in best]

    # -------------------------------------------------------------- persistence
    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, w_in=self.w_in, w_out=self.w_out, counts=self.counts,
                                words=np.array(self.words, dtype=str),
                                params=np.array([self.dim, self.window, self.negatives]))
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str, seed: int = 0) -> "WordEmbeddings":
        if not os.path.exists(path):
            return cls(seed=seed)
        data = np.load(path)
        dim, window, negatives = (int(x) for x in data["params"])
        emb = cls(dim=dim, window=window, negatives=negatives, seed=seed)
        emb.words = [str(w) for w in data["words"]]
        emb.vocab = {w: i for i, w in enumerate(emb.words)}
        emb.w_in, emb.w_out, emb.counts = data["w_in"], data["w_out"], data["counts"]
        return emb
