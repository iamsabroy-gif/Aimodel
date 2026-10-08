"""Optional pretrained word vectors (GloVe / fastText / word2vec text format).

These were trained by others on billions of words, so loading them gives the
model a head start on word meanings (e.g. that "job" and "work" are related)
before it has read much of your own data. Download e.g. glove.6B.50d.txt
(on Kaggle, add the "glove6b50dtxt" dataset) and use /vectors <path>.
"""

from __future__ import annotations

import io
import os
import zipfile

import numpy as np

from .text import STOPWORDS


BUILTIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "glove50_30k.npz")


class PretrainedVectors:
    def __init__(self, words: list[str], matrix: np.ndarray):
        self.words = words
        self.vocab = {w: i for i, w in enumerate(words)}
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self.matrix = (matrix / np.maximum(norms, 1e-9)).astype(np.float16)

    def __len__(self) -> int:
        return len(self.words)

    @classmethod
    def from_text(cls, path: str, max_words: int = 50_000) -> "PretrainedVectors":
        """Read the first `max_words` vectors (these files list common words first)."""
        path = os.path.expanduser(path)
        if path.endswith(".zip"):
            archive = zipfile.ZipFile(path)
            name = next(n for n in archive.namelist() if n.endswith((".txt", ".vec")))
            handle = io.TextIOWrapper(archive.open(name), encoding="utf-8", errors="replace")
        else:
            handle = open(path, encoding="utf-8", errors="replace")
        words, rows, dim = [], [], None
        with handle:
            for n, line in enumerate(handle):
                parts = line.rstrip().split(" ")
                if n == 0 and len(parts) == 2:  # fastText/word2vec header line
                    continue
                if dim is None:
                    dim = len(parts) - 1
                if len(parts) != dim + 1:
                    continue
                words.append(parts[0].lower())
                rows.append(np.asarray(parts[1:], dtype=np.float32))
                if len(words) >= max_words:
                    break
        if not words:
            raise ValueError(f"no word vectors found in {path}")
        return cls(words, np.vstack(rows))

    @classmethod
    def builtin(cls) -> "PretrainedVectors | None":
        """The small set that ships with Aimodel (GloVe 6B, 50 dimensions), or None if it is missing."""
        return cls.load(BUILTIN)

    def similar(self, word: str, k: int = 15) -> list[tuple[str, float]]:
        i = self.vocab.get(word.lower())
        if i is None:
            return []
        sims = self.matrix @ self.matrix[i]
        best = np.argpartition(-sims, min(k + 1, len(sims) - 1))[:k + 1]
        best = sorted(best, key=lambda j: -sims[j])
        return [(self.words[j], float(sims[j])) for j in best if j != i][:k]

    def sentence_vectors(self, token_lists: list[list[str]]) -> np.ndarray:
        """Unit-length mean vector of the meaningful words in each sentence."""
        dim = self.matrix.shape[1]
        out = np.zeros((len(token_lists), dim), dtype=np.float32)
        for n, toks in enumerate(token_lists):
            ids = [self.vocab[t] for t in toks if t in self.vocab and t not in STOPWORDS]
            if ids:
                out[n] = self.matrix[ids].astype(np.float32).mean(axis=0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)

    def save(self, path: str) -> None:
        tmp = path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez_compressed(f, words=np.array(self.words, dtype=str), matrix=self.matrix)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "PretrainedVectors | None":
        if not os.path.exists(path):
            return None
        data = np.load(path)
        vec = cls.__new__(cls)
        vec.words = [str(w) for w in data["words"]]
        vec.vocab = {w: i for i, w in enumerate(vec.words)}
        vec.matrix = data["matrix"]
        return vec
