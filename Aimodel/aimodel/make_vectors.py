"""Shrink a GloVe / fastText text file to the most common words, for the bundled word vectors.

    python -m aimodel.make_vectors glove.6B.zip --name glove.6B.50d.txt --words 30000 --out aimodel/data/glove50_30k.npz

The big files list common words first, so keeping the first N words keeps the useful ones.
"""

from __future__ import annotations

import argparse
import io
import os
import zipfile

import numpy as np

from .vectors import PretrainedVectors


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="a GloVe/fastText .txt file or a .zip that holds one")
    ap.add_argument("--name", help="which file inside the zip (default: the first .txt)")
    ap.add_argument("--words", type=int, default=30_000)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                                                  "glove50_30k.npz"))
    args = ap.parse_args(argv)
    path = args.source
    if path.endswith(".zip") and args.name:  # read just the member we want, without unpacking the rest
        with zipfile.ZipFile(path) as z, z.open(args.name) as raw:
            lines = io.TextIOWrapper(raw, encoding="utf-8", errors="replace")
            words, rows = [], []
            for line in lines:
                parts = line.rstrip().split(" ")
                words.append(parts[0].lower())
                rows.append(np.asarray(parts[1:], dtype=np.float32))
                if len(words) >= args.words:
                    break
        vectors = PretrainedVectors(words, np.vstack(rows))
    else:
        vectors = PretrainedVectors.from_text(path, args.words)
    vectors.save(args.out)
    print(f"Saved {len(vectors)} words x {vectors.matrix.shape[1]} dimensions to {args.out} "
          f"({os.path.getsize(args.out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
