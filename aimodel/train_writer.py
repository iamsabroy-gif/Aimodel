"""Train the tiny transformer writer. Run this on Kaggle (Settings: GPU on).

    python -m aimodel.train_writer --data writer_data --out writer.npz

`--data` is the folder written by /export: writer_data.jsonl (question /
evidence / answer examples) and corpus.txt (everything the model has read).
Training happens in two mixed parts:

* language practice: predict the next character of text it has read, so it
  learns spelling and grammar;
* writing practice: given a question and evidence, write the answer. Only the
  answer characters are scored, so it learns to answer, not to copy prompts.

The result is saved as an .npz file that runs with numpy only (no PyTorch),
on Kaggle, a computer or a phone: /writer load writer.npz

Use --resume old_writer.npz to keep training a writer you already have
(gradual learning: export again after you've taught it more, then resume).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter

import numpy as np

from .transformer import END, PAD, BPETokenizer, CharTokenizer, TinyTransformer

SIZES = {  # name: (layers, heads, dimensions, tokens of context)
    "tiny": (2, 2, 64, 128),     # for a quick test on a CPU
    "small": (4, 4, 128, 256),
    "base": (6, 6, 192, 384),    # about 3M parameters: the Kaggle default
}


def load_data(folder: str) -> tuple[list[dict], list[str]]:
    with open(os.path.join(folder, "writer_data.jsonl"), encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]
    corpus_path = os.path.join(folder, "corpus.txt")
    corpus = []
    if os.path.exists(corpus_path):
        with open(corpus_path, encoding="utf-8") as f:
            corpus = [line.rstrip("\n") for line in f if line.strip()]
    return examples, corpus


def _torch_model(torch, n_vocab: int, block: int, d: int, n_layer: int, n_head: int, dropout: float):
    """A GPT-style model that matches TinyTransformer's numpy maths exactly."""
    nn, F = torch.nn, torch.nn.functional

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
            self.attn, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
            self.fc, self.fc_proj = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
            self.drop = nn.Dropout(dropout)

        def forward(self, x):
            b, t, _ = x.shape
            q, k, v = self.attn(self.ln1(x)).split(d, dim=2)
            q, k, v = (z.view(b, t, n_head, d // n_head).transpose(1, 2) for z in (q, k, v))
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                               dropout_p=dropout if self.training else 0.0)
            x = x + self.drop(self.proj(y.transpose(1, 2).reshape(b, t, d)))
            return x + self.drop(self.fc_proj(F.gelu(self.fc(self.ln2(x)), approximate="tanh")))

    class GPT(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok, self.pos = nn.Embedding(n_vocab, d), nn.Embedding(block, d)
            self.blocks = nn.ModuleList(Block() for _ in range(n_layer))
            self.ln_f, self.drop = nn.LayerNorm(d), nn.Dropout(dropout)
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.normal_(p, std=0.02)

        def forward(self, idx):
            x = self.drop(self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device)))
            for blk in self.blocks:
                x = blk(x)
            return self.ln_f(x) @ self.tok.weight.T  # the output layer shares the embeddings

    return GPT()


def to_numpy(model, n_layer: int) -> dict:
    """PyTorch weights -> the names TinyTransformer uses (Linear weights transposed)."""
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    w = {"tok_emb": sd["tok.weight"], "pos_emb": sd["pos.weight"],
         "ln_f.g": sd["ln_f.weight"], "ln_f.b": sd["ln_f.bias"]}
    for l in range(n_layer):
        p = f"blocks.{l}."
        w.update({f"h{l}.ln1.g": sd[p + "ln1.weight"], f"h{l}.ln1.b": sd[p + "ln1.bias"],
                  f"h{l}.attn.w": sd[p + "attn.weight"].T, f"h{l}.attn.b": sd[p + "attn.bias"],
                  f"h{l}.attn.proj.w": sd[p + "proj.weight"].T, f"h{l}.attn.proj.b": sd[p + "proj.bias"],
                  f"h{l}.ln2.g": sd[p + "ln2.weight"], f"h{l}.ln2.b": sd[p + "ln2.bias"],
                  f"h{l}.mlp.fc.w": sd[p + "fc.weight"].T, f"h{l}.mlp.fc.b": sd[p + "fc.bias"],
                  f"h{l}.mlp.proj.w": sd[p + "fc_proj.weight"].T, f"h{l}.mlp.proj.b": sd[p + "fc_proj.bias"]})
    return w


def from_numpy(torch, model, weights: dict, n_layer: int) -> None:
    """Load a saved writer back into PyTorch (to keep training it)."""
    sd = {"tok.weight": weights["tok_emb"], "pos.weight": weights["pos_emb"],
          "ln_f.weight": weights["ln_f.g"], "ln_f.bias": weights["ln_f.b"]}
    for l in range(n_layer):
        p = f"blocks.{l}."
        sd.update({p + "ln1.weight": weights[f"h{l}.ln1.g"], p + "ln1.bias": weights[f"h{l}.ln1.b"],
                   p + "attn.weight": weights[f"h{l}.attn.w"].T, p + "attn.bias": weights[f"h{l}.attn.b"],
                   p + "proj.weight": weights[f"h{l}.attn.proj.w"].T, p + "proj.bias": weights[f"h{l}.attn.proj.b"],
                   p + "ln2.weight": weights[f"h{l}.ln2.g"], p + "ln2.bias": weights[f"h{l}.ln2.b"],
                   p + "fc.weight": weights[f"h{l}.mlp.fc.w"].T, p + "fc.bias": weights[f"h{l}.mlp.fc.b"],
                   p + "fc_proj.weight": weights[f"h{l}.mlp.proj.w"].T,
                   p + "fc_proj.bias": weights[f"h{l}.mlp.proj.b"]})
    model.load_state_dict({k: torch.tensor(np.ascontiguousarray(v)) for k, v in sd.items()})


_WORD = re.compile(r"[A-Za-z]{3,}")
_ONSETS = ["", "b", "c", "d", "f", "g", "h", "j", "k", "l", "m", "n", "p", "r", "s", "t", "v", "w", "z",
           "br", "cr", "dr", "fl", "gr", "pl", "pr", "st", "str", "tr", "ch", "sh", "th", "wh", "sp",
           "sk", "sl", "sn", "gl", "cl", "bl", "fr"]
_NUCLEI = ["a", "e", "i", "o", "u", "a", "e", "i", "o", "u", "oo", "ea", "ai", "ou", "ie", "au", "ee"]
_CODAS = ["", "", "", "n", "r", "s", "t", "l", "m", "k", "x", "nd", "st", "rt", "ng", "sh", "ck", "se"]


_SIMPLE = ["b", "d", "f", "g", "k", "l", "m", "n", "p", "r", "s", "t", "v", "z"]


def fake_word(rng: random.Random) -> str:
    """A pronounceable made-up word, 3-9 letters, of varied shape (moose-like, panda-like...)."""
    for _ in range(20):
        n = rng.choices([1, 2, 3], [3, 6, 2])[0]
        parts = []
        for i in range(n):
            onset = rng.choice(_ONSETS) if i == 0 else rng.choice(_SIMPLE)
            nucleus = rng.choice(_NUCLEI) if (i == 0 or rng.random() < 0.25) else rng.choice("aeiou")
            coda = rng.choice(_CODAS) if i == n - 1 else (rng.choice("nrs") if rng.random() < 0.2 else "")
            parts.append(onset + nucleus + coda)
        word = "".join(parts)
        if 3 <= len(word) <= 9:
            return word
    return rng.choice(_SIMPLE) + rng.choice("aeiou") + rng.choice(_SIMPLE) + rng.choice("aeiou")


def name_stats(texts: list[str]) -> tuple[set[str], list[str]]:
    """(ordinary words, rare words). Ordinary words occur in many different texts; the rest
    are names (moose, Canberra...) that the model must copy from the evidence."""
    spread = Counter(w for t in texts for w in {w.lower() for w in _WORD.findall(t)})
    floor = max(3, 0.02 * len(texts))
    common = {w for w, n in spread.items() if n >= floor}
    rare = sorted(w for w, n in spread.items() if n < floor and len(w) >= 4)
    return common, rare


def swap_names(question: str, evidence: list[str], answer: str, rng: random.Random,
               common: set[str], rare: list[str], most: int = 4, real: float = 0.4):
    """Replace the names the answer copies from the evidence, everywhere and consistently,
    by made-up words (or other real words from the data). Memorising a name then never helps:
    the only way to answer is to read the evidence."""
    text = " ".join(evidence)
    names = sorted({w for w in _WORD.findall(answer) if w.lower() not in common
                    and re.search(r"\b%s\b" % re.escape(w), text)})
    if not names:
        return question, evidence, answer
    rng.shuffle(names)
    used = {w.lower() for w in _WORD.findall(question + " " + text + " " + answer)}
    swap = {}
    for name in names[:most]:
        for _ in range(5):
            new = rng.choice(rare) if rare and rng.random() < real else fake_word(rng)
            if new.lower() not in used:
                break
        used.add(new.lower())
        swap[name.lower()], swap[name.capitalize()] = new.lower(), new.capitalize()
    pattern = re.compile(r"\b(%s)\b" % "|".join(map(re.escape, swap)))
    sub = lambda t: pattern.sub(lambda m: swap[m.group(1)], t)
    return sub(question), [sub(e) for e in evidence], sub(answer)


class Batches:
    """Mixes writing practice (answers only are scored) with language practice.

    Name swapping (see `swap_names`) is applied to most examples so the model learns to
    copy names from the evidence instead of reciting facts it memorised.
    """

    def __init__(self, tokenizer, examples, corpus, block, rng, swap: float = 0.85):
        self.tok, self.examples, self.block, self.rng = tokenizer, examples, block, rng
        self.weights = [float(e.get("weight", 1.0)) for e in examples]
        self.corpus, self.swap = corpus, swap
        self.common, self.rare = name_stats(corpus + [e["answer"] for e in examples])
        stream = tokenizer.encode("\n".join(corpus)) if corpus else []
        self.stream = np.array(stream, dtype=np.int64)

    def _swap_names(self, question: str, evidence: list[str], answer: str):
        return swap_names(question, evidence, answer, self.rng, self.common, self.rare)

    def _qa(self):
        ex = self.rng.choices(self.examples, weights=self.weights)[0]
        evidence = list(ex["evidence"])
        if len(evidence) > 1 and self.rng.random() < 0.5:
            self.rng.shuffle(evidence)  # order shouldn't matter
        if self.corpus and self.rng.random() < 0.3:  # an unrelated sentence: learn to ignore it
            evidence.insert(self.rng.randrange(len(evidence) + 1), self.rng.choice(self.corpus))
        question, answer = ex["question"], ex["answer"]
        if evidence and self.rng.random() < self.swap:
            question, evidence, answer = self._swap_names(question, evidence, answer)
        ids, start = self.tok.example(question, evidence, answer, self.block)
        mask = [0] * (start - 1) + [1] * (len(ids) - start)
        return ids, mask

    def _text(self):
        if len(self.stream) <= self.block + 1:
            ids = self.stream.tolist()
        else:
            i = self.rng.randrange(len(self.stream) - self.block - 1)
            ids = self.stream[i:i + self.block + 1].tolist()
        return ids, [1] * (len(ids) - 1)

    def get(self, n: int, lm_mix: float):
        xs, ys, ms = [], [], []
        for _ in range(n):
            use_text = len(self.stream) > 16 and (not self.examples or self.rng.random() < lm_mix)
            ids, mask = self._text() if use_text else self._qa()
            ids = ids[: self.block + 1]
            x, y, m = ids[:-1], ids[1:], mask[: len(ids) - 1]
            pad = self.block - len(x)
            xs.append(x + [PAD] * pad)
            ys.append(y + [PAD] * pad)
            ms.append(m + [0] * pad)
        return np.array(xs), np.array(ys), np.array(ms, dtype=np.float32)


def evaluate(net: TinyTransformer, held_out: list[dict], limit: int = 50,
             common: set[str] | None = None, rare: list[str] | None = None) -> dict:
    """On questions it never trained on: how often its answers pass the evidence check and
    how often they say exactly what the reference says. Done twice: as written, and with
    every name replaced by one it has never seen, which is the honest test of reading."""
    from .writer import grounded
    norm = lambda t: re.sub(r"\W+", " ", t.lower()).strip()
    results = {}
    for label, rng in (("seen", None), ("new", random.Random(7))):
        passed = same = 0
        samples = []
        for ex in held_out[:limit]:
            q, ev, ans = ex["question"], ex["evidence"], ex["answer"]
            if rng is not None:
                q, ev, ans = swap_names(q, ev, ans, rng, common or set(), rare or [], real=0.3)
            draft = net.write(q, ev, max_new=200)
            ok, _ = grounded(draft, ev + [q])
            passed += ok
            same += norm(draft) == norm(ans)
            samples.append((q, draft, ok))
        n = min(limit, len(held_out))
        results[label] = {"grounded": passed / n if n else None, "same": same / n if n else None,
                          "checked": n, "samples": samples[:4]}
    return results


def main(argv: list[str] | None = None) -> TinyTransformer:
    try:
        import torch
    except ImportError:
        raise SystemExit("Training needs PyTorch (Kaggle has it). Running the writer only needs numpy.")
    ap = argparse.ArgumentParser(description="Train the tiny transformer writer.")
    ap.add_argument("--data", required=True, help="folder written by /export")
    ap.add_argument("--out", default="writer.npz")
    ap.add_argument("--size", choices=SIZES, default="base")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="small models learn to copy from the evidence much sooner at 1e-3")
    ap.add_argument("--lm-mix", type=float, default=0.3, help="share of language practice")
    ap.add_argument("--resume", help="a writer .npz to keep training")
    ap.add_argument("--tokens", choices=("bpe", "char"), default="bpe",
                    help="bpe: word pieces like GPT (default); char: one character per token")
    ap.add_argument("--vocab", type=int, default=2048, help="BPE vocabulary size")
    ap.add_argument("--swap", type=float, default=0.85,
                    help="share of examples whose names are swapped for made-up words")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    examples, corpus = load_data(args.data)
    if not examples and not corpus:
        raise SystemExit("No training data found. Run /export first.")

    old = TinyTransformer.load(args.resume) if args.resume else None
    if old:
        tokenizer, cfg = old.tokenizer, dict(old.cfg)
        n_layer, n_head, d, block = cfg["n_layer"], cfg["n_head"], cfg["d_model"], cfg["block_size"]
    else:
        n_layer, n_head, d, block = SIZES[args.size]
        texts = corpus + [t for e in examples for t in [e["question"], e["answer"], *e["evidence"]]]
        tokenizer = (BPETokenizer.build(texts, vocab_size=args.vocab) if args.tokens == "bpe"
                     else CharTokenizer.build(texts))
        cfg = {"n_layer": n_layer, "n_head": n_head, "d_model": d, "block_size": block,
               "vocab_size": len(tokenizer)}

    rng.shuffle(examples)
    n_held = min(max(1, len(examples) // 10), 200) if len(examples) >= 10 else 0
    held_out, train = examples[:n_held], examples[n_held:]
    print(f"Training on {device}: {len(train)} examples (+{n_held} held out), "
          f"{len(corpus)} lines of text, {len(tokenizer)} {tokenizer.kind} tokens.")

    model = _torch_model(torch, len(tokenizer), block, d, n_layer, n_head, dropout=0.1).to(device)
    if old:
        from_numpy(torch, model, old.w, n_layer)
    print(f"Model: {n_layer} layers x {d} dims ({sum(p.numel() for p in model.parameters()):,} parameters)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    batches = Batches(tokenizer, train, corpus, block, rng, swap=args.swap)
    warmup, start, avg = min(200, args.steps // 10), time.time(), None
    report_every, window = max(1, args.steps // 20), []

    model.train()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / max(1, warmup))
        lr *= 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / args.steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        x, y, m = (torch.tensor(a).to(device) for a in batches.get(args.batch, args.lm_mix))
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.float().view(-1, logits.shape[-1]), y.view(-1), reduction="none")
            loss = (loss * m.view(-1)).sum() / m.sum().clamp(min=1)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        window.append(loss.item())
        if step % report_every == 0 or step == args.steps:
            avg, window = sum(window) / len(window), []
            print(f"step {step}/{args.steps}  loss {avg:.3f}  ({time.time() - start:.0f}s)", flush=True)

    model.eval()
    weights = to_numpy(model, n_layer)
    meta = dict(old.meta) if old else {}
    meta.update({"trained_steps": meta.get("trained_steps", 0) + args.steps,
                 "examples": len(train), "final_loss": round(avg or 0.0, 4),
                 "trained_on": time.strftime("%Y-%m-%d")})
    net = TinyTransformer(cfg, weights, tokenizer, meta)

    # The numpy copy must think exactly like the PyTorch model it came from.
    probe = (tokenizer.prompt("check", ["The numpy copy matches."], block // 2) + [END])[:block]
    with torch.no_grad():
        ref = model.float()(torch.tensor([probe]).to(device))[0].cpu().numpy()
    diff = float(np.abs(ref - net.forward(probe)).max())
    print(f"numpy/PyTorch agreement: max difference {diff:.2e}")
    if diff > 1e-2:
        raise SystemExit("The numpy copy doesn't match the trained model; not saving it.")

    if held_out:
        common, rare = name_stats(corpus + [e["answer"] for e in examples])
        report = evaluate(net, held_out, common=common, rare=rare)
        seen, new = report["seen"], report["new"]
        net.meta.update({"grounded_rate": seen["grounded"], "same_rate": seen["same"],
                         "new_names_grounded": new["grounded"], "new_names_same": new["same"]})
        print(f"Held-out questions ({seen['checked']}): {seen['grounded']:.0%} used only words from "
              f"their evidence; {seen['same']:.0%} matched the reference answer.")
        print(f"Same questions with NEW names it has never seen: {new['grounded']:.0%} grounded; "
              f"{new['same']:.0%} matched the reference. (This is the real test of reading evidence.)")
        for q, draft, ok in new["samples"]:
            print(f"  Q: {q}\n  A: {draft}  [{'grounded' if ok else 'NOT grounded'}]")
    net.save(args.out)
    print(f"Saved {args.out}. Load it with: /writer load {os.path.basename(args.out)}")
    return net


if __name__ == "__main__":
    main()
