"""Turn Hugging Face datasets into files this app can read (it cannot read Parquet).

    python -m aimodel.hf_convert sciq              --out hf_data
    python -m aimodel.hf_convert squad             --out hf_data --limit 20
    python -m aimodel.hf_convert simple-wikipedia  --out hf_data --limit 60
    python -m aimodel.hf_convert simple-wikipedia  --out hf_data --titles starter     # ~140 broad topics
    python -m aimodel.hf_convert simple-wikipedia  --out hf_data --titles "Moon,Tea,Bicycle"
    python -m aimodel.hf_convert tinystories       --out hf_data --limit 3

Each writes plain .txt study files (upload them on the Train tab) and, where the dataset has questions,
an `eval/` folder that `python -m aimodel.devset hf_data/<name>/eval` can score. Question and answer
pairs are NOT turned into taught replies: those match by similarity and answer the wrong question, so only
the passages are for studying and the questions are for testing.

Needs pyarrow for the Parquet datasets (pip install pyarrow). The files are downloaded from huggingface.co
once and kept in <out>/_cache.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import urllib.request

HF = "https://huggingface.co/datasets/"
FILES = {
    "sciq": {"train": "allenai/sciq/resolve/main/data/train-00000-of-00001.parquet",
             "validation": "allenai/sciq/resolve/main/data/validation-00000-of-00001.parquet"},
    "squad": {"train": "rajpurkar/squad/resolve/main/plain_text/train-00000-of-00001.parquet",
              "validation": "rajpurkar/squad/resolve/main/plain_text/validation-00000-of-00001.parquet"},
    "simple-wikipedia": {"train": "wikimedia/wikipedia/resolve/main/20231101.simple/train-00000-of-00001.parquet"},
    "tinystories": {"valid": "roneneldan/TinyStories/resolve/main/TinyStories-valid.txt"},
}
LICENSES = {
    "sciq": "SciQ (allenai/sciq), CC BY-NC 3.0: non-commercial use only.",
    "squad": "SQuAD (rajpurkar/squad), CC BY-SA 4.0: Wikipedia text, share alike with attribution.",
    "simple-wikipedia": "Simple English Wikipedia (wikimedia/wikipedia), CC BY-SA 3.0 and GFDL: share alike with attribution.",
    "tinystories": "TinyStories (roneneldan/TinyStories), CDLA-Sharing 1.0.",
}
# Broad, well-known topics: a better start than a random sample of the 200,000 Simple English articles.
STARTER = ("Sun,Moon,Earth,Mars,Jupiter,Saturn,Venus,Solar System,Star,Galaxy,Water,Air,Fire,Electricity,Gravity,Atom,"
           "Energy,Light,Sound,Magnet,Photosynthesis,Plant,Tree,Flower,Animal,Mammal,Bird,Fish,Insect,Reptile,"
           "Dinosaur,Elephant,Whale,Tiger,Lion,Dog,Cat,Horse,Honey bee,Ant,Spider,Shark,Penguin,Heart,Brain,Lung,"
           "Skeleton,Muscle,Blood,Skin,Eye,Ear,Tooth,Stomach,Volcano,Earthquake,Ocean,River,Lake,Mountain,Desert,"
           "Rainforest,Glacier,Weather,Climate,Cloud,Rain,Wind,Season,Continent,Africa,Asia,Europe,Australia,"
           "North America,South America,Antarctica,France,Germany,India,China,Japan,Egypt,Brazil,Canada,Russia,"
           "Italy,Spain,United Kingdom,United States,Mexico,Roman Empire,Ancient Egypt,Ancient Greece,Middle Ages,"
           "Renaissance,Industrial Revolution,World War I,World War II,Albert Einstein,Isaac Newton,Charles Darwin,"
           "Leonardo da Vinci,Mahatma Gandhi,Computer,Internet,Telephone,Television,Bicycle,Car,Train,Airplane,Ship,"
           "Tea,Coffee,Bread,Rice,Sugar,Salt,Milk,Chocolate,Music,Piano,Guitar,Football,Olympic Games,Language,Book,"
           "Mathematics,Number,Medicine,Vaccine,Money,Democracy").split(",")
CHUNK_CHARS = 700_000  # about 14 seconds to study: small enough to upload in pieces


def download(path: str, cache: str) -> str:
    target = os.path.join(cache, path.replace("/", "__"))
    if not os.path.exists(target):
        os.makedirs(cache, exist_ok=True)
        print("downloading", HF + path)
        req = urllib.request.Request(HF + path, headers={"User-Agent": "aimodel-hf-convert"})
        with urllib.request.urlopen(req, timeout=120) as r, open(target + ".part", "wb") as f:
            while chunk := r.read(1 << 20):
                f.write(chunk)
        os.replace(target + ".part", target)
    return target


def rows(path: str) -> list[dict]:
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("Reading Parquet needs pyarrow: run  pip install pyarrow  and try again.") from None
    return pq.read_table(path).to_pylist()


def squeeze(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def write_chunks(folder: str, prefix: str, paragraphs: list[str], size: int = CHUNK_CHARS) -> list[str]:
    os.makedirs(folder, exist_ok=True)
    names, current, length = [], [], 0

    def flush():
        nonlocal current, length
        if current:
            name = f"{prefix}_{len(names) + 1:03d}.txt"
            with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
                f.write("\n\n".join(current))
            names.append(name)
            current, length = [], 0
    for p in paragraphs:
        if length + len(p) > size:
            flush()
        current.append(p)
        length += len(p) + 2
    flush()
    return names


def write_questions(folder: str, items: list) -> None:
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "questions.json"), "w", encoding="utf-8") as f:
        f.write("[\n" + ",\n".join(" " + json.dumps(i, ensure_ascii=False) for i in items) + "\n]\n")


def words(answer: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", answer.lower())


def sciq(out: str, cache: str, limit: int | None, seed: int = 0) -> dict:
    train, val = (rows(download(FILES["sciq"][k], cache)) for k in ("train", "validation"))
    seen, paragraphs = set(), []
    for r in train:
        p = squeeze(r["support"])
        if p and p not in seen:
            seen.add(p)
            paragraphs.append(p)
    if limit:
        paragraphs = paragraphs[:limit * 40]
    names = write_chunks(os.path.join(out, "sciq"), "SciQ_support", paragraphs)
    usable = [r for r in val if squeeze(r["support"]) and r["correct_answer"].lower() in squeeze(r["support"]).lower()]
    random.Random(seed).shuffle(usable)
    test = usable[:150]
    folder = os.path.join(out, "sciq", "eval")
    write_chunks(folder, "SciQ_eval", list(dict.fromkeys(squeeze(r["support"]) for r in test)))
    write_questions(folder, [["lookup", r["question"], words(r["correct_answer"]), {}] for r in test])
    return {"study files": names, "paragraphs": len(paragraphs), "test questions": len(test)}


def squad(out: str, cache: str, limit: int | None) -> dict:
    train, val = (rows(download(FILES["squad"][k], cache)) for k in ("train", "validation"))

    def by_title(data):
        articles: dict[str, list[str]] = {}
        for r in data:
            ctx = squeeze(r["context"])
            if ctx not in articles.setdefault(r["title"], []):
                articles[r["title"]].append(ctx)
        return articles
    articles = by_title(train)
    chosen = list(articles)[:limit or 20]
    folder = os.path.join(out, "squad")
    os.makedirs(folder, exist_ok=True)
    for title in chosen:  # one file per article: the file name is the topic the model files it under
        with open(os.path.join(folder, title + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(articles[title]))
    test_articles = by_title(val)
    pick = list(test_articles)[:8]
    ev = os.path.join(folder, "eval")
    os.makedirs(ev, exist_ok=True)
    for title in pick:
        with open(os.path.join(ev, title + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(test_articles[title]))
    qs = [["lookup", r["question"], words(r["answers"]["text"][0]), {}] for r in val
          if r["title"] in pick and r["answers"]["text"]]
    write_questions(ev, qs[:150])
    return {"study files": len(chosen), "test questions": min(150, len(qs))}


def simple_wikipedia(out: str, cache: str, limit: int | None, titles: list[str] | None, seed: int = 0) -> dict:
    data = rows(download(FILES["simple-wikipedia"]["train"], cache))
    if titles:
        wanted = {t.lower() for t in (STARTER if titles == ["starter"] else titles)}
        chosen = [r for r in data if r["title"].lower() in wanted]
    else:
        pool = [r for r in data if 2500 <= len(r["text"]) <= 20000 and ":" not in r["title"]
                and not r["title"].lower().startswith(("list of", "years in"))]
        random.Random(seed).shuffle(pool)
        chosen = pool[:limit or 60]
    folder = os.path.join(out, "simple-wikipedia")
    os.makedirs(folder, exist_ok=True)
    written = []
    for r in chosen:
        lines = [squeeze(l) for l in r["text"].splitlines()]
        keep = [l for l in lines if l and (l.endswith((".", "!", "?", '"')) or len(l.split()) >= 8)]
        if sum(len(l) for l in keep) < 300:  # a redirect or a stub: nothing to study
            continue
        name = re.sub(r"[^\w .-]", "", r["title"]).strip().replace(" ", "_")
        with open(os.path.join(folder, name + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n\n".join(keep))
        written.append(r["title"])
    info = {"study files": len(written), "titles": written[:12]}
    if titles:
        lower = {w.lower() for w in written}
        info["not found or too short"] = [t for t in (STARTER if titles == ["starter"] else titles) if t.lower() not in lower]
    return info


def tinystories(out: str, cache: str, limit: int | None) -> dict:
    path = download(FILES["tinystories"]["valid"], cache)
    with open(path, encoding="utf-8") as f:
        stories = [squeeze(s) for s in f.read().split("<|endoftext|>") if s.strip()]
    names = write_chunks(os.path.join(out, "tinystories"), "TinyStories", stories)[:limit or 3]
    return {"study files": names, "stories": len(stories)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("dataset", choices=sorted(FILES))
    ap.add_argument("--out", default="hf_data")
    ap.add_argument("--limit", type=int, help="how many articles (or chunks) to prepare")
    ap.add_argument("--titles", help="simple-wikipedia: comma-separated article titles to prepare")
    args = ap.parse_args(argv)
    cache = os.path.join(args.out, "_cache")
    if args.dataset == "sciq":
        info = sciq(args.out, cache, args.limit)
    elif args.dataset == "squad":
        info = squad(args.out, cache, args.limit)
    elif args.dataset == "simple-wikipedia":
        info = simple_wikipedia(args.out, cache, args.limit, args.titles.split(",") if args.titles else None)
    else:
        info = tinystories(args.out, cache, args.limit)
    os.makedirs(os.path.join(args.out, args.dataset), exist_ok=True)
    with open(os.path.join(args.out, args.dataset, "LICENSE.txt"), "w", encoding="utf-8") as f:
        f.write(LICENSES[args.dataset] + "\nKeep this notice with the files.\n")
    print(json.dumps(info, indent=1, ensure_ascii=False))
    print("Files are in", os.path.join(args.out, args.dataset))


if __name__ == "__main__":
    main()
