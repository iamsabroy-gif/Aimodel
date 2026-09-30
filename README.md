# Aimodel — a tiny AI that learns from you

A small AI model (Python 3.9+, only needs numpy) that starts knowing nothing and
learns from **you**, from **your data**, and from **the internet**.

## Run it

```bash
pip install -r requirements.txt
python3 -m aimodel                 # saves what it learns to brain.json (+ brain.neural.npz)
python3 -m aimodel --offline       # never look things up on the web
python3 -m aimodel --brain me.json # use a different brain file
```

## Run it on Android

**Pydroid 3 (easiest):** install *Pydroid 3* from the Play Store, then
Menu → Pip → install `numpy`. Download this repo as a ZIP (Code → Download
ZIP), unzip it, open `run.py` in Pydroid and press ▶.

**Termux:** install Termux from F-Droid (not the Play Store), then:

```bash
pkg update && pkg install python python-numpy git
git clone -b ccr-34fc447e-og7rx9 https://github.com/iamsabroy-gif/Aimodel.git
cd Aimodel && python run.py
```

Run `termux-setup-storage` once to read your files, e.g.
`/read /sdcard/Download/notes.txt`. Use `pkg install python-numpy`, not
`pip install numpy` (pip tries to build numpy on the phone).

## Run it on Kaggle

Create a notebook and turn **Internet on** (Session options). Then:

```python
!git clone -b ccr-34fc447e-og7rx9 https://github.com/iamsabroy-gif/Aimodel.git
%cd Aimodel

from aimodel.cli import main                # interactive chat; /quit to stop
main(["--brain", "/kaggle/working/brain.json"])
```

Or use it from code (see *Use it from code* below). Upload your `.txt` files
with **+ Add Input** and `m.read("/kaggle/input/<dataset>/notes.txt")`.
Kaggle wipes the session when it ends, so download `brain.json` and
`brain.neural.npz` from `/kaggle/working/`, or save them as a dataset and copy
them back next time.

Your brain files work anywhere: train on Kaggle, keep using them on your phone.

## How it works

When you type something, it:

1. **Checks what you taught it:** replies you taught it, matched by meaning.
2. **Remembers facts you state:** "My sister lives in Delhi" becomes a fact.
   Later, "Where does my sister live?" gets "Your sister lives in Delhi."
3. **Reasons over what it knows** (your files, pages it read, facts) and
   writes an answer in its own words.
4. **Looks things up:** for a question it can't answer, it reads Wikipedia
   and tries again (if web lookups are on).
5. **Asks you:** if it still doesn't know, you teach it.

### Reasoning and its own explanations

Every sentence it reads is turned into simple **facts** (subject → verb →
object), e.g. "Cats are mammals" → `Cats | are | mammals`. It uses them to:

- **Prove things it was never told** by chaining facts:
  *Is a cat a living thing?* → "Yes. Cats are mammals, and mammals are
  warm-blooded animals, and animals are living things, so a cat is a living thing."
- **Say no** when a fact rules it out: *Is a snake a mammal?* → "No. Snakes are not mammals."
- **Explain why** by finding a stated cause: *Why is the sky blue?* →
  "The reason is that air molecules scatter blue sunlight more than red sunlight."
- **Explain how** as ordered steps: *How do I make tea?* → "Here's how:
  First, … Then, … Finally, …"
- **Connect facts:** *What is my dog called?* → "Your dog is called Rex,
  which is a golden retriever."
- **Talk to you about you:** "I", "my" in your notes become "you", "your".

`/why` shows every step: which facts and evidence it used, where each came
from, and what it inferred. `/facts <topic>` lists what it knows.

### Neural networks

- **Word-meaning network** (`neural.py`): a word2vec (skip-gram) neural
  network written in numpy. It trains on everything it reads and learns which
  words are related (`/similar <word>`, `/train`). It **grows automatically**:
  once it knows 8,000 words it rebuilds itself with bigger vectors (48 → 96),
  and again at 30,000 words (→ 160). Its opinion counts more as it reads more
  (full trust after about 50,000 words), so a young network can't cause wrong
  matches.
- **Feedback network** (`ranker.py`): a small neural network that learns from
  your `/good` and `/bad` which kind of evidence makes a good answer, so its
  choices move toward your taste.
- **Pretrained word vectors** (optional, `vectors.py`): load GloVe or fastText
  vectors with `/vectors <file>` to give it knowledge of billions of words from
  day one (e.g. that "job" ≈ "work"). Get `glove.6B.50d.txt` from
  <https://nlp.stanford.edu/projects/glove/>, or on Kaggle add the
  "glove6b50dtxt" dataset and run
  `/vectors /kaggle/input/glove6b50dtxt/glove.6B.50d.txt`.

## Commands

| Command | What it does |
|---|---|
| *(just type)* | chat or ask a question |
| `/teach hi => Hello!` | teach a reply directly |
| `/good` | trust the last reply more (a good researched answer becomes a memory) |
| `/bad` | trust it less and give a better answer |
| `/read <file or url>` | learn from a text file or web page |
| `/web <question>` | look it up on Wikipedia now |
| `/online on\|off` | allow automatic web lookups |
| `/why` | show the reasoning behind the last answer |
| `/facts [topic]` | list facts it has learned |
| `/vectors <file> [max words]` | load pretrained word vectors (GloVe / fastText) |
| `/train [epochs]` | extra neural network training on everything it knows |
| `/similar <word>` | words the network thinks are related |
| `/gen [words]` | generate text in your writing style |
| `/forget <text>`, `/stats`, `/help`, `/quit` | |

Everything is saved after every step, so it keeps learning between sessions.

## Example

```
you> /read notes.txt
ai> I read 5 new sentences and trained my network on them.
you> My sister lives in Delhi
ai> Got it: Your sister lives in Delhi.
you> Is a cat a living thing?
ai> Yes. Cats are mammals, and mammals are warm-blooded animals, and animals are
    living things, so a cat is a living thing.
    (from reasoning over 1 source, confidence 0.80; /why, /good or /bad)
you> /why
ai> To answer 'Is a cat a living thing?' I used:
  1. [fact] Cats are mammals.
     source: notes.txt
  2. [fact] Mammals are warm-blooded animals.
     source: notes.txt
  3. [fact] Animals are living things.
     source: notes.txt
  4. [reasoning] Chained 3 facts: cat -> mammal -> animal -> thing
```

## Limits (honestly)

This is a *small* model that runs on a phone. Its explanations are built from
facts and sentences it has read, combined with its own sentence patterns and
reasoning steps. It does not invent new ideas or write long essays like a
large language model (those have billions of parameters). Fact extraction
uses simple grammar patterns, so complicated sentences are used as evidence
rather than as facts. It gets better the more you feed it: `/read` your data,
let it read the web, load pretrained vectors, and give `/good` / `/bad` feedback.

## Use it from code

```python
from aimodel import LearningModel

m = LearningModel.load("brain.json")
m.read("notes.txt")                          # your data
m.learn("what's your name", "I'm Sage.")    # teach it
reply, confidence = m.respond("What is photosynthesis?", use_web=True)
print(reply, m.last_trace)                   # answer + reasoning steps
m.respond("My sister lives in Delhi")         # it learns facts you state
m.load_vectors("glove.6B.50d.txt")           # optional word knowledge
m.feedback(good=True)
m.save("brain.json")
```

## Tests

```bash
python3 -m unittest    # runs offline; the internet is faked in tests
```
