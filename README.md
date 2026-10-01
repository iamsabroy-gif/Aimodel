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

### Study mode: learning like a child

Reading isn't learning. A child sorts what matters, looks up words, gets
tested, forgets what they never use and sleeps on the rest. `study.py` does the same:

| A child... | The model... |
|---|---|
| Doesn't memorise a whole book | **Sorts by importance.** A long web page is studied only for its important sentences (the opening, definitions, ideas that recur, your interests, the question you asked). The rest go on a **shelf** and are looked up when a question needs them |
| Remembers what is used and confirmed, forgets the rest | **Memory strength.** Every sentence has a strength: it rises when used, confirmed (`/good`) or recalled in an exam and fades if never used. After about two weeks unused web text moves to the shelf. **What you taught or gave it is never forgotten** |
| Reviews at growing intervals | **Spaced repetition.** Each correct recall doubles the wait before that fact is due again; a miss brings it back tomorrow |
| Looks up words in a dictionary | **Dictionary.** Unknown words in your question are looked up and learned (`/define <word>`) |
| Takes exams | **`/quiz`**: it writes questions from its facts, answers them *from memory* (no peeking at the source) and grades itself. `/quiz me` quizzes you |
| Sleeps on it | **`/sleep`** consolidates: fades unused things, merges repeats, settles contradictions (keeps what you said over the web, asks you when it can't tell) and replays what it studied. It also runs by itself after a day away |
| Updates what they know | Tell it "My sister lives in Mumbai" and it replaces "Delhi" (but "I like tea" and "I like coffee" are both kept) |
| Gets curious about what it missed | Questions it couldn't answer and exam misses become **study goals**. `/curious` reads up on them (shelf, dictionary, web) and checks it can now answer. `/study` does a whole school day: curiosity, an exam, then sleep |

`/progress` shows how it's going, including exam scores over time.

### Your own transformer: the writer

Everything above decides **what** is true. The writer decides **how to say it**.
Out of the box, answers are worded with sentence templates. You can train your
own tiny transformer (GPT-style, about 3 million parameters) to word them instead.
It is trained on **your** data on **Kaggle**, and it runs on your phone with numpy
only. Nothing is sent to any AI company.

```
question -> memory, facts, reasoning, study -> evidence -> [ transformer writes ] -> check -> answer
```

- **How it learns** (`train_writer.py`): it practises predicting the next piece of
  text it has read (language), and writing answers from evidence (only the answer is
  scored). Text is split into **BPE tokens** like GPT: common pieces such as
  " mammal" become one token, so it trains faster and copies names more easily.
- **What it learns from** (`/export`): your real questions and its answers, your
  `/good` ratings (weight 2) and `/bad` corrections (weight 3, the strongest
  signal), practice questions written from everything it knows, and replies you taught it.
- **It reads instead of reciting.** During training, names in the examples are
  randomly swapped for made-up words (consistently in question, evidence and
  answer). Memorising never pays, so it learns to copy facts *from the evidence*.
- **It can't make things up or leave things out.** Every draft is checked: each
  meaningful word must appear in the question or the evidence, and it must say
  (almost) everything the template answer says. Otherwise the template answer is
  used. `/why` says which writer worded the answer, or why a draft was rejected.
- **The model** (`transformer.py`): token and position embeddings, then layers of
  masked multi-head self-attention and a feed-forward network, each with layer
  norm and a residual connection, then a softmax over the vocabulary. It writes one
  token at a time and caches past keys and values so each new token is cheap.

**Train it on Kaggle (gradually):**

1. In Aimodel: `/export` creates a `writer_data` folder.
2. On Kaggle: **Create → New Dataset**, upload the folder's files, keep it **Private**.
3. Open `notebooks/train_writer_on_kaggle.ipynb` on Kaggle (File → Import Notebook), add
   your dataset as an input, turn on the **GPU** and **Internet**, then **Run all**.
4. Download `writer.npz` and in Aimodel type `/writer load writer.npz`.
5. Later, after teaching it more: export again and train with
   `--resume writer.npz` so it keeps what it already learned.

**What to expect.** In a test on this repo (a CPU, the smallest size, 6,000
steps, about 650 examples about made-up animals), the writer answered questions
about animals it had never seen: 97% of its answers used only words from their
evidence and 90% were word-for-word what the template would say. Its misses
left things out rather than inventing them. Before name swapping, a higher
learning rate and enough steps, the same model fluently wrote about the *wrong*
animal every time: the evidence check caught all of those. At first it imitates
the templates. It moves beyond them through your `/good` ratings and `/bad`
corrections, which count two and three times as much in training. The Kaggle
notebook trains the larger `base` size on a GPU (about 3M parameters).

| Command | What it does |
|---|---|
| `/export [folder]` | save training data for the writer |
| `/writer` | which writer is wording answers, and how often the transformer passes the check |
| `/writer load <file>` | use a writer you trained |
| `/writer on` / `off` | switch it on or off |

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
| `/quiz [n]` / `/quiz me [n]` | it takes an exam and grades itself / it quizzes you |
| `/sleep` | consolidate: forget unused, merge repeats, settle contradictions, replay |
| `/curious` | study its open questions (shelf, dictionary, web) |
| `/study` | a full session: be curious, take an exam, sleep |
| `/define <word>` | look a word up in the dictionary |
| `/progress` | how studying is going |
| `/export [folder]` | save training data for your transformer writer |
| `/writer [load <file> \| on \| off]` | use a writer you trained on Kaggle |
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
you> /quiz 3
  [OK ] What are cats?
         I said: Cats are mammals, which are warm-blooded animals.
  ...
ai> Self-exam: 3/3 correct.
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
reasoning steps. The transformer writer only rewords answers the reasoning has
already found; it doesn't add knowledge or reasoning of its own. It does not invent new ideas or write long essays like a
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
exam = m.quiz(5)                             # it quizzes itself: exam["score"], exam["results"]
m.sleep()                                    # consolidate (forget unused, merge, replay)
m.study_session(use_web=True)                # curiosity + exam + sleep
m.feedback(good=True)
m.save("brain.json")
```

## Tests

```bash
python3 -m unittest    # runs offline; the internet is faked in tests
```

The training test runs only where PyTorch is installed (it is on Kaggle). Training
needs PyTorch; using a trained writer needs only numpy.
