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

## How it works

When you ask something, it tries, in order:

1. **What you taught it:** replies you taught it, matched by meaning.
2. **Your data and what it has read:** sentences from files and pages you gave it.
3. **The internet:** if it's a question and web lookups are on, it searches
   Wikipedia, reads the top articles, learns from them and answers.
4. **You:** if it still doesn't know, it asks you and remembers your answer.

### The neural network

`aimodel/neural.py` is a small **word2vec (skip-gram + negative sampling)
neural network** written in numpy. It trains on everything the model sees (your
messages, your files, web pages) and learns a vector for every word, so that
words used in similar ways end up close together. Sentence vectors built from
these let it match questions to answers by *meaning*, not just by exact words.
Try `/similar <word>` to see what it has learned, and `/train` to give it
extra practice.

### Reasoning over evidence

To answer from what it has read, it ranks sentences (TF-IDF + neural
similarity), picks the best one, then adds sentences that cover the parts of
your question the answer doesn't cover yet, so it can combine facts from
different places. It only answers when the evidence covers at least half of
what you asked about. `/why` shows each piece of evidence, where it came
from, and which words matched.

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
| `/why` | show the evidence behind the last answer |
| `/train [epochs]` | extra neural network training on everything it knows |
| `/similar <word>` | words the network thinks are related |
| `/gen [words]` | generate text in your writing style |
| `/forget <text>`, `/stats`, `/help`, `/quit` | |

Everything is saved after every step, so it keeps learning between sessions.

## Example

```
you> /read notes.txt
ai> I read 3 new sentences and trained my network on them.
you> who is my friend?
ai> I go running every Saturday morning with my friend Ravi.
    (from reading, confidence 0.38; /why, /good or /bad)
you> /why
ai> To answer 'who is my friend?' I used:
  1. I go running every Saturday morning with my friend Ravi.
     source: notes.txt | match 0.38 on: friend
```

## Limits (honestly)

This is a *small* model. It answers by finding and combining sentences it has
read; it does not write new explanations like a large language model. The
neural network gets better with more text: with only a few sentences it
can't yet know that, say, "job" and "work" mean similar things. Feed it more
of your data (`/read`) and let it read the web to improve it.

## Use it from code

```python
from aimodel import LearningModel

m = LearningModel.load("brain.json")
m.read("notes.txt")                          # your data
m.learn("what's your name", "I'm Sage.")    # teach it
reply, confidence = m.respond("What is photosynthesis?", use_web=True)
print(reply, m.last_trace)                   # answer + evidence
m.feedback(good=True)
m.save("brain.json")
```

## Tests

```bash
python3 -m unittest    # runs offline; the internet is faked in tests
```
