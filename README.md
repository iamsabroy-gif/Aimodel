# Aimodel — a tiny AI that learns from you

A small, dependency-free AI model (pure Python 3.9+) that starts knowing nothing
and learns entirely from your inputs.

## Run it

```bash
python3 -m aimodel                 # saves what it learns to brain.json
python3 -m aimodel --brain me.json # use a different brain file
```

## How it learns

| You do | It learns |
|---|---|
| Chat normally | Replies with the closest thing you've taught it (TF-IDF similarity, so different wording still matches) |
| It doesn't know an answer | It asks you what to say and remembers your answer |
| `/teach hi => Hello!` | Learns a reply directly |
| `/good` | Trusts that reply more |
| `/bad` | Trusts it less and lets you give a better answer |
| Anything you type | Trains a word-level Markov model of your writing style, so `/gen` writes text that sounds like you |

Other commands: `/forget <text>`, `/stats`, `/help`, `/quit`.
Everything is saved after every step, so it keeps learning between sessions.

## Example

```
you> hello
ai> I don't know how to reply yet. What should I say? Hi there, friend!
ai> Learned it!
you> hey hello
ai> Hi there, friend!   (confidence 0.88; /good or /bad)
you> /good
ai> Thanks! I'll remember that.
```

## Use it from code

```python
from aimodel import LearningModel

m = LearningModel.load("brain.json")
m.learn("what's your name", "I'm Sage.")
print(m.respond("what is your name?"))  # ("I'm Sage.", 0.8...)
m.feedback(good=True)
print(m.generate("I"))
m.save("brain.json")
```

## Tests

```bash
python3 -m unittest
```
