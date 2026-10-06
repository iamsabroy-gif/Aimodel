"""Interactive chat that learns from you, your data and the web."""

from __future__ import annotations

import argparse
import os
import urllib.error

from .model import LearningModel
from .reports import run_user_quiz, show_curiosity, show_exam, show_progress, show_sleep

HELP = """\
Just type to chat or ask questions. I answer from what you taught me, from
data you gave me, or (if online) by reading Wikipedia, and I explain the
answer in my own words. Tell me facts ("My sister lives in Delhi") and I'll
remember them. If I don't know something, I'll ask you to teach me.

Teaching:
  /teach <prompt> => <response>   teach me a reply directly
  /good                           my last reply was good (I'll remember/trust it)
  /bad                            my last reply was wrong (you can correct me)
  /forget <text>                  forget replies whose prompt contains <text>

Greetings (hello, how are you, thanks, bye - I already know many):
  /greet                          show the greetings I know, your name and your greetings
  /greet <phrase> => <reply>      teach me a greeting or one more reply (any language;
                                  {{name}} = your name). e.g. /greet namaste => Namaste {{name}}!
  /greet forget <phrase>          forget a greeting you taught me
  Tell me "My name is Sam" and I'll greet you by name. /good or /bad after a greeting
  teaches me which replies you like.

Your data and the internet:
  /read <file or url>             learn from a text file or web page
  /web <question>                 look something up on Wikipedia and answer
  /online on|off                  allow automatic web lookups (now: {online})
  /why                            show my reasoning for the last answer
  /facts [topic]                  show facts I've learned (about a topic)

Checking how good I am (run it from time to time):
  /evaluate                       the standard scorecard: fixed tests, your brain and your writer;
                                  compares with the last run and saves it (about 20 seconds)
  /evaluate quick                 only the fixed tests (a second): "did a code change break anything?"
  /evaluate history               the trend over your past runs

Studying (how I learn like a child):
  /quiz [n]                       I take an exam on what I know and grade myself
  /quiz me [n]                    I quiz you instead
  /sleep                          consolidate: forget unused things, merge repeats, replay
  /curious                        study my open questions (shelf, dictionary, web)
  /study                          a full session: be curious, take an exam, then sleep
  /define <word>                  look a word up in the dictionary
  /progress                       how my studying is going

Your own transformer (the writer, trained on Kaggle):
  /export [folder]                save training data for the writer (default: writer_data)
  /writer                         show which writer words my answers
  /writer load <file>             use a writer you trained (writer.npz)
  /writer on|off                  switch the transformer writer on or off

Neural network:
  /train [epochs]                 give the network extra practice on all I know
  /similar <word>                 words the network thinks mean something similar
  /vectors builtin                use the small set of word vectors that ships with Aimodel (on by default)
  /vectors off                    stop using word vectors
  /vectors <file> [max words]     load pretrained word vectors (GloVe/fastText .txt)

Other:
  /gen [start words]              generate text in your style
  /stats                          show what I've learned
  /help                           show this help
  /quit                           save and exit
"""

_NET_ERRORS = (urllib.error.URLError, TimeoutError, OSError, ValueError)


def _show_reply(model: LearningModel, reply: str, confidence: float) -> None:
    print(f"ai> {reply}")
    if model.last_source in ("noted", "smalltalk"):
        return
    if model.last_source == "memory":
        where = "memory"
    else:
        sources = {s["source"] for s in model.last_trace
                   if s["source"] not in ("my reasoning", "my transformer")}
        where = f"reasoning over {len(sources)} source{'s' if len(sources) != 1 else ''}"
        if any(s["kind"] == "writer" and s["text"].startswith("Worded") for s in model.last_trace):
            where += ", worded by my transformer"
    print(f"    (from {where}, confidence {confidence:.2f}; /why, /good or /bad)")


# Jupyter raises a RuntimeError (StdinNotImplementedError) when nobody can type, e.g. in a saved run.
_NO_INPUT = (EOFError, KeyboardInterrupt, RuntimeError)


def ask(question: str) -> str:
    try:
        return input(question).strip()
    except _NO_INPUT:
        return ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="A tiny AI that learns from you.")
    parser.add_argument("--brain", default="brain.json", help="where to save what it learns")
    parser.add_argument("--offline", action="store_true", help="never look things up on the web")
    args = parser.parse_args(argv)

    model = LearningModel.load(args.brain)
    model.notify = lambda message: print("ai>", message)
    sibling = os.path.join(os.path.dirname(os.path.abspath(args.brain)), "writer.npz")
    if model.writer is None and os.path.exists(sibling):  # a writer you trained, next to your brain
        try:
            net = model.load_writer(sibling)
            print(f"Using your trained writer ({net.n_params:,} parameters) from {sibling}.")
        except (OSError, ValueError, KeyError) as e:
            print(f"(Found {sibling} but couldn't load it: {e})")
    online = not args.offline
    s = model.stats()
    print(f"Loaded brain from {args.brain}: {s['memories']} memories, {s['facts']} facts, "
          f"{s['knowledge sentences']} sentences, {s['neural vocabulary']} words in the network.")
    print(f"Web lookups are {'on' if online else 'off'}. Type /help for commands.\n")
    overnight = model.sleep_if_due()  # like sleeping: a day has passed since last time
    if overnight:
        print("ai> Good to see you again! Time passed, so I consolidated what I know.")
        show_sleep(overnight)
        print()

    queued = None  # a command typed at one of my questions, run next
    while True:
        if queued:
            text, queued = queued, None
        else:
            try:
                text = input("you> ").strip()
            except _NO_INPUT:
                print()
                break
        if not text:
            continue

        if text.startswith("/"):
            cmd, _, arg = text.partition(" ")
            arg = arg.strip()
            if cmd in ("/quit", "/exit"):
                break
            elif cmd == "/help":
                print(HELP.format(online="on" if online else "off"))
            elif cmd == "/teach":
                prompt, sep, response = arg.partition("=>")
                if not sep or not prompt.strip() or not response.strip():
                    print("usage: /teach <prompt> => <response>")
                    continue
                model.learn(prompt, response)
                print("ai> Got it, I learned that.")
            elif cmd == "/good":
                print("ai> Thanks! I'll remember that." if model.feedback(True)
                      else "ai> Nothing to rate yet.")
            elif cmd == "/bad":
                if model.last_reply is None:
                    print("ai> Nothing to rate yet.")
                    continue
                better = ask(f"   What should I have said to '{model.last_query}'? (enter to skip) ")
                if better.startswith("/"):
                    queued = better
                elif better:
                    model.correct(model.last_query, better)
                    print("ai> Thanks, I corrected myself.")
                else:
                    model.feedback(False)
                    print("ai> Okay, I'll trust that answer less.")
            elif cmd == "/read":
                if not arg:
                    print("usage: /read <file or url>")
                    continue
                try:
                    n = model.read(arg)
                except (FileNotFoundError, IsADirectoryError, PermissionError) as e:
                    print(f"ai> I couldn't open that file: {e}")
                    continue
                except _NET_ERRORS as e:
                    print(f"ai> I couldn't fetch that page: {e}")
                    continue
                read = model.last_read
                print(f"ai> I read {n} new sentences and trained my network on them."
                      if not read["shelved"] else
                      f"ai> I studied the {n} most important sentences and put {read['shelved']} "
                      "on the shelf to look up later if needed.")
            elif cmd == "/web":
                if not arg:
                    print("usage: /web <question>")
                    continue
                print("ai> Searching the web...")
                try:
                    n = model.search_web(arg)
                except _NET_ERRORS as e:
                    print(f"ai> The web search failed: {e}")
                    continue
                print(f"ai> I read {n} new sentences.")
                reply, confidence = model.respond(arg)
                if reply is not None:
                    _show_reply(model, reply, confidence)
                else:
                    print("ai> I still couldn't find a good answer to that.")
            elif cmd == "/online":
                if arg in ("on", "off"):
                    online = arg == "on"
                print(f"ai> Web lookups are {'on' if online else 'off'}.")
            elif cmd == "/why":
                if not model.last_trace:
                    print("ai> I haven't given an answer I can explain yet.")
                    continue
                print(f"ai> To answer '{model.last_query}' I used:")
                labels = {"fact": "fact", "evidence": "evidence", "inference": "reasoning",
                          "memory": "memory", "writer": "writer", "smalltalk": "greeting"}
                for i, step in enumerate(model.last_trace, 1):
                    print(f"  {i}. [{labels.get(step['kind'], step['kind'])}] {step['text']}")
                    detail = f"source: {step['source']}"
                    if "score" in step:
                        matched = ", ".join(step.get("matched", [])) or "similar meaning"
                        detail += f" | match {step['score']:.2f} on: {matched}"
                    if step["source"] != "my reasoning":
                        print(f"     {detail}")
            elif cmd == "/quiz":
                parts = arg.split()
                who = "me" if parts and parts[0] == "me" else "ai"
                nums = [p for p in parts if p.isdigit()]
                n = int(nums[0]) if nums else 5
                if who == "me":
                    run_user_quiz(model, n, ask)
                else:
                    show_exam(model.quiz(n))
            elif cmd == "/sleep":
                show_sleep(model.sleep())
            elif cmd == "/curious":
                print("ai> Let me look into the things I couldn't answer...")
                show_curiosity(model.be_curious(use_web=online))
            elif cmd == "/study":
                print("ai> Study session! First, my open questions...")
                result = model.study_session(use_web=online)
                show_curiosity(result["curiosity"])
                print("ai> Now an exam on what I know...")
                show_exam(result["exam"])
                print("ai> And now I sleep on it...")
                show_sleep(result["sleep"])
            elif cmd == "/define":
                if not arg:
                    print("usage: /define <word>")
                    continue
                try:
                    n = model.define(arg.split()[0])
                except _NET_ERRORS as e:
                    print(f"ai> I couldn't reach the dictionary: {e}")
                    continue
                if n:
                    reply, _ = model.respond(f"What is {arg.split()[0]}?", learn=False)
                    print(f"ai> {reply}")
                else:
                    print("ai> I couldn't find a definition for that.")
            elif cmd == "/greet":
                phrase, sep, reply = arg.partition("=>")
                if not arg:
                    info = model.greeting_summary()
                    kinds = ", ".join(info["built-in"])
                    print(f"ai> I know {len(info['built-in'])} kinds of small talk: {kinds}.")
                    print(f"    Your name: {info['name'] or 'not told yet (say: My name is ...)'}")
                    for item in info["custom"]:
                        print(f"    you taught me: {item['phrase']} => {item['reply']}")
                    if not info["custom"]:
                        print("    Teach me your own: /greet <phrase> => <reply>")
                elif arg.startswith("forget "):
                    n = model.forget_greeting(arg[7:])
                    print(f"ai> Forgot {n} greeting{'s' if n != 1 else ''}.")
                elif not sep:
                    print("usage: /greet <phrase> => <reply>    (or /greet, /greet forget <phrase>)")
                    continue
                else:
                    try:
                        model.add_greeting(phrase, reply)
                    except ValueError as e:
                        print(f"ai> I can't use that: {e}")
                        continue
                    print(f"ai> Learned. When you say '{phrase.strip()}' I'll say '{reply.strip()}'.")
            elif cmd == "/export":
                info = model.export_training_data(arg or "writer_data")
                print(f"ai> Saved {info['examples']} training examples and {info['corpus lines']} "
                      f"lines of text to {info['folder']}/")
                print("    " + ", ".join(f"{n} {k}" for k, n in info["by kind"].items()))
                print("    It holds your private notes: upload it to Kaggle as a PRIVATE dataset and")
                print("    train with: python -m aimodel.train_writer --data <folder> --out writer.npz")
            elif cmd == "/writer":
                parts = arg.split(maxsplit=1)
                if parts and parts[0] == "load" and len(parts) == 2:
                    try:
                        net = model.load_writer(parts[1])
                    except (OSError, ValueError, KeyError) as e:
                        print(f"ai> I couldn't load that writer: {e}")
                        continue
                    rate = net.meta.get("grounded_rate")
                    print(f"ai> Loaded a writer with {net.n_params:,} parameters, trained "
                          f"{net.meta.get('trained_steps', '?')} steps"
                          + (f"; {rate:.0%} of its test answers passed the evidence check." if rate is not None else "."))
                    print("    Its drafts are checked against the evidence; failing ones fall back to my templates.")
                elif parts and parts[0] in ("on", "off"):
                    model.writer_enabled = parts[0] == "on"
                    print(f"ai> Transformer writer is {parts[0]}.")
                elif model.writer is None:
                    print(f"ai> My answers are worded with templates. I've logged {len(model.answer_log)} "
                          "conversations to train a transformer on: /export, then train on Kaggle.")
                else:
                    used = sum(1 for e in model.answer_log if e.get("by") == "transformer")
                    print(f"ai> Writer: transformer with {model.writer.n_params:,} parameters "
                          f"({'on' if model.writer_enabled else 'off'}), trained "
                          f"{model.writer.meta.get('trained_steps', '?')} steps on "
                          f"{model.writer.meta.get('trained_on', '?')}. It worded {used} of my last "
                          f"{len(model.answer_log)} answers.")
            elif cmd == "/evaluate":
                from .evaluation import (evaluate_all, format_history, format_report, history_path,
                                         load_history, save_run)
                hist = history_path(args.brain)
                if arg == "history":
                    print(format_history(load_history(hist), last=15))
                else:
                    quick = arg == "quick"
                    print("ai> Running the standard tests" + (" (fixed tests only)..." if quick
                                                              else " (about 20 seconds)..."))
                    report = evaluate_all(model, args.brain, quick=quick)
                    runs = load_history(hist)
                    print(format_report(report, runs[-1] if runs else None))
                    save_run(hist, report)
                    print(f"\nSaved to {hist}. /evaluate history shows the trend.")
            elif cmd == "/progress":
                show_progress(model.progress())
            elif cmd == "/facts":
                topic = {w.lower() for w in arg.split()}
                from .reasoning import state, stem
                found = [f for f in model.facts
                         if not topic or {stem(t) for t in topic} & {stem(w) for w in f["subj"].lower().split() + f["obj"].lower().split()}]
                if not found:
                    print("ai> I don't know any facts about that yet.")
                for f in found[-20:]:
                    print(f"   - {state(f)}  ({f['source']})")
                if len(found) > 20:
                    print(f"   ... and {len(found) - 20} more")
            elif cmd == "/vectors":
                parts = arg.split()
                if not parts:
                    print("usage: /vectors builtin | off   or   /vectors <glove or fastText .txt/.zip file> [max words]")
                    continue
                if parts[0] == "off":
                    model.vectors_off()
                    print("ai> Word vectors are off. /vectors builtin turns them back on.")
                    continue
                if parts[0] == "builtin":
                    n = model.load_builtin_vectors()
                    print(f"ai> Loaded {n} pretrained words. I now understand word meanings much better."
                          if n else "ai> The bundled word vectors are missing from this copy.")
                    continue
                print("ai> Loading word vectors (this can take a minute)...")
                try:
                    n = model.load_vectors(parts[0], int(parts[1]) if len(parts) > 1 else 50_000)
                except (OSError, ValueError, StopIteration) as e:
                    print(f"ai> I couldn't load those vectors: {e}")
                    continue
                print(f"ai> Loaded {n} pretrained words. I now understand word meanings much better.")
            elif cmd == "/train":
                epochs = int(arg) if arg.isdigit() else 5
                loss = model.retrain(epochs)
                print("ai> Nothing to train on yet." if loss is None
                      else f"ai> Trained {epochs} epochs, loss {loss:.3f} (lower is better).")
            elif cmd == "/similar":
                pairs = model.neural.similar(arg.lower())
                print("ai>", ", ".join(f"{w} ({s:.2f})" for w, s in pairs) if pairs
                      else "ai> I haven't seen that word yet.")
            elif cmd == "/gen":
                print("ai>", model.generate(arg) or "(I need more of your text first)")
            elif cmd == "/forget":
                if not arg:
                    print("usage: /forget <text>")
                    continue
                print(f"ai> Forgot {model.forget(arg)} memories.")
            elif cmd == "/stats":
                for k, v in model.stats().items():
                    print(f"   {k}: {v}")
            else:
                print("Unknown command. Type /help.")
            model.save(args.brain)
            continue

        try:
            reply, confidence = model.respond(text, use_web=online)
        except _NET_ERRORS as e:
            print(f"ai> (web lookup failed: {e})")
            reply, confidence = model.respond(text)
        if reply is not None:
            _show_reply(model, reply, confidence)
        else:
            taught = ask("ai> I don't know how to reply yet. What should I say? (enter to skip) ")
            if taught.startswith("/"):
                queued = taught
            elif taught:
                model.learn(text, taught)
                print("ai> Learned it!")
        model.save(args.brain)

    model.save(args.brain)
    print(f"Saved to {args.brain}. Bye!")


if __name__ == "__main__":
    main()
