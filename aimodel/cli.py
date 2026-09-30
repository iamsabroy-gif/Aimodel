"""Interactive chat that learns from you, your data and the web."""

from __future__ import annotations

import argparse
import urllib.error

from .model import LearningModel

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

Your data and the internet:
  /read <file or url>             learn from a text file or web page
  /web <question>                 look something up on Wikipedia and answer
  /online on|off                  allow automatic web lookups (now: {online})
  /why                            show my reasoning for the last answer
  /facts [topic]                  show facts I've learned (about a topic)

Neural network:
  /train [epochs]                 give the network extra practice on all I know
  /similar <word>                 words the network thinks mean something similar
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
    if model.last_source == "noted":
        return
    if model.last_source == "memory":
        where = "memory"
    else:
        sources = {s["source"] for s in model.last_trace if s["source"] != "my reasoning"}
        where = f"reasoning over {len(sources)} source{'s' if len(sources) != 1 else ''}"
    print(f"    (from {where}, confidence {confidence:.2f}; /why, /good or /bad)")


def ask(question: str) -> str:
    try:
        return input(question).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="A tiny AI that learns from you.")
    parser.add_argument("--brain", default="brain.json", help="where to save what it learns")
    parser.add_argument("--offline", action="store_true", help="never look things up on the web")
    args = parser.parse_args(argv)

    model = LearningModel.load(args.brain)
    model.notify = lambda message: print("ai>", message)
    online = not args.offline
    s = model.stats()
    print(f"Loaded brain from {args.brain}: {s['memories']} memories, {s['facts']} facts, "
          f"{s['knowledge sentences']} sentences, {s['neural vocabulary']} words in the network.")
    print(f"Web lookups are {'on' if online else 'off'}. Type /help for commands.\n")

    queued = None  # a command typed at one of my questions, run next
    while True:
        if queued:
            text, queued = queued, None
        else:
            try:
                text = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
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
                print(f"ai> I read {n} new sentences and trained my network on them.")
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
                          "memory": "memory"}
                for i, step in enumerate(model.last_trace, 1):
                    print(f"  {i}. [{labels.get(step['kind'], step['kind'])}] {step['text']}")
                    detail = f"source: {step['source']}"
                    if "score" in step:
                        matched = ", ".join(step.get("matched", [])) or "similar meaning"
                        detail += f" | match {step['score']:.2f} on: {matched}"
                    if step["source"] != "my reasoning":
                        print(f"     {detail}")
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
                    print("usage: /vectors <glove or fastText .txt/.zip file> [max words]")
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
