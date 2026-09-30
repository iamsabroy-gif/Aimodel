"""Interactive chat that learns from you."""

from __future__ import annotations

import argparse

from .model import LearningModel

HELP = """\
Just type to chat. If I don't know how to reply, I'll ask you to teach me.

Commands:
  /teach <prompt> => <response>   teach me a reply directly
  /good                           my last reply was good (I'll trust it more)
  /bad                            my last reply was wrong (you can correct me)
  /gen [start words]              generate text in your style
  /forget <text>                  forget replies whose prompt contains <text>
  /stats                          show what I've learned
  /help                           show this help
  /quit                           save and exit
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="A tiny AI that learns from you.")
    parser.add_argument("--brain", default="brain.json", help="where to save what it learns")
    args = parser.parse_args(argv)

    model = LearningModel.load(args.brain)
    s = model.stats()
    print(f"Loaded brain from {args.brain}: {s['memories']} memories, {s['vocabulary']} words.")
    print("Type /help for commands.\n")

    last_input = None
    while True:
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
                print(HELP)
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
                if last_input is None or model.last_match is None:
                    print("ai> Nothing to rate yet.")
                    continue
                better = input(f"   What should I have said to '{last_input}'? (enter to skip) ").strip()
                if better:
                    model.correct(last_input, better)
                    print("ai> Thanks, I corrected myself.")
                else:
                    model.feedback(False)
                    print("ai> Okay, I'll trust that answer less.")
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

        last_input = text
        reply, confidence = model.respond(text)
        if reply is not None:
            print(f"ai> {reply}   (confidence {confidence:.2f}; /good or /bad)")
        else:
            taught = input("ai> I don't know how to reply yet. What should I say? (enter to skip) ").strip()
            if taught:
                model.learn(text, taught)
                print("ai> Learned it!")
        model.save(args.brain)

    model.save(args.brain)
    print(f"Saved to {args.brain}. Bye!")


if __name__ == "__main__":
    main()
