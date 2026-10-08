"""Printing helpers for the study commands (kept out of cli.py to keep it readable)."""

from __future__ import annotations


def show_exam(result: dict) -> None:
    if not result["total"]:
        print("ai> I have nothing to quiz myself on yet. Teach me or let me read something first.")
        return
    for r in result["results"]:
        mark = "OK " if r["correct"] else "MISS"
        print(f"  [{mark}] {r['q']}")
        print(f"         I said: {r['reply'] or '(nothing)'}")
        if not r["correct"]:
            print(f"         It should be: {r['answer']}")
    print(f"ai> Self-exam: {result['score']}/{result['total']} correct."
          + ("" if result["score"] == result["total"] else " I'll review what I missed."))


def show_sleep(rep: dict) -> None:
    print(f"ai> I slept on it (study day {rep['day']}): {rep['kept']} sentences studied, "
          f"{rep['strong']} of them strong, {rep['shelf']} on the shelf.")
    if rep["faded"]:
        print(f"    I forgot {len(rep['faded'])} things I never used (they're on the shelf if I need them).")
    if rep["merged"]:
        print(f"    I merged {rep['merged']} repeated things.")
    for c in rep["contradictions"]:
        if c["kept"]:
            print(f"    Two sources disagreed ('{c['a']}' vs '{c['b']}'); I kept: {c['kept']}")
        else:
            print(f"    I can't tell which is right: '{c['a']}' or '{c['b']}'. You can /teach me.")


def show_curiosity(report: list[dict]) -> None:
    if not report:
        print("ai> I have no open questions right now.")
    for r in report:
        if r["resolved"]:
            status = "now I can answer it"
        elif r["error"]:
            status = f"couldn't look it up ({r['error']})"
        else:
            status = "still not sure"
        print(f"  - {r['question']} -> read {r['learned']} sentences, {status}")


def show_progress(info: dict) -> None:
    for key, value in info.items():
        if key == "exams":
            continue
        print(f"   {key}: {value}")
    if info["exams"]:
        print("   recent exams: " + ", ".join(
            f"day {e['day']}: {e['score']}/{e['total']} ({e['who']})" for e in info["exams"]))


def run_user_quiz(model, n: int, ask) -> None:
    """Quiz the user: ask questions from what was learned, grade their answers."""
    questions = model.make_questions(n)
    if not questions:
        print("ai> I don't know enough yet to quiz you.")
        return
    score = 0
    for i, q in enumerate(questions, 1):
        answer = ask(f"  Q{i}. {q['q']}\n  your answer> ")
        correct = model.grade(q, answer, lenient=True)
        score += correct
        print("     Correct!" if correct else f"     Not quite. {q['answer']}")
    model.log_exam(score, len(questions), "you")
    print(f"ai> You got {score}/{len(questions)}.")
