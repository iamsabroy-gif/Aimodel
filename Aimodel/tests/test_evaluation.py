import os
import subprocess
import sys
import tempfile
import unittest

from aimodel import LearningModel
from aimodel.evaluation import (CORRUPTIONS, GREETINGS, ITEMS, METRICS, NOT_GREETINGS, benchmark_model, change,
                                evaluate_all, format_history, format_report, history_path, judge, load_history,
                                regression_index, run_benchmark, run_greetings, run_safety_check, save_run, show,
                                status)
from aimodel.sample_data import build_sample_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def stable(metrics):
    """The metrics that don't depend on how fast the computer is."""
    return {k: v for k, v in metrics.items() if not k.endswith("_ms") and not k.endswith("disk_mb") and v is not None}


class TestFixedTests(unittest.TestCase):
    """These are the regression guards: if one fails after a change, the change made it worse."""

    @classmethod
    def setUpClass(cls):
        cls.bench = run_benchmark()["metrics"]
        cls.safety = run_safety_check()
        cls.greet = run_greetings()

    def test_reasoning_benchmark_standards(self):
        self.assertGreaterEqual(self.bench["bench.accuracy"], 0.90)
        self.assertGreaterEqual(self.bench["bench.abstain"], 0.90)
        self.assertEqual(self.bench["bench.unsupported"], 0)
        self.assertLessEqual(self.bench["bench.wrong_rate"], 0.05)

    def test_stretch_questions_leave_room_to_grow(self):
        self.assertLess(self.bench["bench.stretch"], 1.0)  # a benchmark that is always 100% shows no progress

    def test_safety_check_standards(self):
        self.assertGreaterEqual(self.safety["metrics"]["check.catch_rate"], 0.95, self.safety["missed"])
        self.assertLessEqual(self.safety["metrics"]["check.false_reject"], 0.10)
        self.assertGreater(self.safety["corrupted"], 50)
        self.assertEqual(set(self.safety["by_kind"]), set(CORRUPTIONS))  # every kind of corruption is exercised

    def test_greeting_standards(self):
        self.assertGreaterEqual(self.greet["metrics"]["greet.recall"], 0.95, self.greet["missed"])
        self.assertEqual(self.greet["metrics"]["greet.false_positive"], 0, self.greet["false_alarms"])

    def test_test_sets_are_not_tiny_and_do_not_overlap(self):
        self.assertGreaterEqual(len(ITEMS), 40)
        self.assertGreaterEqual(len(GREETINGS), 40)
        self.assertFalse({t.lower() for t in GREETINGS} & {t.lower() for t in NOT_GREETINGS})
        self.assertEqual(len({i[1] for i in ITEMS}), len(ITEMS))

    def test_regression_index(self):
        full = {"bench.accuracy": 1, "bench.abstain": 1, "check.catch_rate": 1, "greet.recall": 1,
                "check.false_reject": 0, "greet.false_positive": 0, "bench.wrong_rate": 0}
        self.assertEqual(regression_index(full), 100)
        self.assertEqual(regression_index({**full, "bench.accuracy": 0}), 100 * 6 / 7)
        self.assertIsNone(regression_index({}))
        self.assertGreaterEqual(regression_index({**self.bench, **self.safety["metrics"], **self.greet["metrics"]}), 95)


class TestJudging(unittest.TestCase):
    def test_answers(self):
        self.assertTrue(judge("The mimbat is a mammal.", "lookup", ["mammal"], {"no": ["bird"]}))
        self.assertFalse(judge("The mimbat is a bird.", "lookup", ["mammal"], {"no": ["bird"]}))   # a forbidden word
        self.assertFalse(judge("The mimbat is a mammal.", "lookup", ["mammal", "caves"], {}))      # one is missing
        self.assertFalse(judge(None, "lookup", ["mammal"], {}))

    def test_order_and_yes_no_and_declining(self):
        steps = {"ordered": True}
        self.assertTrue(judge("First, mix. Then knead. Finally, roll.", "steps", ["mix", "knead", "roll"], steps))
        self.assertFalse(judge("First, roll. Then mix. Finally, knead.", "steps", ["mix", "knead", "roll"], steps))
        self.assertTrue(judge("Yes. A mimbat is an animal.", "chain", None, {"verdict": "yes"}))
        self.assertTrue(judge("Probably yes. It is.", "chain", None, {"verdict": "yes"}))
        self.assertFalse(judge("No. Whales are not fish.", "chain", None, {"verdict": "yes"}))
        self.assertTrue(judge("No. Zarnaks are not mammals.", "negation", None, {"verdict": "no"}))
        self.assertTrue(judge(None, "abstain", None, {}))
        self.assertFalse(judge("The glorb eats fish.", "abstain", None, {}))

    def test_the_benchmark_world_cannot_be_answered_from_the_web_or_memory(self):
        m = LearningModel(seed=0)  # knows nothing: every non-abstain question must fail
        for skill, question, gold, options in ITEMS:
            if skill != "abstain":
                self.assertFalse(judge(m.respond(question, learn=False)[0], skill, gold, options), question)


class TestCorruptions(unittest.TestCase):
    def test_each_kind_changes_an_answer_the_way_it_says(self):
        self.assertEqual(CORRUPTIONS["wrong yes/no"]("Yes. A cat is an animal."), "No. A cat is an animal.")
        self.assertEqual(CORRUPTIONS["dropped 'not'"]("Whales are not fish."), "Whales are fish.")
        self.assertIsNone(CORRUPTIONS["dropped 'not'"]("Whales are fish."))
        self.assertEqual(CORRUPTIONS["roles swapped"]("The lorpan eats grass."), "The grass eats the lorpan.")
        self.assertIn("lorpan", CORRUPTIONS["wrong entity"]("The mimbat is a mammal."))
        self.assertTrue(CORRUPTIONS["invented fact"]("Yes.").endswith("moonlight."))
        self.assertEqual(CORRUPTIONS["repeated phrase"]("A b. The long second sentence here."),
                         "A b. The long second sentence here. The long second sentence here.")
        self.assertIsNone(CORRUPTIONS["left things out"]("One. Two."))


class TestScoring(unittest.TestCase):
    def test_status_and_display(self):
        self.assertEqual(status("bench.accuracy", 0.95), "PASS")
        self.assertEqual(status("bench.accuracy", 0.80), "WARN")
        self.assertEqual(status("bench.accuracy", 0.50), "FAIL")
        self.assertEqual(status("bench.wrong_rate", 0.02), "PASS")          # lower is better
        self.assertEqual(status("bench.wrong_rate", 0.50), "FAIL")
        self.assertEqual(status("brain.facts", 5), "")                       # information only
        self.assertEqual(status("bench.accuracy", None), "")
        self.assertEqual((show("bench.accuracy", 0.876), show("brain.facts", 12345), show("brain.disk_mb", None)),
                         ("88%", "12,345", "-"))

    def test_change_says_better_or_worse_by_what_matters(self):
        self.assertEqual(change("bench.accuracy", 0.95, 0.90), "+5 pts better")
        self.assertEqual(change("bench.accuracy", 0.85, 0.90), "-5 pts WORSE")
        self.assertEqual(change("bench.wrong_rate", 0.10, 0.02), "+8 pts WORSE")    # more wrong is worse
        self.assertEqual(change("bench.wrong_rate", 0.0, 0.02), "-2 pts better")
        self.assertEqual(change("bench.accuracy", 0.901, 0.900), "same")
        self.assertEqual(change("brain.facts", 120, 100), "+20")                    # information only
        self.assertEqual(change("bench.accuracy", None, 0.9), "")
        for key in METRICS:  # every metric is complete and sensible
            label, group, direction, target, warn, kind = METRICS[key]
            self.assertIn(direction, ("higher", "lower", "info"))
            self.assertEqual(direction == "info", target is None)
            self.assertIn(kind, ("pct", "int", "mb", "ms"))


class TestRunsAndHistory(unittest.TestCase):
    def test_a_full_run_on_a_brain_with_data(self):
        model, _ = build_sample_model()
        report = evaluate_all(model, None)
        metrics = report["metrics"]
        self.assertGreaterEqual(metrics["brain.recall"], 0.85)
        self.assertEqual(metrics["brain.facts"], len(model.facts))
        self.assertEqual(metrics["brain.parameters"], model.parameter_counts()["total"])
        self.assertIsNone(metrics.get("brain.disk_mb"))
        self.assertNotIn("writer.pass_seen", metrics)                              # no writer loaded
        text = format_report(report)
        for expect in ("A  Reasoning benchmark", "D  Your brain", "Regression index", "Needs attention"):
            self.assertIn(expect, text)

    def test_an_empty_brain_does_not_crash(self):
        report = evaluate_all(LearningModel(seed=0), None)
        self.assertNotIn("brain.recall", report["metrics"])
        self.assertEqual(report["metrics"]["brain.facts"], 0)
        self.assertIn("Regression index", format_report(report))

    def test_quick_runs_only_the_fixed_tests(self):
        report = evaluate_all(build_sample_model()[0], None, quick=True)
        self.assertTrue(report["quick"])
        self.assertNotIn("brain.recall", report["metrics"])
        self.assertIn("bench.accuracy", report["metrics"])

    def test_evaluating_changes_nothing_and_is_repeatable(self):
        model, _ = build_sample_model()
        model.respond("hello")
        before = (model.to_dict(), model.neural.w_in.copy(), len(model.answer_log), model.last_reply, model.day)
        first = evaluate_all(model, None)["metrics"]
        second = evaluate_all(model, None)["metrics"]
        after = (model.to_dict(), model.neural.w_in, len(model.answer_log), model.last_reply, model.day)
        self.assertEqual(before[0], after[0])
        self.assertTrue((before[1] == after[1]).all())
        self.assertEqual(before[2:], after[2:])
        self.assertEqual(stable(first), stable(second))                             # same questions every time

    def test_history_is_saved_compared_and_trimmed(self):
        with tempfile.TemporaryDirectory() as d:
            path = history_path(os.path.join(d, "brain.json"))
            self.assertEqual(load_history(path), [])
            self.assertIn("No evaluations saved", format_history([]))
            report = evaluate_all(None, None, quick=True)
            save_run(path, report, "baseline")
            runs = load_history(path)
            self.assertEqual((len(runs), runs[0]["label"], runs[0]["quick"]), (1, "baseline", True))
            self.assertAlmostEqual(runs[0]["metrics"]["index"], report["metrics"]["index"])
            for _ in range(105):
                save_run(path, report)
            self.assertEqual(len(load_history(path)), 100)
            self.assertIn("index", format_history(load_history(path)))

    def test_a_regression_is_called_out(self):
        report = evaluate_all(None, None, quick=True)
        better_before = {"when": "earlier", "metrics": {**report["metrics"], "bench.accuracy": 1.0,
                                                        "greet.false_positive": 0.0}}
        worse_now = {**report, "metrics": {**report["metrics"], "bench.accuracy": 0.8, "greet.false_positive": 0.1}}
        text = format_report(worse_now, better_before)
        self.assertIn("-20 pts WORSE", text)
        self.assertIn("Got worse since last time: Answers correct, Questions mistaken for greetings", text)
        self.assertIn("Needs attention: Answers correct (below target), Questions mistaken for greetings (FAIL)", text)

    def test_a_trained_writer_adds_its_metrics(self):
        from aimodel.transformer import CharTokenizer, TinyTransformer
        model, _ = build_sample_model()
        model.writer = TinyTransformer.random(CharTokenizer.build(["abc"]), block_size=64)
        report = evaluate_all(model, None)
        self.assertIn("writer.pass_seen", report["metrics"])
        self.assertIn("F  Transformer writer", format_report(report))


class TestCommand(unittest.TestCase):
    def run_chat(self, *lines):
        with tempfile.TemporaryDirectory() as d:
            out = subprocess.run([sys.executable, "-m", "aimodel", "--brain", os.path.join(d, "brain.json"),
                                  "--offline"], input="\n".join(lines) + "\n/quit\n", text=True,
                                 capture_output=True, cwd=ROOT, timeout=300).stdout
            return out, os.listdir(d)

    def test_evaluate_quick_then_history(self):
        out, files = self.run_chat("/evaluate quick", "/evaluate quick", "/evaluate history")
        self.assertIn("Scorecard", out)
        self.assertIn("compared with the run of", out)           # the second run is compared with the first
        self.assertIn("Regression index", out)
        self.assertIn("quick", out.split("When")[-1])            # the history table lists the quick runs
        self.assertIn("brain.eval.json", files)

    def test_help_works_and_mentions_it(self):
        out, _ = self.run_chat("/help")  # this once crashed on the {name} in the greetings help
        self.assertIn("/evaluate", out)
        self.assertIn("Namaste {name}!", out)
        self.assertIn("/quit", out)


if __name__ == "__main__":
    unittest.main()
