import json
import os
import random
import re
import tempfile
import unittest

from aimodel.sample_data import ANIMALS, COUNTRIES, PLURAL_ANIMALS, build_sample, export_sample, persona
from aimodel.train_writer import fake_word, name_stats, swap_names
from aimodel.writer import grounded


class TestSampleData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples, cls.lines = build_sample(seed=0, people=6)

    def test_every_answer_is_built_from_its_own_evidence(self):
        self.assertGreater(len(self.examples), 400)
        for ex in self.examples:
            ok, missing = grounded(ex["answer"], ex["evidence"] + [ex["question"]])
            self.assertTrue(ok, (ex["question"], ex["answer"], missing))

    def test_the_format_matches_what_export_writes(self):
        for ex in self.examples:
            self.assertEqual(set(ex), {"question", "evidence", "answer", "weight", "kind"})
            self.assertTrue(ex["evidence"] and ex["question"] and ex["answer"])
        kinds = {ex["kind"] for ex in self.examples}
        self.assertEqual(kinds, {"practice", "your correction"})
        self.assertTrue(all(ex["weight"] == 3.0 for ex in self.examples if ex["kind"] == "your correction"))

    def test_many_ways_of_asking_and_many_kinds_of_answer(self):
        questions = " | ".join(ex["question"] for ex in self.examples)
        for expect in ["What is the red fox?", "Where does the tiger live?", "What does the panda eat".replace("panda", "giant panda"),
                       "Is a cow an animal?", "What are cats?", "What is the capital of France?", "Where is Japan?",
                       "Why is the sky blue?", "How do I make tea?", "What is Python?", "Are lists mutable?"]:
            self.assertIn(expect, questions)
        answers = " ".join(ex["answer"] for ex in self.examples)
        for expect in ["Here's how to plant a seed: First, dig a small hole.", "The reason is that",
                       "Yes. The cow is a mammal, and mammals are warm-blooded animals, so a cow is an animal."]:
            self.assertIn(expect, answers)

    def test_lists_and_plurals_are_kept_whole(self):
        answers = [ex["answer"] for ex in self.examples]
        self.assertTrue(any("seeds, insects and small animals" in a for a in answers))
        self.assertIn("Trout are fish.", answers)

    def test_it_is_repeatable_and_people_never_clash(self):
        again, _ = build_sample(seed=0, people=6)
        self.assertEqual(again, self.examples)
        pets = [ex for ex in self.examples if ex["question"].startswith("What is my ") and "called?" in ex["question"]]
        self.assertGreaterEqual(len(pets), 4)
        for ex in pets:  # one consistent pet name per person, taken from that person's own notes
            name = re.search(r"is called (\w+)", ex["answer"]).group(1)
            self.assertIn(f"is called {name}.", ex["evidence"][0])

    def test_pets_match_their_descriptions(self):
        rng = random.Random(1)
        for _ in range(60):
            notes, _questions = persona(rng)
            pet = notes.split("My ")[1].split(" is called")[0]
            descriptions = {"dog": ("retriever", "dog", "beagle", "terrier", "labrador"), "cat": ("cat",),
                            "parrot": ("parrot",), "rabbit": ("rabbit",), "hamster": ("hamster",), "turtle": ("turtle",)}
            self.assertTrue(any(w in notes.split(" is called ")[1] for w in descriptions[pet]), notes)

    def test_export_writes_the_three_files(self):
        with tempfile.TemporaryDirectory() as d:
            info = export_sample(d, people=3)
            self.assertEqual(sorted(os.listdir(d)), ["ABOUT.txt", "corpus.txt", "writer_data.jsonl"])
            with open(os.path.join(d, "writer_data.jsonl")) as f:
                rows = [json.loads(line) for line in f]
            self.assertEqual(len(rows), info["examples"])
            self.assertIn("Starter training data", open(os.path.join(d, "ABOUT.txt")).read())

    def test_the_lists_are_sane(self):
        self.assertEqual(len({a[0] for a in ANIMALS}), len(ANIMALS))
        self.assertEqual(len({c[0] for c in COUNTRIES}), len(COUNTRIES))
        self.assertEqual(len({p[1] for p in PLURAL_ANIMALS}), len(PLURAL_ANIMALS))


class TestNameSwapping(unittest.TestCase):
    def test_made_up_words_look_like_words(self):
        rng = random.Random(0)
        words = {fake_word(rng) for _ in range(300)}
        self.assertGreater(len(words), 150)  # varied
        self.assertTrue(all(3 <= len(w) <= 9 and w.isalpha() and w.islower() for w in words))
        self.assertTrue(all(any(v in w for v in "aeiou") for w in words))

    def test_names_are_swapped_everywhere_and_consistently(self):
        texts = ["The moose is a mammal.", "The moose lives in forests."] + \
                [f"The animal{i} is a mammal that lives in forests." for i in range(30)]
        common, rare = name_stats(texts)
        self.assertNotIn("mammal", rare)
        question, evidence, answer = swap_names(
            "What is the moose?", ["The moose is a mammal.", "The moose lives in forests."],
            "The moose is a mammal. It also lives in forests.", random.Random(2), common, rare, real=0.0)
        self.assertNotIn("moose", question + " ".join(evidence) + answer)
        name = question.removeprefix("What is the ").rstrip("?")
        self.assertEqual(evidence[0], f"The {name} is a mammal.")
        self.assertEqual(answer, f"The {name} is a mammal. It also lives in forests.")
        self.assertTrue(grounded(answer, evidence + [question])[0])

    def test_nothing_to_swap_leaves_the_example_alone(self):
        common, rare = name_stats(["The cat is a mammal."] * 10 + ["The cow is a mammal."] * 10)
        same = swap_names("What is it?", ["It is a mammal."], "It is a mammal.", random.Random(0), common, rare)
        self.assertEqual(same, ("What is it?", ["It is a mammal."], "It is a mammal."))


if __name__ == "__main__":
    unittest.main()
