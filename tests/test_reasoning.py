import os
import tempfile
import unittest

import numpy as np

from aimodel import LearningModel
from aimodel.ranker import FeedbackRanker
from aimodel.reasoning import extract_fact, flip_person, head, isa_chain, state

FACTS = """Cats are mammals. Mammals are warm-blooded animals. Animals are living things.
Snakes are not mammals.
The sky is blue because air molecules scatter blue sunlight more than red sunlight.
To make tea, boil water in a kettle. Then put a tea bag in a cup.
Pour the hot water over the tea bag. Finally, wait three minutes and remove the bag."""


class TestFacts(unittest.TestCase):
    def test_extracts_subject_verb_object(self):
        f = extract_fact("Plants can make their own food (autotrophs).")
        self.assertEqual((f["subj"], f["rel"], f["verb"], f["obj"]),
                         ("Plants", "make", "can make", "their own food"))
        self.assertTrue(extract_fact("Cats are not reptiles.")["neg"])
        self.assertIsNone(extract_fact("In 1876, Bell was granted a patent."))

    def test_pronoun_resolves_to_topic(self):
        self.assertEqual(extract_fact("It is the fourth planet.", topic="Mars")["subj"], "Mars")
        self.assertIsNone(extract_fact("It is the fourth planet."))

    def test_speaks_to_you_about_yourself(self):
        self.assertEqual(flip_person("I was sure my dog is mine"), "you were sure your dog is yours")
        self.assertEqual(state(extract_fact("My dog is called Rex.")), "Your dog is called Rex.")

    def test_head_noun(self):
        self.assertEqual(head("a small domesticated carnivorous mammal"), "mammal")
        self.assertEqual(head("a type of big cat"), "cat")

    def test_isa_chain(self):
        facts = [extract_fact(s) for s in ["Cats are mammals.", "Mammals are animals."]]
        self.assertEqual(len(isa_chain(facts, "cat", "animal")), 2)
        self.assertIsNone(isa_chain(facts, "animal", "cat"))


class TestReasoning(unittest.TestCase):
    def setUp(self):
        self.m = LearningModel(seed=0)
        self.m.add_document(FACTS, "facts.txt")

    def ask(self, q):
        return self.m.respond(q)[0]

    def test_proves_things_it_was_never_told(self):
        reply = self.ask("Is a cat a living thing?")
        self.assertTrue(reply.startswith("Yes."))
        self.assertIn("so a cat is a living thing", reply)
        kinds = [s["kind"] for s in self.m.last_trace]
        self.assertEqual(kinds.count("fact"), 3)
        self.assertIn("inference", kinds)

    def test_generalizing_from_one_kind_is_only_probable(self):
        self.m.add_document("The sperm whale is a large mammal.", "whales.txt")
        self.assertTrue(self.ask("Is a whale a mammal?").startswith("Probably yes."))

    def test_answer_must_cover_the_key_word(self):
        self.m.add_document("The pigments are arranged to work together.", "plants.txt")
        self.assertIsNone(self.ask("How do vaccines work?"))

    def test_negative_facts(self):
        self.assertTrue(self.ask("Is a snake a mammal?").startswith("No."))

    def test_why_finds_the_reason(self):
        self.assertEqual(self.ask("Why is the sky blue?"),
                         "The reason is that air molecules scatter blue sunlight more than red sunlight.")

    def test_how_gives_steps_in_order(self):
        reply = self.ask("How do I make tea?")
        self.assertTrue(reply.startswith("Here's how: First,"))
        self.assertLess(reply.index("kettle"), reply.index("tea bag in a cup"))
        self.assertIn("Then, put", reply)
        self.assertIn("Finally, wait", reply)

    def test_learns_facts_you_state_in_chat(self):
        reply, _ = self.m.respond("My sister lives in Delhi")
        self.assertEqual(reply, "Got it: Your sister lives in Delhi.")
        self.assertEqual(self.m.last_source, "noted")
        self.assertEqual(self.ask("Where does my sister live?"), "Your sister lives in Delhi.")

    def test_follows_a_link_between_facts(self):
        self.m.add_document("My dog is called Rex. Rex is a golden retriever.", "notes.txt")
        self.assertEqual(self.ask("What is my dog called?"),
                         "Your dog is called Rex, which is a golden retriever.")

    def test_young_network_does_not_cause_false_matches(self):
        self.m.respond("Is a cat a living thing?")
        self.m.feedback(good=True)  # saved as a memory
        self.assertEqual(self.m.respond("My sister lives in Delhi")[1], 1.0)
        self.assertEqual(self.ask("Where does my sister live?"), "Your sister lives in Delhi.")
        self.assertTrue(self.ask("Why is the sky blue?").startswith("The reason is"))

    def test_facts_survive_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "brain.json")
            self.m.save(path)
            loaded = LearningModel.load(path)
        self.assertEqual(loaded.facts, self.m.facts)
        self.assertTrue(loaded.respond("Is a cat an animal?")[0].startswith("Yes."))


class TestFeedbackNetwork(unittest.TestCase):
    def test_starts_neutral_then_learns(self):
        r = FeedbackRanker()
        good, bad = [0.5, 0.5, 1, 1, 1, 0.3, 1, 0], [0.5, 0.5, 1, 0, 0, 0.3, 0, 1]
        self.assertTrue(np.allclose(r.adjust([good, bad]), 0))
        for _ in range(5):
            r.train([good, bad], [1, 0])
        adj = r.adjust([good, bad])
        self.assertGreater(adj[0], 0)
        self.assertLess(adj[1], 0)

    def test_bad_feedback_trains_ranker(self):
        m = LearningModel(seed=0)
        m.add_document("Mars is a red planet. Mars has two small moons.", "space.txt")
        m.respond("Tell me about Mars moons")
        before = m.ranker.examples
        self.assertTrue(m.feedback(good=False))
        self.assertGreater(m.ranker.examples, before)


class TestPretrainedAndGrowth(unittest.TestCase):
    def test_pretrained_vectors_understand_meaning(self):
        rng = np.random.default_rng(0)
        base = rng.normal(size=20)
        rows = {"job": base, "work": base + 0.1 * rng.normal(size=20)}
        for w in ["cat", "blue", "tree", "sky", "music", "data", "analyst", "pune"]:
            rows[w] = rng.normal(size=20)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "vec.txt")
            with open(path, "w") as f:
                for w, v in rows.items():
                    f.write(w + " " + " ".join(f"{x:.4f}" for x in v) + "\n")
            m = LearningModel(seed=0)
            m.add_document("I work as a data analyst in Pune.", "me.txt")
            self.assertIsNone(m.respond("What is my job?")[0])
            self.assertEqual(m.load_vectors(path), len(rows))
            self.assertEqual(m.respond("What is my job?")[0], "You work as a data analyst in Pune.")
            brain = os.path.join(d, "brain.json")
            m.save(brain)
            self.assertEqual(len(LearningModel.load(brain).pretrained), len(rows))

    def test_network_grows_when_it_knows_enough_words(self):
        m = LearningModel(seed=0)
        m.GROWTH = ((30, 64),)
        m.add_document(" ".join(f"Word{i} is thing{i} here." for i in range(20)), "words.txt")
        self.assertEqual(m.neural.dim, 64)
        self.assertGreaterEqual(len(m.neural), 30)


if __name__ == "__main__":
    unittest.main()
