import os
import tempfile
import unittest

from aimodel import LearningModel


class TestLearningModel(unittest.TestCase):
    def setUp(self):
        self.m = LearningModel(seed=0)
        self.m.learn("what is your name", "I'm your tiny AI.")
        self.m.learn("what is the weather like", "Check outside!")
        self.m.learn("tell me a joke", "Why did the bit flip? It was tired of being zero.")

    def test_unknown_input_returns_none(self):
        self.assertEqual(self.m.respond("quantum chromodynamics")[0], None)
        self.assertEqual(LearningModel().respond("zorp flimflam wibble")[0], None)

    def test_matches_similar_wording(self):
        self.assertEqual(self.m.respond("What's your name?")[0], "I'm your tiny AI.")
        self.assertEqual(self.m.respond("tell a joke please")[0],
                         "Why did the bit flip? It was tired of being zero.")

    def test_feedback_changes_preference(self):
        self.m.learn("hello", "Hi!")
        self.m.learn("hello", "Hey there!")
        first, _ = self.m.respond("hello")
        for _ in range(3):
            self.m.respond("hello")
            self.m.feedback(good=False)
        second, _ = self.m.respond("hello")
        self.assertNotEqual(first, second)

    def test_correct_learns_better_answer(self):
        self.m.respond("what is your name")
        self.m.correct("what is your name", "Call me Sage.")
        self.assertEqual(self.m.respond("what is your name")[0], "Call me Sage.")

    def test_forget(self):
        self.assertEqual(self.m.forget("joke"), 1)
        self.assertIsNone(self.m.respond("tell me a joke")[0])

    def test_generate_uses_learned_words(self):
        words = set()
        for mem in self.m.memories:
            words.update(mem["prompt"].split() + mem["response"].split())
        out = self.m.generate("tell")
        self.assertTrue(out.startswith("tell"))
        self.assertTrue(set(out.split()) <= words)

    def test_save_and_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "brain.json")
            self.m.save(path)
            loaded = LearningModel.load(path)
        self.assertEqual(loaded.pretrained is not None, True)  # the bundled word vectors are on by default
        self.m.load_builtin_vectors()
        self.assertEqual(loaded.memories, self.m.memories)
        self.assertEqual(loaded.stats(), self.m.stats())
        self.assertEqual(loaded.respond("your name?")[0], "I'm your tiny AI.")

    def test_learn_rejects_empty(self):
        with self.assertRaises(ValueError):
            self.m.learn("  ", "x")


class TestParameters(unittest.TestCase):
    def test_a_new_brain_has_only_the_tiny_feedback_network(self):
        counts = LearningModel(seed=0).parameter_counts()
        self.assertEqual(counts, {"word network": 0, "feedback network": 81,
                                  "transformer writer": 0, "total": 81})

    def test_the_word_network_grows_with_what_it_reads(self):
        m = LearningModel(seed=0)
        m.add_document("Cats are mammals. Mammals are warm-blooded animals.", "a.txt")
        words, dim = len(m.neural), m.neural.dim
        counts = m.parameter_counts()
        self.assertEqual(counts["word network"], 2 * words * dim)  # an input and an output vector per word
        self.assertEqual(counts["total"], counts["word network"] + 81)

    def test_the_writer_is_counted_and_shown_in_stats(self):
        from aimodel.transformer import CharTokenizer, TinyTransformer
        m = LearningModel(seed=0)
        m.writer = TinyTransformer.random(CharTokenizer.build(["abc"]), block_size=32)
        counts = m.parameter_counts()
        self.assertEqual(counts["transformer writer"], m.writer.n_params)
        self.assertGreater(counts["total"], 81)
        self.assertIn(f"{counts['total']:,}", m.stats()["parameters (total)"])


if __name__ == "__main__":
    unittest.main()
