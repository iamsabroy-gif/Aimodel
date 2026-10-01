import contextlib
import io
import json
import os
import tempfile
import unittest

import numpy as np

from aimodel import LearningModel
from aimodel.transformer import END, PAD, UNK, A, E, Q, BPETokenizer, CharTokenizer, TinyTransformer
from aimodel.writer import (build_examples, check_draft, complete, grounded, repeats, supported,
                            verdict, writer_evidence)

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

NOTES = ("Cats are mammals. Mammals are warm-blooded animals. Animals are living things. "
         "Rex is a golden retriever. The sky is blue because air scatters blue light.")


class StubWriter:
    """Stands in for a trained transformer: always drafts the same text."""
    n_params, meta = 123, {}

    def __init__(self, draft):
        self.draft = draft
        self.calls = []

    def write(self, question, evidence, max_new=300):
        self.calls.append((question, evidence))
        return self.draft


def model():
    m = LearningModel(seed=0)
    m.add_document(NOTES, "notes.txt")
    return m


class TestTokenizer(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer.build(["Cats are mammals.", "Is a cat an animal? Yes."])

    def test_round_trip_and_unknown_characters(self):
        self.assertEqual(self.tok.decode(self.tok.encode("Cats are mammals.")), "Cats are mammals.")
        self.assertEqual(self.tok.encode("₹"), [UNK])  # never seen and not plain ASCII
        self.assertEqual(self.tok.decode(self.tok.encode("Xy-1@")), "Xy-1@")  # ASCII always works

    def test_prompt_layout_and_budget(self):
        ids = self.tok.prompt("Is a cat an animal?", ["Cats are mammals."], 200)
        self.assertEqual((ids[0], ids[-1]), (Q, A))
        self.assertIn(E, ids)
        self.assertLessEqual(len(self.tok.prompt("a" * 500, ["b" * 500], 64)), 64)

    def test_training_example(self):
        ids, start = self.tok.example("Is a cat an animal?", ["Cats are mammals."], "Yes.", 128)
        self.assertEqual(ids[start - 1], A)
        self.assertEqual(self.tok.decode(ids[start:-1]), "Yes.")
        self.assertEqual(ids[-1], END)
        long_ids, _ = self.tok.example("q" * 300, ["e" * 300], "a" * 300, 128)
        self.assertLessEqual(len(long_ids), 129)
        self.assertEqual(long_ids[-1], END)  # the answer's end always survives


class TestBPE(unittest.TestCase):
    TEXTS = ["The zuquen is a small mammal that lives in forests."] * 5 + \
            ["The tober is a small bird that lives in rivers."] * 5

    def test_frequent_words_become_single_tokens(self):
        bpe = BPETokenizer.build(self.TEXTS, vocab_size=400)
        ids = bpe.encode("The zuquen is a small mammal")
        self.assertEqual([bpe.tokens[i - 6] for i in ids],
                         ["The", " zuquen", " is", " a", " small", " mammal"])

    def test_any_text_round_trips(self):
        bpe = BPETokenizer.build(self.TEXTS, vocab_size=400)
        text = "Brand-new: a Xylophone, 42 times!\nOK"
        self.assertEqual(bpe.decode(bpe.encode(text)), text)
        self.assertLess(len(bpe.encode(self.TEXTS[0])), len(self.TEXTS[0]) / 3)

    def test_saved_with_the_model(self):
        bpe = BPETokenizer.build(self.TEXTS, vocab_size=400)
        net = TinyTransformer.random(bpe, block_size=64)
        with tempfile.TemporaryDirectory() as d:
            net.save(os.path.join(d, "w.npz"))
            loaded = TinyTransformer.load(os.path.join(d, "w.npz"))
        self.assertEqual(loaded.tokenizer.kind, "bpe")
        self.assertEqual(loaded.tokenizer.encode(self.TEXTS[1]), bpe.encode(self.TEXTS[1]))


class TestTinyTransformer(unittest.TestCase):
    def setUp(self):
        self.tok = CharTokenizer.build([NOTES])
        self.net = TinyTransformer.random(self.tok, n_layer=2, n_head=2, d_model=32, block_size=96)
        self.ids = self.tok.prompt("What are cats?", ["Cats are mammals."], 60)

    def test_cached_steps_match_the_full_forward_pass(self):
        nxt = [7, 9, 11]
        full = self.net.forward(self.ids + nxt)
        cache = []
        self.net.forward(self.ids, cache)
        for i, token in enumerate(nxt):
            step = self.net._step(token, len(self.ids) + i, cache)
            self.assertLess(np.abs(step - full[len(self.ids) + i]).max(), 1e-5)

    def test_generation_never_writes_structure_tokens(self):
        out = self.net.generate(self.ids, max_new=20)
        self.assertLessEqual(len(out), 20)
        self.assertFalse({PAD, Q, E, A, END} & set(out))

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.npz")
            self.net.meta = {"trained_steps": 5}
            self.net.save(path)
            loaded = TinyTransformer.load(path)
        self.assertEqual(loaded.tokenizer.chars, self.tok.chars)
        self.assertEqual(loaded.meta, {"trained_steps": 5})
        self.assertEqual(loaded.cfg, self.net.cfg)
        self.assertEqual(float(np.abs(loaded.forward(self.ids) - self.net.forward(self.ids)).max()), 0.0)

    def test_write_fits_a_small_context(self):
        text = self.net.write("What are cats?" * 20, ["Cats are mammals."] * 20)
        self.assertIsInstance(text, str)


class TestGrounding(unittest.TestCase):
    EVIDENCE = ["Cats are mammals.", "Mammals are warm-blooded animals."]

    def test_answers_made_of_evidence_pass(self):
        ok, missing = grounded("Yes. Cats are mammals, and mammals are warm-blooded animals.",
                               self.EVIDENCE + ["Is a cat an animal?"])
        self.assertTrue(ok, missing)

    def test_invented_words_fail(self):
        ok, missing = grounded("Cats are reptiles that fly.", self.EVIDENCE)
        self.assertFalse(ok)
        self.assertIn("reptiles", missing)

    def test_empty_unrelated_and_repetitive_drafts_fail(self):
        self.assertFalse(grounded("", self.EVIDENCE)[0])
        self.assertFalse(grounded("Also, then.", self.EVIDENCE)[0])
        self.assertFalse(grounded("cats cats cats are mammals", self.EVIDENCE)[0])

    def test_complete_drafts_say_what_the_template_says(self):
        ref = "The gorzu is a small bird. In addition, the gorzu eats fish and berries."
        self.assertTrue(complete("The gorzu, a small bird, eats fish and berries.", ref)[0])
        ok, left_out = complete("The gorzu is a small bird.", ref)
        self.assertFalse(ok)
        self.assertEqual(left_out, ["eats", "fish", "berries"])

    def test_writer_evidence_speaks_to_you(self):
        trace = [{"kind": "evidence", "text": "I live in Pune.", "source": "me.txt"},
                 {"kind": "evidence", "text": "I think, therefore I am.", "source": "https://x.org"},
                 {"kind": "inference", "text": "Chained 2 facts", "source": "my reasoning"}]
        self.assertEqual(writer_evidence(trace), ["You live in Pune.", "I think, therefore I am."])


class TestSafetyCheck(unittest.TestCase):
    EV = ["Whales are not fish.", "The sheep eats grass.", "The moose eats plants.", "The seal eats fish.",
          "Cats are mammals.", "Mammals are warm-blooded animals.", "You like painting."]

    def check(self, draft, reference):
        return check_draft(draft, self.EV, "question", reference=reference)

    def test_a_wrong_yes_or_no_is_rejected(self):
        ok, problems = self.check("Yes. Whales are not fish.", "No. Whales are not fish.")
        self.assertFalse(ok)
        self.assertIn("yes", problems[0])
        self.assertTrue(self.check("No. Whales are not fish.", "No. Whales are not fish.")[0])
        self.assertEqual((verdict("Probably yes. A cat is an animal."), verdict("Cats are mammals.")),
                         ("probably yes", None))

    def test_the_same_words_with_the_wrong_meaning_are_rejected(self):
        self.assertFalse(self.check("The grass eats sheep.", "The sheep eats grass.")[0])      # roles swapped
        self.assertFalse(self.check("The seal eats plants.", "The moose eats plants.")[0])     # wrong animal
        self.assertFalse(self.check("Whales are fish.", "Whales are not fish.")[0])            # polarity dropped
        for good in ("The sheep eats grass.", "The moose eats plants."):
            self.assertTrue(self.check(good, good)[0])

    def test_reasoning_steps_and_asides_are_allowed(self):
        chain = "Yes. Cats are mammals, and mammals are warm-blooded animals, so a cat is an animal."
        self.assertTrue(self.check(chain, chain)[0])
        aside = "Cats are mammals, which are warm-blooded animals."
        self.assertTrue(self.check(aside, aside)[0])

    def test_repeats_are_rejected_even_when_reworded(self):
        self.assertTrue(repeats("Mount Kenya is a mountain in Africa in Africa."))
        self.assertTrue(repeats("You like painting. You also like painting."))
        self.assertFalse(repeats("The moose is a mammal. It also lives in forests."))
        self.assertFalse(self.check("You like painting. You also like painting.", "You like painting.")[0])

    def test_supported_reads_statements_as_facts(self):
        self.assertEqual(supported("The sheep eats grass.", self.EV), (True, []))
        ok, wrong = supported("The grass eats sheep.", self.EV)
        self.assertFalse(ok)
        self.assertEqual(wrong, ["The grass eats sheep."])
        self.assertTrue(supported("Hello there, welcome.", self.EV)[0])  # nothing to verify


class TestWriterInModel(unittest.TestCase):
    def test_a_wrong_yes_never_reaches_you(self):
        m = LearningModel(seed=0)
        m.add_document("Whales are not fish. Fish are animals that live in water.", "notes.txt")
        m.writer = StubWriter("Yes. Whales are not fish.")
        reply, _ = m.respond("Is a whale a fish?")
        self.assertTrue(reply.startswith("No."))
        self.assertIn("rejected", m.last_trace[-1]["text"])
        self.assertEqual(m.answer_log[-1]["by"], "templates")

    def test_a_grounded_draft_is_used(self):
        m = model()
        m.writer = StubWriter("Yes, cats are mammals and mammals are warm-blooded animals, so cats are animals")
        reply, _ = m.respond("Is a cat an animal?")
        self.assertEqual(reply, "Yes, cats are mammals and mammals are warm-blooded animals, so cats are animals.")
        self.assertEqual(m.last_trace[-1]["kind"], "writer")
        self.assertEqual(m.answer_log[-1]["by"], "transformer")
        self.assertTrue(m.answer_log[-1]["template"].startswith("Yes. Cats are mammals"))

    def test_an_invented_draft_falls_back_to_templates(self):
        m = model()
        m.writer = StubWriter("Cats are reptiles from Mars.")
        reply, _ = m.respond("Is a cat an animal?")
        self.assertTrue(reply.startswith("Yes. Cats are mammals"))
        self.assertIn("reptiles", m.last_trace[-1]["text"])
        self.assertEqual(m.answer_log[-1]["by"], "templates")

    def test_an_incomplete_draft_falls_back_to_templates(self):
        m = model()
        m.add_document("The gorzu is a small bird. The gorzu eats fish and berries.", "zoo.txt")
        m.writer = StubWriter("The gorzu is a small bird.")
        reply, _ = m.respond("Tell me about the gorzu")
        self.assertIn("berries", reply)
        self.assertIn("left things out", m.last_trace[-1]["text"])

    def test_writer_can_be_switched_off_and_skips_memories(self):
        m = model()
        m.writer = StubWriter("Cats are mammals.")
        m.writer_enabled = False
        m.respond("What are cats?")
        m.writer_enabled = True
        m.learn("hi", "Hello!")
        self.assertEqual(m.respond("hi")[0], "Hello!")
        self.assertEqual(m.writer.calls, [])

    def test_ratings_and_corrections_are_logged(self):
        m = model()
        m.respond("What is Rex?")
        m.feedback(good=True)
        self.assertEqual(m.answer_log[-1]["rating"], "good")
        m.respond("Why is the sky blue?")
        m.correct("Why is the sky blue?", "Because air scatters blue light the most.")
        self.assertEqual(m.answer_log[-1]["rating"], "bad")
        self.assertEqual(m.answer_log[-1]["correction"], "Because air scatters blue light the most.")

    def test_exams_are_not_logged_as_conversations(self):
        m = model()
        m.quiz(5)
        self.assertEqual(m.answer_log, [])

    def test_writer_and_log_are_saved_with_the_brain(self):
        m = model()
        m.respond("What is Rex?")
        m.writer = TinyTransformer.random(CharTokenizer.build([NOTES]), block_size=64)
        m._writer_changed = True
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "brain.json")
            m.save(path)
            self.assertTrue(os.path.exists(os.path.join(d, "brain.writer.npz")))
            loaded = LearningModel.load(path)
        self.assertEqual(loaded.answer_log, m.answer_log)
        self.assertEqual(loaded.writer.n_params, m.writer.n_params)


class TestTrainingData(unittest.TestCase):
    def test_examples_weigh_your_feedback_most(self):
        m = model()
        m.learn("hi", "Hello!")
        m.respond("What is Rex?")
        m.feedback(good=True)
        m.respond("Why is the sky blue?")
        m.correct("Why is the sky blue?", "Because air scatters blue light the most.")
        m.respond("What are cats?")
        m.feedback(good=False)
        kinds = {}
        for ex in build_examples(m):
            kinds.setdefault(ex["kind"], []).append(ex)
        self.assertEqual(kinds["your correction"][0]["weight"], 3.0)
        self.assertEqual(kinds["rated good"][0]["question"], "What is Rex?")
        self.assertIn("practice", kinds)
        self.assertEqual(kinds["taught"][0]["answer"], "Hello!")
        bad = [e for e in build_examples(m) if e["kind"] == "asked" and e["question"] == "What are cats?"]
        self.assertEqual(bad, [])  # answers you rated bad aren't taught

    def test_export_writes_the_files(self):
        m = model()
        m.respond("What is Rex?")
        with tempfile.TemporaryDirectory() as d:
            info = m.export_training_data(d)
            self.assertEqual(sorted(os.listdir(d)), ["ABOUT.txt", "corpus.txt", "writer_data.jsonl"])
            with open(os.path.join(d, "writer_data.jsonl")) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), info["examples"])
        self.assertTrue(all({"question", "evidence", "answer", "weight", "kind"} <= set(r) for r in rows))


@unittest.skipUnless(HAVE_TORCH, "training needs PyTorch (on Kaggle)")
class TestTraining(unittest.TestCase):
    def test_train_save_load_and_resume(self):
        from aimodel import train_writer
        m = model()
        for q in ["What is Rex?", "Is a cat an animal?", "Why is the sky blue?"]:
            m.respond(q)
        with tempfile.TemporaryDirectory() as d:
            m.export_training_data(d)
            out = os.path.join(d, "writer.npz")
            with contextlib.redirect_stdout(io.StringIO()) as log:
                train_writer.main(["--data", d, "--out", out, "--size", "tiny", "--steps", "6",
                                   "--batch", "4"])
            self.assertIn("numpy/PyTorch agreement", log.getvalue())
            net = m.load_writer(out)
            self.assertEqual(net.meta["trained_steps"], 6)
            with contextlib.redirect_stdout(io.StringIO()):
                train_writer.main(["--data", d, "--out", out, "--resume", out, "--steps", "4",
                                   "--batch", "4"])
            self.assertEqual(TinyTransformer.load(out).meta["trained_steps"], 10)
        reply, _ = m.respond("Is a cat an animal?")
        self.assertTrue(reply)  # untrained drafts fail the check and fall back safely


if __name__ == "__main__":
    unittest.main()
