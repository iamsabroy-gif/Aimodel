import json
import os
import tempfile
import unittest
import urllib.parse

from aimodel import LearningModel
from aimodel.neural import WordEmbeddings
from aimodel.text import split_sentences
from aimodel.web import html_to_text

NOTES = """My dog is called Rex. Rex is a golden retriever who loves swimming.
Our cat Luna sleeps all day on the sofa.
The project deadline is on Friday. The team meets every Monday at 10am in room 4."""

ARTICLE = """Photosynthesis is a process used by plants to convert light energy into chemical energy.
Plants absorb light primarily using the pigment chlorophyll.
== History ==
The process was discovered in 1779 by Jan Ingenhousz.
== References ==
Some reference text that should be dropped."""


def fake_fetch(url):
    """Stands in for the internet so tests are fast and offline."""
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    if q.get("list") == ["search"]:
        return json.dumps({"query": {"search": [{"title": "Photosynthesis"}]}})
    if q.get("prop") == ["extracts"]:
        return json.dumps({"query": {"pages": {"1": {"title": "Photosynthesis", "extract": ARTICLE}}}})
    return "<html><head><title>T</title><script>x()</script></head><body><p>Mars is the fourth planet from the Sun.</p></body></html>"


class TestKnowledge(unittest.TestCase):
    def setUp(self):
        self.m = LearningModel(seed=0)
        self.m.fetch = fake_fetch
        self.m.add_document(NOTES, "notes.txt")

    def test_answers_from_user_data_with_source(self):
        reply, conf = self.m.respond("What is my dog called?")
        self.assertIn("Rex", reply)
        self.assertEqual(self.m.last_source, "knowledge")
        self.assertEqual(self.m.last_trace[0]["source"], "notes.txt")

    def test_combines_evidence_across_sentences(self):
        reply, _ = self.m.respond("When is the deadline and when does the team meet?")
        self.assertIn("Friday", reply)
        self.assertIn("Monday", reply)
        self.assertGreaterEqual(len(self.m.last_trace), 2)

    def test_unrelated_questions_are_not_answered(self):
        self.assertIsNone(self.m.respond("who invented the telephone?")[0])
        self.assertIsNone(self.m.respond("zorp flimflam wibble")[0])

    def test_web_lookup_only_when_enabled(self):
        self.assertIsNone(self.m.respond("What is photosynthesis?")[0])
        reply, _ = self.m.respond("What is photosynthesis?", use_web=True)
        self.assertTrue(reply.startswith("Photosynthesis is a process"))
        self.assertIn("wikipedia.org", self.m.last_trace[0]["source"])
        texts = [k["text"] for k in self.m.knowledge]
        self.assertFalse(any("reference text" in t for t in texts))

    def test_read_web_page(self):
        self.assertEqual(self.m.read("https://example.com/mars"), 1)
        self.assertIn("fourth planet", self.m.respond("Which planet is Mars?")[0])

    def test_good_feedback_turns_answer_into_memory(self):
        reply, _ = self.m.respond("What is my dog called?")
        self.assertTrue(self.m.feedback(good=True))
        self.assertEqual(self.m.memories[-1]["response"], reply)
        self.m.respond("What is my dog called?")
        self.assertEqual(self.m.last_source, "memory")

    def test_save_and_load_keeps_knowledge_and_network(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "brain.json")
            self.m.save(path)
            self.assertTrue(os.path.exists(os.path.join(d, "brain.neural.npz")))
            loaded = LearningModel.load(path)
        self.assertEqual(loaded.knowledge, self.m.knowledge)
        self.assertEqual(loaded.neural.words, self.m.neural.words)
        self.assertIn("Rex", loaded.respond("What is my dog called?")[0])


class TestNeural(unittest.TestCase):
    def test_learns_words_used_in_similar_ways(self):
        groups = {"animal": ["cat", "dog", "horse"], "food": ["apple", "bread", "rice"]}
        sents = []
        for i in range(300):
            a, f = groups["animal"][i % 3], groups["food"][i % 3]
            sents.append(f"the {a} runs and sleeps in the barn".split())
            sents.append(f"i ate some {f} for lunch today".split())
        emb = WordEmbeddings(seed=0)
        first = emb.train(sents, epochs=1)
        last = emb.train(sents, epochs=5, new=False)
        self.assertLess(last, first)
        self.assertIn(emb.similar("cat", 2)[0][0], {"dog", "horse"})
        v = emb.sentence_vectors([["dog", "sleeps"], ["cat", "runs"], ["ate", "bread"]])
        self.assertGreater(v[0] @ v[1], v[0] @ v[2])

    def test_network_helps_match_different_wording(self):
        m = LearningModel(seed=0)
        m.neural.MATURE_WORDS = 5000  # a small, very regular corpus is enough here
        corpus = []
        for a in ["cat", "dog", "horse", "cow"]:  # enough text for the network to be trusted
            corpus += [f"my {a} runs and sleeps in the barn on night {i}." for i in range(150)]
        m.add_document(" ".join(corpus), "animals")
        m.learn("where does the cat sleep", "In the barn.")
        self.assertEqual(m.neural.maturity, 1.0)
        self.assertEqual(m.respond("where does the horse sleep")[0], "In the barn.")


class TestText(unittest.TestCase):
    def test_split_sentences_skips_headings(self):
        self.assertEqual(split_sentences("== Title ==\nOne two three. Four five six!"),
                         ["One two three.", "Four five six!"])

    def test_split_sentences_lowercase_and_abbreviations(self):
        self.assertEqual(split_sentences("my cat runs fast. my dog sleeps a lot. Dr. Rao saw e.g. cats here."),
                         ["my cat runs fast.", "my dog sleeps a lot.", "Dr. Rao saw e.g. cats here."])

    def test_html_to_text_drops_scripts(self):
        title, text = html_to_text("<title>Hi</title><script>bad()</script><p>Good\ntext.</p>")
        self.assertEqual(title, "Hi")
        self.assertNotIn("bad", text)
        self.assertIn("Good text", text)

    def test_html_to_text_drops_menus(self):
        _, text = html_to_text("<ul><li><a href='#a'>Home</a></li><li>About us</li></ul>"
                               "<p>Mars is the fourth planet.</p>")
        self.assertEqual(text, "Mars is the fourth planet.")


if __name__ == "__main__":
    unittest.main()
