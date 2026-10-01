import contextlib
import io
import json
import os
import tempfile
import unittest
import urllib.error
import urllib.parse

from aimodel import LearningModel
from aimodel.reports import run_user_quiz

NOTES = ("Cats are mammals. Mammals are warm-blooded animals. Animals are living things. "
         "Rex is a golden retriever. Plants absorb light primarily using the pigment chlorophyll.")
WEB = "https://example.org/zoo"

DICT = {"zebra": [{"meanings": [{"partOfSpeech": "noun", "definitions": [
    {"definition": "An African wild horse with black and white stripes."}]}]}]}
OKAPI = "The okapi is a mammal native to the Congo. It has striped legs like a zebra."


def fake_fetch(url):
    """Stands in for the internet: a dictionary and a one-article Wikipedia."""
    if "dictionaryapi.dev" in url:
        word = urllib.parse.unquote(url.rsplit("/", 1)[1])
        if word in DICT:
            return json.dumps(DICT[word])
        raise urllib.error.HTTPError(url, 404, "not found", {}, None)
    q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
    if q.get("list") == ["search"]:
        return json.dumps({"query": {"search": [{"title": "Okapi"}]}})
    if q.get("prop") == ["extracts"]:
        return json.dumps({"query": {"pages": {"1": {"title": "Okapi", "extract": OKAPI}}}})
    raise urllib.error.URLError("offline")


def long_page():
    """A long web page: one recurring topic plus one-off trivia."""
    main = [f"A volcano erupts when magma rises through the crust in eruption number {i}."
            for i in range(60)]
    main[0] = "A volcano is an opening in the crust of a planet."
    trivia = ["Mount Zephyria was first climbed by a baker named Orlando in 1873.",
              "The village of Quillon sells pottery decorated with blue herons."]
    other = [f"Sentence about unrelated subject {i} mentions topic{i} and thing{i} only once."
             for i in range(60)]
    return "\n".join(main[:30] + trivia + main[30:] + other)


def model(*docs):
    m = LearningModel(seed=0)
    m.fetch = fake_fetch
    for text, source in docs:
        m.add_document(text, source)
    return m


class TestExams(unittest.TestCase):
    def setUp(self):
        self.m = model((NOTES, "notes.txt"), ("My dog is called Rex.", "you said"))

    def test_self_exam_is_graded_and_reviews_are_spaced(self):
        self.m.last_reply = "keep me"
        result = self.m.quiz(10)
        self.assertGreaterEqual(result["total"], 4)
        self.assertEqual(result["score"], result["total"])
        self.assertEqual(self.m.last_reply, "keep me")  # an exam doesn't disturb the chat
        recalled = [e for r in result["results"] for e in r["items"]]
        self.assertTrue(all(e["iv"] >= 2 and e["due"] > self.m.day for e in recalled))
        self.assertEqual(self.m.exam_log[-1]["score"], result["score"])

    def test_questions_do_not_give_the_answer_away(self):
        for q in self.m.make_questions(20):
            if q["expect"]:
                self.assertNotIn(q["expect"], q["q"].lower().replace("?", "").split())

    def test_inference_question(self):
        qs = {q["q"]: q for q in self.m.make_questions(30)}
        q = qs["Is a cat an animal?"]
        self.assertEqual(q["kind"], "infer")
        self.assertTrue(self.m.grade(q, "Yes. Cats are mammals, so a cat is an animal."))
        self.assertFalse(self.m.grade(q, "I don't know."))

    def test_grading_models_and_people(self):
        q = next(q for q in self.m.make_questions(30) if q["q"] == "What is Rex?")
        self.assertTrue(self.m.grade(q, "Rex is a golden retriever."))
        self.assertFalse(self.m.grade(q, "Rex is a cat."))
        self.assertFalse(self.m.grade(q, "golden"))              # a sentence must hold the key word
        self.assertTrue(self.m.grade(q, "golden", lenient=True))  # a person's short answer is enough

    def test_a_miss_is_reviewed_and_becomes_a_study_goal(self):
        q = next(q for q in self.m.make_questions(30) if q["q"] == "What is Rex?")
        self.m.record(q, False)
        entry = q["items"][0]
        self.assertEqual((entry["bad"], entry["iv"], entry["due"]), (1, 1, self.m.day))
        self.assertEqual(self.m.study_goals()[0]["why"], "forgot")
        report = self.m.be_curious()
        self.assertTrue(report[0]["resolved"])  # I do know it, so the gap closes
        self.assertEqual(self.m.gaps, [])

    def test_quizzing_the_user_does_not_touch_my_memory(self):
        before = [dict(e) for e in self.m.knowledge]
        shown = []
        with contextlib.redirect_stdout(io.StringIO()):
            run_user_quiz(self.m, 3, lambda prompt: shown.append(prompt) or "no idea")
        self.assertEqual(len(shown), 3)
        self.assertEqual(self.m.knowledge, before)
        self.assertEqual(self.m.gaps, [])
        self.assertEqual(self.m.exam_log[-1]["who"], "you")

    def test_nothing_to_quiz_on(self):
        self.assertEqual(LearningModel(seed=0).quiz(), {"results": [], "score": 0, "total": 0})


class TestImportanceAndShelf(unittest.TestCase):
    def test_long_web_pages_are_sorted_by_importance(self):
        m = model()
        m.add_document(long_page(), WEB, topic="Volcano")
        self.assertGreater(m.last_read["shelved"], 0)
        self.assertEqual(len(m.shelf), m.last_read["shelved"])
        studied = [k["text"] for k in m.knowledge]
        self.assertIn("A volcano is an opening in the crust of a planet.", studied)
        self.assertTrue(any("Zephyria" in e["text"] for e in m.shelf))  # one-off trivia waits

    def test_shelved_sentences_are_looked_up_when_asked(self):
        m = model()
        m.add_document(long_page(), WEB, topic="Volcano")
        shelf = len(m.shelf)
        self.assertIsNone(m.respond("Who first climbed Mount Zephyria?", learn=False)[0])
        self.assertIn("baker", m.respond("Who first climbed Mount Zephyria?")[0])
        self.assertEqual(len(m.shelf), shelf - 1)

    def test_your_own_data_is_never_shelved(self):
        m = model()
        m.add_document(long_page(), "my_notes.txt")
        self.assertEqual((m.last_read["shelved"], len(m.shelf)), (0, 0))

    def test_reading_something_for_a_question_favours_that_question(self):
        m = model()
        m.add_document(long_page(), WEB, topic="Volcano", focus="Who climbed Mount Zephyria?")
        self.assertTrue(any("Zephyria" in k["text"] for k in m.knowledge))

    def test_reading_a_shelved_sentence_again_brings_it_back(self):
        m = model()
        m.add_document(long_page(), WEB)
        trivia = "The village of Quillon sells pottery decorated with blue herons."
        self.assertTrue(any(e["text"] == trivia for e in m.shelf))
        m.add_document(trivia, WEB)
        self.assertTrue(any(k["text"] == trivia for k in m.knowledge))


class TestSleep(unittest.TestCase):
    def setUp(self):
        self.m = model((OKAPI, WEB), ("My sister lives in Delhi.", "you said"),
                       ("The pigment chlorophyll absorbs light.", "plants.txt"))

    def test_unused_web_knowledge_fades_but_yours_does_not(self):
        report = self.m.sleep(days=15)
        self.assertEqual(len(report["faded"]), 2)
        self.assertTrue(all(k["source"] != WEB for k in self.m.knowledge))
        self.assertEqual({k["source"] for k in self.m.knowledge}, {"you said", "plants.txt"})
        self.assertEqual(len(self.m.shelf), 2)
        self.assertFalse(any(WEB in f["source"] for f in self.m.facts))

    def test_faded_knowledge_comes_back_from_the_shelf(self):
        self.m.sleep(days=15)
        self.assertIn("native to the Congo", self.m.respond("What is the okapi?")[0])
        self.assertTrue(any(k["source"] == WEB for k in self.m.knowledge))

    def test_used_and_confirmed_knowledge_survives(self):
        self.m.respond("What is the okapi?")
        self.m.feedback(good=True)
        self.m.sleep(days=15)
        self.assertTrue(any("okapi" in k["text"] for k in self.m.knowledge))

    def test_repeats_are_merged(self):
        m = model(("Cats are small mammals.", "a.txt"), ("Small mammals are cats.", "b.txt"))
        self.assertEqual(len(m.knowledge), 2)
        self.assertEqual(m.sleep()["merged"], 1)
        self.assertEqual(len(m.knowledge), 1)

    def test_contradictions_keep_what_you_said(self):
        m = model(("Whales are fish.", "https://example.org/old"))
        m.respond("Whales are not fish.")
        report = m.sleep()
        self.assertEqual(report["contradictions"][0]["kept"], "Whales are not fish.")
        self.assertEqual([k["text"] for k in m.knowledge], ["Whales are not fish."])
        self.assertTrue(m.shelf)  # the losing sentence is only shelved

    def test_contradictions_between_equals_are_left_for_you(self):
        m = model(("Whales are fish.", "a.txt"), ("Whales are not fish.", "b.txt"))
        report = m.sleep()
        self.assertIsNone(report["contradictions"][0]["kept"])
        self.assertEqual(len(m.knowledge), 2)

    def test_sleeps_by_itself_once_a_day_has_passed(self):
        now = [1000.0]
        self.m.clock = lambda: now[0]
        self.assertIsNone(self.m.sleep_if_due())
        now[0] += 3600
        self.assertIsNone(self.m.sleep_if_due())
        now[0] += 2.5 * 86400
        self.assertEqual(self.m.sleep_if_due()["day"], 2)
        self.assertIsNone(self.m.sleep_if_due())

    def test_sleep_with_nothing_learned(self):
        self.assertIsNone(LearningModel(seed=0).sleep()["retrained"])


class TestFactUpdates(unittest.TestCase):
    def test_a_new_fact_replaces_an_old_one(self):
        m = model()
        m.respond("My sister lives in Delhi")
        reply, _ = m.respond("My sister lives in Mumbai")
        self.assertIn("Updated", reply)
        self.assertIn("Delhi", reply)
        self.assertEqual(m.respond("Where does my sister live?")[0], "Your sister lives in Mumbai.")
        self.assertIn("Got it", m.respond("My sister lives in Delhi")[0])  # it can be re-learned

    def test_things_you_can_have_several_of_are_kept(self):
        m = model()
        m.respond("I like tea")
        m.respond("I like coffee")
        self.assertEqual(m.respond("What do I like?")[0], "You like tea. You also like coffee.")


class TestDictionary(unittest.TestCase):
    def test_define_learns_a_word(self):
        m = model()
        self.assertEqual(m.define("zebra"), 1)
        self.assertEqual(m.respond("What is a zebra?")[0],
                         "Zebra is an African wild horse with black and white stripes.")
        self.assertEqual(m.define("qwertyx"), 0)
        self.assertIn("qwertyx", m.no_definition)

    def test_unknown_words_are_looked_up_when_asked(self):
        m = model()
        seen = []
        m.notify = seen.append
        self.assertIn("African wild horse", m.respond("What is a zebra?", use_web=True)[0])
        self.assertIn("Looking up 'zebra' in the dictionary...", seen)
        self.assertEqual(m.respond("Is a zebra a horse?")[0][:4], "Yes.")


class TestCuriosity(unittest.TestCase):
    def test_unanswered_questions_become_study_goals(self):
        m = model()
        m.respond("Where does the okapi live?")
        m.respond("Where does the okapi live?")
        self.assertEqual([(g["topic"], g["count"]) for g in m.gaps], [("okapi live", 2)])

    def test_curiosity_reads_up_and_resolves_the_gap(self):
        m = model()
        m.respond("Where does the okapi live?")
        report = m.be_curious(use_web=True)
        self.assertTrue(report[0]["resolved"])
        self.assertGreater(report[0]["learned"], 0)
        self.assertEqual(m.gaps, [])
        self.assertIn("Congo", m.respond("Where does the okapi live?")[0])

    def test_gives_up_after_a_few_tries(self):
        m = model()
        m.respond("What is the airspeed of a swallow?")
        for _ in range(3):
            self.assertFalse(m.be_curious()[0]["resolved"])
        self.assertEqual(m.gaps, [])

    def test_offline_failures_are_reported_not_raised(self):
        m = model()
        m.fetch = lambda url: (_ for _ in ()).throw(urllib.error.URLError("no network"))
        m.respond("What is the okapi?")
        report = m.be_curious(use_web=True)
        self.assertIn("no network", report[0]["error"])

    def test_study_session(self):
        m = model((NOTES, "notes.txt"))
        m.respond("What is the okapi?")
        result = m.study_session(use_web=True)
        self.assertEqual(set(result), {"curiosity", "exam", "sleep"})
        self.assertGreater(result["exam"]["total"], 0)
        self.assertEqual(result["sleep"]["day"], 1)
        self.assertEqual(m.progress()["day"], 1)


class TestPersistenceAndCompatibility(unittest.TestCase):
    def test_study_state_survives_save_and_load(self):
        m = model((long_page(), WEB))
        m.respond("What is a volcano?")
        m.respond("Where does the okapi live?")
        m.quiz(3)
        m.sleep()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "brain.json")
            m.save(path)
            loaded = LearningModel.load(path)
        self.assertEqual(loaded.shelf, m.shelf)
        self.assertEqual((loaded.day, loaded.gaps, loaded.exam_log),
                         (m.day, m.gaps, m.exam_log))
        self.assertEqual(loaded.knowledge, m.knowledge)
        self.assertEqual(loaded.interests, m.interests)
        self.assertEqual(loaded.stats()["shelf sentences"], len(m.shelf))

    def test_brains_from_before_study_mode_still_work(self):
        old = {"memories": [{"prompt": "hi", "response": "Hello!", "weight": 1.0}],
               "knowledge": [{"text": "Cats are mammals.", "source": "notes.txt", "pos": 0},
                             {"text": "Mammals are animals.", "source": "https://x.org", "pos": 1}]}
        m = LearningModel.from_dict(old)
        self.assertEqual(m.respond("hi")[0], "Hello!")
        self.assertEqual(m.quiz(5)["total"], 4)  # hi, cats, mammals, "is a cat an animal?"
        m.sleep()
        self.assertEqual(len(m.knowledge), 2)  # young web sentence hasn't faded yet

    def test_memory_matching_ignores_filler_words(self):
        m = model()
        m.learn("what is the capital of france", "Paris.")
        self.assertIsNone(m.respond("What is a cat?")[0])
        self.assertEqual(m.respond("capital of France?")[0], "Paris.")


if __name__ == "__main__":
    unittest.main()
