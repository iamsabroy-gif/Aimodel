import os
import tempfile
import unittest

from aimodel import LearningModel
from aimodel.smalltalk import INTENTS, PHRASES, render


def model():
    m = LearningModel(seed=0)
    m.add_document("Rex is a golden retriever. The capital of France is Paris.", "notes.txt")
    return m


class TestGreetings(unittest.TestCase):
    def reply(self, m, text):
        return m.respond(text)[0]

    def test_greetings_are_answered_without_any_teaching(self):
        m = LearningModel(seed=0)
        for text, expect in [("hello", ["hello", "hi", "hey"]), ("Hi there!", ["hello", "hi", "hey"]),
                             ("good morning", ["morning"]), ("good evening", ["evening"]),
                             ("good night", ["night"]), ("how are you?", ["well", "great", "good", "doing"]),
                             ("what's up", ["up", "learning", "studying"]),
                             ("thank you so much", ["welcome", "happy", "anytime"]),
                             ("goodbye", ["goodbye", "see you", "take care"]),
                             ("nice to meet you", ["meet"]), ("sorry", ["problem", "okay"]),
                             ("who are you?", ["aimodel"]), ("what can you do?", ["answer", "read"])]:
            reply = self.reply(m, text)
            self.assertIsNotNone(reply, text)
            self.assertTrue(any(w in reply.lower() for w in expect), (text, reply))
            self.assertEqual(m.last_source, "smalltalk")
            self.assertNotIn("{name}", reply)

    def test_stretched_and_shortened_forms(self):
        m = LearningModel(seed=0)
        for text in ["heyyy", "hii", "HELLO!!!", "thx", "byeee", "gm", "wassup", "Hi buddy", "hello everyone"]:
            self.assertIsNotNone(self.reply(m, text), text)

    def test_real_questions_are_never_mistaken_for_small_talk(self):
        m = model()
        for text in ["What is the name of the capital?", "hello world program", "What is the capital of France?",
                     "tell me about the good things", "who made the capital"]:
            self.reply(m, text)
            self.assertNotEqual(m.last_source, "smalltalk", text)
        self.assertIn("Paris", self.reply(m, "What is the capital of France?"))

    def test_a_greeting_followed_by_a_question(self):
        m = model()
        reply = self.reply(m, "Hi, what is Rex?")
        self.assertIn("golden retriever", reply)
        self.assertTrue(reply.lower().startswith(("hello", "hi", "hey")))
        both = self.reply(m, "Hi, how are you?")
        self.assertGreater(len(both), 40)
        self.assertEqual(m.last_source, "smalltalk")

    def test_a_greeting_with_a_question_i_cannot_answer_is_not_swallowed(self):
        m = LearningModel(seed=0)
        self.assertIsNone(self.reply(m, "hi, what is the airspeed of a swallow?"))

    def test_replies_you_taught_win(self):
        m = LearningModel(seed=0)
        m.learn("hello", "Yo, what's good?")
        self.assertEqual(self.reply(m, "hello"), "Yo, what's good?")
        self.assertEqual(m.last_source, "memory")

    def test_greetings_do_not_clutter_what_it_learns(self):
        m = model()
        before = (len(m.facts), len(m.gaps), len(m.knowledge))
        for text in ["hello", "I'm fine", "how are you", "bye"]:
            self.reply(m, text)
        self.assertEqual((len(m.facts), len(m.gaps), len(m.knowledge)), before)


class TestName(unittest.TestCase):
    def test_it_learns_your_name_and_uses_it(self):
        m = LearningModel(seed=0)
        self.assertIsNone(m.user_name())
        self.assertNotIn("Sam", m.respond("hello")[0])
        reply = m.respond("My name is Sam")[0]
        self.assertEqual(reply, "Got it: Your name is Sam. Nice to meet you!")
        self.assertEqual(m.user_name(), "Sam")
        seen = {m.respond("hello")[0] for _ in range(40)}
        self.assertTrue(all("Sam" in r or "Ask me" in r or "Hey" in r or "Hi" in r for r in seen))
        self.assertTrue(any("Sam" in r for r in seen))

    def test_name_forms_and_corrections(self):
        for text, name in [("call me Alex", "Alex"), ("I'm Priya", "Priya"), ("my name is raj", "Raj"),
                           ("I am Sam", "Sam")]:
            m = LearningModel(seed=0)
            m.respond(text)
            self.assertEqual(m.user_name(), name, text)
        m = LearningModel(seed=0)
        m.respond("My name is Sam")
        reply = m.respond("Actually call me Sammy")[0]
        self.assertEqual(m.user_name(), "Sammy")
        self.assertIn("Updated", reply)

    def test_feelings_and_chores_are_not_names(self):
        m = LearningModel(seed=0)
        for text in ["I'm Fine", "I am Happy", "call me later", "I'm tired"]:
            m.respond(text)
            self.assertIsNone(m.user_name(), text)

    def test_your_name_is_left_out_of_greetings_you_type(self):
        m = LearningModel(seed=0)
        m.respond("My name is Sam")
        self.assertIsNotNone(m.respond("hello Sam")[0])
        self.assertEqual(m.last_source, "smalltalk")

    def test_render_without_a_name_leaves_no_stray_punctuation(self):
        self.assertEqual(render("Hello {name}! Hi.", None), "Hello! Hi.")
        self.assertEqual(render("Good night, {name}! Sleep well.", None), "Good night! Sleep well.")
        self.assertEqual(render("Good night, {name}! Sleep well.", "Sam"), "Good night, Sam! Sleep well.")
        self.assertEqual(render("See you, {name}.", None), "See you.")
        for _phrases, replies in INTENTS.values():
            for r in replies:
                self.assertNotIn("{", render(r, None))
                self.assertNotIn(" ,", render(r, None))
                self.assertNotIn(" !", render(r, None).replace("Bonjour !", ""))


class TestTraining(unittest.TestCase):
    def test_you_can_teach_a_greeting_in_any_language(self):
        m = LearningModel(seed=0)
        self.assertIsNone(m.respond("kem cho")[0])
        m.add_greeting("kem cho", "Majama {name}!")
        self.assertEqual(m.respond("kem cho")[0], "Majama!")
        m.respond("My name is Sam")
        self.assertEqual(m.respond("Kem cho!")[0], "Majama Sam!")
        self.assertEqual(m.respond("kem choooo")[0], "Majama Sam!")

    def test_your_replies_are_used_first_for_a_phrase_you_taught(self):
        m = LearningModel(seed=0)
        m.add_greeting("hello", "Greetings, traveller.")
        self.assertEqual({m.respond("hello")[0] for _ in range(10)}, {"Greetings, traveller."})
        m.add_greeting("hello", "Well met.")
        self.assertEqual({m.respond("hello")[0] for _ in range(60)}, {"Greetings, traveller.", "Well met."})

    def test_forgetting_a_greeting(self):
        m = LearningModel(seed=0)
        m.add_greeting("kem cho", "Majama!")
        self.assertEqual(m.forget_greeting("Kem cho"), 1)
        self.assertEqual(m.forget_greeting("kem cho"), 0)
        self.assertIsNone(m.respond("kem cho")[0])
        with self.assertRaises(ValueError):
            m.add_greeting("   ", "x")
        with self.assertRaises(ValueError):
            m.add_greeting("so very much", "x")  # only filler words: nothing to recognise

    def test_good_and_bad_change_what_it_says(self):
        m = LearningModel(seed=0)
        liked, disliked = INTENTS["hello"][1][:2]
        for _ in range(6):
            m._last_talk, m.last_source = [disliked], "smalltalk"
            m.feedback(good=False)
            m._last_talk, m.last_source = [liked], "smalltalk"
            m.feedback(good=True)
        self.assertEqual(m.smalltalk_weights[liked], 7.0)
        self.assertEqual(m.smalltalk_weights[disliked], 0.05)  # a floor: it is never silenced completely
        counts = {}
        for _ in range(400):
            reply = m.respond("hello")[0]
            counts[reply] = counts.get(reply, 0) + 1
        self.assertGreater(counts[render(liked, None)], 4 * counts.get(render(disliked, None), 0))
        self.assertGreaterEqual(len(counts), 3)  # still varied

    def test_feedback_after_a_greeting_works_like_everywhere_else(self):
        m = LearningModel(seed=0)
        m.respond("hello")
        self.assertTrue(m.feedback(good=True))
        self.assertTrue(any(w > 1 for w in m.smalltalk_weights.values()))
        m.respond("hello")
        m.correct("hello", "Salutations!")  # /bad + the right answer: now a taught reply
        self.assertEqual(m.respond("hello")[0], "Salutations!")

    def test_greetings_name_and_weights_survive_saving(self):
        m = LearningModel(seed=0)
        m.add_greeting("kem cho", "Majama {name}!")
        m.respond("My name is Sam")
        m.respond("hello")
        m.feedback(good=True)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "brain.json")
            m.save(path)
            loaded = LearningModel.load(path)
        self.assertEqual(loaded.smalltalk_custom, m.smalltalk_custom)
        self.assertEqual(loaded.smalltalk_weights, m.smalltalk_weights)
        self.assertEqual(loaded.user_name(), "Sam")
        self.assertEqual(loaded.respond("kem cho")[0], "Majama Sam!")
        self.assertEqual(loaded.stats()["greetings you taught"], 1)

    def test_every_phrase_belongs_to_one_kind_and_has_replies(self):
        self.assertGreater(len(PHRASES), 150)
        for name, (phrases, replies) in INTENTS.items():
            self.assertTrue(phrases and replies, name)

    def test_exams_ignore_greetings(self):
        m = model()
        m.respond("hello")
        self.assertTrue(all(q["kind"] != "memory" for q in m.make_questions(20)))


if __name__ == "__main__":
    unittest.main()
