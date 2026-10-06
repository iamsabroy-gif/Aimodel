import unittest

from aimodel import devset
from aimodel.model import LearningModel


class TestRealText(unittest.TestCase):
    """Real Wikipedia intros (sample_data/devset). The code was tuned on this set, so these are floors
    that stop it from getting worse, not proof that it works on text it has never seen."""

    @classmethod
    def setUpClass(cls):
        cls.result = devset.run()

    def test_answers_enough_and_declines_what_it_cannot_know(self):
        ok, total = self.result["answerable"]
        self.assertGreaterEqual(ok, 33, f"{ok}/{total}")
        declined, asked = self.result["declined_correctly"]
        self.assertEqual(declined, asked)

    def test_it_is_resolved_to_the_thing_the_text_is_about(self):
        m = LearningModel(seed=0)
        m.add_document("Mount Everest is the highest mountain on Earth. It lies in the Himalayas. "
                       "Its height was measured in 2020 as 8,848 m.", "everest.txt")
        self.assertIn("Himalayas", m.respond("Where is Mount Everest?", learn=False)[0])
        self.assertIn("8,848", m.respond("How tall is Mount Everest?", learn=False)[0])

    def test_amounts_need_the_right_kind_of_number(self):
        m = LearningModel(seed=0)
        m.add_document("Mount Everest is the highest mountain on Earth. Its height was measured in 2020 "
                       "as 8,848 m. Many climbers visit Everest every year.", "everest.txt")
        self.assertIsNone(m.respond("How old is Mount Everest?", learn=False)[0])  # no age given
        self.assertIsNone(m.respond("What is the population of Nepal?", learn=False)[0])

    def test_similar_words_find_the_sentence(self):
        m = LearningModel(seed=0)
        m.add_document("Penguins are flightless birds. Most penguins feed on krill and fish.", "p.txt")
        self.assertIn("krill", m.respond("What do penguins eat?", learn=False)[0])

    def test_a_sentence_that_says_it_answers_yes_no(self):
        m = LearningModel(seed=0)
        m.add_document("All octopuses are venomous, but only a few are deadly to humans.", "o.txt")
        self.assertTrue(m.respond("Are octopuses venomous?", learn=False)[0].startswith("Probably yes"))


if __name__ == "__main__":
    unittest.main()


class TestWhoIsIt(unittest.TestCase):
    name = staticmethod(LearningModel._name_the_subject)

    def test_a_later_it_is_the_subject_of_the_text(self):
        self.assertEqual(self.name("At 7,088 km long, it is the longest river.", "The Nile", "The Nile"),
                         "At 7,088 km long, the Nile is the longest river.")
        self.assertEqual(self.name("Reaching 30 m long and weighing 190 t, it is the largest animal.",
                                   "blue whale", "The blue whale"),
                         "Reaching 30 m long and weighing 190 t, the blue whale is the largest animal.")

    def test_it_is_left_alone_when_it_may_mean_something_else(self):
        for sentence, subject, about in [
                ("The siphon is used for respiration and it helps.", "The siphon", "An octopus"),
                ("It is thought to be old.", "The Nile", "The Nile"),
                ("I like it.", "The Nile", "The Nile"),
                ("To plant a seed, dig a hole, then cover it with soil.", None, None)]:
            self.assertEqual(self.name(sentence, subject, about), sentence)

    def test_a_how_to_is_not_rewritten(self):
        m = LearningModel(seed=0)
        m.add_document("To plant a seed, dig a small hole. Place the seed inside. Cover it with soil.", "h.txt")
        self.assertTrue(any("Cover it with soil" in k["text"] for k in m.knowledge))


class TestRealTextRules(unittest.TestCase):
    def ask(self, text, question):
        m = LearningModel(seed=0)
        m.add_document(text, "t.txt")
        return m.respond(question, learn=False)[0]

    def test_a_short_aside_in_brackets_stays_in_the_answer(self):
        reply = self.ask("Bees produce honey by gathering the sugary secretions of plants (primarily floral nectar).",
                         "How do bees make honey?")
        self.assertIn("(primarily floral nectar)", reply)
        self.assertNotIn("Apis", self.ask("Honey is made by bees (Apis mellifera) and stored in hives.",
                                          "Who makes honey?") or "")

    def test_a_when_question_needs_a_date(self):
        text = "The statue is a colossal sculpture in New York. The statue was dedicated on October 28, 1886."
        self.assertIn("1886", self.ask(text, "When was the statue dedicated?"))
        self.assertIsNone(self.ask("The statue is a colossal sculpture in New York.", "When was the statue dedicated?"))

    def test_how_old_needs_an_age_not_any_number_of_years(self):
        text = "Saturn is the sixth planet. Saturn orbits the Sun with an orbital period of 29.45 years."
        self.assertIsNone(self.ask(text, "How old is Saturn?"))
        self.assertIn("4.5 billion years ago", self.ask(text + " Saturn formed 4.5 billion years ago.",
                                                        "How old is Saturn?"))

    def test_other_forms_of_a_word_match(self):
        self.assertIn("Bartholdi", self.ask("The statue was designed by the sculptor Bartholdi.",
                                            "Who created the statue?"))
        self.assertIn("1912", self.ask("The Titanic sank in 1912.", "When did the Titanic sink?"))
        self.assertIn("plantains", self.ask("Cooking bananas are called plantains in some countries.",
                                            "What is a plantain?"))

    def test_how_long_to_live_asks_for_a_time_not_a_length(self):
        text = "Calves rely on their mothers for as long as three years. Elephants can live up to 70 years."
        self.assertIn("70 years", self.ask(text, "How long can elephants live?"))


BIO = ("Leonardo da Vinci (15 April 1452 – 2 May 1519) was an Italian polymath of the High Renaissance. "
       "Born out of wedlock in Vinci in Tuscany, he was educated in Florence by the painter Verrocchio. "
       "Upon the invitation of Francis I, he spent his last three years in France, where he died in 1519. "
       "Revered for his ingenuity, he conceptualised flying machines.")


class TestPeople(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = LearningModel(seed=0)
        cls.m.add_document(BIO, "Leonardo_da_Vinci.txt")

    def ask(self, q):
        return self.m.respond(q, learn=False)[0]

    def test_he_is_the_person_the_text_is_about(self):
        texts = [k["text"] for k in self.m.knowledge]
        self.assertIn("Upon the invitation of Francis I, Leonardo da Vinci spent his last three years in France, "
                      "where he died in 1519.", texts)  # "Francis I" is a name, not "I"
        self.assertTrue(any(t.startswith("Born out of wedlock") and "Leonardo da Vinci was educated" in t
                            for t in texts))

    def test_dates_in_brackets_answer_when_questions(self):
        self.assertIn("1452", self.ask("When was Leonardo da Vinci born?"))
        self.assertIn("(15 April 1452 – 2 May 1519)", self.ask("When was Leonardo da Vinci born?"))

    def test_where_and_who_questions_use_the_sentence_about_it(self):
        self.assertIn("France", self.ask("Where did Leonardo da Vinci die?"))
        self.assertIn("Verrocchio", self.ask("Who taught Leonardo da Vinci?"))
        self.assertIn("flying machines", self.ask("What did Leonardo conceptualise?"))

    def test_a_vector_stand_in_does_not_make_a_word_rare(self):
        m = LearningModel(seed=0)
        m.load_builtin_vectors()
        m.add_document("Mount Fuji is an active stratovolcano with a summit elevation of 3,776 m.", "fuji.txt")
        m.add_document("Oxygen is a colorless gas at standard temperature and pressure.", "oxygen.txt")
        self.assertIsNone(m.respond("What is the temperature at Mount Fuji?", learn=False)[0])
        # "summit" has a vector stand-in ("meeting") that I never read; that must not make it the rarest word
        key = m._key_word(m._wanted("What is the temperature at the summit of Mount Fuji?"))
        self.assertEqual(key, "temperature")

    def test_a_unit_at_the_end_of_a_sentence_is_not_an_abbreviation(self):
        from aimodel.text import split_sentences
        self.assertEqual(split_sentences("It rises to 3,776 m. Water boils there at a lower temperature."),
                         ["It rises to 3,776 m.", "Water boils there at a lower temperature."])
        self.assertEqual(len(split_sentences("Use a tool, e.g. a hammer, to do it.")), 1)
