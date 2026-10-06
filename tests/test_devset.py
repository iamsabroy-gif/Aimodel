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
