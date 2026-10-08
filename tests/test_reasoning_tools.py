import unittest

from aimodel import LearningModel
from aimodel.quantities import Quantities, find_names
from aimodel.relations import Relations

FACTS = [
    "Mount Everest is 8849 meters high.", "K2 is 8611 meters high.", "Mont Blanc is 4806 meters high.",
    "The Nile is 6650 kilometers long.", "The Thames is 346 kilometers long.",
    "Maple School has 450 students.", "Oak School has 620 students.",
    "Maple School teaches French, Spanish and Latin.", "Oak School teaches German and Spanish.",
    "Anna is the mother of Ben.", "Ben is the father of Carla.", "Dev is the brother of Ben.",
    "Carla lives in Paris.", "Paris is the capital of France.", "France is in Europe.",
    "Dev lives in Delhi.", "Delhi is in India.",
    "The Johnson family has three children: Ann, Bob and Cleo.",
    "Ann is older than Bob.", "Bob is older than Cleo.",
]


def source(text):
    """Documents are named after their topic, which is how 'the shortest river' finds the rivers."""
    return next((f"{name}.txt" for word, name in (("kilometers", "Rivers"), ("meters", "Mountains"),
                                                  ("students", "Schools")) if word in text), "Families.txt")


def sentences():
    return [{"text": t, "source": source(t)} for t in FACTS]


class TestQuantities(unittest.TestCase):
    def setUp(self):
        self.q = Quantities(sentences())

    def reply(self, question):
        got = self.q.answer(question)
        return got and got[0]

    def test_comparing_and_ranking(self):
        self.assertIn("Everest", self.reply("Which is higher, Everest or K2?"))
        self.assertTrue(self.reply("Is Mont Blanc higher than K2?").startswith("No"))
        self.assertIn("Thames", self.reply("What is the shortest river?"))
        self.assertIn("Oak School", self.reply("Which school has more students, Maple School or Oak School?"))
        self.assertEqual(self.reply("How many mountains are higher than 5000 meters?").split(":")[0], "Two mountains")

    def test_counting_and_lists(self):
        self.assertIn("three languages", self.reply("How many languages does Maple School teach?"))
        self.assertIn("three children", self.reply("How many children does the Johnson family have?"))
        self.assertTrue(self.reply("Does Oak School teach French?").startswith("No"))
        self.assertTrue(self.reply("Does Maple School teach Latin?").startswith("Yes"))

    def test_unclear_questions_are_left_alone(self):
        for q in ["Which mountain is the most beautiful?", "Which school is the best?", "What is the highest river?",
                  "Who is the tallest of the Johnson children?"]:
            self.assertIsNone(self.reply(q), q)

    def test_a_clause_is_not_a_list(self):
        q = Quantities([{"text": "An adult has 32 teeth, and a child has 20 baby teeth.", "source": "x"}])
        self.assertEqual(q.lists, [])

    def test_short_names_need_to_be_unambiguous(self):
        self.assertEqual(find_names("Which school has more?", ["Maple School", "Oak School"]), [])
        self.assertEqual(find_names("Is Everest high?", ["Mount Everest"]), ["Mount Everest"])


class TestRelations(unittest.TestCase):
    def setUp(self):
        self.r = Relations(sentences())

    def reply(self, question):
        got = self.r.answer(question)
        return got and got[0]

    def test_chains(self):
        self.assertIn("France", self.reply("In which country does Ben's daughter live?"))
        self.assertIn("India", self.reply("In which country does Ben's brother live?"))
        self.assertIn("Europe", self.reply("On which continent does Carla live?"))
        self.assertTrue(self.reply("Does Carla live in Europe?").startswith("Yes"))

    def test_ordering(self):
        self.assertEqual(self.reply("Who is the youngest of the Johnson children?"), "Cleo.")
        self.assertEqual(self.reply("Who is the oldest of the Johnson children?"), "Ann.")

    def test_a_hedged_no_only_when_everything_points_elsewhere(self):
        self.assertTrue(self.r.closed_world("Does Carla live in India?")[0].startswith("No, not as far as I know"))
        self.assertIsNone(self.r.closed_world("Does Zed live in India?"))

    def test_plain_lookups_are_left_to_the_ordinary_reasoning(self):
        self.assertIsNone(self.reply("Where does Dev live?"))
        self.assertIsNone(self.reply("Does Dev live in Delhi?"))


class TestInTheModel(unittest.TestCase):
    def test_the_model_uses_them(self):
        m = LearningModel(seed=0)
        for name, words in (("Mountains.txt", "meters"), ("Rivers.txt", "kilometers"), ("Schools.txt", "students")):
            m.add_document(" ".join(t for t in FACTS if source(t) == name), name)
        m.add_document(" ".join(t for t in FACTS if source(t) == "Families.txt"), "Families.txt")
        self.assertIn("Everest", m.respond("Which is higher, Mount Everest or K2?", learn=False)[0])
        self.assertIn("France", m.respond("In which country does Ben's daughter live?", learn=False)[0])
        self.assertIn("Cleo", m.respond("Who is the youngest of the Johnson children?", learn=False)[0])
        self.assertIn("Everest", m.respond("What is the highest mountain?", learn=False)[0])


if __name__ == "__main__":
    unittest.main()
