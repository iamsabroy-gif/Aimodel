import unittest

from aimodel.evaluation import benchmark_model
from aimodel.understand import fix_typos, members_question, use_common_words

VOCAB = {"where", "were", "does", "the", "lorpan", "live", "seed", "seeds", "animals", "animal"}


class TestUnderstand(unittest.TestCase):
    def test_typos_are_fixed_but_real_words_and_names_are_left_alone(self):
        self.assertEqual(fix_typos("Whre does the lorpan live?", VOCAB), "Where does the lorpan live?")
        self.assertEqual(fix_typos("Where does the lorpam live?", VOCAB), "Where does the lorpan live?")
        self.assertEqual(fix_typos("What is the speed of light?", VOCAB), "What is the speed of light?")
        self.assertEqual(fix_typos("Is Zorbo an animal?", VOCAB), "Is Zorbo an animal?")

    def test_synonyms(self):
        self.assertEqual(use_common_words("Where does it dwell?"), "Where does it live?")

    def test_set_questions_are_recognised(self):
        self.assertEqual(members_question("Which animals are birds?"), ("animal", "bird"))
        self.assertEqual(members_question("Name all the birds."), ("", "bird"))
        self.assertIsNone(members_question("What is a bird?"))
        self.assertIsNone(members_question("Which is bigger?"))


class TestReasoning(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = benchmark_model()

    def ask(self, q):
        return self.m.respond(q, learn=False)[0]

    def test_typo_and_synonym_questions(self):
        self.assertIn("mountains", self.ask("Whre does the lorpan live?"))
        self.assertIn("mountains", self.ask("Where does the lorpan dwell?"))

    def test_which_and_name_questions_list_every_member(self):
        for q in ("Which animals are birds?", "Name all the birds."):
            reply = self.ask(q)
            self.assertIn("quillet", reply)
            self.assertIn("druxle", reply)
            self.assertNotIn("mimbat", reply)

    def test_either_or_questions(self):
        reply = self.ask("Is the mimbat a bird or a mammal?")
        self.assertIn("mammal", reply)
        self.assertNotIn("quillet", reply)

    def test_unknown_things_are_still_declined(self):
        for q in ("What is the speed of light?", "What does the glorb eat?"):
            self.assertIsNone(self.ask(q))


if __name__ == "__main__":
    unittest.main()


class TestClaimsAndComparisons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m = benchmark_model()

    def ask(self, q):
        return self.m.respond(q, learn=False)[0]

    def test_does_questions_say_yes_and_no_honestly(self):
        self.assertTrue(self.ask("Does the lorpan eat grass?").startswith("Yes"))
        no = self.ask("Does the lorpan eat meat?")
        self.assertTrue(no.startswith("No, not as far as I know"))
        self.assertIn("grass", no)  # it says what it does know
        self.assertIsNone(self.ask("Does the glorb eat meat?"))  # never heard of it: no guess

    def test_several_subjects_are_not_misread_as_one(self):
        self.assertNotIn("not as far as I know",
                         self.ask("Do the mimbat and the lorpan both live in mountains?") or "")

    def test_what_two_things_have_in_common(self):
        self.assertIn("animals", self.ask("What do mammals and birds have in common?"))
        reply = self.ask("How are the mimbat and the quillet alike?")
        self.assertTrue(reply.startswith("The mimbat and the quillet are both animals"))
        self.assertNotIn("lorpan", reply)
        self.assertIsNone(self.ask("What do cats and glorbs have in common?"))
