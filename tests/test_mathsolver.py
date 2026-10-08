import unittest

from aimodel import LearningModel
from aimodel.mathsolver import solve


class TestMathSolver(unittest.TestCase):
    def test_calculations(self):
        for text, answer in [("what is 12 * (3 + 4)?", "84"), ("15% of 200", "30"), ("square root of 144", "12"),
                             ("twelve plus five", "17"), ("what is 2^10", "1024"), ("10 / 4", "2.5"),
                             ("7 is what percent of 28", "25%"), ("average of 4, 8 and 12", "8"),
                             ("what is 3 squared", "9"), ("twenty one times two", "42"), ("5!", "120"),
                             ("What is 100 minus 37?", "63"), ("2 + 3 * 4", "14"), ("1/3 + 1/6", "0.5")]:
            self.assertEqual(solve(text)["answer"], answer, text)

    def test_steps_are_shown(self):
        self.assertEqual(solve("(3 + 4) * 5")["steps"], ["3 + 4 = 7", "7 * 5 = 35"])

    def test_division_by_zero_is_reported(self):
        self.assertIn("zero", solve("1/0")["error"])

    def test_ordinary_text_is_not_maths(self):
        for text in ["Rex is 5 years old", "capital of France", "hello", "I am 5 and she is 3", "42"]:
            self.assertIsNone(solve(text), text)

    def test_model_calculates_instead_of_searching(self):
        m = LearningModel(seed=0)
        reply, confidence = m.respond("what is 12 * (3 + 4)?")
        self.assertIn("84", reply)
        self.assertEqual(m.last_source, "calculation")
        self.assertEqual(confidence, 1.0)
        self.assertTrue(m.last_trace)

    def test_a_taught_memory_never_overrides_the_calculator(self):
        m = LearningModel(seed=0)
        m.learn("3+3", "6")
        for text, answer in [("5 + 7", "12"), ("what is 25 + 17?", "42"), ("add 5 and 7", "12"),
                             ("what is the sum of 12 and 30", "42"), ("how much is 9 + 8", "17")]:
            self.assertIn(answer, m.respond(text)[0], text)
            self.assertEqual(m.last_source, "calculation", text)


if __name__ == "__main__":
    unittest.main()
