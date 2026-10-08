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


class TestStories(unittest.TestCase):
    def test_simple_stories_are_worked_out(self):
        from aimodel.wordproblems import solve_story
        for text, answer in [
            ("Tom has 5 apples and buys 3 more. How many apples does he have now?", "8"),
            ("Tom has 5 apples. He eats 2. How many apples are left?", "3"),
            ("A shop had 50 pens. It sold 20 and then received 30 more. How many pens does it have now?", "60"),
            ("There are 4 boxes with 6 pencils each. How many pencils are there in total?", "24"),
            ("24 sweets are shared equally among 6 children. How many sweets does each child get?", "4"),
            ("Ann has 12 books. Bob has 8 books. How many more books does Ann have than Bob?", "4"),
            ("There are 15 birds on a tree. 6 fly away. How many birds remain?", "9")]:
            self.assertEqual(solve_story(text)["answer"], answer, text)

    def test_unclear_stories_are_not_guessed(self):
        from aimodel.wordproblems import solve_story
        for text in ["Tom has 5 apples and 3 oranges. How many apples does he have?",
                     "Tom has 5 apples. He does something with 2. How many are left?",
                     "The Nile is 6650 kilometers long. How long is it?", "Tom has 5 apples."]:
            self.assertIsNone(solve_story(text), text)

    def test_the_model_answers_a_story(self):
        m = LearningModel(seed=0)
        reply = m.respond("Tom has 5 apples and buys 3 more. How many apples does he have now?")[0]
        self.assertIn("8", reply)
        self.assertEqual(m.last_source, "calculation")


class TestUnits(unittest.TestCase):
    def test_conversions(self):
        from aimodel.units import convert
        for text, answer in [("convert 5 km to miles", "3.10686 miles"), ("how many meters in 3 km?", "3000 meters"),
                             ("how many minutes are in 2 hours", "120 minutes"), ("what is 12 inches in feet", "1 feet"),
                             ("100 degrees celsius to fahrenheit", "212 fahrenheit"), ("two hours in minutes", "120 minutes")]:
            self.assertEqual(convert(text)["answer"], answer, text)

    def test_things_that_are_not_conversions(self):
        from aimodel.units import convert
        for text in ["5 km in France", "convert 5 kg to meters", "how many days in a year", "hello"]:
            self.assertIsNone(convert(text), text)
