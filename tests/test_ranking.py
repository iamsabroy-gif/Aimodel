import unittest

from aimodel import ranking
from aimodel.model import RANK_WEIGHTS_PATH, LearningModel, _learned_weights

DOC = ("Penguins are flightless birds that live in the Southern Hemisphere. "
       "Most penguins feed on krill, fish and squid. Some penguins are tall. "
       "Emperor penguins can live for twenty years.")


def row(question, cands):
    return {"question": question, "exists": True, "cands": cands}


class TestLearnedRanking(unittest.TestCase):
    def test_the_shipped_weights_cover_every_feature(self):
        weights = _learned_weights()
        self.assertIsNotNone(weights, RANK_WEIGHTS_PATH)
        self.assertEqual(set(ranking.FEATURES) - set(weights), set())
        self.assertTrue(all(abs(v) <= ranking.MAX_WEIGHT for v in weights.values()))

    def test_training_gives_weight_to_what_separates_right_from_wrong(self):
        base = {n: 0.0 for n in ranking.FEATURES}
        rows = []
        for i in range(12):  # "content" is what makes a sentence right; "length" is noise
            rows.append(row(f"q{i}", [({**base, "content": 1.0, "length": 0.3 + 0.1 * (i % 3)}, True),
                                      ({**base, "content": 0.0, "length": 0.5}, False),
                                      ({**base, "content": 0.0, "length": 0.2 + 0.1 * (i % 2)}, False)]))
        weights = ranking.train(rows)
        self.assertGreater(weights["content"], 1.0)
        self.assertGreater(weights["content"], abs(weights["length"]) * 3)
        self.assertEqual(ranking.metrics(rows, weights)["first"], 12)
        self.assertEqual(ranking.metrics(rows, {"content": -1.0})["first"], 0)  # ranking the wrong way round is worse

    def test_it_refuses_to_learn_from_nothing(self):
        with self.assertRaises(ValueError):
            ranking.train([row("q", [({n: 0.0 for n in ranking.FEATURES}, False)])])

    def test_answers_work_with_and_without_learned_weights(self):
        for learned in (True, False):
            m = LearningModel(seed=0)
            if not learned:
                m.learned_weights = None
            m.add_document(DOC, "p.txt")
            self.assertIn("krill", m.respond("What do penguins eat?", learn=False)[0])
            self.assertIn("Southern Hemisphere", m.respond("Where do penguins live?", learn=False)[0])

    def test_the_best_sentence_comes_first(self):
        m = LearningModel(seed=0)
        m.add_document(DOC, "p.txt")
        reply = m.answer_from_knowledge("What do penguins eat?")[0]
        self.assertTrue(reply.startswith("Most penguins feed on krill"))


if __name__ == "__main__":
    unittest.main()
