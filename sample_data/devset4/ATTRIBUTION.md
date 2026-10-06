Introductions of English Wikipedia articles (Banana, Elephant, Honey, Saturn, Statue of Liberty), fetched
2026-10-06 and used under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/); authors are listed in
each article's history. questions.json: 53 questions written before the model was run.

This was an untouched HOLDOUT for the word-vector work: the vectors were tuned on devset, devset2 and devset3
only, and this set was scored once afterwards. Its misses have not been read yet.
Run: python -m aimodel.devset sample_data/devset4 --vectors builtin
