Introductions of English Wikipedia articles (Tiger, Great Wall of China, Mars, Titanic, Coffee), fetched
2026-10-06 and used under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/); authors are listed in
each article's history. questions.json: 55 questions written before the model was run.

This was an untouched HOLDOUT when first scored: 29/45 answerable right, 9/10 declined, 4 answered wrongly.
The code from before the real-text fixes scored the same 29/45 but answered 10 wrongly. Its misses have been
read since, so it is a development set now.
Run: python -m aimodel.devset sample_data/devset3 --misses
