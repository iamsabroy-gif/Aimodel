Introductions of English Wikipedia articles (Chocolate, Dolphin, Great Pyramid of Giza, Rainbow, Venus), fetched
2026-10-06 and used under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/); authors are listed in
each article's history. questions.json: 55 questions written before the model was run.

This was an untouched HOLDOUT for the devset4 fixes: scored once afterwards (34/45 answerable right, 3 wrong,
9/10 declined, with word vectors). Its misses have not been read, so it is still a clean holdout.
Run: python -m aimodel.devset sample_data/devset5 --vectors builtin
