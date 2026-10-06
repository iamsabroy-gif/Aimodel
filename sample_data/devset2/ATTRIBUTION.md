Introductions of English Wikipedia articles (Giraffe, Blue whale, Sahara, Eiffel Tower, Volcano, Nile),
fetched 2026-10-06 and used under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/); authors
are listed in each article's history. questions.json: 54 questions written before the model was run.

This set was the HOLDOUT for the first round of fixes: it scored 52% (23/44) before them and 59% (26/44)
after, with the code tuned on `devset/` only. Its misses were then read, so it is a second development
set now. Check real progress with articles that nobody has looked at.
Run: python -m aimodel.devset sample_data/devset2 --misses
