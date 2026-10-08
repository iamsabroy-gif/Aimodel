Introductions of English Wikipedia articles (Albert Einstein, Bread, Great Barrier Reef, Jupiter, Mount
Kilimanjaro), fetched 2026-10-06 and used under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/);
authors are listed in each article's history. questions.json: 57 questions written before the model was run.

Scored ONCE with the code after the devset6 fixes: 35/47 answerable right, 2 wrong, 10/10 declined without word
vectors; 38/47, 1 wrong, 8/10 with them. The code before those fixes scored 30/47 and 34/47 on the same set.
Its misses have NOT been read, so this is a clean holdout: do not tune against it, and say so if you read them.
Run: python -m aimodel.devset sample_data/devset7 [--vectors builtin]
