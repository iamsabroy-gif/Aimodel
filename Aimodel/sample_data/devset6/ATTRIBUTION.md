Introductions of English Wikipedia articles (Kangaroo, Leonardo da Vinci, Mount Fuji, Oxygen, Tea), fetched
2026-10-06 and used under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/); authors are listed in
each article's history. questions.json: 58 questions written before the model was run.

Scored ONCE, with no tuning afterwards: 38/48 answerable right (79%), 5 wrong, 10/10 declined without word
vectors; 38/48, 6 wrong, 9/10 with them. The same code from before the real-text work scored 31/48. Its misses
have NOT been read, so this is still a clean holdout: do not tune against it, and say so if you read them.
Run: python -m aimodel.devset sample_data/devset6 [--vectors builtin]
