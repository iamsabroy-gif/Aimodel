Human body facts written for Aimodel (original text, no copied sources; facts are standard
anatomy). The .txt files are study material; questions.json holds 71 questions (lookup) and 10 that the
documents cannot answer (abstain), in the same shape as the other dev sets.

Run:    python -m aimodel.devset sample_data/human_body --vectors builtin
Teach:  Train tab -> choose human_body_qa.csv (question/answer pairs), and add the .txt files as documents.
