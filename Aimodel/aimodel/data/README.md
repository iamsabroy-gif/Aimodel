glove50_30k.npz: the 30,000 most common words of GloVe 6B (Wikipedia 2014 + Gigaword 5), 50 dimensions,
stored as unit vectors in float16. GloVe: Pennington, Socher and Manning, 2014 (https://nlp.stanford.edu/projects/glove/),
released under the Public Domain Dedication and License v1.0.
Rebuild with: python -m aimodel.make_vectors glove.6B.zip --name glove.6B.50d.txt --words 30000
