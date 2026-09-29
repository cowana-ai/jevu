"""Minimal end-to-end example.

Run with real JEV + embeddings by setting OPENROUTER_API_KEY (and providing your own
embeddings). This script uses tiny synthetic embeddings + explicit labels so it runs
offline; swap in your model's embeddings and a `texts=` argument to use JEV labeling.
"""
import numpy as np

from jevu import ConceptScrubber, concept_auc

# --- your data ---
rng = np.random.default_rng(0)
n, d = 500, 32
gender = rng.integers(0, 2, size=n)                    # the concept to erase (here: known)
embeddings = rng.standard_normal((n, d))
embeddings[:, 0] += 4.0 * gender                       # pretend gender lives on dim 0

# --- erase the concept ---
# With JEV instead of explicit labels you would write:
#   scrubber = ConceptScrubber(concept="Does the text describe a woman?", method="leace")
#   clean = scrubber.fit_transform(embeddings, texts=texts)     # needs OPENROUTER_API_KEY
scrubber = ConceptScrubber(concept="Does the text describe a woman?", method="leace")
clean = scrubber.fit_transform(embeddings, labels=gender)

print("concept AUC before:", round(concept_auc(embeddings, gender), 3))
print("concept AUC after :", round(concept_auc(clean, gender), 3))
