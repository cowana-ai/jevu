# Erasing and Suppressing Concepts in Embeddings

*How to remove a protected attribute like gender from text embeddings, or keep unsafe content out of
retrieval, running everything on your own machine. I'll focus on the intuition and the practical use
cases: when you want to **erase** a concept, and when you actually want to **suppress** it instead.*

---

Pre-trained embeddings carry a lot more than the task you care about. Take any off-the-shelf model,
train a simple linear probe on top of the frozen vectors, and you can read a person's gender off a
professional bio at around 0.99 AUC. Do the same on user comments and the probe tells you whether the
text is toxic at around 0.92. You never trained the embeddings for either of these — the information is
just sitting there. So anything you build on top (a classifier, a search ranker, a recommender) picks it
up too, along with whatever bias or liability comes with it.

Now, there are two different things you might want to do about a concept like this, and in practice
people mix them up:

- **Erase it.** Make the concept unreadable so nothing downstream can use it. This is what you want for
  a *protected* attribute: your search ranker should not be able to tell gender at all.
- **Suppress it.** Keep the concept's signal and use it to *exclude* matching items. This is what you
  want for *safety*: unsafe content should not show up, which means you need to detect it, not forget it.

These are opposite operations on the same quantity, and that distinction is the thing I want to get
across. The good news is that both come out of a single fit, and both run locally. Let me build up the
method first, then go through each use case with real numbers and a few examples.

## A local pipeline

One practical constraint drives the whole design: the text you most want to control (toxic, NSFW,
anything PII-adjacent) is exactly the text you can't send off to a hosted API. So every piece runs
on-device:

- local embeddings (I used `sentence-transformers/all-MiniLM-L6-v2`),
- a local scorer, [laya](https://huggingface.co/convaiinnovations/laya), which takes a text plus a yes/no
  question and returns a calibrated probability in `[0, 1]` — no labels, no thresholds, no network,
- a pure-numpy eraser, [LEACE](https://arxiv.org/abs/2306.03819).

Nothing leaves your machine, which is really the whole point when the data is sensitive.

## How erasure works

A concept is *linearly encoded* in an embedding `x` when some direction `û` predicts it: a probe reads
`σ(wᵀx)`. If you strip out the part of `x` that lies along `û`, the probe just sees a constant and the
concept becomes unreadable along that axis.

The obvious thing to try is to subtract that direction, `x − (ûᵀx) û`. That only behaves well if the
coordinates of your embedding are uncorrelated and have equal variance, and they never are. Embeddings
sit in a stretched, tilted ellipse, so projecting along a raw axis also drags along every direction
correlated with it, and you end up removing more than you meant to.

The key idea is to first move into a space where the cloud is a round ball, because *there* a plain
perpendicular projection is the least-damaging way to delete a direction. That move is whitening: if
`Σ = cov(X)`, then `W = Σ^{-1/2}` makes `cov(WX) = I`. So LEACE is just three steps — whiten, project the
concept out, un-whiten:

```
r(x) = μ + Σ^{1/2} · (I − U Uᵀ) · Σ^{-1/2} · (x − μ)
            un-whiten   project out    whiten
```

`U` is an orthonormal basis for the concept subspace (in the whitened space), so `I − UUᵀ` keeps
everything except the concept. You can prove this gives `cov(r(X), concept) = 0` with the smallest
possible change to `x`. In other words, no linear probe can read the concept back out, and you've
disturbed the embedding as little as the constraint allows.

One rule of thumb falls out of this, and it matters later: erasing a `k`-dimensional concept means
removing about `k` directions. Gender is basically one direction, so a single question kills it. A
28-way attribute like occupation is high-rank and much harder, which I'll come back to at the end.

---

## Use case 1: erase a protected attribute (gender)

Gender is binary, so two questions are enough to span it. The whole pipeline is three lines:

```python
from jevu import ConceptScrubber, LayaLabeler
GENDER_Q = ["Does the text describe a woman or a female person?",
            "Does the text describe a man or a male person?"]
scrubber = ConceptScrubber(labeler=LayaLabeler(questions=GENDER_Q))
X_clean = scrubber.fit_transform(X, texts=texts)      # X = your own embeddings
```

I ran this on `bias_in_bios` with local embeddings. The gender probe goes from **0.986 AUC down to
0.109** — it can no longer tell the genders apart, and if you look at a PCA plot the two clouds that
used to be clearly separated collapse into one.

The reason you'd actually do this is to debias search. Since the eraser is a fixed matrix, you clean the
documents once at index time and clean the query the same way at search time, which keeps both in the
same space. A profession query that used to lean heavily toward one gender comes back balanced:

| query | top-10 % female, before → after |
|---|---|
| *"a registered nurse providing patient care"* | **90% → 40%** |
| *"a university professor"* | 30% → 60% |
| *"a software engineer…"* | 30% → 30% |

Two things stand out. The shift is biggest exactly where the ranking was skewed in the first place (the
nurse query), and the software-engineer query barely moves because its ranking wasn't driven by gender
to begin with. That's what you want: it takes out the bias without touching the relevance.

## Use case 2: suppress unsafe content in retrieval

Toxicity is low-rank too, so the same two-question recipe erases it cleanly. On `wiki_toxic` the unsafe
probe drops from **0.918 to 0.321**, k-means purity drops from 0.80 to 0.56, and retrieval homophily
(toxic docs pulling in other toxic docs) drops from 0.65 to 0.53.

But here's where erase and suppress come apart. Take a bunch of safe queries and count how many unsafe
documents land in the top-10:

| retrieval over… | unsafe in top-10 |
|---|---|
| raw embeddings | 0.45 |
| concept-erased embeddings | 0.47 |

Erasing toxicity doesn't actually keep toxic documents out of your results. Once you've deleted the axis,
you've also deleted the signal you'd need to filter on. If the goal is to *exclude* unsafe content, you
have to do the opposite and keep the direction `û`, then penalize it at ranking time.

The nice trick is that you can fold that penalty straight into the query. Shift the query by the concept
direction, `q' = q − α·û`, and write out the similarity:

```
sim(q', x) = q·x − α·(û·x) = sim(q, x) − α·(unsafe score of x)
```

Since `û·x` is just the document's unsafe score, the shifted query subtracts a penalty in proportion to
how unsafe each document is. So an ordinary dot-product search over the *original* index now pushes
unsafe documents down, with no re-indexing and no per-document bookkeeping, just one subtraction on the
query. `α` controls how hard you push:

| α | 0 | 0.3 | 0.6 | 1.0 | 1.5 |
|---|---|---|---|---|---|
| unsafe in top-10 | 0.45 | 0.24 | 0.13 | 0.07 | 0.04 |

To make it concrete, here's the query *"why did you delete my contribution"*, the kind of edit-dispute
query that tends to drag in flames.

Before, the top-5 is 60% unsafe:
```
1. [safe]    I noticed that you deleted peacecorpswiki — why did you delete this…
2. [UNSAFE]  niggerballs — why did you delete the page …[redacted]
3. [safe]    what responsibility?, why are you so stubborn???…
4. [UNSAFE]  Question — can you tell me what …[redacted]
5. [UNSAFE]  Listen asshole, you do NOT delete my pages …[redacted]
```

After the shift `q' = q − α·û`, the top-5 is 0% unsafe:
```
1. [safe]    I noticed that you deleted peacecorpswiki — why did you delete this…
2. [safe]    I reverted the page to the last stable form…
3. [safe]    I have added the paragraph that answers this question.
4. [safe]    As the deleting administrator, I will comment further…
5. [safe]    …
```

The relevant result is still on top, and the slurs and insults are gone.

## Detecting unsafe embeddings when all you have is the vector

At serve time you usually only get the embedding of incoming data, not the text, so laya can't run on it
directly. The question becomes how to flag an unsafe *vector*.

You don't need a second model for this, because the direction `û` is already a byproduct of the LEACE
fit. On its way to building the eraser, LEACE computes the cross-covariance `Σ_XZ = cov(X, Z)`. Each
column of it is a score-weighted average of your embeddings, `(1/n) Σᵢ z_ij x̃ᵢ`, which is itself a vector
in embedding space pointing from the safe region toward the unsafe one. (For a binary score it's exactly
the difference of the two class means.) Sum the columns, normalize, and you get one unit direction `û`
whose dot product with any embedding is that embedding's concept score.

So putting everything together, a single fitted object covers all three jobs:

```python
eraser.transform(x)        # erase   — make the concept invisible      (whitened projection)
eraser.score(x)            # detect  — (x − μ)·û, text-free, on embeddings alone
eraser.steer(q, alpha)     # suppress — q − α·û, the retrieval gate
```

Text-free detection on held-out embeddings gets to **0.917 AUC**, with no separate classifier and no
text at inference. One subtlety that tripped me up: for detection you want the *raw* cross-covariance
direction, not the whitened erase-direction. Whitening is tuned for minimal-damage erasure, which is the
wrong objective for detection — the raw direction scores 0.917 while the whitened one only gets 0.808. (A
dedicated logistic probe does slightly better at 0.934 if you need it.)

So in the end it's really one direction used three ways: project it out to erase, project onto it to
detect, subtract it from the query to suppress.

## What this doesn't do

A few honest limits, because they matter in practice:

- This works on the *latent* direction, not the words. It's not a content filter — a slur is still a
  slur in the raw text. If you need to block content outright, keep a filter in front.
- High-cardinality attributes are hard. Gender is one direction; a 28-way occupation is around 27, and a
  few questions only cover part of it. You'll knock down top-1 accuracy while a probe can still recover
  the identity by AUC. For those cases the better move is to extract the attribute per document with an
  LLM and one-hot it, but that's a heavier setup. The clean wins are the low-rank attributes above.
- The directions are fit on your data, so they won't generalize to concepts your corpus never saw.
  Re-fit when the distribution shifts — laya labels are free and local, so this is cheap.

## Wrapping up

- Keep the whole thing local: local embeddings, local laya scoring, numpy LEACE.
- Erase a protected attribute to make it unreadable; suppress unsafe content to keep it out of results.
  Erasing toxicity won't exclude it from search — for that you need the suppression gate.
- One LEACE fit gives you all three operations (`transform`, `score`, `steer`) because they share the
  same concept direction.
- The query-shift gate `q − α·û` is about the cheapest safety control you can add: one subtraction per
  query, and your index stays exactly as it is.

Code and runnable notebooks (gender and toxicity) are at
[github.com/cowana-ai/jevu](https://github.com/cowana-ai/jevu). It builds on LEACE (Belrose et al.,
NeurIPS 2023) and laya (convaiinnovations).
