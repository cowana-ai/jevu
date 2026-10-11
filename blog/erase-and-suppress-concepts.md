# Erasing and Suppressing Concepts in Embeddings

*Removing a protected attribute (gender, age) from text embeddings, or keeping unsafe content out of
retrieval — on your own machine, with a local scorer and a few lines of linear algebra. This post
focuses on the intuition, the minimal derivation, and the practical use cases: when to **erase** a
concept and when to **suppress** it.*

---

Pre-trained embeddings encode far more than the signal you need. Embed a professional bio and a simple
linear probe recovers the person's gender at ~0.99 AUC; embed a user comment and the same probe answers
"is this toxic?" at ~0.92 — in both cases *before any task-specific training*. Anything you build on top
(a classifier, a ranker, a recommender) inherits those signals, and with them the bias or the liability.

There are two distinct things you might want to do about such a concept, and they are easy to confuse:

- **Erase it** — make the concept unreadable, so it cannot influence anything downstream. This is what
  you want for a **protected attribute**: a search ranker should not be able to use gender at all.
- **Suppress it** — keep the concept's signal and use it to **exclude** matching items from results.
  This is what you want for **safety**: unsafe content should not surface, which means you need the
  signal, not its absence.

The key idea of this post is that these are *opposite* operations on the same quantity, and that both
can be done locally, from a single fit. We'll build up the method, then walk through results for each
use case with qualitative examples.

## The setup: a local pipeline

A practical constraint shapes the whole design. The text you most want to control — toxic, NSFW,
PII-adjacent — is exactly the text you cannot send to a hosted API. So every component runs on-device:

- **Local embeddings** (here, `sentence-transformers/all-MiniLM-L6-v2`).
- **A local scorer**, [laya](https://huggingface.co/convaiinnovations/laya): an open-weights model that
  takes a text and a yes/no question and returns a *calibrated probability* in `[0, 1]`. No labels, no
  threshold tuning, no network call.
- **A pure-numpy eraser**, [LEACE](https://arxiv.org/abs/2306.03819).

Nothing leaves the machine, which is the point for sensitive data.

## How erasure works (intuition, then the map)

A concept is *linearly encoded* in an embedding `x` if some direction `û` predicts it: a probe reads
`σ(wᵀx)`. If you remove the component of `x` along `û`, the probe reads a constant — the concept becomes
unreadable along that axis.

The naive version is to subtract that one direction: `x − (ûᵀx) û`. This works only if the embedding's
coordinates are uncorrelated and equal-variance. They are not — embeddings form a stretched, tilted
ellipse, so projecting along a raw axis disturbs every direction correlated with it and removes more
than necessary.

The key idea is to first move to a space where the cloud is a round ball, because there, ordinary
perpendicular projection *is* the minimal-damage way to delete a subspace. That move is called
**whitening**: with `Σ = cov(X)`, the transform `W = Σ^{-1/2}` makes `cov(WX) = I`. Putting everything
together, LEACE is three steps — whiten, project the concept out, un-whiten:

```
r(x) = μ + Σ^{1/2} · (I − U Uᵀ) · Σ^{-1/2} · (x − μ)
            └─┬──┘    └────┬───┘    └──┬───┘
         un-whiten    project out    whiten
```

Here `U` is an orthonormal basis of the concept subspace (in whitened coordinates), so `I − UUᵀ` keeps
everything *except* the concept. The result provably satisfies `cov(r(X), concept) = 0` with the
smallest possible change to `x`. In other words, no linear probe can recover the concept afterward, and
you've distorted the embedding as little as the constraint allows.

One rule of thumb follows directly:

> Erasing a `k`-dimensional concept requires removing about `k` directions. A binary attribute (gender)
> is roughly one direction, so a single question erases it. A many-valued identity (occupation) is
> high-rank, and is genuinely harder — we return to this at the end.

---

## Use case 1 — Erase a protected attribute (gender)

For a binary attribute, two questions span the concept. The full pipeline is three lines:

```python
from jevu import ConceptScrubber, LayaLabeler
GENDER_Q = ["Does the text describe a woman or a female person?",
            "Does the text describe a man or a male person?"]
scrubber = ConceptScrubber(labeler=LayaLabeler(questions=GENDER_Q))
X_clean = scrubber.fit_transform(X, texts=texts)      # X = your own embeddings
```

On `bias_in_bios` with local embeddings, the **gender-probe AUC drops from 0.986 to 0.109** — a probe
can no longer separate the genders, and in PCA the two clouds collapse into one.

The practical payoff is debiased search. The eraser is a fixed matrix, so you scrub documents at index
time and the query at search time with the *same* map (keeping them in one space). A profession query
that used to return a gender-skewed set now returns a balanced one:

| query | top-10 % female, before → after |
|---|---|
| *"a registered nurse providing patient care"* | **90% → 40%** |
| *"a university professor"* | 30% → 60% |
| *"a software engineer…"* | 30% → 30% |

Two things are worth noting. First, the shift is largest exactly where ranking *was* gender-skewed (the
nurse query). Second, the software-engineer query does not move — its ranking was not gender-driven to
begin with. This is the behavior you want: the method removes the bias, not the relevance.

## Use case 2 — Suppress unsafe content in retrieval

Toxicity and NSFW are also low-rank, so the same two-question recipe erases them cleanly: on
`wiki_toxic`, the unsafe-probe AUC drops **0.918 → 0.321**, k-means purity drops 0.80 → 0.56, and
retrieval homophily (toxic docs retrieving toxic docs) drops 0.65 → 0.53.

But here is the catch, and the reason erase and suppress are different tools. Take safe queries and
measure how much unsafe content sits in the top-10:

| retrieval over… | unsafe in top-10 |
|---|---|
| raw embeddings | 0.45 |
| concept-erased embeddings | 0.47 |

**Erasing toxicity does not remove toxic documents from the results.** Once the axis is gone, so is the
signal you would have filtered on. To *exclude* unsafe content you do the opposite: keep the direction
`û` and penalize it at ranking time.

The key idea is that this penalty can be folded into the query. Shift the query by the concept direction,
`q' = q − α·û`, and expand the similarity:

```
sim(q', x) = q·x − α·(û·x) = sim(q, x) − α·(unsafe score of x)
```

Because `û·x` *is* the document's unsafe score, the steered query subtracts a penalty proportional to how
unsafe each document is. In other words, ordinary dot-product search over the *original* index now sinks
unsafe documents — no re-indexing, no per-document scores, just one vector subtraction per query. `α`
tunes the strength:

| α | 0 | 0.3 | 0.6 | 1.0 | 1.5 |
|---|---|---|---|---|---|
| unsafe in top-10 | 0.45 | 0.24 | 0.13 | 0.07 | 0.04 |

Qualitatively, for the query *"why did you delete my contribution"* (an edit-dispute query that tends to
pull in flames):

**Before** — top-5, 60% unsafe:
```
1. [safe]    I noticed that you deleted peacecorpswiki — why did you delete this…
2. [UNSAFE]  niggerballs — why did you delete the page …[redacted]
3. [safe]    what responsibility?, why are you so stubborn???…
4. [UNSAFE]  Question — can you tell me what …[redacted]
5. [UNSAFE]  Listen asshole, you do NOT delete my pages …[redacted]
```

**After** `q' = q − α·û` — top-5, 0% unsafe:
```
1. [safe]    I noticed that you deleted peacecorpswiki — why did you delete this…
2. [safe]    I reverted the page to the last stable form…
3. [safe]    I have added the paragraph that answers this question.
4. [safe]    As the deleting administrator, I will comment further…
5. [safe]    …
```

The on-topic result stays at the top; the slurs and insults drop out.

## Detecting unsafe embeddings when you only have the vector

At serve time you often see only the **embedding** of incoming data, not the text — so laya, which needs
text, cannot run on it. The question is how to detect or exclude an unsafe *vector*.

You do not need a second model, because the concept direction `û` is already a byproduct of the LEACE
fit. Specifically, LEACE computes the cross-covariance `Σ_XZ = cov(X, Z)` on the way to building the
eraser. Each column of `Σ_XZ` is a score-weighted average of the embeddings — `(1/n) Σᵢ z_ij x̃ᵢ` — which
is itself a vector in embedding space pointing from the safe region toward the unsafe one (for a binary
score it is exactly the difference of class means). Summing the columns and normalizing gives a single
unit direction `û` whose dot product with any embedding *is* that embedding's concept score.

Putting everything together, one fitted object does three jobs:

```python
eraser.transform(x)        # ERASE   — make the concept invisible      (whitened projection)
eraser.score(x)            # DETECT  — (x − μ)·û, text-free, on embeddings alone
eraser.steer(q, alpha)     # SUPPRESS — q − α·û, the retrieval gate
```

Text-free detection on held-out embeddings reaches **AUC 0.917** — no separate classifier, no text at
inference. One subtlety: use the *raw* cross-covariance direction for detection, not the whitened
erase-direction. Whitening is tuned for minimal-damage erasure, which is the wrong objective for
detection; empirically the raw direction scores 0.917 while the whitened one scores only 0.808. (A
dedicated logistic probe reaches 0.934 if you want the last bit of accuracy.)

So the whole toolkit reduces to one direction used three ways: **project it out** to erase, **project
onto it** to detect, **subtract it from the query** to suppress.

## Limitations

A few honest boundaries:

- These operations act on the *latent* direction, not the words. This is not a content filter — a slur
  is still lexically a slur. Pair it with filtering if you need to block content outright.
- High-cardinality identities are hard. A binary attribute is ~1 direction; a 28-way occupation is ~27,
  and a few questions only partly span it — you collapse top-1 accuracy while a probe still recovers the
  identity by AUC. The better recipe there is to extract the attribute per document with an LLM and
  one-hot it, but it is a heavier regime. The clean wins are the low-rank protected attributes above.
- The directions are fit on your data, so they do not generalize to concepts your corpus never
  contained. Re-fit when the distribution shifts — laya labels are free and local.

## Summary

- Keep sensitive pipelines local: local embeddings, local laya scoring, numpy LEACE.
- Erase protected attributes to make them unreadable; suppress unsafe content to keep it out of results.
  Erasing toxicity does not exclude it from search — that requires the suppression gate.
- A single LEACE fit gives you all three operations — `transform` (erase), `score` (text-free detect),
  `steer` (query-shift suppress) — because they share one concept direction.
- The query-shift gate, `q − α·û`, is the cheapest safety control: one subtraction per query, with the
  existing index left untouched.

Code and runnable notebooks (gender, toxicity) are at
[github.com/cowana-ai/jevu](https://github.com/cowana-ai/jevu). The method builds on LEACE (Belrose et
al., NeurIPS 2023) and laya (convaiinnovations).
