# Erase vs. Suppress: local, laya-based concept control for embeddings

*How to remove a protected attribute (gender, age) from embeddings, or keep unsafe/NSFW content out of
retrieval — entirely on your machine, with no text at inference. Plus the geometry of why the obvious
tricks fail and the efficient ones work.*

---

## The setup: sensitive data can't leave the box

The text you most want to control — toxic, NSFW, PII-adjacent — is exactly the text you **can't** send
to a hosted API. So the whole pipeline here is **local**:

- **Local embeddings** (e.g. `sentence-transformers/all-MiniLM-L6-v2`),
- **Local scoring** with [**laya**](https://huggingface.co/convaiinnovations/laya) — an open-weights,
  on-device calibrated yes/no scorer (`noul = P(true)`),
- **Pure-numpy erasers** ([LEACE](https://arxiv.org/abs/2306.03819)).

Nothing leaves the machine. We use the open-source `wiki_toxic` dataset (binary toxic/safe) as a
shareable stand-in for "unsafe/NSFW" — a **low-rank** concept, which is laya's sweet spot: one or two
yes/no questions span it.

## Two goals that are opposites: erase vs. suppress

The single most important distinction in this whole area:

| goal | operation | use it for | what it does to the signal |
|---|---|---|---|
| **Erase** | project the concept direction *out* (LEACE) | **protected** attributes (gender, age) | **destroys** it — the attribute becomes invisible, can't bias ranking |
| **Suppress** | **keep** the concept score, *penalize* it | **safety** (unsafe/NSFW) | **uses** it — unsafe content is excluded from results |

They pull in opposite directions. Erasing "unsafe" would make toxic docs *blend in* — the exact
opposite of keeping them out. For safety you want to **keep the signal and gate on it.**

## Erasing a low-rank concept (the easy, clean case)

Two yes/no questions, laya-scored locally, LEACE applied:

```python
from jevu import ConceptScrubber, LayaLabeler
Q = ["Does the text contain toxic, hateful, or unsafe content?",
     "Does the text use insults, profanity, or offensive language?"]
X_clean = ConceptScrubber(labeler=LayaLabeler(questions=Q, cache_dir=".cache")).fit_transform(X, texts=texts)
```

Result on `wiki_toxic` (local MiniLM embeddings):

- **unsafe-probe AUC: 0.92 → 0.32** (gone; chance = 0.5)
- **k-means purity: 0.80 → 0.56** (clustering no longer organizes by safety)
- **retrieval homophily: 0.65 → 0.53** (unsafe docs no longer attract unsafe)

For a **binary/low-rank** concept, one or two questions span it, so LEACE drives the probe to chance.
(High-cardinality *identities* — occupation, topic — are a different, harder regime; see the repo's
limits section.)

## The inference problem: at serve time you only have embeddings

laya needs the **text**. But a deployed retriever/classifier often sees only the **embedding** of
incoming data. So how do you detect/exclude unsafe *vectors*?

**Distil laya into a direction, once.** Where you have text, laya labels a reference corpus; train a
linear probe on the embeddings → a single unit vector `û` (the "unsafe direction"). After that,
everything is a dot product — no text, no laya at inference:

```python
labels = (laya_score(ref_texts) >= 0.5)
w = LogisticRegression(C=0.3).fit(ref_embeds, labels).coef_[0]
u = w / norm(w)                       # the unsafe direction

s = sigmoid(u @ x)                    # text-free unsafe score for ANY incoming embedding
```

Text-free detection AUC on held-out embeddings: **~0.93**. laya is used *only* to make the labels —
and laya labels are **free and local**, so "not enough labels" is rarely the real constraint.

### Low-data behaviour (measured)

| n_train | probe (C=1) | probe (C=0.05) | LEACE-displacement direction |
|---|---|---|---|
| 20 | 0.826 | 0.829 | 0.635 |
| 40 | 0.871 | 0.873 | 0.628 |
| 80 | 0.896 | 0.898 | 0.698 |
| 150 | 0.930 | 0.924 | 0.821 |

The probe **degrades gracefully** and barely overfits (even n=20 → 0.83). The eraser-derived direction
is the *fragile* one at low n, because LEACE **whitens** (inverts a d×d covariance), which needs
`n ≳ d` rows. So: use the regularized probe when data is scarce; don't reuse the eraser direction.

## The geometry: why magnitude fails and sign works

Tempting idea: "if LEACE moves an embedding a lot, it must be unsafe → exclude it." **It doesn't
work** — displacement *magnitude* `‖x − r(x)‖` gives AUC **0.51 (chance)**.

Why: LEACE collapses the concept axis onto a point, so it moves both very-safe and very-unsafe
embeddings by the *same amount*, in **opposite directions**. On a number line with safe ≈ −1, unsafe
≈ +1, mean = 0, erasure projects everyone onto 0 — both ends move by 1. Magnitude is **sign-blind**;
it measures *extremeness*, not *which side*.

The fix is the **sign**: the signed projection `û·x` (which side of the mean) → AUC **0.808**. And a
dedicated probe (which optimizes separation, not minimal-damage erasure) is sharper still → **0.933**.

> One direction, three uses: **project it out** to erase, **signed-project onto it** to detect,
> **train a probe on it** for the best detector.

## The efficient safety gate: shift the query

To keep unsafe docs out of retrieval, you don't touch the index at all — you shift the **query**:

```
q' = q − α·û
sim(q', x) = q·x − α·(û·x) = sim(q, x) − α·(unsafe score of x)
```

Because `û·x` *is* the unsafe score, subtracting `û` from the query turns ordinary cosine/dot-product
search into a safety gate **automatically**. Unsafe docs lose similarity in proportion to how unsafe
they are.

Measured (safe queries, `wiki_toxic`), unsafe fraction in top-10:

| α | 0 | 0.3 | 0.6 | 1.0 | 1.5 |
|---|---|---|---|---|---|
| unsafe in top-10 | 0.45 | 0.24 | 0.13 | 0.07 | 0.04 |

**Why this is the cheapest option:**

| approach | per-query cost | index change | per-doc scores |
|---|---|---|---|
| rerank penalty `sim − α·s(x)` | rerank candidates | none | yes (store `s`) |
| penalty dimension `[x; −α·s]` | none | **re-index docs** | baked in |
| **query shift `q − α·û`** | **1 vector subtract** | **none** | **no** |

It runs on an existing, deployed ANN index unchanged. Caveats: (1) shifting the query drifts it off the
user's intent if `α` is large — keep it moderate (0.3–0.6 gives strong suppression with little drift);
(2) use the sharper **probe** direction for `û`.

## Putting it together: a RetrievalGuard

One object, one operation per attribute:

```
protected attribute (gender) → ERASE   (LEACE)           → neutral ranking, can't be biased
unsafe / NSFW                → STEER   (q − α·û)          → excluded from results, index untouched
```

- **Erase** what should be *invisible* to ranking (fairness).
- **Steer/gate** what should be *excluded* from ranking (safety).
- Both derive from laya scores at fit time; at inference it's all dot products on embeddings — **no
  text, no laya, no re-indexing.**

## Honest limits

- **Erasure/steering act on the latent direction, not the words.** Not a content filter — pair with
  filtering if you must *block* content.
- **Subspace-specific.** These directions are fit on your data; they do **not** generalize to concepts
  your corpus never contained. Re-fit when the distribution shifts (cheap — laya labels are free).
- **Low-rank concepts only (for the easy wins).** Binary attributes (gender, age, safety) → one or two
  questions → chance. High-cardinality identities need a different, heavier recipe.

## Takeaways

1. **Keep sensitive pipelines local** — laya + local embeddings + numpy LEACE; nothing leaves the box.
2. **Erase ≠ suppress.** Erase protected attributes (invisible); suppress unsafe content (excluded).
3. **Distil laya into a direction** → text-free detection (`û·x`) on embeddings alone.
4. **Steer the query (`q − α·û`)** for the cheapest safety gate — no index changes, one subtract.
5. **Trust the sign, not the magnitude** (displacement magnitude is chance); and the **probe** beats
   the eraser direction, especially at low n.
