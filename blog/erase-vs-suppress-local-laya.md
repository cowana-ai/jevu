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
(Many-valued *identities* are a harder regime, out of scope for this local/low-rank story.)

## The inference problem: at serve time you only have embeddings

laya needs the **text**. But a deployed retriever/classifier often sees only the **embedding**. So how
do you detect/exclude unsafe *vectors*?

**Reuse the LEACE fit — no second model.** LEACE already computes the cross-covariance `Σ_XZ` (the raw
concept direction) while fitting the eraser. `jevu` exposes it as `eraser.concept_direction_`, so the
*same* fitted object does all three jobs — everything is then a dot product, no text, no laya:

```python
from jevu import LeaceEraser
eraser = LeaceEraser().fit(ref_embeds, laya_scores)   # ONE fit

eraser.transform(x)        # ERASE   — make the concept invisible (whitened projection)
eraser.score(x)            # DETECT  — (x − μ)·û, text-free, on embeddings alone
eraser.steer(q, alpha)     # SUPPRESS — q − α·û for retrieval
```

`û = concept_direction_` is the raw `Σ_XZ` direction (summed over the questions, unit-normalized,
oriented so + = concept present). Text-free detection AUC on held-out embeddings: **0.917** — basically
a probe's quality, with **no extra training**. laya is used only to make the labels, and laya labels
are free/local, so "not enough labels" is rarely the real constraint.

### Which direction to reuse — they are not equal

Three directions come out of (or near) the same fit:

| direction | detection AUC | low-n | note |
|---|---|---|---|
| **raw `Σ_XZ`** (`concept_direction_`) | **0.917** | robust (no whitening) | **use this** — reuses the fit |
| whitened erase-displacement | 0.808 | fragile (needs `n ≳ d` to whiten) | tuned for *erasure*, wrong for detection |
| logistic probe (separate fit) | 0.934 | graceful | marginally sharper, if you want it |

The **whitened erase-displacement is the fragile one at low n** — it inverts a `d×d` covariance, so it
needs rows. The raw cross-cov doesn't whiten → robust. (A probe degrades gracefully too: AUC 0.83 at
n=20, 0.93 at n=150.) So **reuse `concept_direction_`**; reach for a probe only to chase the last ~0.02.

## The geometry: why the *magnitude* of erasure is useless

A tempting shortcut: "if LEACE *moves* an embedding a lot, it must be unsafe → exclude it." **It
doesn't work** — displacement *magnitude* `‖x − r(x)‖` gives AUC **0.51 (chance)**.

Why: LEACE collapses the concept axis onto a point, moving both very-safe and very-unsafe embeddings by
the *same amount*, in **opposite directions**. On a number line with safe ≈ −1, unsafe ≈ +1, mean = 0,
erasure projects everyone onto 0 — both ends move by 1. Magnitude is **sign-blind**; it measures
*extremeness*, not *which side*. So you need a **signed projection** onto the direction, not the move
size — which is exactly what `eraser.score` (= `(x−μ)·û`) gives.

> One fitted `LeaceEraser`, three uses of the same concept direction: **`transform`** projects it out
> (erase), **`score`** signed-projects onto it (detect), **`steer`** subtracts it from a query (suppress).

## The efficient safety gate: shift the query

To keep unsafe docs out of retrieval, you don't touch the index at all — you shift the **query** with
`eraser.steer(q, α)`:

```
q' = q − α·û                                          # eraser.steer(q, α)
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
(2) use `concept_direction_` (the raw `Σ_XZ` direction) as `û` — not the whitened erase-direction.

## Putting it together: one `LeaceEraser` per attribute

Fit a `LeaceEraser` per sensitive attribute, then apply the operation that fits the goal:

```
protected attribute (gender) → eraser.transform(docs & query)   → neutral ranking, can't be biased
unsafe / NSFW                → eraser.steer(query, α)            → excluded from results, index untouched
```

- **`transform`** (erase) what should be *invisible* to ranking (fairness) — applied to docs + query.
- **`steer`** (gate) what should be *excluded* from ranking (safety) — applied to the query only.
- Both come from the *same* fit (laya labels → `Σ_XZ`); at inference it's dot products on embeddings —
  **no text, no laya, no re-indexing.**

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
3. **One `LeaceEraser` does both** — `transform` (erase), `score` (text-free detect, AUC 0.917),
   `steer` (query-shift suppress) — all from the raw `Σ_XZ` `concept_direction_`, no second model.
4. **Steer the query (`q − α·û`)** for the cheapest safety gate — no index changes, one subtract.
5. **Use `concept_direction_`, not the move size** — displacement *magnitude* is chance (sign-blind);
   the raw cross-cov direction is robust at low n (no whitening). A probe buys only ~0.02 more AUC.
