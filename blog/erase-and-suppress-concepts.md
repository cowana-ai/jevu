# Erasing — and Suppressing — Concepts in Embeddings

*Off-the-shelf embeddings quietly encode gender, toxicity, and more. Here's how to take a concept back
out — or keep it out of your search results — **entirely on your own machine**, with a local scorer and
a few lines of linear algebra. And why "erase" and "suppress" are opposite tools you shouldn't confuse.*

---

When I reach for an embedding model, I'm thinking about one thing: does it capture meaning well enough
for my task. I'm almost never thinking about what *else* it captured along the way. That's the mistake.

Embeddings are good at their job, so they encode far more than the thing you asked for. Embed a
professional bio and a one-line linear probe reads the person's **gender at ~0.99 AUC** — before you've
trained anything. Embed a user comment and the same trick reads **"is this toxic?" at ~0.92**. Every
classifier, ranker, and recommender you build on top inherits those signals, and with them the bias (or
the liability).

I wrote a small library, [`jevu`](https://github.com/cowana-ai/jevu), to do something about it. Two
design choices shape everything below:

1. **It runs locally.** The sensitive text you most want to control — toxic, NSFW, PII-adjacent — is
   exactly the text you *can't* ship to a hosted API. So the whole pipeline is on-device: local
   embeddings, a local scorer ([laya](https://huggingface.co/convaiinnovations/laya)), and a pure-numpy
   eraser. Nothing leaves the box.
2. **It separates two goals people constantly conflate** — *erasing* a concept and *suppressing* it.

## Two things you can do to a concept

This is the distinction the rest of the post hangs on:

| | **Erase** | **Suppress** |
|---|---|---|
| what it does | makes the concept **invisible** to any linear reader | **excludes** concept-y items from results |
| on the signal | **destroys** it | **keeps** it and gates on it |
| use it for | **protected** attributes (gender, age) — don't let them bias ranking | **safety** (toxic/NSFW) — keep it out of results |
| operation | project the concept direction *out* | subtract the concept *score* at ranking time |

They pull in **opposite directions**. If you *erase* "toxicity," toxic docs blend in — you've thrown
away the very signal you'd need to filter them. If you want toxic content *gone from results*, you keep
the signal and demote it. Same machinery, opposite move. Most "debiasing" write-ups blur this; keeping
it straight is half the battle.

## The method, briefly

**The scorer (laya).** A concept is a yes/no question — `"Does the text describe a woman?"`,
`"Does the text contain toxic or unsafe content?"`. laya is an open-weights, on-device model that
answers such a question with a **calibrated probability in `[0,1]`** — no training set, no threshold
tuning, no API. For a low-rank attribute, one or two questions are enough.

**The eraser (LEACE).** A concept is *linearly encoded* if some direction `û` in the embedding predicts
it. Remove that direction and no linear probe can read it. The subtlety: naively projecting along `û`
over-damages the embedding, because its dimensions have unequal variance and are correlated (a tilted,
stretched ellipse). [LEACE](https://arxiv.org/abs/2306.03819) does it right, in three moves:

```
r(x) = μ + Σ^{1/2} · (I − U Uᵀ) · Σ^{-1/2} · (x − μ)
            └─┬──┘   └────┬────┘   └──┬───┘
         un-whiten    project out    whiten
                     the concept
```

**Whiten** (`Σ^{-1/2}`) to turn the ellipse into a round ball, where perpendicular projection is the
*minimal-damage* way to delete a subspace; **project out** the concept subspace there (`I − UUᵀ`);
**un-whiten** back. One closed-form shot, provably `cov(r(X), concept) = 0` with the least change to
`x`.

One idea carries the whole thing:

> **Erasing a `k`-dimensional concept means removing ~`k` directions.** A binary attribute (gender) is
> ~1 direction, so one question erases it. A many-valued identity (occupation) is high-rank and
> genuinely hard — more on that at the end.

---

## Erase in action

### Gender, locally, on `bias_in_bios`

Local embeddings (`all-MiniLM-L6-v2`), two laya questions (woman / man), one LEACE fit:

```python
from jevu import ConceptScrubber, LayaLabeler
GENDER_Q = ["Does the text describe a woman or a female person?",
            "Does the text describe a man or a male person?"]
scrubber = ConceptScrubber(labeler=LayaLabeler(questions=GENDER_Q))
X_clean = scrubber.fit_transform(X, texts=texts)     # X = your embeddings
```

**Gender-probe AUC: 0.986 → 0.109.** It's gone — a probe can no longer tell the genders apart. In PCA,
the clean male/female separation collapses into one cloud.

**The payoff is debiased search.** Run a profession query over the *erased* embeddings and the results
stop being gender-skewed (the query is scrubbed with the same map, so docs and query stay in one space):

| query | top-10 % female, before → after |
|---|---|
| *"a registered nurse providing patient care"* | **90% → 40%** |
| *"a university professor"* | 30% → 60% |
| *"a software engineer…"* | 30% → 30% (ranking wasn't gender-driven — honest null) |

The effect is largest exactly where ranking *was* skewed, and leaves already-neutral queries alone. It
removes the bias, not the relevance.

### Toxicity / NSFW, fully local, on `wiki_toxic`

Same recipe, two "unsafe" questions:

- **unsafe-probe AUC: 0.918 → 0.321** (chance = 0.5) — toxicity no longer linearly readable.
- **k-means purity: 0.80 → 0.56** — clustering stops organizing by safety.
- **retrieval homophily: 0.65 → 0.53** — toxic docs stop attracting toxic docs.

Erasure makes "unsafe" *invisible*. Which — and this is the twist — is **not** what you want if your
actual goal is to keep unsafe content out of results.

---

## Suppress in action: the retrieval safety gate

Here's the finding that forces the erase-vs-suppress distinction. Take safe queries and ask: how much
unsafe content is in the top-10?

| retrieval over… | unsafe in top-10 |
|---|---|
| raw embeddings | 0.45 |
| **concept-erased embeddings** | **0.47** ← erasure did *nothing* to exclude |

**Erasing toxicity doesn't remove toxic docs from results** — once the axis is gone, so is the signal
you'd filter on. To *exclude* unsafe content you do the opposite: keep the score and **penalize** it.

The elegant part is that this is a one-line change at query time. Keep the unsafe direction `û`;
subtracting it from the query turns ordinary similarity search into a safety gate, for free:

```
q' = q − α·û
sim(q', x) = q·x − α·(û·x) = sim(q, x) − α·(unsafe score of x)
```

Because `û·x` *is* the document's unsafe score, the steered query sinks unsafe docs in proportion to how
unsafe they are. No re-indexing, no per-doc score lookup — **one vector subtract per query**, over your
existing index. Measured, unsafe fraction in top-10:

| α | 0 | 0.3 | 0.6 | 1.0 | 1.5 |
|---|---|---|---|---|---|
| unsafe in top-10 | 0.45 | 0.24 | 0.13 | 0.07 | 0.04 |

### Qualitative: watch it work

Query: *"why did you delete my contribution"* (the kind of edit-dispute query that drags in flames).

**Before** (top-5, 60% unsafe):
```
1. [✓ safe ]  I noticed that you deleted peacecorpswiki — why did you delete this…
2. [⚠ UNSAFE] niggerballs — why did you delete the page …[redacted]
3. [✓ safe ]  what responsibility?, why are you so stubborn???…
4. [⚠ UNSAFE] Question — can you tell me what …[redacted]
5. [⚠ UNSAFE] Listen asshole, you do NOT delete my pages …[redacted]
```

**After** `q' = q − α·û` (top-5, **0% unsafe**):
```
1. [✓ safe ]  I noticed that you deleted peacecorpswiki — why did you delete this…
2. [✓ safe ]  I reverted the page to the last stable form…
3. [✓ safe ]  I have added the paragraph that answers this question.
4. [✓ safe ]  As the deleting administrator, I will comment further…
5. [✓ safe ]  …
```

The on-topic result stays #1; the slurs and insults are gone. One subtraction.

---

## The catch at inference — and why one LEACE fit is enough

laya needs the **text**. But a deployed retriever usually sees only the **embedding** of incoming data,
and laya can't run on a bare vector. So how do you detect/exclude unsafe *embeddings*?

You don't need a second model. The **same LEACE fit** already contains the concept direction — the raw
cross-covariance `Σ_XZ` it computed on the way to building the eraser. It's a weighted average of your
embeddings (weighted by how unsafe each text is), so it's a vector in embedding space whose dot product
with any embedding *is* that embedding's concept score. `jevu` exposes it, so **one fitted object does
three jobs**:

```python
eraser.transform(x)        # ERASE   — make the concept invisible      (whitened projection)
eraser.score(x)            # DETECT  — (x−μ)·û, text-free, on embeddings alone
eraser.steer(q, alpha)     # SUPPRESS — q − α·û, the retrieval gate
```

Text-free detection on held-out embeddings — **AUC 0.917**, no separate classifier, no text, no laya at
inference. (One subtlety worth knowing: use the *raw* cross-cov direction for detection, **not** the
whitened erase-direction — whitening is tuned for minimal-damage erasure, the wrong objective for
detection. Raw scores 0.917; whitened only 0.808. A dedicated probe buys ~0.02 more, if you want it.)

So the whole toolkit is one direction, used three ways:

> **`transform`** projects the concept *out* (erase · for protected attributes) ·
> **`score`** projects *onto* it (detect · text-free) ·
> **`steer`** subtracts it from the query (suppress · for safety).

---

## Honest limits

- **Erasure/steering act on the *latent* direction, not the words.** This is not a content filter — a
  slur is still a slur lexically. Pair it with filtering if you must *block* content.
- **High-cardinality identities are hard.** A binary attribute is ~1 direction; a 28-way *occupation*
  is ~27, and a handful of questions only partly span it (you collapse top-1 accuracy while a probe
  still recovers it by AUC). The better recipe there is to *read the label off each document* with an
  LLM and one-hot it — but it's a heavier regime. The clean wins are the low-rank protected attributes
  above.
- **It doesn't generalize out-of-distribution.** An eraser fit on your data removes the subspace your
  data spans — occupations (or toxicity styles) it never saw stay recoverable. Re-fit when the
  distribution shifts (cheap — laya labels are free and local).

## Takeaways

1. **Keep sensitive pipelines local.** Local embeddings + local laya + numpy LEACE — the data never
   leaves your machine, which is the whole point for NSFW/PII-adjacent text.
2. **Erase ≠ suppress.** Erase *protected* attributes to make them invisible; suppress *unsafe* content
   to keep it out of results. Erasing toxicity does **not** remove it from your search results.
3. **One `LeaceEraser` does all three** — `transform` / `score` / `steer` — from a single fit.
4. **The query-shift gate is the cheapest safety control there is:** one vector subtract, existing index
   untouched, tunable strength.
5. **Match questions to rank.** Low-rank attributes go to chance with one or two questions; that's the
   sweet spot. High-cardinality identities are a different, harder animal.

An embedding compresses everything it saw, including the parts you'd never have chosen to carry forward.
Erasing — or suppressing — is just deciding, out loud and on the record, which of that you refuse to
keep. The nice part is you can do it on your own machine, and read back exactly what you let go.

---

*Code and runnable notebooks (gender, NSFW/toxicity) at
[github.com/cowana-ai/jevu](https://github.com/cowana-ai/jevu). Built on
LEACE (Belrose et al., NeurIPS 2023) and laya (convaiinnovations).*
