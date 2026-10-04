# Erasing Concepts from Embeddings — in Plain English

*How to remove gender, occupation, or any attribute you can name from off-the-shelf
embeddings, label it zero-shot with an LLM scorer, and actually see which parts of your
data get cleaned.*

---

## The problem

Off-the-shelf text embeddings are convenient, but they quietly encode attributes you may not
want in a downstream model. Embed a professional bio with a standard model and a simple linear
probe can read off the person's **gender** at ~1.00 AUC and their **occupation** at ~0.65 AUC —
*before you do anything*. Any classifier, ranker, or recommender you build on top inherits that
signal, and with it the bias.

The usual fix is a full annotation campaign plus adversarial training. That's expensive, and it
only works for attributes you already have labels for.

**jevu** takes a different route:

> Bring **(1) your embeddings** (from any model), **(2) your raw texts**, and **(3) a concept
> named in plain English** — even a single ambiguous word like `"gender"` or `"occupation"`. An LLM
> turns the word into concrete questions, a scorer labels each one per text, and **LEACE** removes
> the whole thing in closed form — leaving everything else intact.

```python
from jevu import ConceptScrubber

X = your_model.encode(texts)                         # embeddings from ANY model
scrubber = ConceptScrubber(concept="occupation", method="leace",  # just the bare word
                           expand="auto")            # LLM sizes & writes the questions
scrubber.fit(X, texts=texts)                         # expand -> score -> select -> fit the eraser
X_clean = scrubber.transform(X_new)                  # erase it from new embeddings

scrubber.audit(X, texts=texts)
# {'concept_auc_before': 0.65, 'concept_auc_after': 0.32, 'n_concepts': 15}
```

No annotation. No retraining of the embedding model. Just numpy and scikit-learn plus a scorer.

---



## The method: a four-step pipeline

You hand in one ambiguous word — `"gender"` or `"occupation"` — and get back an eraser. Under the
hood that word flows through four steps: **expand → score → select → erase**. We'll use `"occupation"`
throughout.

A key piece of intuition sets up the whole thing:

> **Erasing a `k`-dimensional concept needs ~`k` directions to project out.**

"Gender" is essentially binary — it lives on *one* direction, so one question can carry it. A 28-way
occupation *identity* lives in a ~27-dimensional subspace — no single question can capture it. That's
why an ambiguous word has to become *several* questions before we can erase it.

### Step 1 — Expand the word into questions (LLM)

A bare noun like `"occupation"` is not something you can score directly ("is this text occupation-y?"
is meaningless). So an LLM first **expands** it into a set of concrete, discriminative yes/no
questions that *together cover* the concept's facets:

```python
scrubber = ConceptScrubber(concept="occupation", expand="auto")  # or expand=True, n_questions=15
scrubber.fit(X, texts=texts)
scrubber.concept_questions_
# ['Does the text describe a healthcare professional?',
#  'Does the text describe an educator or academic?',
#  'Does the text describe an artist or performer?',
#  'Does the text describe a legal professional?', ...]   # ~15 covering questions
```

With `expand="auto"` the LLM also *sizes* the set from the concept's cardinality — **1–2 questions
for a binary attribute** (gender, sentiment), a **covering set (~15) for a many-valued identity**
(occupation, topic). Fewer questions = fewer scoring calls downstream.

### Step 2 — Score a sample with JEV (build the matrix `Z`)

Each question is now scorable. **JEV** is a calibrated per-question scorer: give it a text and a
yes/no question and it returns a number in `[0, 1]` — no training set, no threshold tuning. We run it
over a **sample of `n` rows × the `p` questions** to build a score matrix

```
Z ∈ ℝ^{n × p},   Z[i, j] = JEV(text_i, question_j) ∈ [0, 1]
```

This `Z` is a little interpretable embedding: every column is a *named* axis. We sample (e.g. 250
rows) rather than scoring everything, because the next step only needs enough rows to *rank* the
questions — not the full dataset.

### Step 3 — Select the `k` questions that matter (greedy, label-free)

A pool of 15 questions is good for coverage, but many overlap ("healthcare professional" and "works
in medicine" carry nearly the same column). We keep the `k` that best **reconstruct the whole pool**
— a classic *column subset selection*, done **label-free**: we look only at `Z`, never at the true
occupation labels.

Concretely, center `Z` and greedily, one at a time, add the question whose column most reduces the
least-squares reconstruction residual of the *entire* pool from the columns chosen so far. With `A`
the matrix of already-selected columns plus a candidate `c`:

```
pick c that minimizes   ‖Z − A A⁺ Z‖_F
```

where `A⁺` is the Moore–Penrose pseudoinverse, so `A A⁺ Z` is the orthogonal projection of every
column of `Z` onto the span of the selected columns, and `‖·‖_F` is the Frobenius norm (total
residual over all questions). Intuitively: *"which few questions let me linearly predict all the
others?"* Three usually suffice:

```python
scrubber = ConceptScrubber(concept="occupation", expand=True, n_questions=15,
                           select_k=3, select_sample=250)   # select on a 250-row sample
scrubber.fit(X, texts=texts)
scrubber.selected_questions_
# ['Does the text identify someone as a researcher/scholar?',
#  'Does the text mention a healthcare profession?',
#  'Does the text indicate an artist or performer?']
```

Then only those `k` winners are scored on the *full* dataset — cost drops from `pool × N` to
`pool × sample + k × N` — and the result is a small, **reusable** eraser you can apply to new data.
*Which* data those `k` questions end up cleaning is a story in itself — see the next section.

### Step 4 — Erase with LEACE (the linear algebra)

Now we have embeddings `X ∈ ℝ^{N × d}` and the selected concept scores `Z ∈ ℝ^{N × k}`. **LEACE**
(*LEAst-squares Concept Erasure*, Belrose et al. NeurIPS 2023) finds the affine map `r(x)` that makes
the concept **linearly unreadable** — `cov(r(X), Z) = 0` *exactly* — while changing `x` as little as
possible (minimum expected squared distance). Here's why it looks the way it does.

**Warm-up: one direction.** Suppose the concept were a single unit direction `û` in `x`. Projecting
it out is

```
P = I − û ûᵀ,     x' = P x = x − (ûᵀx) û
```

This zeroes the concept coordinate: `ûᵀx' = ûᵀx − (ûᵀx)(ûᵀû) = 0` (since `ûᵀû = 1`). A probe that
relied on `û` now reads a constant. The catch: projecting along the *raw* axis `û` is only
minimal-damage if the embedding's coordinates are uncorrelated and equal-variance. Real embeddings
are not — some directions carry far more variance than others — so LEACE first **whitens**.

**The full map, in three moves:**

1. **Whiten.** Let `Σ = cov(X)` be the embedding covariance. The whitening transform `W = Σ^{-1/2}`
   maps `X` to a space where `cov(WX) = I` — all directions equal-variance, so ordinary orthogonal
   projection *is* the minimal-damage operation.

2. **Project out the concept subspace.** In whitened space the concept lives in the column space of
   `W Σ_{XZ}`, where `Σ_{XZ} = cov(X, Z)` is the cross-covariance (how each embedding direction
   co-varies with each concept question). Take an orthonormal basis `U` of that column space and
   build the orthogonal-complement projector `P = I − U Uᵀ`. For a `k`-dim concept this removes `k`
   directions — the concrete form of "erasing a `k`-dim concept needs ~`k` directions."

3. **Un-whiten and re-center.** Map back with `W⁻¹ = Σ^{1/2}` and restore the mean `μ`:

```
r(x) = μ + Σ^{1/2} · P · Σ^{-1/2} · (x − μ),      P = I − U Uᵀ
```

Because the projection kills exactly the whitened concept subspace, the result provably satisfies
`cov(r(X), Z) = 0`: **no linear probe can recover the concept** from `r(X)`. And because it's an
orthogonal projection *in the whitened metric*, it's the least-squares-optimal such map — the
smallest distortion of `x` that achieves erasure. One closed-form shot, no iteration. (If you only
have hard labels instead of JEV scores, pass those as `Z` — the math is identical.)

---



## Does it work? The numbers

On **Bias in Bios** (professional bios, 28 occupations, binary gender) with
`text-embedding-3-small`, evaluated on a held-out test set.

**Gender** (binary) — linear-probe **AUC**:


| Concept | Before | After (jevu) | Chance |
| ------- | ------ | ------------ | ------ |
| Gender  | 0.996  | **0.515**    | ~0.50  |


Gender collapses to chance.

**Occupation** (28-way) — linear-classifier **accuracy**, swept over how many selected questions
`k` we erase:


| Erase with…              | Accuracy  | Notes                              |
| ------------------------ | --------- | ---------------------------------- |
| nothing (before)         | 0.648     | the embedding leaks occupation     |
| **k = 3** questions      | **0.378** | cleans most of the data mass       |
| **k = 6** questions      | **0.367** |                                    |
| **k = 15** questions     | **0.322** | full covering set                  |
| true labels              | 0.304     | ground-truth erasure (the ceiling) |
| — majority-class floor — | 0.304     | you can't drop below this          |


The *label ceiling* is what you'd get erasing with the **true** one-hot occupation labels — jevu's
zero-shot JEV labels (0.322) land right next to it, and the floor (0.304) is the majority-class
baseline you can't beat.

Two things to notice. **(1)** Just **k = 3** questions do most of the work — 0.648 → 0.378, roughly
two-thirds of the way to the floor — because those three themes clean the bulk of the data (more on
this below). **(2)** Going 3 → 6 → 15 only buys the long tail (0.378 → 0.367 → 0.322), closing the
last gap to the ground-truth erasure.

And it's **selective** — erasing gender leaves occupation readable, and vice versa.

---



## The interesting part: *which* data gets cleaned

We saw that greedy top-k selection keeps the three themes *researcher/scholar*, *healthcare*, and
*artist/performer*. With just **k=3** questions, the natural question is: *what, exactly, did those
three erase?* We
scored each selected question per occupation and marked a profession "covered" (its signal is
captured, so it gets erased) when a selected question's mean score ≥ 0.5.

The result is more subtle — and more useful — than "it covers the biggest classes":

**Coverage is theme-driven, not frequency-driven.** The correlation between a profession's
*coverage* and its *row count* is only **0.16**. The three questions follow semantic themes
(research, healthcare, arts), and coverage follows the themes regardless of how common a profession
is:


| occupation     | count  | coverage | outcome                        |
| -------------- | ------ | -------- | ------------------------------ |
| professor      | 191    | 0.83     | covered (research)             |
| physician      | 56     | 0.98     | covered (healthcare)           |
| **attorney**   | **43** | **0.09** | **survives** — no law question |
| **journalist** | 35     | 0.19     | **survives**                   |
| nurse          | 34     | 0.94     | covered                        |
| chiropractor   | 5      | 0.97     | covered (healthcare)           |
| **rapper**     | **2**  | **0.97** | **covered** (performer)        |
| paralegal      | 3      | 0.02     | survives                       |


A rare rapper (2 rows) is erased because it fits "performer"; a common attorney (43 rows) survives
because no legal question was selected.

**But by data *mass*, it does erase the head of the distribution.** Because the greedy objective
maximizes reconstruction of the pool's **variance**, and variance is dominated by the high-frequency
professions, those frequent professions *drive their theme into the selection* and get captured
first:

- **75% of training rows** belong to professions that k=3 covers — three questions clean three
quarters of the data.
- **6 of the 8 most-represented** professions are covered (only attorney and journalist slip
through).
- The rare tail is weaker and hit-or-miss (**57%** covered).

So the honest statement is: **frequency makes a profession *likely* to be covered — because it drives
a theme into the selection — but not *guaranteed*.** A frequent profession with no matching theme
question still survives.

This is exactly why `k=3` already slashes occupation accuracy (it cleans the bulk of the data), and
why pushing toward the 27-dim label ceiling only buys you the long tail — attorney, journalist,
paralegal, yoga, interior design — each a small slice needing its own theme question.

---



## Making it cheap

The cost is the scoring calls (network-bound). Beyond the select-on-a-sample trick from step 3:

- **Why fit the eraser on the full set, not the sample?** Selection (step 3) only needs enough rows to
*rank* questions; but LEACE (step 4) estimates the covariance `Σ` in the full embedding dimension
(1536), which *needs* rows to be well-conditioned. Scoring the `k` winners is cheap, so you get both —
select on 250, fit on all.
- **Threaded, cached, calibrated.** All `questions × texts` cells score concurrently through one
pooled HTTP client; every score is cached per `(model, text, question)`, so re-runs and repeated
texts are free.

---



## Takeaways

1. **You can erase any concept you can name** — no labels, no retraining the embedding model. Bring
  embeddings from any model plus one English sentence.
2. **Match the number of directions to the concept's rank.** Binary attributes (gender) need ~1
  question; many-valued identities (occupation) need a covering set (~k for a k-way concept). Use
   `expand="auto"` and let the LLM size it.
3. **Zero-shot JEV labels ≈ ground-truth labels for erasure.** Occupation erased to 0.322 vs. a
  0.304 label ceiling; gender to chance.
4. **Erasure is selective.** Removing one concept leaves the others readable — you trade a little
  utility on the target attribute, not everything.
5. **Label-free greedy selection cleans by *variance*, which means by *data mass*.** k=3 erases ~75%
  of rows because the frequent professions drive their themes into the selection. But coverage is
   mediated by theme, not raw count — a frequent class with no matching question (attorney) survives,
   a rare class that fits a theme (rapper) gets erased. Correlation of coverage with frequency is
   only 0.16; correlation with *theme fit* is what matters.
6. **Interpretability is the real payoff.** Because every dimension is a readable question, you can
  point at *exactly* which concepts (and thus which slices of your data) an eraser removes — and
   which survive. Dense-vector debiasing can't tell you that.

---



## Try it

```bash
pip install "jevu[jev,openai]"
```

See the runnable notebooks in [`examples/`](../examples): `occupation_erasure.ipynb` (top-k sweep,
before/after PCA, profession-coverage charts), `gender_erasure.ipynb`, and `search_debias.ipynb`
(top-k retrieval before/after erasing gender from query + document embeddings).

### References

- Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman. *LEACE: Perfect Linear Concept
Erasure in Closed Form.* NeurIPS 2023.
- De-Arteaga et al. *Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes
Setting.* FAT\* 2019. (evaluation dataset)

