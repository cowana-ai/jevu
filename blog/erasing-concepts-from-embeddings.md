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
> described in plain English**. An LLM scorer labels the concept per text, and a provable linear
> method removes it — leaving everything else intact.

```python
from jevu import ConceptScrubber

X = your_model.encode(texts)                        # embeddings from ANY model
scrubber = ConceptScrubber(concept="Does the text describe a woman?", method="leace")
scrubber.fit(X, texts=texts)                        # JEV labels the concept, fits the eraser
X_clean = scrubber.transform(X_new)                 # erase it from new embeddings

scrubber.audit(X, texts=texts)
# {'concept_auc_before': 1.00, 'concept_auc_after': 0.55, 'n_concepts': 1}
```

No annotation. No retraining of the embedding model. Just numpy and scikit-learn plus a scorer.

---



## The method, in four layers



### 1. Label the concept zero-shot (JEV)

A concept is just a yes/no question — `"Does the text describe a woman?"`. We score every text
against it with **JEV**, a calibrated per-question scorer that returns a number in `[0, 1]`. No
training set, no threshold tuning: one sentence replaces an annotation campaign, and the labels
come out calibrated. If you *already* have labels, skip JEV entirely — the erasure math only needs
*a* label per example.

### 2. Erase the linear signal (LEACE / INLP)

A concept `c` is **linearly encoded** in an embedding `x` if a linear probe reads it:
`p(c|x) = σ(wᵀx + b)`. The unit direction `û = w/‖w‖` is the axis the concept lives on. Split
`x = x∥ + x⊥` into its component along `û` and the rest. The projection

```
P = I − û ûᵀ        x' = P x = x − (ûᵀx) û
```

zeroes the concept coordinate: `ûᵀx' = ûᵀx − (ûᵀx)(ûᵀû) = 0`. A probe that relied on `û` now reads
a constant — the concept is unreadable along that axis.

- **INLP** (*Null It Out*, Ravfogel et al. ACL 2020) repeats {fit probe → project out its
direction} until no linear probe beats chance. Robust and iterative; can over-project if run too
long.
- **LEACE** (*Perfect Linear Concept Erasure*, Belrose et al. NeurIPS 2023, the default) is the
closed-form optimum: an affine map that makes `cov(r(X), c) = 0` **exactly**, with the least
squared change to `x`. One shot, minimal collateral damage.



### 3. Handle multi-faceted concepts (LLM expansion)

"Gender" is one direction. "Occupation" is not. A key piece of intuition:

> **Erasing a k-dimensional concept needs ~k directions to project out.**

Gender is essentially binary — one direction. A 28-way occupation *identity* lives in a ~27-dim
subspace. You can't kill it with a single question. So for a high-level attribute you pass the bare
word and let an LLM **expand** it into a covering set of concrete, discriminative yes/no questions:

```python
scrubber = ConceptScrubber(concept="occupation", expand=True, n_questions=15)
scrubber.fit(X, texts=texts)
print(scrubber.concept_questions_)   # ["Does the text describe a healthcare professional?", ...]
```

Or let the LLM decide the count with `expand="auto"` — **1–2 questions for a binary concept**, a
covering set (~15) for a many-valued identity. Fewer questions = fewer scoring calls.

### 4. Keep only the questions that matter (top-k selection)

Over-generating a pool of 15 questions is good for *coverage*, but you rarely need all 15 to erase —
many overlap. So after expansion we **greedily select the** `k` **questions that best reconstruct the
whole pool**, and erase just those. This is a classic *column subset selection* done **label-free**:
we never look at the occupation labels, only at the matrix of JEV scores.

The greedy objective, concretely: let `Z` be the `(n_texts × 15)` matrix of scores (mean-centered).
Forward selection adds, one at a time, the question whose column most reduces the residual of
least-squares reconstructing the *whole* pool from the questions picked so far:

```
pick the column c that minimizes   ‖Z − A·A⁺·Z‖_F ,   where A = columns selected ∪ {c}
```

Each step logs its pick. Three questions typically suffice:

```python
scrubber = ConceptScrubber(concept="occupation", expand=True, n_questions=15,
                           select_k=3, select_sample=250)   # select on a 250-row sample
scrubber.fit(X, texts=texts)
scrubber.selected_questions_
# ['Does the text identify someone as a researcher/scholar?',
#  'Does the text mention a healthcare profession?',
#  'Does the text indicate an artist or performer?']
```

Why this is cheap: `select_sample=250` scores the *pool* on only 250 rows (selection is a
low-dimensional ranking problem — it doesn't need many rows), picks the `k` winners, then scores just
those `k` on the full data to fit the eraser. Cost drops from `pool × n` to
`pool × select_sample + k × n`. The selected questions are a small, **reusable** eraser you can deploy
on new data.

The *consequence* of selecting by reconstruction — what those `k` questions actually end up cleaning —
is the subject of the next two sections.

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

The cost is the scoring calls (network-bound). Beyond the sample-then-fit trick from layer 4:

- **Why fit on the full set at all?** Selection only needs enough rows to *rank* questions; but LEACE
estimates a covariance in the full embedding dimension (1536), which *needs* rows to be
well-conditioned. Scoring the `k` winners is cheap, so you get both — select on 250, fit on all.
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

See the runnable notebooks in `[examples/](../examples)`: `occupation_erasure.ipynb` (top-k sweep,
before/after PCA, profession-coverage charts), `gender_erasure.ipynb`, and `search_debias.ipynb`
(top-k retrieval before/after erasing gender from query + document embeddings).

### References

- Ravfogel, Elazar, Gonen, Twiton, Goldberg. *Null It Out: Guarding Protected Attributes by
Iterative Nullspace Projection.* ACL 2020. (INLP)
- Belrose et al. *LEACE: Perfect Linear Concept Erasure in Closed Form.* NeurIPS 2023.
- De-Arteaga et al. *Bias in Bios.* FAT 2019. (evaluation dataset)

