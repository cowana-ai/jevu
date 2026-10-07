# Erasing Concepts from Embeddings — in Plain English

*You can remove gender, occupation, or any attribute you can name from off-the-shelf embeddings —
label it zero-shot with an LLM, erase it in closed form, and actually see which parts of your data
got cleaned.*

---

When I reach for an off-the-shelf embedding model, I'm thinking about one thing: does it capture
meaning well enough for my task. I'm almost never thinking about what *else* it captured along the
way. That turns out to be the mistake.

Because embeddings are good at their job, they encode far more than the thing you care about. Embed a
professional bio with a standard model and a simple linear probe reads off the person's **gender at
~1.00 AUC** and their **occupation at ~0.65 AUC** — *before you've trained anything*. Every
classifier, ranker, and recommender you build on top quietly inherits that signal. And with it, the
bias.

This post is about taking it back out. I wrote a small library, `jevu`, to do exactly this, and the
part I find genuinely interesting isn't that it works — it's that it tells you, concept by concept
and slice by slice, *what it removed*. Most debiasing methods can't.

## The usual fix is too expensive, so nobody does it

The textbook answer is: annotate the attribute on your whole corpus, then adversarially train it out.
That's two problems. It costs an annotation campaign you'll never budget for, and it only works for
attributes you *already* labeled. The long tail of "wait, is my search ranker leaking gender?"
questions never gets answered, because answering each one is a project.

So `jevu` takes a different route. You bring three things:

> **(1) your embeddings** (from any model), **(2) your raw texts**, and **(3) a concept named in
> plain English** — even a single ambiguous word like `"gender"` or `"occupation"`. An LLM turns the
> word into concrete questions, a scorer labels each one per text, and **LEACE** removes the whole
> thing in closed form — leaving everything else intact.

```python
from jevu import ConceptScrubber

X = your_model.encode(texts)                         # embeddings from ANY model
scrubber = ConceptScrubber(concept="occupation", method="leace",  # just the bare word
                           expand=True, n_questions=15)            # LLM writes a pool of 15 questions
scrubber.fit(X, texts=texts)                         # expand -> score -> select -> fit the eraser
X_clean = scrubber.transform(X_new)                  # erase it from new embeddings

scrubber.audit(X, texts=texts)
# {'concept_auc_before': 0.65, 'concept_auc_after': 0.32, 'n_concepts': 15}
```

No annotation. No retraining the embedding model. Just numpy, scikit-learn, and a scorer. That's the
whole pitch — now let me show you how each piece works, because the *how* is where it gets fun.

---

## Erasing a concept is four steps, and only one of them is hard

You hand in one ambiguous word and get back an eraser. Under the hood that word flows through four
steps: **expand → score → select → erase.** I'll use `"occupation"` the whole way through.

One piece of intuition carries the entire design:

> **Erasing a `k`-dimensional concept means projecting out ~`k` directions.**

"Gender" is essentially binary — it lives on *one* direction, so one question can carry it. A 28-way
occupation *identity* lives in a ~27-dimensional subspace — no single question could ever capture it.
This is why a bare word has to become *several* questions before you can erase it. Everything that
follows is in service of that one sentence.

### Step 1 — Turn the word into questions (the LLM's job)

A bare noun like `"occupation"` isn't something you can score directly. "Is this text occupation-y?"
is meaningless. So an LLM first **expands** it into concrete, discriminative yes/no questions that
*together cover* the concept's facets:

```python
scrubber = ConceptScrubber(concept="occupation", expand=True, n_questions=15)
scrubber.fit(X, texts=texts)
scrubber.concept_questions_
# ['Does the text describe a healthcare professional?',
#  'Does the text describe an educator or academic?',
#  'Does the text describe an artist or performer?',
#  'Does the text describe a legal professional?', ...]   # 15 covering questions
```

Notice I ask for **more questions than I'll actually erase with** — a pool of 15. That redundancy is
the point: a bigger pool covers more facets, and Step 3 prunes it back down to the few that matter.
(If you'd rather let the model size the set itself — 1–2 for something binary, a covering set for an
identity — pass `expand="auto"`. But the select stage below wants an explicit pool, so here I fix
`n_questions=15`.)

### Step 2 — Score a sample into a matrix `Z`

Now the questions are scorable. **JEV** is a calibrated per-question scorer: hand it a text and a
yes/no question, get back a number in `[0, 1]`. No training set, no threshold tuning. Run it over a
**sample of `n` rows × the `p` questions** and you get a score matrix:

```
Z ∈ ℝ^{n × p},   Z[i, j] = JEV(text_i, question_j) ∈ [0, 1]
```

Here's the way I think about `Z`: it's a tiny, *interpretable* embedding. Every column is a named
axis. And I only score a sample — 250 rows, say — because the next step just needs enough data to
*rank* questions, not the whole corpus.

### Step 3 — Keep only the questions that matter (greedy, label-free)

Fifteen questions is good for coverage, but many of them overlap — "healthcare professional" and
"works in medicine" are nearly the same column. So I keep the `k` that best **reconstruct the whole
pool.** This is classic *column subset selection*, and I do it **label-free**: the selector looks
only at `Z`, never at the true occupation labels.

Concretely: center `Z`, then greedily add, one at a time, the question whose column most reduces the
least-squares reconstruction residual of the *entire* pool from the columns chosen so far. With `A`
the already-selected columns plus a candidate `c`:

```
pick c that minimizes   ‖Z − A A⁺ Z‖_F
```

`A⁺` is the Moore–Penrose pseudoinverse, so `A A⁺ Z` is the orthogonal projection of every column of
`Z` onto the span of the selected columns, and `‖·‖_F` is the Frobenius norm — the total residual
across all questions. In plain words the objective asks: *which few questions let me linearly predict
all the others?*

<details>
<summary><b>Why <code>A A⁺</code> is a projection (the SVD view)</b></summary>

The cleanest way to see what `A A⁺` *does* is through the SVD of the selected columns,
`A = U Σ Vᵀ`, where `U`'s columns are an **orthonormal basis of `col(A)`** (the subspace the selected
questions span). The pseudoinverse is `A⁺ = V Σ⁻¹ Uᵀ`, so the scaling and the input-side rotation
cancel and you're left with just:

```
A A⁺ = (U Σ Vᵀ)(V Σ⁻¹ Uᵀ) = U (Σ Σ⁻¹) Uᵀ = U Uᵀ
```

So `A A⁺ = U Uᵀ`, the standard **orthogonal projector onto `col(A)`**. Note `U` and `U Uᵀ` are
different objects doing different jobs:

- **`U` (shape `n × r`) is a *basis*** — a set of orthonormal axes *describing* the subspace. You
  can't drop an arbitrary vector onto the subspace by multiplying by `U`; its input is coordinates,
  not an ambient vector.
- **`U Uᵀ` (shape `n × n`) is a *transformation*** — the machine that *performs* the projection.
  `Uᵀ` reads a vector's coordinates along the orthonormal axes (`UᵀU = I`), and `U` rebuilds them,
  so `U Uᵀ x = Σᵢ (uᵢᵀx) uᵢ` is `x` flattened straight down onto the subspace.

Picture `col(A)` as a tabletop inside a room: `U` is two meter-sticks taped to the table marking its
axes; `U Uᵀ` is a laser in the ceiling that, for any object `x` in the room, points to the spot on the
table directly beneath it. It's idempotent (`(U Uᵀ)² = U Uᵀ` — projecting twice changes nothing) and
symmetric (`(U Uᵀ)ᵀ = U Uᵀ`), the two hallmarks of an orthogonal projector. (If the selected columns
ever spanned *all* of `ℝⁿ`, `U` would be square and `U Uᵀ = I` — projecting onto everything is a
no-op; here `r < n`, so it genuinely flattens.)

In code we only have `A`, not its orthonormal basis `U`, so `np.linalg.pinv` runs the SVD for us and
`A A⁺` reduces to exactly this `U Uᵀ` projection.

</details>

In practice three questions usually suffice:

```python
scrubber = ConceptScrubber(concept="occupation", expand=True, n_questions=15,
                           select_k=3, select_sample=250)   # select on a 250-row sample
scrubber.fit(X, texts=texts)
scrubber.selected_questions_
# ['Does the text identify someone as a researcher/scholar?',
#  'Does the text mention a healthcare profession?',
#  'Does the text indicate an artist or performer?']
```

Then only those `k` winners get scored on the *full* dataset — cost drops from `pool × N` to
`pool × sample + k × N` — and what you're left with is a small, **reusable** eraser you can apply to
new data forever. *Which* data those three questions end up cleaning is a story in itself, and it's my
favorite part of this whole thing. More on that below.

### Step 4 — Erase it with LEACE

Now I have embeddings `X ∈ ℝ^{N × d}` and the selected concept scores `Z ∈ ℝ^{N × k}`. **LEACE**
(*LEAst-squares Concept Erasure*, Belrose et al., NeurIPS 2023) finds the affine map `r(x)` that makes
the concept **linearly unreadable** — `cov(r(X), Z) = 0` *exactly* — while changing `x` as little as
possible. Here's why it looks the way it does.

Start with the easy case: one direction. Suppose the concept were a single unit direction `û`.
Projecting it out is just

```
P = I − û ûᵀ,     x' = P x = x − (ûᵀx) û
```

which zeroes the concept coordinate: `ûᵀx' = ûᵀx − (ûᵀx)(ûᵀû) = 0`, since `ûᵀû = 1`. A probe that
relied on `û` now reads a constant. **But there's a catch**, and it's the whole reason LEACE isn't a
one-liner: projecting along the *raw* axis is only minimal-damage if your coordinates are uncorrelated
and equal-variance. Real embeddings are neither — some directions carry far more variance than others.
So LEACE whitens first.

The full map is three moves:

1. **Whiten.** Let `Σ = cov(X)`. The transform `W = Σ^{-1/2}` sends `X` to a space where
   `cov(WX) = I` — every direction equal-variance, so ordinary orthogonal projection finally *is* the
   minimal-damage operation.

2. **Project out the concept subspace.** In whitened space the concept lives in the column space of
   `W Σ_{XZ}`, where `Σ_{XZ} = cov(X, Z)` is the cross-covariance — how each embedding direction
   co-varies with each question. Take an orthonormal basis `U` of that column space and build
   `P = I − U Uᵀ`. For a `k`-dim concept this removes `k` directions. (There's that intuition again,
   now made literal.)

3. **Un-whiten and re-center.** Map back with `W⁻¹ = Σ^{1/2}` and restore the mean `μ`:

```
r(x) = μ + Σ^{1/2} · P · Σ^{-1/2} · (x − μ),      P = I − U Uᵀ
```

Because the projection kills exactly the whitened concept subspace, you get `cov(r(X), Z) = 0` for
free — **no linear probe can recover the concept.** And because it's an orthogonal projection *in the
whitened metric*, it's the least-squares-optimal such map: the smallest distortion of `x` that
achieves erasure. One closed-form shot, no iteration. (If you already have hard labels instead of JEV
scores, pass those as `Z` — the math doesn't care.)

---

## It works, and the numbers say how well

I evaluated on **Bias in Bios** (professional bios, 28 occupations, binary gender) with
`text-embedding-3-small`, on a held-out test set.

**Gender** collapses to chance. Linear-probe **AUC**:

| Concept | Before | After (jevu) | Chance |
| ------- | ------ | ------------ | ------ |
| Gender  | 0.996  | **0.515**    | ~0.50  |

**Occupation** is the interesting one, because it isn't binary. This is the canonical pipeline —
expand `occupation` into **15 diverse, fine-grained questions**, greedily **select k=3**, erase —
reported on **both** metrics, because they disagree and the disagreement is the whole point:

| metric | before | erased (k=3) | true-label ceiling |
| --- | --- | --- | --- |
| top-1 accuracy | 0.648 | **0.404** | 0.304 |
| one-vs-rest AUC | 0.978 | **0.947** | 0.519 |

Look at the two rows. Accuracy **craters** (0.648 → 0.404, heading for the 0.304 majority floor) — but
AUC **barely moves** (0.978 → 0.947). The k=3 pipeline collapses the *top-1 guess* while leaving the
occupation signal almost fully recoverable. If I'd only reported accuracy, I'd have told you occupation
was erased. It isn't.

### Why accuracy flatters this — and the honest metric

Accuracy saturates at the majority floor: once the probe gives up and predicts *professor* for everyone
(it does, for **89%** of examples at k=3), it reads ~0.40 whether the signal is **gone** or merely **no
longer top-1**. One-vs-rest AUC ignores priors and asks the real question — is the signal still
linearly recoverable? — and at k=3 the answer is *yes*. Keep the whole pool (**k=15**) and AUC finally
moves, but *unevenly*:

| erase | occupation AUC | dominant / head | minority / tail |
| --- | --- | --- | --- |
| before | 0.978 | 0.972 | 0.981 |
| k = 3 | 0.947 | 0.902 | 0.973 |
| k = 15 | 0.797 | **0.582** | 0.912 |
| true labels | 0.519 | 0.470 | 0.547 |

The **head gets erased** (dominant AUC 0.97 → 0.58, nearing the 0.47 ceiling) once you keep enough
fine-grained questions — "nurse / physician / professor" separate what the "healthcare" umbrella used
to merge. The **tail survives** (minority 0.91), because 15 questions get spent naming *common* jobs,
not rare ones. Feed LEACE the **true labels** — a target that spans the full ~27-dim identity — and AUC
collapses to chance (0.52), cross-covariance ~`1e-17`.

That's the **"erase a `k`-dim concept → remove ~`k` directions"** rule, made concrete. Gender is 1-dim,
so one question hits chance. Occupation is ~27-dim, so 15 questions only partly span it, and k=3 — three
head-representative picks — barely dents the AUC at all. The method is as complete as your question set
is high-rank.

So the fair summary: **for a binary attribute, zero-shot erasure ≈ ground truth; for a high-cardinality
identity, k=3 kills top-1 but leaves the signal, and you approach full erasure only as the question set
approaches the concept's rank.** Accuracy says "done"; AUC says "mostly, for the common classes." I
trust AUC. The *which-classes* part is the next section.

And one more property that matters more than it might seem: erasure is **selective.** Remove gender
and occupation stays readable; remove occupation and gender stays readable. You're not nuking the
embedding, you're excising one concept.

---

## The interesting part: *which* data gets cleaned

The three themes the greedy selector kept were *researcher/scholar*, *healthcare*, and
*artist/performer*. So the obvious question is: with only three questions, *what exactly did they
erase?* I scored each selected question per occupation and called a profession "covered" — its signal
captured, so it gets erased — when a selected question's mean score cleared 0.5.

My first guess was "it covers the biggest classes." That guess was wrong, and the real answer is more
useful.

**Coverage is theme-driven, not frequency-driven.** The correlation between a profession's *coverage*
and its *row count* is only **0.16** — basically nothing. The three questions follow semantic themes,
and coverage follows the themes no matter how common a profession is:

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

Look at the extremes. A **rapper with 2 rows** gets erased, because it fits "performer." A **common
attorney with 43 rows** survives, because no legal question was ever selected. Frequency isn't driving
this; *theme fit* is.

**And yet — by data mass, it really does erase the head of the distribution.** The two statements
aren't in tension. The greedy objective maximizes reconstruction of the pool's *variance*, variance is
dominated by the high-frequency professions, and so those frequent professions are the ones that
*drive their theme into the selection* in the first place. The numbers:

- **75% of training rows** belong to professions that `k=3` covers. Three questions clean three
  quarters of the data.
- **6 of the 8 most-represented** professions are covered (only attorney and journalist slip through).
- The rare tail is weaker and hit-or-miss — **57%** covered.

So here's the honest version: **frequency makes a profession *likely* to be covered — because it
drives a theme into the selection — but never *guaranteed*.** A frequent profession with no matching
theme question still walks right through.

This is also the mechanism behind the k-sweep. `k=3` already slashes accuracy because it cleans the
bulk of the data, and pushing toward the label ceiling only buys the long tail — attorney, journalist,
paralegal, yoga, interior design — each a small slice that needs its own theme question. The math and
the data tell the same story.

---

## Making it cheap

The cost here is the scoring calls, which are network-bound. Two things keep it manageable, beyond the
select-on-a-sample trick from Step 3:

- **I fit the eraser on the full set, not the sample — on purpose.** Selection only needs enough rows
  to *rank* questions, but LEACE estimates a covariance in the full 1536-dim embedding space, which
  needs rows to be well-conditioned. Scoring the three winners on everything is cheap, so I select on
  250 and fit on all. Best of both.
- **Everything is threaded and cached.** All `questions × texts` cells score concurrently through one
  pooled HTTP client, and every score is cached per `(model, text, question)` — so re-runs and
  repeated texts are free.

---

## What I'd take away from this

1. **You can erase any concept you can name.** No labels, no retraining the embedding model — just
   embeddings from any model plus one English sentence.
2. **Match the number of directions to the concept's rank.** Binary attributes need ~1 question;
   many-valued identities need a covering set (~`k` for a `k`-way concept). Over-generate a pool and
   let Step 3 prune it, or let the LLM size it with `expand="auto"`.
3. **How complete the erasure is depends on the concept's rank — and on your metric.** Gender
   (1-dim) goes to chance; zero-shot labels there ≈ ground truth. Occupation (~27-dim) only gets a
   *covering* set of ≤15 questions, so it collapses top-1 accuracy (0.648 → 0.322) but a probe still
   recovers it by AUC (~0.90 vs 0.52 for true-label erasure). Measure with AUC, not accuracy — and add
   questions until the AUC, not the accuracy, stops moving.
4. **Erasure is selective.** You trade a little utility on the target attribute, not everything else.
5. **Label-free selection cleans by variance, which means by data mass.** `k=3` erases ~75% of rows,
   but coverage is mediated by *theme*, not raw count — a frequent class with no matching question
   survives, a rare class that fits a theme gets erased. Correlation with frequency is only 0.16;
   correlation with theme fit is what matters.
6. **Interpretability is the real payoff.** Because every dimension is a readable question, you can
   point at *exactly* which concepts — and which slices of your data — an eraser touches, and which it
   misses. Dense-vector debiasing can't tell you that, and I think that's the whole game.

An embedding is a compression of meaning, and it compresses everything it saw — including the things
you'd never have chosen to carry forward. Erasure is just deciding, out loud and on the record, which
meaning you refuse to keep. The nice part is that with `jevu` you don't have to take that decision on
faith — you can read, concept by concept, exactly what you let go.

---

## Try it

```bash
pip install "jevu[jev,openai]"
```

The runnable notebooks are in [`examples/`](../examples): `occupation_erasure.ipynb` (the top-k sweep,
before/after PCA, and the profession-coverage charts), `gender_erasure.ipynb`, and `search_debias.ipynb`
(top-k retrieval before and after erasing gender from query + document embeddings).

### References

- Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman. *LEACE: Perfect Linear Concept
  Erasure in Closed Form.* NeurIPS 2023.
- De-Arteaga et al. *Bias in Bios: A Case Study of Semantic Representation Bias in a High-Stakes
  Setting.* FAT\* 2019. (evaluation dataset)
