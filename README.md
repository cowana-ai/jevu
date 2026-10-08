# jevu

Erase a **protected attribute** (gender, age, sentiment, "unsafe/NSFW") from text embeddings —
**locally, for free, offline**. Define the concept in plain English, label it zero-shot with
[**laya**](https://huggingface.co/convaiinnovations/laya) (an open-weights, on-device calibrated
scorer), and remove its linear signal with a provable eraser
([LEACE](https://arxiv.org/abs/2306.03819) / [INLP](https://aclanthology.org/2020.acl-main.647/)).

You bring **(1) a concept as 1–2 yes/no questions, (2) your raw texts, (3) their embeddings** from any
model. `jevu` labels the concept per text (no annotation), and returns embeddings with that concept's
*linear* signal removed — leaving the rest intact. **Nothing leaves your machine**: laya runs locally,
the erasers are pure numpy. That's exactly what sensitive data (NSFW, toxic, PII-adjacent) needs.

```python
from jevu import ConceptScrubber, LayaLabeler

X = your_model.encode(texts)                           # embeddings from ANY model (jevu never embeds)
# fully local: you give the questions, laya scores them on-device, LEACE erases — no API, no key
scrubber = ConceptScrubber(labeler=LayaLabeler(questions=[
    "Does the text describe a woman?", "Does the text describe a man?"]))
X_clean = scrubber.fit_transform(X, texts=texts)

print(scrubber.audit(X, texts=texts))
# {'concept_auc_before': 1.00, 'concept_auc_after': 0.55, 'n_concepts': 2}
```

Prefer to just name the attribute? `ConceptScrubber("gender")` has an LLM *write* the questions for
you — convenient, but that one step calls OpenAI (needs `OPENAI_API_KEY`); laya scoring and erasure
stay local. Already have labels? Skip the scorer entirely — then it's pure numpy/scikit-learn:

```python
scrubber.fit(X, labels=y)                             # your embeddings and labels, no scorer at all
```

## Two paths — pick by the concept's *rank*

`jevu`'s sweet spot is **low-rank / binary protected attributes**, where one or two yes/no questions
span the concept and laya drives the probe to chance. The `binary` flag selects the path:

```python
# binary=True (default): LLM writes 2-3 yes/no questions, laya scores them locally, LEACE erases.
#   Ideal for gender, age, sentiment, tone, "unsafe/NSFW" — laya's home turf.
ConceptScrubber("gender", binary=True).fit_transform(X, texts=texts)

# binary=False: LLM reads the attribute off each document (open-vocabulary), one-hots it, LEACE erases.
#   For high-cardinality identities (occupation, topic). See the honest caveats below.
ConceptScrubber("occupation", binary=False).fit_transform(X, texts=texts)
```

Pass your own questions for full control:

```python
from jevu import ConceptScrubber, LayaLabeler
Q = ["Does the text contain toxic, hateful, or unsafe content?",
     "Does the text use insults, profanity, or offensive language?"]
scrubber = ConceptScrubber(labeler=LayaLabeler(questions=Q, cache_dir=".cache"))
X_clean = scrubber.fit_transform(X, texts=texts)
```

**Why laya.** It's an open-weights, local, calibrated typed-question scorer (same `noul = P(true)`
output as hosted scorers) — no API, no per-call cost, offline, private. For a **low-rank** concept a
couple of questions fully span it, so you get chance-level erasure on-device. (For high-cardinality
*identities*, a single scorer under-spans the concept — see below.)

## Install

```bash
pip install jevu              # core: numpy + scikit-learn only (the erasers)
pip install "jevu[laya]"      # + laya (torch/transformers), for local zero-shot concept labeling
pip install "jevu[openai]"    # + openai, only for LLM question-expansion / entity extraction
pip install "jevu[examples]"  # + sentence-transformers/matplotlib/jupyter, to run the notebooks
pip install "jevu[dev]"       # everything, incl. pytest (development)
```

**jevu never computes embeddings** — you pass them in from any model (local or hosted). Set
`OPENAI_API_KEY` only if you use `expand=True` / `binary=False` (the LLM step); laya scoring needs no
key. Pass `cache_dir=...` to cache scores on disk so re-runs are free.

## Examples

Runnable notebooks in [`examples/`](examples):

- **`nsfw_erasure.ipynb`** — erase "unsafe/NSFW" content **fully locally** (local embeddings + laya +
  LEACE) on the open-source `wiki_toxic` data; probe AUC 0.92 → ~0.3, with PCA, clustering, and
  retrieval before/after. The flagship for *private, offline* erasure.
- **`gender_erasure.ipynb`** — gender erasure with before/after PCA.
- **`search_debias.ipynb`** — top-k retrieval before/after erasing gender from query + document
  embeddings.

## How it works (the math)

A concept `c` is **linearly encoded** in an embedding `x` if a linear probe reads it:
`p(c|x) = σ(wᵀx + b)`. The unit direction `û = w/‖w‖` is the axis the concept varies along.

**Remove the direction (nullspace projection).** With `P = I − û ûᵀ`, `x' = Px` zeroes the concept
coordinate (`ûᵀx' = 0`), so a probe that relied on `û` now reads a constant.

- **LEACE** (`method="leace"`, default) is the closed-form optimum: an affine map that makes
  `cov(r(X), c) = 0` *exactly*, with the least squared change to `x`. It whitens, projects out the
  concept subspace, un-whitens — removing ~`rank` directions in one shot.
- **INLP** (`method="inlp"`) iterates {fit probe → project out its direction} until no probe beats
  chance. Robust; can over-project if run too long (early-stopped here).

Erasing a `k`-dimensional concept needs ~`k` directions — which is why **binary attributes are easy**
(1–2 questions) and high-cardinality identities are hard.

## API

- `ConceptScrubber(concept=None, method="leace", binary=True, labeler=None, **eraser_kwargs)`
  - `.fit(embeddings, texts=None, labels=None)`, `.transform(X)`, `.fit_transform(...)`, `.audit(...)`.
- `LayaLabeler(concept=None, questions=[...], model="convaiinnovations/laya", device=None, cache_dir=...)`
  — local scorer: `.score(texts)`, `.score_questions(qs, texts)`, `.label(texts)`.
- `EntityExtractor(concept, llm_model=..., cache_dir=...)` — per-doc extraction → one-hot (`binary=False`).
- `LeaceEraser()`, `InlpEraser(max_iters=20, tol_auc=0.55)` — low-level `fit(X, y)/transform(X)`.
- `concept_auc(X, y)`, `erasure_report(X_before, X_after, y)`, `tpr_gap(true, pred, group)` — auditing.

## Performance & logging

laya scoring runs on-device (GPU/`mps`/CPU; set `device=...`, `batch_size=...`), shows a `tqdm` bar,
and **caches every `(model, question, text)` score** (`cache_dir=...`) so re-runs are free. The
erasers are fast numpy. The library logs to the `jevu` logger with a `NullHandler` — opt in with
`logging.basicConfig(level=logging.INFO)`.

## Honest limits (high-cardinality & OOD)

- **High-cardinality identities** (occupation, topic) are a different regime: one scorer under-spans
  the ~`k`-dim concept, so `binary=False` (per-doc extraction → one-hot) erases far more than scored
  questions — but completeness trades off against collateral damage, and it's O(N) LLM calls.
- **Linear erasure is subspace-specific.** It removes exactly the directions spanned by your fit data;
  it does **not** generalize to concepts/values your corpus never contained (re-fit when the
  distribution shifts).
- **Erasure ≠ content filter.** It removes the *latent* direction, not the words. Pair with filtering
  if you must block content.

## References

- Belrose et al. *LEACE: Perfect Linear Concept Erasure in Closed Form.* NeurIPS 2023.
- Ravfogel et al. *Null It Out (INLP).* ACL 2020.
- laya — *convaiinnovations/laya* (open-weights calibrated decision model). `wiki_toxic` (evaluation).

## License

MIT
