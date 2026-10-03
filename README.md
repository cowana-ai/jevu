# jevu

Erase a **target concept** from text embeddings — defined in plain English, labeled
zero-shot with [JEV](https://openrouter.ai/), and removed with a provable linear method
([INLP](https://aclanthology.org/2020.acl-main.647/) or
[LEACE](https://arxiv.org/abs/2306.03819)).

You bring **(1) a concept as a question, (2) your raw texts, (3) their embeddings** from any
off-the-shelf model. `jevu` labels the concept per text (no annotation needed) and
returns embeddings with that concept's *linear* signal removed — while leaving the rest intact.
Already have labels? Skip JEV and pass them directly.

Bring **embeddings from any model** plus a **concept** — JEV labels the concept, the eraser removes it:

```python
from jevu import ConceptScrubber

X = your_model.encode(texts)                 # embeddings from ANY model (jevu never embeds)
scrubber = ConceptScrubber(concept="Does the text describe a woman?", method="leace")
scrubber.fit(X, texts=texts)                 # JEV labels the concept, fits the eraser
X_clean = scrubber.transform(X_new)          # erase the concept from new embeddings

print(scrubber.audit(X, texts=texts))
# {'concept_auc_before': 1.00, 'concept_auc_after': 0.55, 'n_concepts': 1}
```

Already have labels? Skip JEV entirely — it's then pure numpy/scikit-learn:

```python
scrubber.fit(X, labels=y)                    # your embeddings and labels
```

### Multi-faceted concepts (LLM expansion)

Pass a high-level — even **terse or ambiguous** — attribute (`"gender"`, `"genders"`, `"age"`,
`"tone"`) and let an LLM expand it into several concrete, discriminative yes/no questions (woman,
man, gendered pronouns, ...); JEV scores each and the whole multi-dimensional concept is erased at
once. This is the robust way to erase a bare word — a single raw noun often yields non-discriminative
JEV scores and erases nothing:

```python
scrubber = ConceptScrubber(concept="gender", expand=True, n_questions=6)
scrubber.fit(X, texts=texts)                # X = your embeddings
print(scrubber.concept_questions_)          # the sub-questions the LLM generated
print(scrubber.audit(X, texts=texts))       # {'concept_auc_before':…, 'after':…, 'n_concepts': 6}
```

**Let the LLM decide how many** with `expand="auto"`: it judges the concept's cardinality and uses the
*fewest* questions needed — **1–2 for a binary concept** (gender, sentiment) so it's fast, a covering
set only for a **many-valued identity** (occupation, topic). Fewer questions = fewer JEV calls:

```python
ConceptScrubber(concept="gender",     expand="auto").fit(X, texts=texts)   # -> ~1-2 questions
ConceptScrubber(concept="occupation", expand="auto").fit(X, texts=texts)   # -> a covering set (~15)
```

## Install

```bash
pip install jevu              # core: numpy + scikit-learn only (the erasers)
pip install "jevu[jev]"       # + httpx, for zero-shot concept labeling via JEV
pip install "jevu[openai]"    # + openai, only for LLM concept expansion (expand=True)
pip install "jevu[examples]"  # + pandas/matplotlib/jupyter, to run the example notebooks
pip install "jevu[dev]"       # everything, incl. pytest (development)
```

**jevu never computes embeddings** — you pass them in from any model. The core install is pure
numpy/scikit-learn; `httpx` (JEV) and `openai` (concept expansion) are lazy extras. Set
`OPENROUTER_API_KEY` for JEV labeling (and `OPENAI_API_KEY` if you use `expand=True`); pass
`cache_dir=...` to cache scores on disk so re-runs are free.

## How it works (the math)

A concept `c` is **linearly encoded** in an embedding `x` if a linear probe reads it:
`p(c|x) = σ(wᵀx + b)`. The unit direction `û = w/‖w‖` is the axis along which the concept varies.

**Remove one direction (nullspace projection).** Split `x = x∥ + x⊥` into its component along
`û` and the rest. The projection

```
P = I − û ûᵀ,      x' = P x = x − (ûᵀx) û
```

zeroes the concept coordinate: `ûᵀx' = ûᵀx − (ûᵀx)(ûᵀû) = 0` (using `ûᵀû = 1`). A probe that
relied on `û` now reads a constant → the concept is unreadable along that axis.

- **INLP** (`method="inlp"`) repeats {fit probe → project out its direction} until no linear
  probe beats chance. Robust, iterative; can over-project if run too long (early-stopped here).
- **LEACE** (`method="leace"`, default) is the closed-form optimum: an affine map that makes
  `cov(r(X), c) = 0` *exactly*, with the least squared change to `x`. One shot, minimal damage.

Because the concept often correlates with what you want to keep, erasure trades a little utility
for fairness — measure it with `audit()` / `tpr_gap()`.

## API

- `ConceptScrubber(concept=None, method="leace", labeler=None, **eraser_kwargs)` — high level.
  - `.fit(embeddings, texts=None, labels=None)`, `.transform(X)`, `.fit_transform(...)`, `.audit(...)`.
- `LeaceEraser()`, `InlpEraser(max_iters=20, tol_auc=0.55)` — low-level `fit(X, y)/transform(X)`.
- `JevLabeler(concept, cache_dir=..., client=...)` — `.score(texts) -> [0,1]`, `.label(texts) -> {0,1}`.
- `concept_auc(X, y)`, `erasure_report(X_before, X_after, y)`, `tpr_gap(true, pred, group)` — auditing.

## Progress & logging

JEV scoring shows a `tqdm` progress bar by default (disable with `progress=False` on
`ConceptScrubber(..., )` components, or it silently no-ops if `tqdm` isn't installed). The library
logs to the `jevu` logger via Python's standard `logging` and attaches a `NullHandler`, so it's quiet
until you opt in:

```python
import logging
logging.basicConfig(level=logging.INFO)    # see labeling / expansion / erasure details
# keep jevu logs but silence the OpenAI/OpenRouter (httpx) request spam:
for _n in ("httpx", "httpcore", "openai"):
    logging.getLogger(_n).setLevel(logging.WARNING)
```

## Performance

The cost is the JEV HTTP calls (network-bound), so **threading is the right tool and is already used**
— calls run concurrently via a thread pool with a shared, pooled HTTP client. Multiprocessing would
not help (it's for CPU-bound work) and would add overhead. To go faster:

- **Raise concurrency:** `ConceptScrubber(..., max_workers=24)` (default 8). Higher = more parallel
  JEV calls, subject to rate limits (429s are retried with backoff).
- **`expand=True` scores all `questions x texts` cells in one pool**, not one question at a time.
- **Cache:** pass `cache_dir=...` — every score is cached per `(model, text, question)`, so re-runs and
  repeated texts are free. The slow case is always a cache *miss* (new concept/text).

The erasers themselves (LEACE/INLP) are fast numpy and not the bottleneck.

## When to use JEV vs. your own labels

The erasure math only needs *a* label per example. If you **already have** the attribute labeled,
use them — they're exact. JEV's value is when the attribute is **unlabeled** (most real corpora),
or the concept is bespoke ("sounds formal", "mentions a competitor") with no dataset — then one
sentence replaces an annotation campaign, and the labels are calibrated.

## References

- Ravfogel, Elazar, Gonen, Twiton, Goldberg. *Null It Out: Guarding Protected Attributes by
  Iterative Nullspace Projection.* ACL 2020. (INLP)
- Belrose et al. *LEACE: Perfect Linear Concept Erasure in Closed Form.* NeurIPS 2023.
- De-Arteaga et al. *Bias in Bios.* FAT* 2019. (evaluation dataset)

## License

MIT
