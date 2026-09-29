# jevu

Erase a **target concept** from text embeddings — defined in plain English, labeled
zero-shot with [JEV](https://openrouter.ai/), and removed with a provable linear method
([INLP](https://aclanthology.org/2020.acl-main.647/) or
[LEACE](https://arxiv.org/abs/2306.03819)).

You bring **(1) a concept as a question, (2) your raw texts, (3) their embeddings** from any
off-the-shelf model. `jevu` labels the concept per text (no annotation needed) and
returns embeddings with that concept's *linear* signal removed — while leaving the rest intact.
Already have labels? Skip JEV and pass them directly.

```python
from jevu import ConceptScrubber

scrubber = ConceptScrubber(concept="Does the text describe a woman?", method="leace")
scrubber.fit(embeddings=X, texts=texts)     # JEV scores the concept, fits the eraser
X_clean = scrubber.transform(X_new)         # scrub unseen embeddings (fit once, apply forever)

print(scrubber.audit(embeddings=X, texts=texts))
# {'concept_auc_before': 1.00, 'concept_auc_after': 0.55}
```

Bring-your-own labels (no JEV call):

```python
scrubber = ConceptScrubber(method="leace")
X_clean = scrubber.fit_transform(X, labels=y)
```

## Install

```bash
pip install jevu            # core (numpy, scikit-learn)
pip install "jevu[jev]"     # + httpx, for zero-shot concept labeling via JEV
```

Set `OPENROUTER_API_KEY` to use JEV labeling. Pass `cache_dir=...` to `JevLabeler` (or via
`ConceptScrubber.fit(..., cache_dir=...)`) to cache scores on disk so re-runs are free.

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
