# A Linear Algebra Lesson: Concept Erasure

Let's build LEACE from the ground up. I'll assume you know vectors, matrices, and dot products, and we'll develop everything else. Think of this as a lecture with a running example.

---

## Part 1 — The objects on the table

We have two things, both as matrices where **each row is one data point**:

- `X` — your embeddings. `N` points (texts), each a vector in `ℝᵈ` (say `d = 1536`).
- `Z` — the concept. Same `N` points, each a vector in `ℝᵏ` (for gender, `k = 1`; for occupation, `k ≈ 15`).

**The question of the whole lesson:** can we transform `X` so that `Z` becomes *impossible to read with a straight ruler* (a linear probe), while barely moving the points?

Keep a picture in your head: `d = 2`. `X` is a cloud of dots in the plane. Each dot also has a color `z` (its concept value). We want to smear the cloud so color can't be predicted from position by any straight line — moving dots as little as possible.

---

## Part 2 — "Linearly readable" = correlation = cross-covariance

A linear probe reads `z` from `x` by taking a dot product with some weight vector `w`: `prediction = wᵀx`. This works **only if position and color are correlated** — if moving along some direction in `X` tends to change `z`.

That correlation is one matrix, the **cross-covariance**. Center both (subtract the mean so we measure *variation*, not location):

```
Σ_XZ = (1/N) · X̃ᵀ Z̃          (shape d × k)
```

Entry `(i, j)` = "how much does embedding-coordinate `i` move together with concept-coordinate `j`."

**The key equivalence of the lesson:**

```
no linear probe can read Z   ⟺   Σ_XZ = 0
```

If `Σ_XZ = 0`, there is literally no direction along which position predicts color. So our goal becomes concrete: **produce `r(X)` with `cov(r(X), Z) = 0`.**

The columns of `Σ_XZ` point in the **directions of `X` that leak the concept**. Call `col(Σ_XZ)` the *concept subspace*. Erasing = removing that subspace from `X`.

---

## Part 3 — Projection: the tool for "remove a subspace"

Remember projections. If `U` holds an **orthonormal basis** of a subspace `S`, then

```
U Uᵀ       = projector ONTO S          (drop a vector straight down onto S)
I − U Uᵀ   = projector OFF S           (keep only the part orthogonal to S)
```

So the *naive* erasure is: find an orthonormal basis `U` of the concept subspace `col(Σ_XZ)`, and apply `I − U Uᵀ` to every point. Flatten the cloud along the color-carrying directions.

**This is almost right, and it's wrong in an important way.** Here's the subtlety that is the entire reason LEACE exists.

---

## Part 4 — Why naive projection over-damages: the metric problem

Orthogonal projection assumes the coordinate axes are **interchangeable** — same scale, uncorrelated. Real embeddings are not. Some directions have huge variance, some tiny; pairs of directions are correlated (the cloud is a stretched, tilted ellipse, not a round ball).

In a tilted ellipse, "perpendicular" in the naive sense is *not* the direction of least damage. If you project straight down along the concept direction, you also disturb every direction correlated with it, and you remove more signal than necessary. The "closest point" in a stretched space is measured differently than in a round one.

Linear algebra has a fix: **change to a basis where the cloud is round.** Then ordinary perpendicular projection *is* minimal-damage. That change of basis is called whitening.

---

## Part 5 — Covariance as a geometry, and its square root

The shape of the cloud is the **covariance matrix**:

```
Σ = (1/N) · X̃ᵀ X̃          (shape d × d, symmetric, positive-definite)
```

`Σ` *is* the ellipse. A symmetric positive-definite matrix has an **eigendecomposition** `Σ = Q Λ Qᵀ`:
- `Q` = rotation to the ellipse's principal axes,
- `Λ` = diagonal of variances along those axes (how stretched each axis is).

Because it's diagonalizable with positive eigenvalues, we can take a **matrix square root** — scale each axis by `√λ`:

```
Σ^{1/2}  = Q Λ^{1/2} Qᵀ       (rotate → stretch by √λ → rotate back)
Σ^{-1/2} = Q Λ^{-1/2} Qᵀ      (rotate → shrink by 1/√λ → rotate back)
```

This `Σ^{-1/2}` is the magic wand.

---

## Part 6 — Whitening: make the cloud round

Define the **whitening transform** `W = Σ^{-1/2}` and look at the cloud in whitened coordinates `u = W x̃`:

```
cov(WX) = W Σ Wᵀ = Σ^{-1/2} Σ Σ^{-1/2} = I
```

The covariance becomes the **identity** — the ellipse becomes a unit ball. All directions now have equal variance and are uncorrelated. **In this space, perpendicular projection is finally the right, minimal-damage operation.** And `W⁻¹ = Σ^{1/2}` un-whitens, taking us back to the real ellipse.

---

## Part 7 — Assemble LEACE: whiten → project → un-whiten

Now we just do the naive thing, but in the whitened world:

1. **Whiten.** Send `x̃ ↦ W x̃`. Round ball.
2. **Locate the concept there.** The concept directions transform too: in whitened space they span `col(M)` where `M = W Σ_XZ`. Take an orthonormal basis `U` of `col(M)` (via SVD of `M`). The number of basis vectors — `rank(M)` — is **how many dimensions the concept occupies** (≈ `k`: one for gender, several for occupation).
3. **Project the concept out.** Apply `I − U Uᵀ`: flatten the ball along exactly the concept directions.
4. **Un-whiten.** Send it back with `W⁻¹ = Σ^{1/2}`, and add the mean back.

Chaining the maps (right-to-left), the eraser is:

```
r(x) = μ + Σ^{1/2} (I − U Uᵀ) Σ^{-1/2} (x − μ)
        └────┘ └──────┬──────┘ └──────┘
      un-whiten   project off    whiten
                 concept subspace
```

Read it as a sentence: *shift to the mean-centered frame, whiten so the cloud is round, delete the concept subspace with an honest perpendicular projection, un-whiten back to the real geometry, restore the mean.*

---

## Part 8 — The two theorems you get for free

1. **Erasure is exact.** We removed precisely the whitened subspace that correlated with `Z`, so `cov(r(X), Z) = 0`. No linear probe survives. *(Proof sketch: the remaining directions are, by construction, orthogonal to `col(M)` in whitened space, which is exactly the condition that the un-whitened cross-covariance vanishes.)*

2. **Damage is minimal.** Among *all* affine maps that achieve `cov = 0`, this one minimizes the expected squared move `E‖r(x) − x‖²`. Why: orthogonal projection is the closest-point map, and we performed it in the whitened metric — which is the correct notion of distance once you account for the cloud's shape. Do it in any other basis and you'd move points farther than necessary.

This is why LEACE is "closed-form optimal": no iteration, no training loop — just eigendecomposition, SVD, and three matrix multiplies.

---

## The one-paragraph summary

A linear probe reads a concept only through the **cross-covariance** `Σ_XZ`, so erasing means zeroing it. You'd love to just **project off** the concept subspace with `I − UUᵀ`, but the embedding cloud is a stretched, tilted ellipse, and perpendicular projection in a stretched space damages more than necessary. So you first **whiten** with `Σ^{-1/2}` to turn the ellipse into a round ball, do the honest orthogonal projection there, then **un-whiten** with `Σ^{1/2}`. The result, `r(x) = μ + Σ^{1/2}(I − UUᵀ)Σ^{-1/2}(x − μ)`, is the unique minimal-damage affine map that makes the concept linearly unreadable.

---

## Key facts to remember

- **Linear readability lives in one matrix.** A linear probe can recover `Z` from `X` **iff** the cross-covariance `Σ_XZ ≠ 0`. Erasing a concept = making `Σ_XZ = 0`. Nothing else matters to a *linear* probe.
- **The concept subspace is `col(Σ_XZ)`.** Its dimension (`≈ k`) is how many directions you must remove — 1 for gender, several for occupation. "Erase a `k`-dim concept ⇒ remove ~`k` directions."
- **`U Uᵀ` projects onto a subspace; `I − U Uᵀ` projects it away.** Requires `U` to have *orthonormal* columns (`UᵀU = I`). This is the one tool doing the removal.
- **Naive projection over-damages** because embeddings are a tilted, stretched ellipse (`Σ ≠ I`). "Perpendicular" in a stretched space is not the direction of least change.
- **Whitening makes the metric honest.** `W = Σ^{-1/2}` turns the ellipse into a unit ball, so ordinary orthogonal projection becomes the minimal-damage operation. `Σ^{1/2}` undoes it.
- **The whole eraser is three moves:** `r(x) = μ + Σ^{1/2} (I − UUᵀ) Σ^{-1/2} (x − μ)` — un-whiten ∘ project-off ∘ whiten, around the mean.
- **You get two guarantees for free:** erasure is *exact* (`cov(r(X), Z) = 0`) and *minimal-damage* (smallest `E‖r(x) − x‖²`). Closed form — eigendecomposition + SVD + a few matmuls, no iteration.

---

## Appendix: two derivations worked out

Convention: each data point is a **row** of the centered matrix `X̃` (shape `N × d`); `x̃ᵢ ∈ ℝᵈ` is point `i` written as a **column** vector.

### A. Why `Σ = (1/N) · X̃ᵀ X̃`

**A1 — Definition of covariance.** For scalars `a, b`, `cov(a,b) = (1/N) Σᵢ (aᵢ − ā)(bᵢ − b̄)`. The data is **centered**, so `ā = b̄ = 0` and it reduces to the average product `(1/N) Σᵢ aᵢ bᵢ`.

**A2 — The covariance matrix is this for every coordinate pair.** `Σ` is `d × d` with

```
Σⱼₖ = (1/N) Σᵢ x̃ᵢ[j] · x̃ᵢ[k]
```

(diagonal = variances, off-diagonal = how coordinate pairs co-move).

**A3 — That double sum *is* a matrix product.** `X̃` is `N × d`, so `X̃ᵀX̃` is `d × d`, and its `(j,k)` entry is the dot product of columns `j` and `k` of `X̃`:

```
(X̃ᵀ X̃)ⱼₖ = Σᵢ X̃[i,j] · X̃[i,k] = Σᵢ x̃ᵢ[j] · x̃ᵢ[k]
```

That's exactly `Σⱼₖ` without the `1/N`. Hence `Σ = (1/N) X̃ᵀ X̃`. ∎

**A4 — Equivalent outer-product form** (used in Part B): `Σ = (1/N) Σᵢ x̃ᵢ x̃ᵢᵀ`.

> **Why the transpose seems to flip** (`X̃ᵀX̃` vs. `Σᵢ x̃ᵢ x̃ᵢᵀ`): it's the *same* matrix. By
> convention `x̃ᵢ` is a **column** (`d×1`), but inside `X̃` each sample lies down as a **row**
> (row `i` = `x̃ᵢᵀ`). So `X̃ᵀ` has the `x̃ᵢ` as its **columns**, and a matrix product is a sum over
> the shared index of (column of left)(row of right): `X̃ᵀX̃ = Σᵢ (col i of X̃ᵀ)(row i of X̃) =
> Σᵢ x̃ᵢ x̃ᵢᵀ`. The transpose only *appears* to move because the standalone vector is a column while
> its copy inside `X̃` is a row.

### B. Why `cov(WX) = W Σ Wᵀ = Σ^{-1/2} Σ Σ^{-1/2} = I`

**B1 — The transformation rule (matrix form).** Whiten the whole cloud at once. Each sample sits
in `X̃` as a row `x̃ᵢᵀ`, so its whitened version is the row `(W x̃ᵢ)ᵀ = x̃ᵢᵀ Wᵀ`; stacking all rows,
the **whitened data matrix** is

```
X̃_W = X̃ Wᵀ          (shape N × d — this is what "WX" means: W applied to every point)
```

Now apply the covariance formula from A3 to `X̃_W`:

```
cov(WX) = (1/N) X̃_Wᵀ X̃_W
        = (1/N) (X̃ Wᵀ)ᵀ (X̃ Wᵀ)
        = (1/N) W X̃ᵀ X̃ Wᵀ          (since (X̃ Wᵀ)ᵀ = W X̃ᵀ)
        = W · [ (1/N) X̃ᵀ X̃ ] · Wᵀ
        = W Σ Wᵀ
```

So covariance always transforms as `Σ ↦ W Σ Wᵀ`. (The `Σᵢ x̃ᵢ x̃ᵢᵀ` outer-product version from A4
gives the identical result — pick whichever notation you like.) ∎

**B2 — Plug in `W = Σ^{-1/2}`.** `Σ^{-1/2}` is **symmetric** (from `Σ = Q Λ Qᵀ`, define `Σ^{-1/2} = Q Λ^{-1/2} Qᵀ`; transposing leaves it unchanged since `Λ^{-1/2}` is diagonal), so `(Σ^{-1/2})ᵀ = Σ^{-1/2}` and

```
cov(WX) = Σ^{-1/2} Σ Σ^{-1/2}
```

**B3 — Split the middle `Σ` into two square roots** (`Σ = Σ^{1/2} Σ^{1/2}`):

```
Σ^{-1/2} Σ Σ^{-1/2} = (Σ^{-1/2} Σ^{1/2})(Σ^{1/2} Σ^{-1/2})
```

Each factor collapses via the eigendecomposition (`QᵀQ = I`):

```
Σ^{-1/2} Σ^{1/2} = Q Λ^{-1/2} Qᵀ Q Λ^{1/2} Qᵀ = Q Λ^{-1/2} Λ^{1/2} Qᵀ = Q Λ⁰ Qᵀ = Q Qᵀ = I
```

Therefore `cov(WX) = I · I = I`. ∎

**Intuition:** `Σ = (1/N)X̃ᵀX̃` because covariance is just "average product of coordinates," and `X̃ᵀX̃` is the machine that forms every coordinate-pair product and sums over points. `cov(WX) = WΣWᵀ` because transforming the data by `W` stretches the ellipse on *both* sides — once per vector inside the covariance. And it equals `I` because whitening divides the covariance out by its own square root, turning the tilted ellipse into a round unit ball.
