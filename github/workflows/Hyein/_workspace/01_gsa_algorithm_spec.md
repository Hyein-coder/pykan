# Sectional GSA (AGSM) Algorithm Specification

> Specification for the `gsa-developer` agent. This is the single source of truth for
> implementing the **Argument-value Sectional Global Sensitivity Measure (AGSM)** as a
> baseline alongside the existing KAN attribution analysis.

## Source

**Paper:** Pannier, S. & Graf, W. (2015). *"Sectional global sensitivity measures."*
Reliability Engineering and System Safety, **134**, 110-117.
DOI: `10.1016/j.ress.2014.09.009`. Institute for Structural Analysis, TU Dresden.

- PDF: `D:\ObsidianVault\KAN\VibeCoding\References\KAN\RESS_2015_Pannier_Sectional global sensitivity measures.pdf`
- User notes: `D:\ObsidianVault\KAN\VibeCoding\Projects\KAN\Sectional GSA as a baseline.md`

The paper defines two families of sectional measures:
- **AGSM** — subdivides the **input space** (argument values). *This is what we implement.*
- **FGSM** — subdivides the **codomain** (function/result values). *Out of scope.* It is
  noted here only so the developer does not confuse the two: FGSM partitions output ranges
  and is a quantitative measure; AGSM partitions one input's domain and is qualitative.

The underlying global sensitivity measure used throughout is the **derivative-based GSM**
using the **absolute** partial derivative (Pannier Eq. 4). The absolute value is essential:
the paper explicitly shows (function `a(x,y)=x+y^2`) that the plain signed derivative
expectation can be zero even when the input clearly matters, so we must integrate
`|∂f/∂x_i|`, never the signed derivative.

---

## Mathematical Definition

### Global sensitivity measure (Pannier 2015, Eqs. 3-4)

A differentiable function `f : H ⊂ ℝⁿ → B ⊂ ℝ` is given. The unnormalized derivative-based
measure for input `x_i` over the full domain `H` is:

$$
\hat{S}_i \;=\; G_i \;=\; \frac{1}{V(H)} \int_H \left| \frac{\partial f}{\partial x_i}(x) \right| \, dx
$$

where:
- `H` is the domain of definition of `f` (and of `|∂f/∂x_i|`).
- `V(H) = ∫_H dx` is the volume of `H` (product of the side lengths for a hyperrectangle).
- If `G_i = 0`, then `x_i` has no impact on `f`.

The **normalized** global sensitivity measure (a single value in `[0,1]`):

$$
S_i \;=\; \frac{\hat{S}_i}{\sum_{j=1}^{n} \hat{S}_j}
$$

### AGSM Sectional measure (Pannier 2015, Eqs. 8-14)

Only the domain `A_i = Q_i(H)` of the **investigated input `x_i`** is segmented into `N_i`
disjoint intervals `A_{i,[k]}` (`k = 1,…,N_i`) that satisfy:

$$
A_{i,[k]} \cap A_{i,[j]} = \varnothing \;\; (k \neq j), \qquad
\bigcup_{k=1}^{N_i} A_{i,[k]} = A_i .
$$

All **other** input domains `A_j` (`j ≠ i`) remain unchanged. This defines the **subinput
space** `C_{i,[k]} ⊂ ℝⁿ` (see "Subdomain Construction" below).

The unnormalized sectional measure on subinput space `C_{i,[k]}`:

$$
\hat{S}^{a}_{i,[k]} \;=\; \frac{1}{V(H)} \int_{C_{i,[k]}} \left| \frac{\partial f}{\partial x_i}(x) \right| dx .
$$

> **Critical normalization fact (Eq. 13):** the section measures sum to the global measure:
> $\sum_{k=1}^{N_i} \hat{S}^{a}_{i,[k]} = \hat{S}_i$. This holds **only because** the
> normalizing volume is `V(H)` (the full domain) in *both* `\hat{S}_i` and
> `\hat{S}^{a}_{i,[k]}` — do **not** normalize a section by its own sub-volume here.

The **within-input** relative section measure (compares sections of the *same* input
`x_i`; values lie in `[0,1]`, summing to 1 over `k`):

$$
R_{i,[k]} \;=\; \frac{\hat{S}^{a}_{i,[k]}}{\sum_{l=1}^{N_i}\hat{S}^{a}_{i,[l]}} \;=\; \frac{\hat{S}^{a}_{i,[k]}}{\hat{S}_i}.
$$

The **cross-input comparable** AGSM (Eq. 14) scales `R_{i,[k]}` by the global `S_i` so that
sections of *different* inputs can be compared on one chart:

$$
S^{a}_{i,[k]} \;=\; R_{i,[k]} \cdot S_i \;=\; \frac{\hat{S}^{a}_{i,[k]}}{\hat{S}_i}\cdot S_i \;=\; \frac{\hat{S}^{a}_{i,[k]}}{\sum_{j=1}^{n}\hat{S}_j} \;=\; \frac{\hat{S}^{a}_{i,[k]}}{\sum_{j=1}^{n}\sum_{l=1}^{N_j}\hat{S}^{a}_{j,[l]}} .
$$

Value-range notes from the paper:
- `S_i ∈ [0,1]`.
- `R_{i,[k]} ∈ [0, N_i]` (a single section can carry up to `N_i`× the average weight).
- `S^a_{i,[k]} ∈ [0, N_i]` as well. AGSM is therefore a **qualitative** functional-interrelation
  descriptor with a quantitative weighting — **not** a global quantitative measure.
- Sanity check to assert in code: `mean_k(S^a_{i,[k]}) == S_i` and `sum_k(R_{i,[k]}) == 1`.

### Monte Carlo approximation (Pannier 2015, Eqs. 21-22)

The integral over `C_{i,[k]}` is approximated by sampling. The paper draws a single
quasi-random sequence `x_1,…,x_{nsim} ∈ H` and bins points by which section `A_{i,[k]}`
their `i`-th coordinate falls into; `n_{sim,k}` is the count in section `k`, with
`Σ_k n_{sim,k} = n_{sim}`.

$$
\hat{S}^{a}_{i,[k]} \;=\; \frac{1}{V(C_{i,[k]})}\int_{C_{i,[k]}}\left|\frac{\partial f}{\partial x_i}(x)\right|dx \;\approx\; \frac{1}{n_{sim,k}} \sum_{l=1}^{n_{sim,k}} \left| \frac{\partial f}{\partial x_i}(x_l) \right|.
$$

> **Subtlety to resolve carefully.** Eq. 21 writes the *integral mean over `C_{i,[k]}`*
> (normalized by `V(C_{i,[k]})`), but Eqs. 13-14 require the *contribution* normalized by
> `V(H)`. The two differ by the volume fraction `V(C_{i,[k]})/V(H)`. With **equal-width
> sections of one input** and the other inputs spanning their full domain,
> `V(C_{i,[k]})/V(H) = 1/N_i` for every `k`. Therefore the version of `\hat{S}^a_{i,[k]}`
> that is consistent with `Σ_k \hat{S}^a_{i,[k]} = \hat{S}_i` is:
>
> $$
> \hat{S}^{a}_{i,[k]} \;=\; \frac{V(C_{i,[k]})}{V(H)} \cdot \big(\text{mean of } |\partial f/\partial x_i| \text{ over points in section } k\big).
> $$
>
> For **equal-width** sections this is `(1/N_i) · mean_k`. For **quantile** or **KAN-grid**
> sections the widths differ, so you **must** use the explicit volume-fraction factor
> `width_k / total_width` (the other dimensions cancel since they are identical across
> sections). Implementing it as `(width_k / A_i_width) * mean_of_abs_grad_in_section_k`
> is correct for all three section modes and is the recommended formulation.
>
> Equivalently and more robustly with finite samples, estimate `\hat{S}^a_{i,[k]}` as a
> Riemann/empirical mean over the **full** sample:
> `\hat{S}^a_{i,[k]} ≈ (1/n_sim) · Σ_{l : x_l ∈ section k} |∂f/∂x_i(x_l)|`
> **when** points are drawn uniformly over `A_i` (each section's expected count is
> proportional to its width). This automatically embeds the width weighting and guarantees
> `Σ_k \hat{S}^a_{i,[k]} = (1/n_sim) Σ_l |∂f/∂x_i(x_l)| = \hat{S}_i`. **Prefer this
> full-sample form** — it is the cleanest way to keep the summation identity exact.

---

## Subdomain Construction (`C_{i,[k]}`)

For the investigated input `x_i` and section index `k`:

1. Take the full bounds `[a_i, b_i] = A_i = Q_i(H)`.
2. Partition `[a_i, b_i]` into `N` contiguous, non-overlapping intervals
   `A_{i,[1]},…,A_{i,[N]}` (see section modes below). These tile `A_i` exactly.
3. `C_{i,[k]}` is the box where:
   - `x_i` is **restricted** to `A_{i,[k]}` (the `k`-th section), and
   - **every other** input `x_j` (`j ≠ i`) spans its **full** domain `A_j = Q_j(H)`.

In words: *one coordinate is sliced into a band; all other coordinates keep their full
range.* Sampling `C_{i,[k]}` = draw `x_i` uniformly within `A_{i,[k]}` and draw every other
`x_j` uniformly within its full `[a_j, b_j]`.

> This is the defining difference from "local effects" / full grid: AGSM never simultaneously
> segments multiple inputs, so it produces `n·N` values (linear), not `∏ N_i` (exponential).

---

## Implementation Algorithm (pseudocode)

```text
INPUT:
  f            : callable, batch numpy func  X[m,n] -> y[m]   (see "Function wrapping")
  bounds       : list of [a_j, b_j] for j = 1..n
  targets      : list of input indices to analyze (default: top-2 by global S_i)
  N            : number of sections per input (default: KAN grid count)
  mode         : "equal" | "quantile" | "kan_grid"
  n_sim        : total MC samples per input (e.g. 20000)
  X_data       : optional sample matrix [M,n] in raw space (needed for "quantile")
  kan_knots    : optional dict {input_idx: knot array in raw space} (needed for "kan_grid")
  h            : finite-difference step for ∂f/∂x_i (relative to each input's width)

# ---- STEP 1: global unnormalized measures S_hat[j] for ALL inputs j=1..n ----
for j in 1..n:
    Xs = uniform_sample(bounds, n_sim)          # full domain H
    g  = abs_partial_derivative(f, Xs, j, h)    # |∂f/∂x_j| at each sample, shape [n_sim]
    S_hat[j] = mean(g)                          # = (1/V(H)) ∫_H |∂f/∂x_j| dx  (MC)
S = S_hat / sum(S_hat)                           # normalized global S_i, Eq.3

# ---- STEP 2: section edges for each target input ----
for i in targets:
    edges[i] = build_section_edges(bounds[i], N, mode, X_data[:,i], kan_knots[i])
    # edges has length N+1, strictly increasing, edges[0]=a_i, edges[-1]=b_i

# ---- STEP 3: sectional unnormalized measures S_hat_a[i][k] ----
for i in targets:
    # draw ONE sample over full H, then bin by x_i's section (full-sample form)
    Xs = uniform_sample(bounds, n_sim)           # x_i uniform over [a_i,b_i], others full
    g  = abs_partial_derivative(f, Xs, i, h)     # [n_sim]
    for k in 1..N:
        in_k = (Xs[:,i] > edges[k-1]) & (Xs[:,i] <= edges[k])   # half-open; see edge cases
        # contribution form: keeps Σ_k S_hat_a == S_hat[i]
        S_hat_a[i][k] = sum( g[in_k] ) / n_sim
    # numerical guard: rescale so Σ_k S_hat_a[i][k] == S_hat[i] exactly
    S_hat_a[i] *= S_hat[i] / sum(S_hat_a[i])     # (only if sum>0; see edge cases)

# ---- STEP 4: derived measures ----
for i in targets:
    R[i][k]   = S_hat_a[i][k] / S_hat[i]                 # Eq.13, within-input, sums to 1
    Sa[i][k]  = R[i][k] * S[i]                           # Eq.14, cross-input comparable
    # equivalently Sa[i][k] = S_hat_a[i][k] / sum_j(S_hat[j])

OUTPUT:
  S_global  = S[i]            per target input
  edges     = edges[i]        section boundaries (raw space)
  centers   = midpoints       for plotting
  R         = R[i][k]         within-input relative
  Sa        = Sa[i][k]        AGSM cross-input comparable
  dominant  = argmax_i Sa[i][k]   per section k (for transition detection)
```

### `build_section_edges(bound, N, mode, col_data, knots)`

- **"equal"** (paper default, "intervals of same length", Eq. ~21 text):
  `np.linspace(a_i, b_i, N+1)`.
- **"quantile"** (data-driven): `np.quantile(col_data, np.linspace(0,1,N+1))`, then clip to
  `[a_i,b_i]` and deduplicate. With ties this can yield `<N` usable sections — handle as an
  edge case.
- **"kan_grid"** (from KAN model): use the KAN spline knot positions for that input,
  converted to raw space, as edges. May not be evenly spaced; first/last edges must be
  snapped to `[a_i, b_i]` to fully tile `A_i`. See "KAN grid count extraction" for how
  `N` and the knots are obtained.

### `abs_partial_derivative(f, X, i, h)`

Central finite difference along axis `i` (the FUNCTION_ZOO funcs are analytic but not
guaranteed differentiable symbolically here, so use numerics):

```python
step = h * (bounds[i][1] - bounds[i][0])      # scale step to the input's range
Xp = X.copy(); Xp[:, i] += step
Xm = X.copy(); Xm[:, i] -= step
grad = (f(Xp) - f(Xm)) / (2 * step)
return np.abs(grad)
```

Use `h ≈ 1e-4` to `1e-3`. For the KAN surrogate, `∂f/∂x_i` may instead be obtained via
autograd on the torch model (more accurate) — both are acceptable; keep finite-difference
as the default since it works uniformly for analytic funcs and the surrogate.

---

## Transition Point Detection

Goal: find input values where the **dominant feature changes** between consecutive sections.
This is the AGSM analogue of KAN inflection points and is the headline comparison artifact.

Algorithm (operate on the comparable AGSM `Sa`, since only `Sa` is cross-input comparable):

```text
For each section k = 1..N:
    dominant[k] = argmax over analyzed inputs i of  Sa[i][k]
For k = 1..N-1:
    if dominant[k] != dominant[k+1]:
        transition value ≈ shared edge between section k and k+1  =  edges[k]
        record (transition_x = edges[k],
                from_feature = dominant[k],
                to_feature   = dominant[k+1])
```

Notes:
- A "transition point" is reported in the **raw input space** at the section boundary
  `edges[k]`. (Resolution is limited to section granularity `N`; finer localization is not
  defined by AGSM.)
- Optionally also detect transitions **within a single input** in `R[i][k]` — a strong local
  peak / sign of where that input's influence concentrates — but the cross-input dominant
  switch above is the primary signal to compare against KAN inflection points.
- These transition `x`-values are what should be overlaid against
  `inflection_points_per_input` from the KAN analysis (see Integration).

---

## User Implementation Guidelines (from notes)

From `Sectional GSA as a baseline.md`, the "AGSM as a baseline" guidelines:

- **Top-2 rank features.** Build subdomains only for the two highest-ranked inputs
  `(x_i, x_j)`. Ranking is by the global measure `S` (Step 1) — or, for parity with the KAN
  pipeline, by the KAN global attribution score `scores_tot` (see Integration). Make the
  ranking source a parameter; default to the AGSM-internal `S` so the baseline is
  self-contained, but allow passing the KAN score order.
- **`N` sections = KAN grid count.** The number of sections `N` equals the number of grid
  intervals in the trained KAN model (see "KAN grid count extraction"). Same `N` for both
  analyzed inputs (`N_i = N_j`), matching the paper's recommendation.
- **Section modes (all three required):**
  1. `equal` — equal-distance partition of `[a_i,b_i]`.
  2. `quantile` — quantiles of the data for that input.
  3. `kan_grid` — use the KAN model's spline grid knots as edges.
- **Which features to compute:** compute `S^a_{l,[k]}` for `l ∈ {i, j}` (both top-2 inputs)
  and `k ∈ {1,…,N}`.

---

## Integration with Existing KAN Code

Reference files:
- `D:\pykan\github\workflows\Hyein\toy_KAN_analyze.py` — KAN loading, inflection points, grids.
- `D:\pykan\github\workflows\Hyein\toy_KAN_sweep.py` — `KANRegressor`, `FUNCTION_ZOO`.
- `D:\pykan\github\workflows\Hyein\toy_analytic_SHAP_Sobol.py` — `run_analysis_suite` pattern
  (Sobol/SHAP) to mirror for outputs (problem dict with `num_vars`/`names`/`bounds`,
  `model_func(X)` batch call, CSV + plot per measure).

### KAN grid count extraction

The trained model is a `MultKAN` (`model = model_wrapper.model`). For layer `l = 0`,
`act = model.act_fun[0]`. The actual interior knot positions for input `i` (already used in
`toy_KAN_analyze.py`) are:

```python
knots = act.grid[i, model.k - 1 : -2].cpu().detach().numpy()   # raw-space-aligned knots
```

- `model.k` is the spline order; the slice drops the `k-1` boundary knots on each side.
- `N = number of grid intervals = len(knots) - 1` (sections between consecutive knots).
  This is the value to use as the default `N` for sectioning. Confirm against the model's
  configured grid count; both top-2 inputs should use the same `N`.
- For `mode="kan_grid"`, use these `knots` directly as section edges (snap ends to the input
  bounds). Note these knots live in the **normalized** input space (`X` is scaled by
  `scaler_X` before the KAN; analysis there works in the `[~0.1, 0.9]` normalized band), so
  to compare against raw-space transition points you must inverse-transform via `scaler_X`
  (consistent with the contour code at lines ~414-416 which denormalizes inflection points).

### KAN inflection point extraction

`inflection_points_per_input` is already computed in `toy_KAN_analyze.py` (Section 3,
lines ~211-289) and saved into the results dict (line ~547):

- Per input `i`, it analyzes the layer-0 spline coefficients `act.coef`, computes 1st/2nd
  finite differences of the coefficients (`slope`, `slope_2nd`), finds sign reversals via
  `find_indices_sign_revert` (from `kan.experiments.analysis`), and maps the reverting knot
  indices back to knot positions `knot_points_actual = act.grid[i, k-1:-2]`.
- Result: `inflection_points_per_input[i] = sorted(set(inflection knot values))` — a list of
  input values (normalized space) where the activation's curvature flips.
- **These are the KAN-side comparison targets** for the AGSM transition points from the
  "Transition Point Detection" section. Align both in the same space (denormalize the KAN
  knots to raw, or normalize the AGSM edges) before comparing/overlaying.

### Function wrapping

`FUNCTION_ZOO[name]["func"]` is written for a **single sample indexed by position**, e.g.
`lambda x: np.sin(2*x[0]) + 5*x[1]`. It expects `x[0], x[1], …`, i.e. one row at a time.
The existing code batches it with `np.apply_along_axis(target_func, 1, X)` (see
`toy_KAN_analyze.py` line 125 and `toy_KAN_sweep.py`). Provide a batch wrapper so AGSM can
call `f(X[m,n]) -> y[m]`:

```python
def make_batch_func(single_func):
    def f(X):                      # X: [m, n] raw-space numpy
        X = np.atleast_2d(X)
        return np.apply_along_axis(single_func, 1, X).astype(float).ravel()
    return f
```

- Use `make_batch_func(config["func"])` for the **analytic ground-truth** AGSM baseline
  (operating in raw input space with `config["bounds"]`).
- For the **KAN surrogate** AGSM, wrap the model instead: scale `X` with `scaler_X`, run
  `model(torch.tensor(...))`, inverse-transform with `scaler_y`, return `[m]` numpy — mirror
  the forward/scaling logic at `toy_KAN_analyze.py` lines ~133-151. The same SHAP/Sobol
  `model_func` convention in `toy_analytic_SHAP_Sobol.py` (`Y = model_func(X)`) applies.
- Note the `conditional` function in the ZOO uses a Python `if` on a scalar (`x[0] < 0`), so
  it only works per-row — `apply_along_axis` (not vectorized numpy) is required; do not try
  to vectorize the wrapper.

---

## Numerical Considerations

- **Absolute value, always.** Integrate `|∂f/∂x_i|`. Never let signs cancel (paper Eq. 4
  rationale). The finite-difference helper already applies `np.abs`.
- **Empty sections.** A section `k` may receive zero samples (sparse data in `quantile`
  mode, or very fine `N`). Set `S_hat_a[i][k] = 0`, `R = 0`, `Sa = 0`, and flag it. With the
  full-sample MC form, empty sections naturally contribute 0; just guard the rescale
  division.
- **Degenerate / duplicate quantile edges.** Ties in `col_data` collapse quantile edges;
  deduplicate and reduce effective `N`, or fall back to `equal`. Warn, do not crash.
- **Zero global measure.** If `S_hat[i] == 0` (input has no effect, e.g. Ishigami `x2`'s
  zero-effect partner, or `x1` in the polynomial example), then `R_{i,[k]}` is `0/0`. Define
  `R = 0` and `Sa = 0` for that input and skip its transition analysis. Guard all divisions
  with `if denom > eps`.
- **Summation identity check.** After Step 3, assert
  `abs(sum_k(S_hat_a[i]) - S_hat[i]) < tol`; the optional rescale in Step 3 enforces it. This
  is the cheapest correctness test and should be logged per input.
- **Half-open binning.** Use `(x > edge[k-1]) & (x <= edge[k])` (matches the KAN interval
  binning in `toy_KAN_analyze.py` lines ~326-327). Ensure the very first edge includes its
  left endpoint — either use `>=` for the first bin or nudge `edges[0]` slightly below
  `a_i`. Verify every sample lands in exactly one bin (no dropped points).
- **NaN / inf from `f`.** Some ZOO funcs blow up near singularities (`convolution`:
  `x1+1.08` denominator; `logarithm`/`log2`: `log(20*(x0+1.2))` near the lower bound; large
  Rosenbrock values). Mask `~np.isfinite(grad)` before averaging; if a whole section is
  non-finite, treat as empty. Optionally clip extreme gradients (robust mean) but document if
  you do.
- **MC variance / seeding.** Use a fixed seed (the repo uses `seed=42`, `random_state=42`)
  for reproducibility. Use enough samples (`n_sim ≥ 1e4`, ideally a Sobol/quasi-random
  sequence as the paper recommends) so section means are stable; report or sanity-check that
  doubling `n_sim` does not materially change `Sa`.
- **Step-size sensitivity.** Finite-difference `h` too large smooths real curvature; too
  small amplifies float noise. Scale `h` to each input's width (as in the helper) and keep
  `h ~ 1e-4`·width. For non-smooth funcs (`conditional`, `rosenbrock` corners) expect a
  spike at the kink — that is genuine signal, not error.
- **Space consistency.** Decide once whether AGSM runs in raw input space (analytic
  baseline) or normalized KAN space (surrogate), and convert KAN knots / inflection points
  to the matching space before any comparison. Mixing spaces is the most likely silent bug.
