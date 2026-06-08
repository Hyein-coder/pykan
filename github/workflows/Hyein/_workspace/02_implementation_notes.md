# `sectional_gsa.py` — Implementation Notes

Implements Sectional GSA (AGSM, Pannier & Graf 2015). Self-contained, raw-space
only. Imports: `numpy`, `matplotlib`, `warnings`. No `torch`, no `pandas`/`scipy`
(not needed), no local imports from `kan`/`toy_KAN_sweep`.

File: `D:\pykan\github\workflows\Hyein\sectional_gsa.py`

## Function signatures (for gsa-integrator)

```python
make_batch_func(single_func) -> f(X[m,n]) -> y[m]
    # Wraps a per-row FUNCTION_ZOO callable via np.apply_along_axis.
    # Use for analytic baseline. For KAN surrogate, the caller supplies its
    # own batch func (scale -> model -> inverse-scale); shape [m,n]->[m].

make_sections(lo, hi, n_sections, mode='equal',
              data_col=None, kan_knots_raw=None) -> (edges, centers)
    # mode in {'equal','quantile','kan'}.
    # 'quantile' needs data_col (raw-space column of the investigated input).
    # 'kan' needs kan_knots_raw (raw-space knots; caller denormalizes first).
    # Returns edges [n_eff+1], centers [n_eff]; n_eff <= n_sections if
    # degenerate/duplicate edges were deduplicated (warned, never crashes).

numerical_gradient_abs(func_batch, X, feat_idx, eps=1e-5) -> |df/dx| [N]
    # Central finite difference. NON-finite entries returned as np.nan.
    # NOTE: eps is an ABSOLUTE step here. compute_gradient_agsm scales its
    # own fd step by each input's width before calling this.

compute_gradient_agsm(func_batch, bounds, feat_names, top2_idx,
                      n_sections=10, n_samples_per_section=512, seed=42,
                      section_mode='equal', data_X=None, kan_knots_raw=None,
                      fd_eps=1e-5)
    -> (section_centers, S_hat, S_a, R)   # each a dict {feat_idx: np.ndarray}
    # section_centers[l] : midpoints [n_eff]
    # S_hat[l]           : raw mean |df/dx_l| per section [n_eff] (NaN if empty)
    # S_a[l]             : normalized cross-input comparable AGSM [n_eff]
    # R[l]               : within-feature relative measure, sums to 1 [n_eff]
    # kan_knots_raw is a dict {feat_idx: knots_array} (not a single array).

find_agsm_transition_points(section_centers_i, S_a_i,
                            section_centers_j, S_a_j,
                            feat_name_i, feat_name_j) -> list[dict]
    # Each dict: {'point': float, 'from_feat': str, 'to_feat': str,
    #             'section_k': int}. Linear-interpolated S_a crossings.
    # If the two share different centers/bounds, S_a_j is np.interp-resampled
    # onto section_centers_i.

compute_global_from_sectional(S_hat, section_centers, bounds_l) -> float
    # Width-weighted reconstruction of the global mean |df/dx_l|.
    # Widths recovered from centers (interior edges = midpoints between
    # consecutive centers; outer edges = bounds_l).

plot_agsm_vs_kan(section_centers, S_a, top2_idx, feat_names,
                 kan_inflection_points, agsm_transition_points,
                 save_path, title='')
    # Saves save_path + {.png,.svg,.eps}. Step plot (steps-mid), green dashed
    # KAN inflection lines, orange dotted AGSM transition lines.
    # If both features share domain bounds -> single shared axis; else two
    # stacked subplots.
    # kan_inflection_points may be a flat list (drawn on all panels) OR a dict
    # {feat_idx: [values]} (drawn on the matching panel). agsm_transition_points
    # is the list returned by find_agsm_transition_points (uses each 'point').
    # SA_RC style dict is applied via plt.rc_context (module-level constant).
```

## Normalization (as specified)

- Per section the **raw mean** of `|df/dx_l|` is `S_hat[l][k]` (no width factor).
- Width-weighted contribution `c[l][k] = S_hat[l][k] * width_k / (hi-lo)`.
- `S_hat_global[l] = sum_k c[l][k]` (= full-domain mean `|df/dx_l|`).
- `total_denom = sum over targets l, sum_k c[l][k]`.
- `S_a[l][k] = c[l][k] / total_denom`  → `sum over all l,k of S_a == 1`,
  cross-input comparable (Eq. 14).
- `R[l][k] = c[l][k] / S_hat_global[l]` → `sum_k R[l][k] == 1` (Eq. 13).

Verified empirically on `original = sin(2*x0)+5*x1`: `sum_k R == 1` per feature,
`sum_all S_a == 1`, `compute_global_from_sectional` for x1 == 5.0 exactly,
`mean_k(S_a[1]) * N == S_global[1] == 0.797` (matches `5/(1.27+5)`).

## Deviations from the spec

1. **Sampling form.** The spec offers two equivalent MC forms. The prose says
   "prefer the full-sample form" (draw one sample over all of H, bin by section),
   while the `compute_gradient_agsm` signature describes per-section sampling
   (`n_samples_per_section`, sample `x_l ~ U(a_k,b_k)`). I implemented the
   **per-section** form per the signature: it draws `n_samples_per_section`
   points per section with `x_l` uniform in that section. The width weighting is
   then applied explicitly via `width_k/(hi-lo)`, which is mathematically
   identical to the full-sample form for the summation identity and is more
   robust against empty sections in quantile/kan modes. The Eq.13 identity
   `sum_k c == S_hat_global` holds exactly by construction (no rescale needed).

2. **No explicit Step-3 rescale.** Because each section is sampled directly and
   weighted by its own width fraction, the summation identity is exact without
   the optional `S_hat_a *= S_hat / sum(...)` numerical-guard rescale in the
   pseudocode. Omitted as unnecessary.

3. **`numerical_gradient_abs` `eps` is absolute.** The spec helper scales the
   step by the input width inside the gradient routine. I kept
   `numerical_gradient_abs` width-agnostic (absolute `eps`) and do the
   width-scaling in `compute_gradient_agsm` (`step = fd_eps * full_width`). This
   keeps the gradient helper reusable/standalone. Default `fd_eps=1e-5` →
   effective step `~1e-5 * width`, within the recommended `1e-4`..`1e-3`·width
   band when widths are O(1)–O(10); raise `fd_eps` to `1e-4` for stiffer funcs.

4. **Transition `point` is the interpolated crossing**, not the raw section
   edge `edges[k]`. The spec's Transition Point Detection section reports the
   shared edge at section granularity, but the `find_agsm_transition_points`
   signature explicitly asks for linear-interpolated crossings of `S_a_i` and
   `S_a_j`. I implemented the interpolation form (finer than edge granularity);
   `section_k` gives the left section so the caller can recover the edge if
   wanted.

## Edge-case handling

- **Empty / sparse section** (<5 valid samples): `S_hat[l][k] = np.nan`, warned;
  contributes 0 to `S_a`/`R`/global.
- **Non-finite gradient** (e.g. `log`/`convolution` near singularities): set to
  `np.nan` in `numerical_gradient_abs`, masked out of the section mean.
- **Zero global measure** (`S_hat_global[l] ~ 0`): `R[l]` and `S_a[l]` set to all
  zeros, warned (avoids 0/0).
- **Zero total denominator**: `S_a` set to zeros, warned.
- **Degenerate quantile / kan edges**: deduplicated (`np.unique`), ends snapped
  to bounds; if collapsed below 2 edges, falls back to `equal` with a warning.
  Effective `N` may be `< n_sections` (returned arrays shrink accordingly).
- **`conditional` (scalar `if`)**: handled by `make_batch_func`'s
  `np.apply_along_axis`; verified `sum_k R == 1`.

## Known limitations

- **Ranking not done here.** `top2_idx` must be supplied by the caller (rank by
  AGSM-internal global `S` or by KAN `scores_tot`). The module computes global
  measures only for the requested targets, not all inputs, so it does not
  itself produce the global ranking. If the integrator wants AGSM-based ranking,
  it should call with all input indices and rank by `compute_global_from_sectional`.
- **Resolution of transitions** is bounded by section count `N` and FD step;
  the interpolated crossing assumes piecewise-linear `S_a` between centers.
- **Space.** Raw space only. KAN knots / inflection points must be denormalized
  to raw space by the caller before being passed in (`kan_knots_raw`,
  `kan_inflection_points`). Mixing spaces is the most likely silent bug.
- **EPS transparency.** The PostScript (.eps) backend renders the semi-
  transparent overlay lines as opaque (harmless matplotlib warning); .png/.svg
  preserve alpha.
- **`plot_agsm_vs_kan` shared-axis detection** compares the min/max span of the
  two features' centers (rtol 1e-3). Features with coincidentally equal spans
  but different distributions would share an axis; acceptable for the intended
  analytic-function use where equal bounds imply equal domains.
```
