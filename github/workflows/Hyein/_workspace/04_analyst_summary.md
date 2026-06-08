# Analyst Summary: AGSM Sectional GSA vs KAN Inflection Points

Date: 2026-06-05

Comparison of sectional GSA (AGSM) dominant-feature transition points against
KAN-derived inflection points on the four analytic datasets. The only dataset
with a closed-form transition is `exponential`.

## Summary table

```
Dataset         | True transition | KAN inflection | AGSM transition | KAN err | AGSM err
----------------|-----------------|----------------|-----------------|---------|---------
exponential     |         -0.3466 |        -0.0244 |          0.3523 |  0.3222 |   0.6988
logarithm       |             N/A |           none |         -0.1910 |       - |        -
log2            |             N/A |        -0.2383 |         -0.1470 |       - |        -
rosenbrock      |             N/A | -0.5982,-0.3893,0.1726,0.5916 |  -0.2516,0.2437 |       - |        -
```

- True transition: `f(x0,x1)=exp(-2*x0)+x1` -> `|df/dx0|=|df/dx1|` ->
  `2*exp(-2*x0)=1` -> `x0 = -ln(2)/2 = -0.3466`. Only computable analytically
  for `exponential`; the others are marked `N/A`.
- KAN inflection points are from `range_split_data.pkl`
  (`inflection_points_per_input`, normalized [0.1,0.9]) denormalized to raw
  space via `scaler_X.inverse_transform`.
- AGSM transition points are the raw-space crossings of the top-2 features'
  `S_a` step curves, recomputed from `{name}_agsm_sectional.csv` (linear
  interpolation between section centers). These match the integration report.
- Errors are absolute distance to the analytical transition. For `exponential`
  the KAN inflection (err 0.322) is **closer** to truth than the AGSM crossing
  (err 0.699).

## Alignment metrics (top feature only)

| Dataset     | KAN IP (raw) | Nearest AGSM TP | abs diff | rel error | in section |
|-------------|--------------|-----------------|----------|-----------|------------|
| exponential | -0.0244      | 0.3523          | 0.3766   | 0.1884    | False      |
| logarithm   | none on x0   | -0.1910         | NaN      | NaN       | NaN        |
| log2        | -0.2383      | -0.1470         | 0.0913   | 0.0457    | False      |
| rosenbrock  | -0.5982      | -0.2516         | 0.3467   | 0.1735    | False      |
| rosenbrock  | -0.3893      | -0.2516         | 0.1378   | 0.0690    | False      |
| rosenbrock  | 0.1726       | 0.2437          | 0.0711   | 0.0356    | False      |
| rosenbrock  | 0.5916       | 0.2437          | 0.3479   | 0.1742    | False      |

Relative error = abs diff / domain width (each top feature spans ~[-1, 1], width ~2.0).
Full per-pair table: `figures_for_paper/VS_sectional_gsa_metrics.csv`.

## Interpretation

**Does KAN outperform AGSM?** On the one dataset where ground truth exists
(`exponential`), the KAN inflection point is markedly closer to the true
transition (-0.3466) than the AGSM crossing: KAN error 0.32 vs AGSM error 0.70,
roughly a 2x improvement. AGSM lands on the *wrong side* of the origin (+0.35),
because its dominant-feature crossing measures where the **integrated sensitivity
mass** of the decaying exponential drops below the constant linear term, not
where the local gradients balance. KAN's inflection detection tracks the local
curvature of the learned activation and therefore sits nearer the analytical
gradient-balance point.

**Is the transition well-located?** Partially. Both methods identify a single
transition region for `exponential` and a near-coincident transition for `log2`
(KAN -0.238, AGSM -0.147; abs diff 0.09, ~4.6% of domain) — good qualitative
agreement. The `kan_in_agsm_section` flag is `False` everywhere, meaning no KAN
inflection falls inside the exact equal-width AGSM section that contains the
crossing; the two methods agree on *region* but not on the same coarse bin
(13 sections -> bin half-width ~0.077, smaller than the typical 0.09-0.38 gap).

**Per-dataset notes:**
- `exponential`: KAN closer to truth; AGSM offset to the positive side. The
  cleanest demonstration of KAN's advantage.
- `logarithm`: No KAN inflection on the dominant feature (x0); KAN instead placed
  both inflections on x1. AGSM still reports an x0-vs-x1 crossing at -0.19. This
  is the weakest alignment case (no direct KAN counterpart on the top feature).
- `log2`: Tightest agreement (rel err 4.6%). Both methods locate the
  log/linear dominance switch slightly left of origin.
- `rosenbrock`: Symmetric structure produces 4 KAN inflections and 2 AGSM
  crossings (near +/-0.25). The two inner KAN inflections (-0.389, +0.173) pair
  well with the AGSM crossings (rel err 7% and 3.6%); the two outer KAN
  inflections capture additional curvature AGSM does not flag.

## Recommendations for the paper narrative

1. **Lead with `exponential`** as the headline result: it is the only case with
   a closed-form transition, and KAN beats AGSM by ~2x on absolute error. Show
   the analytical line, KAN inflection, and AGSM crossing on one panel (already
   in `VS_sectional_gsa_Analytic.png`, top-left).
2. **Frame AGSM and KAN as complementary, not redundant.** AGSM measures
   section-integrated sensitivity dominance; KAN inflections measure local
   activation curvature. The gap (especially the sign flip in `exponential`) is
   a feature to discuss, not an error to hide: integrated-mass crossings lag the
   local gradient-balance point for rapidly decaying functions.
3. **Use `log2` as the agreement case** to show the two methods converge when the
   sensitivity profiles are smooth and monotone (rel err < 5%).
4. **Use `rosenbrock` to argue KAN richness**: KAN recovers more transition
   structure (4 vs 2) consistent with the quartic/parabolic geometry.
5. **Caveat `logarithm`**: KAN placed inflections on the non-dominant feature, so
   present it as a limitation / edge case for inflection-on-top-feature
   alignment, or re-examine the inflection detector's feature selection there.
6. Report the relative-error metric (abs diff / domain width) rather than raw
   distances, so the four datasets are comparable on a common scale.

## Output files

- `figures_for_paper/VS_sectional_gsa_metrics.csv` — per-pair alignment metrics.
- `figures_for_paper/VS_sectional_gsa_Analytic.{png,svg,eps}` — 2x2 comparison.
- Generator: `_workspace/_gen_sectional_gsa.py`.
