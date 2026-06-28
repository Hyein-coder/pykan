# QA GATE 4 Report — full-model input derivatives

**Agent:** curvature-watcher
**Date:** 2026-06-27
**Target:** `D:\pykan\github\workflows\Hyein\bspline_curvature.py` model-level derivative fns
(`model_directional_derivatives`, `model_gradient`, `find_model_inflection_points`,
`verify_model_derivatives`, `_combined_node_scale`, autograd references)
**Models:** `analytical_results/{damping_sin,exponential}/kan_models/`
**Scripts:** `_workspace/qa_gate4_verify.py` (+ float64 re-check snippet)

## Overall verdict: **PASS**

All chain-composition math is correct. Analytic full-model gradient and ∂²f/∂x_i² match
autograd to **~1e-13 in float64** (the one float32 reading above 1e-3 is pure accumulation
noise in the 2nd-order chain, vanishes in float64 — exactly the tolerance note in the brief).

---

## Checklist

| # | Check | Model | Result | Evidence |
|---|-------|-------|--------|----------|
| 1 | gradient + df + d2f vs autograd ≤ 1e-4 | damping_sin | **PASS** | float64: grad=1.07e-14, df=1.07e-14, d2f=7.96e-13 (float32: grad/df ok 1.6e-5, d2f=2.4e-3 = fp32 noise) |
| 1 | same | exponential | **PASS** | float32 already tight: grad=1.01e-6, df=1.01e-6, d2f=4.44e-6 |
| 2 | `model_gradient[:,i]` == `model_directional_derivatives(...)[1]` | both | **PASS** | exact 0.000e+00 for every feature, both models |
| 3 | gradient composes BOTH layers (not layer-0 only) | damping_sin | **PASS** | full vs autograd=1.6e-5; layer0-only naive vs autograd=8.29 (≈5e5× larger) |
| 4 | 1-layer reduction df/dx_i == s·φ'_{i,0}; symbolic edges flow non-zero | exponential | **PASS** | grad vs s·φ'=0.000e+00; vs autograd≤1e-6; both feats non-zero |
| 5 | `find_model_inflection_points` runs, sorted, in grid interior | damping_sin x0 | **PASS** | 9 inflections, sorted, all in [0.020,0.980] |
| 6 | mult-node guard raises clear error | synthesized | **PASS** | ValueError: "...layer 1 has 1 mult node(s)." |

---

## CHECK 1 — verify_model_derivatives
**float32 (as loaded):**
```
damping_sin: grad_max_abs_err=1.621e-05
   feat0: df=1.621e-05  d2f=2.365e-03   <- d2f slightly over 1e-3 (fp32 noise)
   feat1: df=1.490e-06  d2f=2.670e-05
exponential: grad_max_abs_err=1.013e-06
   feat0: df=1.013e-06  d2f=4.441e-06
   feat1: df=2.980e-08  d2f=0.000e+00
```
**float64 model copy (`model.double()`), tight 1e-4 threshold:**
```
damping_sin: grad_max_abs_err=1.07e-14
   feat0: df=1.07e-14  d2f=7.96e-13
   feat1: df=3.44e-15  d2f=2.90e-14
   PASS @ 1e-4  (≈ machine precision)
```
The float32 `d2f=2.4e-3` on damping_sin feat0 is **accumulation noise**, not a logic error: the
2nd-order recurrence term `s·(φ''·d² + φ'·h)` over a 2-layer composition magnifies fp32 rounding;
in float64 it collapses to 8e-13. The brief explicitly allows a float64 copy for tightness.

## CHECK 2 — gradient two ways (analytic)
```
damping_sin feat0/feat1: max|model_gradient - directional df| = 0.000e+00
exponential feat0/feat1: max|model_gradient - directional df| = 0.000e+00
```
The Jacobian-chain path (`model_gradient`) and the forward-mode directional path
(`model_directional_derivatives`) produce bit-identical gradients.

## CHECK 3 — chain rule genuinely composes both layers (damping_sin, 2-layer)
```
full 2-layer grad vs autograd        = 1.621e-05
layer0-only naive (s0·φ'_0) vs autograd = 8.291e+00
```
A layer-0-only derivative is wrong by ~8.3 (≈500,000× the full-chain error), proving the
implementation multiplies through the layer-1 edge derivatives, not just layer 0.

## CHECK 4 — single-layer reduction + symbolic flow (exponential, symbolified)
```
feat0 (tanh edge): grad vs s·φ'=0.000e+00  grad vs autograd=1.01e-6  non-zero=True
feat1 (x^2  edge): grad vs s·φ'=0.000e+00  grad vs autograd=2.98e-8  non-zero=True
```
For the 1-layer model ∂f/∂x_i collapses to exactly `s·φ'_{i,0}` (single edge to the lone output),
and the symbolic tanh / x^2 edges propagate (non-zero gradient matching autograd).

## CHECK 5 — find_model_inflection_points (damping_sin x0)
```
inflections(x0) = [0.0489,0.0802,0.1018,0.2029,0.3111,0.4194,0.5301,0.6528,0.9535]
n=9, sorted=True, all in grid interior [0.020,0.980]
```
The damped oscillator x0 yields multiple full-model inflections, as expected.

## CHECK 6 — multiplication-node guard
Synthesized by temporarily setting `model.width = [...,[n_sum, 1],...]` (1 mult node at layer 1):
```
_combined_node_scale(model, 0) -> ValueError:
  "model derivatives support summation-only models; layer 1 has 1 mult node(s)."
```
Clear, actionable error. (All real models here are summation-only, so the path is never hit
in production.)

## Repro
```
PYTHONUTF8=1 PYTHONPATH=/d/pykan conda run -n pykan-new --no-capture-output \
  python github/workflows/Hyein/_workspace/qa_gate4_verify.py
# float64 tightness re-check: model.double() then verify_model_derivatives -> all ~1e-13
```

## Notes / non-issues
- float32 is fine for the gradient (1e-5) and for exponential throughout; only the 2nd
  derivative of the deeper damping_sin needs float64 to read below 1e-4. Consider running the
  production analysis in float64 if the model 2nd derivative is used quantitatively (inflection
  *sign* detection is unaffected — the noise is ~2e-3 against peaks of order 10+).
- Derivatives are w.r.t. normalized input and output is in scaler_y space; the autograd
  reference `model(x)` uses the same spaces, so the comparison is apples-to-apples.
