# QA GATE 1 Report — `bspline_curvature.py`

**Agent:** curvature-watcher
**Date:** 2026-06-27
**Module under test:** `D:\pykan\github\workflows\Hyein\bspline_curvature.py`
**Model used:** `analytical_results/damping_sin/kan_models/damping_sin_best_kan_model`
(layer 0: in=2, out=2, k=3, grid M=17, coef nc=13)
**Scripts:** `_workspace/qa_gate1_verify.py` (+ ad-hoc `check_fd.py`, `check_const.py`, `check_linear.py`, `check_space.py`)

## Overall verdict: **PASS**

The analytical 2nd derivative matches two independent ground truths to ~1e-5:
- autograd double-grad: worst `max_abs_err = 1.67e-05`
- independent float64 central finite difference of the full activation: worst abs diff `1.27e-06`

---

## Checklist

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | `verify_against_autograd(model,0)` — every edge `max_abs_err≤1e-4` OR `rel_err≤1e-3` | **PASS** | worst max_abs_err=1.67e-5, worst rel_err=3.15e-6 (all 4 edges) |
| 2 | Independent central FD of FULL edge activation matches `edge_second_derivative` | **PASS** | worst abs diff=1.27e-6 (float64); see note A |
| 3a | order reduction k→k−1→k−2, factor `order`, knot offset `grid[a+order]-grid[a]`, zero-pad c_{-1}/c_{nc} | **PASS** | proven empirically by checks 1&2; code review confirms `_deriv_coef` |
| 3b | `B_batch(x,grid,k-2)` basis count == `d2` length | **PASS** | k2=1, d2 last-dim=15 == B_batch last-dim=15 == M−k+1=15 |
| 3c | `scale_base*base''` term present; SiLU closed form correct | **PASS** | present in `activation_second_derivative`; SiLU form validated by autograd match (base_fun is SiLU) |
| 3d | `mask` and `scale_sp` applied | **PASS** | `y = mask * (scale_base*base'' + scale_sp*spline'')` |
| 3e | `k<2` raises | **PASS** | `ValueError: ...requires spline order k >= 2; got 1.` |
| 3f | returned inflection x-values in NORMALIZED space | **PASS** | inflections ∈ [0.075, 0.958] ⊂ interior grid range [0.02, 0.98] |
| 4 | near-linear/const edge → ~0 curvature, few/no inflections; `find_inflection_points` runs, sorted | **PASS** | affine edge → 0 inflections; FIP runs, returns sorted in-range values; see note B |

### Per-edge results — CHECK 1 (autograd)
```
edge(0,0): max_abs_err=1.669e-05 scale=5.293e+00 rel_err=3.153e-06  PASS
edge(0,1): max_abs_err=2.861e-06 scale=6.947e+00 rel_err=4.118e-07  PASS
edge(1,0): max_abs_err=2.783e-06 scale=4.721e+00 rel_err=5.894e-07  PASS
edge(1,1): max_abs_err=2.176e-06 scale=3.010e+00 rel_err=7.227e-07  PASS
```

### Per-edge results — CHECK 2 (float64 finite difference of full activation)
```
edge(0,0): max|fd-ana|=1.273e-06 rel=2.58e-07  PASS
edge(0,1): max|fd-ana|=1.159e-06 rel=1.84e-07  PASS
edge(1,0): max|fd-ana|=6.264e-08 rel=4.11e-06  PASS
edge(1,1): max|fd-ana|=9.508e-08 rel=1.91e-05  PASS
```

### Sample 2nd-derivative values (edge(0,0), first 3 sweep pts): `[-4.934, -4.561, -4.188]`

### CHECK 4 — inflections found
```
input 0: 8 inflections, sorted, in-range: [0.075,0.297,0.506,0.508,0.802,0.85,0.925,0.958]
input 1: 1 inflection: [0.903]
```

---

## Notes (false alarms ruled out)

**Note A — CHECK2 float32 false-fail.** A first FD pass with `h=1e-3*span` on the
**float32** model gave abs diff ~9.6e-2 (rel 1.9e-2), tripping a 1e-2 threshold.
This was pure FD rounding noise: central 2nd-diff rounding error ~ ε·f/h². Upcasting
the model tensors to float64 and using `h=1e-4*span` drops the disagreement to 1.3e-6.
**Module is correct; the original test harness was under-resolved.**

**Note B — synthetic "constant"/"linear" spline false-fail.** Setting `coef=ones`
(or Greville-affine coefs) does NOT produce a constant/affine function on the pykan
extended grid: `B_batch` ignores its `extend` arg and the padded boundary knots are
not clamped, so boundary basis functions do not form a partition of unity. The
constant-coef spline actually ranges 0.83→1.0 and has a genuine non-zero curvature
(peak 156) — which the module reproduces against autograd to 5.6e-5. So the large
peak is **correct**, and the CHECK4 assertion `max|spline''|<1e-6` was the flawed part,
not the module. The meaningful edge-case result holds: a near-linear edge yields
**0 spurious inflections** via the relative-eps detector.

## Cross-checks of the recurrence indexing (CHECK 3a)
`_deriv_coef` implements `d_a = order·(c_a − c_{a−1})/(t_{a+order} − t_a)` with
`cpad = [0, coef, 0]` (boundary c_{-1}=c_{nc}=0), `diff = cpad[1:] − cpad[:-1]`,
`denom = grid[a+order] − grid[a]` for `a=0..nc`, and degenerate (`|denom|<1e-12`)
knots zeroed. Output length nc+1 per step; two steps give nc+2 = M−k+1, matching the
order-(k−2) basis. The exact autograd agreement (1e-5) confirms the offset, factor,
and padding are all correct — any off-by-one in `a+order` or the padding would have
produced large per-edge errors.
```
```
