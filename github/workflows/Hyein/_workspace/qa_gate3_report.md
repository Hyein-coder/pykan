# QA GATE 3 Report — symbolic-aware `bspline_curvature.py`

**Agent:** curvature-watcher
**Date:** 2026-06-27
**Target:** `D:\pykan\github\workflows\Hyein\bspline_curvature.py` (symbolic-aware additions) + §3.9 figure block in `toy_KAN_analyze.py`
**Models:** `analytical_results/{exponential,logarithm,damping_sin}/kan_models/`
**Scripts:** `_workspace/qa_gate3_verify.py`; end-to-end `python -m github.workflows.Hyein.toy_KAN_analyze <name>`

## Overall verdict: **FAIL** (one real bug)

The derivative/value math is fully correct (CHECK 1, 2, 3, 6 all PASS to ~1e-6 or exact).
But the **inflection detector (`find_inflection_points`) skips every symbolic edge** because its
guard tests only the spline mask (`act.mask[i,j]==0`), which symbolification sets to 0.
Symbolic edges can therefore never produce inflection points (CHECK 4). This is a correctness
bug on the detection path that GATE 3 is specifically meant to catch.

---

## Checklist

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | exponential — analytic vs autograd, every active edge `max_abs_err≤1e-4` OR `rel_err≤1e-3` | **PASS** | tanh: max_abs_err=4.53e-6 rel=3.3e-7; x^2: 0.0 exact |
| 2 | exponential — `edge_curves` phi matches `symbolic_fun` postacts; not flat zero | **PASS** | tanh phi∈[0.080,0.838] ptp=0.76; x^2 phi∈[0.022,0.216] ptp=0.19; max|phi-true|≤2.4e-7 |
| 3 | sympy closed-form deriv vs independent autograd double-grad (symbolic edge) | **PASS** | tanh: 2.4e-14; x^2: 5.0e-16 |
| 4 | semantics: x^2→φ''=2ca²(const, no inflection); tanh→inflection near x=-b/a | **FAIL** | x^2 OK (φ''≡0.1342=2ca², 0 inflections). **tanh: `find_inflection_points` returns [] even on a sweep containing x=-b/a=-0.231 where φ'' clearly flips sign** |
| 5 | logarithm — checks 1–2 + no NaN leak | **PASS** | log: max_abs_err=7.6e-6; phi matches to 6e-8; no non-finite leaked |
| 6 | damping_sin regression — verify still PASS AND `_symbolic_branch`==0 | **PASS** | worst max_abs_err=1.67e-5 (unchanged from GATE 1); `_symbolic_branch` all-zero |

---

## CHECK 1 — verify_against_autograd
```
exponential:
  edge(0,0) [sym:tanh]: max_abs_err=4.530e-06 scale=1.364e+01 rel_err=3.32e-07  PASS
  edge(1,0) [sym:x^2 ]: max_abs_err=0.000e+00 scale=1.342e-01 rel_err=0.00e+00  PASS
logarithm:
  edge(0,0) [sym:log ]: max_abs_err=7.629e-06 scale=7.920e+01 rel_err=9.63e-08  PASS
  edge(1,0) [sym:x   ]: max_abs_err=0.000e+00 (degenerate scale)               PASS
damping_sin (regression, all spline):
  worst max_abs_err=1.669e-05  (== GATE 1)   PASS;  _symbolic_branch all-zero = True
```
Symbolic edges are now VERIFIED (not skipped/zero), because `autograd_edge_second_derivative`
includes the symbolic value term.

## CHECK 2 — phi value (symbolic edges are real, not flat zero)
```
exponential tanh: phi range [0.0802, 0.8377] ptp=0.7575  max|phi-symbolic_fun|=2.38e-7
exponential x^2 : phi range [0.0216, 0.2159] ptp=0.1943  max|phi-symbolic_fun|=5.96e-8
logarithm   log : phi range [0.0694, 0.6294] ptp=0.5600  max|phi-symbolic_fun|=5.96e-8
logarithm   x   : phi range [-0.1150,0.2988] ptp=0.4138  max|phi-symbolic_fun|=5.96e-8
```

## CHECK 3 — sympy closed form == autograd (symbolic 2nd deriv)
```
tanh: max|closed-autograd|=2.40e-14    x^2: 5.00e-16   (float64, away from singularity)
```

## CHECK 4 — semantics  ← FAIL on tanh inflection
```
x^2 edge: a=-7.584,b=-8.916,c=0.0012 -> phi''=2ca^2=0.1342 (mean=0.1342, ptp=0), 0 inflections  OK
tanh edge: a=-2.788,b=-0.644 -> arg-zero at x=-b/a=-0.2308
   default grid range = [0.0200, 0.9800]  (x=-0.2308 is OUTSIDE -> legitimately no in-range inflection)
   BUT on an extended sweep x in [-0.6, 0.6] (which DOES contain -0.2308):
     phi'' sweeps from -14.39 to +14.39 (clear sign flip at idx where x=-0.2331)
     find_indices_sign_revert(d2, eps) directly -> [124] -> x=-0.227   (correct!)
     find_inflection_points(model,0,0, x_grid=[-0.6,0.6], j_list=[0]) -> []   (WRONG)
```

### Root cause (bug)
`find_inflection_points` (bspline_curvature.py line 473):
```python
for j in js:
    if float(act.mask[i, j].detach().cpu()) == 0.0:
        continue  # pruned edge contributes no activation   <-- skips ALL symbolic edges
    d2 = edge_second_derivative(...)
```
For a symbolified model the spline `act.mask` is 0 for the symbolic edges (verified:
`act_fun[0].mask = [[0],[0]]`, while `symbolic_fun[0].mask = [[1,1]]`). So every symbolic
edge is `continue`-skipped and never reaches `edge_second_derivative` / the detector.
The figure's per-edge inflection vlines (toy_KAN_analyze.py line 723) call the same function,
so symbolic edges also get no green vline.

### Repro
```
PYTHONUTF8=1 PYTHONPATH=/d/pykan conda run -n pykan-new --no-capture-output python - <<'PY'
import os,numpy as np; from github.workflows.Hyein.toy_KAN_sweep import KANRegressor
import github.workflows.Hyein.bspline_curvature as bc
m=KANRegressor(device='cpu'); m.load_model(os.path.join('github','workflows','Hyein','analytical_results','exponential','kan_models','exponential_best_kan_model')); m=m.model
xs=np.linspace(-0.6,0.6,400)
print("wrapper:", bc.find_inflection_points(m,0,0,x_grid=xs,j_list=[0]))   # [] (bug)
from kan.experiments.analysis import find_indices_sign_revert
_,_,d2=bc.edge_curves(m,0,0,0,xs); ep=1e-2*np.nanmax(np.abs(d2))
idx=find_indices_sign_revert(list(d2),ep); print("direct:", [round(xs[t],4) for t in idx])  # [-0.227]
PY
```

### Suggested fix (for the module author)
Replace the spline-only guard with an effective-activity check that also honors the symbolic
mask, e.g.:
```python
sym_info = symbolic_edge_info(model, l)   # {(i,j): name}
...
spline_active = float(act.mask[i, j].detach().cpu()) != 0.0
if not (spline_active or (i, j) in sym_info):
    continue
```
With this change the verify script's CHECK 4 detects tanh inflection at x≈-0.227 (== -b/a) and
still 0 for x^2.

## End-to-end runs (exit 0 for all three)
| function | exit | `⚠️ Symbolic edges detected` | curvature inflections (norm) | AGSM transitions (raw) |
|----------|------|------------------------------|------------------------------|------------------------|
| exponential | 0 | YES | `{0: [], 1: []}` | `[]` |
| logarithm   | 0 | YES | `{0: [], 1: []}` | `[]` |
| damping_sin | 0 | (none — pure spline) | `{0:[0.075,0.297,0.506,0.508,0.802,0.850,0.925,0.958], 1:[0.903]}` | `[-0.333]` |

- `{exponential,logarithm}_activation_derivatives_L0.{png,svg,eps}` regenerated (timestamp today).
  **Verified visually:** exponential PNG shows the tanh edge (φ decay 1.0→0.08, titled
  `[symbolic: tanh]`) and the x^2 edge (parabola, constant φ''≈0.13, titled `[symbolic: x^2]`) —
  real curves, NOT flat zero. The `[symbolic: …]` titles are present.
- `{name}_curvature_inflection.{png,svg,eps,csv}` produced for all three.

NOTE the empty exponential/logarithm inflection sets are a DIRECT consequence of the CHECK-4 bug:
even setting aside that this model's tanh inflection (x=-0.23) is outside the default [0.02,0.98]
sweep, the mask-skip means symbolic edges can NEVER contribute an inflection. Any symbolic edge
whose inflection falls in-range would be silently missed.

## What is correct (do not regress these)
- Symbolic value/1st/2nd derivative math: closed-form (sympy) and autograd agree to 1e-14;
  full-edge autograd reference now includes the symbolic term.
- `[out=j][in=i]` transposed indexing for symbolic funs/affine/mask is correct (validated by the
  exact value match against `symbolic_fun` postacts `[:, j, i]`).
- damping_sin (pure spline) unchanged; `_symbolic_branch` returns all-zeros when symbolic disabled.
- NaN guard (`np.nan_to_num`) holds — no non-finite leaked for logarithm's log edge.
