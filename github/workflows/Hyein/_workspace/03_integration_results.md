# Sectional GSA (AGSM) Integration Results

Integration of `sectional_gsa.py` into the KAN analysis pipeline
(`toy_KAN_analyze.py`, new **Section 3.8**) and run on the analytic datasets.

Date: 2026-06-05

## What was integrated

A new try/except-wrapped **Section 3.8 "Sectional GSA (AGSM) vs KAN inflection
points"** was inserted into `toy_KAN_analyze.py`, immediately after Section 3.7
(Contour Analysis) and before Section 4 (Range-Based Attribution Scoring). It:

1. Derives `n_sections` from the KAN grid (`len(act.grid[0]) - model.k - 1`).
2. Picks the top-2 features by KAN global attribution (`scores_tot`).
3. Wraps the analytic `target_func` via `make_batch_func` and runs
   `compute_gradient_agsm` (raw space, `section_mode='equal'`, 512 samples/section).
4. Denormalizes KAN inflection points from normalized [0.1,0.9] -> raw space.
5. Finds AGSM transition points (dominant-feature crossings).
6. Saves `{name}_agsm_sectional.csv` and `{name}_agsm_vs_kan.{png,svg,eps}`.

The whole block is wrapped in try/except so an AGSM failure never breaks the
existing KAN analysis (verified — the pipeline completed Sections 4-6 even when
AGSM first errored during debugging).

## Datasets run — all 4 succeeded

| Dataset     | Top-2 features (by KAN score)        | AGSM transition points (raw)            | KAN inflection points (raw)                                                                 |
|-------------|--------------------------------------|-----------------------------------------|---------------------------------------------------------------------------------------------|
| exponential | Exponent (x0), Linear (x1)           | [0.3523]                                | x0: [-0.0244];  x1: []                                                                       |
| logarithm   | Log (x0), Linear (x1)                | [-0.1910]                               | x0: [];  x1: [0.0168, 0.6077]                                                               |
| log2        | Log (x0), Linear (x1)                | [-0.1470]                               | x0: [-0.2383];  x1: []                                                                      |
| rosenbrock  | Quadratic (x0), Parabolic (x1)       | [-0.2516, 0.2437]                       | x0: [-0.5982, -0.3893, 0.1726, 0.5916];  x1: [-0.2008, 0.3980]                              |

Notes:
- KAN inflection points are reported per top-2 feature index, denormalized to raw
  space and filtered to the valid normalized window (0.05 < ip < 0.95) before
  denormalization.
- `exponential` sanity check: `Linear (x1)` yields `S_hat == 1.0` in every section
  (df/dx1 of the linear term is constant), and `Exponent (x0)` shows the expected
  monotonic decay of `|d/dx exp|`. The single AGSM transition (~0.35) is exactly
  where the exponent's contribution falls below the constant linear term.
- `rosenbrock` produces two AGSM transitions (symmetric-ish around 0), consistent
  with the quadratic/parabolic structure, and the richest set of KAN inflections.

## Output files (per dataset, in `analytical_results/{name}/kan_models/`)

- `{name}_agsm_sectional.csv` — columns: Feature, Feature_idx, Section_k,
  Section_center, S_hat, S_a, R.
- `{name}_agsm_vs_kan.png` / `.svg` / `.eps` — step plot of S_a per feature with
  green-dashed KAN inflection lines and orange-dotted AGSM transition lines.

All 16 files (4 datasets x [csv, png, svg, eps]) were verified present.

## Issues encountered and resolved

1. **Python environment.** Base conda env lacks `torch`. Used
   `C:\Users\user\miniconda3\envs\pykan-new\python.exe` (torch 2.2.2+cpu, plus
   SALib/sklearn/yaml/joblib). Several other `pykan-*` envs also work.

2. **Console encoding.** The script prints emoji; the default `cp949` Windows
   console codec raised `UnicodeEncodeError`. Ran with `PYTHONIOENCODING=utf-8`
   and `PYTHONUTF8=1`. (No code change — runtime env only.)

3. **`plot_agsm_vs_kan` argument type (code fix).** The task spec instructed
   passing `agsm_tp_values = [tp['point'] for tp in agsm_tps]` (a flat list of
   floats) to `plot_agsm_vs_kan(agsm_transition_points=...)`. But that function
   internally does `[t['point'] for t in transitions]`, i.e. it expects the **list
   of dicts** returned by `find_agsm_transition_points`, not the pre-extracted
   floats. Passing floats raised `'float' object is not subscriptable`. Fixed by
   passing the dict list (`agsm_tps`) to the plot; `agsm_tp_values` is still kept
   for the CSV-side / console reporting.

## How to reproduce

From `D:\pykan`:

```
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
C:\Users\user\miniconda3\envs\pykan-new\python.exe -m github.workflows.Hyein.toy_KAN_analyze <name>
```

where `<name>` in {exponential, logarithm, log2, rosenbrock}.
