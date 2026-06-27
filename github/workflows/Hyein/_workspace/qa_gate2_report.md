# QA GATE 2 Report — §3.9 Curvature-Inflection Integration

**Agent:** curvature-watcher
**Date:** 2026-06-27
**Target:** `D:\pykan\github\workflows\Hyein\toy_KAN_analyze.py` §3.9 (lines 664–823, "3.9 Curvature-Based Inflection")
**Models:** `analytical_results/{damping_sin,conditional}/kan_models/`
**Scripts:** `_workspace/qa_gate2_verify.py`; end-to-end `python -m github.workflows.Hyein.toy_KAN_analyze conditional`

## Overall verdict: **PASS**

---

## Cross-interface checklist

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Single source of truth — AGSM `custom_edges`, attribution masks, vlines all from ONE `custom_edges` | **PASS** | `ip_lines == custom_edges[ci][1:-1]`; `make_sections(mode='custom')` edges == `custom_edges` for both feats |
| 2 | raw↔normalized round-trip consistency | **PASS** | raw interior edges → `scaler_X.transform` → normalized lands on original norm inflections to 1e-6 |
| 2b | band filter `0.05<v<0.95` AND interior filter `lo<v<hi` both applied | **PASS** | spurious raw points >hi/<lo excluded (see CHECK3) |
| 3 | Linear feature (damping_sin x1) ends with NO interior edges (1 section) | **PASS** | x1: spurious norm 0.903 → raw **1.0059** (> hi=1.0) filtered → `custom_edges[1]=[-1,1]`, 1 section |
| 4a | §3 coefficient-based detector preserved | **PASS** | lines 266–281 intact, writes `inflection_points_per_input`, used later by §4 |
| 4b | §3.8 AGSM preserved, in try/except | **PASS** | lines 455–662 own try/except |
| 4c | §3.9 wrapped in try/except (can't crash pipeline) | **PASS** | lines 671–823 `try: ... except Exception: traceback` |
| 5 | End-to-end on `conditional` exits 0, produces png/svg/eps/csv | **PASS** | exit code 0; 4 files written |

---

## CHECK 1 — single source of truth (damping_sin, ci=feat0)
```
ip_lines (plot vlines)        = [-0.507, 0.01534, 0.02134, 0.75382, 0.8739]
custom_edges[ci][1:-1]        = [-0.507, 0.01534, 0.02134, 0.75382, 0.8739]   -> EQUAL
AGSM make_sections feat0      = [-1.0, -0.507, 0.01534, 0.02134, 0.75382, 0.8739, 1.0]  == custom_edges -> True
AGSM make_sections feat1      = [-1.0, 1.0]                                              == custom_edges -> True
```
The AGSM custom branch (`make_sections(mode='custom')`, sectional_gsa.py:171-185) consumes the full
`[lo, …interior…, hi]` array — exactly the format §3.9 builds (`[lo_raw] + interior + [hi_raw]`).
Attribution masks (block §6) recompute `edges_norm = scaler_X.transform(custom_edges[feat])`, i.e.
the SAME array. All three consumers trace to one object. No independent recomputation.

## CHECK 2 — round-trip (damping_sin feat0)
```
raw[-0.507, 0.01534, 0.02134, 0.75382, 0.8739]
 -> norm[0.296692, 0.506015, 0.508421, 0.801955, 0.850075]
 vs  orig norm[0.296692, 0.506015, 0.508421, 0.801955, 0.850075]   match=True
```

## CHECK 3 — linear-feature filtering (damping_sin)
```
feat 0 (x0): n_sections=6
feat 1 (x1): n_sections=1  (1 section - no interior)   <-- linear feature, correct
ALL norm inflections feat1 = [0.903] -> raw [1.0059]  (bounds [-1,1])  -> filtered (>hi)
ALL norm inflections feat0 = [0.0753,0.2967,0.506,0.5084,0.802,0.8501,0.9247,0.9583]
              -> raw [-1.0594,-0.507,0.0153,0.0213,0.7538,0.8739,1.06,1.1441]
              filtered out: -1.0594(<lo), 1.06/1.1441(>hi)  -> 5 interior survive
```
The spurious linear-feature inflection (raw ~1.006) is removed by the `lo_raw < v < hi_raw`
interior filter before it can enter `custom_edges`, so x1 contributes no transition to AGSM or
attribution. Confirms the GATE 2 requirement.

## End-to-end on `conditional` (kink at raw x0=0)
Command: `$env:PYTHONUTF8=1; conda run -n pykan-new --no-capture-output python -m github.workflows.Hyein.toy_KAN_analyze conditional`
- **Exit code 0.** §3.9 ran (`🧭 ...`), no exception.
- Output files (all present): `conditional_curvature_inflection.{png,svg,eps,csv}` in `analytical_results/conditional/kan_models/`.
- top2 features = {0: Conditional(x0), 1: Linear(x1)}; ci=feat0.
- **Curvature inflection near x0≈0 raw:** feat0 raw inflections include `0.0310` and `-0.1166/-0.1223` —
  straddling the x0=0 kink (closest interior edge **0.031 raw**, ~0.03 from the true kink).
- **AGSM transitions (raw):** `[-0.808, -0.358, -0.227]`.
- CSV `conditional_curvature_inflection.csv` shows the per-interval dual measure; `Conditional(x0)`
  S_hat drops from ~2.0 to ~1.60 at section center ≈ -0.043 (crossing x0≈0), consistent with the kink.

Full normalized/raw inflection sets are in the run log (relayed in summary).

## Notes
- §3.9 uses `scores_tot` (global feature_score) to pick top2; the verify script reproduced
  top2=[0,1] for damping_sin and the pipeline reported top2={0,1} for conditional. Consistent.
- A pre-existing `std() dof<=0` UserWarning fires for tiny intervals elsewhere in the pipeline
  (not §3.9-specific) — harmless, does not affect exit code.
