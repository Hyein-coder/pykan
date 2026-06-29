# QA GATE 4 Report — Symbolification-Robustness Driver

**Agent:** curvature-watcher
**Date:** 2026-06-28
**Target:** `github/workflows/Hyein/robustness_symbolify.py`
**Notes:** `_workspace/robustness_symbolify_notes.md`
**Env:** `pykan-new`, run from `D:\pykan` with `PYTHONPATH=.`
**Check scripts:** `_workspace/qa_gate4_check1.py`, `qa_gate4_deadedge.py`,
`qa_gate4_forced_fail.py`, `qa_gate4_check45.py`

## Overall verdict: **PASS** (no correctives required)

The driver reproduces clean on the full 7-function set (all `status=ok`, both
construction directions exercised), all six boundary cross-checks hold, and the
two corrective-loop failure branches were forced and confirmed to emit the
correct statuses + rows (never silent skips).

---

## Boundary cross-check checklist

| # | Check | Result | Evidence |
|---|-------|--------|----------|
| 1 | Spline reading genuinely spline (sym branch off; only symbolified edges' mask flipped; spline s_i non-degenerate) | **PASS** | `qa_gate4_check1.py` |
| 1b | Dead/pruned edges (mask 0, not symbolified) NOT revived | **PASS (logic)** | `qa_gate4_deadedge.py` — no saved model has dead+symbolified mix, so verified by code, not data |
| 2 | Same `x_grid` / `rel_thresh` / `layer` to both `find_ranking_transitions` | **PASS** | driver L296,301-302,337-338 — one grid object |
| 3 | Already-symbolified used as-saved (no re-auto_symbolic); pure-spline symbolified with KANRegressor production defaults; branch on `symbolic_edge_info` non-empty | **PASS** | driver L159 `if sym_info:`; `SYM_*` consts == `toy_KAN_sweep` defaults (lib 12-fn, a/b_range ±10, r2_threshold 0, weight_simple 0) |
| 4 | Raw-space consistency: both norm point sets → raw via same `scaler_X.inverse_transform` | **PASS** | `qa_gate4_check45.py` (diff 0.0); driver L383-384 |
| 5 | Deepcopy-before-forward; original never forwarded | **PASS** | `qa_gate4_check45.py` — un-forwarded deepcopies clean; forwarded copy fails with the dev's exact error |
| 6 | Outputs present: `{func}_symbolify_robustness.{png,svg,eps,csv}` ×7 + summary CSV with `status` col | **PASS** | all 28 per-func files + `_workspace/robustness_symbolify_summary.csv` present |

### Check 1 evidence (logarithm, invert-symbolify)
- saved symbolic edges `[((0,0),'log'), ((1,0),'x')]`; `mask BEFORE = [[0],[0]]`.
- `_build_spline_reading`: flipped `(0,0):0→1`, `(1,0):0→1` (only the symbolified
  edges, both 0→1); `symbolic_enabled=False`.
- spline-side peak `|s_i|` = 2.55 (s_0) / 1.01 (s_1) → non-degenerate.
- contrast: toggling symbolic branch ON changes s_0 by 160.3 → disabling it is
  load-bearing; the spline reading is genuinely the spline curve, not symbolic.

### Check 3 — production defaults match
Driver `SYM_LIB / SYM_A_RANGE / SYM_B_RANGE / SYM_R2_THRESHOLD / SYM_WEIGHT_SIMPLE`
= `['sin','cos','x','x^2','x^3','x^4','exp','log','sqrt','tanh','1/x','1/x^2']`,
(-10,10), (-10,10), 0.0, 0.0 — identical to `toy_KAN_sweep.py` L140-203.

---

## Forced-failure drill (corrective loop) — `qa_gate4_forced_fail.py`

Monkeypatched `kan.custom_multkan_ddp.MultKAN.auto_symbolic` (the class the
persisted model actually instantiates — NOT `kan.MultKAN.MultKAN`) on a
pure-spline function (`conditional`, introduce-symbolify) so the auto_symbolic
path is reached.

| Drill | Forced condition | Expected | Observed | Result |
|-------|------------------|----------|----------|--------|
| (a) | `auto_symbolic` raises | `symbolify_failed`; spline-only row; continue | status=`symbolify_failed`, n_spline=12, n_sym=0, CSV statuses=`['symbolify_failed']` (12 rows) | **PASS** |
| (b) | `auto_symbolic` → 0 edges | `no_symbolification`; null contrast; continue | status=`no_symbolification`, n_sym=0, CSV statuses=`['no_symbolification']` (12 rows) | **PASS** |

Both statuses are written to the per-function CSV and surface in `result['summary']`
(never a silent skip). Restored clean run returns `ok` / 4 edges, and the full
set was re-run afterward so the on-disk CSVs + summary are the real results.

`symbolic_degenerate` / `spline_reading_unavailable` branches were verified by
code path (status determination block L323-348) but the saved set does not
trigger them; they are reachable and correctly labeled.

---

## Sanity spot-check (logarithm: 6 spline vs 1 symbolic)

Figure `logarithm_symbolify_robustness.png`:
- both panels show solid blue (spline) + dashed red (symbolic) s_i overlays.
- x0 (Log): 1 symbolic vline (red dotted, left edge where symbolic-log derivative
  spikes then crosses τ) + 2 spline vlines (~0.77, 0.93, near-edge wiggle).
- x1 (Linear): 4 spline vlines (near both edges where spline dips below τ);
  symbolic `x` is flat, no crossing.
- CSV: 6 spline-branch (dropped) + 1 symbolic-branch (added) + 0 matched rows,
  exactly matching summary `n_spline=6, n_sym=1, n_matched=0, n_added=1,
  n_dropped=6`. Figure ↔ CSV consistent.

---

## Reproduce
```
# clean full set
PYTHONPATH=. python github/workflows/Hyein/robustness_symbolify.py
# boundary checks
PYTHONPATH=. python github/workflows/Hyein/_workspace/qa_gate4_check1.py
PYTHONPATH=. python github/workflows/Hyein/_workspace/qa_gate4_deadedge.py
PYTHONPATH=. python github/workflows/Hyein/_workspace/qa_gate4_check45.py
# corrective-loop drill (restores CSVs afterward)
PYTHONPATH=. python github/workflows/Hyein/_workspace/qa_gate4_forced_fail.py
```

## Minor observations (non-blocking, not FAILs)
1. Dead-edge protection (1b) is correct by construction but untested with data —
   no saved invert-symbolify model has a genuinely-dead edge (mask 0, not
   symbolified) alongside symbolified ones. If such a model ever appears, the
   `for (i,j) in sym_info` loop already restricts the flip correctly.
2. `symbolic_degenerate` / `spline_reading_unavailable` statuses are reachable
   and labeled correctly but not exercised by the current saved set.
