# QA Gate 5 — Symbolify-robustness paper aggregation

## Reproduce

```powershell
# from D:\pykan, in the pykan-new conda env
$env:PYTHONPATH="."; conda run -n pykan-new python github/workflows/Hyein/build_symbolify_robustness_paper.py
```

Inputs consumed (already gate-4 verified, read-only):
- `_workspace/robustness_symbolify_summary.csv` (one row per function)
- `analytical_results/{func}/kan_models/{func}_symbolify_robustness.csv` (paired transitions)

Outputs written:
- `figures_for_paper/symbolify_robustness_summary.{png,svg,eps}`
- `figures_for_paper/symbolify_robustness_summary.csv`

No pykan library files or existing Hyein scripts were modified. Only the new
aggregation script `build_symbolify_robustness_paper.py` was added.

## Derivation notes
- `matched/added/dropped` are counted directly from the per-function detail CSV
  (`matched==1`, `added==1`, `dropped==1`) so the table is self-contained and
  cross-checks against the workspace summary's `n_matched/n_added/n_dropped`.
- `drift_frac` in the per-function CSVs is already the fraction of domain width;
  `domain_width` is recovered as `|drift| / |drift_frac|` (constant per function,
  ~2.0 for the normalized toy domains).
- Drift statistics use the absolute value of drift over matched transitions only.
  Functions with 0 matched transitions have undefined drift (NaN) — flagged in the
  figure as "no match (count changed)".

## Per-function table

| func | direction | status | n_edges_symb | n_spline | n_sym | Δcount | matched | added | dropped | count_match_rate | median drift (raw) | max drift (raw) | median drift (%) | max drift (%) |
|------|-----------|--------|------|------|------|------|------|------|------|------|------|------|------|------|
| conditional | introduce | ok | 4 | 12 | 3 | -9 | 1 | 2 | 11 | 0.071 | 0.0454 | 0.0454 | 2.27% | 2.27% |
| damping_sin | introduce | ok | 4 | 8 | 9 | +1 | 6 | 3 | 2 | 0.545 | 0.0390 | 0.0841 | 1.95% | 4.20% |
| log2 | invert | ok | 4 | 4 | 1 | -3 | 0 | 1 | 4 | 0.000 | — | — | — | — |
| exponential | invert | ok | 2 | 4 | 1 | -3 | 0 | 1 | 4 | 0.000 | — | — | — | — |
| logarithm | invert | ok | 2 | 6 | 1 | -5 | 0 | 1 | 6 | 0.000 | — | — | — | — |
| rosenbrock | invert | ok | 4 | 3 | 0 | -3 | 0 | 0 | 3 | 0.000 | — | — | — | — |
| ishigami | introduce | ok | 9 | 12 | 1 | -11 | 0 | 1 | 12 | 0.000 | — | — | — | — |

All functions report `status=ok` (no non-ok flags).

## Robustness conclusion
- **Transition COUNT is not preserved under symbolification for any of the 7
  functions.** The `invert-symbolify` cases (log2, exponential, logarithm,
  rosenbrock) collapse from several spline transitions to 0–1 symbolic ones,
  and `conditional`/`ishigami` (`introduce`) drop the vast majority of
  transitions (12→3 and 12→1). Only `damping_sin` is near-stable in count
  (8→9, Δ=+1).
- **Where transitions DO match, location drift is small.** The only two
  functions with matched transitions — conditional (median 2.27% of domain
  width) and damping_sin (median 1.95%, max 4.20%) — keep matched transition
  points within ~2% and well under 5% of the domain width. So the *positions*
  that survive are reliable even though the *set* of transitions is not.
- **Non-robust (count clearly changes):** log2, exponential, logarithm,
  rosenbrock, ishigami, conditional. **Near-stable count:** damping_sin only.
- **Caveat — refit asymmetry.** The saved-symbolic reading is post-refit
  (auto_symbolic was followed by a fit when the model was saved), whereas the
  freshly-symbolified pure-spline reading is taken without a refit. The two
  branches are therefore not a perfectly controlled spline-vs-symbolic A/B at
  identical parameters; part of the count change reflects this refit asymmetry,
  not symbolification alone.
