"""Gate-4 forced-failure drill: exercise the corrective-loop branches that the
clean saved set never triggered.

(a) auto_symbolic RAISES  -> expect status 'symbolify_failed', spline-only row,
    run continues, n_sym=0.
(b) auto_symbolic produces 0 symbolified edges -> expect 'no_symbolification',
    null contrast (symbolic side empty).

We monkeypatch MultKAN.auto_symbolic so the deepcopy in _build_symbolic_reading
hits the patched method. Use a pure-spline (introduce-symbolify) function so the
auto_symbolic path is actually reached: 'conditional'.
"""
import os, numpy as np, pandas as pd
import github.workflows.Hyein.robustness_symbolify as R

FUNC = 'conditional'
csv_path = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein',
                        'analytical_results', FUNC, 'kan_models',
                        f'{FUNC}_symbolify_robustness.csv')


def read_csv_statuses():
    df = pd.read_csv(csv_path)
    return sorted(set(df['status'].astype(str))), len(df)


# ---- baseline (no patch) so we can restore the real CSV afterwards ----------
import importlib
# The persisted model is kan.custom_multkan_ddp.MultKAN (see MRO), which is the
# class that actually defines auto_symbolic. Patch THAT class.
MK = importlib.import_module('kan.custom_multkan_ddp')
MultKANClass = MK.MultKAN
assert 'auto_symbolic' in MultKANClass.__dict__, "auto_symbolic not on this class"
ORIG = MultKANClass.auto_symbolic

# ===== (a) auto_symbolic raises =====
def raising_auto_symbolic(self, *a, **k):
    raise RuntimeError("FORCED auto_symbolic failure (drill a)")

MultKANClass.auto_symbolic = raising_auto_symbolic
res_a = R.analyze_function(FUNC, rel_thresh=0.1, refit=False, match_frac=0.05)
print("\n=== DRILL (a) auto_symbolic raises ===")
print("status        :", res_a['status'], "(expect symbolify_failed)")
print("reason        :", res_a['reason'])
print("n_edges_sym   :", res_a['n_edges_symbolified'], "(expect 0)")
sm = res_a['summary']
print("summary n_sym :", sm['n_sym'], "(expect 0)")
print("summary n_spline:", sm['n_spline'], "(spline-only row, expect >0)")
stat_a, nrows_a = read_csv_statuses()
print("CSV statuses  :", stat_a, "rows:", nrows_a)
ok_a = (res_a['status'] == R.STATUS_SYMBOLIFY_FAILED and sm['n_sym'] == 0
        and R.STATUS_SYMBOLIFY_FAILED in stat_a)
print("DRILL (a) ->", "PASS" if ok_a else "FAIL")

# ===== (b) auto_symbolic symbolifies 0 edges =====
def noop_auto_symbolic(self, *a, **k):
    # Do nothing: leave all spline masks, no symbolic edges introduced.
    return None

MultKANClass.auto_symbolic = noop_auto_symbolic
res_b = R.analyze_function(FUNC, rel_thresh=0.1, refit=False, match_frac=0.05)
print("\n=== DRILL (b) auto_symbolic symbolifies 0 edges ===")
print("status        :", res_b['status'], "(expect no_symbolification)")
print("reason        :", res_b['reason'])
print("n_edges_sym   :", res_b['n_edges_symbolified'], "(expect 0)")
sm_b = res_b['summary']
print("summary n_sym :", sm_b['n_sym'], "(expect 0, null contrast)")
print("summary n_spline:", sm_b['n_spline'])
stat_b, nrows_b = read_csv_statuses()
print("CSV statuses  :", stat_b, "rows:", nrows_b)
ok_b = (res_b['status'] == R.STATUS_NO_SYMBOLIFICATION and sm_b['n_sym'] == 0
        and R.STATUS_NO_SYMBOLIFICATION in stat_b)
print("DRILL (b) ->", "PASS" if ok_b else "FAIL")

# ---- restore + re-run clean so the on-disk CSV is the real one --------------
MultKANClass.auto_symbolic = ORIG
res_clean = R.analyze_function(FUNC, rel_thresh=0.1, refit=False, match_frac=0.05)
print("\n=== restored clean run ===")
print("status:", res_clean['status'], "n_edges_sym:", res_clean['n_edges_symbolified'])

print("\nFORCED-FAILURE DRILL VERDICT:", "PASS" if (ok_a and ok_b) else "FAIL")
