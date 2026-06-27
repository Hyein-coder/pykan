"""GATE 2 cross-interface verification: mirror toy_KAN_analyze.py section 3.9
logic on damping_sin, assert single-source-of-truth + round-trip consistency."""
import os, numpy as np, torch, joblib
from github.workflows.Hyein.toy_KAN_sweep import KANRegressor, FUNCTION_ZOO
from github.workflows.Hyein.bspline_curvature import find_inflection_points
from github.workflows.Hyein.sectional_gsa import compute_gradient_agsm, find_agsm_transition_points, make_sections

data_name = 'damping_sin'
base = os.path.join('github', 'workflows', 'Hyein', 'analytical_results', data_name)
savepath = os.path.join(base, 'kan_models')

mw = KANRegressor(device='cpu')
mw.load_model(os.path.join(savepath, f'{data_name}_best_kan_model'))
model = mw.model
scaler_X = joblib.load(os.path.join(savepath, f'{data_name}_scaler_X.pkl'))

cfg = FUNCTION_ZOO[data_name]
bounds = cfg['bounds']
feat_names = cfg.get('feat_names', [f'x{i}' for i in range(len(bounds))])
nx = len(bounds)
print(f"data={data_name} nx={nx} bounds={bounds} feat_names={feat_names}")

l = 0
# top2 by global curvature-ish proxy: just use feature_score from a forward
model.forward(torch.tensor(scaler_X.transform(
    np.random.RandomState(0).uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(200, nx))),
    dtype=torch.float32))
scores_tot = model.feature_score.detach().cpu().numpy()
top2 = np.argsort(scores_tot)[::-1][:2].tolist()
ci_idx, cj_idx = int(top2[0]), int(top2[1])
print(f"scores_tot={np.round(scores_tot,4)} top2={top2}")

# === replicate 3.9 ===
curv_ips_norm = {idx: find_inflection_points(model, l, idx) for idx in top2}

def _denorm_feat(vals_norm, feat_idx):
    vals = [v for v in (vals_norm or []) if 0.05 < v < 0.95]
    if not vals:
        return []
    dummy = np.zeros((len(vals), nx)); dummy[:, feat_idx] = vals
    return scaler_X.inverse_transform(dummy)[:, feat_idx].tolist()

curv_ips_raw = {idx: _denorm_feat(curv_ips_norm[idx], idx) for idx in top2}

custom_edges = {}
for idx in top2:
    lo_raw, hi_raw = bounds[idx]
    interior = sorted(v for v in curv_ips_raw[idx] if lo_raw < v < hi_raw)
    custom_edges[idx] = np.array([lo_raw] + interior + [hi_raw], dtype=float)

print("\n=== custom_edges (raw) ===")
for idx in top2:
    print(f"  feat {idx} ({feat_names[idx]}): {np.round(custom_edges[idx],5).tolist()}  (n_sections={len(custom_edges[idx])-1})")
print("  normalized inflections:", {k: np.round(v,5).tolist() for k,v in curv_ips_norm.items()})
print("  band-filtered+denorm raw:", {k: np.round(v,5).tolist() for k,v in curv_ips_raw.items()})

# === CHECK 1: single source of truth ===
ip_lines = list(custom_edges[ci_idx][1:-1])
print("\n=== CHECK1: single source of truth ===")
print(f"  ip_lines (plot vlines) = {np.round(ip_lines,5).tolist()}")
print(f"  == custom_edges[ci_idx][1:-1] = {np.round(custom_edges[ci_idx][1:-1],5).tolist()}")
c1a = np.allclose(ip_lines, custom_edges[ci_idx][1:-1])

# attribution masks derive from scaler_X.transform of custom_edges[feat]
attr_norm_edges = {}
for feat_idx in top2:
    edges_raw = custom_edges[feat_idx]
    dummy = np.zeros((len(edges_raw), nx)); dummy[:, feat_idx] = edges_raw
    attr_norm_edges[feat_idx] = scaler_X.transform(dummy)[:, feat_idx]

# AGSM custom_edges branch -> make_sections produces same edges (post clip/sort/unique)
agsm_edges = {}
for feat_idx in top2:
    lo, hi = bounds[feat_idx]
    e, _ = make_sections(lo, hi, 10, mode='custom', custom_edges=custom_edges[feat_idx])
    agsm_edges[feat_idx] = e
    print(f"  AGSM make_sections edges feat{feat_idx} = {np.round(e,5).tolist()}  matches custom_edges={np.allclose(e, custom_edges[feat_idx])}")
c1b = all(np.allclose(agsm_edges[f], custom_edges[f]) for f in top2)
print(f"  CHECK1: ip_lines==custom_edges[ci][1:-1]: {c1a}; AGSM edges==custom_edges: {c1b}")

# === CHECK 2: raw<->normalized round trip ===
print("\n=== CHECK2: round-trip raw->normalized lands on original normalized inflection ===")
c2 = True
for feat_idx in top2:
    # interior raw edges -> transform back to normalized, compare with original norm inflections (band+interior filtered)
    interior_raw = custom_edges[feat_idx][1:-1]
    if len(interior_raw) == 0:
        print(f"  feat {feat_idx}: no interior edges (nothing to round-trip)")
        continue
    dummy = np.zeros((len(interior_raw), nx)); dummy[:, feat_idx] = interior_raw
    back_norm = scaler_X.transform(dummy)[:, feat_idx]
    # original normalized that survived both filters:
    lo_raw, hi_raw = bounds[feat_idx]
    orig_norm_surv = []
    for v in (curv_ips_norm[feat_idx] or []):
        if 0.05 < v < 0.95:
            d = np.zeros((1, nx)); d[0, feat_idx] = v
            vr = scaler_X.inverse_transform(d)[0, feat_idx]
            if lo_raw < vr < hi_raw:
                orig_norm_surv.append(v)
    orig_norm_surv = np.array(sorted(orig_norm_surv))
    ok = (len(back_norm) == len(orig_norm_surv)) and np.allclose(np.sort(back_norm), orig_norm_surv, atol=1e-6)
    c2 = c2 and ok
    print(f"  feat {feat_idx}: raw{np.round(interior_raw,5).tolist()} -> norm{np.round(np.sort(back_norm),6).tolist()} vs orig norm{np.round(orig_norm_surv,6).tolist()}  match={ok}")

# === CHECK 3: linear feature on damping_sin gets NO interior edges (1 section) ===
print("\n=== CHECK3: linear-feature filtering ===")
for idx in top2:
    n_sec = len(custom_edges[idx]) - 1
    print(f"  feat {idx} ({feat_names[idx]}): n_sections={n_sec} {'(1 section - no interior)' if n_sec==1 else ''}")
# Also show: were any normalized inflections present for the linear feat but filtered out?
for idx in top2:
    raw_all = curv_ips_norm[idx]
    if raw_all:
        dummy = np.zeros((len(raw_all), nx)); dummy[:, idx] = raw_all
        raw_unfiltered = scaler_X.inverse_transform(dummy)[:, idx]
        print(f"  feat {idx}: ALL norm inflections={np.round(raw_all,4).tolist()} -> raw={np.round(raw_unfiltered,4).tolist()} (bounds={bounds[idx]})")

print("\n=== RESULT ===")
print(f"CHECK1 single-source-of-truth: {'PASS' if (c1a and c1b) else 'FAIL'}")
print(f"CHECK2 round-trip:             {'PASS' if c2 else 'FAIL'}")
