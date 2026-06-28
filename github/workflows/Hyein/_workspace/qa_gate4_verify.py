"""GATE 4 verification: full-model input derivatives (chain across layers)."""
import os, numpy as np, torch
from github.workflows.Hyein.toy_KAN_sweep import KANRegressor, FUNCTION_ZOO
import github.workflows.Hyein.bspline_curvature as bc


def load(name):
    sp = os.path.join('github', 'workflows', 'Hyein', 'analytical_results', name, 'kan_models')
    mw = KANRegressor(device='cpu')
    mw.load_model(os.path.join(sp, f'{name}_best_kan_model'))
    return mw.model


def make_batch(model, name, N=64, seed=0):
    """Normalized inputs inside the grid interior (avoids exact knots)."""
    act0 = model.act_fun[0]; k = act0.k; n0 = act0.coef.shape[0]
    rng = np.random.RandomState(seed)
    X = np.zeros((N, n0))
    for i in range(n0):
        kn = act0.grid[i, k - 1:-2].detach().cpu().numpy()
        lo, hi = float(kn.min()), float(kn.max())
        X[:, i] = rng.uniform(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo), N)
    return X


def run_one(name):
    print(f"\n############ {name} ############")
    model = load(name)
    L = len(model.act_fun)
    n_in = model.act_fun[0].coef.shape[0]
    print(f"  layers={L} width={model.width} width_in={model.width_in}")
    X = make_batch(model, name)

    # CHECK 1: verify_model_derivatives
    print("  --- CHECK1: verify_model_derivatives (analytic vs autograd) ---")
    rep = bc.verify_model_derivatives(model, X)
    g_err = rep['grad_max_abs_err']
    print(f"    grad_max_abs_err = {g_err:.3e}")
    df_worst = d2_worst = 0.0
    for i, d in sorted(rep['curv'].items()):
        print(f"    feat{i}: df_max_abs_err={d['df_max_abs_err']:.3e}  d2f_max_abs_err={d['d2f_max_abs_err']:.3e}")
        df_worst = max(df_worst, d['df_max_abs_err']); d2_worst = max(d2_worst, d['d2f_max_abs_err'])
    # float32 model -> accept 1e-3
    c1 = (g_err <= 1e-3) and (df_worst <= 1e-3) and (d2_worst <= 1e-3)
    print(f"    CHECK1: {'PASS' if c1 else 'FAIL'} (grad={g_err:.2e} df={df_worst:.2e} d2f={d2_worst:.2e}; threshold 1e-3 float32)")

    # CHECK 2: model_gradient[:,i] == model_directional_derivatives(...)[1]
    print("  --- CHECK2: gradient two ways agree ---")
    grad = bc.model_gradient(model, X)
    c2 = True
    for i in range(n_in):
        _, df_i, _ = bc.model_directional_derivatives(model, X, i)
        diff = float(np.max(np.abs(grad[:, i] - df_i)))
        ok = diff <= 1e-6
        c2 = c2 and ok
        print(f"    feat{i}: max|model_gradient - directional df| = {diff:.3e} {'PASS' if ok else 'FAIL'}")
    print(f"    CHECK2: {'PASS' if c2 else 'FAIL'}")

    return model, X, rep, grad, c1, c2


def main():
    # ===== damping_sin (2-layer pure spline) =====
    md, Xd, repd, gradd, c1d, c2d = run_one('damping_sin')

    # CHECK 3: chain-rule sanity - gradient is NOT just layer-0 edge derivative
    print("  --- CHECK3: chain-rule composes BOTH layers (damping_sin) ---")
    # layer-0-only "gradient" would be s0 * phi'_{i,j} (no layer-1 factor). Build it and show it differs.
    acts = bc._layer_acts(md, Xd)
    phi1_0 = bc.activation_first_derivative(md, 0, acts[0]).detach().cpu().numpy()  # (b,n0,n1)
    s0 = bc._combined_node_scale(md, 0).detach().cpu().numpy()                       # (n1,)
    # naive layer-0-only attempt: sum_j s0_j phi'_{i,j} (wrong: ignores layer1)
    naive = np.einsum('j,bij->bi', s0, phi1_0)   # (b,n0)
    full = gradd                                  # correct 2-layer gradient
    # autograd reference
    g_ref = bc.model_gradient_autograd(md, Xd)
    diff_full = float(np.max(np.abs(full - g_ref)))
    diff_naive = float(np.max(np.abs(naive - g_ref)))
    c3 = (diff_full <= 1e-3) and (diff_naive > 10 * max(diff_full, 1e-9))
    print(f"    full 2-layer grad vs autograd: {diff_full:.3e}")
    print(f"    layer0-only naive grad vs autograd: {diff_naive:.3e}  (should be MUCH larger)")
    print(f"    CHECK3: {'PASS' if c3 else 'FAIL'} (full matches autograd AND differs from layer0-only)")

    # CHECK 5: find_model_inflection_points on damping_sin x0
    print("  --- CHECK5: find_model_inflection_points (damping_sin x0) ---")
    try:
        ips = bc.find_model_inflection_points(md, 0)
        act0 = md.act_fun[0]; k = act0.k
        kn = act0.grid[0, k-1:-2].detach().cpu().numpy(); lo, hi = float(kn.min()), float(kn.max())
        sorted_ok = (ips == sorted(ips)); inrange = all(lo - 1e-9 <= p <= hi + 1e-9 for p in ips)
        c5 = sorted_ok and inrange
        print(f"    inflections(x0)={np.round(ips,4).tolist()} n={len(ips)} sorted={sorted_ok} in_range[{lo:.3f},{hi:.3f}]={inrange}")
        print(f"    CHECK5: {'PASS' if c5 else 'FAIL'}")
    except Exception as e:
        c5 = False; print(f"    CHECK5: FAIL exception {type(e).__name__}: {e}")

    # ===== exponential (1-layer symbolified) =====
    me, Xe, repe, grade, c1e, c2e = run_one('exponential')

    # CHECK 4: single-layer reduction df/dx_i == s * phi'_{i,0}, and symbolic edges flow (non-zero, matches autograd)
    print("  --- CHECK4: single-layer reduction + symbolic edges flow (exponential) ---")
    acts_e = bc._layer_acts(me, Xe)
    phi1_e = bc.activation_first_derivative(me, 0, acts_e[0]).detach().cpu().numpy()  # (b,n0,1)
    s_e = bc._combined_node_scale(me, 0).detach().cpu().numpy()                        # (1,)
    g_ref_e = bc.model_gradient_autograd(me, Xe)
    c4 = True
    for i in range(me.act_fun[0].coef.shape[0]):
        manual = s_e[0] * phi1_e[:, i, 0]            # one edge to single output
        diff_manual = float(np.max(np.abs(grade[:, i] - manual)))
        diff_ag = float(np.max(np.abs(grade[:, i] - g_ref_e[:, i])))
        nonzero = float(np.max(np.abs(grade[:, i]))) > 1e-6
        ok = (diff_manual <= 1e-6) and (diff_ag <= 1e-3) and nonzero
        c4 = c4 and ok
        print(f"    feat{i}: grad vs s*phi'={diff_manual:.3e}  grad vs autograd={diff_ag:.3e}  nonzero={nonzero} {'PASS' if ok else 'FAIL'}")
    print(f"    CHECK4: {'PASS' if c4 else 'FAIL'}")

    # CHECK 6: mult-node guard
    print("\n  --- CHECK6: mult-node guard raises clear error ---")
    import types
    orig = md.width
    try:
        # synthesize a mult node at layer 1: width[1] = [n_sum, n_mult>0]
        md.width = [list(orig[0]), [orig[1][0], 1], list(orig[2])]
        raised = False
        try:
            bc._combined_node_scale(md, 0)
        except ValueError as e:
            raised = True; msg = str(e)
        print(f"    _combined_node_scale with mult node: {'raised ValueError' if raised else 'DID NOT RAISE'}")
        if raised:
            print(f"      message: {msg}")
        c6 = raised and ('mult' in msg.lower() or 'summation' in msg.lower())
    finally:
        md.width = orig
    print(f"    CHECK6: {'PASS' if c6 else 'FAIL'}")

    print("\n############ GATE 4 SUMMARY ############")
    print(f"damping_sin CHECK1(verify):        {'PASS' if c1d else 'FAIL'}")
    print(f"damping_sin CHECK2(grad two ways): {'PASS' if c2d else 'FAIL'}")
    print(f"damping_sin CHECK3(chain rule):    {'PASS' if c3 else 'FAIL'}")
    print(f"damping_sin CHECK5(model inflect): {'PASS' if c5 else 'FAIL'}")
    print(f"exponential CHECK1(verify):        {'PASS' if c1e else 'FAIL'}")
    print(f"exponential CHECK2(grad two ways): {'PASS' if c2e else 'FAIL'}")
    print(f"exponential CHECK4(1-layer+symb):  {'PASS' if c4 else 'FAIL'}")
    print(f"CHECK6(mult-node guard):           {'PASS' if c6 else 'FAIL'}")


if __name__ == '__main__':
    main()
