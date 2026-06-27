"""GATE 1 verification for bspline_curvature.py"""
import os, glob, numpy as np, torch, joblib

from github.workflows.Hyein.toy_KAN_sweep import KANRegressor, FUNCTION_ZOO
import github.workflows.Hyein.bspline_curvature as bc


def load_model(data_name):
    savepath = os.path.join('github', 'workflows', 'Hyein',
                            'analytical_results', data_name, 'kan_models')
    mw = KANRegressor(device='cpu')
    mw.load_model(os.path.join(savepath, f'{data_name}_best_kan_model'))
    return mw.model


def main():
    data_name = 'damping_sin'
    print(f"=== Loading model: {data_name} ===")
    model = load_model(data_name)
    act = model.act_fun[0]
    in_dim, out_dim = act.coef.shape[:2]
    k = act.k
    print(f"layer0: in={in_dim} out={out_dim} k={k} grid.shape={tuple(act.grid.shape)} coef.shape={tuple(act.coef.shape)}")
    print(f"grid M = {act.grid.shape[1]}, expected nc=M-k-1={act.grid.shape[1]-k-1}, actual coef nc={act.coef.shape[2]}")

    # --- shape check for d2 / basis ---
    d2coef, k2 = bc.spline_second_derivative_coef(act.grid, act.coef, k)
    M = act.grid.shape[1]
    b = bc.B_batch(torch.zeros(3, in_dim), act.grid, k=k2)
    print(f"\n[shape] k2={k2}, d2coef last-dim={d2coef.shape[-1]}, B_batch(k2) last-dim={b.shape[-1]}, expected M-k+1={M-k+1}")
    assert d2coef.shape[-1] == b.shape[-1] == (M - k + 1), "BASIS COUNT MISMATCH"
    print("[shape] PASS: basis count == d2 coef count == M-k+1")

    # --- CHECK 1: verify_against_autograd ---
    print("\n=== CHECK 1: verify_against_autograd(model, 0) ===")
    rep = bc.verify_against_autograd(model, 0)
    worst = 0.0
    all_pass = True
    for (i, j), d in sorted(rep.items()):
        mae, sc, rel = d['max_abs_err'], d['scale'], d['rel_err']
        ok = (mae <= 1e-4) or (rel <= 1e-3)
        all_pass = all_pass and ok
        worst = max(worst, mae)
        print(f"  edge({i},{j}): max_abs_err={mae:.3e} scale={sc:.3e} rel_err={rel:.3e}  {'PASS' if ok else 'FAIL'}")
    print(f"CHECK1 overall: {'PASS' if all_pass else 'FAIL'}  (worst max_abs_err={worst:.3e})")

    # --- CHECK 2: independent central finite difference of FULL activation ---
    print("\n=== CHECK 2: independent central finite-difference 2nd derivative ===")
    from kan.spline import coef2curve

    def full_edge_value(i, j, xs):
        xt = torch.zeros(len(xs), in_dim, dtype=act.coef.dtype)
        xt[:, i] = torch.as_tensor(xs, dtype=act.coef.dtype)
        with torch.no_grad():
            base = act.base_fun(xt)
            yspl = coef2curve(xt, act.grid, act.coef, act.k)
            y = (act.scale_base[None] * base[:, :, None] + act.scale_sp[None] * yspl)
            y = act.mask[None] * y
        return y[:, i, j].cpu().numpy()

    c2_worst = 0.0
    c2_pass = True
    for (i, j) in [(0, 0)] + [(i, 0) for i in range(1, min(in_dim, 3))]:
        knots = act.grid[i, k - 1:-2].detach().cpu().numpy()
        lo, hi = float(np.min(knots)), float(np.max(knots))
        span = hi - lo
        xs = np.linspace(lo + 0.1 * span, hi - 0.1 * span, 50)
        h = 1e-3 * span
        fd = (full_edge_value(i, j, xs + h) - 2 * full_edge_value(i, j, xs)
              + full_edge_value(i, j, xs - h)) / h ** 2
        ana = bc.edge_second_derivative(model, 0, i, j, xs)
        diff = np.max(np.abs(fd - ana))
        scale = np.max(np.abs(fd)) + 1e-12
        rel = diff / scale
        ok = (diff <= 1e-2) or (rel <= 1e-2)
        c2_pass = c2_pass and ok
        c2_worst = max(c2_worst, diff)
        print(f"  edge({i},{j}): max|fd-ana|={diff:.3e} scale={scale:.3e} rel={rel:.3e} sample_ana[:3]={np.round(ana[:3],4)}  {'PASS' if ok else 'FAIL'}")
    print(f"CHECK2 overall: {'PASS' if c2_pass else 'FAIL'} (worst abs diff={c2_worst:.3e})")

    # --- CHECK 3 logic: k<2 raises ---
    print("\n=== CHECK 3: k<2 raises ValueError ===")
    try:
        bc.spline_second_derivative_coef(act.grid, act.coef, 1)
        print("  FAIL: k=1 did not raise")
        k_raise = False
    except ValueError as e:
        print(f"  PASS: raised ValueError: {e}")
        k_raise = True

    # --- CHECK 4: near-linear / constant edge -> ~0 second derivative & few inflections ---
    print("\n=== CHECK 4: edge cases ===")
    # find a near-linear edge: low max|2nd deriv|. Also test find_inflection_points runs.
    edge_peaks = {}
    for i in range(in_dim):
        for j in range(out_dim):
            if float(act.mask[i, j]) == 0.0:
                continue
            knots = act.grid[i, k - 1:-2].detach().cpu().numpy()
            lo, hi = float(np.min(knots)), float(np.max(knots))
            xs = np.linspace(lo, hi, 200)
            d2 = bc.edge_second_derivative(model, 0, i, j, xs)
            edge_peaks[(i, j)] = float(np.nanmax(np.abs(d2)))
    sorted_edges = sorted(edge_peaks.items(), key=lambda kv: kv[1])
    print("  edge peak|2nd-deriv| (sorted ascending, first 5):")
    for (i, j), p in sorted_edges[:5]:
        print(f"    edge({i},{j}): peak|phi''|={p:.4e}")

    # find_inflection_points runs & returns sorted normalized values
    print("\n  find_inflection_points per input:")
    fip_ok = True
    for i in range(in_dim):
        try:
            pts = bc.find_inflection_points(model, 0, i)
            sorted_ok = (pts == sorted(pts))
            knots = act.grid[i, k - 1:-2].detach().cpu().numpy()
            lo, hi = float(np.min(knots)), float(np.max(knots))
            in_range = all(lo - 1e-6 <= p <= hi + 1e-6 for p in pts)
            print(f"    input {i}: n_inflections={len(pts)} sorted={sorted_ok} in_grid_range={in_range} pts={np.round(pts,3).tolist()}")
            fip_ok = fip_ok and sorted_ok and in_range
        except Exception as e:
            print(f"    input {i}: FAIL exception {type(e).__name__}: {e}")
            fip_ok = False

    # synthetic constant/linear edge test (controlled): overwrite a copy's coef
    print("\n  synthetic constant & linear spline test:")
    # constant: all coef equal -> spline const -> spline'' = 0
    grid1 = act.grid[:1].clone()
    nc = act.coef.shape[2]
    coef_const = torch.ones(1, 1, nc, dtype=act.coef.dtype)
    xs = np.linspace(float(grid1[0, k-1]), float(grid1[0, -3]), 100)
    xt = torch.zeros(len(xs), 1, dtype=act.coef.dtype); xt[:, 0] = torch.as_tensor(xs, dtype=act.coef.dtype)
    sp2_const = bc.eval_spline_second_derivative(grid1, coef_const, k, xt)[:, 0, 0].numpy()
    print(f"    constant spline: max|spline''|={np.max(np.abs(sp2_const)):.3e} (expect ~0)")
    # linear: coef = linspace -> for uniform knots spline is ~linear -> spline''~0
    coef_lin = torch.linspace(0, 1, nc, dtype=act.coef.dtype)[None, None, :]
    sp2_lin = bc.eval_spline_second_derivative(grid1, coef_lin, k, xt)[:, 0, 0].numpy()
    print(f"    linear-coef spline: max|spline''|={np.max(np.abs(sp2_lin)):.3e}")
    const_ok = np.max(np.abs(sp2_const)) < 1e-6

    print("\n=== SUMMARY ===")
    print(f"CHECK1 (autograd):  {'PASS' if all_pass else 'FAIL'}  worst max_abs_err={worst:.3e}")
    print(f"CHECK2 (fd):        {'PASS' if c2_pass else 'FAIL'}  worst abs diff={c2_worst:.3e}")
    print(f"CHECK3 (k<2 raise): {'PASS' if k_raise else 'FAIL'}")
    print(f"CHECK4 (edge cases/const): {'PASS' if (fip_ok and const_ok) else 'FAIL'}")


if __name__ == '__main__':
    main()
