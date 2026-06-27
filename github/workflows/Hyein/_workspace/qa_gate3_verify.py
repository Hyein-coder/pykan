"""GATE 3 verification: symbolic-aware bspline_curvature.py."""
import os, numpy as np, torch
from github.workflows.Hyein.toy_KAN_sweep import KANRegressor, FUNCTION_ZOO
import github.workflows.Hyein.bspline_curvature as bc


def load(data_name):
    sp = os.path.join('github', 'workflows', 'Hyein', 'analytical_results', data_name, 'kan_models')
    mw = KANRegressor(device='cpu')
    mw.load_model(os.path.join(sp, f'{data_name}_best_kan_model'))
    return mw.model


def describe_symbolic(model):
    print(f"  symbolic_enabled={getattr(model,'symbolic_enabled',None)}")
    info = bc.symbolic_edge_info(model, 0)
    print(f"  active symbolic edges (i,j)->fun: {info}")
    act = model.act_fun[0]
    print(f"  act_fun[0].mask=\n{act.mask.detach().cpu().numpy()}")
    return info


def check_verify(model, name):
    print(f"\n=== [{name}] CHECK1: verify_against_autograd ===")
    rep = bc.verify_against_autograd(model, 0)
    act = model.act_fun[0]
    info = bc.symbolic_edge_info(model, 0)
    allpass = True
    worst = 0.0
    for (i, j), d in sorted(rep.items()):
        spline_active = float(act.mask[i, j]) != 0.0
        sym_active = (i, j) in info
        active = spline_active or sym_active
        mae, sc, rel = d['max_abs_err'], d['scale'], d['rel_err']
        if not active:
            tag = 'inactive(skip)'
            ok = True
        else:
            ok = (mae <= 1e-4) or (rel <= 1e-3)
            tag = ('sym:' + info[(i, j)]) if sym_active else 'spline'
            allpass = allpass and ok
            if np.isfinite(mae):
                worst = max(worst, mae)
        print(f"  edge({i},{j}) [{tag}]: max_abs_err={mae:.3e} scale={sc:.3e} rel_err={rel:.3e}  {'PASS' if ok else 'FAIL'}")
    print(f"  CHECK1 [{name}]: {'PASS' if allpass else 'FAIL'} (worst active max_abs_err={worst:.3e})")
    return allpass, worst


def check_value(model, name):
    """CHECK2: phi value from edge_curves matches model.symbolic_fun postacts."""
    print(f"\n=== [{name}] CHECK2: phi value vs symbolic_fun postacts ===")
    act = model.act_fun[0]
    in_dim, out_dim = act.coef.shape[:2]
    k = act.k
    info = bc.symbolic_edge_info(model, 0)
    sym = model.symbolic_fun[0]
    allpass = True
    for (i, j), fname in info.items():
        knots = act.grid[i, k - 1:-2].detach().cpu().numpy()
        lo, hi = float(knots.min()), float(knots.max())
        xs = np.linspace(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo), 100)
        phi, _, _ = bc.edge_curves(model, 0, i, j, xs)
        # model symbolic_fun forward: returns postacts shape (batch, out, in)
        x_eval = torch.zeros(len(xs), in_dim, dtype=torch.float32)
        x_eval[:, i] = torch.as_tensor(xs, dtype=torch.float32)
        with torch.no_grad():
            res = sym(x_eval)
        # Symbolic_KANLayer.forward returns (y, postacts); postacts is (batch,out,in)
        postacts = res[1] if isinstance(res, (tuple, list)) else res
        true_edge = postacts[:, j, i].cpu().numpy()
        diff = np.max(np.abs(phi - true_edge))
        scale = np.max(np.abs(true_edge)) + 1e-12
        flat = np.ptp(phi) < 1e-9
        ok = (diff <= 1e-3) or (diff / scale <= 1e-3)
        allpass = allpass and ok
        print(f"  edge({i},{j}) [{fname}]: phi range=[{phi.min():.4f},{phi.max():.4f}] (ptp={np.ptp(phi):.4f}, flat={flat})")
        print(f"      max|phi-true|={diff:.3e} rel={diff/scale:.3e}  {'PASS' if ok else 'FAIL'}")
    print(f"  CHECK2 [{name}]: {'PASS' if allpass else 'FAIL'}")
    return allpass


def check_symbolic_deriv_xcheck(model, name):
    """CHECK3: closed-form _symbolic_branch order2 vs independent autograd double-grad."""
    print(f"\n=== [{name}] CHECK3: sympy closed-form vs autograd (symbolic deriv) ===")
    act = model.act_fun[0]
    in_dim = act.coef.shape[:2][0]
    k = act.k
    info = bc.symbolic_edge_info(model, 0)
    sym = model.symbolic_fun[0]
    allpass = True
    for (i, j), fname in info.items():
        knots = act.grid[i, k - 1:-2].detach().cpu().numpy()
        lo, hi = float(knots.min()), float(knots.max())
        xs = np.linspace(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), 80)
        x_eval = torch.zeros(len(xs), in_dim, dtype=torch.float64)
        x_eval[:, i] = torch.as_tensor(xs, dtype=torch.float64)
        closed = bc._symbolic_branch(model, 0, x_eval, order=2)[:, i, j].numpy()
        # independent autograd double grad of c*f(a*x+b)+d
        a = float(sym.affine[j, i, 0]); b = float(sym.affine[j, i, 1])
        c = float(sym.affine[j, i, 2]); d = float(sym.affine[j, i, 3])
        xt = torch.tensor(xs, dtype=torch.float64, requires_grad=True)
        y = c * sym.funs[j][i](a * xt + b) + d
        g1, = torch.autograd.grad(y.sum(), xt, create_graph=True)
        g2, = torch.autograd.grad(g1.sum(), xt)
        ag = g2.detach().numpy()
        fin = np.isfinite(closed) & np.isfinite(ag)
        diff = np.max(np.abs(closed[fin] - ag[fin]))
        scale = np.max(np.abs(ag[fin])) + 1e-12
        ok = (diff <= 1e-4) or (diff / scale <= 1e-3)
        allpass = allpass and ok
        print(f"  edge({i},{j}) [{fname}]: max|closed-autograd|={diff:.3e} rel={diff/scale:.3e}  {'PASS' if ok else 'FAIL'}")
    print(f"  CHECK3 [{name}]: {'PASS' if allpass else 'FAIL'}")
    return allpass


def check_semantics(model, name):
    """CHECK4: x^2 -> phi''=const (no inflection); tanh -> inflection near x=-b/a."""
    print(f"\n=== [{name}] CHECK4: semantics (x^2 const, tanh inflection) ===")
    act = model.act_fun[0]
    in_dim = act.coef.shape[0]; k = act.k
    info = bc.symbolic_edge_info(model, 0)
    sym = model.symbolic_fun[0]
    for (i, j), fname in info.items():
        knots = act.grid[i, k - 1:-2].detach().cpu().numpy()
        lo, hi = float(knots.min()), float(knots.max())
        xs = np.linspace(lo, hi, 300)
        _, _, d2 = bc.edge_curves(model, 0, i, j, xs)
        a = float(sym.affine[j, i, 0]); b = float(sym.affine[j, i, 1]); c = float(sym.affine[j, i, 2])
        ptp = np.ptp(d2[np.isfinite(d2)]) if np.any(np.isfinite(d2)) else np.nan
        ips = bc.find_inflection_points(model, 0, i, j_list=[j])
        print(f"  edge({i},{j}) [{fname}]: phi'' range ptp={ptp:.3e}, a={a:.4f},b={b:.4f},c={c:.4f}, inflections(j={j})={np.round(ips,4).tolist()}")
        if fname in ('x^2', 'x_squared'):
            expected = 2 * c * a * a
            print(f"      x^2 expected phi''=2ca^2={expected:.4f}; mean phi''={np.nanmean(d2):.4f}; const(ptp<1e-3)={ptp<1e-3}; n_inflection={len(ips)} (expect 0)")
        if fname in ('tanh',):
            x_zero = -b / a if a != 0 else np.nan
            print(f"      tanh arg-zero at x=-b/a={x_zero:.4f}; inflections={np.round(ips,4).tolist()}")


def main():
    # exponential
    print("######## EXPONENTIAL ########")
    me = load('exponential')
    describe_symbolic(me)
    e1, ew = check_verify(me, 'exponential')
    e2 = check_value(me, 'exponential')
    e3 = check_symbolic_deriv_xcheck(me, 'exponential')
    check_semantics(me, 'exponential')

    # logarithm
    print("\n\n######## LOGARITHM ########")
    ml = load('logarithm')
    describe_symbolic(ml)
    l1, lw = check_verify(ml, 'logarithm')
    l2 = check_value(ml, 'logarithm')
    # NaN leak check
    act = ml.act_fun[0]; in_dim = act.coef.shape[0]; k = act.k
    nan_ok = True
    for i in range(in_dim):
        knots = act.grid[i, k-1:-2].detach().cpu().numpy(); lo, hi = float(knots.min()), float(knots.max())
        xs = np.linspace(lo, hi, 200)
        for j in range(act.coef.shape[1]):
            _, _, d2 = bc.edge_curves(ml, 0, i, j, xs)
            if np.any(~np.isfinite(d2)):
                nan_ok = False
                print(f"  [logarithm] NON-FINITE leaked in edge({i},{j})!")
    print(f"  logarithm NaN-guard: {'PASS' if nan_ok else 'FAIL'} (no non-finite leaked)")

    # damping_sin regression (pure spline)
    print("\n\n######## DAMPING_SIN (regression, pure spline) ########")
    md = load('damping_sin')
    print(f"  symbolic_enabled={getattr(md,'symbolic_enabled',None)}")
    x_eval = torch.zeros(50, md.act_fun[0].coef.shape[0], dtype=torch.float64)
    x_eval[:, 0] = torch.linspace(0.1, 0.9, 50, dtype=torch.float64)
    sb = bc._symbolic_branch(md, 0, x_eval, order=2)
    sb_zero = bool(torch.all(sb == 0).item())
    print(f"  _symbolic_branch all-zero (symbolic disabled): {sb_zero}")
    d1, dw = check_verify(md, 'damping_sin')

    print("\n\n######## GATE 3 SUMMARY ########")
    print(f"exponential CHECK1(autograd): {'PASS' if e1 else 'FAIL'} worst={ew:.3e}")
    print(f"exponential CHECK2(value):    {'PASS' if e2 else 'FAIL'}")
    print(f"exponential CHECK3(sym-deriv):{'PASS' if e3 else 'FAIL'}")
    print(f"logarithm   CHECK1(autograd): {'PASS' if l1 else 'FAIL'} worst={lw:.3e}")
    print(f"logarithm   CHECK2(value):    {'PASS' if l2 else 'FAIL'}")
    print(f"logarithm   NaN-guard:        {'PASS' if nan_ok else 'FAIL'}")
    print(f"damping_sin CHECK6(regress):  {'PASS' if (d1 and sb_zero) else 'FAIL'} worst={dw:.3e} sym_branch_zero={sb_zero}")


if __name__ == '__main__':
    main()
