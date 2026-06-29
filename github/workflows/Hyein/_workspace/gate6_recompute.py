"""Independent recompute of GATE 6 numbers — does NOT import sweep_rel_threshold."""
import os
import numpy as np
import yaml

def _tuple_constructor(loader, node):
    return tuple(loader.construct_sequence(node))
yaml.add_constructor('tag:yaml.org,2002:python/tuple', _tuple_constructor, Loader=yaml.SafeLoader)
try:
    yaml.add_constructor('tag:yaml.org,2002:python/tuple', _tuple_constructor, Loader=yaml.Loader)
except AttributeError:
    pass

from github.workflows.Hyein.toy_KAN_sweep import KANRegressor
from github.workflows.Hyein.bspline_curvature import find_ranking_transitions

RESULTS_ROOT = os.path.join('github', 'workflows', 'Hyein', 'analytical_results')
name = 'conditional'
savepath = os.path.join(RESULTS_ROOT, name, 'kan_models')
ckpt = os.path.join(savepath, f'{name}_best_kan_model')

mw = KANRegressor(device='cpu')
mw.load_model(ckpt)
model = mw.model

TRUE_KINK = 0.5
FEAT = 0

for rt in [0.10, 0.20]:
    transitions, info = find_ranking_transitions(model, rel_thresh=rt, layer=0)
    down = [t['point'] for t in transitions
            if t['feat_idx'] == FEAT and t['direction'] == 'down']
    up = [t['point'] for t in transitions
          if t['feat_idx'] == FEAT and t['direction'] == 'up']
    n_down = len(down)
    if n_down:
        loc_err = float(np.min(np.abs(np.array(down) - TRUE_KINK)))
    else:
        loc_err = float('nan')
    print(f"rel_thresh={rt:.2f}  tau={info['tau']:.6f}  n_down={n_down}  n_up={len(up)}  "
          f"down_pts={[round(d,4) for d in down]}  loc_err={loc_err:.6f}")
