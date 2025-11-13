import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_log_sum_function(n_inputs, m_min=1, m_max=5, _device='cpu', seed=None):
    """
    A function factory that creates a target function of the form:
    f(x) = sum(log(c_i * x_i))

    Args:
        n_inputs (int): The number of input features (x_i).
        m_min (float): The minimum value for the random multipliers.
        m_max (float): The maximum value for the random multipliers.
        _device (str): The device to store the multipliers on.

    Returns:
        tuple: (target_function, multipliers)
            - target_function: The new PyTorch function.
            - multipliers: The 1D tensor of random multipliers used.
    """
    generator = torch.Generator(device=_device)
    if seed is not None:
        generator.manual_seed(seed)

    multipliers = m_min + (m_max - m_min) * torch.rand(n_inputs, device=_device)

    def target_function(x):
        """
        Calculates sum(log(c_i * x_i)).

        Args:
            x (torch.Tensor): The input tensor, expected shape [batch_size, n_inputs].
                              All values in x * multipliers must be positive.
        """
        safe_x = x + torch.ones_like(x) * (1 + 1e-3)
        multiplied_x = safe_x * (5**multipliers)
        log_x = torch.log(multiplied_x + 100.)
        final_sum = torch.sum(log_x, dim=1)
        return final_sum

    return target_function, multipliers

if __name__ == "__main__":
    from kan import create_dataset
    from kan.custom import MultKAN
    from kan.utils import ex_round
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from kan.custom_processing import find_index_sign_revert

    import json
    import os
    import datetime

    # num_inputs = [50]
    num_inputs = [2, 5, 10, 30, 50]
    save_heading = os.path.join(os.getcwd(), 'github', 'workflows', 'Hyein', 'multvariable',
                                "toy_7_multiple_sweep_" + datetime.datetime.now().strftime('%Y%m%d_%H%M'))

    class NumpyJSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if obj is None:
                return None
            return super(NumpyJSONEncoder, self).default(obj)
    # ---

    models = []
    dummy_data = np.linspace(-1, 1, 100)
    for nx in num_inputs:
        f_test, mult_test = create_log_sum_function(nx, _device=device.type, seed=0)
        digits = int(np.ceil(nx / 10))
        sorted_multiplier_indices = np.argsort(mult_test.detach().cpu().numpy())[::-1].tolist()

        cmap = cm.get_cmap('viridis')
        for i in sorted_multiplier_indices:
            input_i = torch.Tensor(
                [np.zeros_like(dummy_data)] * i + [dummy_data] + [np.zeros_like(dummy_data)] * (nx - i - 1)).T
            f_vary = f_test(input_i)
            plt.plot(dummy_data, f_vary, label=str(i), color=cmap(i/(nx-1)))
        plt.grid()
        plt.savefig(save_heading + f'_nx{nx:02d}_vary_inputs.png')
        plt.show()

        dataset = create_dataset(f_test, n_var=nx, train_num=1000, test_num=100, device=device, normalize_label=True)

        # grids_to_sym = [3, 5, 10, 20]
        grids_to_sym = [10]

        train_rmse = []

        model = MultKAN(width=[nx, nx, 1], grid=3, k=3, seed=0, device=device)
        model.fit(dataset, opt='LBFGS', steps=20,
                  lamb=0.01, lamb_entropy=0.1, lamb_coef=0.1, lamb_coefdiff=0.5)
        model = model.prune(edge_th=0.03, node_th=0.01)

        for i in range(len(grids_to_sym)):
            model = model.refine(grids_to_sym[i])
            results = model.fit(dataset, opt='LBFGS', steps=50, stop_grid_update_step=20)
            train_rmse.append((results['train_loss'][-1].item(), results['test_loss'][-1].item()))

        model.auto_symbolic()
        sym_fun = ex_round(model.symbolic_formula()[0][0], 4)
        sym_res = model.evaluate(dataset)

        l = 0
        act = model.act_fun[l]
        ni, no = act.coef.shape[:2]
        coef = act.coef.tolist()
        inflection_points_per_input = []

        for i in range(ni):
            for j in range(no):
                coef_node = coef[i][j]
                num_knot = act.grid.shape[1]
                spline_radius = int((num_knot - len(coef_node)) / 2)

                slope = [x - y for x, y in zip(coef_node[1:], coef_node[:-1])]
                slope_2nd = [(x - y) * 10 for x, y in zip(slope[1:], slope[:-1])]

                idx_sign_revert = find_index_sign_revert(slope)
                if idx_sign_revert is None:
                    inflection_points_per_input.append(None)
                else:
                    inflection_val = act.grid[i, spline_radius + find_index_sign_revert(slope)].item()
                    inflection_points_per_input.append(inflection_val)

        model.forward(dataset['train_input'])
        scores_tot = model.feature_score.detach().cpu().numpy()
        input_std_tot = dataset['train_input'].std(dim=0).detach().cpu().numpy()
        scores_tot_norm = scores_tot / input_std_tot

        sorted_indices = np.argsort(scores_tot)[::-1]

        mask_idx = None
        mask_inflection_val = None

        for idx in sorted_indices:
            if inflection_points_per_input[idx] is not None:
                mask_idx = idx
                mask_inflection_val = inflection_points_per_input[idx]
                break

        if mask_inflection_val is None:
            mask_interval = None
            scores_interval = None
            scores_interval_norm = None
        else:
            mask_interval = [-1, mask_inflection_val, 1]

            x_mask = dataset['train_input'][:, mask_idx]
            y_vals = dataset['train_label'].ravel()

            masks = [((x_mask > lb) & (x_mask <= ub)) for lb, ub in zip(mask_interval[:-1], mask_interval[1:])]
            labels = [f'x{mask_idx} <= {ub:.2f}' for lb, ub in zip(mask_interval[:-1], mask_interval[1:])]
            print([sum(x) for x in masks])

            scores_interval = []
            scores_interval_norm = []
            for mask in masks:
                if np.any(mask.numpy()):
                    x_tensor_masked = dataset['train_input'][mask, :]
                    x_std = torch.std(x_tensor_masked, dim=0).detach().cpu().numpy()
                    model.forward(x_tensor_masked)

                    score_masked = model.feature_score.detach().cpu().numpy()
                    score_norm = score_masked / x_std
                    scores_interval.append(score_masked)
                    scores_interval_norm.append(score_norm)
                else:
                    scores_interval.append(np.zeros(scores_tot.shape))
                    scores_interval_norm.append(np.zeros(scores_tot.shape))

            width = 0.25

            fig, ax = plt.subplots(figsize=(8 * digits, 3 * digits))
            xticks = np.arange(len(masks)+1) * (width * scores_tot.shape[0] * 1.2)
            xticklabels = ['Total'] + labels
            max_score = max([max(s) for s in scores_interval_norm])
            for idx in range(scores_tot.shape[0]):
                bars = ax.bar(xticks + idx * width, [scores_tot_norm[idx]] + [s[idx] for s in scores_interval_norm], width, label=f"x{idx}")
                ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=3)
            ax.margins(x=0.1)
            ax.set_ylim(0, max_score * 1.1)

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=10, ha='center', fontsize=8)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, ncol=digits)
            ax.set_title("Input-std Normalized Attribution Scores")
            plt.tight_layout(rect=[0, 0, 1 - .01 * digits, 1])
            plt.savefig(save_heading + f'_nx{nx:02d}_scores_L0_interval_normalized.png')
            plt.show()

            fig, ax = plt.subplots(figsize=(6 * digits, 3 * digits))
            max_score = max([max(s) for s in scores_interval])
            for idx in range(scores_tot.shape[0]):
                bars = ax.bar(xticks + idx * width, [scores_tot[idx]] + [s[idx] for s in scores_interval], width, label=f"x{idx}")
                ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=3)
            ax.margins(x=0.1)
            ax.set_ylim(0, max_score * 1.1)

            ax.set_xticks(xticks)
            ax.set_xticklabels(xticklabels, rotation=10, ha='center', fontsize=8)
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, ncol=digits)
            ax.set_title("Attribution Scores")
            plt.tight_layout(rect=[0, 0, 1 - .01 * digits, 1])
            plt.savefig(save_heading + f'_nx{nx:02d}_scores_L0_interval.png')
            plt.show()

            fig, ax_tot = plt.subplots(figsize=(6 * digits, 3 * digits))
            xticks_tot = np.arange(2) * (width * scores_tot.shape[0] * 1.2)
            xticklabels_tot = ["Original", "Std-Normalized"]
            scores_tot_combined = [scores_tot, scores_tot_norm]
            max_score = max([max(s) for s in scores_tot_combined])
            for idx in range(scores_tot.shape[0]):
                bars = ax_tot.bar(xticks_tot + idx * width, [s[idx] for s in scores_tot_combined], width, label=f"x{idx}")
                ax_tot.bar_label(bars, fmt='%.2f', fontsize=7, padding=3)
            ax_tot.margins(x=0.1)
            ax_tot.set_ylim(0, max_score * 1.1)

            ax_tot.set_xticks(xticks_tot)
            ax_tot.set_xticklabels(xticklabels_tot, rotation=10, ha='center', fontsize=8)
            ax_tot.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8, ncol=digits)
            ax_tot.set_title("Attribution Scores")
            plt.tight_layout(rect=[0, 0, 1 - .01 * digits, 1])
            plt.savefig(save_heading + f'_nx{nx:02d}_scores_total.png')
            plt.show()

        res = {
            'num_inputs': nx,
            'input_multipliers': mult_test.tolist(),
            'train_rmse': train_rmse,
            'symbolic_fun': str(sym_fun),
            'symbolic_res': sym_res,
            'inflection_points': inflection_points_per_input,
            'attribution_score': scores_tot,
            'attribution_score_norm': scores_tot_norm,
            'first_rank_idx': sorted_indices[0],
            'mask_idx': mask_idx,
            'mask_inflection_val': mask_inflection_val,
            'scores_interval': scores_interval,
            'scores_interval_norm': scores_interval_norm,
        }
        models.append(model)
        with open(save_heading + f'_nx{nx:02d}_res.json', 'w') as f:
            json.dump(res, f, cls=NumpyJSONEncoder)
        torch.save(model.state_dict(), f"{save_heading}_nx{nx:02d}_model.pt")

#%%
    summary = {k: [] for k in res.keys()}
    for nx in num_inputs:
        fn = f"{save_heading}_nx{nx:02d}_res.json"
        with open(fn, 'r') as f:
            res_loaded = json.load(f)
        for k in summary.keys():
            summary[k].append(res_loaded[k])

    max_length = max(summary['num_inputs'])
    mat_multipliers = np.array([m + [0 for _ in range(max_length - len(m))] for m in summary['input_multipliers']]).transpose()
    mat_height, mat_width = mat_multipliers.shape
    extent = [-0.5, mat_width - 0.5, -0.5, mat_height - 0.5]

    fig, axs = plt.subplots(1, 2, figsize=(10, 8))
    ax = axs[0]
    im = ax.imshow(
        mat_multipliers,
        cmap='Blues',         # Colormap for the heatmap
        aspect='auto',        # Allow non-square pixels
        origin='lower',       # Put (0,0) in the bottom-left corner
        extent=extent
    )
    fig.colorbar(im, ax=ax, label='Heatmap Value', pad=0.08)

    ax.set_xticks(np.arange(mat_width))
    ax.set_yticks(summary['num_inputs'])
    ax.set_ylabel('Input Index')
    ax.set_xlabel('Number of Inputs')

    ax.step(
        np.arange(mat_width+1) - 0.5, summary['first_rank_idx'] + summary['first_rank_idx'][-1::],
        where='post', color='red', linewidth=2, label='First Rank Index')
    ax.legend(frameon=False, loc='upper left')

    ax_first_rank = axs[1]
    first_rank_multipliers = [m[i] for m, i in zip(summary['input_multipliers'], summary['first_rank_idx'])]
    reversed_multipliers = [m[i] for m, i in zip(summary['input_multipliers'], summary['mask_idx'])]
    max_multipliers = [max(m) for m in summary['input_multipliers']]
    min_multipliers = [min(m) for m in summary['input_multipliers']]

    ax_first_rank.step(
        np.arange(mat_width + 1) - 0.5, max_multipliers + max_multipliers[-1::],
        where='post', linewidth=2, label='Maximum Multiplier')
    ax_first_rank.step(
        np.arange(mat_width + 1) - 0.5, min_multipliers + min_multipliers[-1::],
        where='post', linewidth=2, label='Minimum Multiplier')
    ax_first_rank.step(
        np.arange(mat_width + 1) - 0.5, first_rank_multipliers + first_rank_multipliers[-1::],
        where='post', linewidth=2, color='r', label='First Rank Multiplier')
    ax_first_rank.step(
        np.arange(mat_width + 1) - 0.5, reversed_multipliers + reversed_multipliers[-1::],
        where='post', linewidth=2, color='k', label='Importance Reversed Multiplier')
    ax_first_rank.set_ylabel('Multiplier')
    ax_first_rank.set_xlabel('Number of Inputs')
    ax_first_rank.legend(frameon=False, loc='upper left')

    plt.savefig(save_heading + '_summary_multipliers.png')
    plt.show()

