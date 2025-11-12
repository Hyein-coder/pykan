import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_log_sum_function(n_inputs, c_min=10, c_max=500, _device='cpu', seed=None):
    """
    A function factory that creates a target function of the form:
    f(x) = sum(log(c_i * x_i))

    Args:
        n_inputs (int): The number of input features (x_i).
        c_min (float): The minimum value for the random multipliers.
        c_max (float): The maximum value for the random multipliers.
        _device (str): The device to store the multipliers on.

    Returns:
        tuple: (target_function, multipliers)
            - target_function: The new PyTorch function.
            - multipliers: The 1D tensor of random multipliers used.
    """
    generator = torch.Generator(device=_device)
    if seed is not None:
        generator.manual_seed(seed)

    multipliers = c_min + (c_max - c_min) * torch.rand(n_inputs, device=_device)

    def target_function(x):
        """
        Calculates sum(log(c_i * x_i)).

        Args:
            x (torch.Tensor): The input tensor, expected shape [batch_size, n_inputs].
                              All values in x * multipliers must be positive.
        """
        safe_x = x + torch.ones_like(x) * 1.2
        multiplied_x = safe_x * multipliers
        log_x = torch.log(multiplied_x)
        final_sum = torch.sum(log_x, dim=1)
        return final_sum

    return target_function, multipliers

if __name__ == "__main__":
    from kan import create_dataset
    from kan.custom import MultKAN
    from kan.utils import ex_round
    import numpy as np
    import matplotlib.pyplot as plt
    from kan.custom_processing import find_index_sign_revert

    num_inputs = [2, 5, 10, 30, 100]
    for nx in num_inputs:
        f_test, mult_test = create_log_sum_function(nx, _device=device.type, seed=0)
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
                    inflection_val = act.grid[i, spline_radius + find_index_sign_revert(slope)]
                    inflection_points_per_input.append(inflection_val)

        model.forward(dataset['train_input'])
        scores_tot = model.feature_score.detach().cpu().numpy()

        sorted_indices = np.argsort(scores_tot)[::-1]

        mask_idx = None
        mask_inflection_val = None

        for idx in sorted_indices:
            if inflection_points_per_input[idx] is not None:
                mask_idx = idx
                mask_inflection_val = inflection_points_per_input[idx]
                break

        if mask_inflection_val is None:
            res = {
                'num_inputs': nx,
                'input_multipliers': mult_test.tolist(),
                'train_rmse': train_rmse,
                'symbolic_fun': sym_fun,
                'symbolic_res': sym_res,
                'inflection_points': inflection_points_per_input,
                'attribution_score': scores_tot,
                'model': model,
                'mask_idx': mask_idx,
                'mask_interval': None,
                'scores_interval': None,
                'scores_interval_norm': None,
            }

            print(res)
            continue

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
        fig, ax = plt.subplots(figsize=(10, 4))
        xticks = np.arange(len(masks)) * (width * scores_tot.shape[0] * 1.2)
        xticklabels = labels
        max_score = max([max(s) for s in scores_interval_norm])
        for idx in range(scores_tot.shape[0]):
            bars = ax.bar(xticks + idx * width, [s[idx] for s in scores_interval_norm], width, label=f"x{idx}")
            ax.bar_label(bars, fmt='%.2f', fontsize=7, padding=3)
        ax.margins(x=0.1)
        ax.set_ylim(0, max_score * 1.1)

        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=10, ha='center', fontsize=8)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
        ax.set_title("Input-std Normalized Attribution Scores")
        plt.tight_layout()
        plt.show()

        res = {
            'num_inputs': nx,
            'input_multipliers': mult_test.tolist(),
            'train_rmse': train_rmse,
            'symbolic_fun': sym_fun,
            'symbolic_res': sym_res,
            'inflection_points': inflection_points_per_input,
            'attribution_score': scores_tot,
            'model': model,
            'mask_idx': mask_idx,
            'mask_interval': mask_interval,
            'scores_interval': scores_interval,
            'scores_interval_norm': scores_interval_norm,
        }
        print(res)


"""
{'num_inputs': 2, 'input_multipliers': [253.1657257080078, 386.4286804199219], 'train_rmse': [(0.0012998419115319848, 0.001278519630432129)], 'symbolic_fun': 1.1354*log(0.8413*x_1 + 1.0093) + 1.1363*log(1.9744*x_2 + 2.3702) - 0.6313, 'symbolic_res': {'test_loss': 0.00030538582359440625, 'n_edge': 0, 'n_grid': 10}, 'inflection_points': [tensor(0.2023), tensor(-0.0146)], 'attribution_score': array([0.6987343, 0.6877852], dtype=float32), 'model': MultKAN(
  (act_fun): ModuleList(
    (0-1): 2 x KANLayer(
      (base_fun): SiLU()
    )
  )
  (base_fun): SiLU()
  (symbolic_fun): ModuleList(
    (0-1): 2 x Symbolic_KANLayer()
  )
), 'mask_idx': 0, 'mask_interval': [-1, tensor(0.2023), 1], 'scores_interval': [array([0.64850837, 0.7499116 ], dtype=float32), array([0.21404329, 0.9874483 ], dtype=float32)], 'scores_interval_norm': [array([1.8500388, 1.3040453], dtype=float32), array([0.9072888, 1.6907924], dtype=float32)]}

{'num_inputs': 5, 'input_multipliers': [253.1657257080078, 386.4286804199219, 53.35394287109375, 74.69493865966797, 160.63717651367188], 'train_rmse': [(0.0003045095654670149, 0.0003607975959312171)], 'symbolic_fun': 0.7219*log(7.0721*x_1 + 8.4859) + 0.7225*log(5.9978*x_2 + 7.2016) + 0.7218*log(1.0602*x_3 + 1.2722) + 0.722*log(0.9654*x_4 + 1.1586) + 0.7223*log(2.004*x_5 + 2.4052) - 3.303, 'symbolic_res': {'test_loss': 0.00018920896400231868, 'n_edge': 0, 'n_grid': 10}, 'inflection_points': [tensor(0.8271), tensor(0.8091), tensor(0.7621), tensor(0.7855), tensor(0.8031)], 'attribution_score': array([0.4441266 , 0.43716258, 0.43397257, 0.44176757, 0.43965313],
      dtype=float32), 'model': MultKAN(
  (act_fun): ModuleList(
    (0-1): 2 x KANLayer(
      (base_fun): SiLU()
    )
  )
  (base_fun): SiLU()
  (symbolic_fun): ModuleList(
    (0-1): 2 x Symbolic_KANLayer()
  )
), 'mask_idx': 0, 'mask_interval': [-1, tensor(0.8271), 1], 'scores_interval': [array([0.43247557, 0.4392632 , 0.42857692, 0.4419582 , 0.44198564],
      dtype=float32), array([0.02049294, 0.47295654, 0.53238916, 0.4963798 , 0.47403318],
      dtype=float32)], 'scores_interval_norm': [array([0.81975466, 0.75947154, 0.7666199 , 0.7768202 , 0.76793516],
      dtype=float32), array([0.37103143, 0.8017579 , 0.89978486, 0.82335776, 0.84379125],
      dtype=float32)]}

{'num_inputs': 10, 'input_multipliers': [253.1657257080078, 386.4286804199219, 53.35394287109375, 74.69493865966797, 160.63717651367188, 320.69854736328125, 250.14576721191406, 449.2579345703125, 233.25770568847656, 319.830078125], 'train_rmse': [(0.0002921565028373152, 0.0008280923357233405)], 'symbolic_fun': 0.2063*log(3.8212*x_1 + 4.6315) + 0.2989*log(5.2906*x_1 + 6.3129) + 0.2176*log(6.6333*x_10 + 8.2507) + 0.2097*log(4.7107*x_2 + 5.732) + 0.2949*log(7.2594*x_2 + 8.633) + 0.2069*log(1.9259*x_3 + 2.3135) + 0.2975*log(6.8726*x_3 + 8.2396) + 0.2126*log(5.2483*x_4 + 6.4208) + 0.292*log(7.5158*x_4 + 8.9018) + 0.2105*log(3.8542*x_5 + 4.6289) + 0.2935*log(4.1009*x_5 + 4.9135) + 0.2008*log(4.3356*x_6 + 5.1638) + 0.304*log(6.749*x_6 + 8.145) + 0.2952*log(2.049*x_7 + 2.4209) + 0.2101*log(5.17720031738281*x_7 + 6.3725) + 0.2918*log(5.3223*x_8 + 6.3093) + 0.2136*log(7.8807*x_8 + 9.6491) + 0.2898*log(3.0827*x_9 + 3.6594) + 0.2149*log(6.2611*x_9 + 7.6351) - 6.887 - 0.7474/sqrt(0.6599*x_10 + 1), 'symbolic_res': {'test_loss': 0.0038956194184720516, 'n_edge': 0, 'n_grid': 10}, 'inflection_points': [None, tensor(-0.2057), None, tensor(0.1964), None, tensor(0.0066), None, tensor(0.3519), None, tensor(0.1837), None, tensor(0.2302), None, tensor(-0.1555), None, tensor(-0.1559), tensor(0.8127), tensor(0.2092), tensor(0.7751), tensor(0.1832)], 'attribution_score': array([0.3102327 , 0.30532676, 0.30309397, 0.30856878, 0.3069896 ,
       0.30434108, 0.2923477 , 0.30499813, 0.31334662, 0.30875897],
      dtype=float32), 'model': MultKAN(
  (act_fun): ModuleList(
    (0-1): 2 x KANLayer(
      (base_fun): SiLU()
    )
  )
  (base_fun): SiLU()
  (symbolic_fun): ModuleList(
    (0-1): 2 x Symbolic_KANLayer()
  )
), 'mask_idx': 8, 'mask_interval': None, 'scores_interval': None, 'scores_interval_norm': None}

{'num_inputs': 30, 'input_multipliers': [253.1657257080078, 386.4286804199219, 53.35394287109375, 74.69493865966797, 160.63717651367188, 320.69854736328125, 250.14576721191406, 449.2579345703125, 233.25770568847656, 319.830078125, 180.95779418945312, 206.8414764404297, 20.939619064331055, 92.74088287353516, 154.00534057617188, 264.07568359375, 351.85711669921875, 402.0055847167969, 88.90443420410156, 148.3115997314453, 343.9881896972656, 458.4450378417969, 204.57896423339844, 438.3363952636719, 215.51007080078125, 280.9244689941406, 476.8416748046875, 27.720762252807617, 100.7632064819336, 192.97451782226562], 'train_rmse': [(0.00029983778949826956, 0.00042533446685411036)], 'symbolic_fun': 0.3033*log(1.8613*x_1 + 2.2337) + 0.3033*log(3.0582*x_10 + 3.6697) + 0.3032*log(8.0885*x_11 + 9.7048) + 0.3032*log(8.0466*x_12 + 9.6546) + 0.3033*log(0.8617*x_13 + 1.0341) + 0.303*log(0.9403*x_14 + 1.1278) + 0.3033*log(6.9965*x_15 + 8.3962) + 0.3032*log(1.0618*x_16 + 1.274) + 0.3032*log(2.0544*x_17 + 2.465) + 0.3033*log(4.9414*x_18 + 5.929) + 0.3033*log(3.0388*x_19 + 3.6466) + 0.3034*log(3.0498*x_2 + 3.661) + 0.3032*log(4.9706*x_20 + 5.9643) + 0.3032*log(5.8624*x_21 + 7.0338) + 0.3029*log(1.9482*x_22 + 2.3357) + 0.303*log(8.1094*x_23 + 9.7257) + 0.3033*log(4.0535*x_24 + 4.8646) + 0.3031*log(0.9017*x_25 + 1.0815) + 0.3033*log(1.9646*x_26 + 2.3574) + 0.3032*log(0.9569*x_27 + 1.1481) + 0.3035*log(4.8967*x_28 + 5.8788) + 0.3032*log(2.1422*x_29 + 2.5702) + 0.3031*log(6.8762*x_3 + 8.2497) + 0.3031*log(2.9871*x_30 + 3.5834) + 0.3031*log(3.8504*x_4 + 4.6182) + 0.3032*log(2.942*x_5 + 3.5295) + 0.3035*log(5.9042*x_6 + 7.0874) + 0.3035*log(2.9532*x_7 + 3.5449) + 0.3034*log(4.888*x_8 + 5.8668) + 0.3032*log(0.8797*x_9 + 1.0554) - 10.1662, 'symbolic_res': {'test_loss': 0.0002287950919708237, 'n_edge': 0, 'n_grid': 10}, 'inflection_points': [tensor(0.2023), tensor(0.1964), tensor(0.0066), tensor(-0.0141), tensor(0.1837), tensor(0.0309), tensor(0.0218), tensor(0.0429), tensor(0.0196), tensor(0.1832), tensor(0.2503), tensor(0.0125), tensor(0.0157), tensor(0.1839), tensor(0.2154), tensor(-0.0061), tensor(0.0149), tensor(0.0170), tensor(0.5952), tensor(-0.0122), tensor(0.0055), tensor(0.0116), tensor(0.2001), tensor(0.0278), tensor(0.0374), tensor(0.0159), tensor(-0.0179), tensor(0.4108), tensor(0.0363), tensor(0.2082)], 'attribution_score': array([0.18652806, 0.18361627, 0.1822513 , 0.18557389, 0.18466108,
       0.18301588, 0.17577529, 0.18343303, 0.18846156, 0.1854124 ,
       0.18195492, 0.18550985, 0.18114968, 0.18428831, 0.18334416,
       0.17521095, 0.18613125, 0.18309428, 0.1839795 , 0.18338424,
       0.18052115, 0.1803456 , 0.18214935, 0.1785978 , 0.1838443 ,
       0.17582951, 0.17799355, 0.18555327, 0.18401894, 0.18776615],
      dtype=float32), 'model': MultKAN(
  (act_fun): ModuleList(
    (0-1): 2 x KANLayer(
      (base_fun): SiLU()
    )
  )
  (base_fun): SiLU()
  (symbolic_fun): ModuleList(
    (0-1): 2 x Symbolic_KANLayer()
  )
), 'mask_idx': 8, 'mask_interval': [-1, tensor(0.0196), 1], 'scores_interval': [array([0.19363458, 0.18537846, 0.1806315 , 0.1886984 , 0.18735649,
       0.18877764, 0.18297102, 0.18753494, 0.15816312, 0.19700404,
       0.1801212 , 0.18160579, 0.17859723, 0.19022842, 0.1916601 ,
       0.18415669, 0.19050364, 0.18502781, 0.1883244 , 0.18600656,
       0.18278301, 0.18149012, 0.17680448, 0.18284184, 0.18365231,
       0.18086372, 0.1825312 , 0.19175196, 0.19122447, 0.1863681 ],
      dtype=float32), array([0.1826194 , 0.18524367, 0.18701139, 0.1857977 , 0.18534917,
       0.18063241, 0.17175338, 0.18274705, 0.05252036, 0.17674698,
       0.18696262, 0.19259521, 0.18666628, 0.18167189, 0.17778125,
       0.16885798, 0.18522468, 0.18456389, 0.18304136, 0.18418212,
       0.18161707, 0.18252613, 0.19023755, 0.17740351, 0.18733548,
       0.17396758, 0.17676575, 0.18273343, 0.18015702, 0.1923384 ],
      dtype=float32)], 'scores_interval_norm': [array([0.32802543, 0.31748804, 0.32191038, 0.3374777 , 0.32953838,
       0.32179826, 0.314538  , 0.32157284, 0.51158285, 0.32934678,
       0.31976667, 0.31663203, 0.30945107, 0.33150983, 0.3291103 ,
       0.3195221 , 0.32075676, 0.32599196, 0.3232878 , 0.32133588,
       0.32306495, 0.3125858 , 0.32205638, 0.31477293, 0.3168734 ,
       0.3098587 , 0.3209492 , 0.3205942 , 0.3252676 , 0.31620744],
      dtype=float32), array([0.3142171 , 0.32204303, 0.33146667, 0.31748095, 0.3197857 ,
       0.31551945, 0.31253695, 0.32304144, 0.18207102, 0.3169336 ,
       0.322424  , 0.31952012, 0.31923193, 0.31741315, 0.3093018 ,
       0.31189188, 0.3217491 , 0.3208014 , 0.31880704, 0.31539568,
       0.31055543, 0.3154932 , 0.3244972 , 0.32021174, 0.32419932,
       0.31001732, 0.32025367, 0.32024798, 0.3133113 , 0.32333693],
      dtype=float32)]}
"""