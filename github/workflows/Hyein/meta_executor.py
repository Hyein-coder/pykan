import subprocess
import os

def main():
    # target_scripts = [
    #     # "github/workflows/Hyein/material_all_NN_SHAP.py",
    #     # "github/workflows/Hyein/material_KAN_sweep.py",
    #     "github/workflows/Hyein/material_KAN_analyze.py",
    # ]
    # target_data = [
    #     # 'AgNP', 'AutoAM', 'Perovskite',
    #     'CO2RRLCA',
    #     # 'P3HT', 'CrossedBarrel',
    #     # 'CO2RRE', 'CO2RRNPV', 'CO2RRRC', 'CO2RRUR',
    #     # 'CO2HE', 'CO2RRCC', 'CO2RRCA', 'CO2RRRA',
    #     # 'CO2RRMSP', 'CO2RREE', 'CO2RRC', 'CO2RRA', 'CO2HP',
    # ]


    target_scripts = [
        # "github/workflows/Hyein/toy_analytic_SHAP_Sobol.py",
        # "github/workflows/Hyein/toy_NN_tuning.py",
        # "github/workflows/Hyein/toy_NN_SHAP_Sobol.py",
        "github/workflows/Hyein/toy_KAN_sweep.py",
        # "github/workflows/Hyein/toy_KAN_analyze.py",
    ]
    target_data = [
        # 'convolution',
        # 'original',
        # 'mult_periodic', 'exponential', 'logarithm',
        # 'log_sum_2d', 'log_sum_5d', 'log_sum_10d', 'log_sum_30d',
    ] + [f'convex_seed_{i}' for i in range(0)]

    for s in target_scripts:
        for data in target_data:
            subprocess.run(['python', s, data])

if __name__ == '__main__':
    main()
