import subprocess
import os

def main():
    target_scripts = [
        # "github/workflows/Hyein/material_all_NN_SHAP.py",
        "github/workflows/Hyein/material_KAN_sweep.py",
        # "github/workflows/Hyein/material_KAN_analyze.py",
    ]
    target_data = [
        # 'AgNP', 'P3HT', 'AutoAM', 'Perovskite',
        # 'CrossedBarrel',
        'CO2HE', 'CO2HP',
        'CO2RRUR', 'CO2RRRC', 'CO2RRRA',
        'CO2RRNPV', 'CO2RRMSP', 'CO2RRLCA',
        'CO2RREE', 'CO2RRE', 'CO2RRCC', 'CO2RRCA', 'CO2RRC', 'CO2RRA',
    ]


    # target_scripts = [
    #     # "github/workflows/Hyein/toy_analytic_SHAP_Sobol.py",
    #     # "github/workflows/Hyein/toy_NN_tuning.py",
    #     # "github/workflows/Hyein/toy_NN_SHAP_Sobol.py",
    #     "github/workflows/Hyein/toy_KAN_sweep.py",
    #     # "github/workflows/Hyein/toy_KAN_analyze.py",
    # ]
    # target_data = [
    #     'convolution',
    #     'original',
    #     'mult_periodic', 'exponential', 'logarithm',
    #     'log_sum_2d', 'log_sum_5d', 'log_sum_10d', 'log_sum_30d',
    # ]

    for s in target_scripts:
        for data in target_data:
            subprocess.run(['python', s, data])

if __name__ == '__main__':
    main()
