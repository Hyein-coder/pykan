import subprocess
import os

def main():
    target_scripts = [
        "github/workflows/Hyein/material_all_NN_SHAP.py",
    ]
    target_data = [
        'AgNP', 'P3HT', 'AutoAM', 'Perovskite', 'CrossedBarrel'
    ]
    for s in target_scripts:
        for data in target_data:
            subprocess.run(['python', s, data])

if __name__ == '__main__':
    main()
