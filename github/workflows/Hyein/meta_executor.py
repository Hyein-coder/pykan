import subprocess
import os

def main():
    target_scripts = [
        "github/workflows/Hyein/material_04_tuning.py",
        "github/workflows/Hyein/material_05_tuning.py",
    ]
    for s in target_scripts:
        subprocess.run(['python', s])

if __name__ == '__main__':
    main()
