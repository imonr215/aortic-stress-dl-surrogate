"""
run_all.py

Runs the full pipeline end to end:
  1. Generate the synthetic aTAA cohort (geometries + FEA-like stress fields)
  2. Train and cross-validate the deep-learning surrogate (PCA + MLP), reporting
     MAE and peak-stress error (APE) per direction
  3. Render the 3D stress-field comparison and the peak-stress parity plot

Run:
    python run_all.py
"""

import subprocess
import sys


def run(script):
    print(f"\n{'#' * 64}\n# {script}\n{'#' * 64}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        sys.exit(result.returncode)


if __name__ == "__main__":
    run("src/generate_dataset.py")
    run("src/surrogate.py")
    run("src/visualize.py")
    print("\nPipeline complete. See results/ for metrics and figures/ for plots.")
