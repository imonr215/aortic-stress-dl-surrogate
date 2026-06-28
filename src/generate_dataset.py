"""
generate_dataset.py

Builds the full synthetic cohort: N patient-specific aTAA geometries, each with
its FEA-like ground-truth wall-stress field, and saves them as a single .npz.

N is set to 169 to match the abstract's clinical cohort size. The real data
came from ECG-gated CT; here each "patient" is a sampled synthetic geometry.

Run:
    python src/generate_dataset.py
Output:
    data/aorta_dataset.npz
"""

import sys

import numpy as np

sys.path.insert(0, "src")
from geometry import N_AXIAL, N_THETA, build_aorta, sample_params
from biomechanics import compute_stress

RANDOM_SEED = 7
N_PATIENTS = 169


def main():
    rng = np.random.default_rng(RANDOM_SEED)

    coords = np.empty((N_PATIENTS, N_AXIAL * N_THETA, 3), dtype=np.float32)
    stress = np.empty((N_PATIENTS, N_AXIAL * N_THETA, 2), dtype=np.float32)

    for i in range(N_PATIENTS):
        p = sample_params(rng)
        geom = build_aorta(p)
        sig = compute_stress(geom, rng)
        coords[i] = geom["coords"]
        stress[i] = sig

    np.savez_compressed(
        "data/aorta_dataset.npz",
        coords=coords,
        stress=stress,
        grid=np.array([N_AXIAL, N_THETA]),
    )

    circ = stress[:, :, 0]
    lon = stress[:, :, 1]
    print(f"Wrote data/aorta_dataset.npz")
    print(f"  patients: {N_PATIENTS}")
    print(f"  nodes/patient: {N_AXIAL * N_THETA} ({N_AXIAL} axial x {N_THETA} circumferential)")
    print(f"  mean circumferential stress: {circ.mean() / 1000:.1f} kPa")
    print(f"  mean longitudinal stress:    {lon.mean() / 1000:.1f} kPa")


if __name__ == "__main__":
    main()
