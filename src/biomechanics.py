"""
biomechanics.py

Computes a physically-grounded, FEA-like ground-truth wall-stress field for a
synthetic aorta. Real finite-element analysis (the ground truth in the abstract,
run in LS-DYNA with fiber-embedded material models) solves the full nonlinear
mechanics; here we use a transparent closed-form approximation so the synthetic
"ground truth" is principled rather than arbitrary, and so the geometry->stress
relationship the surrogate learns is real.

The basis is Laplace's law for a pressurized thin-walled vessel:

    circumferential (hoop) stress   sigma_theta ≈ P * r / t
    longitudinal (axial) stress     sigma_z     ≈ P * r / (2 t)

so hoop stress is ~2x longitudinal — which matches the abstract's reported
means (175 vs 95 kPa). On top of this we add a stress-concentration factor at
the aneurysm bulge (where the wall is thinner and more sharply curved), a
direction-specific weighting, a small smooth per-patient material-variation
field, and minor per-node noise.

Units are SI internally (pressure in Pa, length in m → stress in Pa); results
are reported in kPa elsewhere.
"""

import numpy as np

SYSTOLIC_PRESSURE = 13_000.0   # Pa (calibrated so dataset-mean stresses match the abstract)

# Direction-specific concentration weights, tuned so the dataset-average
# stresses land near the abstract's values (circ ~175 kPa, long ~95 kPa) and
# the bulge produces a realistic stress hot-spot.
_CIRC_CURV_GAIN = 2.0e-3
_LONG_CURV_GAIN = 1.4e-3
_LONG_BASE_FACTOR = 0.56       # longitudinal/hoop base ratio (gives circ/long ~1.8, per abstract)


def compute_stress(geom, rng, material_variation=0.0):
    """
    Parameters
    ----------
    geom : dict from geometry.build_aorta (needs radius, thickness, axial_curv, grid)
    rng  : numpy Generator, for the (small) node-level noise
    material_variation : std of an optional smooth per-patient multiplicative
        field. Left at 0 by default: real FEA stress is a deterministic function
        of geometry under a fixed material model, so the surrogate's target
        should be too (a geometry-independent random field would be impossible
        to learn and would only inflate the error floor).

    Returns
    -------
    stress : (n_nodes, 2) array, columns = [circumferential, longitudinal], in Pa
    """
    r = geom["radius"]
    t = geom["thickness"]
    kappa = geom["axial_curv"]
    n_axial, n_theta = geom["grid"]

    # Base Laplace stresses.
    hoop_base = SYSTOLIC_PRESSURE * r / t
    long_base = _LONG_BASE_FACTOR * SYSTOLIC_PRESSURE * r / t

    # Stress concentration at the bulge: curvature of the wall meridian raises
    # stress, more strongly in the circumferential direction.
    circ_conc = 1.0 + _CIRC_CURV_GAIN * kappa / (t + 1e-6)
    long_conc = 1.0 + _LONG_CURV_GAIN * kappa / (t + 1e-6)

    sigma_circ = hoop_base * circ_conc
    sigma_long = long_base * long_conc

    # Optional smooth per-patient field (off by default; see docstring).
    if material_variation > 0:
        field = _smooth_field(n_axial, n_theta, rng, scale=material_variation)
        sigma_circ = sigma_circ * (1.0 + field)
        sigma_long = sigma_long * (1.0 + 0.8 * field)

    # Small high-frequency measurement-like noise (keeps the fit non-trivial).
    sigma_circ += rng.normal(0, 0.004 * sigma_circ.mean(), size=sigma_circ.shape)
    sigma_long += rng.normal(0, 0.004 * sigma_long.mean(), size=sigma_long.shape)

    return np.stack([sigma_circ, sigma_long], axis=1)


def _smooth_field(n_axial, n_theta, rng, scale):
    """A low-frequency random field over the (axial, theta) grid, flattened."""
    # Build from a few low-order Fourier modes so it varies smoothly.
    s = np.linspace(0, 1, n_axial)[:, None]
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)[None, :]
    field = np.zeros((n_axial, n_theta))
    for k_s in range(1, 4):
        for k_th in range(0, 3):
            amp = rng.normal(0, 1.0)
            phase = rng.uniform(0, 2 * np.pi)
            field += amp * np.sin(np.pi * k_s * s + phase) * np.cos(k_th * th)
    field /= np.abs(field).max() + 1e-9
    return (scale * field).reshape(-1)
