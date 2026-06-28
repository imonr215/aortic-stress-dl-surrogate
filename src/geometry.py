"""
geometry.py

Generates synthetic patient-specific ascending thoracic aortic aneurysm (aTAA)
geometries as structured surface meshes.

Each aorta is a curved tube parameterized on a (theta, s) grid:
  - s in [0, 1]  : position along the vessel (axial)
  - theta in [0, 2*pi) : angle around the circumference

The vessel has a gently curved centerline, a baseline radius that tapers along
its length, and a Gaussian "bulge" representing the aneurysm. The wall thins
slightly at the bulge apex, which is the physical reason wall stress
concentrates there (a thinner, more sharply curved wall under the same pressure
carries higher stress).

The real study used hexahedral meshes with 9648 nodes / 6336 solid elements
derived from ECG-gated CT. Here we use a comparable-resolution structured
surface mesh; the relative geometry is what the surrogate model learns from.

Returns, per patient, the node coordinates plus the local geometric quantities
(radius, wall thickness, axial wall curvature) that biomechanics.py turns into
a ground-truth stress field.
"""

from dataclasses import dataclass

import numpy as np

N_THETA = 48     # nodes around the circumference
N_AXIAL = 100    # nodes along the vessel
VESSEL_LENGTH = 0.10   # metres (~100 mm ascending aorta segment)
WALL_THICKNESS = 0.0020  # metres (2 mm), uniform across the cohort


@dataclass
class AortaParams:
    r_prox: float       # proximal baseline radius (m)
    r_dist: float       # distal baseline radius (m)
    bulge_amp: float    # aneurysm bulge amplitude (m)
    bulge_pos: float    # aneurysm centre, s in [0,1]
    bulge_width: float  # aneurysm width (in s units)
    wall_thick: float   # nominal wall thickness (m)
    wall_thin: float    # fractional wall thinning at bulge apex (0-1)
    curvature: float    # total centerline turn angle (radians)


def sample_params(rng) -> AortaParams:
    """Draw a plausible aTAA geometry. Radii are aneurysm-scale (dilated).

    Wall thickness is held constant across patients (WALL_THICKNESS): the
    surrogate sees only the surface geometry, so — as in the deep-learning FEA
    surrogate this reproduces — stress must be driven by shape (radius and wall
    curvature), not by a hidden per-patient thickness the model can't observe.
    """
    return AortaParams(
        r_prox=rng.uniform(0.018, 0.024),
        r_dist=rng.uniform(0.016, 0.022),
        bulge_amp=rng.uniform(0.004, 0.012),
        bulge_pos=rng.uniform(0.40, 0.60),
        bulge_width=rng.uniform(0.10, 0.18),
        wall_thick=WALL_THICKNESS,   # uniform across the cohort
        wall_thin=0.0,               # no hidden thickness variation
        curvature=rng.uniform(0.45, 0.75),
    )


def _radius_profile(s, p: AortaParams):
    taper = p.r_prox + (p.r_dist - p.r_prox) * s
    bulge = p.bulge_amp * np.exp(-((s - p.bulge_pos) / p.bulge_width) ** 2)
    return taper + bulge


def _thickness_profile(s, p: AortaParams):
    thinning = 1.0 - p.wall_thin * np.exp(-((s - p.bulge_pos) / p.bulge_width) ** 2)
    return p.wall_thick * thinning


def build_aorta(p: AortaParams):
    """
    Construct one aorta surface mesh from parameters.

    Returns a dict with:
      coords      : (N_AXIAL*N_THETA, 3) node positions (m)
      radius      : (N_AXIAL*N_THETA,)   local radius at each node (m)
      thickness   : (N_AXIAL*N_THETA,)   local wall thickness at each node (m)
      axial_curv  : (N_AXIAL*N_THETA,)   |d^2 r / d s^2|, a stress-concentration proxy
      grid        : (N_AXIAL, N_THETA)   shape for reshaping flattened arrays
    """
    s = np.linspace(0.0, 1.0, N_AXIAL)
    theta = np.linspace(0.0, 2 * np.pi, N_THETA, endpoint=False)

    r_s = _radius_profile(s, p)            # (N_AXIAL,)
    t_s = _thickness_profile(s, p)         # (N_AXIAL,)

    # Axial curvature proxy: second derivative of the radius profile.
    ds = s[1] - s[0]
    d2r = np.gradient(np.gradient(r_s, ds), ds)
    axial_curv_s = np.abs(d2r)

    # Centerline: planar arc in the x-z plane, tangent angle sweeps with s.
    alpha = p.curvature * (s - 0.5)                      # tangent angle (N_AXIAL,)
    tangent = np.stack([np.sin(alpha), np.zeros_like(alpha), np.cos(alpha)], axis=1)
    seg_len = (VESSEL_LENGTH / (N_AXIAL - 1))
    centerline = np.cumsum(tangent * seg_len, axis=0)
    centerline -= centerline.mean(axis=0)               # centre at origin

    # Local frame: N1 in-plane normal, N2 out-of-plane (constant y).
    n1 = np.stack([np.cos(alpha), np.zeros_like(alpha), -np.sin(alpha)], axis=1)
    n2 = np.tile(np.array([0.0, 1.0, 0.0]), (N_AXIAL, 1))

    # Build the surface: for each axial station, a ring of N_THETA nodes.
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    coords = np.empty((N_AXIAL, N_THETA, 3))
    for i in range(N_AXIAL):
        ring = (r_s[i] * (np.outer(cos_t, n1[i]) + np.outer(sin_t, n2[i])))
        coords[i] = centerline[i] + ring

    # Broadcast per-axial-station scalars across the circumference.
    radius = np.repeat(r_s[:, None], N_THETA, axis=1)
    thickness = np.repeat(t_s[:, None], N_THETA, axis=1)
    axial_curv = np.repeat(axial_curv_s[:, None], N_THETA, axis=1)

    return {
        "coords": coords.reshape(-1, 3),
        "radius": radius.reshape(-1),
        "thickness": thickness.reshape(-1),
        "axial_curv": axial_curv.reshape(-1),
        "grid": (N_AXIAL, N_THETA),
    }
