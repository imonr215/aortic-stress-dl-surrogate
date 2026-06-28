"""
visualize.py

Two figures that tell the surrogate's story:

  1. fig_stress_field_3d.png — one held-out patient's aorta rendered three ways:
     the FEA ground-truth circumferential stress, the deep-learning surrogate's
     prediction, and the absolute error. The point is that the DL field is
     visually indistinguishable from FEA, with error concentrated only at the
     aneurysm's high-stress apex.

  2. fig_peak_stress_parity.png — predicted vs FEA peak stress across all
     held-out patients (one per CV fold's cached example is not enough, so this
     re-runs a quick hold-out), the standard way to show a stress surrogate
     tracks the clinically important peak.

Run (after surrogate.py):
    python src/visualize.py
Outputs:
    figures/fig_stress_field_3d.png
    figures/fig_peak_stress_parity.png
"""

import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

sys.path.insert(0, "src")

EXAMPLE_PATH = "results/example_prediction.npz"
DATA_PATH = "data/aorta_dataset.npz"
FIG_DIR = "figures"


def _surface_panel(fig, pos, coords, values, grid, title, vmin, vmax, cmap="viridis"):
    n_axial, n_theta = grid
    X = coords[:, 0].reshape(n_axial, n_theta)
    Y = coords[:, 1].reshape(n_axial, n_theta)
    Z = coords[:, 2].reshape(n_axial, n_theta)
    C = values.reshape(n_axial, n_theta)

    # Wrap the circumferential seam so the tube closes visually.
    X = np.concatenate([X, X[:, :1]], axis=1)
    Y = np.concatenate([Y, Y[:, :1]], axis=1)
    Z = np.concatenate([Z, Z[:, :1]], axis=1)
    C = np.concatenate([C, C[:, :1]], axis=1)

    ax = fig.add_subplot(pos, projection="3d")
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    facecolors = plt.get_cmap(cmap)(norm(C))
    ax.plot_surface(X * 1000, Y * 1000, Z * 1000, facecolors=facecolors,
                    rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.set_title(title, fontsize=10, pad=0)
    ax.set_box_aspect((np.ptp(X), np.ptp(Y), np.ptp(Z)))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.grid(False)
    # Clean panes
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_alpha(0.0)
    return norm


def plot_stress_field_3d(out_path):
    d = np.load(EXAMPLE_PATH)
    coords = d["coords"]
    circ_true, circ_pred = d["circ_true"], d["circ_pred"]
    grid = tuple(d["grid"])

    err = np.abs(circ_true - circ_pred)
    vmin = min(circ_true.min(), circ_pred.min())
    vmax = max(circ_true.max(), circ_pred.max())

    fig = plt.figure(figsize=(13, 4.6))
    _surface_panel(fig, 131, coords, circ_true, grid,
                   "FEA ground truth\n(circumferential stress)", vmin, vmax)
    _surface_panel(fig, 132, coords, circ_pred, grid,
                   "Deep-learning surrogate\n(prediction)", vmin, vmax)
    norm_err = _surface_panel(fig, 133, coords, err, grid,
                              "Absolute error", 0, vmax - vmin, cmap="magma")

    # Shared colorbars
    sm = cm.ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap="viridis")
    cbar = fig.colorbar(sm, ax=fig.axes[:2], shrink=0.6, pad=0.02)
    cbar.set_label("Stress (kPa)")
    sm_e = cm.ScalarMappable(norm=plt.Normalize(0, vmax - vmin), cmap="magma")
    cbar_e = fig.colorbar(sm_e, ax=fig.axes[2], shrink=0.6, pad=0.02)
    cbar_e.set_label("|error| (kPa)")

    fig.suptitle("Aortic wall stress: FEA vs. deep-learning surrogate (held-out patient)",
                 fontsize=12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_peak_parity(out_path):
    """Quick hold-out: train on 80%, predict peak stress on 20%, plot parity."""
    from sklearn.decomposition import PCA
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import warnings

    d = np.load(DATA_PATH)
    coords = d["coords"]
    n = coords.shape[0]
    X = coords.reshape(n, -1)
    Y_circ = d["stress"][:, :, 0] / 1000.0
    Y_long = d["stress"][:, :, 1] / 1000.0

    tr, te = train_test_split(np.arange(n), test_size=0.25, random_state=1)

    def predict_peaks(Y):
        sh_scaler = StandardScaler().fit(X[tr])
        sh_pca = PCA(n_components=20, random_state=1)
        c_tr = sh_pca.fit_transform(sh_scaler.transform(X[tr]))
        c_te = sh_pca.transform(sh_scaler.transform(X[te]))
        y_mean = Y[tr].mean(axis=0, keepdims=True)
        st_pca = PCA(n_components=20, random_state=1)
        sc_tr = st_pca.fit_transform(Y[tr] - y_mean)
        cs = StandardScaler().fit(c_tr)
        mlp = MLPRegressor(hidden_layer_sizes=(100, 100), alpha=3e-3,
                           max_iter=2500, tol=1e-6, random_state=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp.fit(cs.transform(c_tr), sc_tr)
        pred = st_pca.inverse_transform(mlp.predict(cs.transform(c_te))) + y_mean
        return Y[te].max(axis=1), pred.max(axis=1)

    circ_true_pk, circ_pred_pk = predict_peaks(Y_circ)
    long_true_pk, long_pred_pk = predict_peaks(Y_long)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(circ_true_pk, circ_pred_pk, c="#B23A48", s=30, alpha=0.8,
               label="Circumferential", edgecolor="white", linewidth=0.5)
    ax.scatter(long_true_pk, long_pred_pk, c="#3F6FA8", s=30, alpha=0.8,
               label="Longitudinal", edgecolor="white", linewidth=0.5)
    lo = min(long_true_pk.min(), long_pred_pk.min()) - 10
    hi = max(circ_true_pk.max(), circ_pred_pk.max()) + 10
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="Perfect agreement")
    ax.set_xlabel("FEA peak stress (kPa)")
    ax.set_ylabel("Surrogate peak stress (kPa)")
    ax.set_title("Predicted vs. FEA peak wall stress\n(held-out patients)")
    ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    plot_stress_field_3d(f"{FIG_DIR}/fig_stress_field_3d.png")
    plot_peak_parity(f"{FIG_DIR}/fig_peak_stress_parity.png")
