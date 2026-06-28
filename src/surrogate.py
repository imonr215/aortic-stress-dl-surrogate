"""
surrogate.py

The deep-learning surrogate for FEA wall-stress prediction, following the
architecture described in the abstract: an unsupervised stage plus a supervised
stage.

  Unsupervised (dimensionality reduction):
    - PCA on the aortic shapes (node coordinates) -> compact shape codes
    - PCA on the stress fields                      -> compact stress codes
    Both capture that shapes and stress fields, though high-dimensional, live on
    a low-dimensional manifold across patients.

  Supervised (regression):
    - A neural network (multilayer perceptron) maps shape codes -> stress codes.
    - The predicted stress codes are inverse-transformed through the stress PCA
      basis to reconstruct the full per-node stress field.

Evaluation follows the abstract: 10-fold cross-validation, reporting mean
absolute error (MAE) over all nodes and absolute error of the peak stress (APE)
per patient, separately for the circumferential and longitudinal directions.

To avoid data leakage, both PCA bases and the network are fit on the training
split of each fold only, then applied to the held-out patients.

Run:
    python src/surrogate.py
Outputs:
    results/cv_metrics.json
    results/fold_metrics.csv
    predictions cached in results/example_prediction.npz (first held-out patient)
"""

import json
import sys
import warnings

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "src")

DATA_PATH = "data/aorta_dataset.npz"
RESULTS_DIR = "results"

N_SHAPE_MODES = 20      # PCA components for the shape representation
N_STRESS_MODES = 20     # PCA components for each stress-direction field
N_FOLDS = 10
RANDOM_SEED = 7


def load_dataset():
    d = np.load(DATA_PATH)
    coords = d["coords"]                       # (N, n_nodes, 3)
    stress = d["stress"]                        # (N, n_nodes, 2) in Pa
    grid = tuple(d["grid"])
    n = coords.shape[0]
    X = coords.reshape(n, -1)                   # (N, n_nodes*3) shape vectors
    # Work in kPa for readable error numbers.
    Y_circ = stress[:, :, 0] / 1000.0           # (N, n_nodes)
    Y_long = stress[:, :, 1] / 1000.0
    return X, Y_circ, Y_long, grid


def _fit_predict_direction(X_tr, X_te, Y_tr, n_stress_modes, seed):
    """
    Fit shape-PCA + stress-PCA + MLP on the training split for one stress
    direction, and predict the full-resolution stress field for the test split.
    Shape PCA is shared across directions in principle, but refit here per call
    for simplicity; results are identical because the input X is the same.
    """
    # --- Unsupervised: shape PCA (fit on train only) ---
    shape_scaler = StandardScaler().fit(X_tr)
    shape_pca = PCA(n_components=N_SHAPE_MODES, random_state=seed)
    codes_tr = shape_pca.fit_transform(shape_scaler.transform(X_tr))
    codes_te = shape_pca.transform(shape_scaler.transform(X_te))

    # --- Unsupervised: stress PCA (fit on train only) ---
    stress_mean = Y_tr.mean(axis=0, keepdims=True)
    stress_pca = PCA(n_components=n_stress_modes, random_state=seed)
    stress_codes_tr = stress_pca.fit_transform(Y_tr - stress_mean)

    # --- Supervised: MLP from shape codes -> stress codes ---
    code_scaler = StandardScaler().fit(codes_tr)
    mlp = MLPRegressor(
        hidden_layer_sizes=(100, 100),
        activation="relu",
        solver="adam",
        alpha=3e-3,                 # L2 regularization for generalization
        learning_rate_init=1e-3,
        max_iter=2500,
        tol=1e-6,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # benign non-convergence on small folds
        mlp.fit(code_scaler.transform(codes_tr), stress_codes_tr)

    pred_codes_te = mlp.predict(code_scaler.transform(codes_te))
    if pred_codes_te.ndim == 1:
        pred_codes_te = pred_codes_te[:, None]
    Y_pred = stress_pca.inverse_transform(pred_codes_te) + stress_mean
    return Y_pred


def _metrics(Y_true, Y_pred):
    """MAE over all nodes and absolute peak-stress error (APE), per patient."""
    mae = np.abs(Y_true - Y_pred).mean(axis=1)                 # (n_test,)
    ape = np.abs(Y_true.max(axis=1) - Y_pred.max(axis=1))      # (n_test,)
    return mae, ape


def run_cv():
    X, Y_circ, Y_long, grid = load_dataset()
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    fold_rows = []
    saved_example = False
    for fold, (tr, te) in enumerate(kf.split(X), start=1):
        circ_pred = _fit_predict_direction(X[tr], X[te], Y_circ[tr], N_STRESS_MODES, RANDOM_SEED)
        long_pred = _fit_predict_direction(X[tr], X[te], Y_long[tr], N_STRESS_MODES, RANDOM_SEED)

        circ_mae, circ_ape = _metrics(Y_circ[te], circ_pred)
        long_mae, long_ape = _metrics(Y_long[te], long_pred)

        fold_rows.append({
            "fold": fold,
            "circ_mae": float(circ_mae.mean()), "circ_ape": float(circ_ape.mean()),
            "long_mae": float(long_mae.mean()), "long_ape": float(long_ape.mean()),
        })

        # Cache one held-out patient's prediction for the visualization.
        if not saved_example:
            idx = te[0]
            np.savez_compressed(
                f"{RESULTS_DIR}/example_prediction.npz",
                coords=X[idx].reshape(-1, 3),
                circ_true=Y_circ[idx], circ_pred=circ_pred[0],
                long_true=Y_long[idx], long_pred=long_pred[0],
                grid=np.array(grid),
            )
            saved_example = True

    return fold_rows


def summarize(fold_rows):
    import csv
    keys = ["circ_mae", "circ_ape", "long_mae", "long_ape"]
    arr = {k: np.array([r[k] for r in fold_rows]) for k in keys}

    summary = {
        "n_folds": N_FOLDS,
        "circumferential": {
            "mae_kpa": [round(arr["circ_mae"].mean(), 2), round(arr["circ_mae"].std(), 2)],
            "ape_kpa": [round(arr["circ_ape"].mean(), 2), round(arr["circ_ape"].std(), 2)],
        },
        "longitudinal": {
            "mae_kpa": [round(arr["long_mae"].mean(), 2), round(arr["long_mae"].std(), 2)],
            "ape_kpa": [round(arr["long_ape"].mean(), 2), round(arr["long_ape"].std(), 2)],
        },
    }

    with open(f"{RESULTS_DIR}/fold_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold"] + keys)
        w.writeheader()
        w.writerows(fold_rows)
    with open(f"{RESULTS_DIR}/cv_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    import os
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fold_rows = run_cv()
    summary = summarize(fold_rows)

    c, l = summary["circumferential"], summary["longitudinal"]
    print("=== 10-fold cross-validated surrogate accuracy (synthetic cohort) ===\n")
    print("Circumferential direction:")
    print(f"  MAE = {c['mae_kpa'][0]:.2f} ± {c['mae_kpa'][1]:.2f} kPa")
    print(f"  APE = {c['ape_kpa'][0]:.2f} ± {c['ape_kpa'][1]:.2f} kPa")
    print("Longitudinal direction:")
    print(f"  MAE = {l['mae_kpa'][0]:.2f} ± {l['mae_kpa'][1]:.2f} kPa")
    print(f"  APE = {l['ape_kpa'][0]:.2f} ± {l['ape_kpa'][1]:.2f} kPa")
    print(f"\nResults saved to {RESULTS_DIR}/cv_metrics.json and fold_metrics.csv")


if __name__ == "__main__":
    main()
