"""
prediction_noise.py
===================================================================
Inject realistic prediction error into the PREDICTED patient
attributes that drive routing and prioritization.

Each patient carries two versions of every predicted attribute:
  * TRUE value  -> drives the simulation physics
  * PRED value  -> drives the routing/priority decision only
                   (7-attribute code -> similarity -> pod + priority)

In the perfect-foresight baseline PRED == TRUE. This module produces a
corrupted PRED that reflects how the predictive models actually err, so
the simulation can be re-run on imperfect inputs and the resulting
performance change measured. Only predicted attributes are corrupted;
CTAS (attribute A) is observed at triage and left untouched.

Two noise modes:
  1. Confusion-matrix mode (realistic): draw the predicted class
     p ~ P(p | t) from the model's row-normalized confusion matrix.
     An `alpha` knob scales the off-diagonal mass (alpha=0 perfect,
     alpha=1 observed error, alpha=2 twice the error, ...).
  2. Flip-rate mode (simple sweep): with probability epsilon, replace
     the true class with a uniformly random different class.
"""

from __future__ import annotations
import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# Core: confusion-matrix based corruption
# ----------------------------------------------------------------------
def row_normalize(cm: np.ndarray) -> np.ndarray:
    """Row-normalize a confusion-matrix-like array into P(pred | true)."""
    cm = np.asarray(cm, dtype=float)
    rs = cm.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return cm / rs


def scale_offdiagonal(P: np.ndarray, alpha: float) -> np.ndarray:
    """
    Scale off-diagonal (error) probability mass by `alpha`, keeping rows
    valid probability distributions.

        alpha = 0  -> identity (no error)
        alpha = 1  -> observed error structure
        alpha > 1  -> amplified error (stress test)

    The diagonal is set to 1 - alpha*(observed off-diagonal mass) and
    clipped to [0, 1]; if a row saturates, the off-diagonal is
    renormalized to fill the remainder.
    """
    P = np.asarray(P, dtype=float).copy()
    n = P.shape[0]
    out = np.zeros_like(P)
    for i in range(n):
        diag = P[i, i]
        off = P[i].copy()
        off[i] = 0.0
        off_mass = off.sum()
        new_off_mass = min(alpha * off_mass, 1.0)
        if off_mass > 0:
            off = off / off_mass * new_off_mass
        out[i] = off
        out[i, i] = max(0.0, 1.0 - new_off_mass)
        # numerical safety: renormalize
        s = out[i].sum()
        if s > 0:
            out[i] /= s
    return out


def corrupt_from_confusion(
    true_labels: np.ndarray,
    cm: np.ndarray,
    classes: list,
    alpha: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Corrupt categorical labels using a confusion matrix.

    Parameters
    ----------
    true_labels : array of the patients' TRUE classes (any hashable labels)
    cm          : raw confusion matrix, rows = true class, cols = pred class,
                  ordered to match `classes`
    classes     : ordered list of class labels matching cm rows/cols
    alpha       : off-diagonal scaling (see scale_offdiagonal)
    rng         : numpy Generator (for reproducibility)

    Returns
    -------
    predicted labels (same length / dtype family as true_labels)
    """
    rng = rng or np.random.default_rng()
    P = scale_offdiagonal(row_normalize(cm), alpha)
    idx = {c: i for i, c in enumerate(classes)}
    classes_arr = np.array(classes, dtype=object)
    true = np.asarray(true_labels, dtype=object)
    out = np.empty(len(true), dtype=object)
    for i, t in enumerate(true):
        row = P[idx[t]]
        out[i] = classes_arr[rng.choice(len(classes), p=row)]
    return out


# ----------------------------------------------------------------------
# Simple flip-rate corruption (no confusion matrix needed)
# ----------------------------------------------------------------------
def corrupt_flip(
    true_labels: np.ndarray,
    classes: list,
    epsilon: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """With prob. epsilon, replace label with a uniformly random DIFFERENT class."""
    rng = rng or np.random.default_rng()
    classes_arr = np.array(classes, dtype=object)
    true = np.asarray(true_labels, dtype=object)
    out = true.copy()
    flip = rng.random(len(true)) < epsilon
    for i in np.where(flip)[0]:
        alts = classes_arr[classes_arr != true[i]]
        if len(alts):
            out[i] = rng.choice(alts)
    return out


# ----------------------------------------------------------------------
# Multi-attribute driver
# ----------------------------------------------------------------------
# Spec for one predicted attribute.
#   - 'classes': ordered class labels
#   - 'cm': confusion matrix (rows=true) OR None to use flip-rate mode
# CTAS (attribute A) is observed -> do NOT list it here.
def corrupt_predictions(
    df: pd.DataFrame,
    attr_specs: dict,
    mode: str = "confusion",
    level: float = 1.0,
    pred_suffix: str = "_pred",
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Add a corrupted *_pred column for every predicted attribute.

    Parameters
    ----------
    df         : one row per patient; must contain the TRUE columns named
                 by the keys of attr_specs.
    attr_specs : {true_col_name: {'classes': [...], 'cm': ndarray|None}}
    mode       : 'confusion' (uses cm + alpha=level) or 'flip' (epsilon=level)
    level      : alpha (confusion mode) or epsilon (flip mode)
    pred_suffix: suffix for the new corrupted columns
    seed       : RNG seed for reproducibility

    Returns
    -------
    copy of df with added '<attr><pred_suffix>' columns
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col, spec in attr_specs.items():
        classes = spec["classes"]
        if mode == "confusion":
            cm = spec.get("cm")
            if cm is None:
                raise ValueError(f"confusion mode needs a 'cm' for '{col}'")
            out[col + pred_suffix] = corrupt_from_confusion(
                out[col].values, cm, classes, alpha=level, rng=rng
            )
        elif mode == "flip":
            out[col + pred_suffix] = corrupt_flip(
                out[col].values, classes, epsilon=level, rng=rng
            )
        else:
            raise ValueError("mode must be 'confusion' or 'flip'")
    return out


def achieved_error_rates(df: pd.DataFrame, attr_specs: dict,
                         pred_suffix: str = "_pred") -> pd.DataFrame:
    """Report realized misclassification rate per attribute (sanity check)."""
    rows = []
    for col in attr_specs:
        pred = col + pred_suffix
        if pred in df.columns:
            err = float((df[col].values != df[pred].values).mean())
            rows.append({"attribute": col, "error_rate": round(err, 4)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Demo / self-test (synthetic data)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    N = 20000

    # --- Synthetic TRUE attributes (replace with your real dataset) ---
    # Predicted attributes in the coding scheme (CTAS = A is observed):
    #   B Service time (5 classes), C Admission (2), D Disposition (4),
    #   E Diagnostic test (2), F Resource use (4), G Consultation (2)
    df = pd.DataFrame({
        "service_time": rng.integers(1, 6, N),   # B: 1..5
        "admission":    rng.integers(1, 3, N),   # C: 1..2
        "disposition":  rng.integers(1, 5, N),   # D: 1..4
        "diag_test":    rng.integers(1, 3, N),   # E: 1..2
        "resource_use": rng.integers(1, 5, N),   # F: 1..4
        "consult":      rng.integers(1, 3, N),   # G: 1..2
    })

    def cm_from_accuracy(k, acc, conc=4.0, seed=1):
        """Toy confusion matrix: ~acc on diagonal, errors spread to neighbours.
        REPLACE with real sklearn confusion_matrix() output from your models."""
        r = np.random.default_rng(seed)
        M = np.zeros((k, k))
        for i in range(k):
            M[i, i] = acc
            rem = 1 - acc
            w = np.array([1.0 / (abs(i - j) + 1) ** conc if j != i else 0
                          for j in range(k)])
            w = w / w.sum() * rem
            M[i] += w
        return M

    # Illustrative default accuracies; replace cm with the fitted
    # confusion matrices (see fit_models.py / confusion_matrices.py):
    attr_specs = {
        "service_time": {"classes": [1,2,3,4,5], "cm": cm_from_accuracy(5, 0.67)},
        "admission":    {"classes": [1,2],       "cm": cm_from_accuracy(2, 0.89)},
        "disposition":  {"classes": [1,2,3,4],   "cm": cm_from_accuracy(4, 0.79)},
        "diag_test":    {"classes": [1,2],       "cm": cm_from_accuracy(2, 0.90)},
        "resource_use": {"classes": [1,2,3,4],   "cm": cm_from_accuracy(4, 0.71)},
        "consult":      {"classes": [1,2],       "cm": cm_from_accuracy(2, 0.865)},
    }

    print("=== Confusion-matrix mode: error vs alpha (stress) ===")
    for alpha in [0.0, 0.5, 1.0, 1.5, 2.0]:
        d = corrupt_predictions(df, attr_specs, mode="confusion",
                                level=alpha, seed=42)
        er = achieved_error_rates(d, attr_specs).set_index("attribute")["error_rate"]
        print(f"alpha={alpha:>3}:  " +
              "  ".join(f"{k}={v:.3f}" for k, v in er.items()))

    print("\n=== Flip-rate mode: error vs epsilon ===")
    for eps in [0.0, 0.1, 0.2, 0.3]:
        d = corrupt_predictions(df, attr_specs, mode="flip", level=eps, seed=42)
        er = achieved_error_rates(d, attr_specs).set_index("attribute")["error_rate"]
        print(f"eps={eps:>3}:  " +
              "  ".join(f"{k}={v:.3f}" for k, v in er.items()))

    print("\nOK - module self-test passed.")
