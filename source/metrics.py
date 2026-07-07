"""
metrics.py
==========
Reviewer-required evaluation metrics for DG3Net paper.

All metrics operate in PHYSICAL (denormalised) units unless stated otherwise.
Computes per-sample and aggregate versions of:

    1. Test MSE          (mean squared error on displacement, physical units)
    2. Mean rel L2       (relative L2 error on displacement field)
    3. Peak disp error   (abs error on peak displacement magnitude, mean over test)
    4. Max nodal error   (largest single-node error across the whole test set)
    5. Top-5% node RelL2 (rel L2 restricted to top-5% displacement-magnitude nodes)

Also provides:
    - r2_score          (per-sample coefficient of determination)
    - per_channel_mse   (breakdown across u,v,w or full 21 channels)
    - cross_validation  utilities (k-fold train/test)

Author: Rutwik / DG3Net revision
"""

import numpy as np
import torch


# =============================================================================
# Core per-sample metrics (numpy or torch tensors accepted)
# =============================================================================

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def mse_per_sample(pred, true):
    """
    Mean squared error, per sample, per full-node tensor.
    pred, true: [N_nodes, C] arrays. C is displacement dim (3) or full field (21).
    Returns: scalar MSE.
    """
    pred = _to_numpy(pred)
    true = _to_numpy(true)
    return float(np.mean((pred - true) ** 2))


def rel_l2_per_sample(pred, true, eps=1e-8):
    """
    Relative L2 error: ||pred - true||_2 / (||true||_2 + eps).
    Computed on the displacement magnitude field (first 3 channels).
    pred, true: [N_nodes, C>=3].
    Returns: scalar.
    """
    pred = _to_numpy(pred)[:, :3]
    true = _to_numpy(true)[:, :3]
    # Displacement magnitude per node
    p_mag = np.linalg.norm(pred, axis=1)
    t_mag = np.linalg.norm(true, axis=1)
    return float(np.linalg.norm(p_mag - t_mag) / (np.linalg.norm(t_mag) + eps))


def peak_disp_error_per_sample(pred, true):
    """
    Absolute error on the peak displacement magnitude for this sample.
    pred, true: [N_nodes, C>=3].
    Returns: scalar, |max(||pred||) - max(||true||)|.
    """
    pred = _to_numpy(pred)[:, :3]
    true = _to_numpy(true)[:, :3]
    p_peak = float(np.linalg.norm(pred, axis=1).max())
    t_peak = float(np.linalg.norm(true, axis=1).max())
    return abs(p_peak - t_peak)


def max_nodal_error_per_sample(pred, true):
    """
    Largest absolute error on any single node's displacement magnitude, this sample.
    pred, true: [N_nodes, C>=3].
    Returns: scalar.
    """
    pred = _to_numpy(pred)[:, :3]
    true = _to_numpy(true)[:, :3]
    return float(np.abs(np.linalg.norm(pred, axis=1) - np.linalg.norm(true, axis=1)).max())


def top5_rel_l2_per_sample(pred, true, top_pct=0.05, eps=1e-8):
    """
    Relative L2 error restricted to the top-`top_pct` fraction of nodes by
    reference displacement magnitude. Captures accuracy in high-deformation zones.
    pred, true: [N_nodes, C>=3].
    """
    pred = _to_numpy(pred)[:, :3]
    true = _to_numpy(true)[:, :3]
    t_mag = np.linalg.norm(true, axis=1)
    p_mag = np.linalg.norm(pred, axis=1)
    k = max(1, int(np.ceil(top_pct * len(t_mag))))
    idx = np.argsort(t_mag)[-k:]
    return float(np.linalg.norm(p_mag[idx] - t_mag[idx]) / (np.linalg.norm(t_mag[idx]) + eps))


def r2_per_sample(pred, true, eps=1e-12):
    """
    Coefficient of determination on the displacement magnitude field.
    """
    pred = _to_numpy(pred)[:, :3]
    true = _to_numpy(true)[:, :3]
    p_mag = np.linalg.norm(pred, axis=1)
    t_mag = np.linalg.norm(true, axis=1)
    ss_res = float(np.sum((t_mag - p_mag) ** 2))
    ss_tot = float(np.sum((t_mag - t_mag.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + eps)


def per_channel_mse(pred, true):
    """
    Per-output-channel MSE (useful when output_dim=21).
    Returns: array of length C.
    """
    pred = _to_numpy(pred)
    true = _to_numpy(true)
    return ((pred - true) ** 2).mean(axis=0)


# =============================================================================
# Aggregate over a test set
# =============================================================================

def evaluate_over_set(preds, trues, top_pct=0.05):
    """
    preds, trues: lists of length N_test, each element [N_nodes_i, C].
    Assumes physical (denormalised) units.

    Returns a dict with per-sample lists and aggregate statistics.
    """
    assert len(preds) == len(trues), "preds and trues must have same length"

    mse_list = []
    rel_l2_list = []
    peak_err_list = []
    max_err_list = []
    top5_list = []
    r2_list = []
    peak_true_list = []
    peak_pred_list = []
    n_nodes_list = []

    for p, t in zip(preds, trues):
        mse_list.append(mse_per_sample(p, t))
        rel_l2_list.append(rel_l2_per_sample(p, t))
        peak_err_list.append(peak_disp_error_per_sample(p, t))
        max_err_list.append(max_nodal_error_per_sample(p, t))
        top5_list.append(top5_rel_l2_per_sample(p, t, top_pct=top_pct))
        r2_list.append(r2_per_sample(p, t))
        p_np = _to_numpy(p)[:, :3]
        t_np = _to_numpy(t)[:, :3]
        peak_true_list.append(float(np.linalg.norm(t_np, axis=1).max()))
        peak_pred_list.append(float(np.linalg.norm(p_np, axis=1).max()))
        n_nodes_list.append(int(t_np.shape[0]))

    mse_arr = np.array(mse_list)
    rel_l2_arr = np.array(rel_l2_list)
    peak_arr = np.array(peak_err_list)
    max_arr = np.array(max_err_list)
    top5_arr = np.array(top5_list)
    r2_arr = np.array(r2_list)

    return {
        # per-sample lists (useful for scatter/histograms and later stats)
        "per_sample": {
            "mse":       mse_list,
            "rel_l2":    rel_l2_list,
            "peak_err":  peak_err_list,
            "max_err":   max_err_list,
            "top5":      top5_list,
            "r2":        r2_list,
            "peak_true": peak_true_list,
            "peak_pred": peak_pred_list,
            "n_nodes":   n_nodes_list,
        },
        # aggregate stats (mean ± std across the test set)
        "aggregate": {
            "test_mse_mean":       float(mse_arr.mean()),
            "test_mse_std":        float(mse_arr.std()),
            "rel_l2_mean":         float(rel_l2_arr.mean()),
            "rel_l2_std":          float(rel_l2_arr.std()),
            "peak_err_mean":       float(peak_arr.mean()),
            "peak_err_std":        float(peak_arr.std()),
            "max_nodal_error":     float(max_arr.max()),   # worst over the whole test set
            "max_nodal_err_mean":  float(max_arr.mean()),  # avg worst-per-sample
            "top5_rel_l2_mean":    float(top5_arr.mean()),
            "top5_rel_l2_std":     float(top5_arr.std()),
            "r2_mean":             float(r2_arr.mean()),
            "r2_std":              float(r2_arr.std()),
        },
    }


def format_aggregate_table(agg, name=""):
    """One-line printable summary for a model. `agg` from evaluate_over_set()['aggregate']."""
    a = agg["aggregate"] if "aggregate" in agg else agg
    return (
        f"{name:20s}  "
        f"MSE={a['test_mse_mean']:.4e}±{a['test_mse_std']:.2e}  "
        f"RelL2={a['rel_l2_mean']:.3e}±{a['rel_l2_std']:.2e}  "
        f"Peak={a['peak_err_mean']:.3e}  "
        f"MaxNode={a['max_nodal_error']:.3e}  "
        f"Top5%={a['top5_rel_l2_mean']:.3e}  "
        f"R2={a['r2_mean']:.4f}"
    )


# =============================================================================
# Cross-validation utility (for the "need to run CV" item)
# =============================================================================

def kfold_indices(n_samples, k=5, seed=42, stratify_labels=None):
    """
    Generate k-fold splits.
    If stratify_labels is provided (length n_samples), splits are stratified.
    Returns list of k dicts each with 'train_idx' and 'test_idx'.
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(n_samples)

    if stratify_labels is None:
        rng.shuffle(idx)
        folds = np.array_split(idx, k)
        splits = []
        for i in range(k):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            splits.append({"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()})
        return splits

    # Stratified
    labels = np.asarray(stratify_labels)
    unique_labels = np.unique(labels)
    per_label_folds = {}
    for lab in unique_labels:
        lab_idx = np.where(labels == lab)[0].copy()
        rng.shuffle(lab_idx)
        per_label_folds[lab] = np.array_split(lab_idx, k)

    splits = []
    for i in range(k):
        test_idx = np.concatenate([per_label_folds[lab][i] for lab in unique_labels])
        train_idx = np.concatenate(
            [per_label_folds[lab][j] for lab in unique_labels for j in range(k) if j != i]
        )
        splits.append({"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()})
    return splits


def stratified_train_val_test_split(n_samples, stratify_labels,
                                    test_frac=0.20, val_frac_of_train=0.10, seed=42):
    """
    Produce a (train, val, test) index split that matches the paper's protocol:
        Step 1: STRATIFIED 80/20 train-pool / test split
                (test set is stratified across labels)
        Step 2: GLOBAL 90/10 train / val split on the 80% pool
                (val is drawn randomly from the pool, NOT stratified)

    For 225 samples across 5 balanced classes (9 test/class):
        test = 45, val = round(0.10 * 180) = 18, train = 162.

    Returns: (train_idx, val_idx, test_idx) as lists of ints.
    """
    rng = np.random.RandomState(seed)
    labels = np.asarray(stratify_labels)
    unique_labels = np.unique(labels)

    # Step 1: stratified test extraction (target ~test_frac per label)
    test_idx, pool_idx = [], []
    for lab in unique_labels:
        lab_idx = np.where(labels == lab)[0].copy()
        rng.shuffle(lab_idx)
        n = len(lab_idx)
        n_test = max(1, int(round(test_frac * n)))
        test_idx.extend(lab_idx[:n_test].tolist())
        pool_idx.extend(lab_idx[n_test:].tolist())

    # Step 2: random 90/10 train/val on the pool
    pool_idx = np.array(pool_idx)
    rng.shuffle(pool_idx)
    n_val = int(round(val_frac_of_train * len(pool_idx)))
    val_idx = pool_idx[:n_val].tolist()
    train_idx = pool_idx[n_val:].tolist()

    return train_idx, val_idx, test_idx
