import math
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def extract_theta_columns(df: pd.DataFrame):
    """Extract and sort theta columns from dataframe."""
    theta_cols = [col for col in df.columns if col.startswith("theta")]
    if not theta_cols:
        raise ValueError("No theta columns detected in CSV")
    # why: ensure consistent ordering; what: sort by numeric suffix
    def _theta_key(name: str) -> tuple:
        match = re.search(r"(\d+)$", name)
        return (int(match.group(1)) if match else float("inf"), name)
    return sorted(theta_cols, key=_theta_key)


def mahalanobis_bounds(df: pd.DataFrame, alpha: float = 0.05, seed: int = 42):
    """Compute Mahalanobis bounds for a dataset dataframe."""
    theta_cols = extract_theta_columns(df)
    thetas = df[theta_cols].to_numpy(dtype=np.float32)
    
    if thetas.shape[0] < 2:
        raise ValueError("Need at least two samples to compute covariance")
    
    n_rows = thetas.shape[0]
    ids_arr = df["image_id"].to_numpy() if "image_id" in df.columns else np.arange(1, n_rows + 1)
    flat = thetas.reshape(n_rows, -1)

    # why: match rhoComp split/setup; what: deterministic random partition
    X_train_np, X_test_np, _, ids_test = train_test_split(
        flat, ids_arr, test_size=0.2, random_state=seed, shuffle=True
    )

    # why: support both GPU and CPU; what: use CUDA if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_train = torch.from_numpy(X_train_np).float().to(device)
    X_test = torch.from_numpy(X_test_np).float().to(device)

    # why: follow line-for-line math; what: Mahalanobis stats + per-sample rho
    mean_train = X_train.mean(dim=0, keepdim=True)
    Xc = X_train - mean_train
    n_train = Xc.size(0)
    cov = (Xc.t() @ Xc) / (n_train - 1)
    inv_cov = torch.linalg.inv(cov)
    train_quad = (Xc @ inv_cov * Xc).sum(dim=1)

    cross_train = Xc @ inv_cov @ Xc.t()
    d2_train = train_quad.unsqueeze(1) + train_quad.unsqueeze(0) - 2 * cross_train
    d2_train = torch.clamp(d2_train, min=0.0)
    dist_mat = torch.sqrt(d2_train)
    mean_dists = dist_mat.sum(dim=1) / (n_train - 1)
    overall_mean = mean_dists.mean().item()

    X_test_c = X_test - mean_train
    q_inv = X_test_c @ inv_cov
    q_quad = (q_inv * X_test_c).sum(dim=1)
    cross_test = q_inv @ Xc.t()
    d2_test = q_quad.unsqueeze(1) + train_quad.unsqueeze(0) - 2 * cross_test
    d2_test = torch.clamp(d2_test, min=0.0)
    mean_dists_test = torch.sqrt(d2_test).mean(dim=1)

    mean_dists_test_rho = mean_dists_test - overall_mean
    sorted_mean_dists, sorted_idx = mean_dists_test_rho.sort(dim=0)
    ids_test_sorted = np.asarray(ids_test)[sorted_idx.cpu().numpy()]

    m = sorted_mean_dists.size(0)
    pos_one_based = math.ceil((m / 2 + 1) * (1 - alpha))
    idx = min(max(pos_one_based - 1, 0), m - 1)
    lower_p, upper_p = 100 * (alpha / 2), 100 * (1 - alpha / 2)
    lower_bound, upper_bound = np.percentile(
        sorted_mean_dists.cpu().numpy(), [lower_p, upper_p]
    )

    return {
        "lower": float(lower_bound),
        "upper": float(upper_bound),
        "overall": overall_mean,
        "selected_id": ids_test_sorted[idx],
        "selected_value": float(sorted_mean_dists[idx].item()),
        "alpha": alpha,
        "count": int(m),
    }


def check_coverage(source_df: pd.DataFrame, target_df: pd.DataFrame, bounds: dict, seed: int = 42):
    """Check how much of target dataset falls within source dataset bounds."""
    source_theta_cols = extract_theta_columns(source_df)
    target_theta_cols = extract_theta_columns(target_df)
    
    missing_cols = [col for col in source_theta_cols if col not in target_theta_cols]
    if missing_cols:
        raise ValueError(f"Target missing theta cols: {missing_cols}")
    
    # why: use same columns for both datasets; what: ensure feature alignment
    source_data = source_df[source_theta_cols].to_numpy(dtype=np.float32)
    target_data = target_df[source_theta_cols].to_numpy(dtype=np.float32)

    # why: match testing_dcb split; what: use train portions for distance computation
    X_train, _ = train_test_split(source_data, test_size=0.01, random_state=seed, shuffle=True)
    X_train_target, _ = train_test_split(target_data, test_size=0.01, random_state=seed, shuffle=True)

    # why: support both GPU and CPU; what: use CUDA if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(X_train).float().to(device)
    x1 = torch.from_numpy(X_train_target).float().to(device)

    # why: compute Mahalanobis distances; what: center target data with source mean
    mean_x = x.mean(dim=0, keepdim=True)
    Xc = x - mean_x
    cov = (Xc.t() @ Xc) / (Xc.size(0) - 1)
    inv_cov = torch.linalg.inv(cov)
    train_quad = (Xc @ inv_cov * Xc).sum(dim=1)

    X1c = x1 - mean_x
    q_inv = X1c @ inv_cov
    q_quad = (q_inv * X1c).sum(dim=1)
    cross = q_inv @ Xc.t()
    d2 = q_quad.unsqueeze(1) + train_quad.unsqueeze(0) - 2 * cross
    d2 = torch.clamp(d2, min=0.0)
    dist_mat = torch.sqrt(d2)

    # why: match testing_dcb logic; what: use minimum distance per target sample
    mean_dists = dist_mat.min(dim=1).values
    md_np = mean_dists.cpu().numpy()

    # why: check bounds coverage; what: count samples within [lower, upper]
    lb, ub = bounds["lower"], bounds["upper"]
    in_range_mask = (md_np >= lb) & (md_np <= ub)
    count_in_range = int(in_range_mask.sum())
    percent_in_range = 100 * count_in_range / md_np.shape[0]

    return {
        "count_in_range": count_in_range,
        "total_samples": int(md_np.shape[0]),
        "percent_in_range": float(percent_in_range),
    }

