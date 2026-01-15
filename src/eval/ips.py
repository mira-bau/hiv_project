import numpy as np
import pandas as pd


def estimate_ips(reward: np.ndarray, logged_action: np.ndarray, target_action: np.ndarray, propensities: np.ndarray = None) -> float:
    """
    Simple Inverse Propensity Scoring (IPS).
    If propensities are not provided, assume uniform logging over observed actions (fallback: 1.0 where match, 0 otherwise).
    """
    if propensities is None:
        # Avoid zero division; assume prob=1 for matches as a naive fallback
        propensities = np.where(logged_action == logged_action, 1.0, 1.0)
    mask = (logged_action == target_action)
    weights = np.zeros_like(reward, dtype=float)
    weights[mask] = 1.0 / np.clip(propensities[mask], 1e-6, None)
    ips = np.mean(weights[mask] * reward[mask]) if mask.any() else 0.0
    return float(ips)


def estimate_snips(reward: np.ndarray, logged_action: np.ndarray, target_action: np.ndarray, propensities: np.ndarray = None) -> float:
    if propensities is None:
        propensities = np.where(logged_action == logged_action, 1.0, 1.0)
    mask = (logged_action == target_action)
    w = np.zeros_like(reward, dtype=float)
    w[mask] = 1.0 / np.clip(propensities[mask], 1e-6, None)
    denom = w[mask].sum()
    if denom <= 0:
        return 0.0
    snips = (w[mask] * reward[mask]).sum() / denom
    return float(snips)


def estimate_dr(reward: np.ndarray,
                logged_action: np.ndarray,
                target_action: np.ndarray,
                q_hat: np.ndarray,
                propensities: np.ndarray = None) -> float:
    """
    Doubly Robust (DR) estimator.
    - reward: observed rewards
    - logged_action: actions taken in data
    - target_action: actions proposed by target policy
    - q_hat: model predictions of reward for the target action for each row
    - propensities: logging probs P(A=a|X); if unknown, fallback to 1.0 (naive)
    """
    if propensities is None:
        propensities = np.ones_like(reward, dtype=float)
    # Indicator of match
    match = (logged_action == target_action).astype(float)
    ips_term = match * (reward - q_hat) / np.clip(propensities, 1e-6, None)
    dr = np.mean(q_hat + ips_term)
    return float(dr)
