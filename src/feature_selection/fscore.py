import numpy as np
from typing import List, Tuple

from ..utils.config import Config


def calculate_fscore(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_positive = X[y == 1]
    X_negative = X[y == 0]

    n_positive = len(X_positive)
    n_negative = len(X_negative)

    mean_overall = np.mean(X, axis=0)
    mean_positive = np.mean(X_positive, axis=0)
    mean_negative = np.mean(X_negative, axis=0)

    numerator = (
        (mean_positive - mean_overall) ** 2 +
        (mean_negative - mean_overall) ** 2
    )

    variance_positive = np.sum((X_positive - mean_positive) ** 2, axis=0) / (n_positive - 1)
    variance_negative = np.sum((X_negative - mean_negative) ** 2, axis=0) / (n_negative - 1)

    denominator = variance_positive + variance_negative

    fscores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator != 0
    )

    return fscores


def rank_features_by_fscore(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fscores = calculate_fscore(X, y)

    sorted_indices = np.argsort(fscores)[::-1]

    return sorted_indices, fscores[sorted_indices]


def select_features(
    X: np.ndarray,
    feature_indices: np.ndarray,
    n_features: int
) -> np.ndarray:
    if n_features < 1 or n_features > X.shape[1]:
        raise ValueError(
            f"n_features must be between 1 and {X.shape[1]}, got {n_features}"
        )

    selected_indices = feature_indices[:n_features]
    return X[:, selected_indices]


def print_fscore_ranking(
    feature_indices: np.ndarray,
    fscores: np.ndarray,
    feature_names: List[str] = None
) -> None:
    if feature_names is None:
        feature_names = Config.FEATURE_NAMES

    print("\nFeature Ranking by F-score:")
    print("=" * 70)
    print(f"{ 'Rank':<6} {'Feature Index':<15} {'Feature Name':<35} {'F-score':<10}")
    print("-" * 70)

    for rank, (idx, score) in enumerate(zip(feature_indices, fscores), 1):
        feature_name = feature_names[idx] if idx < len(feature_names) else f"F{idx+1}"
        print(f"{rank:<6} {idx+1:<15} {feature_name:<35} {score:<10.4f}")

    print("=" * 70)
