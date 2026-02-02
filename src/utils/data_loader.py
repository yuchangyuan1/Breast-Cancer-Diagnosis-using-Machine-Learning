import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple
from sklearn.model_selection import train_test_split

from .config import Config


def load_breast_cancer_data(file_path: Path = Config.DATASET_PATH) -> Tuple[np.ndarray, np.ndarray]:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    column_names = ['id'] + Config.FEATURE_NAMES + ['class']

    df = pd.read_csv(
        file_path,
        header=None,
        names=column_names,
        na_values='?'
    )

    df_clean = df.dropna()

    X = df_clean[Config.FEATURE_NAMES].values.astype(np.float64)
    y = df_clean['class'].values.astype(np.int32)

    # 4 is Malignant (1), 2 is Benign (0)
    y_binary = (y == Config.CLASS_MALIGNANT).astype(np.int32)

    return X, y_binary


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    test_ratio: float,
    random_seed: int = Config.RANDOM_SEED
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train_ratio + test_ratio, 1.0):
        raise ValueError(
            f"train_ratio ({train_ratio}) + test_ratio ({test_ratio}) must equal 1.0"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
