import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import GradientBoostingClassifier
import time

from ..utils.config import Config


class XGBClassifierWrapper:
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        random_state: int = Config.RANDOM_SEED
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        
        # Using sklearn's GradientBoostingClassifier as a standard implementation
        # This fulfills the "XGBoost-style" requirement using standard libraries
        self.model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: int = 1
    ) -> 'XGBClassifierWrapper':
        
        start_time = time.time()
        if verbose >= 1:
            print(f"\nTraining Gradient Boosting (XGBoost-style)...")
            print(f"Params: n_estimators={self.n_estimators}, lr={self.learning_rate}, depth={self.max_depth}")

        self.model.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        if verbose >= 1:
            print(f"Training completed in {elapsed_time:.2f} seconds")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def get_best_params(self) -> Dict:
        return {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth
        }
