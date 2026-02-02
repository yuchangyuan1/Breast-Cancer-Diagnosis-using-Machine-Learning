import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time

from ..utils.config import Config


class RFClassifier:
    def __init__(
        self,
        n_estimators: int = 300,
        random_state: int = Config.RANDOM_SEED
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight="balanced",
            max_features="sqrt",
            random_state=self.random_state
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: int = 1
    ) -> 'RFClassifier':
        
        start_time = time.time()
        if verbose >= 1:
            print(f"\nTraining Random Forest with {self.n_estimators} trees...")

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
            "max_features": "sqrt",
            "class_weight": "balanced"
        }

