import numpy as np
from typing import Dict, Optional
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time

from ..utils.config import Config


class SVMClassifier:
    def __init__(
        self,
        C_range: list = None,
        gamma_range: list = None,
        cv_folds: int = Config.CV_FOLDS,
        kernel: str = Config.KERNEL,
        random_state: int = Config.RANDOM_SEED
    ):
        self.C_range = C_range if C_range is not None else Config.C_RANGE
        self.gamma_range = gamma_range if gamma_range is not None else Config.GAMMA_RANGE
        self.cv_folds = cv_folds
        self.kernel = kernel
        self.random_state = random_state

        self.best_estimator_: Optional[SVC] = None
        self.best_params_: Optional[Dict] = None
        self.best_score_: Optional[float] = None
        self.grid_search_: Optional[GridSearchCV] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        verbose: int = 1
    ) -> 'SVMClassifier':
        param_grid = {
            'C': self.C_range,
            'gamma': self.gamma_range
        }

        if verbose >= 1:
            print(f"\nStarting Grid Search with {self.cv_folds}-fold CV...")
            print(f"Parameter grid size: {len(self.C_range)} C values × {len(self.gamma_range)} gamma values")
            print(f"Total combinations: {len(self.C_range) * len(self.gamma_range)}")

        start_time = time.time()

        base_svm = SVC(
            kernel=self.kernel,
            random_state=self.random_state,
            class_weight='balanced'
        )

        self.grid_search_ = GridSearchCV(
            estimator=base_svm,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=verbose,
            return_train_score=True
        )

        self.grid_search_.fit(X_train, y_train)

        self.best_estimator_ = self.grid_search_.best_estimator_
        self.best_params_ = self.grid_search_.best_params_
        self.best_score_ = self.grid_search_.best_score_

        elapsed_time = time.time() - start_time

        if verbose >= 1:
            print(f"\nGrid Search completed in {elapsed_time:.2f} seconds")
            print(f"Best CV accuracy: {self.best_score_:.4f}")
            print(f"Best parameters:")
            print(f"  C = {self.best_params_['C']:.6f}")
            print(f"  gamma = {self.best_params_['gamma']:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.best_estimator_ is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        return self.best_estimator_.predict(X)

    def get_best_params(self) -> Dict:
        if self.best_params_ is None:
            raise ValueError("Model must be fitted first. Call fit() first.")

        return self.best_params_.copy()
