import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np

from src.utils.data_loader import load_breast_cancer_data, split_data
from src.feature_selection.fscore import (
    rank_features_by_fscore,
    select_features,
    print_fscore_ranking
)
from src.svm.classifier import SVMClassifier
from src.rf.classifier import RFClassifier
from src.xgboost.classifier import XGBClassifierWrapper
from src.utils.metrics import calculate_metrics, print_metrics
from src.utils.plots import (
    plot_fscore,
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_metrics_table
)
from src.utils.config import Config


class ExperimentRunner:
    def __init__(self, model_type: str):
        self.model_type = model_type.lower()
        self.results = {
            'model_type': self.model_type,
            'fscores': {},
            'accuracies': {},
            'model5_metrics': {},
            'best_params': {},
        }
        Config.ensure_results_dir()

    def get_classifier(self):
        if self.model_type == 'svm':
            return SVMClassifier()
        elif self.model_type == 'rf':
            return RFClassifier()
        elif self.model_type == 'xgboost':
            return XGBClassifierWrapper()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def run_all_experiments(self) -> None:
        print("=" * 80)
        print(f"BREAST CANCER DIAGNOSIS - {self.model_type.upper()}")
        print("=" * 80)
        
        print("\nLoading dataset...")
        try:
            X, y = load_breast_cancer_data()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please ensure 'data/breast-cancer-wisconsin.data' exists.")
            return

        for train_ratio, test_ratio in Config.SPLIT_RATIOS:
            split_name = Config.get_split_name(train_ratio, test_ratio)
            print(f"\n{'=' * 80}")
            print(f"EXPERIMENT: {split_name} Train-Test Split")
            print(f"{'=' * 80}")

            self.run_experiment_for_split(X, y, train_ratio, test_ratio, split_name)

        self.generate_visualizations()
        self.save_results()

        print("\n" + "=" * 80)
        print("ALL EXPERIMENTS COMPLETED!")
        print("=" * 80)

    def run_experiment_for_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float,
        test_ratio: float,
        split_name: str
    ) -> None:
        X_train, X_test, y_train, y_test = split_data(X, y, train_ratio, test_ratio)

        print("\nCalculating F-scores for feature ranking...")
        feature_indices, fscores = rank_features_by_fscore(X_train, y_train)

        self.results['fscores'][split_name] = {
            'indices': feature_indices.tolist(),
            'scores': fscores.tolist()
        }

        print_fscore_ranking(feature_indices, fscores)

        self.results['accuracies'][split_name] = {}

        for n_features in range(1, Config.NUM_FEATURES + 1):
            print(f"\n{'-' * 80}")
            print(f"Model #{n_features}: Using top {n_features} feature(s)")
            print(f"{'-' * 80}")

            X_train_selected = select_features(X_train, feature_indices, n_features)
            X_test_selected = select_features(X_test, feature_indices, n_features)

            selected_feature_names = [
                Config.FEATURE_NAMES[idx] for idx in feature_indices[:n_features]
            ]
            print(f"Selected features: {', '.join(selected_feature_names)}")

            start_time = time.time()

            classifier = self.get_classifier()
            classifier.fit(X_train_selected, y_train, verbose=0) # Less verbose for batch run

            training_time = time.time() - start_time

            y_pred = classifier.predict(X_test_selected)

            metrics = calculate_metrics(y_test, y_pred)

            self.results['accuracies'][split_name][n_features] = metrics['accuracy']

            if n_features == 5:
                self.results['model5_metrics'][split_name] = metrics
                self.results['best_params'][f"{split_name}_model5"] = classifier.get_best_params()

            print(f"Training time: {training_time:.2f} seconds | Accuracy: {metrics['accuracy']:.2f}%")
            
            # Save best params for every model
            self.results['best_params'][f"{split_name}_model{n_features}"] = classifier.get_best_params()

    def generate_visualizations(self) -> None:
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)

        results_dir = Config.RESULTS_PATH / self.model_type
        results_dir.mkdir(parents=True, exist_ok=True)

        if '80-20' in self.results['fscores']:
            fscore_data = self.results['fscores']['80-20']
            indices = np.array(fscore_data['indices'])
            scores = np.array(fscore_data['scores'])

            print("\nGenerating F-score plot...")
            plot_fscore(
                indices,
                scores,
                save_path=results_dir / 'fscore_ranking.png'
            )

        print("\nGenerating accuracy comparison plot...")
        plot_accuracy_comparison(
            self.results['accuracies'],
            save_path=results_dir / 'accuracy_comparison.png'
        )

        print("\nGenerating Model #5 metrics table...")
        plot_metrics_table(
            self.results['model5_metrics'],
            save_path=results_dir / 'model5_metrics_table.png'
        )

        print("\nGenerating confusion matrices for Model #5...")
        for split_name, metrics in self.results['model5_metrics'].items():
            cm = self._reconstruct_cm(metrics)
            plot_confusion_matrix(
                cm,
                split_name,
                f"{self.model_type.upper()} Model #5",
                save_path=results_dir / f'confusion_matrix_{split_name}.png'
            )

        print(f"\nAll visualizations saved to: {results_dir}")

    def _reconstruct_cm(self, metrics):
        from src.utils.metrics import ConfusionMatrix
        return ConfusionMatrix(
            true_positive=metrics['tp'],
            true_negative=metrics['tn'],
            false_positive=metrics['fp'],
            false_negative=metrics['fn']
        )

    def save_results(self) -> None:
        results_dir = Config.RESULTS_PATH / self.model_type
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / 'experiment_results.json'

        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser(description="Breast Cancer Diagnosis Experiment Runner")
    parser.add_argument(
        '--model', 
        type=str, 
        choices=['svm', 'rf', 'xgboost', 'all'], 
        default='all',
        help='Model to run (svm, rf, xgboost, or all)'
    )
    
    args = parser.parse_args()

    models_to_run = [args.model] if args.model != 'all' else ['svm', 'rf', 'xgboost']

    for model in models_to_run:
        runner = ExperimentRunner(model)
        runner.run_all_experiments()

if __name__ == "__main__":
    main()
