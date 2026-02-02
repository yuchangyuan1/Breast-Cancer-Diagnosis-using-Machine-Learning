import numpy as np
from dataclasses import dataclass
from typing import Dict
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, f1_score


@dataclass
class ConfusionMatrix:
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int

    @classmethod
    def from_predictions(cls, y_true: np.ndarray, y_pred: np.ndarray) -> 'ConfusionMatrix':
        # Assuming 1 is Malignant (Positive) and 0 is Benign (Negative)
        cm = sk_confusion_matrix(y_true, y_pred, labels=[0, 1])
        return cls(
            true_negative=int(cm[0, 0]),
            false_positive=int(cm[0, 1]),
            false_negative=int(cm[1, 0]),
            true_positive=int(cm[1, 1])
        )

    def total_samples(self) -> int:
        return self.true_positive + self.true_negative + self.false_positive + self.false_negative

    def __str__(self) -> str:
        return (
            f"Confusion Matrix:\n"
            f"                 Predicted\n"
            f"                 Benign  Malignant\n"
            f"Actual Benign      {self.true_negative:4d}    {self.false_positive:4d}\n"
            f"       Malignant   {self.false_negative:4d}    {self.true_positive:4d}"
        )


def calculate_accuracy(cm: ConfusionMatrix) -> float:
    total = cm.total_samples()
    if total == 0:
        return 0.0

    return ((cm.true_positive + cm.true_negative) / total) * 100


def calculate_sensitivity(cm: ConfusionMatrix) -> float:
    denominator = cm.true_positive + cm.false_negative
    if denominator == 0:
        return 0.0

    return (cm.true_positive / denominator) * 100


def calculate_specificity(cm: ConfusionMatrix) -> float:
    denominator = cm.false_positive + cm.true_negative
    if denominator == 0:
        return 0.0

    return (cm.true_negative / denominator) * 100


def calculate_ppv(cm: ConfusionMatrix) -> float:
    denominator = cm.true_positive + cm.false_positive
    if denominator == 0:
        return 0.0

    return (cm.true_positive / denominator) * 100


def calculate_npv(cm: ConfusionMatrix) -> float:
    denominator = cm.false_negative + cm.true_negative
    if denominator == 0:
        return 0.0

    return (cm.true_negative / denominator) * 100


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    cm = ConfusionMatrix.from_predictions(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    metrics = {
        'accuracy': calculate_accuracy(cm),
        'sensitivity': calculate_sensitivity(cm),
        'specificity': calculate_specificity(cm),
        'ppv': calculate_ppv(cm),
        'npv': calculate_npv(cm),
        'f1': f1,
        'tp': cm.true_positive,
        'tn': cm.true_negative,
        'fp': cm.false_positive,
        'fn': cm.false_negative
    }

    return metrics


def print_metrics(metrics: Dict[str, float], model_name: str = "Model") -> None:
    print(f"\n{'=' * 60}")
    print(f"{model_name} Performance Metrics")
    print(f"{'=' * 60}")
    print(f"Classification Accuracy: {metrics['accuracy']:6.2f}%")
    print(f"Sensitivity (Recall):    {metrics['sensitivity']:6.2f}%")
    print(f"Specificity:             {metrics['specificity']:6.2f}%")
    print(f"Positive Predictive Val: {metrics['ppv']:6.2f}%")
    print(f"Negative Predictive Val: {metrics['npv']:6.2f}%")
    print(f"F1 Score:                {metrics['f1']:6.4f}")
    print(f"{'-' * 60}")
    print(f"Confusion Matrix:")
    print(f"  TP: {metrics['tp']:4d}  FP: {metrics['fp']:4d}")
    print(f"  FN: {metrics['fn']:4d}  TN: {metrics['tn']:4d}")
    print(f"{'=' * 60}")
