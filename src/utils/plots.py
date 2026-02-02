import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from .config import Config
from .metrics import ConfusionMatrix


sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def plot_fscore(
    feature_indices: np.ndarray,
    fscores: np.ndarray,
    save_path: Optional[Path] = None,
    feature_names: List[str] = None
) -> None:
    if feature_names is None:
        feature_names = Config.FEATURE_NAMES

    labels = [f"F{idx+1}" for idx in feature_indices]

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(len(fscores)), fscores, color='steelblue', alpha=0.8)

    for i in range(min(5, len(bars))):
        bars[i].set_color('darkred')
        bars[i].set_alpha(0.8)

    ax.set_xlabel('Features (ranked by F-score)', fontsize=12, fontweight='bold')
    ax.set_ylabel('F-score Value', fontsize=12, fontweight='bold')
    ax.set_title('Feature Importance using F-score\n(Top 5 highlighted for Model #5)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(axis='y', alpha=0.3)

    for bar, score in zip(bars, fscores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"F-score plot saved to: {save_path}")

    plt.close()


def plot_accuracy_comparison(
    results: Dict[str, Dict[int, float]],
    save_path: Optional[Path] = None
) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))

    colors = {'50-50': 'blue', '70-30': 'green', '80-20': 'red'}
    markers = {'50-50': 'o', '70-30': 's', '80-20': '^'}

    for split_name, accuracies in results.items():
        n_features_list = sorted(accuracies.keys())
        accuracy_list = [accuracies[n] for n in n_features_list]

        ax.plot(n_features_list, accuracy_list,
                marker=markers.get(split_name, 'o'),
                label=f'{split_name} split',
                color=colors.get(split_name, 'black'),
                linewidth=2, markersize=8, alpha=0.7)

    ax.set_xlabel('Number of Features (Model #N)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Classification Accuracy vs. Number of Features\nfor Different Train-Test Splits',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(1, 10))
    ax.set_xticklabels([f'#{i}' for i in range(1, 10)])
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([90, 100.5])

    ax.axhline(y=99.51, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.text(9.2, 99.51, '99.51%\n(paper)', va='center', fontsize=9, color='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Accuracy comparison plot saved to: {save_path}")

    plt.close()


def plot_confusion_matrix(
    cm: ConfusionMatrix,
    split_name: str,
    model_name: str,
    save_path: Optional[Path] = None
) -> None:
    matrix = np.array([
        [cm.true_negative, cm.false_positive],
        [cm.false_negative, cm.true_positive]
    ])

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues',
                cbar_kws={'label': 'Count'},
                ax=ax, annot_kws={'fontsize': 16, 'fontweight': 'bold'})

    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix: {model_name} ({split_name} split)',
                 fontsize=14, fontweight='bold')
    ax.set_xticklabels(['Benign (0)', 'Malignant (1)'], fontsize=11)
    ax.set_yticklabels(['Benign (0)', 'Malignant (1)'], fontsize=11, rotation=0)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix plot saved to: {save_path}")

    plt.close()


def plot_metrics_table(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[Path] = None
) -> None:
    df = pd.DataFrame(results).T
    df = df[['accuracy', 'sensitivity', 'specificity', 'ppv', 'npv']]

    df.columns = ['Accuracy', 'Sensitivity', 'Specificity', 'PPV', 'NPV']

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('tight')
    ax.axis('off')

    table_data = []
    for split_name, row in df.iterrows():
        table_data.append([split_name] + [f"{v:.2f}%" for v in row.values])

    table = ax.table(cellText=table_data,
                     colLabels=['Split'] + list(df.columns),
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    for i in range(len(df.columns) + 1):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    for i in range(1, len(table_data) + 1):
        table[(i, 0)].set_facecolor('#f0f0f0')
        table[(i, 0)].set_text_props(weight='bold')

    plt.title('Model #5 Performance Metrics Across Different Splits',
              fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Metrics table saved to: {save_path}")

    plt.close()
