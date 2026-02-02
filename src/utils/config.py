from pathlib import Path
from typing import List, Tuple


class Config:
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATASET_PATH = PROJECT_ROOT / "data" / "breast-cancer-wisconsin.data"
    RESULTS_PATH = PROJECT_ROOT / "results"

    RANDOM_SEED = 8

    SPLIT_RATIOS: List[Tuple[float, float]] = [
        (0.5, 0.5),
        (0.7, 0.3),
        (0.8, 0.2),
    ]

    FEATURE_NAMES = [
        'F1_clump_thickness',
        'F2_uniformity_cell_size',
        'F3_uniformity_cell_shape',
        'F4_marginal_adhesion',
        'F5_single_epithelial_cell_size',
        'F6_bare_nucleoi',
        'F7_bland_chromatin',
        'F8_normal_nuclei',
        'F9_mitoses'
    ]

    NUM_FEATURES = 9
    C_RANGE = [2**i for i in range(-5, 16)]
    GAMMA_RANGE = [2**i for i in range(-15, 4, 4)]
    CV_FOLDS = 10
    KERNEL = 'rbf'
    CLASS_BENIGN = 2
    CLASS_MALIGNANT = 4

    @classmethod
    def ensure_results_dir(cls) -> None:
        cls.RESULTS_PATH.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_split_name(cls, train_ratio: float, test_ratio: float) -> str:
        return f"{int(train_ratio * 100)}-{int(test_ratio * 100)}"
