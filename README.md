# Breast Cancer Diagnosis using Machine Learning

This project implements three machine learning approaches for breast cancer diagnosis using the Breast Cancer Wisconsin dataset:
1.  **Support Vector Machine (SVM)**
2.  **Random Forest (RF)**
3.  **Gradient Boosting (XGBoost-style)**

The project includes feature selection using F-score, experimental evaluation on multiple train-test splits (50-50, 70-30, 80-20), and comprehensive visualization of results.

## Project Structure

```
project/
├── data/
│   └── breast-cancer-wisconsin.data  # Raw dataset
├── results/                          # Generated plots and JSON results
│   ├── svm/
│   ├── rf/
│   └── xgboost/
├── src/
│   ├── feature_selection/            # F-score calculation
│   ├── svm/                          # SVM implementation
│   ├── rf/                           # Random Forest implementation
│   ├── xgboost/                      # Gradient Boosting implementation
│   └── utils/                        # Shared utilities (data loader, metrics, config)
├── main.py                           # Unified entry point for experiments
├── requirements.txt                  # Python dependencies
└── paper.pdf                         # Project report
```

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the experiments using `main.py`. You can specify which model to run.

**Run all models (default):**
```bash
python main.py
```

**Run a specific model:**
```bash
python main.py --model svm
python main.py --model rf
python main.py --model xgboost
```

## Methodology

*   **Data Preprocessing:** Handling missing values, binary classification (Benign vs Malignant).
*   **Feature Selection:** Features are ranked using the F-score metric. Models are trained iteratively using the top $N$ features ($N=1$ to $9$).
*   **Evaluation:**
    *   3 Train-Test Splits: 50-50, 70-30, 80-20.
    *   Metrics: Accuracy, Sensitivity, Specificity, PPV, NPV, F1-score.
    *   Confusion Matrices.

## Results

Results (JSON and Plots) are saved in the `results/` directory, organized by model type.
