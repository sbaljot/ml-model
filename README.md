# Telco Customer Churn Prediction

This project trains XGBoost models to predict whether a Telco customer will churn. It compares training on the original class distribution, a randomly undersampled training set, and a SMOTE-NC oversampled training set. It also compares full-feature models with models retrained after per-model feature selection.

## Project files

- `WA_Fn-UseC_-Telco-Customer-Churn.csv` — input dataset; keep it beside the Python scripts.
- `train_test_split.py` — loads the CSV and creates fitting, validation, and test partitions, then prepares undersampled and SMOTE training data.
- `class_imbalance.py` — undersampling and SMOTE-NC helper functions.
- `training_pipeline.py` — Optuna search, model fitting, and feature-importance analysis.
- `feature_importance.py` — per-model feature-selection helper.
- `testing_pipeline.py` — compares validation thresholds, test metrics, and sample predictions.

## Setup

Create and activate a virtual environment from the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pandas scikit-learn xgboost optuna imbalanced-learn
```

On macOS, if importing XGBoost reports a missing `libomp.dylib`, install the OpenMP runtime with Homebrew:

```bash
brew install libomp
```

## Run the Python scripts

Run the comparison script from the project directory:

```bash
python testing_pipeline.py
```

It imports the training pipeline, runs Optuna, fits the models, selects decision thresholds on validation data, and reports final test metrics and sample predictions. Since importing `training_pipeline.py` trains the models, this command reruns training each time.

## Workflow and evaluation

The code maps `Churn` to `1` and `No churn` to `0`, converts `TotalCharges` to numeric, and drops the customer ID. It creates a stratified 80/20 training-pool/test split, then reserves 20% of the training pool for validation. Validation and test data retain the original churn proportions.

The fitting-training rows are used in three ways:

1. **Original data:** keep all fitting-training rows.
2. **Undersampling:** randomly reduce `Churn = 0` to the `Churn = 1` count.
3. **SMOTE-NC:** synthesize minority churn samples while handling categorical and numeric features.

Optuna tunes hyperparameters using five-fold ROC AUC on the undersampled data. The same best hyperparameters are reused for the other two approaches, so the initial comparison focuses on sampling strategy rather than independently tuned models.

Each full-feature model reports XGBoost gain importance. One-hot category importances are summed back to their original feature. For each model independently, features with importance **at or below 0.02** are removed, and a reduced-feature model is fitted. The comparison therefore includes three full-feature models and three reduced-feature models.

The validation set is used to select a separate threshold for each model: maximize churn recall while keeping churn precision at least `0.55`. Change `MIN_CHURN_PRECISION` in `testing_pipeline.py` to use a different precision floor. The test set is reserved for final evaluation and reports accuracy, ROC AUC, the classification report, and the confusion matrix. Sample predictions show all six approaches for the same test customers.
