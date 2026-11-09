"""Compare churn predictions across sampling and feature-selection approaches."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

import training_pipeline as tp


MIN_CHURN_PRECISION = 0.55
THRESHOLD_VALUES = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

models = {
    "Undersampled": tp.model,
    "Original training data": tp.model_without_undersampling,
    "SMOTE oversampled": tp.model_with_smote,
    "Undersampled, selected features": tp.reduced_model,
    "Original training data, selected features": tp.reduced_model_without_undersampling,
    "SMOTE oversampled, selected features": tp.reduced_model_with_smote,
}
feature_selection_keys = {
    "Undersampled, selected features": "Undersampled",
    "Original training data, selected features": "Original training data",
    "SMOTE oversampled, selected features": "SMOTE oversampled",
}

X_val = tp.split.X_val
y_val = tp.split.y_val
X_test = tp.split.X_test
y_test = tp.split.y_test

selected_thresholds = {}
test_probabilities = {}
test_predictions = {}
threshold_rows = []

for model_name, estimator in models.items():
    selected_features = tp.selected_features_by_model.get(
        feature_selection_keys.get(model_name)
    )
    val_features = X_val[selected_features] if selected_features else X_val
    val_probabilities = estimator.predict_proba(val_features)[:, 1]

    # Show how this model's precision and recall change over common thresholds.
    for threshold in THRESHOLD_VALUES:
        val_predictions = (val_probabilities >= threshold).astype(int)
        threshold_rows.append(
            {
                "Model": model_name,
                "Threshold": threshold,
                "Churn precision": precision_score(
                    y_val, val_predictions, zero_division=0
                ),
                "Churn recall": recall_score(
                    y_val, val_predictions, zero_division=0
                ),
                "Churn F1": f1_score(y_val, val_predictions, zero_division=0),
            }
        )

    # Select each model's threshold on the natural-prevalence validation set.
    precision, recall, thresholds = precision_recall_curve(y_val, val_probabilities)
    eligible = np.flatnonzero(precision[:-1] >= MIN_CHURN_PRECISION)
    if eligible.size == 0:
        raise ValueError(
            f"{model_name} did not reach minimum churn precision "
            f"{MIN_CHURN_PRECISION:.2f} on validation data."
        )
    best_idx = eligible[np.argmax(recall[:-1][eligible])]
    threshold = thresholds[best_idx]
    selected_thresholds[model_name] = threshold
    print(
        f"{model_name} selected threshold: {threshold:.3f} "
        f"(validation precision={precision[best_idx]:.3f}, "
        f"recall={recall[best_idx]:.3f})"
    )

    test_features = X_test[selected_features] if selected_features else X_test
    test_probabilities[model_name] = estimator.predict_proba(test_features)[:, 1]
    test_predictions[model_name] = (
        test_probabilities[model_name] >= threshold
    ).astype(int)

print("\nValidation results by threshold:")
print(pd.DataFrame(threshold_rows).round(3).to_string(index=False))

for model_name in models:
    predictions = test_predictions[model_name]
    probabilities = test_probabilities[model_name]
    print(f"\n--- Held-out test results: {model_name} ---")
    print("Threshold:", round(selected_thresholds[model_name], 3))
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("ROC AUC:", roc_auc_score(y_test, probabilities))
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["No churn", "Churn"],
            zero_division=0,
        )
    )
    print(
        "Confusion matrix (actual rows; predicted columns):\n",
        confusion_matrix(y_test, predictions, labels=[0, 1]),
    )

# Show both models' predictions for the same held-out customers.
sample = X_test.head(10)
sample_results = pd.DataFrame(index=sample.index)
sample_results["ActualChurn"] = y_test.loc[sample.index]
column_prefixes = {
    "Undersampled": "Undersampled",
    "Original training data": "Original",
    "SMOTE oversampled": "SMOTE",
    "Undersampled, selected features": "UndersampledSelected",
    "Original training data, selected features": "OriginalSelected",
    "SMOTE oversampled, selected features": "SMOTESelected",
}
for model_name in models:
    column_prefix = column_prefixes[model_name]
    sample_results[f"{column_prefix}PredictedChurn"] = test_predictions[
        model_name
    ][: len(sample)]
    sample_results[f"{column_prefix}ChurnProbability"] = test_probabilities[
        model_name
    ][: len(sample)]

print("\nSample predictions on the same test customers:")
print(sample_results.round(3).to_string())
