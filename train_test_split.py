"""Load the churn data and create a held-out, stratified train/test split."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from class_imbalance import (
    oversample_churn_with_smote,
    undersample_majority_class,
)


DATA_PATH = Path(__file__).with_name("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df = pd.read_csv(DATA_PATH)

# Convert the target to binary labels and TotalCharges to numeric. In this CSV,
# blank TotalCharges entries become NaN and will be imputed inside the pipeline.
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Drop the identifier. Keep the other feature columns raw; the training
# pipeline will impute and one-hot encode them separately inside each CV fold.
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"].astype("int64")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# Set aside validation data at the original class distribution. It will be used
# for threshold selection; the test set remains reserved for final evaluation.
X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42,
    stratify=y_train,
)

# Keep the original, unbalanced fitting data so both training approaches can
# be compared on the same validation and test customers.
X_train_original = X_train.copy()
y_train_original = y_train.copy()
X_train_smote, y_train_smote = oversample_churn_with_smote(
    X_train_original,
    y_train_original,
    churn_class=1,
    random_state=42,
)

# Undersample only the model-fitting partition. Keep validation and test data
# at their original class distributions so precision/recall remain meaningful.
X_train, y_train = undersample_majority_class(
    X_train,
    y_train,
    majority_class=0,
    random_state=42,
)

training_counts = y_train.value_counts().reindex([0, 1], fill_value=0)
print("Training non-churn users after undersampling:", training_counts.loc[0])
print("Training churn users after undersampling:", training_counts.loc[1])
smote_counts = y_train_smote.value_counts().reindex([0, 1], fill_value=0)
print("SMOTE non-churn users:", smote_counts.loc[0])
print("SMOTE churn users:", smote_counts.loc[1])
print("Training rows:", len(X_train))
print("SMOTE training rows:", len(X_train_smote))
print("Validation rows:", len(X_val))
print("Test rows:", len(X_test))
