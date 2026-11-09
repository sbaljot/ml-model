"""Tune and fit an XGBoost churn classifier with Optuna."""

import optuna
import pandas as pd
from xgboost import XGBClassifier
from feature_importance import select_features_by_model_importance
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import clone

import train_test_split as split


numeric_columns = split.X_train.select_dtypes(include="number").columns
categorical_columns = split.X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns

def make_model_pipeline(feature_columns):
    """Build a preprocessing and XGBoost pipeline for the selected columns."""
    selected_data = split.X_train.loc[:, feature_columns]
    numeric_features = selected_data.select_dtypes(include="number").columns
    categorical_features = selected_data.select_dtypes(
        include=["object", "category", "bool"]
    ).columns

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    transformers = []
    if len(numeric_features):
        transformers.append(("numeric", numeric_pipeline, numeric_features))
    if len(categorical_features):
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )
    preprocessor = ColumnTransformer(transformers=transformers)

    # Preprocessing is inside the estimator, so CV fits it within each fold.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="auc",
                    importance_type="gain",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )


base_model = make_model_pipeline(split.X_train.columns.tolist())

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def print_feature_importance(model_name, fitted_model, top_n=20):
    """Print gain importance aggregated back to the original input columns."""
    preprocessor = fitted_model.named_steps["preprocessor"]
    classifier = fitted_model.named_steps["classifier"]
    encoded_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    importance_table = pd.DataFrame(
        {"Encoded feature": encoded_names, "Importance": importances}
    )

    def get_original_feature(encoded_name):
        feature_name = encoded_name.split("__", maxsplit=1)[-1]
        if feature_name in numeric_columns:
            return feature_name
        for column in categorical_columns:
            if feature_name.startswith(f"{column}_"):
                return column
        return feature_name

    importance_table["Feature"] = importance_table["Encoded feature"].map(
        get_original_feature
    )
    grouped = (
        importance_table.groupby("Feature", as_index=False)["Importance"]
        .sum()
        .sort_values("Importance", ascending=False)
        .head(top_n)
    )

    print(f"\nTop {top_n} feature importances — {model_name} (gain):")
    print(grouped.to_string(index=False, formatters={"Importance": "{:.4f}".format}))


def objective(trial):
    """Return mean five-fold ROC AUC for one hyperparameter combination."""
    params = {
        "classifier__n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "classifier__max_depth": trial.suggest_int("max_depth", 2, 8),
        "classifier__learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.3, log=True
        ),
        "classifier__min_child_weight": trial.suggest_float(
            "min_child_weight", 0.5, 10.0, log=True
        ),
        "classifier__subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "classifier__colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.6, 1.0
        ),
        "classifier__gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "classifier__reg_alpha": trial.suggest_float(
            "reg_alpha", 1e-8, 10.0, log=True
        ),
        "classifier__reg_lambda": trial.suggest_float(
            "reg_lambda", 1e-3, 20.0, log=True
        ),
    }

    candidate = clone(base_model).set_params(**params)
    fold_scores = cross_val_score(
        candidate,
        split.X_train,
        split.y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
    )
    return fold_scores.mean()


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
)
study.optimize(objective, n_trials=30)

print(f"Best mean CV ROC AUC: {study.best_value:.3f}")
print("Best hyperparameters:", study.best_params)

# Refit the pipeline with the best hyperparameters on all undersampled rows.
best_params = {
    f"classifier__{name}": value for name, value in study.best_params.items()
}
model = clone(base_model).set_params(**best_params)
model.fit(split.X_train, split.y_train)
print("Undersampled-training model fitted.")

# Fit a comparison model on the original, unbalanced training rows using the
# same tuned hyperparameters. This isolates the effect of undersampling.
model_without_undersampling = clone(base_model).set_params(**best_params)
model_without_undersampling.fit(
    split.X_train_original,
    split.y_train_original,
)
print("Original-training model fitted.")

# Fit a third model on the SMOTE-NC oversampled training data. Validation and
# test partitions are not resampled.
model_with_smote = clone(base_model).set_params(**best_params)
model_with_smote.fit(split.X_train_smote, split.y_train_smote)
print("SMOTE-training model fitted.")

for model_name, fitted_model in (
    ("Undersampled", model),
    ("Original training data", model_without_undersampling),
    ("SMOTE oversampled", model_with_smote),
):
    print_feature_importance(model_name, fitted_model)

# Select features separately for each full-feature model. Importance is
# calculated from training fits only; validation/test data do not select columns.
selected_features_by_model, feature_importance_by_model = select_features_by_model_importance(
    {
        "Undersampled": model,
        "Original training data": model_without_undersampling,
        "SMOTE oversampled": model_with_smote,
    },
    threshold=0.02,
)

undersampled_features = selected_features_by_model["Undersampled"]
original_features = selected_features_by_model["Original training data"]
smote_features = selected_features_by_model["SMOTE oversampled"]

reduced_undersampled_base = make_model_pipeline(undersampled_features)
reduced_base_model = make_model_pipeline(original_features)
reduced_smote_base = make_model_pipeline(smote_features)

reduced_model = clone(reduced_undersampled_base).set_params(**best_params)
reduced_model.fit(split.X_train[undersampled_features], split.y_train)

reduced_model_without_undersampling = clone(reduced_base_model).set_params(
    **best_params
)
reduced_model_without_undersampling.fit(
    split.X_train_original[original_features],
    split.y_train_original,
)

reduced_model_with_smote = clone(reduced_smote_base).set_params(**best_params)
reduced_model_with_smote.fit(
    split.X_train_smote[smote_features],
    split.y_train_smote,
)

print("\nReduced-feature models fitted.")
for model_name, fitted_model in (
    ("Undersampled, selected features", reduced_model),
    (
        "Original training data, selected features",
        reduced_model_without_undersampling,
    ),
    ("SMOTE oversampled, selected features", reduced_model_with_smote),
):
    print_feature_importance(model_name, fitted_model)
