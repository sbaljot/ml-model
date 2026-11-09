"""Feature importance helpers for the three churn training approaches."""

import pandas as pd


def select_features_by_model_importance(models, threshold=0.02):
    """Select features independently for each model using its gain importance.

    The model dictionary should contain the fitted undersampled, original-data,
    and SMOTE models. One-hot encoded category importances are summed back to
    their source feature. Each model gets its own selected feature list.
    """
    importance_by_model = {}

    for model_name, fitted_model in models.items():
        preprocessor = fitted_model.named_steps["preprocessor"]
        classifier = fitted_model.named_steps["classifier"]
        encoded_names = preprocessor.get_feature_names_out()
        importances = classifier.feature_importances_

        numeric_features = set(preprocessor.transformers_[0][2])
        categorical_features = list(preprocessor.transformers_[1][2])

        feature_rows = []
        for encoded_name, importance in zip(encoded_names, importances):
            feature_name = encoded_name.split("__", maxsplit=1)[-1]
            if feature_name in numeric_features:
                source_feature = feature_name
            else:
                source_feature = next(
                    (
                        column
                        for column in categorical_features
                        if feature_name.startswith(f"{column}_")
                    ),
                    feature_name,
                )
            feature_rows.append((source_feature, importance))

        importance_by_model[model_name] = (
            pd.DataFrame(feature_rows, columns=["Feature", "Importance"])
            .groupby("Feature")["Importance"]
            .sum()
        )

    summary = pd.DataFrame(importance_by_model).fillna(0.0)
    summary = summary.sort_index()
    selected_by_model = {}

    print("\nFeature importance by original feature and training approach:")
    print(summary.to_string(formatters={col: "{:.4f}".format for col in summary.columns}))
    print(f"\nPer-model cutoff: remove features with importance <= {threshold:.2f}")

    for model_name in models:
        selected = summary.index[summary[model_name] > threshold].tolist()
        removed = summary.index[summary[model_name] <= threshold].tolist()
        if not selected:
            raise ValueError(
                f"No features exceeded importance cutoff {threshold:.2f} "
                f"for {model_name}."
            )
        selected_by_model[model_name] = selected
        print(f"\n{model_name} — kept features:", selected)
        print(f"{model_name} — removed features:", removed)

    return selected_by_model, summary
