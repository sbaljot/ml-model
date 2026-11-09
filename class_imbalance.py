"""Utilities for handling class imbalance in the churn training data."""

import pandas as pd


def undersample_majority_class(X, y, majority_class=0, random_state=42):
    """Randomly reduce the majority class to the minority class size.

    Returns a shuffled, balanced pair of feature and target objects. The input
    objects are not modified.
    """
    class_counts = y.value_counts()
    if majority_class not in class_counts.index:
        raise ValueError(f"Class {majority_class!r} is not present in y.")
    if len(class_counts) != 2:
        raise ValueError("This helper expects a binary target with exactly 2 classes.")

    minority_class = class_counts.drop(index=majority_class).idxmin()
    minority_indices = y[y == minority_class].index
    majority_indices = y[y == majority_class].sample(
        n=len(minority_indices),
        random_state=random_state,
    ).index

    selected_indices = pd.Index(list(minority_indices) + list(majority_indices))
    shuffled_indices = pd.Series(selected_indices).sample(
        frac=1,
        random_state=random_state,
    ).tolist()

    X_balanced = X.loc[shuffled_indices]
    y_balanced = y.loc[shuffled_indices]
    return X_balanced, y_balanced


def oversample_churn_with_smote(X, y, churn_class=1, random_state=42):
    """Use SMOTE-NC to synthesize minority churn rows in mixed-type data.

    SMOTE-NC handles continuous and categorical columns together. Missing
    values are imputed from this training partition before generating samples.
    """
    from imblearn.over_sampling import SMOTENC

    class_counts = y.value_counts()
    if churn_class not in class_counts.index:
        raise ValueError(f"Class {churn_class!r} is not present in y.")
    other_classes = class_counts.drop(index=churn_class)
    if len(class_counts) != 2:
        raise ValueError("This helper expects a binary target with exactly 2 classes.")
    if class_counts.loc[churn_class] >= other_classes.iloc[0]:
        raise ValueError("The churn class is not the minority class to oversample.")
    if class_counts.loc[churn_class] < 2:
        raise ValueError("SMOTE requires at least 2 churn examples in the training data.")

    X_prepared = X.copy()
    categorical_columns = list(
        X_prepared.select_dtypes(include=["object", "category", "bool"]).columns
    )
    # Treat two-valued numeric features (for example SeniorCitizen) as
    # categorical so SMOTE-NC does not create fractional category values.
    for column in X_prepared.select_dtypes(include="number").columns:
        if X_prepared[column].nunique(dropna=True) <= 2:
            categorical_columns.append(column)

    for column in categorical_columns:
        values = X_prepared[column].astype(object)
        X_prepared[column] = values.where(values.notna(), "__missing__")

    numeric_columns = X_prepared.select_dtypes(include="number").columns
    for column in numeric_columns:
        median = X_prepared[column].median()
        X_prepared[column] = X_prepared[column].fillna(median)

    sampler = SMOTENC(
        categorical_features=categorical_columns,
        random_state=random_state,
        k_neighbors=min(5, int(class_counts.loc[churn_class]) - 1),
    )
    X_resampled, y_resampled = sampler.fit_resample(X_prepared, y)
    X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    y_resampled = pd.Series(y_resampled, name=y.name)
    return X_resampled, y_resampled
