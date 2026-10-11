import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics_from_df(df: pd.DataFrame) -> dict:
    """
    Compute classification metrics from a holdout prediction dataframe.
    """

    y_true = df["y_true"].values
    y_score = df["y_score"].values
    y_pred = df["y_pred"].values

    metrics = {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "positives": int((y_true == 1).sum()),
        "negatives": int((y_true == 0).sum()),
    }

    return metrics
