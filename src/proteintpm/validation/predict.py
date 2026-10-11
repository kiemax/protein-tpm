from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from proteintpm.validation.holdout import build_dataset_for_model

REQUIRED_ORIGIN_COLUMNS = ["source_shard", "source_pdb", "source_row"]
OPTIONAL_PROVENANCE_COLUMNS = ["source_type", "source_file", "source_split"]


def load_trained_model(model_path: Path) -> keras.Model:
    """
    Load a saved Keras model.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    return keras.models.load_model(model_path)


def require_metadata_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required metadata columns: {missing}")


def ensure_example_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure every example has a deterministic canonical key.
    """
    out = df.copy()

    if "example_key" not in out.columns:
        require_metadata_columns(out, REQUIRED_ORIGIN_COLUMNS)
        out["example_key"] = (
            out["source_shard"].astype(str)
            + "::"
            + out["source_pdb"].astype(str)
            + "::"
            + out["source_row"].astype(str)
        )

    return out


def assert_unique_example_key(df: pd.DataFrame, where: str) -> None:
    if "example_key" not in df.columns:
        raise ValueError(f"{where}: missing example_key")

    duplicated = df["example_key"].duplicated()
    if duplicated.any():
        examples = df.loc[duplicated, "example_key"].astype(str).tolist()[:10]
        raise ValueError(f"{where}: duplicate example_key values found, e.g. {examples}")


def order_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Put provenance and prediction columns first for readability and safer downstream use.
    """
    preferred = [
        "example_key",
        "source_shard",
        "source_pdb",
        "source_row",
        "source_type",
        "source_file",
        "source_split",
        "y_true",
        "y_score",
        "y_pred",
        "threshold",
    ]
    front = [c for c in preferred if c in df.columns]
    rest = [c for c in df.columns if c not in front]
    return df[front + rest]


def prepare_model_inputs(
    positions: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert holdout arrays into the exact input format expected by the model.

    Current model expects:
        input_dist   : shape (N, max_neighbours, 3), float32
        input_tokens : shape (N, max_neighbours), int32
    """
    x_dist = positions.astype(np.float32)

    if classes.ndim != 3 or classes.shape[-1] != 1:
        raise ValueError(f"Expected classes shape (N, max_neighbours, 1), got {classes.shape}")

    x_tokens = classes[:, :, 0].astype(np.int32)
    return x_dist, x_tokens


def predict_scores(
    model: keras.Model,
    positions: np.ndarray,
    classes: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """
    Run model prediction and return flat score array of shape (N,).
    """
    x_dist, x_tokens = prepare_model_inputs(positions, classes)

    preds = model.predict(
        (x_dist, x_tokens),
        batch_size=batch_size,
        verbose=0,
    )

    preds = np.asarray(preds).reshape(-1)
    return preds


def build_prediction_table(
    metadata_df: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Merge metadata, true labels, and predicted scores into one tidy table.
    """
    if labels.ndim == 2 and labels.shape[1] == 1:
        y_true = labels[:, 0]
    else:
        y_true = labels.reshape(-1)

    if len(metadata_df) != len(y_true) or len(y_true) != len(scores):
        raise ValueError(
            "Length mismatch between metadata, labels, and scores: "
            f"{len(metadata_df)=}, {len(y_true)=}, {len(scores)=}"
        )

    df = metadata_df.copy().reset_index(drop=True)
    df = ensure_example_key(df)
    require_metadata_columns(df, ["example_key", *REQUIRED_ORIGIN_COLUMNS])

    df["y_true"] = y_true.astype(int)
    df["y_score"] = scores.astype(float)
    df["y_pred"] = (df["y_score"] >= threshold).astype(int)
    df["threshold"] = float(threshold)

    assert_unique_example_key(df, "prediction table")
    df = order_prediction_columns(df)
    return df


def predict_for_dataset(
    model_shard: str,
    model_path: Path,
    input_dir: Path,
    sample_ids: list[str],
    max_neighbours: int,
    all_shards: list[str] | None = None,
    batch_size: int = 256,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Build a dataset for one model shard from an explicit list of sample IDs,
    run predictions, and return a tidy table.
    """
    positions, classes, labels, metadata_df = build_dataset_for_model(
        model_shard=model_shard,
        input_dir=input_dir,
        sample_ids=sample_ids,
        max_neighbours=max_neighbours,
        all_shards=all_shards,
    )

    model = load_trained_model(model_path)
    scores = predict_scores(
        model=model,
        positions=positions,
        classes=classes,
        batch_size=batch_size,
    )

    pred_df = build_prediction_table(
        metadata_df=metadata_df,
        labels=labels,
        scores=scores,
        threshold=threshold,
    )
    return pred_df


def predict_and_save_for_dataset(
    model_shard: str,
    model_path: Path,
    input_dir: Path,
    sample_ids: list[str],
    max_neighbours: int,
    output_csv: Path,
    all_shards: list[str] | None = None,
    batch_size: int = 256,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Convenience wrapper: predict scores for an explicit dataset and save the CSV.
    """
    pred_df = predict_for_dataset(
        model_shard=model_shard,
        model_path=model_path,
        input_dir=input_dir,
        sample_ids=sample_ids,
        max_neighbours=max_neighbours,
        all_shards=all_shards,
        batch_size=batch_size,
        threshold=threshold,
    )
    save_prediction_table(pred_df, output_csv)
    return pred_df


def save_prediction_table(pred_df: pd.DataFrame, output_path: Path) -> None:
    """
    Save prediction table as CSV.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(output_path, index=False)
