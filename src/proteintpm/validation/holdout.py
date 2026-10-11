from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from proteintpm.pipeline.stack_brooms import (
    BroomStackConfig,
    passes_filters,
    process_triplet,
)


def load_pdb_split_table(pdb_splits_csv: Path) -> pd.DataFrame:
    """
    Load the PDB split table.

    Expected columns:
        - pdb_id
        - split

    Additional columns are ignored.
    """
    df = pd.read_csv(pdb_splits_csv)

    required = {"pdb_id", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"pdb_splits.csv is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["pdb_id"] = df["pdb_id"].astype(str).str.lower()
    df["split"] = df["split"].astype(str)
    return df


def get_pdb_ids_for_split(pdb_splits_csv: Path, split_name: str = "test") -> set[str]:
    """
    Return the set of PDB ids assigned to the requested split.
    """
    df = load_pdb_split_table(pdb_splits_csv)
    subset = df.loc[df["split"] == split_name, "pdb_id"]
    return set(subset.tolist())


def default_all_shards(input_dir: Path) -> list[str]:
    """
    Discover shard directories from the interfaces input directory.
    """
    return sorted([p.name for p in Path(input_dir).iterdir() if p.is_dir()])


def load_triplets_from_pdb_npz(npz_path: Path) -> np.ndarray:
    """
    Load the `triplets` object array from one per-PDB shard file.
    """
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing NPZ file: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    if "triplets" not in data:
        raise ValueError(f"{npz_path} does not contain a 'triplets' array")
    return data["triplets"]


def iter_shard_holdout_examples(
    shard_name: str,
    input_dir: Path,
    holdout_pdb_ids: set[str],
    cfg: BroomStackConfig,
) -> Iterable[tuple[np.ndarray, np.ndarray, dict]]:
    """
    Yield only triplets that pass the exact same processing/filtering used in training.
    """
    shard_dir = Path(input_dir) / shard_name / "data_by_pdb"
    if not shard_dir.is_dir():
        raise FileNotFoundError(f"Missing shard data_by_pdb directory: {shard_dir}")

    for npz_path in sorted(shard_dir.glob("*.npz")):
        pdb_id = npz_path.stem.lower()
        if pdb_id not in holdout_pdb_ids:
            continue

        triplets = load_triplets_from_pdb_npz(npz_path)

        for row_idx, triplet in enumerate(triplets):
            if isinstance(triplet, np.void):
                triplet = triplet.item()

            try:
                positions, classes, dr2 = process_triplet(triplet, cfg)
                keep, skip_reason = passes_filters(positions, classes, dr2, cfg)
                if not keep:
                    continue
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to parse/filter triplet row {row_idx} in {npz_path}"
                ) from exc

            metadata = {
                "source_shard": shard_name,
                "source_pdb": pdb_id,
                "source_row": row_idx,
                "source_file": str(npz_path),
                "source_split": "holdout",
                "n_neighbors_raw": int(np.asarray(triplet["target_atoms"]).shape[0]),
            }
            yield positions.astype(np.float32), classes.astype(np.int32), metadata


def build_dataset_for_model(
    model_shard: str,
    input_dir: Path,
    sample_ids: list[str] | set[str],
    max_neighbours: int,
    all_shards: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build one evaluation dataset for a model shard from an explicit set of PDB IDs.

    Positive examples come from the model shard itself.
    Negative examples come from all other shards.

    Parameters
    ----------
    model_shard
        The shard whose model is being evaluated.
    input_dir
        Root interfaces directory containing shard subdirectories.
    sample_ids
        Explicit list or set of PDB IDs to include.
    max_neighbours
        Max neighbours passed into BroomStackConfig.
    all_shards
        Optional explicit shard list. If None, discover from input_dir.
    """
    input_dir = Path(input_dir)

    if all_shards is None:
        all_shards = default_all_shards(input_dir)

    if model_shard not in all_shards:
        raise ValueError(f"Model shard '{model_shard}' not found in shard list")

    selected_pdb_ids = {str(pdb_id).lower() for pdb_id in sample_ids}
    if not selected_pdb_ids:
        raise ValueError("sample_ids must not be empty")

    cfg = BroomStackConfig(max_neighbours=max_neighbours)

    positions_list: list[np.ndarray] = []
    classes_list: list[np.ndarray] = []
    labels_list: list[list[int]] = []
    metadata_rows: list[dict] = []

    example_id = 0

    # Positives: examples from the model shard itself
    for positions, classes, meta in iter_shard_holdout_examples(
        shard_name=model_shard,
        input_dir=input_dir,
        holdout_pdb_ids=selected_pdb_ids,
        cfg=cfg,
    ):
        positions_list.append(positions)
        classes_list.append(classes)
        labels_list.append([1])

        row = dict(meta)
        row.update(
            {
                "example_id": example_id,
                "model_shard": model_shard,
                "true_label": 1,
                "source_type": "positive",
            }
        )
        metadata_rows.append(row)
        example_id += 1

    # Negatives: same PDB IDs, but from all other shards
    for shard_name in all_shards:
        if shard_name == model_shard:
            continue

        for positions, classes, meta in iter_shard_holdout_examples(
            shard_name=shard_name,
            input_dir=input_dir,
            holdout_pdb_ids=selected_pdb_ids,
            cfg=cfg,
        ):
            positions_list.append(positions)
            classes_list.append(classes)
            labels_list.append([0])

            row = dict(meta)
            row.update(
                {
                    "example_id": example_id,
                    "model_shard": model_shard,
                    "true_label": 0,
                    "source_type": "shard_negative",
                }
            )
            metadata_rows.append(row)
            example_id += 1

    if not positions_list:
        raise RuntimeError(f"No evaluation examples found for model shard '{model_shard}'")

    positions = np.stack(positions_list, axis=0).astype(np.float32)
    classes = np.stack(classes_list, axis=0).astype(np.int32)
    labels = np.asarray(labels_list, dtype=np.int32)
    metadata_df = pd.DataFrame(metadata_rows)

    return positions, classes, labels, metadata_df


def build_holdout_dataset_for_model(
    model_shard: str,
    input_dir: Path,
    pdb_splits_csv: Path,
    max_neighbours: int,
    all_shards: list[str] | None = None,
    split_name: str = "holdout",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    input_dir = Path(input_dir)
    pdb_splits_csv = Path(pdb_splits_csv)

    holdout_pdb_ids = get_pdb_ids_for_split(
        pdb_splits_csv=pdb_splits_csv,
        split_name=split_name,
    )

    return build_dataset_for_model(
        model_shard=model_shard,
        input_dir=input_dir,
        sample_ids=sorted(holdout_pdb_ids),
        max_neighbours=max_neighbours,
        all_shards=all_shards,
    )


# def compare_single_shard_filtered_counts(
#    shard_name: str,
#    input_dir: Path,
#    stacked_dir: Path,
#    pdb_splits_csv: Path,
#    max_neighbours: int,
#    split_name: str = "test",
# ) -> dict:
#    holdout_pdb_ids = get_pdb_ids_for_split(pdb_splits_csv, split_name)
#    cfg = BroomStackConfig(max_neighbours=max_neighbours)
#
#    count = 0
#    for _pos, _cls, _meta in iter_shard_holdout_examples(
#        shard_name=shard_name,
#        input_dir=input_dir,
#        holdout_pdb_ids=holdout_pdb_ids,
#        cfg=cfg,
#    ):
#        count += 1
#
#    stacked_path = Path(stacked_dir) / shard_name / f"target_positions_{split_name}.npz"
#    stacked = np.load(stacked_path, allow_pickle=True)["arr"]
#
#    return {
#        "shard_name": shard_name,
#        "split_name": split_name,
#        "rebuilt_count": int(count),
#        "stacked_count": int(stacked.shape[0]),
#        "counts_match": bool(count == stacked.shape[0]),
#    }


def save_holdout_dataset(
    output_dir: Path,
    positions: np.ndarray,
    classes: np.ndarray,
    labels: np.ndarray,
    metadata_df: pd.DataFrame,
) -> None:
    """
    Save holdout arrays and metadata to disk.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_dir / "holdout_arrays.npz",
        positions=positions,
        classes=classes,
        labels=labels,
    )
    metadata_df.to_csv(output_dir / "holdout_metadata.csv", index=False)


def build_and_save_holdout_dataset_for_model(
    model_shard: str,
    input_dir: Path,
    pdb_splits_csv: Path,
    max_neighbours: int,
    output_dir: Path,
    all_shards: list[str] | None = None,
    split_name: str = "holdout",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Convenience wrapper to build and save one model's holdout dataset.
    """
    positions, classes, labels, metadata_df = build_holdout_dataset_for_model(
        model_shard=model_shard,
        input_dir=input_dir,
        pdb_splits_csv=pdb_splits_csv,
        max_neighbours=max_neighbours,
        all_shards=all_shards,
        split_name=split_name,
    )
    save_holdout_dataset(
        output_dir=output_dir,
        positions=positions,
        classes=classes,
        labels=labels,
        metadata_df=metadata_df,
    )
    return positions, classes, labels, metadata_df
