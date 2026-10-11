import json
from pathlib import Path

import numpy as np
import pandas as pd

from proteintpm.pipeline.stack_brooms import BroomStackConfig, process_triplet
from proteintpm.validation.chemistry_groups import MOTIF_CFG
from proteintpm.validation.descriptors import compute_descriptor_table
from proteintpm.validation.motif_filters import MotifFilterConfig, compute_motif_feature_table

METADATA_COLUMNS = [
    "example_key",
    "source_shard",
    "source_pdb",
    "source_row",
    "source_type",
    "source_file",
    "source_split",
]


def make_example_key(source_shard: str, source_pdb: str, source_row: int | str) -> str:
    return f"{source_shard}::{source_pdb}::{source_row}"


def ensure_example_key(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "example_key" not in out.columns:
        needed = {"source_shard", "source_pdb", "source_row"}
        missing = needed - set(out.columns)
        if missing:
            raise ValueError(f"Cannot derive example_key; missing columns: {sorted(missing)}")
        out["example_key"] = [
            make_example_key(s, p, r)
            for s, p, r in zip(out["source_shard"], out["source_pdb"], out["source_row"])
        ]
    return out


def _prediction_csvs(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("*/predictions.csv"))


def build_set_metadata_from_predictions(run_dir: Path) -> pd.DataFrame:
    csvs = _prediction_csvs(run_dir)
    if not csvs:
        raise FileNotFoundError(f"No per-model predictions.csv files found under {run_dir}")

    frames: list[pd.DataFrame] = []
    for path in csvs:
        df = pd.read_csv(path)
        df = ensure_example_key(df)
        keep = [c for c in METADATA_COLUMNS if c in df.columns]
        if not keep:
            raise ValueError(f"Prediction file has no metadata columns: {path}")
        frames.append(df[keep].copy())

    meta = pd.concat(frames, ignore_index=True)
    meta = meta.drop_duplicates(subset=["example_key"]).reset_index(drop=True)

    if "source_row" in meta.columns:
        try:
            meta["source_row"] = pd.to_numeric(meta["source_row"], errors="raise").astype(int)
        except Exception:
            pass

    sort_cols = [c for c in ["source_shard", "source_pdb", "source_row"] if c in meta.columns]
    if sort_cols:
        meta = meta.sort_values(sort_cols).reset_index(drop=True)
    return meta


def save_set_metadata(run_dir: Path, metadata_df: pd.DataFrame) -> Path:
    out_dir = Path(run_dir) / "evaluation_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "set_metadata.csv"
    metadata_df.to_csv(out_path, index=False)
    return out_path


def compute_and_save_set_metadata(*, run_dir: Path) -> Path:
    meta = build_set_metadata_from_predictions(run_dir)
    return save_set_metadata(run_dir, meta)


def _load_run_config(run_dir: Path) -> dict:
    run_config_path = Path(run_dir) / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"Missing run config: {run_config_path}")
    with open(run_config_path) as f:
        return json.load(f)


def _resolve_input_dir_and_max_neighbours(
    run_dir: Path, max_neighbours: int | None
) -> tuple[Path, int]:
    cfg = _load_run_config(run_dir)
    input_dir = Path(cfg["input_dir"])
    if max_neighbours is None:
        max_neighbours = int(cfg.get("max_neighbours", 25))
    return input_dir, int(max_neighbours)


def _load_triplets_from_npz(npz_path: Path) -> np.ndarray:
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing NPZ file: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        if "triplets" not in data.files:
            raise ValueError(f"{npz_path} does not contain a 'triplets' array")
        triplets = data["triplets"]
    return triplets


def _triplet_to_dict(triplet):
    if isinstance(triplet, np.void):
        return triplet.item()
    return triplet


def build_example_arrays_from_metadata(
    metadata_df: pd.DataFrame,
    *,
    input_dir: Path,
    max_neighbours: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build model-independent example arrays from set metadata.

    Each metadata row identifies one broom via:
      source_shard / source_pdb / source_row

    Returns
    -------
    positions : (N, max_neighbours, 3)
    classes   : (N, max_neighbours, 1)
    """
    cfg = BroomStackConfig(max_neighbours=max_neighbours)

    positions_out: list[np.ndarray] = []
    classes_out: list[np.ndarray] = []

    current_npz_path: Path | None = None
    current_triplets = None

    for _, row in metadata_df.iterrows():
        source_shard = str(row["source_shard"])
        source_pdb = str(row["source_pdb"])
        source_row = int(row["source_row"])

        npz_path = input_dir / source_shard / "data_by_pdb" / f"{source_pdb}.npz"
        if current_npz_path != npz_path:
            current_triplets = _load_triplets_from_npz(npz_path)
            current_npz_path = npz_path

        if source_row < 0 or source_row >= len(current_triplets):
            raise IndexError(
                f"source_row={source_row} out of bounds for {npz_path} with {len(current_triplets)} triplets"
            )

        triplet = _triplet_to_dict(current_triplets[source_row])
        positions, classes, _dr2 = process_triplet(triplet, cfg)

        positions_out.append(positions.astype(np.float32))
        classes_out.append(classes.astype(np.int32))

    if not positions_out:
        return (
            np.empty((0, max_neighbours, 3), dtype=np.float32),
            np.empty((0, max_neighbours, 1), dtype=np.int32),
        )

    return np.stack(positions_out, axis=0), np.stack(classes_out, axis=0)


def save_example_arrays(
    run_dir: Path, metadata_df: pd.DataFrame, positions: np.ndarray, classes: np.ndarray
) -> Path:
    out_dir = Path(run_dir) / "evaluation_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "examples.npz"
    np.savez_compressed(
        out_path,
        example_key=metadata_df["example_key"].astype(str).to_numpy(),
        positions=positions,
        classes=classes,
    )
    return out_path


def load_example_arrays(run_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    meta_path = Path(run_dir) / "evaluation_set" / "set_metadata.csv"
    examples_path = Path(run_dir) / "evaluation_set" / "examples.npz"

    if not meta_path.exists() or not examples_path.exists():
        raise FileNotFoundError(
            f"Missing cached example artifacts under {Path(run_dir) / 'evaluation_set'}"
        )

    meta = pd.read_csv(meta_path)
    meta = ensure_example_key(meta)

    with np.load(examples_path, allow_pickle=True) as data:
        example_keys = data["example_key"].astype(str)
        positions = data["positions"]
        classes = data["classes"]

    meta_keys = meta["example_key"].astype(str).to_numpy()
    if len(meta_keys) != len(example_keys) or not np.array_equal(meta_keys, example_keys):
        raise ValueError("Cached examples.npz does not align with set_metadata.csv")

    return meta, positions, classes


def compute_and_save_example_arrays(*, run_dir: Path, max_neighbours: int | None = None) -> Path:
    meta_path = Path(run_dir) / "evaluation_set" / "set_metadata.csv"
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
    else:
        meta = build_set_metadata_from_predictions(run_dir)
        save_set_metadata(run_dir, meta)

    meta = ensure_example_key(meta)
    input_dir, max_neighbours = _resolve_input_dir_and_max_neighbours(run_dir, max_neighbours)
    positions, classes = build_example_arrays_from_metadata(
        meta,
        input_dir=input_dir,
        max_neighbours=max_neighbours,
    )
    return save_example_arrays(run_dir, meta, positions, classes)


def load_or_build_example_arrays(
    run_dir: Path,
    max_neighbours: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    examples_path = Path(run_dir) / "evaluation_set" / "examples.npz"
    meta_path = Path(run_dir) / "evaluation_set" / "set_metadata.csv"

    if examples_path.exists() and meta_path.exists():
        return load_example_arrays(run_dir)

    compute_and_save_example_arrays(run_dir=run_dir, max_neighbours=max_neighbours)
    return load_example_arrays(run_dir)


def compute_and_save_set_descriptors(*, run_dir: Path, max_neighbours: int | None = None) -> Path:
    meta, positions, classes = load_or_build_example_arrays(run_dir, max_neighbours=max_neighbours)
    desc = compute_descriptor_table(positions=positions, classes=classes)
    desc["example_key"] = meta["example_key"].values

    cols = ["example_key"] + [c for c in desc.columns if c not in {"example_key", "example_id"}]
    desc = desc[cols]

    out_dir = Path(run_dir) / "evaluation_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "set_descriptors.csv"
    desc.to_csv(out_path, index=False)
    return out_path


def compute_and_save_set_chemical_motifs(
    *, run_dir: Path, max_neighbours: int | None = None, motif_cfg: MotifFilterConfig | None = None
) -> Path:
    cfg = motif_cfg or MOTIF_CFG
    meta, positions, classes = load_or_build_example_arrays(run_dir, max_neighbours=max_neighbours)
    motif = compute_motif_feature_table(positions=positions, classes=classes, cfg=cfg)
    motif["example_key"] = meta["example_key"].values

    cols = ["example_key"] + [c for c in motif.columns if c not in {"example_key", "example_id"}]
    motif = motif[cols]

    out_dir = Path(run_dir) / "evaluation_set"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "set_chemical_motifs.csv"
    motif.to_csv(out_path, index=False)
    return out_path
