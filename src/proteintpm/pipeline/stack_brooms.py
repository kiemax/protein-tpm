import csv
import logging
from collections.abc import Iterable
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from proteintpm.config import stacking as _stk

logger = logging.getLogger(__name__)

PAD_CLASS = 0
ZERO_ATOL = 1e-12


@dataclass(frozen=True)
class BroomStackConfig:
    # Defaults now sourced from config/stacking.py (values unchanged).
    max_neighbours: int = _stk.broom_max_neighbours  # 100
    max_neighbours_for_filter: int = _stk.broom_filter_window  # 25
    lower_threshold: float = _stk.broom_lower_threshold  # 8.0
    min_count_low: int = _stk.broom_min_count_low  # 5
    upper_threshold: float = _stk.broom_upper_threshold  # 13.0
    max_count_high: int = _stk.broom_max_count_high  # 3


class StackError(RuntimeError):
    """Raised for unexpected stacking failures (IO/format issues)."""


def load_split_csv(path: Path) -> dict[str, str]:
    """
    CSV must contain columns: pdb_id, split
    Returns dict of {pdb_id_lower: split_lower}.
    """
    out: dict[str, str] = {}
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        if "pdb_id" not in r.fieldnames or "split" not in r.fieldnames:
            raise ValueError(f"Split CSV must have columns pdb_id,split. Got: {r.fieldnames}")
        for row in r:
            pdb_id = (row["pdb_id"] or "").strip().lower()
            split = (row["split"] or "").strip().lower()
            if pdb_id and split:
                out[pdb_id] = split
    return out


def pad_or_trim(arr: np.ndarray, target_rows: int) -> np.ndarray:
    if arr.shape[0] >= target_rows:
        return arr[:target_rows]
    padding = np.zeros((target_rows - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
    return np.vstack([arr, padding])


def valid_mask(positions: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """
    True for rows that are real atoms (not padding).
    Uses both class!=0 and 'row not all zeros' for safety.
    """
    cls = classes.reshape(-1)
    class_mask = cls != PAD_CLASS
    pos_mask = ~np.all(np.isclose(positions, 0.0, atol=ZERO_ATOL), axis=1)
    return class_mask & pos_mask


def process_triplet(
    triplet: dict, cfg: BroomStackConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract padded arrays from a single triplet:
    - positions: dx,dy,dz
    - classes: target_atom_id as (N,1)
    - dr2: dr_squared as (N,1) (used for filtering, optional)
    """
    ta = triplet["target_atoms"]  # shape (n,5): [target_atom_id, dx, dy, dz, dr_squared]
    if ta.ndim != 2 or ta.shape[1] != 5:
        raise ValueError(f"target_atoms must have shape (n,5); got {ta.shape}")

    positions = pad_or_trim(ta[:, 1:4], cfg.max_neighbours)  # dx,dy,dz
    classes = pad_or_trim(ta[:, 0].reshape(-1, 1), cfg.max_neighbours)  # target_atom_id
    dr2 = pad_or_trim(ta[:, 4].reshape(-1, 1), cfg.max_neighbours)  # dr_squared
    return positions, classes, dr2


def passes_filters(
    positions_padded: np.ndarray,
    classes_padded: np.ndarray,
    dr2_padded: np.ndarray,
    cfg: BroomStackConfig,
) -> tuple[bool, str]:
    """
    Returns (keep?, reason_if_skipped).
    Filtering is done on unpadded valid rows only, but we keep padded arrays for model input.
    """
    mask = valid_mask(positions_padded, classes_padded)
    valid_positions = positions_padded[mask]
    valid_classes = classes_padded[mask]
    valid_dr2 = dr2_padded[mask].reshape(-1)

    if valid_positions.size == 0 or valid_classes.size == 0:
        return False, "no_valid_rows_after_unpadding"

    # Require at least two unique non-padding classes
    unique_classes = np.unique(valid_classes)
    if unique_classes.size < 2:
        return False, "only_one_class_present"

    # Distance-based broom filter (use dr_squared directly; avoid sqrt)
    m = min(cfg.max_neighbours_for_filter, valid_positions.shape[0], valid_dr2.shape[0])
    if m <= 0:
        return False, "no_neighbors_for_distance_filter"

    low2 = cfg.lower_threshold**2
    high2 = cfg.upper_threshold**2

    dr2 = valid_dr2[:m]
    count_low = int(np.sum(dr2 < low2))
    count_high = int(np.sum(dr2 > high2))

    if count_low < cfg.min_count_low or count_high > cfg.max_count_high:
        return False, "distance_filter_failed"

    return True, ""


@dataclass
class StackStats:
    files_seen: int = 0
    triplets_seen: int = 0
    triplets_kept: int = 0
    triplets_skipped: int = 0
    skip_reasons: Counter[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.skip_reasons is None:
            self.skip_reasons = Counter()


def stack_npz_files_to_brooms(
    file_paths: Iterable[Path],
    cfg: BroomStackConfig,
) -> tuple[np.ndarray, np.ndarray, StackStats]:
    """
    Reads per-PDB NPZ files containing object array 'triplets'.
    Returns:
      positions: (N, max_neighbours, 3)
      classes:   (N, max_neighbours, 1)
      stats
    """
    positions_out: list[np.ndarray] = []
    classes_out: list[np.ndarray] = []
    stats = StackStats()

    for path in file_paths:
        stats.files_seen += 1
        try:
            with np.load(path, allow_pickle=True) as data:
                if "triplets" not in data.files:
                    raise ValueError("NPZ missing key 'triplets'")
                triplets = data["triplets"]
        except Exception as e:
            raise StackError(f"Failed to load {path}: {e}") from e

        for triplet in triplets:
            stats.triplets_seen += 1
            try:
                pos, cls, dr2 = process_triplet(triplet, cfg)
                keep, reason = passes_filters(pos, cls, dr2, cfg)
                if not keep:
                    stats.triplets_skipped += 1
                    stats.skip_reasons[reason] += 1
                    continue
                positions_out.append(pos)
                classes_out.append(cls)
                stats.triplets_kept += 1
            except Exception as e:
                stats.triplets_skipped += 1
                stats.skip_reasons["triplet_processing_error"] += 1
                logger.warning("Skipping triplet in %s due to error: %s", path, e)

    if not positions_out:
        # Return empty arrays with correct rank
        return (
            np.empty((0, cfg.max_neighbours, 3), dtype=float),
            np.empty((0, cfg.max_neighbours, 1), dtype=float),
            stats,
        )

    return np.stack(positions_out), np.stack(classes_out), stats


def write_shard_report_csv(report_path: Path, rows: list[dict]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "shard",
        "split",
        "files_seen",
        "triplets_seen",
        "triplets_kept",
        "triplets_skipped",
        "skip_reason",
        "skip_reason_count",
    ]
    with report_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
