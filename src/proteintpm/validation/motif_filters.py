from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

PAD_CLASS = 0
ZERO_ATOL = 1e-12


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------


@dataclass(frozen=True)
class MotifFilterConfig:
    """
    Chemistry-group definitions in terms of target atom class ids.

    You should populate these sets once from your atom typing scheme.
    Example:
        aromatic_ids = { ... ids for aromatic carbons/nitrogens ... }
        negative_ids = { ... ids for Asp/Glu oxygens ... }
        positive_ids = { ... ids for Lys/Arg terminal nitrogens ... }
        hbond_acceptor_ids = { ... }
        hbond_donor_ids = { ... }
    """

    aromatic_ids: frozenset[int] = field(default_factory=frozenset)
    negative_ids: frozenset[int] = field(default_factory=frozenset)
    positive_ids: frozenset[int] = field(default_factory=frozenset)
    hbond_acceptor_ids: frozenset[int] = field(default_factory=frozenset)
    hbond_donor_ids: frozenset[int] = field(default_factory=frozenset)
    hydrophobic_ids: frozenset[int] = field(default_factory=frozenset)

    # geometric thresholds
    close_cutoff: float = 4.0
    medium_cutoff: float = 6.0
    aromatic_patch_cutoff: float = 5.0
    planar_ratio_threshold: float = 0.15

    # minimum counts
    min_aromatic_patch_atoms: int = 3
    min_negative_patch_atoms: int = 2
    min_positive_patch_atoms: int = 2


# ---------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------


def valid_mask(
    positions: np.ndarray,
    classes: np.ndarray,
    pad_class: int = PAD_CLASS,
    zero_atol: float = ZERO_ATOL,
) -> np.ndarray:
    cls = np.asarray(classes).reshape(-1)
    pos = np.asarray(positions)

    class_mask = cls != pad_class
    pos_mask = ~np.all(np.isclose(pos, 0.0, atol=zero_atol), axis=1)
    return class_mask & pos_mask


def extract_valid_rows(
    positions: np.ndarray,
    classes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mask = valid_mask(positions, classes)
    pos = np.asarray(positions)[mask]
    cls = np.asarray(classes).reshape(-1)[mask]
    return pos, cls


def distances(pos: np.ndarray) -> np.ndarray:
    if pos.shape[0] == 0:
        return np.empty((0,), dtype=float)
    return np.linalg.norm(pos, axis=1)


def subset_by_class_ids(
    pos: np.ndarray,
    cls: np.ndarray,
    class_ids: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    class_ids = set(class_ids)
    if len(class_ids) == 0 or pos.shape[0] == 0:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=int)

    mask = np.isin(cls, list(class_ids))
    return pos[mask], cls[mask]


def nearest_distance(
    pos: np.ndarray,
) -> float:
    if pos.shape[0] == 0:
        return np.nan
    return float(np.min(np.linalg.norm(pos, axis=1)))


def count_within_cutoff(
    pos: np.ndarray,
    cutoff: float,
) -> int:
    if pos.shape[0] == 0:
        return 0
    r = np.linalg.norm(pos, axis=1)
    return int(np.sum(r < cutoff))


def fraction_within_cutoff(
    pos: np.ndarray,
    cutoff: float,
) -> float:
    if pos.shape[0] == 0:
        return np.nan
    r = np.linalg.norm(pos, axis=1)
    return float(np.mean(r < cutoff))


def principal_variances(pos: np.ndarray) -> np.ndarray:
    """
    Return normalized PCA variances (descending) for a point cloud.
    """
    if pos.shape[0] < 3:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    centered = pos - pos.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)

    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]

    total = np.sum(eigvals)
    if total <= ZERO_ATOL:
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    return eigvals / total


def planarity_score(pos: np.ndarray) -> float:
    """
    Simple planarity proxy from PCA spectrum.
    Small third variance => more planar.
    Return 1 - lambda3 so larger means more planar.
    """
    lam = principal_variances(pos)
    if np.any(np.isnan(lam)):
        return np.nan
    return float(1.0 - lam[2])


def planar_ratio(pos: np.ndarray) -> float:
    """
    Smaller values indicate stronger planarity.
    """
    lam = principal_variances(pos)
    if np.any(np.isnan(lam)):
        return np.nan
    return float(lam[2])


# ---------------------------------------------------------------------
# Motif-level feature extraction
# ---------------------------------------------------------------------


def compute_motif_features_for_broom(
    positions: np.ndarray,
    classes: np.ndarray,
    cfg: MotifFilterConfig,
) -> dict[str, float | bool]:
    """
    Compute chemistry-aware motif descriptors and boolean filters
    for one broom.
    """
    pos, cls = extract_valid_rows(positions, classes)

    if pos.shape[0] == 0:
        return {
            "n_valid_neighbors": 0,
            "n_aromatic_neighbors": 0,
            "n_negative_neighbors": 0,
            "n_positive_neighbors": 0,
            "n_hbond_acceptor_neighbors": 0,
            "n_hbond_donor_neighbors": 0,
            "nearest_aromatic_distance": np.nan,
            "nearest_negative_distance": np.nan,
            "nearest_positive_distance": np.nan,
            "frac_aromatic_lt4": np.nan,
            "frac_negative_lt4": np.nan,
            "frac_positive_lt4": np.nan,
            "aromatic_planarity_score": np.nan,
            "aromatic_planar_ratio": np.nan,
            "has_aromatic_patch": False,
            "has_negative_patch": False,
            "has_positive_patch": False,
            "has_cation_pi_candidate": False,
            "has_aromatic_negative_combo": False,
        }

    aromatic_pos, _ = subset_by_class_ids(pos, cls, cfg.aromatic_ids)
    negative_pos, _ = subset_by_class_ids(pos, cls, cfg.negative_ids)
    positive_pos, _ = subset_by_class_ids(pos, cls, cfg.positive_ids)
    acceptor_pos, _ = subset_by_class_ids(pos, cls, cfg.hbond_acceptor_ids)
    donor_pos, _ = subset_by_class_ids(pos, cls, cfg.hbond_donor_ids)

    n_aromatic = aromatic_pos.shape[0]
    n_negative = negative_pos.shape[0]
    n_positive = positive_pos.shape[0]

    aromatic_patch_n = count_within_cutoff(aromatic_pos, cfg.aromatic_patch_cutoff)
    negative_patch_n = count_within_cutoff(negative_pos, cfg.medium_cutoff)
    positive_patch_n = count_within_cutoff(positive_pos, cfg.medium_cutoff)

    ar_planarity = planarity_score(aromatic_pos)
    ar_planar_ratio = planar_ratio(aromatic_pos)

    has_aromatic_patch = aromatic_patch_n >= cfg.min_aromatic_patch_atoms and (
        np.isnan(ar_planar_ratio) or ar_planar_ratio < cfg.planar_ratio_threshold
    )

    has_negative_patch = negative_patch_n >= cfg.min_negative_patch_atoms
    has_positive_patch = positive_patch_n >= cfg.min_positive_patch_atoms

    # Useful "interaction opportunity" style composites
    has_cation_pi_candidate = has_aromatic_patch
    has_aromatic_negative_combo = has_aromatic_patch and has_negative_patch

    return {
        "n_valid_neighbors": int(pos.shape[0]),
        "n_aromatic_neighbors": int(n_aromatic),
        "n_negative_neighbors": int(n_negative),
        "n_positive_neighbors": int(n_positive),
        "n_hbond_acceptor_neighbors": int(acceptor_pos.shape[0]),
        "n_hbond_donor_neighbors": int(donor_pos.shape[0]),
        "nearest_aromatic_distance": nearest_distance(aromatic_pos),
        "nearest_negative_distance": nearest_distance(negative_pos),
        "nearest_positive_distance": nearest_distance(positive_pos),
        "frac_aromatic_lt4": fraction_within_cutoff(aromatic_pos, cfg.close_cutoff),
        "frac_negative_lt4": fraction_within_cutoff(negative_pos, cfg.close_cutoff),
        "frac_positive_lt4": fraction_within_cutoff(positive_pos, cfg.close_cutoff),
        "aromatic_planarity_score": ar_planarity,
        "aromatic_planar_ratio": ar_planar_ratio,
        "has_aromatic_patch": bool(has_aromatic_patch),
        "has_negative_patch": bool(has_negative_patch),
        "has_positive_patch": bool(has_positive_patch),
        "has_cation_pi_candidate": bool(has_cation_pi_candidate),
        "has_aromatic_negative_combo": bool(has_aromatic_negative_combo),
    }


def compute_motif_feature_table(
    positions: np.ndarray,
    classes: np.ndarray,
    cfg: MotifFilterConfig,
) -> pd.DataFrame:
    """
    Batch version for a whole dataset.
    """
    rows: list[dict[str, float | bool]] = []

    for i in range(positions.shape[0]):
        row = compute_motif_features_for_broom(
            positions=positions[i],
            classes=classes[i],
            cfg=cfg,
        )
        row["example_id"] = i
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = ["example_id"] + [c for c in df.columns if c != "example_id"]
    return df[cols]
