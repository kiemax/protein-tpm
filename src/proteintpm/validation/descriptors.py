from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

PAD_CLASS = 0
ZERO_ATOL = 1e-12


@dataclass(frozen=True)
class SphereCoverageConfig:
    n_theta: int = 12
    n_phi: int = 24


def valid_mask(
    positions: np.ndarray,
    classes: np.ndarray,
    pad_class: int = PAD_CLASS,
    zero_atol: float = ZERO_ATOL,
) -> np.ndarray:
    """
    True for rows that are real atoms rather than padding.

    positions: (N, 3)
    classes:   (N,) or (N, 1)
    """
    cls = np.asarray(classes).reshape(-1)
    pos = np.asarray(positions)

    class_mask = cls != pad_class
    pos_mask = ~np.all(np.isclose(pos, 0.0, atol=zero_atol), axis=1)
    return class_mask & pos_mask


def safe_unit_vectors(vectors: np.ndarray) -> np.ndarray:
    """
    Normalize vectors row-wise, dropping zero-length rows.
    """
    norms = np.linalg.norm(vectors, axis=1)
    keep = norms > ZERO_ATOL
    if not np.any(keep):
        return np.empty((0, 3), dtype=float)
    return vectors[keep] / norms[keep][:, None]


def pairwise_angles_deg(unit_vectors: np.ndarray) -> np.ndarray:
    """
    Return all pairwise angles (degrees) between unit vectors.
    """
    n = unit_vectors.shape[0]
    if n < 2:
        return np.empty((0,), dtype=float)

    dots = unit_vectors @ unit_vectors.T
    dots = np.clip(dots, -1.0, 1.0)

    iu = np.triu_indices(n, k=1)
    angles = np.degrees(np.arccos(dots[iu]))
    return angles


def spherical_coverage(
    unit_vectors: np.ndarray,
    cfg: SphereCoverageConfig = SphereCoverageConfig(),
) -> float:
    """
    Fraction of occupied bins on the sphere using simple theta/phi binning.

    This is a pragmatic first-pass approximation to solid-angle occupancy.
    """
    if unit_vectors.shape[0] == 0:
        return np.nan

    x = unit_vectors[:, 0]
    y = unit_vectors[:, 1]
    z = unit_vectors[:, 2]

    theta = np.arccos(np.clip(z, -1.0, 1.0))  # [0, pi]
    phi = np.arctan2(y, x)  # [-pi, pi]
    phi = np.mod(phi, 2.0 * np.pi)  # [0, 2pi)

    theta_idx = np.minimum((theta / np.pi * cfg.n_theta).astype(int), cfg.n_theta - 1)
    phi_idx = np.minimum((phi / (2.0 * np.pi) * cfg.n_phi).astype(int), cfg.n_phi - 1)

    occupied = set(zip(theta_idx.tolist(), phi_idx.tolist()))
    return len(occupied) / float(cfg.n_theta * cfg.n_phi)


def _unit_vector_from_angles(theta: float, phi: float) -> np.ndarray:
    """
    theta in [0, pi], phi in [0, 2pi)
    """
    return np.array(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        dtype=float,
    )


def minimum_enclosing_cone(unit_vectors: np.ndarray) -> tuple[float, float]:
    """
    Compute the smallest cone that contains all unit vectors.

    Returns
    -------
    half_angle_rad : float
        Minimal cone half-angle in radians.
    solid_angle_fraction : float
        Cone solid angle divided by 4*pi, in [0, 1].
    """
    if unit_vectors.shape[0] == 0:
        return np.nan, np.nan

    if unit_vectors.shape[0] == 1:
        return 0.0, 0.0

    # good initialization: mean direction if available
    mean_vec = unit_vectors.mean(axis=0)
    mean_norm = np.linalg.norm(mean_vec)

    if mean_norm > ZERO_ATOL:
        u0 = mean_vec / mean_norm
        theta0 = np.arccos(np.clip(u0[2], -1.0, 1.0))
        phi0 = np.mod(np.arctan2(u0[1], u0[0]), 2.0 * np.pi)
    else:
        theta0 = np.pi / 2.0
        phi0 = 0.0

    def objective(x: np.ndarray) -> float:
        theta, phi = x
        u = _unit_vector_from_angles(theta, phi)
        # maximize minimum dot product  <=> minimize negative min dot
        min_dot = np.min(unit_vectors @ u)
        return -min_dot

    res = minimize(
        objective,
        x0=np.array([theta0, phi0]),
        bounds=[(0.0, np.pi), (0.0, 2.0 * np.pi)],
        method="L-BFGS-B",
    )

    theta_opt, phi_opt = res.x
    u_opt = _unit_vector_from_angles(theta_opt, phi_opt)

    min_dot = np.min(unit_vectors @ u_opt)
    min_dot = np.clip(min_dot, -1.0, 1.0)

    half_angle_rad = float(np.arccos(min_dot))

    # solid angle of cone cap, normalized by full sphere 4*pi
    solid_angle_fraction = float((1.0 - np.cos(half_angle_rad)) / 2.0)

    return half_angle_rad, solid_angle_fraction


def compute_broom_descriptors(
    positions: np.ndarray,
    classes: np.ndarray,
    sphere_cfg: SphereCoverageConfig = SphereCoverageConfig(),
) -> dict[str, float]:
    """
    Compute geometric descriptors for one broom.

    positions: (max_neighbours, 3)
    classes:   (max_neighbours,) or (max_neighbours, 1)
    """
    mask = valid_mask(positions, classes)
    pos = np.asarray(positions)[mask]

    if pos.shape[0] == 0:
        return {
            "n_neighbors_valid": 0,
            "mean_r": np.nan,
            "std_r": np.nan,
            "min_r": np.nan,
            "max_r": np.nan,
            "frac_r_lt_4": np.nan,
            "frac_r_lt_6": np.nan,
            "frac_r_lt_8": np.nan,
            "resultant_norm": np.nan,
            "mean_pairwise_angle_deg": np.nan,
            "sphere_coverage": np.nan,
            "cone_half_angle_deg": np.nan,
            "cone_solid_angle_fraction": np.nan,
        }

    r = np.linalg.norm(pos, axis=1)
    unit = safe_unit_vectors(pos)

    if unit.shape[0] > 0:
        resultant = np.linalg.norm(unit.mean(axis=0))
        angles = pairwise_angles_deg(unit)
        mean_pairwise_angle_deg = float(np.mean(angles)) if angles.size > 0 else np.nan
        coverage = spherical_coverage(unit, cfg=sphere_cfg)
        cone_half_angle_rad, cone_solid_angle_fraction = minimum_enclosing_cone(unit)
        cone_half_angle_deg = (
            float(np.degrees(cone_half_angle_rad)) if not np.isnan(cone_half_angle_rad) else np.nan
        )

    else:
        resultant = np.nan
        mean_pairwise_angle_deg = np.nan
        coverage = np.nan
        cone_half_angle_deg = np.nan
        cone_solid_angle_fraction = np.nan

    return {
        "n_neighbors_valid": int(pos.shape[0]),
        "mean_r": float(np.mean(r)),
        "std_r": float(np.std(r)),
        "min_r": float(np.min(r)),
        "max_r": float(np.max(r)),
        "frac_r_lt_4": float(np.mean(r < 4.0)),
        "frac_r_lt_6": float(np.mean(r < 6.0)),
        "frac_r_lt_8": float(np.mean(r < 8.0)),
        "resultant_norm": float(resultant),
        "mean_pairwise_angle_deg": float(mean_pairwise_angle_deg)
        if not np.isnan(mean_pairwise_angle_deg)
        else np.nan,
        "sphere_coverage": float(coverage) if not np.isnan(coverage) else np.nan,
        "cone_half_angle_deg": float(cone_half_angle_deg)
        if not np.isnan(cone_half_angle_deg)
        else np.nan,
        "cone_solid_angle_fraction": float(cone_solid_angle_fraction)
        if not np.isnan(cone_solid_angle_fraction)
        else np.nan,
    }


def compute_descriptor_table(
    positions: np.ndarray,
    classes: np.ndarray,
    sphere_cfg: SphereCoverageConfig = SphereCoverageConfig(),
) -> pd.DataFrame:
    """
    Compute descriptors for a batch of brooms.

    positions: (N, max_neighbours, 3)
    classes:   (N, max_neighbours, 1) or (N, max_neighbours)
    """
    n = positions.shape[0]
    rows: list[dict[str, float]] = []

    for i in range(n):
        rows.append(
            compute_broom_descriptors(
                positions=positions[i],
                classes=classes[i],
                sphere_cfg=sphere_cfg,
            )
        )

    df = pd.DataFrame(rows)
    df.insert(0, "example_id", np.arange(n))
    return df


def augment_prediction_table_with_descriptors(
    pred_df: pd.DataFrame,
    positions: np.ndarray,
    classes: np.ndarray,
    sphere_cfg: SphereCoverageConfig = SphereCoverageConfig(),
) -> pd.DataFrame:
    """
    Merge computed descriptors into a prediction table.

    Assumes example order matches the prediction rows.
    Uses example_id if present in pred_df, otherwise row order.
    """
    desc_df = compute_descriptor_table(
        positions=positions,
        classes=classes,
        sphere_cfg=sphere_cfg,
    )

    pred_df = pred_df.copy().reset_index(drop=True)

    if "example_id" in pred_df.columns:
        out = pred_df.merge(desc_df, on="example_id", how="left", validate="one_to_one")
    else:
        pred_df.insert(0, "example_id", np.arange(len(pred_df)))
        out = pred_df.merge(desc_df, on="example_id", how="left", validate="one_to_one")

    return out
