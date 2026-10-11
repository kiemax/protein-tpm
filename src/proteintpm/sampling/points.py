from collections.abc import Callable

import numpy as np

from proteintpm.sampling.weights import F_exclude_near  # default only

WeightFunc = Callable[[np.ndarray], np.ndarray]


def sample_points(
    grid_box: tuple[np.ndarray, np.ndarray, np.ndarray],
    grid_distances: np.ndarray,
    n_samples: int,
    mask: np.ndarray | None = None,
    weighting_func: WeightFunc = F_exclude_near,  # default
) -> np.ndarray:
    grid_x, grid_y, grid_z = grid_box
    weights = weighting_func(grid_distances)

    if mask is not None:
        weights = np.where(mask, weights, 0.0)

    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("All weights are zero. Check weighting_func() or mask.")

    probs = (weights / total).ravel()
    flat_choices = np.random.choice(probs.size, size=n_samples, p=probs, replace=True)

    nx, ny, nz = grid_x.shape
    i = flat_choices // (ny * nz)
    j = (flat_choices % (ny * nz)) // nz
    k = flat_choices % nz

    return np.column_stack(
        (
            grid_x[i, j, k],
            grid_y[i, j, k],
            grid_z[i, j, k],
        )
    )
