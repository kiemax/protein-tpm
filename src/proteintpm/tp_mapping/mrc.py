from pathlib import Path

import mrcfile
import numpy as np


def write_mrc(
    mrc_path: Path,
    probabilities: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    grid_z: np.ndarray,
    voxel_size: float,
) -> None:
    """
    Write a probability volume to MRC format.

    grid_x.shape is expected to be (nx, ny, nz).
    Probabilities are expected as a flat array of length nx * ny * nz.
    """
    nx, ny, nz = grid_x.shape

    origin_x = grid_x[0, 0, 0] - voxel_size / 2
    origin_y = grid_y[0, 0, 0] - voxel_size / 2
    origin_z = grid_z[0, 0, 0] - voxel_size / 2

    expected_size = nx * ny * nz
    if probabilities.size != expected_size:
        raise ValueError(
            f"Probabilities length mismatch: got {probabilities.size}, "
            f"expected {expected_size} from grid shape {(nx, ny, nz)}"
        )

    volume_xyz = probabilities.reshape((nx, ny, nz))
    volume_zyx = volume_xyz.transpose(2, 1, 0)  # (nz, ny, nx)

    with mrcfile.new(mrc_path, overwrite=True) as mrc:
        mrc.set_data(volume_zyx.astype(np.float32))
        mrc.voxel_size = voxel_size
        mrc.header.origin.x = float(origin_x)
        mrc.header.origin.y = float(origin_y)
        mrc.header.origin.z = float(origin_z)
