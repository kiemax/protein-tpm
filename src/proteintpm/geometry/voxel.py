import numpy as np
from scipy.ndimage import distance_transform_edt


def voxel_grid(data, voxel_size, padding):
    # Create voxel grid
    x_min, y_min, z_min = data[["x", "y", "z"]].min(axis=0)
    x_max, y_max, z_max = data[["x", "y", "z"]].max(axis=0)

    x_centers = np.arange(x_min - padding, x_max + padding, voxel_size) + voxel_size / 2
    y_centers = np.arange(y_min - padding, y_max + padding, voxel_size) + voxel_size / 2
    z_centers = np.arange(z_min - padding, z_max + padding, voxel_size) + voxel_size / 2

    grid_x, grid_y, grid_z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    voxel_centers = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    return (grid_x, grid_y, grid_z), voxel_centers


def calc_voxel_distances(
    atom_cloud: np.array, grid: tuple[np.array, np.array, np.array], voxel_size: float
) -> np.array:

    grid_x, grid_y, grid_z = grid
    point_cloud = atom_cloud
    P = np.asarray(point_cloud, dtype=float)
    occ = np.zeros(grid_x.shape, dtype=bool)
    origin = (grid_x[0][0][0], grid_y[0][0][0], grid_z[0][0][0])
    nx, ny, nz = grid_x.shape
    ijk = np.floor((P - origin) / np.full(3, voxel_size)).astype(int)
    inside = (
        (ijk[:, 0] >= 0)
        & (ijk[:, 0] < nx)
        & (ijk[:, 1] >= 0)
        & (ijk[:, 1] < ny)
        & (ijk[:, 2] >= 0)
        & (ijk[:, 2] < nz)
    )
    ijk = ijk[inside]
    occ[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

    return distance_transform_edt(~occ, sampling=np.full(3, voxel_size))
