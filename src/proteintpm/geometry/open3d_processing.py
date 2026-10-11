from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Open3DOutputs:
    surface_area: float
    inside_mask: np.ndarray  # shape (N,), dtype=bool


def compute_surface_area_and_inside_mask(
    work_dir: str | Path,
    *,
    alpha: float = 3.0,
    ray_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    save_plot: bool = False,
) -> Open3DOutputs:
    """
    Reads:
      - atom_cloud.npy      (M,3)
      - voxel_centers.npy   (N,3)

    Writes:
      - surface_area.npy    (scalar float)
      - inside_mask.npy     (N,) bool

    Optionally writes:
      - exclude_voxels.png

    Keeps file naming compatible with the validated pipeline.
    """
    work_dir = Path(work_dir)

    # Import here so the module can be imported even when open3d isn't installed
    import open3d as o3d  # type: ignore

    point_cloud = np.load(work_dir / "atom_cloud.npy")
    voxels = np.load(work_dir / "voxel_centers.npy")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    area = float(mesh.get_surface_area())
    np.save(work_dir / "surface_area.npy", area)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    direction = np.array([ray_direction], dtype=np.float32)
    rays = np.concatenate(
        [
            voxels.astype(np.float32),
            np.tile(direction, (voxels.shape[0], 1)),
        ],
        axis=1,
    )
    rays_o3d = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
    counts = scene.count_intersections(rays_o3d).numpy()

    inside_mask = counts % 2 == 1
    np.save(work_dir / "inside_mask.npy", inside_mask)

    if save_plot:
        # NOTE: keep plot behavior identical if you relied on it;
        # otherwise it's safe to make it optional.
        import matplotlib.pyplot as plt  # type: ignore

        interior_voxels = voxels[inside_mask]

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            point_cloud[:, 0],
            point_cloud[:, 1],
            point_cloud[:, 2],
            s=20,
            c="blue",
            label="Point Cloud",
        )

        ax.scatter(
            interior_voxels[:, 0],
            interior_voxels[:, 1],
            interior_voxels[:, 2],
            s=10,
            c="red",
            marker="o",
            label="Interior Voxels",
            alpha=0.8,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend()

        plt.tight_layout()
        plt.savefig(work_dir / "exclude_voxels.png")
        plt.close(fig)

    return Open3DOutputs(surface_area=area, inside_mask=inside_mask)
