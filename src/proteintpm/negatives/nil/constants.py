from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NILParams:
    voxel_size: float = 0.5
    padding: float = 4.0

    # sampling parameters
    A0: float = 10000.0
    n0: float = 1000.0
    b: float = 1.0

    # visualization
    bild_radius: float = 0.3

    # open3d subprocess
    conda_env: str = "open3d_env"
    open3d_script: Path | None = None  # if None, must be supplied by caller/CLI

    # cleanup behavior
    cleanup_files: tuple[str, ...] = ("atom_cloud.npy", "voxel_centers.npy", "inside_mask.npy")
