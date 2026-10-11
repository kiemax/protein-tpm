from pathlib import Path

import numpy as np
import pandas as pd


def clean_up(dire: str | Path, files_to_delete: list[str], *, ignore_missing: bool = True) -> None:

    directory = Path(dire)

    for filename in files_to_delete:
        file_path = directory / filename
        try:
            file_path.unlink()
        except FileNotFoundError:
            if not ignore_missing:
                raise
        except Exception as e:
            raise OSError(f"Failed to delete {file_path}: {e}") from e


def save_grid_data(
    data: pd.DataFrame,
    voxel_centers: np.ndarray,
    output_dir: str | Path,
) -> None:
    """
    Save data required for Open3D processing and broom calculation.

    Parameters
    ----------
    data : pd.DataFrame
        Atom dataframe containing at least columns ["x", "y", "z"].
    voxel_centers : np.ndarray
        Array of voxel center coordinates with shape (N, 3).
    output_dir : str | Path
        Directory where artifacts will be written.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_cols = {"x", "y", "z"}
    if not required_cols.issubset(data.columns):
        missing = required_cols - set(data.columns)
        raise KeyError(f"Missing required columns: {missing}")

    atom_cloud = data[["x", "y", "z"]].to_numpy()

    np.save(output_dir / "atom_cloud.npy", atom_cloud)
    np.save(output_dir / "voxel_centers.npy", voxel_centers)

    # Needed later for broom calculations
    data.to_pickle(output_dir / "data_target.pkl")
