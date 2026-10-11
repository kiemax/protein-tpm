from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from proteintpm.data.chains import get_chains_negative
from proteintpm.data.pdb_io import create_dataframe, strip_pdb
from proteintpm.features.atom_types import map_atom
from proteintpm.geometry.voxel import calc_voxel_distances, voxel_grid
from proteintpm.io.numpy_io import clean_up, save_grid_data
from proteintpm.negatives.nil.constants import NILParams
from proteintpm.negatives.nil.open3d_runner import run_open3d_processing
from proteintpm.negatives.nil.paths import make_ab_ag_output_dirs
from proteintpm.sampling.points import sample_points
from proteintpm.sampling.weights import F_exclude_near
from proteintpm.visualization.bild import write_bild_spheres


@dataclass(frozen=True)
class NILRunResult:
    pdb_id: str
    ag_out_dir: Path
    ab_out_dir: Path


def _select_atoms_for_chains(df: pd.DataFrame, chains: list[str]) -> pd.DataFrame:
    return df[df["chain_id"].isin(chains)].copy()


def _add_target_atom_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_atom_id"] = out.apply(lambda r: map_atom(r["atom_name"], r["res_name"]), axis=1)
    return out


def generate_nil_for_pdb(
    pdb_path: str | Path,
    out_base: str | Path,
    params: NILParams,
    *,
    pdb_id: str | None = None,
    exist_ok: bool = True,
) -> NILRunResult:
    """
    Generate NIL negative samples for a single PDB (AB + AG):
      - strips chains into dfs (no re-parsing per chain)
      - builds voxel grid + saves atom_cloud/voxel_centers/data_target.pkl
      - runs Open3D script (conda env)
      - computes distance transform, excludes inside voxels
      - samples points, writes points.bild + random_pts.npy
      - cleans up intermediate files

    Output layout:
      <out_base>/<pdb_id>/AG/...
      <out_base>/<pdb_id>/AB/...
    """
    pdb_path = Path(pdb_path)
    out_base = Path(out_base)
    _pdb_id = pdb_id or pdb_path.stem

    if params.open3d_script is None:
        raise ValueError("params.open3d_script must be set (Path to open3d_processing.py).")

    # ---- parse once
    complex_data = create_dataframe(strip_pdb(str(pdb_path)))

    # ---- chains
    chains_a, chains_b = get_chains_negative(pdb_path)
    ag_chains, ab_chains = chains_a, chains_b

    # ---- make output dirs
    ag_out, ab_out = make_ab_ag_output_dirs(out_base, _pdb_id, exist_ok=exist_ok)

    # ---- prepare dfs (no per-chain PDB re-read)
    ag_df = _add_target_atom_id(_select_atoms_for_chains(complex_data, ag_chains))
    ab_df = _add_target_atom_id(_select_atoms_for_chains(complex_data, ab_chains))

    for out_dir, df in ((ag_out, ag_df), (ab_out, ab_df)):
        # compute grid + centers (moved out of save_grid_data as requested)
        (grid_x, grid_y, grid_z), voxel_centers = voxel_grid(
            data=df[["x", "y", "z"]],
            voxel_size=params.voxel_size,
            padding=params.padding,
        )
        grid = (grid_x, grid_y, grid_z)

        # save artifacts for Open3D + broom calculation
        save_grid_data(data=df, voxel_centers=voxel_centers, output_dir=out_dir)

        # run open3d (produces inside_mask.npy + surface_area.npy)
        run_open3d_processing(
            out_dir,
            conda_env=params.conda_env,
            script_path=params.open3d_script,
        )

        surface_area = float(np.load(out_dir / "surface_area.npy"))
        inside_mask = np.load(out_dir / "inside_mask.npy")

        # distance field (outside distance)
        distances = calc_voxel_distances(
            atom_cloud=df[["x", "y", "z"]].to_numpy(),
            grid=grid,
            voxel_size=params.voxel_size,
        )

        # reshape + exclude interior
        inside_mask = inside_mask.reshape(distances.shape)
        distances[inside_mask] = 0.0

        # same sample-count formula as legacy
        n_samples = round(params.n0 * (surface_area / params.A0) ** params.b)

        pts = sample_points(
            grid_box=grid,
            grid_distances=distances,
            n_samples=n_samples,
            mask=None,
            weighting_func=F_exclude_near,
        )

        write_bild_spheres(out_dir / "points.bild", points=pts, radius=params.bild_radius)
        np.save(out_dir / "random_pts.npy", pts)

        clean_up(out_dir, files_to_delete=list(params.cleanup_files))

    return NILRunResult(pdb_id=_pdb_id, ag_out_dir=ag_out, ab_out_dir=ab_out)
