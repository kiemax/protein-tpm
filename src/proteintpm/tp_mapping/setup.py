import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from proteintpm.data.pdb_io import create_dataframe, strip_pdb
from proteintpm.features.atom_types import map_atom
from proteintpm.tp_mapping.jobs import (
    BoxSource,
    MappingJob,
    ModelMaterialized,
    ResidueSelector,
    build_mapping_job,
    make_model_id,
    make_output_dir,
    make_run_id,
    write_json,
)

# ----------------------------
# Generic helpers
# ----------------------------


def validate_file_exists(p: Path, label: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{label} does not exist: {p}")
    if not p.is_file():
        raise ValueError(f"{label} is not a file: {p}")


def validate_dir_exists(p: Path, label: str) -> None:
    if not p.exists():
        raise FileNotFoundError(f"{label} does not exist: {p}")
    if not p.is_dir():
        raise ValueError(f"{label} is not a directory: {p}")


def try_symlink(src: Path, dst: Path, is_dir: bool = True) -> bool:
    """
    Try to create a symlink. Returns True on success, False otherwise.
    """
    try:
        if dst.exists() or dst.is_symlink():
            return True
        dst.symlink_to(src.resolve(), target_is_directory=is_dir)
        return True
    except OSError:
        return False


def voxel_grid(data_xyz: pd.DataFrame, voxel_size: float, padding: float):
    """
    Build a voxel center grid based on min/max of x,y,z in data_xyz.
    data_xyz must have columns: x,y,z

    Returns:
        (grid_x, grid_y, grid_z), voxel_centers
    """
    x_min, y_min, z_min = data_xyz[["x", "y", "z"]].min(axis=0)
    x_max, y_max, z_max = data_xyz[["x", "y", "z"]].max(axis=0)

    x_centers = np.arange(x_min - padding, x_max + padding, voxel_size) + voxel_size / 2
    y_centers = np.arange(y_min - padding, y_max + padding, voxel_size) + voxel_size / 2
    z_centers = np.arange(z_min - padding, z_max + padding, voxel_size) + voxel_size / 2

    grid_x, grid_y, grid_z = np.meshgrid(x_centers, y_centers, z_centers, indexing="ij")
    voxel_centers = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

    return (grid_x, grid_y, grid_z), voxel_centers


# ----------------------------
# Data loading / selection
# ----------------------------


def load_pdb_dataframe(pdb_path: Path) -> pd.DataFrame:
    """
    Load a PDB file using the existing proteintpm helpers.
    """
    validate_file_exists(pdb_path, "PDB file")
    df = create_dataframe(strip_pdb(str(pdb_path)))
    if df.empty:
        raise ValueError(f"No atoms found in PDB file: {pdb_path}")
    return df.reset_index(drop=True)


def filter_target_chains(data: pd.DataFrame, target_chains: list[str]) -> pd.DataFrame:
    """
    Restrict protein dataframe to the selected target chains.
    """
    chains = sorted({c.strip() for c in target_chains if c.strip()})
    if not chains:
        raise ValueError("target_chains must contain at least one non-empty chain.")

    filtered = data[data["chain_id"].isin(chains)].reset_index(drop=True)
    if filtered.empty:
        raise ValueError(f"No atoms found for target chains {chains}.")
    return filtered


def annotate_target_atom_ids(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add the target atom class column used by the model.
    Keeps compatibility with the old setup script naming.
    """
    annotated = data.copy()
    annotated["target atom_id"] = annotated.apply(
        lambda row: map_atom(row["atom_name"], row["res_name"]),
        axis=1,
    )
    return annotated


def filter_residue(
    data: pd.DataFrame,
    residue: ResidueSelector,
) -> pd.DataFrame:
    """
    Restrict dataframe to a single residue selection.
    res_num is matched as string, consistent with create_dataframe().
    """
    residue = residue.normalized()

    filtered = data[
        (data["chain_id"] == residue.chain_id)
        & (data["res_num"].astype(str).str.strip() == residue.res_num)
    ]

    if residue.res_name is not None:
        filtered = filtered[filtered["res_name"] == residue.res_name]

    filtered = filtered.reset_index(drop=True)

    if filtered.empty:
        if residue.res_name is None:
            raise ValueError(
                f"No atoms found for residue selection chain={residue.chain_id}, "
                f"res_num={residue.res_num}."
            )
        raise ValueError(
            f"No atoms found for residue selection chain={residue.chain_id}, "
            f"res_num={residue.res_num}, res_name={residue.res_name}."
        )

    return filtered


# def get_box_dataframe(
#    target_data: pd.DataFrame,
#    box_source: BoxSource,
#    ligand_path: Optional[Path] = None,
# ) -> pd.DataFrame:
#    """
#    Select the atom coordinates that define the bounding box for the voxel grid.
#
#    Modes:
#      - ligand: load ligand PDB and use its coordinates
#      - chains: use all atoms in target_data
#      - residue: use only the selected residue atoms from target_data
#    """
#    box_source = box_source.normalized()
#
#    if box_source.mode == "ligand":
#        if ligand_path is None:
#            raise ValueError("ligand_path is required when box_source.mode='ligand'.")
#        ligand_df = load_pdb_dataframe(ligand_path)
#        return ligand_df
#
#    if box_source.mode == "chains":
#        return target_data.reset_index(drop=True)
#
#    if box_source.mode == "residue":
#        if box_source.residue is None:
#            raise ValueError("box_source.mode='residue' requires a residue selector.")
#        return filter_residue(target_data, box_source.residue)
#
#    raise RuntimeError(f"Unhandled box source mode: {box_source.mode}")


def get_box_dataframe(
    target_data: pd.DataFrame,
    box_source: BoxSource,
    ligand_path: Path | None = None,
) -> pd.DataFrame:
    """
    Select the atom coordinates that define the bounding box for the voxel grid.

    Modes:
      - ligand: use all atoms from ligand_path
      - chains: use all atoms in target_data
      - residue: use the selected residue, preferably from ligand_path if provided,
                 otherwise from target_data
    """
    box_source = box_source.normalized()

    if box_source.mode == "ligand":
        if ligand_path is None:
            raise ValueError("ligand_path is required when box_source.mode='ligand'.")
        ligand_df = load_pdb_dataframe(ligand_path)
        return ligand_df

    if box_source.mode == "chains":
        return target_data.reset_index(drop=True)

    if box_source.mode == "residue":
        if box_source.residue is None:
            raise ValueError("box_source.mode='residue' requires a residue selector.")

        # Prefer ligand structure for residue-box selection when available.
        if ligand_path is not None:
            ligand_df = load_pdb_dataframe(ligand_path)
            try:
                return filter_residue(ligand_df, box_source.residue)
            except ValueError:
                pass

        # Fallback: try the receptor/target dataframe.
        return filter_residue(target_data, box_source.residue)

    raise RuntimeError(f"Unhandled box source mode: {box_source.mode}")


# ----------------------------
# Filesystem / output materialization
# ----------------------------


def create_output_dirs(output_dir: Path, overwrite: bool = False) -> tuple[Path, Path, Path]:
    """
    Create output_dir with setup/, inference/, model/ subdirectories.
    """
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}\n"
                f"Use a different run_id or enable overwrite."
            )

    output_dir.mkdir(parents=True, exist_ok=False)

    setup_dir = output_dir / "setup"
    infer_dir = output_dir / "inference"
    model_dir = output_dir / "model"

    setup_dir.mkdir()
    infer_dir.mkdir()
    model_dir.mkdir()

    return setup_dir, infer_dir, model_dir


def materialize_model_logs(
    model_logs: Path,
    model_dir: Path,
    mode: str = "symlink",
) -> ModelMaterialized:
    """
    Materialize the model logs into the job directory, or only record the path.

    mode:
      - symlink
      - copy
      - path_only
    """
    validate_dir_exists(model_logs, "Model logs directory")

    mode = mode.strip().lower()
    if mode not in {"symlink", "copy", "path_only"}:
        raise ValueError("mode must be one of: symlink, copy, path_only")

    if mode == "path_only":
        return ModelMaterialized(
            mode="path_only",
            path=str(model_logs.resolve()),
            ok=True,
        )

    dst = model_dir / model_logs.name

    if mode == "copy":
        shutil.copytree(model_logs, dst, dirs_exist_ok=True)
        return ModelMaterialized(
            mode="copy",
            path=str(dst.resolve()),
            ok=True,
        )

    ok = try_symlink(model_logs, dst, is_dir=True)
    if ok:
        return ModelMaterialized(
            mode="symlink",
            path=str(dst.resolve()),
            ok=True,
        )

    return ModelMaterialized(
        mode="path_only",
        path=str(model_logs.resolve()),
        ok=False,
    )


# ----------------------------
# Main reusable setup entrypoint
# ----------------------------


def setup_mapping_job(
    *,
    pdb_path: Path,
    target_chains: list[str],
    map_name: str,
    model_logs: Path,
    outdir: Path,
    voxel_size: float = 0.5,
    padding: float = 1.0,
    run_id: str | None = None,
    ligand_path: Path | None = None,
    box_source: BoxSource | None = None,
    model_link_mode: str = "symlink",
    overwrite: bool = False,
) -> MappingJob:
    """
    Prepare a tp_mapping job directory and write all setup artifacts.

    Default behavior matches the old standalone script:
      - box_source defaults to ligand mode
      - ligand_path is required in ligand mode
      - target atoms come from selected receptor chains
    """
    validate_file_exists(pdb_path, "Receptor PDB")
    validate_dir_exists(model_logs, "Model logs directory")

    map_name = str(map_name).strip()
    if not map_name:
        raise ValueError("map_name must not be empty.")

    voxel_size = float(voxel_size)
    padding = float(padding)
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0.")
    if padding < 0:
        raise ValueError("padding must be >= 0.")

    box_source = box_source.normalized() if box_source is not None else BoxSource(mode="ligand")

    if box_source.mode == "ligand":
        if ligand_path is None:
            raise ValueError("ligand_path is required when box_source.mode='ligand'.")
        validate_file_exists(ligand_path, "Ligand PDB")

    model_id = make_model_id(model_logs)
    run_id = make_run_id(run_id)

    output_dir = make_output_dir(
        outdir=outdir,
        pdb_path=pdb_path,
        target_chains=target_chains,
        ligand_path=ligand_path,
        model_id=model_id,
        map_name=map_name,
        run_id=run_id,
        box_source=box_source,
    )

    setup_dir, infer_dir, model_dir = create_output_dirs(output_dir, overwrite=overwrite)

    model_materialized = materialize_model_logs(
        model_logs=model_logs,
        model_dir=model_dir,
        mode=model_link_mode,
    )

    # Load receptor and filter to target chains
    receptor_df = load_pdb_dataframe(pdb_path)
    target_df = filter_target_chains(receptor_df, target_chains)
    target_df = annotate_target_atom_ids(target_df)

    if target_df.empty:
        raise ValueError(
            f"No atoms found after filtering receptor to target chains {target_chains}."
        )

    # Protein atom cloud used for neighbor search during prediction
    atom_cloud = target_df[["x", "y", "z"]].to_numpy(dtype=np.float32)

    # Box-defining coordinates
    box_df = get_box_dataframe(
        target_data=target_df,
        box_source=box_source,
        ligand_path=ligand_path,
    )

    if box_df.empty:
        raise ValueError("Box-defining dataframe is empty; cannot construct voxel grid.")

    (grid_x, grid_y, grid_z), voxel_centers = voxel_grid(
        data_xyz=box_df[["x", "y", "z"]],
        voxel_size=voxel_size,
        padding=padding,
    )

    # Save setup artifacts
    np.save(setup_dir / "atom_cloud.npy", atom_cloud)
    np.save(setup_dir / "voxel_centers.npy", voxel_centers)
    np.save(setup_dir / "grid_x.npy", grid_x)
    np.save(setup_dir / "grid_y.npy", grid_y)
    np.save(setup_dir / "grid_z.npy", grid_z)
    target_df.to_pickle(setup_dir / "data_target.pkl")

    job = build_mapping_job(
        pdb_path=pdb_path,
        target_chains=target_chains,
        ligand_path=ligand_path,
        box_source=box_source,
        voxel_size=voxel_size,
        padding=padding,
        model_logs=model_logs,
        model_materialized=model_materialized,
        outdir=outdir,
        output_dir=output_dir,
        setup_dir=setup_dir,
        inference_dir=infer_dir,
        model_dir=model_dir,
        map_name=map_name,
        run_id=run_id,
    )

    job.write(output_dir / "job.json")

    # Optional compatibility file for older code paths
    setup_params_compat = {
        "PDB": str(pdb_path.resolve()),
        "Target_chain": list(target_chains),
        "Model": str(model_logs.resolve()),
        "VOXEL_SIZE": voxel_size,
        "PADDING": padding,
        "BOX_SOURCE_MODE": box_source.mode,
    }
    if ligand_path is not None:
        setup_params_compat["LIGAND"] = str(ligand_path.resolve())
    if box_source.mode == "residue" and box_source.residue is not None:
        setup_params_compat["BOX_RESIDUE"] = box_source.residue.to_dict()

    # from proteintpm.tp_mapping.jobs import write_json
    write_json(output_dir / "setup_params.json", setup_params_compat)

    return job
