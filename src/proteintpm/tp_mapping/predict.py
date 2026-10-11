import json
import os
from pathlib import Path
import gc

# import mrcfile
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.spatial import cKDTree

from proteintpm.tp_mapping.jobs import MappingJob, load_json, resolve_job_dir
from proteintpm.tp_mapping.mrc import write_mrc

# ----------------------------
# Small helpers
# ----------------------------


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def configure_device(device: str) -> None:
    """
    device:
      - 'cpu'
      - 'gpu' (use all visible GPUs)
      - 'gpu:0' (use only GPU 0)
      - 'gpu:1' (use only GPU 1), etc.

    Must be called before TensorFlow initializes GPU state.
    """
    device = device.strip().lower()

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return

    if device == "gpu":
        return

    if device.startswith("gpu:"):
        idx = device.split(":", 1)[1].strip()
        if idx == "":
            raise ValueError("Invalid device format. Use gpu:0, gpu:1, ...")
        os.environ["CUDA_VISIBLE_DEVICES"] = idx
        return

    raise ValueError("Invalid device. Use one of: cpu | gpu | gpu:<index>")


def find_run_dir(model_logs_dir: Path) -> Path:
    """
    Compatibility with the old model directory convention.

    If model_logs_dir itself contains hyperparameters.json, treat it as the run dir.
    Otherwise, expect exactly one subdirectory whose name contains 'run'.
    """
    if (model_logs_dir / "hyperparameters.json").exists():
        return model_logs_dir

    run_dirs = [p for p in model_logs_dir.iterdir() if p.is_dir() and "run" in p.name]

    if len(run_dirs) != 1:
        raise RuntimeError(
            f"Expected exactly one run folder in {model_logs_dir}, found {len(run_dirs)}: "
            f"{[p.name for p in run_dirs]}"
        )

    return run_dirs[0]


def load_model_from_run_dir(run_dir: Path) -> tuple[tf.keras.Model, dict]:
    """
    Load model + hyperparameters from a run directory.

    Selection policy:
      - list *.keras in run_dir
      - ignore 'final_model.keras'
      - if one candidate remains -> load it
      - if multiple remain -> load newest by mtime
      - if none remain -> error
    """
    hp_path = run_dir / "hyperparameters.json"
    hparams = load_json(hp_path) if hp_path.exists() else {}

    keras_files = sorted(run_dir.glob("*.keras"))
    if not keras_files:
        raise FileNotFoundError(f"No *.keras files found in {run_dir}")

    candidates = [p for p in keras_files if p.name != "final_model.keras"]
    if not candidates:
        raise FileNotFoundError(
            f"Only found final_model.keras in {run_dir}, but it is excluded by policy."
        )

    if len(candidates) == 1:
        model_path = candidates[0]
    else:
        model_path = max(candidates, key=lambda p: p.stat().st_mtime)
        print(
            f"[tp_mapping] multiple candidate models found (excluding final_model.keras): "
            f"{[p.name for p in candidates]}. choosing newest: {model_path.name}"
        )

    model = tf.keras.models.load_model(model_path)
    return model, hparams


def get_max_neighbours(hparams: dict, run_dir: Path) -> int:
    """
    Extract MAX_NEIGHBOURS from hyperparameters.
    """
    if "MAX_NEIGHBOURS" not in hparams:
        raise KeyError(
            f"MAX_NEIGHBOURS not found in {run_dir / 'hyperparameters.json'}. "
            "Please add it or define a fallback policy."
        )

    k = int(hparams["MAX_NEIGHBOURS"])
    if k <= 0:
        raise ValueError(f"MAX_NEIGHBOURS must be > 0, got {k}.")

    return k


# ----------------------------
# Job / legacy layout resolution
# ----------------------------


def resolve_prediction_layout(
    *,
    job: str | None = None,
    mapdata_path: str | None = None,
) -> dict:
    """
    Resolve either:
      - new tp_mapping job layout (job.json), or
      - legacy setup directory layout

    Returns a dict with all paths needed for prediction.
    """
    job_dir = resolve_job_dir(job, mapdata_path)
    job_json_path = job_dir / "job.json"
    is_new_layout = job_json_path.exists()

    if is_new_layout:
        mapping_job = MappingJob.from_json(job_json_path)

        setup_dir = Path(mapping_job.outputs.setup_dir)
        infer_dir = Path(mapping_job.outputs.inference_dir)
        map_name = mapping_job.outputs.map_name
        voxel_size = float(mapping_job.grid.voxel_size)
        model_logs = Path(mapping_job.model.model_logs)

        atom_cloud_path = setup_dir / "atom_cloud.npy"
        voxel_centers_path = setup_dir / "voxel_centers.npy"
        grid_x_path = setup_dir / "grid_x.npy"
        grid_y_path = setup_dir / "grid_y.npy"
        grid_z_path = setup_dir / "grid_z.npy"
        data_target_path = setup_dir / "data_target.pkl"

        return {
            "is_new_layout": True,
            "job_dir": job_dir,
            "job": mapping_job,
            "setup_dir": setup_dir,
            "infer_dir": infer_dir,
            "map_name": map_name,
            "voxel_size": voxel_size,
            "model_logs": model_logs,
            "atom_cloud_path": atom_cloud_path,
            "voxel_centers_path": voxel_centers_path,
            "grid_x_path": grid_x_path,
            "grid_y_path": grid_y_path,
            "grid_z_path": grid_z_path,
            "data_target_path": data_target_path,
        }

    # legacy layout
    setup_dir = job_dir
    infer_dir = job_dir
    setup_params_path = job_dir / "setup_params.json"
    if not setup_params_path.exists():
        raise FileNotFoundError(f"Legacy layout requires setup_params.json in {job_dir}.")

    setup_params = load_json(setup_params_path)

    voxel_size = float(setup_params["VOXEL_SIZE"])
    model_logs = Path(setup_params["Model"])
    map_name = setup_params.get("MAP_NAME", "my_map")

    return {
        "is_new_layout": False,
        "job_dir": job_dir,
        "job": None,
        "setup_dir": setup_dir,
        "infer_dir": infer_dir,
        "map_name": map_name,
        "voxel_size": voxel_size,
        "model_logs": model_logs,
        "atom_cloud_path": setup_dir / "atom_cloud.npy",
        "voxel_centers_path": setup_dir / "voxel_centers.npy",
        "grid_x_path": setup_dir / "grid_x.npy",
        "grid_y_path": setup_dir / "grid_y.npy",
        "grid_z_path": setup_dir / "grid_z.npy",
        "data_target_path": setup_dir / "data_target.pkl",
    }


# ----------------------------
# Feature construction
# ----------------------------


def load_setup_artifacts(layout: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load the setup artifacts required for prediction.
    """
    point_cloud = np.load(layout["atom_cloud_path"])
    voxel_centers = np.load(layout["voxel_centers_path"])
    data_target = pd.read_pickle(layout["data_target_path"])

    return point_cloud, voxel_centers, data_target


def build_neighbour_features(
    point_cloud: np.ndarray,
    voxel_centers: np.ndarray,
    data_target: pd.DataFrame,
    k: int,
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Build model inputs from protein coordinates and voxel centers.

    Returns:
      positions_tf: (N, K, 3)
      classes_tf:   (N, K, 1)
    """
    if point_cloud.ndim != 2 or point_cloud.shape[1] != 3:
        raise ValueError(f"point_cloud must have shape (M, 3), got {point_cloud.shape}")

    if voxel_centers.ndim != 2 or voxel_centers.shape[1] != 3:
        raise ValueError(f"voxel_centers must have shape (N, 3), got {voxel_centers.shape}")

    if len(point_cloud) == 0:
        raise ValueError("point_cloud is empty.")
    if len(voxel_centers) == 0:
        raise ValueError("voxel_centers is empty.")

    protein_xyz = data_target[["x", "y", "z"]].to_numpy(dtype=np.float32)
    protein_cls = data_target["target atom_id"].to_numpy(dtype=np.float32)

    if len(protein_xyz) != len(point_cloud):
        raise ValueError(
            f"Mismatch between point_cloud ({len(point_cloud)}) and data_target ({len(protein_xyz)})."
        )

    if k > len(point_cloud):
        raise ValueError(
            f"Requested k={k} neighbours, but point cloud only has {len(point_cloud)} atoms."
        )

    tree = cKDTree(point_cloud)
    _, neighbors = tree.query(voxel_centers, k=k)

    # cKDTree returns shape (N,) if k == 1, normalize to (N, 1)
    if k == 1:
        neighbors = neighbors[:, np.newaxis]

    nbr_xyz = protein_xyz[neighbors]  # (N, K, 3)
    nbr_cls = protein_cls[neighbors][..., np.newaxis]  # (N, K, 1)

    positions_np = nbr_xyz - voxel_centers[:, None, :]  # (N, K, 3)
    classes_np = nbr_cls  # (N, K, 1)

    positions_tf = tf.convert_to_tensor(positions_np, dtype=tf.float32)
    classes_tf = tf.convert_to_tensor(classes_np, dtype=tf.float32)

    return positions_tf, classes_tf


def predict_probabilities(
    model: tf.keras.Model,
    positions_tf: tf.Tensor,
    classes_tf: tf.Tensor,
    batch_size: int = 0,
) -> np.ndarray:
    """
    Run model prediction and return a flat float32 array.
    """
    if batch_size and batch_size > 0:
        probs = model.predict(
            [positions_tf, classes_tf],
            batch_size=batch_size,
            verbose=0,
        ).reshape(-1)
    else:
        probs = model.predict(
            [positions_tf, classes_tf],
            verbose=0,
        ).reshape(-1)

    return probs.astype(np.float32)


# ----------------------------
# Output writing
# ----------------------------


def determine_output_paths(
    *,
    infer_dir: Path,
    map_name: str,
    is_new_layout: bool,
    mrc_name: str | None = None,
) -> tuple[Path, Path, Path]:
    """
    Determine standard prediction output paths.
    """
    if is_new_layout:
        mrc_filename = mrc_name or f"{map_name}.mrc"
    else:
        mrc_filename = mrc_name or "my_map.mrc"

    mrc_path = infer_dir / mrc_filename
    preds_path = infer_dir / "tpm_predictions.npy"
    meta_path = infer_dir / "inference_metadata.json"

    return mrc_path, preds_path, meta_path


def check_overwrite_allowed(
    *,
    overwrite: bool,
    paths: list[Path],
) -> None:
    existing = [p for p in paths if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing inference outputs: {existing}. "
            "Use overwrite=True to replace."
        )


# ----------------------------
# Main reusable entrypoint
# ----------------------------


def run_mapping_job(
    *,
    job: str | None = None,
    mapdata_path: str | None = None,
    device: str = "gpu",
    overwrite: bool = False,
    batch_size: int = 0,
    mrc_name: str | None = None,
) -> dict:
    """
    Run tp_mapping prediction from either:
      - a new job directory / job.json
      - a legacy setup directory

    Returns a metadata dict that is also written to disk.
    """
    configure_device(device)

    layout = resolve_prediction_layout(job=job, mapdata_path=mapdata_path)
    infer_dir = layout["infer_dir"]
    infer_dir.mkdir(parents=True, exist_ok=True)

    mrc_path, preds_path, meta_path = determine_output_paths(
        infer_dir=infer_dir,
        map_name=layout["map_name"],
        is_new_layout=layout["is_new_layout"],
        mrc_name=mrc_name,
    )

    check_overwrite_allowed(
        overwrite=overwrite,
        paths=[mrc_path, preds_path, meta_path],
    )

    run_dir = find_run_dir(layout["model_logs"])
    model, hparams = load_model_from_run_dir(run_dir)
    k = get_max_neighbours(hparams, run_dir)

    point_cloud, voxel_centers, data_target = load_setup_artifacts(layout)

    positions_tf, classes_tf = build_neighbour_features(
        point_cloud=point_cloud,
        voxel_centers=voxel_centers,
        data_target=data_target,
        k=k,
    )

    probabilities = predict_probabilities(
        model=model,
        positions_tf=positions_tf,
        classes_tf=classes_tf,
        batch_size=batch_size,
    )

    np.save(preds_path, probabilities)

    grid_x = np.load(layout["grid_x_path"])
    grid_y = np.load(layout["grid_y_path"])
    grid_z = np.load(layout["grid_z_path"])

    write_mrc(
        mrc_path=mrc_path,
        probabilities=probabilities,
        grid_x=grid_x,
        grid_y=grid_y,
        grid_z=grid_z,
        voxel_size=layout["voxel_size"],
    )

    meta = {
        "job_dir": str(layout["job_dir"].resolve()),
        "setup_dir": str(layout["setup_dir"].resolve()),
        "inference_dir": str(infer_dir.resolve()),
        "model_run_dir": str(run_dir.resolve()),
        "device": device,
        "max_neighbours": k,
        "num_voxels": int(voxel_centers.shape[0]),
        "outputs": {
            "mrc": str(mrc_path.resolve()),
            "predictions": str(preds_path.resolve()),
        },
    }

    write_json(meta_path, meta)
    # release per-job memory
    del model
    del positions_tf, classes_tf, probabilities
    del point_cloud, voxel_centers, data_target
    del grid_x, grid_y, grid_z
    tf.keras.backend.clear_session() 
    gc.collect()

    return meta
