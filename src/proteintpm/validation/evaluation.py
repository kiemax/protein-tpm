import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from proteintpm.validation.predict import predict_and_save_for_dataset

DatasetType = Literal["split_csv", "explicit_ids"]


@dataclass(frozen=True)
class EvaluationDatasetConfig:
    name: str
    type: DatasetType
    input_dir: Path
    index_csv: Path
    sample_id_column: str = "pdb_id"
    selection_column: str | None = None
    selection_value: str | None = None


@dataclass(frozen=True)
class EvaluationConfig:
    model_config: Path
    dataset_config: Path
    output_root: Path
    run_name: str
    max_neighbours: int = 25
    batch_size: int = 256
    threshold: float = 0.5
    overwrite: bool = False
    run_predict: bool = True
    run_descriptors: bool = True
    run_motifs: bool = True


def config_from_args(args: Any) -> EvaluationConfig:
    requested = {
        "predict": bool(getattr(args, "predict", False)),
        "descriptors": bool(getattr(args, "descriptors", False)),
        "chemical_motifs": bool(getattr(args, "chemical_motifs", False)),
        "all": bool(getattr(args, "all", False)),
    }

    if requested["all"] or not any(requested.values()):
        run_predict = True
        run_descriptors = True
        run_motifs = True
    else:
        run_predict = requested["predict"]
        run_descriptors = requested["descriptors"]
        run_motifs = requested["chemical_motifs"]

    return EvaluationConfig(
        model_config=Path(args.model_config),
        dataset_config=Path(args.dataset_config),
        output_root=Path(args.output_root),
        run_name=args.run_name,
        max_neighbours=args.max_neighbours,
        batch_size=args.batch_size,
        threshold=args.threshold,
        overwrite=args.overwrite,
        run_predict=run_predict,
        run_descriptors=run_descriptors,
        run_motifs=run_motifs,
    )


def load_model_map(path: Path) -> dict[str, str]:
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Model config must be a JSON object mapping shard -> run_dir.")

    for shard, run_dir in data.items():
        if not isinstance(shard, str) or not isinstance(run_dir, str):
            raise ValueError("Model config must be dict[str, str].")

    return data


def load_dataset_config(path: Path) -> EvaluationDatasetConfig:
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Dataset config must be a JSON object.")

    dataset_type = data.get("type")
    if dataset_type not in {"split_csv", "explicit_ids"}:
        raise ValueError("Dataset config 'type' must be one of: 'split_csv', 'explicit_ids'.")

    selection = data.get("selection", {})
    if selection is None:
        selection = {}

    config = EvaluationDatasetConfig(
        name=data["name"],
        type=dataset_type,
        input_dir=Path(data["input_dir"]),
        index_csv=Path(data["index_csv"]),
        sample_id_column=data.get("sample_id_column", "pdb_id"),
        selection_column=selection.get("column"),
        selection_value=selection.get("value"),
    )

    validate_dataset_config(config)
    return config


def validate_dataset_config(config: EvaluationDatasetConfig) -> None:
    if config.type == "split_csv":
        if not config.selection_column or config.selection_value is None:
            raise ValueError(
                "split_csv dataset config requires selection.column and selection.value."
            )

    if config.type == "explicit_ids":
        if config.selection_column is not None or config.selection_value is not None:
            raise ValueError("explicit_ids dataset config must not define selection.column/value.")


def prepare_run_dir(output_root: Path, run_name: str, overwrite: bool) -> Path:
    run_dir = output_root / run_name
    if run_dir.exists():
        if overwrite:
            shutil.rmtree(run_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def build_model_path(model_run_dir: Path, shard: str) -> Path:
    return model_run_dir / f"best_{shard}_model.keras"


def determine_output_path(run_dir: Path, shard: str) -> Path:
    return run_dir / shard / "predictions.csv"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def copy_run_inputs(
    *,
    model_config: Path,
    dataset_config: Path,
    dataset_index_csv: Path,
    run_dir: Path,
) -> None:
    shutil.copy2(model_config, run_dir / "model_config.json")
    shutil.copy2(dataset_config, run_dir / "dataset_config.json")
    shutil.copy2(dataset_index_csv, run_dir / dataset_index_csv.name)


def summarize_prediction_df(pred_df: pd.DataFrame) -> dict[str, Any]:
    y_true = pred_df["y_true"]
    return {
        "rows": int(len(pred_df)),
        "positives": int((y_true == 1).sum()),
        "negatives": int((y_true == 0).sum()),
        "columns": list(pred_df.columns),
    }


def build_run_manifest(
    config: EvaluationConfig,
    dataset: EvaluationDatasetConfig,
    model_map: dict[str, str],
) -> dict[str, Any]:
    return {
        "run_name": config.run_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "evaluation_mode": "dataset_config",
        "model_config": str(config.model_config.resolve()),
        "dataset_config": str(config.dataset_config.resolve()),
        "dataset_name": dataset.name,
        "dataset_type": dataset.type,
        "input_dir": str(dataset.input_dir.resolve()),
        "index_csv": str(dataset.index_csv.resolve()),
        "sample_id_column": dataset.sample_id_column,
        "selection_column": dataset.selection_column,
        "selection_value": dataset.selection_value,
        "output_root": str(config.output_root.resolve()),
        "max_neighbours": config.max_neighbours,
        "batch_size": config.batch_size,
        "threshold": config.threshold,
        "run_predict": config.run_predict,
        "run_descriptors": config.run_descriptors,
        "run_motifs": config.run_motifs,
        "models": model_map,
    }


def evaluate_single_model(
    *,
    shard: str,
    model_run_dir: Path,
    run_dir: Path,
    config: EvaluationConfig,
    dataset: EvaluationDatasetConfig,
    sample_ids: list[str],
    all_shards: list[str],
) -> dict[str, Any]:
    model_path = build_model_path(model_run_dir, shard)
    output_csv = determine_output_path(run_dir, shard)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pred_df = predict_and_save_for_dataset(
        model_shard=shard,
        model_path=model_path,
        input_dir=dataset.input_dir,
        sample_ids=sample_ids,
        max_neighbours=config.max_neighbours,
        output_csv=output_csv,
        all_shards=all_shards,
        batch_size=config.batch_size,
        threshold=config.threshold,
    )

    return {
        "shard": shard,
        "model_run_dir": str(model_run_dir.resolve()),
        "model_path": str(model_path.resolve()),
        "output_csv": str(output_csv.resolve()),
        **summarize_prediction_df(pred_df),
    }


def resolve_sample_ids(dataset: EvaluationDatasetConfig) -> list[str]:
    df = pd.read_csv(dataset.index_csv)

    if dataset.sample_id_column not in df.columns:
        raise ValueError(
            f"Dataset index CSV {dataset.index_csv} does not contain sample_id_column={dataset.sample_id_column!r}."
        )

    if dataset.type == "split_csv":
        if dataset.selection_column is None or dataset.selection_value is None:
            raise ValueError("split_csv dataset requires selection_column and selection_value.")

        if dataset.selection_column not in df.columns:
            raise ValueError(
                f"Dataset index CSV {dataset.index_csv} does not contain selection_column={dataset.selection_column!r}."
            )

        selected = df[df[dataset.selection_column] == dataset.selection_value]
    elif dataset.type == "explicit_ids":
        selected = df
    else:
        raise ValueError(f"Unsupported dataset type: {dataset.type}")

    sample_ids = selected[dataset.sample_id_column].dropna().astype(str).drop_duplicates().tolist()

    if not sample_ids:
        raise ValueError(f"Resolved dataset {dataset.name!r} contains no sample IDs.")

    return sample_ids


def prediction_csv_paths(run_dir: Path, model_map: dict[str, str]) -> dict[str, Path]:
    return {shard: determine_output_path(run_dir, shard) for shard in model_map}


def ensure_prediction_outputs_exist(run_dir: Path, model_map: dict[str, str]) -> None:
    missing = []
    for shard, csv_path in prediction_csv_paths(run_dir, model_map).items():
        if not csv_path.is_file():
            missing.append(f"{shard}: {csv_path}")

    if missing:
        preview = missing[:10]
        raise FileNotFoundError(
            "Descriptor/motif stages require existing per-model predictions.csv files "
            f"in the run directory. Missing: {preview}"
        )


def build_or_refresh_set_metadata(run_dir: Path) -> dict[str, Any]:
    from proteintpm.validation.dataset_features import compute_and_save_set_metadata

    metadata_path = compute_and_save_set_metadata(run_dir=run_dir)
    meta = pd.read_csv(metadata_path)

    return {
        "requested": True,
        "output_csv": str(Path(metadata_path).resolve()),
        "rows": int(len(meta)),
        "columns": list(meta.columns),
    }


def try_compute_set_descriptors(*, run_dir: Path) -> Path:
    from proteintpm.validation.dataset_features import compute_and_save_set_descriptors

    return compute_and_save_set_descriptors(run_dir=run_dir)


def try_compute_set_chemical_motifs(*, run_dir: Path) -> Path:
    from proteintpm.validation.dataset_features import compute_and_save_set_chemical_motifs

    return compute_and_save_set_chemical_motifs(run_dir=run_dir)


def build_run_summary(
    *,
    run_name: str,
    run_dir: Path,
    dataset: EvaluationDatasetConfig,
    model_map: dict[str, str],
    successes: list[dict[str, Any]],
    failures: list[dict[str, str]],
    stage_status: dict[str, Any],
) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "run_dir": str(run_dir.resolve()),
        "dataset_name": dataset.name,
        "dataset_type": dataset.type,
        "n_models_total": len(model_map),
        "n_models_success": len(successes),
        "n_models_failed": len(failures),
        "successes": successes,
        "failures": failures,
        "stages": stage_status,
    }


def run_evaluation(config: EvaluationConfig) -> None:
    model_map = load_model_map(config.model_config)
    dataset = load_dataset_config(config.dataset_config)
    shards = list(model_map.keys())
    sample_ids = resolve_sample_ids(dataset)

    run_dir = prepare_run_dir(config.output_root, config.run_name, config.overwrite)
    copy_run_inputs(
        model_config=config.model_config,
        dataset_config=config.dataset_config,
        dataset_index_csv=dataset.index_csv,
        run_dir=run_dir,
    )

    manifest = build_run_manifest(config, dataset, model_map)
    manifest["n_samples"] = len(sample_ids)
    write_json(run_dir / "run_config.json", manifest)

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    stage_status: dict[str, Any] = {}

    if config.run_predict:
        for shard, model_dir_str in model_map.items():
            model_run_dir = Path(model_dir_str)
            try:
                result = evaluate_single_model(
                    shard=shard,
                    model_run_dir=model_run_dir,
                    run_dir=run_dir,
                    config=config,
                    dataset=dataset,
                    sample_ids=sample_ids,
                    all_shards=shards,
                )
                successes.append(result)
            except Exception as exc:
                failures.append(
                    {
                        "shard": shard,
                        "model_run_dir": str(model_run_dir),
                        "model_path": str(build_model_path(model_run_dir, shard)),
                        "error": str(exc),
                    }
                )

        stage_status["predict"] = {
            "requested": True,
            "n_success": len(successes),
            "n_failed": len(failures),
        }
    else:
        stage_status["predict"] = {"requested": False}

    need_set_level = config.run_descriptors or config.run_motifs

    if need_set_level:
        ensure_prediction_outputs_exist(run_dir, model_map)

        try:
            stage_status["set_metadata"] = build_or_refresh_set_metadata(run_dir)
        except Exception as exc:
            stage_status["set_metadata"] = {
                "requested": True,
                "status": "failed",
                "error": str(exc),
            }
            summary = build_run_summary(
                run_name=config.run_name,
                run_dir=run_dir,
                dataset=dataset,
                model_map=model_map,
                successes=successes,
                failures=failures,
                stage_status=stage_status,
            )
            summary["n_samples"] = len(sample_ids)
            write_json(run_dir / "summary.json", summary)
            raise

    else:
        stage_status["set_metadata"] = {"requested": False}

    if config.run_descriptors:
        try:
            descriptors_path = try_compute_set_descriptors(run_dir=run_dir)
            stage_status["descriptors"] = {
                "requested": True,
                "status": "ok",
                "output_csv": str(Path(descriptors_path).resolve()),
            }
        except Exception as exc:
            stage_status["descriptors"] = {
                "requested": True,
                "status": "failed",
                "error": str(exc),
            }
    else:
        stage_status["descriptors"] = {"requested": False}

    if config.run_motifs:
        try:
            motifs_path = try_compute_set_chemical_motifs(run_dir=run_dir)
            stage_status["chemical_motifs"] = {
                "requested": True,
                "status": "ok",
                "output_csv": str(Path(motifs_path).resolve()),
            }
        except Exception as exc:
            stage_status["chemical_motifs"] = {
                "requested": True,
                "status": "failed",
                "error": str(exc),
            }
    else:
        stage_status["chemical_motifs"] = {"requested": False}

    summary = build_run_summary(
        run_name=config.run_name,
        run_dir=run_dir,
        dataset=dataset,
        model_map=model_map,
        successes=successes,
        failures=failures,
        stage_status=stage_status,
    )
    summary["n_samples"] = len(sample_ids)
    write_json(run_dir / "summary.json", summary)


def run_evaluation_from_args(args: Any) -> None:
    config = config_from_args(args)
    run_evaluation(config)
