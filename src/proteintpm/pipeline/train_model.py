import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tensorflow import keras

from proteintpm.config.training import DEFAULT_SHARDS, build_training_params
from proteintpm.ml.augment import duplicate_and_augment
from proteintpm.ml.datasets import build_fixed_val_set, build_pipeline, build_train_ds_for_epoch
from proteintpm.ml.model_builder import create_model
from proteintpm.ml.negative_sampling import build_random_broom_splits
from proteintpm.ml.normalizers import build_normalizer, build_normalizer_mix_dataset


def get_run_logdir(log_dir: Path, shard: str, fold: int = -1) -> Path:
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return log_dir / shard / f"fold{fold + 1}_{run_id}"


def resolve_ova_shards(shard_pos: str, all_shards: list[str] | None = None) -> list[str]:
    """
    Replace all shards that share the same prefix with the target shard so the
    one-vs-all negative pool excludes the entire target class family.
    """
    if all_shards is None:
        all_shards = DEFAULT_SHARDS

    prefix = shard_pos.split("_")[0]
    return [shard_pos if s.split("_")[0] == prefix else s for s in all_shards]


def save_hyperparameters(params: dict, output_path: Path) -> None:
    serializable = {
        key: str(value) if isinstance(value, Path) else value for key, value in params.items()
    }
    with output_path.open("w") as f:
        json.dump(serializable, f, indent=2)


def save_history(history_records: list[dict], output_path: Path) -> pd.DataFrame:
    df = pd.DataFrame(history_records)
    df.to_csv(output_path, index=False)
    return df


def build_summary(
    shard: str,
    history_df: pd.DataFrame,
    params: dict,
) -> dict:
    if history_df.empty:
        return {
            "shard": shard,
            "epochs_completed": 0,
            "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in params.items()},
        }

    best_idx = history_df["val_loss"].idxmin()
    best_row = history_df.loc[best_idx].to_dict()
    final_row = history_df.iloc[-1].to_dict()

    return {
        "shard": shard,
        "epochs_completed": int(len(history_df)),
        "best_epoch_by_val_loss": int(best_row["epoch"]),
        "best_metrics": {
            k: float(v) if isinstance(v, (np.floating, float, int)) and pd.notna(v) else v
            for k, v in best_row.items()
        },
        "final_metrics": {
            k: float(v) if isinstance(v, (np.floating, float, int)) and pd.notna(v) else v
            for k, v in final_row.items()
        },
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in params.items()},
    }


def save_summary(summary: dict, output_path: Path) -> None:
    with output_path.open("w") as f:
        json.dump(summary, f, indent=2)


def run_train_model(
    shard: str,
    input_dir: Path,
    random_dir: Path,
    log_dir: Path,
    max_neighbours: int = 25,
    num_rotations: int = 9,
    batch_size: int = 24,
    learning_rate: float = 1e-4,
    epochs: int = 15,
    patience: int = 5,
    latent_dim: int = 32,
    normalizer_samples: int = 20000,
    subset_size: int | None = None,
    seed: int = 123,
    split_seed: int = 42,
) -> None:
    shard_pos = shard
    # shards_all = resolve_ova_shards(shard_pos)
    shards_all = DEFAULT_SHARDS

    params = build_training_params(
        input_dir=input_dir,
        random_dir=random_dir,
        max_neighbours=max_neighbours,
        num_rotations=num_rotations,
        batch_size=batch_size,
        learning_rate=learning_rate,
        epochs=epochs,
        patience=patience,
        latent_dim=latent_dim,
        normalizer_samples=normalizer_samples,
        subset_size=subset_size,
    )

    allowed_random_train, allowed_random_val = build_random_broom_splits(
        random_dir=Path(params["RANDOM_DIR"]),
        seed=split_seed,
    )

    rng = np.random.default_rng(seed)

    ch1_val, ch2_val, lbl_val = build_fixed_val_set(
        all_shards=shards_all,
        shard_pos=shard_pos,
        params=params,
        random_dir=Path(params["RANDOM_DIR"]),
        allowed_random_val=allowed_random_val,
        rng=rng,
    )

    val_ds = build_pipeline(
        ch1=ch1_val,
        ch2=ch2_val,
        labels=lbl_val,
        num_rotations=1,
        batch_size=params["BATCH_SIZE"],
        augment_fn=lambda x, y, z: duplicate_and_augment(x, y, z, 0),
        filter_fn=None,
        shuffle=False,
        cache=True,
        repeat=False,
    )

    norm_ds = build_normalizer_mix_dataset(
        shard_pos=shard_pos,
        all_shards=shards_all,
        params=params,
        random_dir=Path(params["RANDOM_DIR"]),
        allowed_random_train=allowed_random_train,
        target_samples=params["normalizer_samples"],
        rng=np.random.default_rng(seed + 1),
    )

    norm_dist_layer = build_normalizer(
        dataset=norm_ds,
        sample_count=params["normalizer_samples"],
    )

    model = create_model(
        normlayer=norm_dist_layer,
        latent_dim=params["latent_dim"],
    )
    model.compile(
        optimizer=keras.optimizers.Adam(params["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.AUC(name="auc"),
        ],
    )

    run_logdir = get_run_logdir(Path(log_dir), shard_pos)
    run_logdir.mkdir(parents=True, exist_ok=True)

    save_hyperparameters(params, run_logdir / "hyperparameters.json")

    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=run_logdir,
        histogram_freq=0,
        write_graph=True,
    )
    ckpt_cb = keras.callbacks.ModelCheckpoint(
        filepath=run_logdir / f"best_{shard_pos}_model.keras",
        save_best_only=True,
        monitor="val_loss",
        mode="min",
    )

    best = float("inf")
    wait = 0
    history_records: list[dict] = []

    for epoch in range(params["epochs"]):
        train_ds, steps_per_epoch = build_train_ds_for_epoch(
            shard_pos=shard_pos,
            all_shards=shards_all,
            params=params,
            random_dir=Path(params["RANDOM_DIR"]),
            allowed_random_train=allowed_random_train,
            rng=rng,
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epoch + 1,
            initial_epoch=epoch,
            steps_per_epoch=steps_per_epoch,
            callbacks=[tensorboard_cb, ckpt_cb],
            verbose=2,
        )

        record = {"epoch": epoch + 1}
        for key, values in history.history.items():
            record[key] = values[-1]
        history_records.append(record)

        val_loss = history.history["val_loss"][-1]
        if val_loss < best:
            best = val_loss
            wait = 0
        else:
            wait += 1

        if wait >= params["patience"]:
            print(f"Early stopping at epoch {epoch + 1} with best val_loss={best:.6f}")
            break

    history_df = save_history(history_records, run_logdir / "history.csv")
    summary = build_summary(shard=shard_pos, history_df=history_df, params=params)
    save_summary(summary, run_logdir / "summary.json")

    model.save(run_logdir / "final_model.keras")
    print(f"Finished training for shard {shard_pos}")
    print(f"Artifacts written to: {run_logdir}")
