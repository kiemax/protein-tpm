from pathlib import Path

from proteintpm.config.training import DEFAULT_SHARDS
from proteintpm.pipeline.train_model import run_train_model


def run_train_all_models(
    input_dir: Path,
    random_dir: Path,
    log_dir: Path,
    max_neighbours: int = 25,
    num_rotations: int = 12,
    batch_size: int = 24,
    learning_rate: float = 1e-4,
    epochs: int = 15,
    patience: int = 5,
    latent_dim: int = 32,
    normalizer_samples: int = 20000,
    subset_size: int | None = None,
    seed: int = 123,
    split_seed: int = 42,
    shards: list[str] | None = None,
    continue_on_error: bool = False,
) -> None:
    shards_to_train = shards if shards is not None else DEFAULT_SHARDS
    failures: list[tuple[str, Exception]] = []

    for i, shard in enumerate(shards_to_train):
        print("=" * 80)
        print(f"Training shard: {shard}")
        print("=" * 80)

        try:
            run_train_model(
                shard=shard,
                input_dir=input_dir,
                random_dir=random_dir,
                log_dir=log_dir,
                max_neighbours=max_neighbours,
                num_rotations=num_rotations,
                batch_size=batch_size,
                learning_rate=learning_rate,
                epochs=epochs,
                patience=patience,
                latent_dim=latent_dim,
                normalizer_samples=normalizer_samples,
                subset_size=subset_size,
                seed=seed + i,
                split_seed=split_seed,
            )
        except Exception as exc:
            failures.append((shard, exc))
            print(f"[ERROR] Training failed for shard {shard}: {exc}")

            if not continue_on_error:
                raise

    print("\nTraining run finished.")
    print(f"Requested shards: {len(shards_to_train)}")
    print(f"Failed shards: {len(failures)}")

    if failures:
        print("\nFailures:")
        for shard, exc in failures:
            print(f"  - {shard}: {exc}")
