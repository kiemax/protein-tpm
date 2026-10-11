import argparse

from proteintpm.config.training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_INPUT_DIR,
    DEFAULT_LATENT_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_DIR,
    DEFAULT_MAX_NEIGHBOURS,
    DEFAULT_NORMALIZER_SAMPLES,
    DEFAULT_NUM_ROTATIONS,
    DEFAULT_PATIENCE,
    DEFAULT_RANDOM_DIR,
    DEFAULT_SEED,
    DEFAULT_SHARDS,
    DEFAULT_SPLIT_SEED,
)
from proteintpm.pipeline.train_all_models import run_train_all_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train binary models for all configured target shards."
    )
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--random-dir", type=str, default=str(DEFAULT_RANDOM_DIR))
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--max-neighbours", type=int, default=DEFAULT_MAX_NEIGHBOURS)
    parser.add_argument("--num-rotations", type=int, default=DEFAULT_NUM_ROTATIONS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--normalizer-samples", type=int, default=DEFAULT_NORMALIZER_SAMPLES)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument(
        "--shards",
        nargs="*",
        default=None,
        help=f"Optional shard subset. Default is all configured shards: {', '.join(DEFAULT_SHARDS)}",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_train_all_models(
        input_dir=args.input_dir,
        random_dir=args.random_dir,
        log_dir=args.log_dir,
        max_neighbours=args.max_neighbours,
        num_rotations=args.num_rotations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        patience=args.patience,
        latent_dim=args.latent_dim,
        normalizer_samples=args.normalizer_samples,
        subset_size=args.subset_size,
        seed=args.seed,
        split_seed=args.split_seed,
        shards=args.shards,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    main()
