import argparse
from pathlib import Path

from proteintpm.tp_mapping.batch import run_mapping_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tp_mapping setup + prediction from a jobs CSV."
    )

    parser.add_argument(
        "--jobs",
        type=str,
        required=True,
        help="Path to jobs CSV. A single-row CSV is fully supported.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Base output directory for tp_mapping results.",
    )

    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.5,
        help="Default voxel size in Å for jobs that do not override it in the CSV.",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=1.0,
        help="Default padding in Å for jobs that do not override it in the CSV.",
    )

    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional fixed run id for all jobs.",
    )

    parser.add_argument(
        "--model-link-mode",
        type=str,
        default="symlink",
        choices=["symlink", "copy", "path_only"],
        help="How to materialize model logs inside each job directory.",
    )

    parser.add_argument(
        "--setup-workers",
        type=int,
        default=4,
        help="Parallel workers for setup stage.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Prediction mode.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU indices, used only when --mode gpu.",
    )
    parser.add_argument(
        "--infer-workers",
        type=int,
        default=2,
        help="CPU prediction workers, used only when --mode cpu.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help=("Default model.predict batch_size. Per-row batch_size in the CSV overrides this."),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs if present.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_mapping_batch(
        jobs_csv=Path(args.jobs),
        outdir=Path(args.outdir),
        voxel_size=args.voxel_size,
        padding=args.padding,
        run_id=args.run_id,
        model_link_mode=args.model_link_mode,
        setup_workers=args.setup_workers,
        mode=args.mode,
        gpus=args.gpus,
        infer_workers=args.infer_workers,
        batch_size=args.batch_size,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
