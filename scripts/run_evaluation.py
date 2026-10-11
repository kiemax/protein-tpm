import argparse
from pathlib import Path

from proteintpm.validation.evaluation import run_evaluation_from_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible evaluation over a set of models."
    )
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--max-neighbours", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")

    stage = parser.add_argument_group("stages")
    stage.add_argument("--all", action="store_true", help="Run all stages (default).")
    stage.add_argument("--predict", action="store_true", help="Run model prediction stage.")
    stage.add_argument(
        "--descriptors", action="store_true", help="Compute shared set descriptors once."
    )
    stage.add_argument(
        "--chemical-motifs",
        action="store_true",
        help="Compute shared chemistry/motif features once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation_from_args(args)


if __name__ == "__main__":
    main()
