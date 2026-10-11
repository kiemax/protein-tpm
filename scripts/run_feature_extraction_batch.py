import argparse
from pathlib import Path

from proteintpm.config.extraction import interaction_threshold, ligand_depth, receptor_depth
from proteintpm.logging_config import setup_logging
from proteintpm.pipeline.batch_extract_features import run_extract_features_batch
from proteintpm.pipeline.extract_features import ExtractionParams

MAX_WORKERS = 4
MAX_TASKS_PER_CHILD = 25


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None)
    p.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    p.add_argument("--maxtasksperchild", type=int, default=MAX_TASKS_PER_CHILD)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(
            f"Output dir {args.output} is not empty. Refusing to write "
            f"(would merge with stale output). Use a fresh directory or delete it."
        )
    setup_logging(log_file=args.output / "batch.log")

    params = ExtractionParams(
        ligand_depth=ligand_depth,
        receptor_depth=receptor_depth,
        interaction_threshold=interaction_threshold,
    )

    raise SystemExit(
        run_extract_features_batch(
            input_dir=args.input,
            out_dir=args.output,
            params=params,
            max_workers=args.max_workers,
            maxtasksperchild=args.maxtasksperchild,
            report_path=args.report,
        )
    )
