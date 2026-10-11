import argparse
from pathlib import Path

from proteintpm.clustering.diagnostics import (
    assert_cluster_disjointness,
    plot_cluster_split_heatmap,
    plot_special_set_vs_train_overlap,
    summarize_splits,
)
from proteintpm.logging_config import setup_logging


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--special-ids", type=Path, default=None, help="Optional text file with one pdb_id per line"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=args.out_dir / "cluster_diagnostics.log")

    assert_cluster_disjointness(args.splits)

    summary = summarize_splits(args.splits)
    summary.to_csv(args.out_dir / "split_report.csv", index=False)

    plot_cluster_split_heatmap(
        args.splits,
        args.out_dir / "cluster_split_heatmap.png",
    )

    if args.special_ids is not None:
        special_ids = {
            line.strip().lower()
            for line in args.special_ids.read_text().splitlines()
            if line.strip()
        }
        plot_special_set_vs_train_overlap(
            args.splits,
            special_ids,
            args.out_dir / "special_vs_train_cluster_overlap.png",
        )
