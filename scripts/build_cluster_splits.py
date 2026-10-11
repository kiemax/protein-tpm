import argparse
from pathlib import Path

from proteintpm.clustering.cdhit import parse_cdhit_clstr, run_cdhit
from proteintpm.clustering.overlap import filter_fasta_by_ids, write_overlap_csv
from proteintpm.clustering.splits import make_cluster_splits
from proteintpm.logging_config import setup_logging


def parse_args():
    p = argparse.ArgumentParser(
        description="Build cluster-based train/val/test splits from CDR FASTA."
    )
    p.add_argument("--positive-report", type=Path, required=True)
    p.add_argument("--cdr-csv", type=Path, required=True)
    p.add_argument("--cdr-fasta", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--identity", type=float, default=0.7)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--memory-mb", type=int, default=16000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument(
        "--force-test-ids",
        type=Path,
        default=None,
        help="Optional text file with one pdb_id per line",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=args.out_dir / "cluster_splits.log")

    overlap_csv = args.out_dir / "cdr_overlap.csv"
    overlap_fasta = args.out_dir / "cdr_overlap.fasta"
    cluster_prefix = args.out_dir / f"cdhit_{int(args.identity * 100)}"
    clusters_csv = args.out_dir / "cdhit_clusters.csv"
    splits_csv = args.out_dir / "pdb_splits.csv"

    overlap_ids = write_overlap_csv(args.positive_report, args.cdr_csv, overlap_csv)
    filter_fasta_by_ids(args.cdr_fasta, overlap_ids, overlap_fasta)

    run_cdhit(
        input_fasta=overlap_fasta,
        output_prefix=cluster_prefix,
        identity=args.identity,
        threads=args.threads,
        memory_mb=args.memory_mb,
    )

    parse_cdhit_clstr(
        cluster_prefix.with_suffix(".fasta.clstr")
        if cluster_prefix.suffix
        else Path(str(cluster_prefix) + ".clstr"),
        clusters_csv,
    )

    force_test_ids = set()
    if args.force_test_ids is not None:
        force_test_ids = {
            line.strip().lower()
            for line in args.force_test_ids.read_text().splitlines()
            if line.strip()
        }

    make_cluster_splits(
        clusters_csv=clusters_csv,
        out_csv=splits_csv,
        seed=args.seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        force_test_ids=force_test_ids,
    )
