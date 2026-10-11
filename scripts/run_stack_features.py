from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from proteintpm.logging_config import setup_logging
from proteintpm.pipeline.stack_brooms import (
    BroomStackConfig,
    load_split_csv,
    stack_npz_files_to_brooms,
    write_shard_report_csv,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stack per-PDB triplet NPZs into broom arrays per split."
    )
    p.add_argument(
        "--input-root", type=Path, required=True, help="Root folder containing shard dirs."
    )
    p.add_argument(
        "--output-root", type=Path, required=True, help="Output folder for stacked brooms."
    )
    p.add_argument("--splits", type=Path, required=True, help="CSV with columns pdb_id,split.")
    p.add_argument("--max-neighbours", type=int, default=100)
    p.add_argument("--log-file", type=Path, default=None)
    return p.parse_args()


def main(input_root: Path, output_root: Path, splits_csv: Path, cfg: BroomStackConfig) -> int:
    split_map = load_split_csv(splits_csv)
    if not split_map:
        logger.error("Split map is empty: %s", splits_csv)
        return 1

    shards = sorted([p for p in input_root.iterdir() if p.is_dir()])
    if not shards:
        logger.error("No shard directories found under %s", input_root)
        return 1

    logger.info("Found %d shards under %s", len(shards), input_root)

    for shard_dir in shards:
        shard_name = shard_dir.name
        data_dir = shard_dir / "data_by_pdb"
        if not data_dir.exists():
            logger.warning("Shard %s has no data_by_pdb directory. Skipping.", shard_name)
            continue

        all_npz = sorted(data_dir.glob("*.npz"))
        if not all_npz:
            logger.warning("Shard %s has no NPZ files. Skipping.", shard_name)
            continue

        # Assign files to splits based on filename stem
        split_files: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
        not_in_map = 0
        for f in all_npz:
            pdb_id = f.stem.lower()
            split = split_map.get(pdb_id)
            if split in split_files:
                split_files[split].append(f)
            else:
                not_in_map += 1

        logger.info(
            "Shard %s: files total=%d train=%d val=%d test=%d not_in_split_map=%d",
            shard_name,
            len(all_npz),
            len(split_files["train"]),
            len(split_files["val"]),
            len(split_files["test"]),
            not_in_map,
        )

        shard_out = output_root / shard_name
        shard_out.mkdir(parents=True, exist_ok=True)

        report_rows: list[dict] = []

        for split, files in split_files.items():
            positions, classes, stats = stack_npz_files_to_brooms(files, cfg)

            np.savez_compressed(shard_out / f"target_positions_{split}.npz", arr=positions)
            np.savez_compressed(shard_out / f"target_classes_{split}.npz", arr=classes)

            # Report: one row per skip reason (or a single row if no skips)
            if stats.skip_reasons:
                for reason, count in stats.skip_reasons.most_common():
                    report_rows.append(
                        {
                            "shard": shard_name,
                            "split": split,
                            "files_seen": stats.files_seen,
                            "triplets_seen": stats.triplets_seen,
                            "triplets_kept": stats.triplets_kept,
                            "triplets_skipped": stats.triplets_skipped,
                            "skip_reason": reason,
                            "skip_reason_count": count,
                        }
                    )
            else:
                report_rows.append(
                    {
                        "shard": shard_name,
                        "split": split,
                        "files_seen": stats.files_seen,
                        "triplets_seen": stats.triplets_seen,
                        "triplets_kept": stats.triplets_kept,
                        "triplets_skipped": stats.triplets_skipped,
                        "skip_reason": "",
                        "skip_reason_count": 0,
                    }
                )

            logger.info(
                "Shard %s split %s: kept=%d skipped=%d (seen=%d)",
                shard_name,
                split,
                stats.triplets_kept,
                stats.triplets_skipped,
                stats.triplets_seen,
            )

        write_shard_report_csv(shard_out / "stacking_report.csv", report_rows)

    logger.info("Done. Output written to %s", output_root)
    return 0


if __name__ == "__main__":
    args = parse_args()
    log_file = (
        args.log_file if args.log_file is not None else (args.output_root / "stack_brooms.log")
    )
    setup_logging(log_file=log_file)

    cfg = BroomStackConfig(max_neighbours=args.max_neighbours)
    raise SystemExit(main(args.input_root, args.output_root, args.splits, cfg))
