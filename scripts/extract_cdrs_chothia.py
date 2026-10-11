import argparse
import csv
import logging
from pathlib import Path

from proteintpm.data.cdrs import build_positive_policy_cdr_signature
from proteintpm.data.pdb_io import create_dataframe, strip_pdb
from proteintpm.logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract Chothia CDR sequences for clustering, mirroring positive-pipeline policy."
    )
    p.add_argument(
        "--pdb-dir", type=Path, required=True, help="Directory containing PDB files (*.pdb)"
    )
    p.add_argument("--out-csv", type=Path, required=True, help="Output CSV path")
    p.add_argument("--out-fasta", type=Path, required=True, help="Output FASTA path")
    p.add_argument(
        "--limit", type=int, default=0, help="Optional limit on number of PDBs processed"
    )
    p.add_argument("--log-file", type=Path, default=None, help="Optional log file")
    return p.parse_args()


def main(pdb_dir: Path, out_csv: Path, out_fasta: Path, limit: int) -> int:
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    if limit and limit > 0:
        pdb_files = pdb_files[:limit]

    if not pdb_files:
        logger.error("No PDB files found in %s", pdb_dir)
        return 1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_fasta.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    fasta_lines: list[str] = []
    n_ok = 0
    n_fail = 0

    for pdb_path in pdb_files:
        pdb_id = pdb_path.stem.lower()
        try:
            df = create_dataframe(strip_pdb(str(pdb_path)))
            row = build_positive_policy_cdr_signature(df, pdb_path)

            rows.append(row)

            # FASTA should not contain separators like "|"
            fasta_seq = row["cdrHL_concat"].replace("|", "")
            if not fasta_seq:
                raise ValueError(f"{pdb_id}: empty FASTA sequence after CDR extraction")

            fasta_lines.append(f">{pdb_id}")
            fasta_lines.append(fasta_seq)

            n_ok += 1

        except Exception as e:
            n_fail += 1
            logger.warning("Failed to extract CDRs for %s: %s", pdb_id, e)

    # union of keys across rows
    fieldnames: list[str] = sorted({k for row in rows for k in row.keys()})

    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    out_fasta.write_text("\n".join(fasta_lines) + ("\n" if fasta_lines else ""))

    logger.info("Done. ok=%d failed=%d wrote=%s and %s", n_ok, n_fail, out_csv, out_fasta)
    return 0 if n_ok else 1


if __name__ == "__main__":
    args = parse_args()
    log_file = (
        args.log_file if args.log_file is not None else (args.out_csv.parent / "extract_cdrs.log")
    )
    setup_logging(log_file=log_file)
    raise SystemExit(main(args.pdb_dir, args.out_csv, args.out_fasta, args.limit))
