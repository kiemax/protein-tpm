import csv
import logging
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

from proteintpm.data.validate import PDBValidationError
from proteintpm.pipeline.extract_features import ExtractionParams, extract_features_for_pdb

logger = logging.getLogger(__name__)


def _init_worker(params_dict: dict, out_dir_str: str) -> None:
    """Runs once per worker process."""
    global _PARAMS, _OUT_DIR
    _PARAMS = ExtractionParams(**params_dict)
    _OUT_DIR = Path(out_dir_str)


def _run_one(pdb_path_str: str) -> dict[str, str]:
    """Worker: process one PDB; return a report row."""
    pdb_path = Path(pdb_path_str)
    pdb_id = pdb_path.stem

    try:
        extract_features_for_pdb(pdb_path, _OUT_DIR, _PARAMS)
        return {
            "pdb_id": pdb_id,
            "path": str(pdb_path),
            "status": "ok",
            "error_type": "",
            "reason": "",
        }
    except PDBValidationError as e:
        return {
            "pdb_id": pdb_id,
            "path": str(pdb_path),
            "status": "skipped",
            "error_type": type(e).__name__,
            "reason": str(e),
        }
    except Exception as e:
        # Don't logger.exception here: this happens in worker processes; you'll get noisy interleaving.
        return {
            "pdb_id": pdb_id,
            "path": str(pdb_path),
            "status": "failed",
            "error_type": type(e).__name__,
            "reason": str(e),
        }


def _write_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pdb_id", "path", "status", "error_type", "reason"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def run_extract_features_batch(
    *,
    input_dir: Path,
    out_dir: Path,
    params: ExtractionParams,
    max_workers: int = 4,
    maxtasksperchild: int = 25,
    report_path: Path | None = None,
) -> int:
    """
    Run feature extraction over all *.pdb files in input_dir.

    Uses multiprocessing with worker recycling (maxtasksperchild) to avoid
    long-run memory growth / fragmentation.

    Returns:
        exit code (0 if no unexpected failures, else 1)
    """
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if report_path is None:
        report_path = out_dir / "pdb_processing_report.csv"

    pdb_files = sorted(input_dir.glob("*.pdb"))
    if not pdb_files:
        logger.error("No PDBs found in %s", input_dir)
        return 1

    logger.info(
        "Starting batch: n_pdb=%d workers=%d maxtasksperchild=%d out=%s",
        len(pdb_files),
        max_workers,
        maxtasksperchild,
        out_dir,
    )

    # Spawn is more predictable for long runs than fork (esp. numpy/pandas memory behavior)
    ctx = get_context("spawn")

    rows: list[dict[str, str]] = []
    params_dict = asdict(params)

    with ctx.Pool(
        processes=max_workers,
        maxtasksperchild=maxtasksperchild,
        initializer=_init_worker,
        initargs=(params_dict, str(out_dir)),
    ) as pool:
        for row in pool.imap_unordered(_run_one, (str(p) for p in pdb_files), chunksize=1):
            rows.append(row)

            status = row["status"]
            pdb_id = row["pdb_id"]
            if status == "ok":
                logger.info("OK      %s", pdb_id)
            elif status == "skipped":
                logger.warning("SKIP    %s (%s)", pdb_id, row["reason"])
            else:
                logger.error("FAILED  %s (%s: %s)", pdb_id, row["error_type"], row["reason"])

    _write_report(report_path, rows)

    n_ok = sum(r["status"] == "ok" for r in rows)
    n_skipped = sum(r["status"] == "skipped" for r in rows)
    n_failed = sum(r["status"] == "failed" for r in rows)

    logger.info(
        "Done: ok=%d skipped=%d failed=%d report=%s", n_ok, n_skipped, n_failed, report_path
    )

    return 1 if n_failed else 0
