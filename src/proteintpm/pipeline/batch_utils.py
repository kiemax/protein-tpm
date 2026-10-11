import csv
import logging
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BatchItemResult:
    pdb_id: str
    path: str
    status: str  # "ok" | "skipped" | "failed"
    error_type: str
    reason: str


def write_csv_report(results: Sequence[BatchItemResult], report_path: Path) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["pdb_id", "path", "status", "error_type", "reason"]
    with report_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results:
            w.writerow(
                {
                    "pdb_id": r.pdb_id,
                    "path": r.path,
                    "status": r.status,
                    "error_type": r.error_type,
                    "reason": r.reason,
                }
            )


def run_batch(
    pdb_files: Sequence[Path],
    worker_fn: Callable[[Path], BatchItemResult],
    *,
    max_workers: int,
    report_path: Path | None = None,
    log_progress: bool = True,
) -> tuple[list[BatchItemResult], int]:
    """
    Generic batch runner for PDB pipelines.

    Returns
    -------
    (results, exit_code)
    exit_code policy:
      - 1 if any failed
      - 0 otherwise
    """
    if not pdb_files:
        logger.warning("No PDB files provided.")
        return [], 1

    logger.info("Found %d PDBs. Using %d workers.", len(pdb_files), max_workers)

    results: list[BatchItemResult] = []
    n_failed = 0
    n_skipped = 0

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker_fn, p) for p in pdb_files]

        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)

            if not log_progress:
                continue

            if r.status == "ok":
                logger.info("OK      %s", r.pdb_id)
            elif r.status == "skipped":
                n_skipped += 1
                logger.warning("SKIP    %s (%s)", r.pdb_id, r.reason)
            else:
                n_failed += 1
                logger.error("FAILED  %s (%s: %s)", r.pdb_id, r.error_type, r.reason)

    if report_path is not None:
        write_csv_report(results, report_path)
        logger.info("Wrote report: %s", report_path)

    logger.info(
        "Summary: ok=%d skipped=%d failed=%d",
        sum(r.status == "ok" for r in results),
        sum(r.status == "skipped" for r in results),
        sum(r.status == "failed" for r in results),
    )

    exit_code = 1 if any(r.status == "failed" for r in results) else 0
    return results, exit_code
