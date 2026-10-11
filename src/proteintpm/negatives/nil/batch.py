import logging
from functools import partial
from pathlib import Path

from proteintpm.data.validate import PDBValidationError  # reuse if you skip on validation
from proteintpm.negatives.nil.constants import NILParams
from proteintpm.negatives.nil.pipeline import generate_nil_for_pdb
from proteintpm.pipeline.batch_utils import BatchItemResult, run_batch

logger = logging.getLogger(__name__)


def run_one_nil(
    pdb_path: Path,
    out_dir: Path,
    params: NILParams,
) -> BatchItemResult:
    try:
        generate_nil_for_pdb(pdb_path=pdb_path, out_base=out_dir, params=params)
        return BatchItemResult(
            pdb_id=pdb_path.stem,
            path=str(pdb_path),
            status="ok",
            error_type="",
            reason="",
        )
    except PDBValidationError as e:
        # expected skip
        return BatchItemResult(
            pdb_id=pdb_path.stem,
            path=str(pdb_path),
            status="skipped",
            error_type=type(e).__name__,
            reason=str(e),
        )
    except Exception as e:
        # unexpected failure
        logger.exception("FAILED pdb=%s path=%s", pdb_path.stem, pdb_path)
        return BatchItemResult(
            pdb_id=pdb_path.stem,
            path=str(pdb_path),
            status="failed",
            error_type=type(e).__name__,
            reason=str(e),
        )


def run_nil_batch(
    *,
    input_dir: Path,
    out_dir: Path,
    params: NILParams,
    max_workers: int,
    report_path: Path | None = None,
) -> int:
    """
    Run NIL generation for all .pdb in input_dir.

    Writes report CSV with columns:
      pdb_id, path, status, error_type, reason

    Exit code policy:
      - 1 if any failed
      - 0 otherwise (skips don't fail the run)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_files = sorted(input_dir.glob("*.pdb"))

    if report_path is None:
        report_path = out_dir / "nil_processing_report.csv"

    worker_fn = partial(run_one_nil, out_dir=out_dir, params=params)

    _, exit_code = run_batch(
        pdb_files,
        worker_fn=worker_fn,
        max_workers=max_workers,
        report_path=report_path,
    )
    return exit_code
