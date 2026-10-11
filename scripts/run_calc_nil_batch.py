import argparse
from pathlib import Path

from proteintpm.logging_config import setup_logging
from proteintpm.negatives.nil.batch import run_nil_batch
from proteintpm.negatives.nil.constants import NILParams


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True, help="Directory containing .pdb files")
    p.add_argument("--output", type=Path, required=True, help="Output base directory")
    p.add_argument(
        "--report",
        type=Path,
        default=None,
        help="CSV report path (default: <output>/nil_processing_report.csv)",
    )

    # parallelism
    p.add_argument("--max-workers", type=int, default=4)

    # Open3D execution
    p.add_argument("--conda-env", type=str, default="open3d_env")
    p.add_argument(
        "--open3d-script",
        type=Path,
        required=True,
        help="Path to the script executed inside the Open3D conda env (e.g. scripts/open3d_processing.py)",
    )

    # grid params (keep your validated defaults)
    p.add_argument("--voxel-size", type=float, default=0.5)
    p.add_argument("--padding", type=float, default=4.0)

    # sampling params (keep your validated defaults)
    p.add_argument("--A0", type=float, default=10000.0)
    p.add_argument("--n0", type=float, default=1000.0)
    p.add_argument("--b", type=float, default=1.0)

    # visualization
    p.add_argument("--bild-radius", type=float, default=0.3)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging(log_file=args.output / "nil_batch.log")

    params = NILParams(
        voxel_size=args.voxel_size,
        padding=args.padding,
        A0=args.A0,
        n0=args.n0,
        b=args.b,
        bild_radius=args.bild_radius,
        conda_env=args.conda_env,
        open3d_script=args.open3d_script,
    )

    raise SystemExit(
        run_nil_batch(
            input_dir=args.input,
            out_dir=args.output,
            params=params,
            max_workers=args.max_workers,
            report_path=args.report,
        )
    )
