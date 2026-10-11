import argparse
from pathlib import Path

from proteintpm.negatives.nil.constants import NILParams
from proteintpm.negatives.nil.pipeline import generate_nil_for_pdb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate NIL negatives for a single PDB.")
    p.add_argument("pdb_path", type=Path, help="Path to input .pdb file")
    p.add_argument("--out-base", type=Path, required=True, help="Base output directory")
    p.add_argument("--open3d-script", type=Path, required=True, help="Path to open3d_processing.py")
    p.add_argument("--conda-env", type=str, default="open3d_env")

    p.add_argument("--voxel-size", type=float, default=0.5)
    p.add_argument("--padding", type=float, default=4.0)

    return p.parse_args()


def main() -> None:
    args = parse_args()

    params = NILParams(
        voxel_size=args.voxel_size,
        padding=args.padding,
        conda_env=args.conda_env,
        open3d_script=args.open3d_script,
    )

    generate_nil_for_pdb(args.pdb_path, args.out_base, params)


if __name__ == "__main__":
    main()
