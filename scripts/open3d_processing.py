import argparse
from pathlib import Path

from proteintpm.geometry.open3d_processing import compute_surface_area_and_inside_mask


def main() -> None:
    p = argparse.ArgumentParser(description="Open3D step (surface area + inside mask).")
    p.add_argument(
        "path", type=Path, help="Directory containing atom_cloud.npy and voxel_centers.npy"
    )
    p.add_argument("--alpha", type=float, default=3.0)
    p.add_argument("--plot", action="store_true")
    args = p.parse_args()

    compute_surface_area_and_inside_mask(
        args.path,
        alpha=args.alpha,
        save_plot=args.plot,
    )


if __name__ == "__main__":
    main()
