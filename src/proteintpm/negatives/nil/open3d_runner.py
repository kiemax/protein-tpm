import subprocess
from pathlib import Path


def run_open3d_processing(
    output_dir: Path,
    *,
    conda_env: str,
    script_path: Path,
) -> None:
    """
    Run Open3D preprocessing that produces:
      - surface_area.npy
      - inside_mask.npy
    in output_dir.
    """
    output_dir = Path(output_dir)
    script_path = Path(script_path)

    subprocess.run(
        ["conda", "run", "-n", conda_env, "python", str(script_path), str(output_dir)],
        check=True,
    )
