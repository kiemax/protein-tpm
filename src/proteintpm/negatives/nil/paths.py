from pathlib import Path


def make_ab_ag_output_dirs(
    base: str | Path,
    pdb_id: str,
    *,
    exist_ok: bool = True,
    parents: bool = True,
) -> tuple[Path, Path]:
    """
    Create output directories following project convention:
      <base>/<pdb_id>/AG
      <base>/<pdb_id>/AB
    """
    base_path = Path(base)
    ag_path = base_path / pdb_id / "AG"
    ab_path = base_path / pdb_id / "AB"

    ag_path.mkdir(parents=parents, exist_ok=exist_ok)
    ab_path.mkdir(parents=parents, exist_ok=exist_ok)
    return ag_path, ab_path
