import csv
import re
import subprocess
from pathlib import Path


def run_cdhit(
    input_fasta: Path,
    output_prefix: Path,
    identity: float,
    threads: int = 8,
    memory_mb: int = 16000,
    conda_env: str = "fasta-env",
) -> None:
    """
    Run CD-HIT clustering using a dedicated conda environment.
    """

    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    if identity >= 0.7:
        word_size = 5
    elif identity >= 0.6:
        word_size = 4
    elif identity >= 0.5:
        word_size = 3
    else:
        raise ValueError("Identity <0.5 requires manual CD-HIT tuning")

    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "cd-hit",
        "-i",
        str(input_fasta),
        "-o",
        str(output_prefix),
        "-c",
        str(identity),
        "-n",
        str(word_size),
        "-d",
        "0",
        "-M",
        str(memory_mb),
        "-T",
        str(threads),
    ]

    subprocess.run(cmd, check=True)


def parse_cdhit_clstr(clstr_path: Path, out_csv: Path) -> None:
    """
    Parse CD-HIT .clstr file into CSV with columns:
    pdb_id, cluster_id
    """
    cluster_id = None
    rows = []
    pat = re.compile(r">([^\.]+)\.\.\.")

    for line in clstr_path.read_text().splitlines():
        line = line.strip()
        if line.startswith(">Cluster "):
            cluster_id = int(line.split()[-1])
            continue

        if not line or cluster_id is None:
            continue

        m = pat.search(line)
        if not m:
            continue

        pdb_id = m.group(1).strip().lower()
        rows.append({"pdb_id": pdb_id, "cluster_id": cluster_id})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb_id", "cluster_id"])
        w.writeheader()
        w.writerows(rows)
