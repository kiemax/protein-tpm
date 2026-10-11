import csv
from pathlib import Path


def load_positive_ok_ids(report_csv: Path) -> set[str]:
    rows = csv.DictReader(report_csv.open())
    return {
        row["pdb_id"].strip().lower()
        for row in rows
        if row.get("status", "").strip().lower() == "ok"
    }


def load_cdr_ids(cdr_csv: Path) -> set[str]:
    rows = csv.DictReader(cdr_csv.open())
    return {row["pdb_id"].strip().lower() for row in rows}


def write_overlap_csv(
    positive_report_csv: Path,
    cdr_csv: Path,
    out_csv: Path,
) -> set[str]:
    ok_ids = load_positive_ok_ids(positive_report_csv)
    cdr_ids = load_cdr_ids(cdr_csv)
    overlap = ok_ids & cdr_ids

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb_id"])
        w.writeheader()
        for pdb_id in sorted(overlap):
            w.writerow({"pdb_id": pdb_id})

    return overlap


def filter_fasta_by_ids(input_fasta: Path, keep_ids: set[str], output_fasta: Path) -> None:
    output_fasta.parent.mkdir(parents=True, exist_ok=True)

    out_lines: list[str] = []
    with input_fasta.open() as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0
    while i < len(lines):
        if not lines[i].startswith(">"):
            i += 1
            continue

        header = lines[i]
        seq = lines[i + 1] if i + 1 < len(lines) else ""
        pdb_id = header[1:].strip().lower()

        if pdb_id in keep_ids:
            out_lines.append(header)
            out_lines.append(seq)

        i += 2

    output_fasta.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
