from pathlib import Path

import pandas as pd


def pdb_line(row):
    return (
        f"{row['record_type']:<6}"
        f"{int(row['atom_number']):>5} "
        f"{row['atom_name']:>4} "
        f"{row['res_name']:>3} "
        f"{row['chain_id']:>1}"
        f"{(row['res_num']):>4}    "
        f"{float(row['x']):>8.3f}"
        f"{float(row['y']):>8.3f}"
        f"{float(row['z']):>8.3f}"
        f"{float(row['occupancy']):>6.2f}"
        f"{float(row['temp_factor']):>6.2f}          "
        f"{row['element']:>2}"
    )


def save_pdb(data_stripped: pd.DataFrame, output_path: str | Path):

    pdb_lines = data_stripped.apply(pdb_line, axis=1)

    with open(output_path, "w") as f:
        for line in pdb_lines:
            f.write(line + "\n")
