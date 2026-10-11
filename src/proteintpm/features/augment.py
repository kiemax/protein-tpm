import numpy as np
import pandas as pd


def _unit(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    # avoid divide-by-zero
    n[n == 0] = 1.0
    return v / n


def add_backbone_amide_h(df, nh_length=1.00):
    """
    Add backbone amide H for each residue i using N_i, CA_i, C_{i-1}.
    Assumes columns: record_type, atom_number, atom_name, res_name, chain_id, res_num, x,y,z, occupancy, temp_factor, element
    Returns a new dataframe with H rows appended.
    """
    atoms = df[df["record_type"].eq("ATOM")].copy()

    # N and CA of residue i
    need_cols = ["chain_id", "res_num", "res_name", "x", "y", "z"]
    N = atoms[atoms["atom_name"].eq("N")][["atom_number", "atom_name"] + need_cols].rename(
        columns={"x": "Nx", "y": "Ny", "z": "Nz"}
    )
    CA = atoms[atoms["atom_name"].eq("CA")][["chain_id", "res_num", "x", "y", "z"]].rename(
        columns={"x": "CAx", "y": "CAy", "z": "CAz"}
    )

    # C of residue i-1 (shift res_num forward so it lines up with residue i)
    Cprev = atoms[atoms["atom_name"].eq("C")][["chain_id", "res_num", "x", "y", "z"]].rename(
        columns={"x": "Cpx", "y": "Cpy", "z": "Cpz"}
    )
    Cprev["res_num"] = Cprev["res_num"].astype(int) + 1
    Cprev["res_num"] = Cprev["res_num"].astype(str)

    # C of residue i-1 (shift res_num forward so it lines up with residue i)
    Oprev = atoms[atoms["atom_name"].eq("O")][["chain_id", "res_num", "x", "y", "z"]].rename(
        columns={"x": "Opx", "y": "Opy", "z": "Opz"}
    )
    Oprev["res_num"] = Oprev["res_num"].astype(int) + 1
    Oprev["res_num"] = Oprev["res_num"].astype(str)

    for df_part in (N, CA, Cprev, Oprev):
        df_part["res_num"] = df_part["res_num"].astype(str)

    # Merge to get triplets for residue i
    trip = (
        N.merge(CA, on=["chain_id", "res_num"], how="inner")
        .merge(Cprev, on=["chain_id", "res_num"], how="inner")
        .merge(Oprev, on=["chain_id", "res_num"], how="inner")
    )

    # Skip proline (no backbone amide H)
    trip = trip[~trip["res_name"].eq("PRO")].copy()

    # Build vectors at N_i
    rN = trip[["Nx", "Ny", "Nz"]].to_numpy()
    rCA = trip[["CAx", "CAy", "CAz"]].to_numpy()
    rCp = trip[["Cpx", "Cpy", "Cpz"]].to_numpy()
    rOp = trip[["Opx", "Opy", "Opz"]].to_numpy()

    u_CA = _unit(rCA - rN)
    u_Cp = _unit(rCp - rN)
    u_Op = _unit(rOp - rCp)

    # Negative bisector direction (in the peptide plane)
    # dirH = _unit(-(u_CA + u_Cp))



    # H placed anti-parallel to the preceding C=O (planar trans-peptide geometry).
    # NOTE: an angle-bisector construction was tested and rejected — it placed H
    # incorrectly under this pipeline's residue alignment (validated in ChimeraX).
    # Do not "restore" the bisector without re-checking BILD output.

    
    dirH = _unit(-u_Op) 

    # Fallback if bisector degenerates (rare): point opposite to CA
    bad = np.isclose(np.linalg.norm(u_CA + u_Cp, axis=1), 0.0)
    if bad.any():
        dirH[bad] = _unit(-(u_CA[bad]))

    rH = rN + nh_length * dirH

    # Prepare new H rows
    base_cols = [
        "record_type",
        "atom_number",
        "atom_name",
        "res_name",
        "chain_id",
        "res_num",
        "x",
        "y",
        "z",
        "occupancy",
        "temp_factor",
        "element",
    ]
    new = pd.DataFrame(
        {
            "record_type": "ATOM",
            # atom numbers: continue after current max
            "atom_number": np.arange(
                int(atoms["atom_number"].max()) + 1, int(atoms["atom_number"].max()) + 1 + len(trip)
            ),
            "atom_name": "H",
            "res_name": trip["res_name"].values,
            "chain_id": trip["chain_id"].values,
            "res_num": trip["res_num"].values,
            "x": rH[:, 0],
            "y": rH[:, 1],
            "z": rH[:, 2],
            "occupancy": 1.00,
            "temp_factor": "Nan",
            "element": "H",
        }
    )[base_cols]

    # Don’t add H where one already exists (if your file already had hydrogens)
    has_H = atoms[atoms["atom_name"].isin(["H", "H1", "H2", "H3"])][
        ["chain_id", "res_num"]
    ].drop_duplicates()
    new = new.merge(has_H.assign(_hasH=True), on=["chain_id", "res_num"], how="left")
    new = new[new["_hasH"].isna()].drop(columns="_hasH")

    return pd.concat([df, new], ignore_index=True)
