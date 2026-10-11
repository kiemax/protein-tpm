import logging
from pathlib import Path

import pandas as pd

from proteintpm.pipeline.extract_features import ExtractionParams, extract_features_for_pdb

logger = logging.getLogger(__name__)


def run_extract_features_from_manifest(
    manifest_csv: Path,
    out_dir: Path,
    params: ExtractionParams,
) -> None:
    df = pd.read_csv(manifest_csv)

    required = {"pdb_id", "pdb_path", "ligand_chain", "receptor_chain"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    for _, row in df.iterrows():
        pdb_id = str(row["pdb_id"]).strip()
        pdb_path = Path(row["pdb_path"]) / f"{pdb_id}.pdb"

        ligand_chains = [str(row["ligand_chain"]).strip()]
        receptor_chains = [str(row["receptor_chain"]).strip()]

        logger.info(
            "Processing pdb=%s ligand=%s receptor=%s",
            pdb_id,
            ligand_chains,
            receptor_chains,
        )

        extract_features_for_pdb(
            pdb_path=pdb_path,
            out_dir=out_dir,
            params=params,
            ligand_chains=ligand_chains,
            receptor_chains=receptor_chains,
            interface_mode="ligand_to_receptor",
        )
