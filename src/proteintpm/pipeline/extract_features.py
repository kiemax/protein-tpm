import logging
from dataclasses import dataclass
from pathlib import Path

from proteintpm.data.chains import get_chains_positive
from proteintpm.data.interface import build_interface_pair
from proteintpm.data.normalization import renumber_residues
from proteintpm.data.pdb_io import create_dataframe, strip_pdb
from proteintpm.data.shards import SHARDS, get_shard_atoms_for_ligand
from proteintpm.data.validate import validate_pdb_for_extraction
from proteintpm.features.atom_types import map_atom
from proteintpm.features.augment import add_backbone_amide_h
from proteintpm.features.shardwise import calc_interface_shardwise
from proteintpm.io.npz import save_triplets_to_npz

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractionParams:
    ligand_depth: float
    receptor_depth: float
    interaction_threshold: float


def extract_features_for_pdb(
    pdb_path: Path,
    out_dir: Path,
    params: ExtractionParams,
    *,
    ligand_chains: list[str] | None = None,
    receptor_chains: list[str] | None = None,
    interface_mode: str = "bidirectional",
    shards=SHARDS,
) -> None:
    """Extract shard-wise interaction features for a single PDB and save per-shard NPZs."""
    pdb_path = Path(pdb_path)
    out_dir = Path(out_dir)
    pdb_id = pdb_path.stem

    # chains
    if ligand_chains is None or receptor_chains is None:
        ligand_chains, receptor_chains = get_chains_positive(pdb_path)
    logger.info(
        "Selected chains for pdb=%s ligand=%s receptor=%s",
        pdb_path.stem,
        ligand_chains,
        receptor_chains,
    )

    # parse PDB -> dataframe
    complex_data = create_dataframe(strip_pdb(str(pdb_path)))

    validate_pdb_for_extraction(complex_data, ligand_chains, receptor_chains, pdb_path)

    # normalize residue numbering globally
    complex_data = renumber_residues(complex_data)

    if interface_mode not in {"bidirectional", "ligand_to_receptor", "receptor_to_ligand"}:
        raise ValueError(
            f"Unknown interface_mode={interface_mode!r}. "
            "Expected one of {'bidirectional', 'ligand_to_receptor', 'receptor_to_ligand'}."
        )

    interface_triplet_sets = []

    if interface_mode in {"bidirectional", "ligand_to_receptor"}:
        lig2rec_ligand_data, lig2rec_receptor_data = build_interface_pair(
            complex_data=complex_data,
            ligand_chains=ligand_chains,
            receptor_chains=receptor_chains,
            ligand_depth=params.ligand_depth,
            receptor_depth=params.receptor_depth,
        )

        lig2rec_ligand_data = add_backbone_amide_h(lig2rec_ligand_data)
        lig2rec_receptor_data["target_atom_id"] = lig2rec_receptor_data.apply(
            lambda r: map_atom(r["atom_name"], r["res_name"]), axis=1
        )

        lig2rec_receptor_data = lig2rec_receptor_data[
            lig2rec_receptor_data["target_atom_id"].notna()
        ].copy()

        interface_triplet_sets.append((lig2rec_ligand_data, lig2rec_receptor_data))

    if interface_mode in {"bidirectional", "receptor_to_ligand"}:
        rec2lig_ligand_data, rec2lig_receptor_data = build_interface_pair(
            complex_data=complex_data,
            ligand_chains=receptor_chains,
            receptor_chains=ligand_chains,
            ligand_depth=params.ligand_depth,
            receptor_depth=params.receptor_depth,
        )

        rec2lig_ligand_data = add_backbone_amide_h(rec2lig_ligand_data)
        rec2lig_receptor_data["target_atom_id"] = rec2lig_receptor_data.apply(
            lambda r: map_atom(r["atom_name"], r["res_name"]), axis=1
        )

        rec2lig_receptor_data = rec2lig_receptor_data[
            rec2lig_receptor_data["target_atom_id"].notna()
        ].copy()

        interface_triplet_sets.append((rec2lig_ligand_data, rec2lig_receptor_data))

    for shard_name in shards:
        all_triplets: list = []

        for ligand_data_i, receptor_data_i in interface_triplet_sets:
            lig_shard = get_shard_atoms_for_ligand(
                ligand_data=ligand_data_i,
                shard_name=shard_name,
                shard_definitions=shards,
            )
            all_triplets.extend(
                calc_interface_shardwise(
                    shard_name,
                    lig_shard,
                    receptor_data_i,
                    params.interaction_threshold,
                    pdb_id=pdb_id,
                )
            )

        save_dir = out_dir / shard_name / "data_by_pdb" / pdb_id
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        save_triplets_to_npz(all_triplets, str(save_dir))

        logger.info("Saved shard=%s pdb=%s n_triplets=%d", shard_name, pdb_id, len(all_triplets))
