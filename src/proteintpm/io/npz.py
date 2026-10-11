import numpy as np
import logging

logger = logging.getLogger(__name__)

def save_triplets_to_npz(triplet_list, filename):

    if not triplet_list:
        logger.warning("Skipping empty triplet list: %s", filename)
        return

    processed = []

    for triplet in triplet_list:
        shard_position = triplet[1][["x", "y", "z"]]
        target_atoms = triplet[2][["target_atom_id", "dx", "dy", "dz", "dr_squared"]]

        # Convert to numpy arrays (with float32 precision)
        shard_position = shard_position.to_numpy(dtype=np.float32)
        target_atoms = target_atoms.to_numpy(dtype=np.float32)

        processed.append(
            {
                "shard_position": shard_position,
                "target_atoms": target_atoms,
            }
        )

    # Convert list of dicts to np.array with dtype=object to preserve structure
    np.savez_compressed(filename, triplets=np.array(processed, dtype=object))
    logger.info("Saved %d triplets to %s", len(processed), filename)
