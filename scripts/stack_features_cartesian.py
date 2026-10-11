import argparse
import logging
import os
import sys

import numpy as np

sys.path.append("/home/kiemax/Helmholtz/projects/proteinTPM/noripTPM")
import glob

# Configure the root logger
logging.basicConfig(
    filename="./broom.log",
    encoding="utf-8",
    filemode="a",
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_NEIGHBOURS = 100
PAD_CLASS = 0
ZERO_ATOL = 1e-12  # numerical tolerance for "zero"


def _valid_mask(positions: np.ndarray, classes: np.ndarray) -> np.ndarray:
    """
    True for rows that are real atoms (not padding).
    Uses both class!=0 and 'row not all zeros' for safety.
    """
    cls = classes.reshape(-1)
    class_mask = cls != PAD_CLASS
    pos_mask = ~np.all(np.isclose(positions, 0.0, atol=ZERO_ATOL), axis=1)
    return class_mask & pos_mask


def pad_array(arr, target_rows):
    """Pad or trim the array to have exactly target_rows rows."""
    if arr.shape[0] >= target_rows:
        return arr[:target_rows]
    else:
        padding = np.zeros((target_rows - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
        return np.vstack([arr, padding])


def process_triplet(triplet):
    """
    Process a single triplet to extract:
      - target positions (all columns except the first and last)
      - target classes (first column, reshaped as a column vector)
      - chart orientation
    Each of the first two arrays is padded (or trimmed) to have MAX_NEIGHBOURS rows.
    """
    # Extract and pad target positions.
    positions = pad_array(triplet["target_atoms"][:, 1:-1], MAX_NEIGHBOURS)

    # Extract and pad target classes.
    target_atms = triplet["target_atoms"][
        :, 0
    ]  # +1 <--  from old version, problem with zero padding now handled in map_atoms()
    # print(target_atms)
    classes = pad_array(target_atms.reshape(-1, 1), MAX_NEIGHBOURS)

    # Extract chart orientation (assumed to be already in the desired shape).
    # chart_ori = triplet["chart_orientation"]

    return positions, classes  # , chart_ori


def process_files(file_paths):
    """
    Process all files and aggregate the processed triplet information.
    Returns two numpy arrays for positions and classes.
    """
    target_positions = []
    target_classes = []

    for path in file_paths:
        try:
            with np.load(path, allow_pickle=True) as data:
                triplets = data["triplets"]
                for i, triplet in enumerate(triplets):
                    positions, classes = process_triplet(triplet)

                    # Build mask to exclude zero-padded rows
                    mask = _valid_mask(positions, classes)
                    valid_positions = positions[mask]
                    valid_classes = classes[mask]

                    # If nothing valid remains, skip
                    if valid_positions.size == 0 or valid_classes.size == 0:
                        logger.info(f"Skipped triplet {i} in {path}: no valid (unpadded) rows")
                        continue

                    # 1) Require at least two unique non-padding classes
                    unique_classes = np.unique(valid_classes)
                    if unique_classes.size < 2:
                        logger.info(
                            f"Skipped triplet {i} in {path}: only one class present {unique_classes.ravel().tolist()}"
                        )
                        continue

                    # 2) Requirements for valid brooms
                    max_neighbours = 25

                    lower_threshold = 8  # distance cutoff
                    min_count_low = 5  # how many vectors must be below threshold

                    upper_threshold = 13
                    max_count_high = 3

                    distances = np.linalg.norm(valid_positions[:max_neighbours, :], axis=1)

                    count_low = np.sum(distances < lower_threshold)
                    count_high = np.sum(distances > upper_threshold)

                    if count_low < min_count_low or count_high > max_count_high:
                        logger.info(
                            f"Skipped triplet {i} in {path}: only {count_low} < {lower_threshold} "
                            f"and {count_high} > {upper_threshold} "
                            f"(required {min_count_low}, {max_count_high})"
                        )
                        continue

                    # Passed filters: keep original (padded) arrays to preserve shapes
                    target_positions.append(positions)
                    target_classes.append(classes)

        except Exception as e:
            logger.error(f"Error loading {path}: {e}")

    if not target_positions:
        logger.warning("No valid triplets collected from provided files.")
        # To avoid np.stack error, return empty arrays with expected shapes
        return (np.empty((0, MAX_NEIGHBOURS, positions.shape[1])), np.empty((0, MAX_NEIGHBOURS, 1)))

    return np.stack(target_positions), np.stack(target_classes)


if __name__ == "__main__":
    # Command-line arguments parsing
    parser = argparse.ArgumentParser(description="Process a PDB shard and save the results.")
    parser.add_argument(
        "-s", "--shard", type=str, required=True, help="Input PDB shard path to process"
    )
    parser.add_argument("--saveTo", type=str, required=True, help="Output path to save the results")
    args = parser.parse_args()

    # Get the list of files to process
    file_paths = glob.glob(f"{args.shard}/data_by_pdb/*.npz")

    # Helper function to extract PDB ID from filename
    def get_pdbid_from_path(path):
        return os.path.splitext(os.path.basename(path))[0]

    # Process files
    def process_split(file_paths, suffix):
        # target_positions_array, target_classes_array, chart_orientations_array = process_files(file_paths)
        target_positions_array, target_classes_array = process_files(file_paths)

        # Save each output separately
        np.savez_compressed(
            f"{args.saveTo}/target_positions_{suffix}.npz", arr=target_positions_array
        )
        np.savez_compressed(f"{args.saveTo}/target_classes_{suffix}.npz", arr=target_classes_array)
        # np.savez_compressed(f'{args.saveTo}/till_target_positions_{suffix}.npz', arr=target_positions_array)
        # np.savez_compressed(f'{args.saveTo}/till_target_classes_{suffix}.npz', arr=target_classes_array)
        # np.savez_compressed(f'{args.saveTo}/chart_orient_xyz_{suffix}.npz', arr=chart_orientations_array)

    # Run processing for both sets
    # process_split(train_files, 'train')
    # process_split(val_files, 'val')
    process_split(file_paths, "all")
