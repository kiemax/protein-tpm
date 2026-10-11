import argparse
from pathlib import Path

from proteintpm.config.extraction import interaction_threshold, ligand_depth, receptor_depth
from proteintpm.pipeline.extract_features import ExtractionParams
from proteintpm.pipeline.extract_features_from_manifest import run_extract_features_from_manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract shard-wise features for complexes listed in a manifest CSV."
    )
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    params = ExtractionParams(
        interaction_threshold=interaction_threshold,
        ligand_depth=ligand_depth,
        receptor_depth=receptor_depth,
    )

    run_extract_features_from_manifest(
        manifest_csv=args.manifest_csv,
        out_dir=args.out_dir,
        params=params,
    )


if __name__ == "__main__":
    main()
