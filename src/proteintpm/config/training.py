from pathlib import Path

DEFAULT_INPUT_DIR = Path(
    "/home/kiemax/Helmholtz/projects/proteinTPM/noripTPM/data/sabdab/interfaces"
)
DEFAULT_RANDOM_DIR = Path(
    "/home/kiemax/Helmholtz/projects/proteinTPM/noripTPM/data/sabdab/negativeSamples/randomSampling"
)
# DEFAULT_INPUT_DIR = Path("/home/kienlein/Projects/noripTPM/data/sabdab/interfaces")
# DEFAULT_RANDOM_DIR = Path("/home/kienlein/Projects/noripTPM/data/sabdab/negativeSamples/randomSampling")
DEFAULT_LOG_DIR = Path("logs")

DEFAULT_SHARDS = [
    # Neutral polar
    "Hydroxyl_1",
    "Carboxamid_2",
    "Carboxamid_3",
    "Tyr_1",
    "Trp_2",
    "His_2",
    "His_3",
    "Thiol_1",
    "Backbone_2",  # polar backbone
    # Aliphatic / shape probes
    "Gly_1",
    "Ala_1",
    "Leu_1",
    "Val_1",
    "Ile_1",
    "Pro_1",
    "Pro_2",
    "Pro_3",
    # Polar anchors / hydrophobe-ish anchors
    "Hydroxyl_2",
    "Carboxamid_1",
    "Thiol_2",
    # Thioether
    "Thioether_1",
    "Thioether_2",
    # Charged functional atoms
    "Terminal_amino_1",  # Lys NZ
    "Guanidino_2",  # Arg NH1/NH2
    "Carboxyl_2",  # Asp/Glu OD/OE
    # Charged anchors -> treat as hydrophobic/shape-ish
    "Terminal_amino_2",
    "Guanidino_1",
    "Carboxyl_1",
    # Aromatic ring carbons
    "Phe_1",
    "Phe_2",
    "Tyr_2",
    "Trp_1",
    "Trp_3",
    "His_1",
    # Backbone
    "Backbone_1",
    "Backbone_3",
]

DEFAULT_MAX_NEIGHBOURS = 25
DEFAULT_NUM_ROTATIONS = 12
DEFAULT_BATCH_SIZE = 24
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 15
DEFAULT_PATIENCE = 5
DEFAULT_LATENT_DIM = 32
DEFAULT_NORMALIZER_SAMPLES = 20000

DEFAULT_SEED = 123
DEFAULT_SPLIT_SEED = 42


def build_training_params(
    input_dir: Path = DEFAULT_INPUT_DIR,
    random_dir: Path = DEFAULT_RANDOM_DIR,
    max_neighbours: int = DEFAULT_MAX_NEIGHBOURS,
    num_rotations: int = DEFAULT_NUM_ROTATIONS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    epochs: int = DEFAULT_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    latent_dim: int = DEFAULT_LATENT_DIM,
    normalizer_samples: int = DEFAULT_NORMALIZER_SAMPLES,
    subset_size: int | None = None,
) -> dict:
    return {
        "INPUT_DIR": Path(input_dir),
        "RANDOM_DIR": Path(random_dir),
        "MAX_NEIGHBOURS": max_neighbours,
        "NUM_ROTATIONS": num_rotations,
        "BATCH_SIZE": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "patience": patience,
        "latent_dim": latent_dim,
        "normalizer_samples": normalizer_samples,
        "subset_size": subset_size,
    }
