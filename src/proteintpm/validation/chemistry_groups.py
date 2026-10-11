from proteintpm.validation.motif_filters import MotifFilterConfig

# ---------------------------------------------------------------------
# Atom-class chemistry groups derived from map_atom(...)
# ---------------------------------------------------------------------

ALIPHATIC_C_IDS = frozenset({1})
AROMATIC_RING_IDS = frozenset({2, 13, 17})
BACKBONE_DONOR_IDS = frozenset({3})
CARBONYL_C_IDS = frozenset({4, 11, 12, 16})
HYDROXYL_O_IDS = frozenset({5, 7})
ANIONIC_O_IDS = frozenset({6})
AMIDE_O_IDS = frozenset({8})
AMIDE_N_IDS = frozenset({9})
THIOETHER_S_IDS = frozenset({14})
THIOL_S_IDS = frozenset({15})
CATIONIC_N_IDS = frozenset({18, 19})
BACKBONE_O_IDS = frozenset({20, 21})

# broad interaction groups
AROMATIC_IDS = frozenset({2, 13, 17})
NEGATIVE_IDS = frozenset({6})
POSITIVE_IDS = frozenset({18, 19})

HBOND_ACCEPTOR_IDS = frozenset(
    {
        5,  # Ser/Thr O
        6,  # Asp/Glu O
        7,  # Tyr OH
        8,  # Asn/Gln O
        17,  # His ring N (context-dependent)
        20,  # backbone O
        21,  # special O / unknown O class
    }
)

HBOND_DONOR_IDS = frozenset(
    {
        3,  # backbone N
        5,  # Ser/Thr OH
        7,  # Tyr OH
        9,  # Asn/Gln amide N
        13,  # Trp NE1
        17,  # His ring N (context-dependent)
        18,  # Arg NE
        19,  # Arg NH / Lys NZ
    }
)

HYDROPHOBIC_IDS = frozenset(
    {
        1,  # aliphatic carbon
        2,  # aromatic carbon
        14,  # Met sulfur
        15,  # Cys sulfur
    }
)

MOTIF_CFG = MotifFilterConfig(
    aromatic_ids=AROMATIC_IDS,
    negative_ids=NEGATIVE_IDS,
    positive_ids=POSITIVE_IDS,
    hbond_acceptor_ids=HBOND_ACCEPTOR_IDS,
    hbond_donor_ids=HBOND_DONOR_IDS,
    hydrophobic_ids=HYDROPHOBIC_IDS,
    close_cutoff=4.0,
    medium_cutoff=6.0,
    aromatic_patch_cutoff=5.0,
    planar_ratio_threshold=0.15,
    min_aromatic_patch_atoms=3,
    min_negative_patch_atoms=2,
    min_positive_patch_atoms=2,
)
