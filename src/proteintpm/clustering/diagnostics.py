from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def summarize_splits(splits_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(splits_csv)
    summary = (
        df.groupby("split")
        .agg(
            n_pdbs=("pdb_id", "count"),
            n_clusters=("cluster_id", "nunique"),
        )
        .reset_index()
    )
    return summary


def plot_cluster_split_heatmap(splits_csv: Path, out_png: Path) -> None:
    """
    Plot a cluster x split occupancy matrix.
    A clean split should show each cluster occupied in exactly one split.
    """
    df = pd.read_csv(splits_csv)
    pivot = (
        df.assign(value=1)
        .pivot_table(
            index="cluster_id", columns="split", values="value", aggfunc="sum", fill_value=0
        )
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks([])
    ax.set_title("Cluster membership by split")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("PDB count in cluster/split")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def assert_cluster_disjointness(splits_csv: Path) -> None:
    """
    Raises if any cluster appears in more than one split.
    """
    df = pd.read_csv(splits_csv)
    per_cluster = df.groupby("cluster_id")["split"].nunique()
    bad = per_cluster[per_cluster > 1]
    if not bad.empty:
        raise ValueError(
            f"Found {len(bad)} clusters appearing in multiple splits: {bad.index.tolist()[:10]}"
        )


def plot_special_set_vs_train_overlap(
    splits_csv: Path,
    special_ids: set[str],
    out_png: Path,
) -> None:
    """
    Show whether any special-set PDBs share clusters with train.
    Ideally, for a held-out test panel, train overlap should be zero.
    """
    df = pd.read_csv(splits_csv)
    df["pdb_id"] = df["pdb_id"].str.lower()

    special_ids = {x.lower() for x in special_ids}
    special_df = df[df["pdb_id"].isin(special_ids)]
    train_df = df[df["split"] == "train"]

    train_clusters = set(train_df["cluster_id"])
    special_df = special_df.assign(in_train_cluster=special_df["cluster_id"].isin(train_clusters))

    counts = special_df["in_train_cluster"].value_counts().reindex([False, True], fill_value=0)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["No train overlap", "Train overlap"], counts.values)
    ax.set_ylabel("Special-set PDB count")
    ax.set_title("Special-set cluster overlap with train")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
