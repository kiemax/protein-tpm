import csv
import random
from collections import defaultdict
from pathlib import Path


def make_cluster_splits(
    clusters_csv: Path,
    out_csv: Path,
    *,
    seed: int = 42,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    force_test_ids: set[str] | None = None,
) -> None:
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1.0")

    rows = list(csv.DictReader(clusters_csv.open()))
    cluster_to_ids: dict[int, list[str]] = defaultdict(list)

    for row in rows:
        pdb_id = row["pdb_id"].strip().lower()
        cluster_id = int(row["cluster_id"])
        cluster_to_ids[cluster_id].append(pdb_id)

    force_test_ids = {x.lower() for x in (force_test_ids or set())}

    # First mark clusters that contain forced test ids
    forced_test_clusters = {
        cid
        for cid, ids in cluster_to_ids.items()
        if any(pdb_id in force_test_ids for pdb_id in ids)
    }

    remaining_clusters = sorted(set(cluster_to_ids) - forced_test_clusters)
    rng = random.Random(seed)
    rng.shuffle(remaining_clusters)

    n_remaining = len(remaining_clusters)
    n_test = int(round(n_remaining * test_frac))
    n_val = int(round(n_remaining * val_frac))

    test_clusters = set(remaining_clusters[:n_test]) | forced_test_clusters
    val_clusters = set(remaining_clusters[n_test : n_test + n_val])
    train_clusters = set(remaining_clusters[n_test + n_val :])

    out_rows = []
    for cid, ids in cluster_to_ids.items():
        if cid in test_clusters:
            split = "test"
        elif cid in val_clusters:
            split = "val"
        else:
            split = "train"

        for pdb_id in ids:
            out_rows.append({"pdb_id": pdb_id, "cluster_id": cid, "split": split})

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdb_id", "cluster_id", "split"])
        w.writeheader()
        w.writerows(sorted(out_rows, key=lambda r: (r["split"], int(r["cluster_id"]), r["pdb_id"])))
