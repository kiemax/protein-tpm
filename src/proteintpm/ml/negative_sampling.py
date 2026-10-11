import pickle
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from proteintpm.negatives.brooms.build import get_broom


def extract_brooms(protein, start_brooms, max_neighbours):

    tree = cKDTree(np.asarray(protein[["x", "y", "z"]]))
    _distances, neighbors = tree.query(start_brooms, k=max_neighbours)
    pos_list = []
    cls_list = []

    for c, nbr in zip(start_brooms, neighbors):
        p, cl = get_broom(c, protein, nbr)
        # print(p.shape)
        pos_list.append(p)
        cls_list.append(cl)
    try:
        return (np.concatenate(pos_list, axis=0), np.concatenate(cls_list, axis=0))
    except:
        return None


def build_random_broom_splits(random_dir: Path, seed: int = 42, k_val=20):
    """
    Partition RANDOM_DIR into disjoint train/val index maps per (protein_path, type).
    A protein is ELIGIBLE only if BOTH AB/random_pts.npy and AG/random_pts.npy exist.
    split_ratio controls the *partition* within each file (not the counts you later draw).
    """
    rng = np.random.default_rng(seed)
    allowed_train, allowed_val = {}, {}

    random_dir = Path(random_dir)
    for prot_path in sorted(random_dir.glob("*")):
        ab_pts = prot_path / "AB" / "random_pts.npy"
        ag_pts = prot_path / "AG" / "random_pts.npy"

        # exclude if either file is missing
        if not (ab_pts.is_file() and ag_pts.is_file()):
            # optional: print(f"[skip] {prot_path}: missing AB/AG random_pts.npy")
            continue

        for type_str, pts_path in (("AB", ab_pts), ("AG", ag_pts)):
            try:
                arr = np.load(pts_path)
            except Exception:
                # corrupted/unloadable -> skip this protein entirely
                # (skip both AB/AG by continuing outer loop next iteration)
                arr = None
            if arr is None or arr.size == 0:
                # empty -> skip this type
                # optional: print(f"[skip] {pts_path}: empty")
                continue

            n = arr.shape[0]
            all_idx = np.arange(n)
            # k_val = int(round(split_ratio * n))  # just a partition; your draw sizes come later
            # k_val = min(max(k_val, 0), n)

            val_idx = (
                rng.choice(all_idx, size=k_val, replace=False)
                if k_val > 0
                else np.array([], dtype=int)
            )
            train_idx = np.setdiff1d(all_idx, val_idx, assume_unique=False)

            key = (str(prot_path), type_str)
            allowed_train[key] = train_idx
            allowed_val[key] = val_idx

    return allowed_train, allowed_val


def random_neg_brooms(
    data_path: Path,
    n_samples: int,
    max_neighbours: int,
    allowed_idx_map: dict | None = None,
    rng=None,
    max_tries: int = 3,
):
    if rng is None:
        rng = np.random.default_rng()
    data_path = Path(data_path)

    broom_pos, broom_cls = [], []
    buckets = [(p, t) for p in sorted(data_path.glob("*")) for t in ("AB", "AG")]
    if not buckets or n_samples <= 0:
        raise RuntimeError("random_neg_brooms: no buckets or n_samples <= 0")

    q, r = divmod(n_samples, len(buckets))
    quotas = [q + (1 if i < r else 0) for i in range(len(buckets))]

    for (prot_path, type_str), count in zip(buckets, quotas, strict=True):
        if count == 0:
            continue

        pts_path = prot_path / type_str / "random_pts.npy"
        tgt_path = prot_path / type_str / "data_target.pkl"
        if not (pts_path.is_file() and tgt_path.is_file()):
            continue

        try:
            random_pts = np.load(pts_path)
            with open(tgt_path, "rb") as f:
                protein_target = pickle.load(f)
        except Exception:
            continue
        if random_pts.size == 0:
            continue

        key = (str(prot_path), type_str)
        pool_idx = (
            np.arange(random_pts.shape[0])
            if allowed_idx_map is None
            else allowed_idx_map.get(key, np.array([], int))
        )
        if pool_idx.size == 0:
            continue

        tries, got = 0, 0
        local_pool = pool_idx.copy()
        while got < count and tries < max_tries and local_pool.size:
            need = min(count - got, local_pool.size)
            sel = rng.choice(local_pool, size=need, replace=False)
            local_pool = np.setdiff1d(local_pool, sel, assume_unique=False)

            result = extract_brooms(
                protein=protein_target, start_brooms=random_pts[sel], max_neighbours=max_neighbours
            )
            tries += 1
            if not result:
                continue
            pos, cls = result
            take = min(pos.shape[0], count - got)
            if take > 0:
                broom_pos.append(pos[:take])
                broom_cls.append(cls[:take])
                got += take

    if not broom_pos:
        raise RuntimeError("random_neg_brooms: no samples produced")

    broom_pos = np.concatenate(broom_pos, axis=0)
    broom_cls = np.concatenate(broom_cls, axis=0)
    if broom_pos.shape[0] > n_samples:
        broom_pos = broom_pos[:n_samples]
        broom_cls = broom_cls[:n_samples]
    return broom_pos, broom_cls
