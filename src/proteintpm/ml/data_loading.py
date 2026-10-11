from pathlib import Path

import numpy as np


def load_shards(shard_paths, max_neighbours, suffix):
    target_positions = []
    target_classes = []

    for path in shard_paths:
        path = Path(path)
        pos = np.load(path / f"target_positions_{suffix}.npz", allow_pickle=True)["arr"]
        cls = np.load(path / f"target_classes_{suffix}.npz", allow_pickle=True)["arr"]

        pos = pos[:, :max_neighbours, :]
        cls = cls[:, :max_neighbours, :]

        target_positions.append(pos)
        target_classes.append(cls)

    return (
        np.concatenate(target_positions, axis=0),
        np.concatenate(target_classes, axis=0),
    )
