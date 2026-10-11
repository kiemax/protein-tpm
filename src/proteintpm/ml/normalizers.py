import numpy as np
from keras import layers

from proteintpm.ml.augment import duplicate_and_augment
from proteintpm.ml.data_loading import load_shards
from proteintpm.ml.datasets import (
    build_pipeline,
    make_dataset_tensors,
    no_nans_filter,
    sample_subset,
)
from proteintpm.ml.negative_sampling import random_neg_brooms


def build_normalizer(dataset, sample_count):
    # unbatch to get raw values
    raw = dataset.map(lambda x, y: x[0]).unbatch()
    norm = layers.Normalization()
    norm.adapt(raw.take(sample_count))
    print("mean:", norm.mean.numpy(), "var:", norm.variance.numpy())
    return norm


# def build_normalizer_mix_dataset(
#    shard_pos, all_shards, params,
#    train_idx_map, random_dir, allowed_random_train,
#    target_samples=20000, rng=None
# ):
#    """
#    Returns a tf.data.Dataset (no aug, no shuffle) to fit your normalizer.
#    Mix: 50% pos (pos shard), 25% shard-neg (other shards), 25% broom-neg (RANDOM_DIR).
#    """
#    if rng is None:
#        rng = np.random.default_rng(2025)
#
#    pos_path = f"{params['INPUT_DIR']}/{shard_pos}"
#    neg_paths = [f"{params['INPUT_DIR']}/{s}" for s in all_shards if s != shard_pos]
#
#    # 1) Load train-only pools (already filtered by train_idx_map)
#    pos_pos_all, pos_cls_all = load_shards([pos_path], params["MAX_NEIGHBOURS"], "all", row_idx_map=train_idx_map)
#    neg_pos_all, neg_cls_all = load_shards(neg_paths,  params["MAX_NEIGHBOURS"], "all", row_idx_map=train_idx_map)
#
#    # 2) Decide counts for the mix
#    n_total = target_samples
#    n_pos   = max(1, n_total // 2)
#    n_neg_s = max(1, (n_total - n_pos) // 2)
#    n_neg_b = max(1, n_total - n_pos - n_neg_s)
#
#    # 3) Sample from each pool (no replacement, clipped by availability)
#    Pp, Pc = sample_subset((pos_pos_all, pos_cls_all), num_samples=min(n_pos, pos_pos_all.shape[0]), rng=rng, replace=False)
#    Np_s, Nc_s = sample_subset((neg_pos_all, neg_cls_all), num_samples=min(n_neg_s, neg_pos_all.shape[0]), rng=rng, replace=False)
#
#    # 4) Sample broom-negatives from TRAIN partition of RANDOM_DIR
#    Np_b, Nc_b = random_neg_brooms(
#        data_path=random_dir,
#        n_samples=n_neg_b,
#        max_neighbours=params["MAX_NEIGHBOURS"],
#        allowed_idx_map=allowed_random_train,  # <-- disjoint from validation
#        rng=rng
#    )
#
#    # 5) Merge negatives and convert to your model inputs/labels
#    Np_all = np.concatenate([Np_s, Np_b], axis=0) if n_neg_b else Np_s
#    Nc_all = np.concatenate([Nc_s, Nc_b], axis=0) if n_neg_b else Nc_s
#
#    # IMPORTANT: use the same converter you use during training
#    ch1_norm, ch2_norm, lbl_norm = make_dataset_tensors(Pp, Pc, Np_all, Nc_all, params["MAX_NEIGHBOURS"])
#
#    # Build a no-aug, no-shuffle dataset for fitting the normalizer
#    norm_ds = build_pipeline(
#        ch1_norm, ch2_norm, lbl_norm,
#        num_rotations=1,
#        batch_size=params["BATCH_SIZE"],
#        augment_fn=lambda x,y,z: duplicate_and_augment(x,y,z, 0),  # no augmentation for stats
#        filter_fn=no_nans_filter,
#        shuffle=False
#    )
#    return norm_ds


def build_normalizer_mix_dataset(
    shard_pos,
    all_shards,
    params,
    random_dir,
    allowed_random_train,
    target_samples=20000,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng(2025)

    pos_path = f"{params['INPUT_DIR']}/{shard_pos}"
    neg_paths = [f"{params['INPUT_DIR']}/{s}" for s in all_shards if s != shard_pos]

    pos_pos_all, pos_cls_all = load_shards(
        [pos_path],
        params["MAX_NEIGHBOURS"],
        suffix="train",
    )
    neg_pos_all, neg_cls_all = load_shards(
        neg_paths,
        params["MAX_NEIGHBOURS"],
        suffix="train",
    )

    n_total = target_samples
    n_pos = max(1, n_total // 2)
    n_neg_s = max(1, (n_total - n_pos) // 2)
    n_neg_b = max(1, n_total - n_pos - n_neg_s)

    Pp, Pc = sample_subset(
        (pos_pos_all, pos_cls_all),
        num_samples=min(n_pos, pos_pos_all.shape[0]),
        rng=rng,
        replace=False,
    )
    Np_s, Nc_s = sample_subset(
        (neg_pos_all, neg_cls_all),
        num_samples=min(n_neg_s, neg_pos_all.shape[0]),
        rng=rng,
        replace=False,
    )

    Np_b, Nc_b = random_neg_brooms(
        data_path=random_dir,
        n_samples=n_neg_b,
        max_neighbours=params["MAX_NEIGHBOURS"],
        allowed_idx_map=allowed_random_train,
        rng=rng,
    )

    Np_all = np.concatenate([Np_s, Np_b], axis=0)
    Nc_all = np.concatenate([Nc_s, Nc_b], axis=0)

    ch1_norm, ch2_norm, lbl_norm = make_dataset_tensors(
        Pp, Pc, Np_all, Nc_all, params["MAX_NEIGHBOURS"]
    )

    return build_pipeline(
        ch1_norm,
        ch2_norm,
        lbl_norm,
        num_rotations=1,
        batch_size=params["BATCH_SIZE"],
        augment_fn=lambda x, y, z: duplicate_and_augment(x, y, z, 0),
        filter_fn=no_nans_filter,
        shuffle=False,
    )
