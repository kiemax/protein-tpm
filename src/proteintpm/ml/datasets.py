import numpy as np
import tensorflow as tf

from proteintpm.ml.augment import duplicate_and_augment
from proteintpm.ml.data_loading import load_shards
from proteintpm.ml.negative_sampling import random_neg_brooms


def sample_subset(triple, num_samples, axis=0, replace=False, rng=None):
    if num_samples is None or num_samples <= 0:
        return triple
    if rng is None:
        rng = np.random.default_rng()
    N = triple[0].shape[axis]
    idx = rng.choice(N, size=min(num_samples, N), replace=replace)
    return tuple(np.take(item, idx, axis=axis) for item in triple)


def no_nans_filter(inputs, label):
    return not tf.reduce_any([tf.reduce_any(tf.math.is_nan(t)) for t in list(inputs) + [label]])


def make_dataset_tensors(pos_pos, pos_cls, neg_pos, neg_cls, max_neigh):
    # channel 1: positions
    ch1 = np.concatenate((pos_pos, neg_pos))
    ch1 = tf.convert_to_tensor(ch1, tf.float32)

    # channel 2: classes
    ch2 = np.concatenate((pos_cls, neg_cls))
    ch2 = tf.convert_to_tensor(ch2, tf.float32)
    ch2 = tf.reshape(ch2, (ch2.shape[0], max_neigh))

    # labels
    labels = np.concatenate([np.ones(pos_pos.shape[0]), np.zeros(neg_pos.shape[0])])[:, None]
    labels = tf.convert_to_tensor(labels, tf.float32)

    return ch1, ch2, labels


# def build_fixed_val_set(
#    all_shards, shard_pos, params, val_idx_map, random_dir, allowed_random_val, V_pos, rng
# ):
#    # positives (fixed)
#    pos_path = f"{params['INPUT_DIR']}/{shard_pos}"
#    pos_pos_val, pos_cls_val = load_shards([pos_path], params["MAX_NEIGHBOURS"], "all", row_idx_map=val_idx_map)
#    pos_pos_val, pos_cls_val = sample_subset((pos_pos_val, pos_cls_val), num_samples=V_pos, rng=rng, replace=False)
#
#    # negatives (fixed): V_pos/2 shard + V_pos/2 brooms
#    V_neg = V_pos
#    n_shard = V_neg // 2
#    n_broom = V_neg - n_shard
#
#    neg_paths = [f"{params['INPUT_DIR']}/{s}" for s in all_shards if s != shard_pos]
#    neg_pos_pool, neg_cls_pool = load_shards(neg_paths, params["MAX_NEIGHBOURS"], "all", row_idx_map=val_idx_map)
#    neg_pos_val, neg_cls_val = sample_subset((neg_pos_pool, neg_cls_pool), num_samples=n_shard, rng=rng, replace=False)
#
#    rnd_pos_val, rnd_cls_val = random_neg_brooms(
#        data_path=random_dir,
#        n_samples=n_broom,
#        max_neighbours=params["MAX_NEIGHBOURS"],
#        allowed_idx_map=allowed_random_val,
#        rng=rng
#    )
#
#    all_neg_pos = np.concatenate([neg_pos_val, rnd_pos_val], axis=0) if n_broom else neg_pos_val
#    all_neg_cls = np.concatenate([neg_cls_val, rnd_cls_val], axis=0) if n_broom else neg_cls_val
#
#    ch1_val, ch2_val, lbl_val = make_dataset_tensors(pos_pos_val, pos_cls_val, all_neg_pos, all_neg_cls, params["MAX_NEIGHBOURS"])
#    return ch1_val, ch2_val, lbl_val


def build_fixed_val_set(
    all_shards,
    shard_pos,
    params,
    random_dir,
    allowed_random_val,
    rng,
):
    pos_path = f"{params['INPUT_DIR']}/{shard_pos}"

    # Positive validation samples from precomputed val split
    pos_pos_val, pos_cls_val = load_shards(
        [pos_path],
        params["MAX_NEIGHBOURS"],
        suffix="val",
    )

    V_pos = pos_pos_val.shape[0]
    V_neg = V_pos
    n_shard = V_neg // 2
    n_broom = V_neg - n_shard

    # Shard negatives from other shards' val splits
    neg_paths = [f"{params['INPUT_DIR']}/{s}" for s in all_shards if s != shard_pos]
    neg_pos_pool, neg_cls_pool = load_shards(
        neg_paths,
        params["MAX_NEIGHBOURS"],
        suffix="val",
    )

    neg_pos_val, neg_cls_val = sample_subset(
        (neg_pos_pool, neg_cls_pool),
        num_samples=n_shard,
        rng=rng,
        replace=False,
    )

    # Random broom negatives from validation partition
    rnd_pos_val, rnd_cls_val = random_neg_brooms(
        data_path=random_dir,
        n_samples=n_broom,
        max_neighbours=params["MAX_NEIGHBOURS"],
        allowed_idx_map=allowed_random_val,
        rng=rng,
    )

    all_neg_pos = np.concatenate([neg_pos_val, rnd_pos_val], axis=0) if n_broom else neg_pos_val
    all_neg_cls = np.concatenate([neg_cls_val, rnd_cls_val], axis=0) if n_broom else neg_cls_val

    return make_dataset_tensors(
        pos_pos_val,
        pos_cls_val,
        all_neg_pos,
        all_neg_cls,
        params["MAX_NEIGHBOURS"],
    )


def build_pipeline(
    ch1,
    ch2,
    labels,
    num_rotations: int,
    batch_size: int,
    augment_fn=None,
    filter_fn=None,
    shuffle: bool = True,
    cache: bool = False,
    repeat: bool = False,
):
    ds = tf.data.Dataset.from_tensor_slices((ch1, ch2, labels))

    if augment_fn:
        ds = ds.flat_map(lambda a, b, c: augment_fn(a, b, c))
    if filter_fn:
        ds = ds.filter(filter_fn)
    if shuffle:
        ds = ds.shuffle(buffer_size=ch1.shape[0] * num_rotations)
    if repeat:
        ds = ds.repeat()
    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# def build_train_ds_for_epoch(shard_pos, all_shards, params,
#                             train_idx_map, random_dir, allowed_random_train, rng):
#    pos_path = f"{params['INPUT_DIR']}/{shard_pos}"
#    neg_paths = [f"{params['INPUT_DIR']}/{s}" for s in all_shards if s != shard_pos]
#
#    # Pools (train only)
#    pos_pos_all, pos_cls_all = load_shards([pos_path], params["MAX_NEIGHBOURS"], "all", row_idx_map=train_idx_map)
#    neg_pos_all, neg_cls_all = load_shards(neg_paths,  params["MAX_NEIGHBOURS"], "all", row_idx_map=train_idx_map)
#
#    # Sample Np positives (subset_size or all available)
#    Np = min(params.get("subset_size", pos_pos_all.shape[0]), pos_pos_all.shape[0])
#    pp, pc = sample_subset((pos_pos_all, pos_cls_all), num_samples=Np, rng=rng, replace=False)
#
#    # Negatives: Np/2 from shards + Np/2 from brooms (TRAIN)
#    n_shard = Np // 2
#    n_broom = Np - n_shard
#
#    np_shard, nc_shard = sample_subset((neg_pos_all, neg_cls_all), num_samples=n_shard, rng=rng, replace=False)
#    np_broom, nc_broom = random_neg_brooms(
#        data_path=random_dir, n_samples=n_broom,
#        max_neighbours=params["MAX_NEIGHBOURS"],
#        allowed_idx_map=allowed_random_train, rng=rng
#    )
#
#    np_ = np.concatenate([np_shard, np_broom], axis=0) if n_broom else np_shard
#    nc  = np.concatenate([nc_shard, nc_broom], axis=0) if n_broom else nc_shard
#
#    # Convert to model I/O and make the dataset
#    ch1_e, ch2_e, lbl_e = make_dataset_tensors(pp, pc, np_, nc, params["MAX_NEIGHBOURS"])
#    train_ds = build_pipeline(
#        ch1_e, ch2_e, lbl_e,
#        num_rotations=params["NUM_ROTATIONS"],
#        batch_size=params["BATCH_SIZE"],
#        augment_fn=lambda x,y,z: duplicate_and_augment(x,y,z, params["NUM_ROTATIONS"]),
#        filter_fn=no_nans_filter,
#        shuffle=True,
#        repeat=False
#    )
#    steps_per_epoch = int(np.ceil(len(ch1_e) * params["NUM_ROTATIONS"] / params["BATCH_SIZE"]))
#    return train_ds, steps_per_epoch


def build_train_ds_for_epoch(
    shard_pos,
    all_shards,
    params,
    random_dir,
    allowed_random_train,
    rng,
):
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

    # Np = min(params.get("subset_size", pos_pos_all.shape[0]), pos_pos_all.shape[0])

    subset_size = params.get("subset_size")
    if subset_size is None:
        Np = pos_pos_all.shape[0]
    else:
        Np = min(subset_size, pos_pos_all.shape[0])

    pp, pc = sample_subset((pos_pos_all, pos_cls_all), num_samples=Np, rng=rng, replace=False)

    n_shard = Np // 2
    n_broom = Np - n_shard

    np_shard, nc_shard = sample_subset(
        (neg_pos_all, neg_cls_all),
        num_samples=n_shard,
        rng=rng,
        replace=False,
    )

    np_broom, nc_broom = random_neg_brooms(
        data_path=random_dir,
        n_samples=n_broom,
        max_neighbours=params["MAX_NEIGHBOURS"],
        allowed_idx_map=allowed_random_train,
        rng=rng,
    )

    np_ = np.concatenate([np_shard, np_broom], axis=0) if n_broom else np_shard
    nc_ = np.concatenate([nc_shard, nc_broom], axis=0) if n_broom else nc_shard

    ch1_e, ch2_e, lbl_e = make_dataset_tensors(pp, pc, np_, nc_, params["MAX_NEIGHBOURS"])

    train_ds = build_pipeline(
        ch1_e,
        ch2_e,
        lbl_e,
        num_rotations=params["NUM_ROTATIONS"],
        batch_size=params["BATCH_SIZE"],
        augment_fn=lambda x, y, z: duplicate_and_augment(x, y, z, params["NUM_ROTATIONS"]),
        filter_fn=no_nans_filter,
        shuffle=True,
        repeat=False,
    )

    steps_per_epoch = int(np.ceil(len(ch1_e) * params["NUM_ROTATIONS"] / params["BATCH_SIZE"]))
    return train_ds, steps_per_epoch
