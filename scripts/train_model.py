import argparse

from proteintpm.config.training import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_INPUT_DIR,
    DEFAULT_LATENT_DIM,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_DIR,
    DEFAULT_MAX_NEIGHBOURS,
    DEFAULT_NORMALIZER_SAMPLES,
    DEFAULT_NUM_ROTATIONS,
    DEFAULT_PATIENCE,
    DEFAULT_RANDOM_DIR,
    DEFAULT_SEED,
    DEFAULT_SPLIT_SEED,
)
from proteintpm.pipeline.train_model import run_train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a binary model for a single target shard.")
    parser.add_argument("-s", "--shard", required=True, help="Target shard name, e.g. Carboxamid_1")
    parser.add_argument("--input-dir", type=str, default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--random-dir", type=str, default=str(DEFAULT_RANDOM_DIR))
    parser.add_argument("--log-dir", type=str, default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--max-neighbours", type=int, default=DEFAULT_MAX_NEIGHBOURS)
    parser.add_argument("--num-rotations", type=int, default=DEFAULT_NUM_ROTATIONS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    parser.add_argument("--normalizer-samples", type=int, default=DEFAULT_NORMALIZER_SAMPLES)
    parser.add_argument("--subset-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_train_model(
        shard=args.shard,
        input_dir=args.input_dir,
        random_dir=args.random_dir,
        log_dir=args.log_dir,
        max_neighbours=args.max_neighbours,
        num_rotations=args.num_rotations,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        patience=args.patience,
        latent_dim=args.latent_dim,
        normalizer_samples=args.normalizer_samples,
        subset_size=args.subset_size,
        seed=args.seed,
        split_seed=args.split_seed,
    )


if __name__ == "__main__":
    main()


# import numpy as np
# import sys
# from pathlib import Path
# import time
# import tensorflow as tf
# import math
# from tensorflow import keras
# from sklearn.model_selection import StratifiedKFold
# import argparse
# import json
# from collections import defaultdict
#
##SUPERGPU
# sys.path.append('/home/maximillian/Projects/noripTPM')
#
##HOME
##sys.path.append('/home/kienlein/Projects/noripTPM')
#
# from src.data_loader import (sample_subset,
#                             load_shards,
#                             build_shard_splits,
#                             build_normalizer,
#                             random_neg_brooms,
#                             build_random_broom_splits,
#                             build_pipeline,
#                             build_fixed_val_set,
#                             no_nans_filter,
#                             build_train_ds_for_epoch,
#                             build_normalizer_mix_dataset)
#
# from src.feature_engineering import (duplicate_and_augment)
# from src.model_builder import create_model
#
# root_logdir = Path("logs")
#
##shards_all = ["Carboxamid_start_atm",
##               "Guanidino_1",
##               "Hydroxyl",
##               "Terminal_amino",
##               "Carboxyl",
##               "Thiol",
##               "Thioether",
##               "His",
##               "Phe",
##               "Pro",
##               "Trp",
##               "Tyr"]
# shards_all = ["Carboxamid_1",
#               "Guanidino_1",
#               "Hydroxyl_1",
#               "Terminal_amino_1",
#               "Carboxyl_1",
#               "Thiol_1",
#               "Thioether_1",
#               "His_1",
#               "Phe_1",
#               "Pro_1",
#               "Trp_1",
#               "Tyr_1",
#               "Ala_1",
#               "Leu_1",
#               "Val_1",
#               "Ile_1",
#               "Gly_1",
#              "Backbone_1",
#              "Backbone_2",
#              "Backbone_3"
#               ]
#
# def get_run_logdir(shard, fold=-1):
#    run_id = time.strftime("run_%Y_%m_%d-%H_%S")
#    return root_logdir / shard / f"fold{fold+1}_{run_id}"
#
#
# if __name__ == "__main__":
#
#    # Command-line arguments parsing
#    parser = argparse.ArgumentParser(description="Train a model to classify this shard")
#    parser.add_argument("-s", "--shard",
#                        type=str,
#                        required=True,
#                        help="Train a model to classify this shard")
#
#    args = parser.parse_args()
#
#    shard_pos = args.shard
#    prefix = shard_pos.split("_")[0]
#    shards_all = [shard_pos if s.split("_")[0] == prefix else s for s in shards_all]
#
#
#    # Save hyperparameters
#    params = {
#    "INPUT_DIR" : Path("/home/kienlein/Projects/noripTPM/data/sabdab/interfaces"),
#    "RANDOM_DIR" : Path("/home/kienlein/Projects/noripTPM/data/sabdab/negativeSamples/randomSampling"),
#
#    "MAX_NEIGHBOURS" : 25,
#    "NUM_ROTATIONS" : 12,# 2 means add 1 random rotation
#    #"SUBSET_SIZE" : 50,
#    "BATCH_SIZE" : 24,
#    "learning_rate": 1e-4,
#    "optimizer": "Adam",
#    "loss": "binary_crossentropy",
#    "epochs": 15,
#    "patience": 5,
#    "latent_dim" : 32
#    }
#
#    # Pools for train, val in calculated sabdab brooms
#    train_idx_map, val_idx_map = build_shard_splits(
#        shards_all, params["INPUT_DIR"], suffix="all", val_frac=0.1, seed=42)
#
#    # Pools in the random generated negatives
#    allowed_random_train, allowed_random_val = build_random_broom_splits(
#        Path(params["RANDOM_DIR"]), seed=42)
#
#    rng = np.random.default_rng(123)
#
#    # Fixed validation set (choose V_pos once; e.g., min available, or a fixed number)
#    V_pos = len(val_idx_map[str(params["INPUT_DIR"]/shard_pos)]) # example; pick what fits your data
#    ch1_val, ch2_val, lbl_val = build_fixed_val_set(
#        shards_all, shard_pos, params, val_idx_map,
#        Path(params["RANDOM_DIR"]), allowed_random_val, V_pos, rng)
#
#
#    val_ds = build_pipeline(
#        ch1_val, ch2_val, lbl_val,
#        num_rotations=1, batch_size=params["BATCH_SIZE"],
#        augment_fn=lambda x,y,z: duplicate_and_augment(x,y,z, 0),
#        filter_fn=no_nans_filter, shuffle=False, cache=True)
#
#
#
#    norm_ds = build_normalizer_mix_dataset(
#        shard_pos, shards_all, params,
#        train_idx_map=train_idx_map,
#        random_dir=Path(params["RANDOM_DIR"]),
#        allowed_random_train=allowed_random_train,
#        target_samples=params.get("normalizer_samples", 20000),
#        rng=np.random.default_rng(1234))
#
#    norm_dist_layer = build_normalizer(norm_ds, sample_count=params.get("normalizer_samples", 20000))
#
#
#
#
#
#
#    #model = create_model(build_dummy_or_real_normalizer(), latent_dim=params["latent_dim"])
#    model = create_model(norm_dist_layer, latent_dim= params["latent_dim"])
#    model.compile(optimizer=keras.optimizers.Adam(params["learning_rate"]),
#                  loss='binary_crossentropy', metrics=['accuracy'])
#    model.summary()
#
#
#    #logging
#    run_logdir = get_run_logdir(args.shard)
#    run_logdir.mkdir(parents=True, exist_ok=True)
#    params_serializable = {k: str(v) if isinstance(v, Path) else v
#                           for k, v in params.items()}
#    with open(run_logdir / "hyperparameters.json", "w") as f:
#        json.dump(params_serializable, f, indent=4)
#
#
#    tensorboard_cb = keras.callbacks.TensorBoard(log_dir=run_logdir, histogram_freq=0, write_graph=True)
#    ckpt_cb = keras.callbacks.ModelCheckpoint(f"{run_logdir}/my_{args.shard}_model.keras", save_best_only=True)
#
#    # Manual early stopping across looped fits
#    best, wait, patience = float("inf"), 0, params["patience"]
#    for epoch in range(params["epochs"]):
#        train_ds, steps = build_train_ds_for_epoch(
#            shard_pos, shards_all, params, train_idx_map,
#            Path(params["RANDOM_DIR"]), allowed_random_train, rng
#        )
#
#        # (optional) rebuild or freeze your normalizer here
#
#        hist = model.fit(
#            train_ds,
#            validation_data=val_ds,
#            epochs=epoch+1,
#            initial_epoch=epoch,
#            steps_per_epoch=steps,
#            callbacks=[tensorboard_cb, ckpt_cb],
#            verbose=2,
#        )
#
#        val_loss = hist.history["val_loss"][-1]
#        if val_loss < best: best, wait = val_loss, 0
#        else: wait += 1
#        if wait >= patience:
#            print(f"Early stopping at epoch {epoch} with best val_loss={best:.6f}")
#            break
#
#    model.save(run_logdir / "final_model.keras")


# Model init/compile (normalizer: compute once on first epoch’s train_ds, or precompute)
# model = create_model(build_dummy_or_real_normalizer(), latent_dim=params["latent_dim"])
# model.compile(optimizer=keras.optimizers.Adam(params["learning_rate"]),
#              loss='binary_crossentropy', metrics=['accuracy'])

# tensorboard_cb = keras.callbacks.TensorBoard(log_dir=run_logdir, histogram_freq=0, write_graph=True)
# ckpt_cb = keras.callbacks.ModelCheckpoint(f"{run_logdir}/my_{args.shard}_model.keras", save_best_only=True)

## Manual early stopping across looped fits
# best, wait, patience = float("inf"), 0, params["patience"]
# for epoch in range(params["epochs"]):
#    train_ds, steps = build_train_ds_for_epoch(
#        shard_pos, shards_all, params, train_idx_map,
#        Path(params["RANDOM_DIR"]), allowed_random_train, rng
#    )

#    # (optional) rebuild or freeze your normalizer here

#    hist = model.fit(
#        train_ds,
#        validation_data=val_ds,
#        epochs=epoch+1,
#        initial_epoch=epoch,
#        steps_per_epoch=steps,
#        callbacks=[tensorboard_cb, ckpt_cb],
#        verbose=2,
#    )

#    val_loss = hist.history["val_loss"][-1]
#    if val_loss < best: best, wait = val_loss, 0
#    else: wait += 1
#    if wait >= patience:
#        print(f"Early stopping at epoch {epoch} with best val_loss={best:.6f}")
#        break

# model.save(run_logdir / "final_model.keras")


## 1. Build val set once
# pp, pc, np_, nc = load_and_sample(shard_pos,
#                                  shards_all,
#                                  suffix='all',
#                                  subset_size=None,
#                                  max_neighbours=params["MAX_NEIGHBOURS"],
#                                  input_dir=params["INPUT_DIR"])

####--- Add random negative surface brooms
# n_samples_neg = int(nc.shape[0]//2)
# rnd_np_, rnd_nc = random_neg_brooms(data_path = params["RANDOM_DIR"],
#                         n_samples = n_samples_neg,
#                         #n_samples = 1000,
#                         max_neighbours = params["MAX_NEIGHBOURS"])

# np_[:n_samples_neg,:,:] = rnd_np_
# nc[:n_samples_neg,:,:] = rnd_nc

###################----------------------------------------------Add a few negative brooms longer:


##n_elongate = 50 #number elongations
##idx = np.random.choice(np_.shape[0], size=n_elongate, replace=False)
##s = 5  # your elongation factor
##np_[idx, :, :] = stretch_bundles(np_[idx, :, :], s = s)
###################-----------------------------------------------------------------

# ch1_all, ch2_all, lbl_all = make_dataset_tensors(pp, pc, np_, nc, params["MAX_NEIGHBOURS"])

# N = len(lbl_all)  # master size
# rng = np.random.default_rng(seed=42)

## optional: stratify by label if imbalance matters
# val_size = int(0.1 * N)
# val_idx = rng.choice(np.arange(N), size=val_size, replace=False)
# train_pool_idx = np.setdiff1d(np.arange(N), val_idx, assume_unique=False)

## TF tensors if you like to use tf.gather
# val_idx_tf   = tf.convert_to_tensor(val_idx, dtype=tf.int64)
# train_pool_tf = tf.convert_to_tensor(train_pool_idx, dtype=tf.int64)

## Build the *fixed* validation tensors/ds once
# ch1_val = tf.gather(ch1_all, val_idx_tf)
# ch2_val = tf.gather(ch2_all, val_idx_tf)
# lbl_val = tf.gather(lbl_all, val_idx_tf)

# val_ds = build_pipeline(
#    ch1_val, ch2_val, lbl_val,
#    num_rotations=1,
#    batch_size=params["BATCH_SIZE"],
#    augment_fn=lambda x,y,z: duplicate_and_augment(x,y,z, num_augmentations=0),
#    filter_fn=no_nans_filter,
#    shuffle=False,
#    cache=True
# )


# Shuffle original data
# indices = tf.range(start=0, limit=tf.shape(ch1_all)[0], dtype=tf.int32)
# shuffled_indices = tf.random.shuffle(indices, seed=42)


# ch1_all = tf.gather(ch1_all, shuffled_indices)
# ch2_all = tf.gather(ch2_all, shuffled_indices)
# lbl_all = tf.gather(lbl_all, shuffled_indices)


## 3. Convert labels to numpy for stratification
# labels_np = lbl_all.numpy() if hasattr(lbl_all, 'numpy') else lbl_all


##skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

##for fold, (train_idx, val_idx) in enumerate(skf.split(ch1_all, labels_np)):
##    print(f"\n----- Fold {fold+1} -----")
##    print(f"train_idx : {train_idx}, len: {len(train_idx)}")
##    print(f"val_idx : {val_idx}, len: {len(val_idx)}")


##train_idx_tf = tf.constant(train_idx)
##val_idx_tf = tf.constant(val_idx)


#    # Gather tensors by indices
# ch1_train = tf.gather(ch1_all, train_idx_tf)
# ch2_train = tf.gather(ch2_all, train_idx_tf)
# lbl_train = tf.gather(lbl_all, train_idx_tf)

# ch1_val = tf.gather(ch1_all, val_idx_tf)
# ch2_val = tf.gather(ch2_all, val_idx_tf)
# lbl_val = tf.gather(lbl_all, val_idx_tf)

###Create fold-specific data
##ch1_train, ch2_train, lbl_train = ch1_all[train_idx], ch2_all[train_idx], labels_np[train_idx]
##ch1_val, ch2_val, lbl_val = ch1_all[val_idx], ch2_all[val_idx], labels_np[val_idx]

##Build train dataset pipeline
# train_ds = build_pipeline(
#    ch1_train, ch2_train, lbl_train,
#    num_rotations = params["NUM_ROTATIONS"],
#    batch_size=params["BATCH_SIZE"],
#    augment_fn=lambda x,y,z: duplicate_and_augment(x,y,z, params["NUM_ROTATIONS"]),
#    filter_fn=no_nans_filter,
#    shuffle=True,
#    repeat=False)

##Build val dataset pipeline
# val_ds = build_pipeline(
#    ch1_val, ch2_val, lbl_val,
#    num_rotations=1,
#    batch_size=params["BATCH_SIZE"],
#    augment_fn=lambda x,y,z: duplicate_and_augment(x,y,z, num_augmentations=0),
#    filter_fn=no_nans_filter,
#    shuffle=False,
#    cache=True)

##Log parameters

# params["steps_per_epoch"] = math.ceil(len(train_idx) * params["NUM_ROTATIONS"] / params["BATCH_SIZE"])

# norm_dist_layer = build_normalizer(train_ds, sample_count = len(train_idx))

# with open(run_logdir / "hyperparameters.json", "w") as f:
#    json.dump(params, f, indent=4)


###BIG BOY PERCEIVER
# model = create_model(norm_dist_layer, latent_dim= params["latent_dim"])


# optimizer = optimizers.Adam(learning_rate=params["learning_rate"])
# model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# print(f"steps_per_epoch: {params["steps_per_epoch"]}")

# tensorboard_cb = callbacks.TensorBoard(log_dir = run_logdir,
#                                       histogram_freq = 0,
#                                       write_graph = True)

# reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
#    monitor="val_loss",
#    factor=0.2,
#    patience=3,
#    verbose=1,  # logs when it reduces LR
#    min_lr=1e-6 # prevent LR becoming too small
# )


# early_stopping_cb = callbacks.EarlyStopping(patience= params["patience"],
#                                            restore_best_weights= True)
# checkpoint_cb = callbacks.ModelCheckpoint(f"{run_logdir}/my_{args.shard}_model.keras",
#                                          save_best_only= True)
# model.summary()
# with open(run_logdir / "model_summary.txt", "w") as f:
#    model.summary(print_fn=lambda x: f.write(x + "\n"))

# history = model.fit(train_ds,
#                    validation_data=val_ds,
#                    epochs = params["epochs"],
#                    callbacks= [tensorboard_cb, early_stopping_cb, checkpoint_cb, reduce_lr_cb],
#                    verbose=2,
#                    steps_per_epoch= params["steps_per_epoch"])

# model.save(run_logdir / "final_model.keras")
