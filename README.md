# proteinTPM

**Target Preference Maps (TPMs)** — an interpretable, transferable representation of
protein-interface specificity learned from experimentally resolved complexes.

proteinTPM decomposes protein interfaces into local chemical motifs ("shards"),
trains a per-shard model to recognise the atomic environments in which each motif
is compatible with a protein surface, and evaluates those models over a 3D grid to
produce continuous preference fields.

---

## Contents

- [Glossary](#glossary)
- [Installation](#installation)
- [External dependencies](#external-dependencies)
- [Pipeline overview](#pipeline-overview)
- [Stage 0 — CDR extraction and clustering](#stage-0--cdr-extraction-and-clustering)
- [Stage 1 — Positive feature extraction](#stage-1--positive-feature-extraction)
- [Stage 2 — Negative sampling (nil)](#stage-2--negative-sampling-nil)
- [Stage 3 — Broom stacking](#stage-3--broom-stacking)
- [Stage 4 — Training](#stage-4--training)
- [Stage 5 — Evaluation](#stage-5--evaluation)
- [Stage 6 — TPM generation](#stage-6--tpm-generation)
- [Configuration reference](#configuration-reference)
- [Repository layout](#repository-layout)
- [Preparing your own structures](#preparing-your-own-structures)
- [Known limitations](#known-limitations)
- [Data availability](#data-availability)
- [Citing](#citing)

---

## Glossary

The codebase uses project-specific vocabulary:

| Term | Meaning |
|---|---|
| **Shard** | A local interaction motif, defined by one or more (residue, atom) specifications and centred on a single *anchor atom*. Chemical groups are represented by **several** shards (e.g. arginine's guanidinium by Cζ and by its terminal nitrogens), not one per residue. |
| **Anchor** | The atom at the centre of a shard — the origin of all relative position vectors for that environment. Multiple anchors per group replace an explicit orientation frame: the spatial relationship between a group's anchor maps encodes its orientation. |
| **Target environment** | The atoms of the opposing protein surrounding an anchor, encoded as (atom-class label, Δx, Δy, Δz). |
| **Broom** | A *stacked feature tensor*: many environments for one shard class, padded to a fixed neighbour count and concatenated for training. |
| **nil** | The negative class — environments sampled at random points in the solvent-accessible shell around a protein. |
| **TPM** | Target Preference Map: a trained shard model evaluated over a 3D voxel grid, giving a compatibility probability at every point in space. |

---

## Installation

```bash
git clone <repo-url>
cd proteinTPM
pip install -e .
```

Requires Python ≥ 3.10. Core dependencies: `numpy`, `pandas`, `scipy`,
`tensorflow`/`keras`, `scikit-learn`, `mrcfile`, `matplotlib`.

> **GPU.** Training and TPM inference use TensorFlow. A GPU is strongly
> recommended for both; CPU works but is slow for grid inference.

### External dependencies

Two stages shell out to tools that are kept in **separate environments** because
their dependencies conflict with the main environment:

| Tool | Used by | Environment |
|---|---|---|
| **CD-HIT** | Stage 0 (clustering) | conda env, name passed via `--conda-env` (default `fasta-env`) |
| **Open3D** | Stage 2 (negative sampling) | conda env, name passed via `NILParams.conda_env` (default `open3d_env`) |

Create them from the provided specs:

```bash
conda env create -f envs/environment_cdhit.yml     # provides cd-hit
conda env create -f envs/environment_open3d.yml    # provides open3d
```

<!-- TODO: ship these two env files with pinned versions -->


---

## Pipeline overview

```
                    ┌─────────────────────────────────────────┐
  SAbDab PDBs ─────►│ [0] CDR extraction → CD-HIT clustering   │
  (Chothia)         │     → train/val/test split assignment    │
        │           └──────────────────┬──────────────────────┘
        │                              │ split_map.csv
        │                              │
        ├──────────►[1] Feature extraction (positives)
        │              per-PDB shard environments
        │              → interfaces/<shard>/data_by_pdb/*.npz
        │                              │
        ├──────────►[2] Negative sampling (nil)
        │              random solvent-shell environments
        │              → negativeSamples/randomSampling/<pdb>/{AB,AG}/
        │                              │
        │                              ▼
        └──────────►[3] Broom stacking ◄── split_map.csv
                       → brooms/<shard>/target_{positions,classes}_{train,val,test}.npz
                                       │
                                       ▼
                       [4] Training — 36 per-shard Perceiver models
                       → logs/<shard>/run_*/best_<shard>_model.keras
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
              [5] Evaluation                [6] TPM generation
              per-shard AUC, pairwise       3D preference maps (.mrc)
              → evaluation/runs/<name>/     → tp_mapping/<job>/inference/
```

Splits are defined **once** (stage 0) and **applied** at stacking (stage 3).
Stages 1 and 2 are split-agnostic.

---

## Stage 0 — CDR extraction and clustering

Redundancy control operates on **CDR sequences only** — the framework is
deliberately excluded, since the conserved antibody scaffold would otherwise
inflate pairwise identity and merge functionally distinct antibodies.

```bash
# 1. Extract CDR signatures (H1-H3, L1-L3 concatenated per complex)
python scripts/extract_cdrs_chothia.py \
    --input  data/sabdab/data/ \
    --output data/sabdab/clustering/cdrs.fasta

# 2. Cluster + assign splits
python scripts/build_cluster_splits.py \
    --cdr-fasta       data/sabdab/clustering/cdrs.fasta \
    --positive-ok-ids data/sabdab/clustering/positive_ok_ids.txt \
    --out-dir         data/sabdab/clustering/cluster_splits_70 \
    --identity        0.7 \
    --val-frac        0.1 \
    --test-frac       0.1 \
    --seed            42 \
    --force-test-ids  data/splits/sars2_spike_force_test.txt
```

**Outputs:** `cdhit_70*` (CD-HIT artifacts), `cdhit_clusters.csv`,
`pdb_splits.csv` (`pdb_id,cluster_id,split`).

Whole **clusters** are assigned to a split, so no CDR-similar complexes span
train and test. `--force-test-ids` pins specific complexes (and their clusters)
into the test split — used to hold out the SARS-CoV-2 spike complexes underlying
the affinity analysis.

**Verify the split:**

```bash
python scripts/plot_cluster_split_diagnostics.py \
    --splits      data/sabdab/clustering/cluster_splits_70/pdb_splits.csv \
    --out-dir     data/sabdab/clustering/diagnostics \
    --special-ids data/splits/sars2_spike_force_test.txt
```

This asserts cluster disjointness and plots the special-set-vs-train overlap.

---

## Stage 1 — Positive feature extraction

Parses each complex, assigns chain roles from SAbDab `REMARK` annotations
(antigen = shard-bearing/ligand side; antibody = target/receptor side), and
extracts shard-centred environments in **both** interface directions.

```bash
python scripts/run_feature_extraction_batch.py \
    --input       data/sabdab/data/ \
    --output      data/sabdab/interfaces/ \
    --report      data/sabdab/rerun_report.csv \
    --max-workers 4
```

**Outputs:** `interfaces/<shard>/data_by_pdb/<pdb_id>.npz` (triplets), plus a
per-structure processing report (`ok` / `skipped` / `failed` with reasons).

> The output directory must be **empty** — the driver refuses to write into a
> non-empty directory to prevent silently merging with stale output.

### Peptide–protein (single direction)

Peptide chains cannot be resolved from SAbDab annotations, and a short peptide
does not present enough surface to define a well-formed microenvironment. For
peptide sets, chains are supplied explicitly and extraction runs one-way only:

```bash
python scripts/run_extract_features_from_manifest.py \
    --manifest-csv data/peptides/manifest.csv \
    --out-dir      data/peptides/interfaces/
```

Manifest columns: `pdb_id, pdb_path, ligand_chain, receptor_chain`.

---

## Stage 2 — Negative sampling (nil)

Generates the negative class: environments at random points in the
solvent-accessible shell around each protein, encoded identically to positives.

```bash
python scripts/run_calc_nil_batch.py \
    --input        data/sabdab/data/ \
    --output       data/sabdab/negativeSamples/randomSampling/ \
    --open3d-script scripts/open3d_processing.py \
    --conda-env    open3d_env \
    --report       data/sabdab/negativeSamples/nil_report.csv
```

**Outputs:** `<pdb>/{AB,AG}/random_pts.npy`, `data_target.pkl`, BILD files for
visual inspection.

Points are drawn on a voxel grid (0.5 Å, 4 Å padding) with a distance-dependent
weight that excludes positions within 2 Å of any atom and decays with distance
(effective shell ≈ 2–14.5 Å). Point count scales with solvent-accessible surface
area. Both the antibody and antigen sides are sampled independently.

---

## Stage 3 — Broom stacking

Aggregates per-PDB environments into per-shard, per-split tensors, applying the
**validity filter** that defines the training distribution.

```bash
python scripts/run_stack_features.py \
    --input-root  data/sabdab/interfaces/ \
    --output-root data/sabdab/brooms/ \
    --splits      data/sabdab/clustering/cluster_splits_70/pdb_splits.csv
```

**Outputs:** `brooms/<shard>/target_{positions,classes}_{train,val,test}.npz`
and `stacking_report.csv` (kept/skipped counts with reasons).

An environment is retained only if, among its 25 nearest neighbours, ≥5 lie
within 8 Å and ≤3 lie beyond 13 Å, and it contains ≥2 distinct atom classes —
selecting locally dense, well-enclosed environments. Retention varies by
chemistry (≈16 % for aliphatic shards, ≈50 % for polar/aromatic).

---

## Stage 4 — Training

Trains one binary Perceiver model per shard class (36 models).

```bash
# all shards
python scripts/train_all_models.py \
    --input-dir  data/sabdab/interfaces/ \
    --random-dir data/sabdab/negativeSamples/randomSampling/ \
    --log-dir    logs/

# one shard
python scripts/train_model.py --shard Tyr_1
```

**Outputs:** `logs/<shard>/run_<timestamp>/best_<shard>_model.keras`,
`hyperparameters.json`, training history.

**Task.** One-vs-all per shard. Negatives are a 50/50 mix of random surface
environments (stage 2) and **cross-shard negatives** — native environments of
*other* shard classes, including the other anchors of the same functional group.
Requiring each anchor model to reject its sibling anchors is what forces it to
learn the specific anchor position, so the ensemble of anchor maps resolves group
orientation. Negatives are **resampled every epoch**.

**Defaults** (see `config/training.py`): k = 25 neighbours, 12 rotation
augmentations, batch 24, Adam @ 1e-4, ≤15 epochs, early stopping patience 5,
latent dim 32. Best checkpoint selected by validation loss.


---

## Stage 5 — Evaluation

```bash
python scripts/run_evaluation.py \
    --model-config   configs/models.json \
    --dataset-config configs/dataset_test.json \
    --output-root    evaluation/runs/ \
    --run-name       holdout_test \
    --all
```

`--model-config` maps shard → model path. `--dataset-config` selects the
evaluation set, either by split (`split_csv` + `selection_column`/`value`) or by
an explicit id list. Each run copies its configs and writes a manifest, so results
are traceable to exact models and structures.

**Outputs:** `evaluation/runs/<name>/<shard>/predictions.csv` (with
`y_true, y_score, source_shard, source_pdb`), per-model metrics, run manifest.

**Metric definition.** For a shard, positives are that shard's environments and
negatives are **all other shards'** environments from the *same* held-out
structures. Random surface negatives are **training-only** and excluded from
evaluation. Because negatives greatly outnumber positives, ROC-AUC is the
reported metric.

**Plots (incl. the pairwise discrimination matrix):**

```bash
python scripts/plot_evaluation_analysis.py \
    --run-dir evaluation/runs/holdout_test \
    --annotate-heatmaps
```

---

## Stage 6 — TPM generation

Evaluates a trained shard model over a 3D grid around a target.

```bash
python scripts/run_tp_mapping_batch.py \
    --jobs   jobs.csv \
    --outdir tp_maps/ \
    --mode   gpu --gpus 0
```

`jobs.csv` columns — required: `pdb, target_chains, map_name, model_logs`;
optional: `ligand, box_mode, box_chain, box_res_num, box_res_name, voxel_size,
padding, batch_size`.

**Box modes:** `ligand` (grid spans a reference ligand), `chains` (spans all
target atoms), `residue` (spans one residue).

**Outputs:** `<job>/inference/<map_name>.mrc` (open in ChimeraX/PyMOL alongside
the structure), `tpm_predictions.npy`, `inference_metadata.json`, and `job.json`
recording full provenance.

The neighbour count `k` is read from the model's own `hyperparameters.json`, so
inference and training always use an identical input representation.

---

## Configuration reference

All scientific hyperparameters live in `src/proteintpm/config/`:

| File | Contents |
|---|---|
| `extraction.py` | `ligand_depth` (13 Å), `receptor_depth` (35 Å), `interaction_threshold` (40 Å) — permissive pre-filters that bound computation, **not** the operative selection |
| `stacking.py` | Broom validity filter: `broom_max_neighbours` (100), `broom_filter_window` (25), `broom_lower_threshold` (8 Å), `broom_min_count_low` (5), `broom_upper_threshold` (13 Å), `broom_max_count_high` (3) |
| `training.py` | `DEFAULT_MAX_NEIGHBOURS` (25), `DEFAULT_NUM_ROTATIONS` (12), batch/LR/epochs/patience, `DEFAULT_SHARDS` (the 36 shard classes) |

Negative-sampling parameters are in `negatives/nil/constants.py` (`NILParams`).

---

## Repository layout

```
src/proteintpm/
    config/        scientific hyperparameters (extraction, stacking, training)
    data/          PDB parsing, chain assignment, interface + shard definition
    features/      atom-class labelling, amide-H reconstruction, environments
    clustering/    CDR extraction, CD-HIT wrapper, split assignment, diagnostics
    negatives/     nil generation (voxel grid, surface sampling) + broom building
    geometry/      voxel grids, Open3D interface
    sampling/      point sampling and distance weighting
    pipeline/      extraction + stacking + training orchestration
    ml/            datasets, Perceiver model, negative sampling, augmentation
    tp_mapping/    TPM setup, grid inference, MRC output
    validation/    evaluation, metrics, descriptors, motif analysis, plots
    visualization/ BILD output for structural inspection
    io/            NPZ / PDB writers
scripts/           runnable entry points
```

---

## Preparing your own structures

For **inference on your own PDBs**, note the pipeline's assumptions:

- **Single model.** Only the first model is read; NMR ensembles should be split.
- **Alternate conformations** are resolved automatically (highest occupancy).
- **HETATM records are dropped** — waters, ions, ligands, and modified residues
  (including selenomethionine) are excluded.
- **Hydrogens are stripped**; the backbone amide H is reconstructed geometrically.
- **Chain identifiers must be alphabetic** — structures with numeric chain IDs
  are rejected.
- Antibody–antigen structures for stage 0/1 must carry SAbDab-style Chothia
  `REMARK` annotations; otherwise supply chains explicitly via a manifest.

---

## Known limitations

- **Histidine.** Deposited protonation states are unreliable, so all histidine
  variants are treated identically and both imidazole nitrogens share one atom
  class. Histidine environments are therefore represented less faithfully.
- **Neighbour count.** `k = 25` was carried over from earlier experiments; the
  upper bound of 100 was not optimised per shard.
- **Negative point sampling** used an unseeded draw for the published data; the
  archived negative set is provided (see Data availability). Regenerating will
  produce a different — statistically equivalent — set.
- **Confidence intervals** for the reported metrics are not computed by this
  package.
- **TPM model selection** picks the newest non-`final_model.keras` checkpoint in
  a run directory when several are present.
- **Open3D containment** uses a single-ray even-odd test, which can misclassify
  rare grazing voxels.

---

## Data availability

Training data derive from **SAbDab**; evaluation additionally uses an independent
peptide–protein set and a **PDBBind**-derived affinity benchmark. Redistribution
of source structures is subject to the original databases' terms.


---

## Citing
If you use this work, please cite:

Kienlein, Maximilian, et al.  
["Local structural preference maps encode transferable protein-interface energetics."](https://www.biorxiv.org/content/10.64898/2026.09.18.752156v1.abstract)  
*bioRxiv* (2026): 2026-09.