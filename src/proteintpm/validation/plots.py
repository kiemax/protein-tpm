import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve

DISPLAY_NAME_BY_SHARD = {
    "Backbone_1": "BB-Cβ",
    "Backbone_2": "BB-O",
    "Backbone_3": "BB-H",
    "Terminal_amino_1": "Lys-NZ",
    "Terminal_amino_2": "Lys-CE",
    "Guanidino_1": "Arg-CZ",
    "Guanidino_2": "Arg-NH",
    "Hydroxyl_1": "Ser/Thr-O",
    "Hydroxyl_2": "Ser/Thr-Cβ",
    "Thiol_1": "Cys-SG",
    "Thiol_2": "Cys-Cβ",
    "Thioether_1": "Met-SD",
    "Thioether_2": "Met-CE",
    "Carboxyl_1": "Asp/Glu-C",
    "Carboxyl_2": "Asp/Glu-O",
    "Carboxamid_1": "Asn/Gln-C",
    "Carboxamid_2": "Asn/Gln-O",
    "Carboxamid_3": "Asn/Gln-N",
    "Phe_1": "Phe-CG",
    "Phe_2": "Phe-CE",
    "Tyr_1": "Tyr-OH",
    "Tyr_2": "Tyr-CD",
    "Trp_1": "Trp-CG",
    "Trp_2": "Trp-NE1",
    "Trp_3": "Trp-CZ3",
    "His_1": "His-CG",
    "His_2": "His-ND1",
    "His_3": "His-NE2",
    "Pro_1": "Pro-CA",
    "Pro_2": "Pro-CG",
    "Pro_3": "Pro-N",
    "Gly_1": "Gly-CA",
    "Ala_1": "Ala-Cβ",
    "Leu_1": "Leu-CG",
    "Val_1": "Val-Cβ",
    "Ile_1": "Ile-Cβ",
}

SHARD_ORDER = list(DISPLAY_NAME_BY_SHARD.keys())

FAMILY_BY_SHARD: dict[str, str] = {
    "Backbone_1": "hydrophobic",
    "Backbone_2": "polar",
    "Backbone_3": "backbone_misc",
    "Terminal_amino_1": "charged",
    "Guanidino_2": "charged",
    "Carboxyl_2": "charged",
    "Terminal_amino_2": "hydrophobic",
    "Guanidino_1": "hydrophobic",
    "Carboxyl_1": "hydrophobic",
    "Hydroxyl_1": "polar",
    "Carboxamid_2": "polar",
    "Carboxamid_3": "polar",
    "Tyr_1": "polar",
    "Trp_2": "polar",
    "His_2": "polar",
    "His_3": "polar",
    "Thiol_1": "polar",
    "Hydroxyl_2": "hydrophobic",
    "Carboxamid_1": "hydrophobic",
    "Thiol_2": "hydrophobic",
    "Thioether_1": "hydrophobic",
    "Thioether_2": "hydrophobic",
    "Phe_1": "aromatic",
    "Phe_2": "aromatic",
    "Tyr_2": "aromatic",
    "Trp_1": "aromatic",
    "Trp_3": "aromatic",
    "His_1": "aromatic",
    "Gly_1": "hydrophobic",
    "Ala_1": "hydrophobic",
    "Leu_1": "hydrophobic",
    "Val_1": "hydrophobic",
    "Ile_1": "hydrophobic",
    "Pro_1": "hydrophobic",
    "Pro_2": "hydrophobic",
    "Pro_3": "hydrophobic",
}


def get_family(shard_name: str) -> str:
    return FAMILY_BY_SHARD.get(shard_name, "other")


def _ordered_labels(labels: list[str]) -> list[str]:
    preferred = [label for label in SHARD_ORDER if label in labels]
    remainder = sorted(label for label in labels if label not in set(preferred))
    return preferred + remainder


def _reorder_square_like_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    row_order = _ordered_labels(list(matrix.index))
    col_order = _ordered_labels(list(matrix.columns))
    return matrix.loc[row_order, col_order]


def _display_labels(labels: list[str]) -> list[str]:
    return [DISPLAY_NAME_BY_SHARD.get(label, label) for label in labels]


def _family_boundaries(labels: list[str]) -> list[int]:
    if not labels:
        return []

    families = [get_family(label) for label in labels]
    return [i for i in range(1, len(families)) if families[i] != families[i - 1]]


def load_prediction_tables(prediction_csvs: dict[str, Path]) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for shard, path in prediction_csvs.items():
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing prediction CSV for {shard}: {path}")
        out[shard] = pd.read_csv(path)
    return out


def plot_roc_curves(pred_tables: dict[str, pd.DataFrame], output_path: Path) -> None:
    plt.figure(figsize=(8, 8))

    for shard, df in sorted(pred_tables.items()):
        y_true = df["y_true"].to_numpy()
        y_score = df["y_score"].to_numpy()

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{shard} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Holdout ROC Curves")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_pr_curves(pred_tables: dict[str, pd.DataFrame], output_path: Path) -> None:
    plt.figure(figsize=(8, 8))

    for shard, df in sorted(pred_tables.items()):
        y_true = df["y_true"].to_numpy()
        y_score = df["y_score"].to_numpy()

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, label=f"{shard} (AUC={pr_auc:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Holdout Precision-Recall Curves")
    plt.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_score_distributions(
    pred_tables: dict[str, pd.DataFrame],
    output_path: Path,
    bins: int = 50,
) -> None:
    shards = sorted(pred_tables.keys())
    n = len(shards)
    ncols = 4
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, shard in zip(axes, shards):
        df = pred_tables[shard]
        pos = df.loc[df["y_true"] == 1, "y_score"].to_numpy()
        neg = df.loc[df["y_true"] == 0, "y_score"].to_numpy()

        ax.hist(neg, bins=bins, density=True, alpha=0.6, label="negative")
        ax.hist(pos, bins=bins, density=True, alpha=0.6, label="positive")
        ax.set_title(DISPLAY_NAME_BY_SHARD.get(shard, shard))
        ax.set_xlabel("Predicted score")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    for ax in axes[len(shards) :]:
        ax.axis("off")

    fig.suptitle("Holdout Score Distributions", y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_confusion_score_matrix(pred_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for model_shard, df in pred_tables.items():
        tmp = (
            df.groupby("source_shard", dropna=False)["y_score"]
            .mean()
            .reset_index()
            .rename(columns={"y_score": model_shard})
        )
        rows.append(tmp)

    merged = rows[0]
    for row_df in rows[1:]:
        merged = merged.merge(row_df, on="source_shard", how="outer")

    matrix = merged.fillna(np.nan).set_index("source_shard")
    return _reorder_square_like_matrix(matrix)


def _annotate_heatmap(ax, matrix: pd.DataFrame, fmt: str = ".2f") -> None:
    values = matrix.to_numpy()
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return

    midpoint = (np.nanmin(finite_vals) + np.nanmax(finite_vals)) / 2.0

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            if not np.isfinite(val):
                continue
            color = "white" if val > midpoint else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=7, color=color)


def plot_confusion_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str = "Mean Predicted Score by Source Shard",
    annotate: bool = False,
) -> None:
    plot_matrix = _reorder_square_like_matrix(matrix)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(
        plot_matrix.to_numpy(),
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=np.nanmax(plot_matrix.to_numpy()),
    )

    ax.set_xticks(np.arange(plot_matrix.shape[1]))
    ax.set_xticklabels(_display_labels(list(plot_matrix.columns)), rotation=90)
    ax.set_yticks(np.arange(plot_matrix.shape[0]))
    ax.set_yticklabels(_display_labels(list(plot_matrix.index)))

    ax.set_xlabel("Model shard")
    ax.set_ylabel("Source shard")
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=8)

    row_labels = list(plot_matrix.index)
    col_labels = list(plot_matrix.columns)
    for i, row_name in enumerate(row_labels):
        if row_name in col_labels:
            j = col_labels.index(row_name)
            ax.add_patch(
                plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=1.5)
            )

    row_bounds = _family_boundaries(row_labels)
    col_bounds = _family_boundaries(col_labels)
    for boundary in row_bounds:
        ax.axhline(boundary - 0.5, color="white", linewidth=1.5)
    for boundary in col_bounds:
        ax.axvline(boundary - 0.5, color="white", linewidth=1.5)

    if annotate:
        _annotate_heatmap(ax, plot_matrix)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Mean predicted score", rotation=270, labelpad=15)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_model_score_correlation_matrix(pred_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    score_df = pd.DataFrame(
        {model_shard: df["y_score"].to_numpy() for model_shard, df in pred_tables.items()}
    )
    corr = score_df.corr(method="pearson")
    return _reorder_square_like_matrix(corr)


def plot_correlation_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str = "Correlation Between Model Scores",
    annotate: bool = False,
) -> None:
    plot_matrix = _reorder_square_like_matrix(matrix)
    values = plot_matrix.to_numpy(dtype=float)
    n_rows, n_cols = values.shape

    fig, ax = plt.subplots(figsize=(10, 8))

    cmap = plt.cm.coolwarm
    vmin = -1.0
    vmax = 1.0

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    for i in range(n_rows):
        for j in range(n_cols):
            val = values[i, j]

            if not np.isfinite(val):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="lightgrey",
                        edgecolor="none",
                    )
                )
                continue

            scaled = np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0)
            side = np.sqrt(abs(val)) * 0.9
            color = cmap(scaled)

            ax.add_patch(
                plt.Rectangle(
                    (j - side / 2, i - side / 2),
                    side,
                    side,
                    facecolor=color,
                    edgecolor="none",
                )
            )

            if annotate:
                text_color = "white" if abs(val) >= 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    row_bounds = _family_boundaries(list(plot_matrix.index))
    col_bounds = _family_boundaries(list(plot_matrix.columns))
    for boundary in row_bounds:
        ax.axhline(boundary - 0.5, color="white", linewidth=1.5)
    for boundary in col_bounds:
        ax.axvline(boundary - 0.5, color="white", linewidth=1.5)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(_display_labels(list(plot_matrix.columns)), rotation=90)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(_display_labels(list(plot_matrix.index)))

    ax.set_xlabel("Model shard")
    ax.set_ylabel("Model shard")
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=8)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.set_ylabel("Pearson correlation", rotation=270, labelpad=15)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def save_matrix_csv(matrix: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_path)


def build_pairwise_auc_matrix(pred_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    model_shards = _ordered_labels(list(pred_tables.keys()))
    source_shards = _ordered_labels(
        [
            str(shard)
            for shard in {
                shard
                for df in pred_tables.values()
                for shard in df["source_shard"].dropna().unique()
            }
        ]
    )

    matrix = pd.DataFrame(index=model_shards, columns=source_shards, dtype=float)

    for model_shard, df in pred_tables.items():
        pos_scores = df.loc[df["y_true"] == 1, "y_score"].to_numpy()
        if pos_scores.size == 0:
            continue

        neg_df = df.loc[df["y_true"] == 0].copy()

        for source_shard in source_shards:
            neg_scores = neg_df.loc[neg_df["source_shard"] == source_shard, "y_score"].to_numpy()
            if neg_scores.size == 0:
                matrix.loc[model_shard, source_shard] = np.nan
                continue

            y_true = np.concatenate(
                [np.ones(pos_scores.shape[0], dtype=int), np.zeros(neg_scores.shape[0], dtype=int)]
            )
            y_score = np.concatenate([pos_scores, neg_scores])

            try:
                matrix.loc[model_shard, source_shard] = roc_auc_score(y_true, y_score)
            except ValueError:
                matrix.loc[model_shard, source_shard] = np.nan

    return _reorder_square_like_matrix(matrix)


def build_score_dataframe(pred_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    model_names = _ordered_labels(list(pred_tables.keys()))

    lengths = {name: len(df) for name, df in pred_tables.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Prediction tables have inconsistent lengths: {lengths}")

    return pd.DataFrame(
        {model_name: pred_tables[model_name]["y_score"].to_numpy() for model_name in model_names}
    )


def plot_model_vs_all_grid(
    score_df: pd.DataFrame,
    reference_model: str,
    output_path: Path,
    gridsize: int = 35,
) -> None:
    model_names = list(score_df.columns)

    ncols = 6
    nrows = 6
    if len(model_names) != ncols * nrows:
        raise ValueError("Expected 36 models for a 6x6 grid")

    x = score_df[reference_model].to_numpy()
    reference_display = DISPLAY_NAME_BY_SHARD.get(reference_model, reference_model)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 14))
    axes = axes.ravel()

    for ax, model_name in zip(axes, model_names):
        y = score_df[model_name].to_numpy()
        model_display = DISPLAY_NAME_BY_SHARD.get(model_name, model_name)

        if model_name == reference_model:
            ax.text(
                0.5,
                0.5,
                reference_display,
                ha="center",
                va="center",
                fontsize=11,
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        else:
            r = np.corrcoef(x, y)[0, 1]
            ax.hexbin(x, y, gridsize=gridsize, extent=(0, 1, 0, 1), mincnt=1)
            ax.plot([0, 1], [0, 1], linewidth=0.7, alpha=0.5)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{model_display}\nr={r:.2f}", fontsize=8)
            ax.set_xticks([0, 0.5, 1])
            ax.set_yticks([0, 0.5, 1])
            ax.tick_params(labelsize=6)

    fig.suptitle(f"Model score comparisons (x = {reference_display})", y=0.995)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_all_model_vs_all_grids(
    pred_tables: dict[str, pd.DataFrame],
    output_dir: Path,
    gridsize: int = 35,
) -> None:
    score_df = build_score_dataframe(pred_tables)

    for reference_model in score_df.columns:
        plot_model_vs_all_grid(
            score_df=score_df,
            reference_model=reference_model,
            output_path=output_dir / f"{reference_model}_vs_all.png",
            gridsize=gridsize,
        )


def plot_pairwise_auc_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str = "Pairwise Cross-Shard Discrimination (AUC)",
    annotate: bool = False,
) -> None:
    plot_matrix = _reorder_square_like_matrix(matrix)
    values = plot_matrix.to_numpy(dtype=float)
    n_rows, n_cols = values.shape

    fig, ax = plt.subplots(figsize=(11, 9))

    cmap = plt.cm.managua.reversed().copy()
    vmin = 0.5
    vmax = 1.0

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(n_rows - 0.5, -0.5)

    for i in range(n_rows):
        for j in range(n_cols):
            val = values[i, j]

            if not np.isfinite(val):
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="lightgrey",
                        edgecolor="none",
                    )
                )
                continue

            scaled = np.clip((val - vmin) / (vmax - vmin), 0.0, 1.0)
            side = np.sqrt(scaled) * 0.9
            color = cmap(scaled)

            ax.add_patch(
                plt.Rectangle(
                    (j - side / 2, i - side / 2),
                    side,
                    side,
                    facecolor=color,
                    edgecolor="none",
                )
            )

            if annotate:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="black")

    row_bounds = _family_boundaries(list(plot_matrix.index))
    col_bounds = _family_boundaries(list(plot_matrix.columns))
    for boundary in row_bounds:
        ax.axhline(boundary - 0.5, color="white", linewidth=1.5)
    for boundary in col_bounds:
        ax.axvline(boundary - 0.5, color="white", linewidth=1.5)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(_display_labels(list(plot_matrix.columns)), rotation=90)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(_display_labels(list(plot_matrix.index)))

    ax.set_xlabel("Input shard")
    ax.set_ylabel("Model shard")
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=8)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax)
    cbar.ax.set_ylabel("Pairwise ROC-AUC", rotation=270, labelpad=15)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
