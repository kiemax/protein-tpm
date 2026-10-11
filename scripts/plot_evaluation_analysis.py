import argparse
from pathlib import Path

from proteintpm.validation.plots import (
    build_confusion_score_matrix,
    build_model_score_correlation_matrix,
    build_pairwise_auc_matrix,
    load_prediction_tables,
    plot_all_model_vs_all_grids,
    plot_confusion_heatmap,
    plot_correlation_heatmap,
    plot_pairwise_auc_heatmap,
    plot_pr_curves,
    plot_roc_curves,
    plot_score_distributions,
    save_matrix_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate analysis plots for one evaluation run.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Evaluation run directory, e.g. evaluation/runs/<run_name>",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to <run-dir>/plots",
    )
    parser.add_argument(
        "--annotate-heatmaps",
        action="store_true",
        help="Write numeric values into heatmap cells",
    )
    return parser.parse_args()


def load_prediction_csv_map(run_dir: Path) -> dict[str, Path]:
    pred_map: dict[str, Path] = {}

    for child in sorted(run_dir.iterdir()):
        if not child.is_dir():
            continue

        pred_csv = child / "predictions.csv"
        if pred_csv.is_file():
            pred_map[child.name] = pred_csv

    if not pred_map:
        raise RuntimeError(f"No predictions.csv files found under run dir: {run_dir}")

    return pred_map


def main():
    args = parse_args()

    run_dir = args.run_dir
    output_dir = args.output_dir if args.output_dir is not None else run_dir / "plots"

    pred_csvs = load_prediction_csv_map(run_dir)
    pred_tables = load_prediction_tables(pred_csvs)

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_curves(pred_tables, output_dir / "roc_curves.png")
    plot_pr_curves(pred_tables, output_dir / "pr_curves.png")
    plot_score_distributions(pred_tables, output_dir / "score_distributions.png")

    confusion_matrix = build_confusion_score_matrix(pred_tables)
    save_matrix_csv(confusion_matrix, output_dir / "mean_score_by_source_shard.csv")
    plot_confusion_heatmap(
        confusion_matrix,
        output_dir / "mean_score_by_source_shard.png",
        title="Mean Predicted Score by Source Shard",
        annotate=args.annotate_heatmaps,
    )

    corr_matrix = build_model_score_correlation_matrix(pred_tables)
    save_matrix_csv(corr_matrix, output_dir / "model_score_correlation.csv")
    plot_correlation_heatmap(
        corr_matrix,
        output_dir / "model_score_correlation.png",
        title="Correlation Between Model Scores",
        annotate=args.annotate_heatmaps,
    )

    pairwise_auc_matrix = build_pairwise_auc_matrix(pred_tables)
    save_matrix_csv(pairwise_auc_matrix, output_dir / "pairwise_auc_matrix.csv")
    plot_pairwise_auc_heatmap(
        pairwise_auc_matrix,
        output_dir / "pairwise_auc_matrix.png",
        title="Pairwise Cross-Shard Discrimination (AUC)",
        annotate=args.annotate_heatmaps,
    )

    plot_all_model_vs_all_grids(
        pred_tables,
        output_dir / "model_vs_all_grids",
        gridsize=35,
    )


if __name__ == "__main__":
    main()
