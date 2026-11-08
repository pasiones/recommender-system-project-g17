"""Figures and course-template exports for an already computed Part 1 run."""
import json
from pathlib import Path
import platform

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openpyxl import load_workbook

from part_1 import plot_interaction_long_tail


def save_figures(ratings, summary, out):
    """Save descriptive plots without loading data or changing the summary."""
    figure, _ = plot_interaction_long_tail(ratings, out / "long_tail.png")
    plt.close(figure)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    ratings.rating.value_counts().sort_index().plot.bar(ax=axes[0], title="Rating distribution")
    axes[0].set(xlabel="Rating", ylabel="Interactions")
    summary["activity_by_month"].plot.bar(ax=axes[1], title="Monthly rating activity (UTC)")
    axes[1].set(xlabel="Month", ylabel="Interactions")
    fig.tight_layout()
    fig.savefig(out / "ratings_and_time.png", dpi=150)
    plt.close(fig)
    summary["activity_by_month"].to_csv(out / "monthly_activity.csv")


def save_outputs(ratings, summary, split, tuning, results, out):
    """Write metrics, protocol metadata and the workbook from shared inputs."""
    summary = dict(summary)
    summary["activity_by_month"] = {
        str(k): int(v) for k, v in summary["activity_by_month"].items()
    }
    fit = pd.concat([split.train, split.validation])
    eligible = split.test.loc[split.test.rating >= 4, "user_id"].unique()
    fitted_users = set(fit.user_id)
    cold = sum(u not in fitted_users for u in eligible)
    counts = ratings.groupby("movie_id").size().sort_values(ascending=False)
    top_share = counts.iloc[:int(np.ceil(len(counts) * .1))].sum() / len(ratings)
    protocol = {
        "seed": 42, "python": platform.python_version(), "positive_threshold": 4,
        "filtering": "None; 0 rows removed", "target_fractions": [0.8, 0.1, 0.1],
        "train_end_utc": str(split.train_end), "validation_end_utc": str(split.validation_end),
        "split_sizes": [len(split.train), len(split.validation), len(split.test)],
        "evaluated_test_users": len(eligible), "cold_test_users": int(cold),
        "test_users_without_positive": int(split.test.user_id.nunique() - len(eligible)),
        "top_10_percent_active_items_interaction_share": float(top_share),
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, default=str, indent=2), encoding="utf-8")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    tuning.to_csv(out / "validation_grid.csv", index=False)
    best = tuning.iloc[0]
    root = Path(__file__).resolve().parent
    workbook = load_workbook(root / "course_results_template.xlsx")
    sheet = workbook["Part 1 - Baselines"]
    rows = []
    for offset, record in enumerate(results.to_dict("records"), 7):
        knn = record["model"].startswith("item_knn")
        row = ["Item-based kNN" if knn else "Most popular", record["model"],
               best["similarity"] if knn else "N/A", int(best["k_neighbors"]) if knn else "N/A",
               record["precision@10"], record["recall@10"], record["ndcg@10"],
               record["catalogue_coverage"], 42,
               f"Full catalogue; rating>=4 relevance; {len(eligible)} users; fit train+validation"]
        rows.append(row)
        for column, value in enumerate(row, 1):
            sheet.cell(offset, column, value)
    headers = [sheet.cell(6, c).value for c in range(1, 11)]
    pd.DataFrame(rows, columns=headers).to_csv(out / "results.csv", index=False)
    workbook.save(out / "results.xlsx")
