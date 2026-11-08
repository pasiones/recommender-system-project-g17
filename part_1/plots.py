"""Exploratory plots for MovieLens interactions."""

from pathlib import Path
import numpy as np
import pandas as pd


def plot_interaction_long_tail(
    ratings: pd.DataFrame,
    save_path: str | Path | None = None,
):
    """Plot ranked interaction counts for users and items as bar charts."""
    import matplotlib.pyplot as plt

    user_counts = ratings.groupby("user_id").size().sort_values(ascending=False)
    item_counts = ratings.groupby("movie_id").size().sort_values(ascending=False)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, counts, title, x_label, y_label in zip(
        axes,
        (user_counts, item_counts),
        (
            "User activity distribution (long tail)",
            "Item popularity distribution (long tail)",
        ),
        (
            "User rank (1 = most interactions)",
            "Item rank (1 = most interactions)",
        ),
        (
            "Interactions per user",
            "Interactions per item",
        ),
    ):
        ranks = np.arange(1, len(counts) + 1)
        axis.bar(ranks, counts.to_numpy(), width=1.0, align="center")
        axis.set_title(title)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_yscale("log")
        axis.grid(True, which="both", axis="y", alpha=0.25)
    figure.tight_layout()
    if save_path is not None:
        figure.savefig(save_path, bbox_inches="tight")
    return figure, axes
