"""Dataset characterization, temporal splitting and positive feedback."""

from dataclasses import dataclass
from typing import Dict
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalSplit:
    """Chronological train/validation/test data and the two cut timestamps."""

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_end: pd.Timestamp
    validation_end: pd.Timestamp


def characterize_dataset(
    users: pd.DataFrame,
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
) -> Dict[str, object]:
    """Return the dataset facts needed to document the recommender experiment."""
    user_counts = ratings.groupby("user_id").size()
    item_counts = ratings.groupby("movie_id").size()
    timestamp = pd.to_datetime(ratings["timestamp"], unit="s")
    return {
        "users": len(users),
        "items": len(movies),
        "interactions": len(ratings),
        "active_users": user_counts.size,
        "active_items": item_counts.size,
        "density": len(ratings) / (len(users) * len(movies)),
        "active_density": len(ratings) / (user_counts.size * item_counts.size),
        "interactions_per_user": user_counts.describe().to_dict(),
        "interactions_per_item": item_counts.describe().to_dict(),
        "rating_distribution": ratings["rating"].value_counts().sort_index().to_dict(),
        "time_start": timestamp.min(),
        "time_end": timestamp.max(),
        "activity_by_month": timestamp.dt.to_period("M").value_counts().sort_index(),
    }


def temporal_split(
    ratings: pd.DataFrame,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
) -> TemporalSplit:
    """Split ratings chronologically into 80/10/10 train/validation/test sets.

    Target 80/10/10 by row count, moving cuts right to keep timestamp ties
    together. Each held-out timestamp is strictly later than its fit data.
    """
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be below 1.")
    if ratings.empty:
        raise ValueError("Cannot split an empty ratings DataFrame.")

    ordered = ratings.sort_values(
        ["timestamp", "user_id", "movie_id"], kind="mergesort"
    ).reset_index(drop=True)
    train_end_index = int(len(ordered) * train_fraction)
    validation_end_index = int(
        len(ordered) * (train_fraction + validation_fraction)
    )
    if train_end_index < 1 or validation_end_index <= train_end_index:
        raise ValueError("Not enough rows for three nonempty splits.")
    timestamps = ordered["timestamp"].to_numpy()
    train_end_index = int(np.searchsorted(timestamps, timestamps[train_end_index - 1], side="right"))
    validation_end_index = int(np.searchsorted(timestamps, timestamps[validation_end_index - 1], side="right"))
    if not 0 < train_end_index < validation_end_index < len(ordered):
        raise ValueError("Timestamp ties prevent three nonempty temporal splits.")
    train = ordered.iloc[:train_end_index].copy()
    validation = ordered.iloc[train_end_index:validation_end_index].copy()
    test = ordered.iloc[validation_end_index:].copy()
    return TemporalSplit(
        train=train,
        validation=validation,
        test=test,
        train_end=pd.to_datetime(train["timestamp"].iloc[-1], unit="s"),
        validation_end=pd.to_datetime(validation["timestamp"].iloc[-1], unit="s"),
    )


def _validate_rating_columns(ratings: pd.DataFrame) -> None:
    required = {"user_id", "movie_id", "rating", "timestamp"}
    missing = required.difference(ratings.columns)
    if missing:
        raise ValueError(f"ratings is missing required columns: {sorted(missing)}")


def positive_feedback(
    ratings: pd.DataFrame,
    positive_threshold: float = 4.0,
) -> pd.DataFrame:
    """Return one implicit positive interaction per user-item pair."""
    _validate_rating_columns(ratings)
    if not 0 <= positive_threshold <= 5:
        raise ValueError("positive_threshold must be between 0 and 5.")
    positives = ratings[ratings["rating"] >= positive_threshold].copy()
    return positives.drop_duplicates(["user_id", "movie_id"])
