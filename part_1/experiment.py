"""Validation tuning and final baseline evaluation."""

from typing import Iterable, Tuple
import pandas as pd
from .process_data import TemporalSplit, temporal_split
from .models import item_knn_recommendations, popular_recommendations
from .evaluation import evaluate_at_k


def tune_item_knn(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    catalogue: Iterable[int],
    neighborhood_sizes: Iterable[int] = (20, 50, 100),
    similarities: Iterable[str] = ("cosine", "jaccard"),
    positive_threshold: float = 4.0,
) -> pd.DataFrame:
    """Evaluate k and similarity choices on validation and return all scores."""
    users = validation["user_id"].unique()
    rows = []
    for similarity in similarities:
        for k_neighbors in neighborhood_sizes:
            recommendations = item_knn_recommendations(
                train, users, catalogue, 10, k_neighbors, similarity, positive_threshold
            )
            metrics = evaluate_at_k(
                recommendations, validation, catalogue, 10, positive_threshold
            )
            rows.append(
                {
                    "model": "item_knn",
                    "similarity": similarity,
                    "k_neighbors": k_neighbors,
                    **metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["ndcg@10", "recall@10", "precision@10"], ascending=False
    ).reset_index(drop=True)


def run_baseline_experiment(
    ratings: pd.DataFrame | TemporalSplit,
    catalogue: Iterable[int],
    positive_threshold: float = 4.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tune and evaluate an existing split, or split ratings for notebook callers."""
    split = ratings if isinstance(ratings, TemporalSplit) else temporal_split(ratings)
    tuning = tune_item_knn(
        split.train, split.validation, catalogue, positive_threshold=positive_threshold
    )
    best = tuning.iloc[0]
    train_validation = pd.concat([split.train, split.validation], ignore_index=True)
    test_users = split.test["user_id"].unique()
    rows = []
    knn_recommendations = item_knn_recommendations(
        train_validation,
        test_users,
        catalogue,
        k_neighbors=int(best["k_neighbors"]),
        similarity=str(best["similarity"]),
        positive_threshold=positive_threshold,
    )
    rows.append(
        {
            "model": f"item_knn_{best['similarity']}_k{int(best['k_neighbors'])}",
            **evaluate_at_k(knn_recommendations, split.test, catalogue, 10, positive_threshold),
        }
    )
    variants = [
        ("popular_all_time_all_ratings", False, None),
        ("popular_all_time_positive", True, None),
        ("popular_recent_180d_all_ratings", False, 180),
        ("popular_recent_180d_positive", True, 180),
    ]
    for name, count_positive, recent_days in variants:
        recommendations = popular_recommendations(
            train_validation,
            test_users,
            catalogue,
            count_positive=count_positive,
            positive_threshold=positive_threshold,
            recent_days=recent_days,
            reference_timestamp=int(train_validation["timestamp"].max()),
        )
        rows.append(
            {
                "model": name,
                **evaluate_at_k(recommendations, split.test, catalogue, 10, positive_threshold),
            }
        )
    return tuning, pd.DataFrame(rows)
