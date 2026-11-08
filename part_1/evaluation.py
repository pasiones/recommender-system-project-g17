"""Top-k ranking metrics and catalogue coverage."""

from typing import Dict, Iterable
import numpy as np
import pandas as pd


def evaluate_at_k(
    recommendations: Dict[int, list[int]],
    test: pd.DataFrame,
    catalogue: Iterable[int],
    k: int = 10,
    positive_threshold: float = 4.0,
) -> Dict[str, float]:
    """Calculate Precision@k, Recall@k, NDCG@k, and catalogue coverage."""
    if k < 1:
        raise ValueError("k must be at least 1.")
    relevant = (
        test[test["rating"] >= positive_threshold]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )
    evaluated_users = [user_id for user_id in recommendations if user_id in relevant]
    if not evaluated_users:
        raise ValueError("No evaluated users have positive test feedback.")
    precisions, recalls, ndcgs = [], [], []
    recommended_items: set[int] = set()
    for user_id in evaluated_users:
        ranked = recommendations[user_id][:k]
        hits = [int(item_id in relevant[user_id]) for item_id in ranked]
        recommended_items.update(ranked)
        precisions.append(sum(hits) / k)
        recalls.append(sum(hits) / len(relevant[user_id]))
        dcg = sum(hit / np.log2(rank + 2) for rank, hit in enumerate(hits))
        ideal_hits = min(k, len(relevant[user_id]))
        idcg = sum(1 / np.log2(rank + 2) for rank in range(ideal_hits))
        ndcgs.append(dcg / idcg if idcg else 0.0)
    return {
        f"precision@{k}": float(np.mean(precisions)),
        f"recall@{k}": float(np.mean(recalls)),
        f"ndcg@{k}": float(np.mean(ndcgs)),
        "catalogue_coverage": len(recommended_items) / len(set(catalogue)),
        "evaluated_users": float(len(evaluated_users)),
    }
