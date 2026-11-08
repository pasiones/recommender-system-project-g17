"""Popularity and item-based nearest-neighbor recommenders."""

from typing import Dict, Iterable
import numpy as np
import pandas as pd
from .process_data import _validate_rating_columns, positive_feedback


def _catalogue_ids(
    train: pd.DataFrame,
    catalogue: Iterable[int] | None = None,
) -> np.ndarray:
    if catalogue is None:
        catalogue = train["movie_id"].unique()
    return np.asarray(sorted(set(catalogue)))


def _recommend_from_scores(
    scores: np.ndarray,
    item_ids: np.ndarray,
    seen: set[int],
    n: int,
) -> list[int]:
    if n < 1:
        raise ValueError("n must be at least 1.")
    scores = scores.copy()
    for item_id in seen:
        positions = np.flatnonzero(item_ids == item_id)
        if len(positions):
            scores[positions[0]] = -np.inf
    available = min(n, int(np.isfinite(scores).sum()))
    if available == 0:
        return []
    candidate_positions = np.flatnonzero(np.isfinite(scores))
    candidate_positions = candidate_positions[np.lexsort(
        (item_ids[candidate_positions], -scores[candidate_positions])
    )][:available]
    return item_ids[candidate_positions].astype(int).tolist()


def popular_recommendations(
    train: pd.DataFrame,
    users: Iterable[int],
    catalogue: Iterable[int] | None = None,
    n: int = 10,
    count_positive: bool = True,
    positive_threshold: float = 4.0,
    recent_days: int | None = None,
    reference_timestamp: int | None = None,
) -> Dict[int, list[int]]:
    """Recommend popularity-ranked unseen items against the full catalogue."""
    _validate_rating_columns(train)
    if recent_days is not None and recent_days <= 0:
        raise ValueError("recent_days must be positive.")
    item_ids = _catalogue_ids(train, catalogue)
    counted = train
    if recent_days is not None:
        reference = (
            int(train["timestamp"].max())
            if reference_timestamp is None
            else reference_timestamp
        )
        cutoff = reference - recent_days * 24 * 60 * 60
        counted = counted[counted["timestamp"] >= cutoff]
    if count_positive:
        counted = counted[counted["rating"] >= positive_threshold]
    counts = counted.groupby("movie_id").size()
    ranked = sorted(
        item_ids.tolist(),
        key=lambda item_id: (-int(counts.get(item_id, 0)), item_id),
    )
    seen_by_user = train.groupby("user_id")["movie_id"].apply(set).to_dict()
    return {
        int(user_id): _recommend_from_scores(
            np.array([len(item_ids) - rank for rank in range(len(item_ids))], dtype=float),
            np.asarray(ranked),
            seen_by_user.get(user_id, set()),
            n,
        )
        for user_id in users
    }


def item_similarity(
    train: pd.DataFrame,
    catalogue: Iterable[int] | None = None,
    similarity: str = "cosine",
    positive_threshold: float = 4.0,
):
    """Build a sparse item-item cosine or Jaccard similarity matrix."""
    if similarity not in {"cosine", "jaccard"}:
        raise ValueError("similarity must be 'cosine' or 'jaccard'.")
    positives = positive_feedback(train, positive_threshold)
    item_ids = _catalogue_ids(train, catalogue)
    user_ids = np.sort(positives["user_id"].unique())
    from scipy.sparse import csr_matrix, diags

    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    rows = positives["user_id"].map(user_index)
    cols = positives["movie_id"].map(item_index)
    matrix = csr_matrix(
        (np.ones(len(positives)), (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
    )
    cooccurrence = (matrix.T @ matrix).tocsr()
    cooccurrence.setdiag(0)
    cooccurrence.eliminate_zeros()
    if similarity == "cosine":
        degrees = np.asarray(matrix.sum(axis=0)).ravel()
        normalizer = np.zeros_like(degrees, dtype=float)
        np.divide(1, np.sqrt(degrees), out=normalizer, where=degrees > 0)
        result = diags(normalizer) @ cooccurrence @ diags(normalizer)
    else:
        degrees = np.asarray(matrix.sum(axis=0)).ravel()
        result = cooccurrence.tocoo()
        union = degrees[result.row] + degrees[result.col] - result.data
        result.data = np.divide(
            result.data, union, out=np.zeros_like(result.data), where=union != 0
        )
        result = result.tocsr()
    return item_ids, result


def item_knn_recommendations(
    train: pd.DataFrame,
    users: Iterable[int],
    catalogue: Iterable[int] | None = None,
    n: int = 10,
    k_neighbors: int = 50,
    similarity: str = "cosine",
    positive_threshold: float = 4.0,
) -> Dict[int, list[int]]:
    """Recommend with item-based kNN using positive implicit feedback."""
    if k_neighbors < 1:
        raise ValueError("k_neighbors must be at least 1.")
    users = list(users)
    item_ids, similarities = item_similarity(
        train, catalogue, similarity, positive_threshold
    )
    seen_by_user = train.groupby("user_id")["movie_id"].apply(set).to_dict()
    positive = positive_feedback(train, positive_threshold)
    positive_by_user = positive.groupby("user_id")["movie_id"].apply(set).to_dict()
    # Prune each source item's neighborhood once, with deterministic tie breaks.
    similarities = similarities.tocsr()
    for index in range(len(item_ids)):
        start, end = similarities.indptr[index:index + 2]
        order = np.lexsort((item_ids[similarities.indices[start:end]], -similarities.data[start:end]))
        similarities.data[start + order[k_neighbors:]] = 0
    similarities.eliminate_zeros()
    fallback = popular_recommendations(train, users, item_ids, n, positive_threshold=positive_threshold)
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}
    recommendations: Dict[int, list[int]] = {}
    for user_id in users:
        seen = seen_by_user.get(user_id, set())
        indices = [item_index[item_id] for item_id in positive_by_user.get(user_id, set()) if item_id in item_index]
        if not indices:
            recommendations[int(user_id)] = fallback[int(user_id)]
            continue
        scores = np.zeros(len(item_ids), dtype=float)
        for index in indices:
            row = similarities.getrow(index)
            scores[row.indices] += row.data
        recommendations[int(user_id)] = _recommend_from_scores(
            scores, item_ids, seen, n
        )
    return recommendations
