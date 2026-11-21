"""Three additional models: user kNN, biased MF, and content profiles.

The content profile and ranking logic below is the Part 2 self-implementation.
It follows the profile-matching method in Lecture 2.2, rather than calling a
ready-made recommender.
"""

import re

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

from part_1.models import _catalogue_ids, _recommend_from_scores, popular_recommendations
from part_1.process_data import positive_feedback


def content_recommendations(train, movies, users, n=10, year_weight=0.5,
                            profile_weighting="binary", positive_threshold=4.0):
    """Match a liked-movie profile to genre and release-year item features.

    Genres are multi-hot and L2-normalized per movie. The year is centered and
    scaled using catalogue metadata only. Profiles average positively rated
    movies, optionally weighting rating 5 twice as much as rating 4.
    """
    if year_weight < 0 or profile_weighting not in {"binary", "rating"}:
        raise ValueError("Invalid content model settings")
    users = list(map(int, users))
    item_ids = _catalogue_ids(train, movies.movie_id)
    metadata = movies.set_index("movie_id").reindex(item_ids)
    genres = sorted({g for value in metadata.genres.fillna("") for g in str(value).split("|") if g})
    genre_index = {g: i for i, g in enumerate(genres)}
    features = np.zeros((len(item_ids), len(genres) + 1), dtype=np.float32)
    for row, value in enumerate(metadata.genres.fillna("")):
        for genre in str(value).split("|"):
            if genre in genre_index:
                features[row, genre_index[genre]] = 1
    norm = np.linalg.norm(features[:, :len(genres)], axis=1)
    np.divide(features[:, :len(genres)], norm[:, None],
              out=features[:, :len(genres)], where=norm[:, None] > 0)
    years = np.array([int(match.group(1)) if (match := re.search(r"\((\d{4})\)\s*$", str(title)))
                      else np.nan for title in metadata.title], dtype=float)
    valid = np.isfinite(years)
    center = float(np.mean(years[valid])) if valid.any() else 0.0
    scale = float(np.std(years[valid])) if valid.any() else 1.0
    features[:, -1] = np.where(valid, (years - center) / max(scale, 1.0), 0) * year_weight
    item_norm = np.linalg.norm(features, axis=1)
    normalized = np.divide(features, item_norm[:, None],
                           out=np.zeros_like(features), where=item_norm[:, None] > 0)
    index = {item: i for i, item in enumerate(item_ids)}
    positive = positive_feedback(train, positive_threshold)
    profiles = {}
    for user, group in positive.groupby("user_id"):
        positions = [index[item] for item in group.movie_id if item in index]
        if positions:
            weights = np.ones(len(positions)) if profile_weighting == "binary" else (
                group.rating.to_numpy()[:len(positions)] - positive_threshold + 1)
            profile = np.average(features[positions], axis=0, weights=weights)
            length = np.linalg.norm(profile)
            if length:
                profiles[int(user)] = profile / length
    seen = train.groupby("user_id").movie_id.apply(set).to_dict()
    fallback = popular_recommendations(train, users, item_ids, n)
    result = {}
    for user in users:
        if user in profiles:
            scores = normalized @ profiles[user]
            result[user] = _recommend_from_scores(scores, item_ids, seen.get(user, set()), n)
        else:
            result[user] = fallback[user]
    return result


def user_knn_recommendations(train, users, catalogue, n=10, k_neighbors=50,
                             similarity="cosine", positive_threshold=4.0):
    """Positive-feedback user-neighbour CF, with a popularity cold-start fallback."""
    if k_neighbors < 1 or similarity not in {"cosine", "jaccard"}:
        raise ValueError("Invalid user-kNN settings")
    users = list(map(int, users))
    items = _catalogue_ids(train, catalogue)
    positive = positive_feedback(train, positive_threshold)
    known_users = np.sort(train.user_id.unique())
    uindex = {u: i for i, u in enumerate(known_users)}
    iindex = {item: i for i, item in enumerate(items)}
    matrix = csr_matrix((np.ones(len(positive), dtype=np.float32),
                         (positive.user_id.map(uindex), positive.movie_id.map(iindex))),
                        shape=(len(known_users), len(items)))
    matrix.data[:] = 1
    co = (matrix @ matrix.T).tocsr()
    co.setdiag(0)
    co.eliminate_zeros()
    degrees = np.asarray(matrix.sum(axis=1)).ravel()
    if similarity == "cosine":
        inv = np.divide(1, np.sqrt(degrees), out=np.zeros_like(degrees), where=degrees > 0)
        sim = (diags(inv) @ co @ diags(inv)).tocsr()
    else:
        sim = co.tocoo()
        union = degrees[sim.row] + degrees[sim.col] - sim.data
        sim.data = np.divide(sim.data, union, out=np.zeros_like(sim.data), where=union > 0)
        sim = sim.tocsr()
    # Keep the k strongest neighbours per user with user-ID tie breaking.
    for row in range(len(known_users)):
        start, end = sim.indptr[row:row + 2]
        order = np.lexsort((known_users[sim.indices[start:end]], -sim.data[start:end]))
        sim.data[start + order[k_neighbors:]] = 0
    sim.eliminate_zeros()
    seen = train.groupby("user_id").movie_id.apply(set).to_dict()
    fallback = popular_recommendations(train, users, items, n)
    result = {}
    for user in users:
        if user not in uindex or degrees[uindex[user]] == 0:
            result[user] = fallback[user]
            continue
        scores = (sim.getrow(uindex[user]) @ matrix).toarray().ravel()
        if not np.any(scores):
            result[user] = fallback[user]
        else:
            result[user] = _recommend_from_scores(scores, items, seen.get(user, set()), n)
    return result


def mf_recommendations(train, users, catalogue, n=10, factors=24,
                       regularization=0.05, epochs=8, seed=42):
    """Biased regularized matrix factorization trained with mini-batch SGD."""
    if factors < 1 or regularization < 0 or epochs < 1:
        raise ValueError("Invalid MF settings")
    users = list(map(int, users))
    items = _catalogue_ids(train, catalogue)
    known_users = np.sort(train.user_id.unique())
    uindex = {u: i for i, u in enumerate(known_users)}
    iindex = {item: i for i, item in enumerate(items)}
    u = train.user_id.map(uindex).to_numpy(dtype=np.int32)
    i = train.movie_id.map(iindex).to_numpy(dtype=np.int32)
    rating = train.rating.to_numpy(dtype=np.float32)
    mean = float(rating.mean())
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 0.05, (len(known_users), factors)).astype(np.float32)
    q = rng.normal(0, 0.05, (len(items), factors)).astype(np.float32)
    bu = np.zeros(len(known_users), dtype=np.float32)
    bi = np.zeros(len(items), dtype=np.float32)
    for epoch in range(epochs):
        order = rng.permutation(len(rating))
        rate = 0.03 / (1 + epoch * 0.2)
        for batch in np.array_split(order, max(1, int(np.ceil(len(order) / 4096)))):
            ub, ib = u[batch], i[batch]
            pb, qb = p[ub].copy(), q[ib].copy()
            error = rating[batch] - (mean + bu[ub] + bi[ib] + np.sum(pb * qb, axis=1))
            # A batch update averages contributions for repeated users/items.
            du = np.bincount(ub, minlength=len(known_users)).astype(np.float32)
            di = np.bincount(ib, minlength=len(items)).astype(np.float32)
            np.add.at(bu, ub, rate * (error - regularization * bu[ub]) / du[ub])
            np.add.at(bi, ib, rate * (error - regularization * bi[ib]) / di[ib])
            np.add.at(p, ub, rate * (error[:, None] * qb - regularization * pb) / du[ub, None])
            np.add.at(q, ib, rate * (error[:, None] * pb - regularization * qb) / di[ib, None])
    seen = train.groupby("user_id").movie_id.apply(set).to_dict()
    fallback = popular_recommendations(train, users, items, n)
    trained_items = np.bincount(i, minlength=len(items)) > 0
    # Keep unseen catalogue items eligible with the neutral global/user bias.
    q[~trained_items] = 0
    result = {}
    for user in users:
        if user not in uindex:
            result[user] = fallback[user]
        else:
            scores = mean + bu[uindex[user]] + bi + q @ p[uindex[user]]
            result[user] = _recommend_from_scores(scores, items, seen.get(user, set()), n)
    return result
