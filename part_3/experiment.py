"""Deterministic full-catalogue MMR and beyond-accuracy evaluation."""
import numpy as np
import pandas as pd

from part_1.evaluation import evaluate_at_k
from part_2.experiment import recommend, per_user_ndcg, paired_bootstrap

LAMBDAS = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def genre_vectors(movies):
    """Return fractional genre memberships and cosine-normalized vectors."""
    if movies.movie_id.duplicated().any():
        raise ValueError("Movie IDs must be unique")
    labels = [set(str(value).split('|')) if pd.notna(value) else set()
              for value in movies.genres]
    if any(not row or '' in row for row in labels):
        raise ValueError("Every movie must have genre metadata")
    genres = sorted(set.union(*labels))
    matrix = np.array([[g in row for g in genres] for row in labels], dtype=float)
    return (matrix / matrix.sum(axis=1, keepdims=True),
            matrix / np.linalg.norm(matrix, axis=1, keepdims=True))


def mmr_rerank(items, vectors, weight, k=10):
    """Maximize lambda * rank relevance - (1-lambda) * max cosine similarity.

    Relevance is 1/log2(base rank+1), with one-based ranks. Ties retain base
    ordering. At lambda=0, the first item is the base model's first item.
    """
    if not 0 <= weight <= 1 or k < 1:
        raise ValueError("lambda must be in [0, 1] and k must be positive")
    if len(items) != len(vectors) or len(set(items)) != len(items):
        raise ValueError("Candidates must be unique and aligned with vectors")
    if weight == 1 or len(items) == 0:
        return list(items[:k])
    relevance = 1 / np.log2(np.arange(len(items)) + 2)
    redundancy = np.zeros(len(items))
    available = np.ones(len(items), dtype=bool)
    chosen = []
    for _ in range(min(k, len(items))):
        scores = weight * relevance - (1 - weight) * redundancy
        scores[~available] = -np.inf
        position = int(np.argmax(scores))
        chosen.append(items[position])
        available[position] = False
        redundancy = np.maximum(redundancy, vectors @ vectors[position])
    return chosen


def gini(counts):
    """Uncorrected Gini over the entire catalogue, including zero exposures."""
    values = np.sort(np.asarray(counts, dtype=float))
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    return float(2 * np.dot(np.arange(1, len(values) + 1), values)
                 / (len(values) * values.sum()) - (len(values) + 1) / len(values))


def diversity_metrics(recommendations, fit, movies):
    """Macro ILD and TV calibration; unweighted top-10 item exposure Gini.

    Each movie contributes equal total genre mass. Calibration compares liked
    fit-history and recommendation distributions using total variation (TV).
    Users without liked history are excluded from calibration only.
    """
    fractional, vectors = genre_vectors(movies)
    index = {int(item): i for i, item in enumerate(movies.movie_id)}
    histories = fit[fit.rating >= 4].groupby('user_id').movie_id.apply(list).to_dict()
    counts = np.zeros(len(movies))
    ild, calibration = [], []
    for user, items in recommendations.items():
        positions = [index[item] for item in items[:10]]
        np.add.at(counts, positions, 1)
        if len(positions) < 2:
            ild.append(0.0)
        else:
            similarity = vectors[positions] @ vectors[positions].T
            ild.append(float(np.clip(1 - similarity[np.triu_indices(len(positions), 1)], 0, 1).mean()))
        history = histories.get(user, [])
        if history and positions:
            target = fractional[[index[item] for item in history]].mean(axis=0)
            observed = fractional[positions].mean(axis=0)
            calibration.append(float(np.abs(target - observed).sum() / 2))
    return dict(intra_list_diversity=float(np.mean(ild)), gini=gini(counts),
                calibration_error=float(np.mean(calibration)) if calibration else float('nan'),
                calibration_users=len(calibration))


def run_part_3(split, movies, trials, selected, progress=print):
    """Select the Part 2 validation winner, refit, and sweep a fixed grid."""
    best = trials.sort_values(['ndcg@10', 'recall@10', 'precision@10'],
                             ascending=False, kind='stable').iloc[0]
    name = best['model']
    settings = selected[name]
    fit = pd.concat([split.train, split.validation], ignore_index=True)
    users = sorted(split.test.loc[split.test.rating >= 4, 'user_id'].unique())
    progress(f'Part 3A: full-catalogue rankings from {name}')
    candidates = recommend(name, settings, fit, movies, users, n=len(movies))
    _, vectors = genre_vectors(movies)
    index = {int(item): i for i, item in enumerate(movies.movie_id)}
    base = {user: items[:10] for user, items in candidates.items()}
    reference = per_user_ndcg(base, split.test)
    rows, groups = [], []
    activity = fit.groupby('user_id').size().reindex(users, fill_value=0)
    median = activity.median()
    for weight in LAMBDAS:
        progress(f'Part 3A: MMR lambda={weight:g}')
        recs = {user: mmr_rerank(items, vectors[[index[item] for item in items]], weight)
                for user, items in candidates.items()}
        ndcg_ci, delta_ci = paired_bootstrap(per_user_ndcg(recs, split.test), reference)
        rows.append(dict(model=name, trade_off=weight,
                         **evaluate_at_k(recs, split.test, movies.movie_id),
                         **diversity_metrics(recs, fit, movies),
                         ndcg_ci=ndcg_ci, paired_difference_ci=delta_ci))
        for group, mask in [('Low', activity <= median), ('High', activity > median)]:
            subset = {int(u): recs[int(u)] for u in activity[mask].index}
            if subset:
                groups.append(dict(trade_off=weight, activity=group,
                                   **evaluate_at_k(subset, split.test, movies.movie_id)))
    protocol = dict(base_model=name, base_settings=settings, seed=42,
                    selection='Part 2 validation NDCG, Recall, Precision; stable family order for ties',
                    lambdas=list(LAMBDAS), candidates='All unseen catalogue items; all fit ratings excluded',
                    relevance='1/log2(base rank+1), one-based rank; rank-based MMR',
                    objective='lambda*relevance - (1-lambda)*max cosine similarity to selected items',
                    distance='Cosine distance on L2-normalized multi-hot genre vectors',
                    calibration='Total variation of fractional genre distributions; positive fit history (rating>=4); no smoothing',
                    calibration_missing='Exclude users without positive fit history or recommendations; report count',
                    population='Users with positive test feedback, identical for all accuracy/diversity metrics',
                    gini='Unweighted recommendation counts over full catalogue including zero exposures; uncorrected Gini',
                    activity_cutoff=float(median),
                    operating_point='No test-selected optimum; report the entire predeclared sweep',
                    uncertainty='1000 paired user bootstrap samples, seed 42, 95% percentile intervals vs lambda=1')
    return pd.DataFrame(rows), pd.DataFrame(groups), protocol
