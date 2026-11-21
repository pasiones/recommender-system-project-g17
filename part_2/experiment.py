"""Validation tuning, held-out comparisons, and Part 2 analyses."""

import numpy as np
import pandas as pd

from part_1.evaluation import evaluate_at_k
from part_1.models import item_knn_recommendations, popular_recommendations
from .models import content_recommendations, mf_recommendations, user_knn_recommendations


SEED = 42


def settings_grid():
    """Four validation configurations per family, including both baselines."""
    return {
        "Item-based kNN": [dict(similarity=s, k_neighbors=k) for s in ("cosine", "jaccard") for k in (50, 100)],
        "Most popular": [dict(count_positive=p, recent_days=d) for p in (False, True) for d in (None, 180)],
        "User-based kNN": [dict(similarity=s, k_neighbors=k) for s in ("cosine", "jaccard") for k in (20, 50)],
        "Biased MF": [dict(factors=f, regularization=r) for f in (16, 32) for r in (0.02, 0.1)],
        "Content profile": [dict(year_weight=y, profile_weighting=w) for y in (0.0, 0.5) for w in ("binary", "rating")],
    }


def recommend(name, settings, train, movies, users):
    catalogue = movies.movie_id
    if name == "Item-based kNN":
        return item_knn_recommendations(train, users, catalogue, **settings)
    if name == "Most popular":
        return popular_recommendations(train, users, catalogue, **settings)
    if name == "User-based kNN":
        return user_knn_recommendations(train, users, catalogue, **settings)
    if name == "Biased MF":
        return mf_recommendations(train, users, catalogue, seed=SEED, **settings)
    if name == "Content profile":
        return content_recommendations(train, movies, users, **settings)
    raise ValueError(name)


def per_user_ndcg(recommendations, truth, k=10):
    relevant = truth.loc[truth.rating >= 4].groupby("user_id").movie_id.apply(set).to_dict()
    values = {}
    discounts = 1 / np.log2(np.arange(2, k + 2))
    for user, items in relevant.items():
        ranked = recommendations.get(int(user), [])[:k]
        dcg = sum(discounts[j] for j, item in enumerate(ranked) if item in items)
        values[int(user)] = float(dcg / discounts[:min(k, len(items))].sum())
    return pd.Series(values, dtype=float)


def paired_bootstrap(series, reference, seed=SEED, samples=1000):
    """95% percentile CI of macro NDCG and its paired user-level difference."""
    common = series.index.intersection(reference.index)
    rng = np.random.default_rng(seed)
    positions = rng.integers(0, len(common), (samples, len(common)))
    values = series.loc[common].to_numpy()
    differences = values - reference.loc[common].to_numpy()
    return tuple(map(float, np.quantile(values[positions].mean(axis=1), [0.025, 0.975]))), \
        tuple(map(float, np.quantile(differences[positions].mean(axis=1), [0.025, 0.975])))


def subsample(train, fraction, seed=SEED):
    """Nested uniform row samples, independent of validation and test data."""
    if fraction == 1:
        return train
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train))
    return train.iloc[order[:int(len(train) * fraction)]].copy()


def run_part_2(split, movies, progress=print):
    """Tune on validation, refit on train+validation, and analyze test results."""
    catalogue = movies.movie_id
    validation_users = split.validation.user_id.unique()
    test_users = split.test.user_id.unique()
    grid = settings_grid()
    trials = []
    selected = {}
    for name, options in grid.items():
        for settings in options:
            progress(f"Validation: {name} {settings}")
            recs = recommend(name, settings, split.train, movies, validation_users)
            metrics = evaluate_at_k(recs, split.validation, catalogue)
            trials.append({"model": name, "settings": str(settings), **metrics})
        family = [row for row in trials if row["model"] == name]
        best = max(range(len(options)), key=lambda j: (
            family[j]["ndcg@10"], family[j]["recall@10"], family[j]["precision@10"]))
        selected[name] = options[best]
    fit = pd.concat([split.train, split.validation], ignore_index=True)
    predictions = {}
    overall = []
    for name, settings in selected.items():
        progress(f"Test: {name} {settings}")
        recs = recommend(name, settings, fit, movies, test_users)
        predictions[name] = recs
        overall.append({"model": name, "settings": str(settings),
                        **evaluate_at_k(recs, split.test, catalogue)})
    reference = per_user_ndcg(predictions["Item-based kNN"], split.test)
    for row in overall:
        series = per_user_ndcg(predictions[row["model"]], split.test)
        row["ndcg_ci"], row["paired_difference_ci"] = paired_bootstrap(series, reference)
    # Two disaggregations: training-history size and early/late test period.
    fit_counts = fit.groupby("user_id").size()
    eligible = split.test.loc[split.test.rating >= 4, "user_id"].unique()
    cutoff = float(fit_counts.reindex(eligible, fill_value=0).median())
    halves = np.array_split(np.sort(split.test.timestamp.unique()), 2)
    period_cut = int(halves[0][-1])
    groups = [
        ("Training user activity", "Low", f"fit interactions <= {cutoff:g}",
         split.test[split.test.user_id.isin(fit_counts[fit_counts <= cutoff].index.union(
             pd.Index(eligible).difference(fit_counts.index)))]),
        ("Training user activity", "High", f"fit interactions > {cutoff:g}",
         split.test[split.test.user_id.isin(fit_counts[fit_counts > cutoff].index)]),
        ("Test period", "Early", f"timestamp <= {period_cut}",
         split.test[split.test.timestamp <= period_cut]),
        ("Test period", "Late", f"timestamp > {period_cut}",
         split.test[split.test.timestamp > period_cut]),
    ]
    disaggregated = []
    for name, recs in predictions.items():
        for dimension, group, definition, truth in groups:
            if truth.loc[truth.rating >= 4].empty:
                continue
            metrics = evaluate_at_k(recs, truth, catalogue)
            disaggregated.append({"model": name, "dimension": dimension,
                                  "group": group, "definition": definition, **metrics})
    sparsity = []
    for fraction in (1.0, 0.5, 0.2):
        sparse_fit = subsample(fit, fraction)
        for name, settings in selected.items():
            progress(f"Sparsity {fraction:.0%}: {name}")
            metrics = next(row for row in overall if row["model"] == name) if fraction == 1 else \
                evaluate_at_k(recommend(name, settings, sparse_fit, movies, test_users), split.test, catalogue)
            sparsity.append({"model": name, "training_fraction": fraction,
                             "fit_rows": len(sparse_fit), **{key: metrics[key] for key in
                             ("precision@10", "recall@10", "ndcg@10", "catalogue_coverage", "evaluated_users")}})
    return (pd.DataFrame(trials), pd.DataFrame(overall),
            pd.DataFrame(disaggregated), pd.DataFrame(sparsity), selected)
