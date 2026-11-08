"""Focused protocol and metric regression tests: python -m unittest -v."""
import unittest
import ast
from importlib.metadata import distribution
import numpy as np
import pandas as pd
from part_1 import (evaluate_at_k, temporal_split, item_knn_recommendations,
                    popular_recommendations, _recommend_from_scores, item_similarity)


def frame(rows):
    return pd.DataFrame(rows, columns=["user_id", "movie_id", "rating", "timestamp"])


def reference_ndcg(truth, scores, k):
    """Execute sklearn's unchanged NumPy core without unrelated native imports.

    Importing sklearn stalls in scipy.spatial on the tested Windows runtime.
    Extract only its two numerical functions from the installed pinned source;
    no replacement implementation or mocked numerical dependency is used.
    Scores in this fixture have no ties.
    """
    source = distribution("scikit-learn").locate_file("sklearn/metrics/_ranking.py")
    module = ast.parse(source.read_text(encoding="utf-8"))
    names = {"_dcg_sample_scores", "_ndcg_sample_scores"}
    functions = [node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(functions) != 2:
        raise RuntimeError("Reference implementation changed; review the pinned dependency.")
    namespace = {"np": np}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["_ndcg_sample_scores"](truth, scores, k=k, ignore_ties=True).mean()


class ProtocolTests(unittest.TestCase):
    def test_metrics_against_reference(self):
        truth = np.array([[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]])
        scores = np.array([list(range(12, 0, -1))])
        test = frame([(1, i + 1, 5, 100) for i in np.flatnonzero(truth[0])])
        actual = evaluate_at_k({1: list(range(1, 11))}, test, range(1, 13))
        self.assertAlmostEqual(actual["ndcg@10"], reference_ndcg(truth, scores, k=10))
        self.assertAlmostEqual(actual["precision@10"], 2 / 10)
        self.assertAlmostEqual(actual["recall@10"], 2 / 3)
        self.assertAlmostEqual(actual["catalogue_coverage"], 10 / 12)

    def test_timestamp_ties(self):
        data = frame([(1, i, 5, i // 3) for i in range(30)])
        split = temporal_split(data, .75, .1)
        self.assertLess(split.train.timestamp.max(), split.validation.timestamp.min())
        self.assertLess(split.validation.timestamp.max(), split.test.timestamp.min())
        self.assertEqual(sum(map(len, [split.train, split.validation, split.test])), len(data))
        with self.assertRaises(ValueError):
            temporal_split(data.assign(timestamp=1))

    def test_positive_profiles_and_seen_exclusion(self):
        data = frame([(1, 1, 5, 1), (1, 2, 1, 2), (2, 1, 5, 3),
                      (2, 4, 5, 4), (3, 2, 5, 5), (3, 3, 5, 6)])
        recs = item_knn_recommendations(data, [1, 99], [1, 2, 3, 4], n=2)
        self.assertEqual(recs[1], [4, 3])
        self.assertEqual(recs[99], popular_recommendations(data, [99], [1, 2, 3, 4], n=2)[99])

    def test_popularity_choices_and_ties(self):
        data = frame([(1, 1, 1, 1), (2, 1, 1, 2), (3, 2, 5, 3)])
        self.assertEqual(popular_recommendations(data, [99], count_positive=False, n=1)[99], [1])
        self.assertEqual(popular_recommendations(data, [99], count_positive=True, n=1)[99], [2])
        self.assertEqual(_recommend_from_scores(np.zeros(5), np.arange(1, 6), {1}, 2), [2, 3])

    def test_binary_similarity(self):
        data = frame([(1, 1, 5, 1), (1, 2, 5, 2), (2, 1, 5, 3), (2, 2, 1, 4)])
        for measure, expected in [("cosine", 1 / np.sqrt(2)), ("jaccard", .5)]:
            _, matrix = item_similarity(data, [1, 2, 3], measure)
            self.assertAlmostEqual(matrix[0, 1], expected)
            np.testing.assert_array_equal(matrix.diagonal(), 0)
            self.assertEqual(matrix.getrow(2).nnz, 0)


if __name__ == "__main__":
    unittest.main()
