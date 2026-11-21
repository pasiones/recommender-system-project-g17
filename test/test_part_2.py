"""Focused checks for the Part 2 models and analysis pipeline."""

import unittest

import pandas as pd

from part_1 import temporal_split
from part_2.experiment import paired_bootstrap, run_part_2
from part_2.models import content_recommendations, mf_recommendations, user_knn_recommendations


class PartTwoTests(unittest.TestCase):
    def test_content_profile_uses_genres_and_excludes_seen(self):
        movies = pd.DataFrame({"movie_id": [1, 2, 3, 4],
                               "title": ["A (2000)", "B (2000)", "C (2000)", "D (2000)"],
                               "genres": ["Drama", "Drama", "Comedy", "Comedy"]})
        train = pd.DataFrame([(1, 1, 5, 1), (1, 3, 1, 2), (2, 4, 5, 3)],
                             columns=["user_id", "movie_id", "rating", "timestamp"])
        result = content_recommendations(train, movies, [1], n=2, year_weight=0)
        self.assertEqual(result[1][0], 2)
        self.assertNotIn(1, result[1])
        self.assertNotIn(3, result[1])

    def test_user_knn_and_mf_are_deterministic_and_exclude_seen(self):
        train = pd.DataFrame([(1, 1, 5, 1), (2, 1, 5, 2), (2, 2, 5, 3),
                              (3, 3, 4, 4)], columns=["user_id", "movie_id", "rating", "timestamp"])
        self.assertEqual(user_knn_recommendations(train, [1], [1, 2, 3], n=1, k_neighbors=1)[1], [2])
        one = mf_recommendations(train, [1], [1, 2, 3, 4], n=3, factors=2, epochs=2)
        two = mf_recommendations(train, [1], [1, 2, 3, 4], n=3, factors=2, epochs=2)
        self.assertEqual(one, two)
        self.assertNotIn(1, one[1])
        self.assertEqual(len(one[1]), 3)

    def test_bootstrap_and_small_experiment(self):
        scores = pd.Series({1: 0.2, 2: 0.8})
        ci, diff = paired_bootstrap(scores, scores)
        self.assertEqual(diff, (0.0, 0.0))
        self.assertLessEqual(ci[0], 0.5)
        movies = pd.DataFrame({"movie_id": list(range(1, 7)),
                               "title": [f"M{i} (2000)" for i in range(1, 7)],
                               "genres": ["Drama", "Comedy"] * 3})
        ratings = pd.DataFrame([(i % 3 + 1, i % 6 + 1, 5, i + 1000) for i in range(60)],
                               columns=["user_id", "movie_id", "rating", "timestamp"])
        split = temporal_split(ratings)
        trials, models, disagg, sparsity, selected = run_part_2(split, movies, progress=lambda _: None)
        self.assertEqual(len(trials), 20)
        self.assertEqual(len(models), 5)
        self.assertGreaterEqual(len(disagg), 15)
        self.assertEqual(set(disagg.dimension), {"Training user activity", "Test period"})
        self.assertEqual(len(sparsity), 15)
        self.assertEqual(len(selected), 5)


if __name__ == "__main__":
    unittest.main()
