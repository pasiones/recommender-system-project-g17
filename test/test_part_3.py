"""Check MMR endpoints, metric definitions and Track A export integration."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from part_3.experiment import genre_vectors, mmr_rerank, gini, diversity_metrics, run_part_3
from part_3.reporting import save_part_3
from part_2.experiment import recommend


class TrackATests(unittest.TestCase):
    def setUp(self):
        self.movies = pd.DataFrame(dict(movie_id=[1, 2, 3, 4],
                                       genres=['Drama', 'Drama', 'Comedy', 'Drama|Comedy']))
        self.fit = pd.DataFrame(dict(user_id=[1], movie_id=[1], rating=[5]))

    def test_mmr_endpoints_and_exclusion(self):
        _, vectors = genre_vectors(self.movies)
        self.assertEqual(mmr_rerank([1, 2, 3, 4], vectors, 1, 2), [1, 2])
        self.assertEqual(mmr_rerank([1, 2, 3, 4], vectors, 0, 2), [1, 3])
        self.assertEqual(mmr_rerank([2, 3], vectors[[1, 2]], .5), [2, 3])
        self.assertEqual(mmr_rerank([], vectors[:0], .5), [])
        with self.assertRaises(ValueError):
            mmr_rerank([1], vectors[:1], 1.1)

    def test_full_catalogue_dispatch_preserves_top_ten_and_seen_exclusion(self):
        movies = pd.DataFrame(dict(movie_id=range(1, 16),
                                   title=[f'Movie {i} (2000)' for i in range(15)],
                                   genres=['Drama'] * 15))
        fit = pd.DataFrame(dict(user_id=[1, 1, 2, 2], movie_id=[1, 2, 2, 3],
                                rating=[5, 2, 5, 4], timestamp=[1, 2, 3, 4]))
        for name in ['Item-based kNN', 'Most popular', 'User-based kNN',
                     'Biased MF', 'Content profile']:
            with self.subTest(model=name):
                full = recommend(name, {}, fit, movies, [1, 99], n=len(movies))
                original = recommend(name, {}, fit, movies, [1, 99])
                self.assertEqual(full[1][:10], original[1])
                self.assertEqual(full[99][:10], original[99])
                self.assertEqual(set(full[1]), set(range(3, 16)))
                self.assertEqual(set(full[99]), set(range(1, 16)))

    def test_hand_calculated_metrics_and_cold_users(self):
        result = diversity_metrics({1: [2, 3], 2: [1, 3]}, self.fit, self.movies)
        self.assertAlmostEqual(result['intra_list_diversity'], 1)
        self.assertAlmostEqual(result['calibration_error'], .5)
        self.assertEqual(result['calibration_users'], 1)
        self.assertAlmostEqual(result['gini'], .375)
        self.assertEqual(gini([1, 1, 1]), 0)
        self.assertEqual(gini([0, 0]), 0)
        self.assertAlmostEqual(gini([0, 0, 2]), 2/3)
        mixed = diversity_metrics({1: [4]}, self.fit, self.movies)
        self.assertAlmostEqual(mixed['calibration_error'], .5)
        self.assertEqual(mixed['intra_list_diversity'], 0)

    def test_sweep_validation_selection_and_template_export(self):
        validation = self.fit.iloc[:0]
        truth = pd.DataFrame(dict(user_id=[1, 2], movie_id=[3, 4], rating=[5, 5]))
        split = SimpleNamespace(train=self.fit, validation=validation, test=truth)
        trials = pd.DataFrame([dict(model=name, **{'ndcg@10': score, 'recall@10': .1,
                                                  'precision@10': .1})
                               for name, score in [('Most popular', .2), ('Content profile', .1)]])
        with patch('part_3.experiment.recommend', return_value={1: [2, 3, 4], 2: [1, 2, 3, 4]}) as recommend:
            results, groups, protocol = run_part_3(split, self.movies, trials,
                                                   {'Most popular': {}}, progress=lambda _: None)
        self.assertEqual(protocol['base_model'], 'Most popular')
        self.assertEqual(recommend.call_args.kwargs['n'], 4)
        self.assertEqual(len(results), 7)
        self.assertEqual(len(groups), 14)
        self.assertEqual(results.iloc[-1].paired_difference_ci, (0, 0))
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            workbook = load_workbook('course_results_template.xlsx')
            workbook.save(out / 'results.xlsx')
            save_part_3(results, groups, protocol, out)
            saved = load_workbook(out / 'results.xlsx')
            self.assertEqual(saved['Part 3A - Diversity'].cell(13, 2).value, 'Unmodified base')
            self.assertEqual(pd.read_csv(out / 'part_3a_diversity.csv').shape, (7, 12))
            self.assertGreater((out / 'part_3a_frontier.png').stat().st_size, 0)
            self.assertEqual(json.loads((out / 'part_3a_protocol.json').read_text())['base_model'],
                             'Most popular')


if __name__ == '__main__':
    unittest.main()
