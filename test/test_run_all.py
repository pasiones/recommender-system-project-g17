"""Exercise the complete runner with a small MovieLens-format dataset."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import pandas as pd

import part_1
from part_1 import experiment
import run_all


class RunnerTests(unittest.TestCase):
    def test_load_once_and_reuse_split_through_exports(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            (data / "users.dat").write_text("1::M::25::1::00100\n2::F::25::2::00200\n")
            (data / "movies.dat").write_text(
                "".join(f"{i}::Movie {i}::Drama\n" for i in range(1, 7))
            )
            (data / "ratings.dat").write_text(
                "".join(f"{i % 2 + 1}::{i % 6 + 1}::5::{1000 + i}\n" for i in range(30))
            )
            out = root / "results"
            with (
                patch.object(run_all, "load_data", wraps=part_1.load_data) as load,
                patch.object(run_all, "temporal_split", wraps=part_1.temporal_split) as split,
                patch.object(experiment, "temporal_split", side_effect=AssertionError("Repeated split")),
                patch.object(pd, "read_csv", wraps=pd.read_csv) as read,
                redirect_stdout(StringIO()),
            ):
                run_all.main(["--data-dir", str(data), "--output-dir", str(out)])
            load.assert_called_once_with(data)
            split.assert_called_once()
            self.assertEqual(read.call_count, 3)
            self.assertEqual(len(pd.read_csv(out / "validation_grid.csv")), 6)
            self.assertEqual(len(pd.read_csv(out / "results.csv")), 5)
            for name in (
                "results.xlsx", "dataset_summary.json", "protocol.json",
                "monthly_activity.csv", "long_tail.png", "ratings_and_time.png",
            ):
                self.assertGreater((out / name).stat().st_size, 0)
            self.assertFalse((out / "report.docx").exists())


if __name__ == "__main__":
    unittest.main()
