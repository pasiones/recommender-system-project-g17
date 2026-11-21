"""Reproduce the complete implemented Parts 1 and 2 experiment."""
import argparse
from pathlib import Path

from part_1 import (
    characterize_dataset,
    load_data,
    run_baseline_experiment,
    temporal_split,
)
from part_1.reporting import save_figures, save_outputs
import matplotlib
from part_2.experiment import run_part_2
from part_2.reporting import save_part_2

def run(data_dir, output_dir="outputs"):
    """Load MovieLens once and pass the same split through both parts."""

    matplotlib.use("Agg")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Loading MovieLens from {data_dir}...", flush=True)
    users, movies, ratings = load_data(data_dir)
    print(f"Loaded {len(ratings):,} ratings. Characterizing dataset...", flush=True)
    summary = characterize_dataset(users, movies, ratings)
    split = temporal_split(ratings)
    print(
        f"Split: {len(split.train):,} train / "
        f"{len(split.validation):,} validation / {len(split.test):,} test.",
        flush=True,
    )
    save_figures(ratings, summary, out)
    print("Tuning six kNN configurations and evaluating baselines...", flush=True)
    tuning, results = run_baseline_experiment(split, movies["movie_id"])
    print("Writing results and protocol metadata...", flush=True)
    save_outputs(ratings, summary, split, tuning, results, out)


    trials, models, disaggregated, sparsity, selected = run_part_2(split, movies)
    save_part_2(trials, models, disaggregated, sparsity, selected, out)
    print(models.to_string(index=False), flush=True)
    print(results.to_string(index=False), flush=True)
    print(f"Saved results, figures and protocol metadata to {out}", flush=True)
    return tuning, results


def main(argv=None):
    """Parse paths and run Parts 1 and 2 by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("ml-1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
