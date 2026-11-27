"""Reproduce the complete implemented Parts 1, 2 and 3 Track A experiment."""
import argparse
from numbers import Integral, Real
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
from part_3.experiment import run_part_3
from part_3.reporting import save_part_3


def print_results(title, results):
    """Print each model as an aligned block without overly wide table rows."""
    def format_value(value):
        if isinstance(value, Integral):
            return f"{value:,}"
        if isinstance(value, Real):
            return f"{value:.4f}"
        if isinstance(value, (tuple, list)):
            return "[" + ", ".join(format_value(item) for item in value) + "]"
        return str(value)

    labels = {
        "ndcg_ci": "NDCG confidence interval",
        "paired_difference_ci": "Paired difference confidence interval",
    }
    fields = [column for column in results.columns if column != "model"]
    labels.update({column: labels.get(column, column.replace("_", " ").capitalize())
                   for column in fields})
    width = max((len(labels[column]) for column in fields), default=0)
    lines = ["", "=" * 72, title, "=" * 72]
    if results.empty:
        lines.append("No results available.")
    for number, row in enumerate(results.to_dict(orient="records"), start=1):
        lines.append(f"\n{number}. {row.get('model', 'Result')}")
        lines.extend(
            f"   {labels[column]:<{width}} : {format_value(row[column])}"
            for column in fields
        )
    print("\n".join(lines), flush=True)


def run(data_dir, output_dir="outputs"):
    """Load MovieLens once and pass the same split through all three parts."""

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


    trials, models, disaggregated, sparsity, selected = run_part_2(
        split, movies, progress=lambda message: print(message, flush=True))
    save_part_2(trials, models, disaggregated, sparsity, selected, out)
    diversity, activity, protocol = run_part_3(
        split, movies, trials, selected, progress=lambda message: print(message, flush=True))
    save_part_3(diversity, activity, protocol, out)
    print_results("Part 1 - Baseline results", results)
    print_results("Part 2 - Model comparison results", models)
    print_results("Part 3A - Diversity and popularity bias", diversity)
    print(f"\nSaved results, figures and protocol metadata to {out}", flush=True)
    return tuning, results


def main(argv=None):
    """Parse paths and run Parts 1, 2 and 3 Track A by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("ml-1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
