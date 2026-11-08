# Recommender Systems — Part 1

Tested with Python **3.15.2**. Download and extract [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/) and follow its usage terms. The data directory must contain `users.dat`, `movies.dat`, and `ratings.dat`.

Create and activate a Python environment, then install pinned dependencies:

```sh
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

Reproduce all Part 1 tables, figures and protocol metadata with one command:

```sh
python run_all.py --data-dir ml-1m --output-dir outputs
```

For a dataset elsewhere, use `python run_all.py --data-dir /path/to/ml-1m`.
Outputs default to `outputs`. The runner reads each of the three data files once,
characterizes the data, creates one temporal split, and reuses it for tuning,
test evaluation, and protocol reporting. No notebook execution is needed.

The `part_1/` package contains the reusable analysis and recommendation methods:

- `load_data.py`: MovieLens file readers.
- `process_data.py`: dataset characterization, temporal splitting, and positive feedback.
- `models.py`: popularity and item-kNN recommenders.
- `evaluation.py`: ranking metrics and catalogue coverage.
- `experiment.py`: validation tuning and test evaluation.
- `plots.py`: interaction long-tail plots.
- `reporting.py`: figures, tables, and protocol metadata exports.
- `__init__.py`: exports the existing public API, so `from part_1 import ...` still works.

`run_all.py` orchestrates the workflow and prints progress;
`part_1/reporting.py` writes figures, tables, and protocol metadata.
Existing notebook calls to `run_baseline_experiment(ratings, catalogue)` still work;
callers with a prepared split can pass it in place of `ratings` to avoid splitting again.

The data and output paths are configurable. No intermediate files are required. Seed 42 is fixed; all ranking and similarity ties are deterministic. Run focused regression tests, including comparison of metrics against scikit-learn:

```sh
python -m unittest -v
```

The NDCG test executes the unchanged NumPy numerical core from the installed scikit-learn source. This avoids a native `scipy.spatial` import stall observed in this Windows environment. Precision, Recall and coverage are checked against hand-calculated values.

## Protocol

Keep all ratings, without low-activity filtering. Ratings >=4 define relevance and positive kNN profiles; all rated items are excluded from recommendations. Split globally at 80% and 90% of timestamp-ordered rows, moving cuts right to keep tied timestamps together. Tune cosine/Jaccard and k in {20,50,100} on validation NDCG@10 (Recall, then Precision break ties), then refit on train plus validation. kNN sums source-item top-k similarities over positive history. Users without positive history receive positive-popularity fallback. Other zero-score ties use ascending movie ID.

Compare four popularity variants: all-time/recent 180-day window crossed with all-rating/positive counts. Rank the entire metadata catalogue, excluding seen items, without negative sampling. This assumes the full catalogue is available, including training-unseen items. Macro-average metrics over held-out users with positive feedback, including cold users. Precision divides by 10 even for short lists; coverage divides by all 3,883 catalogue items. Reports document exact cutoffs, excluded users and cold-start counts. The monthly plot describes rating-entry times, not viewing dates.

## Outputs and submission

- `outputs/results.xlsx` fills the original course workbook, preserving its sheets and headers. `results.csv` uses exactly the Part 1 sheet's columns.
- `outputs/validation_grid.csv` records all six validation trials.
- `outputs/dataset_summary.json`, `protocol.json`, and `monthly_activity.csv` record dataset and protocol facts.
- `outputs/long_tail.png` and `ratings_and_time.png` show the required distributions.

Prepare the Word report separately using the official template, within the Part 1 four-page limit. Include team information, contributions, repository URL, and AI disclosure. The runner does not generate a Word report.

The official templates can be downloaded from the PDF's links (their labels in the PDF are reversed): [report template](https://docs.google.com/document/d/19nhPiv6es82L3irhuSqo54b10AaXXAoY/edit), [results workbook](https://drive.google.com/file/d/19jE2N5VpbcS0Xwxfw-jIGTTqfU1yqgs7/view). No external service is needed when running the experiment. GitHub publication and personal contribution statements must be completed by the team.
