# Recommender Systems Project

## Requirements

- Python 3.12.3 (tested)
- [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/), downloaded and extracted according to its usage terms

Place the extracted data in `ml-1m/` at the project root. It must contain `users.dat`, `movies.dat`, and `ratings.dat`.

## Install

Create a virtual environment:

```sh
python -m venv .venv
```

Activate it with `.venv/Scripts/Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on Linux and macOS. Then install the pinned packages in [requirements.txt](./requirements.txt):

```sh
python -m pip install -r requirements.txt
```

## Run

From the project root, run [run_all.py](./run_all.py):

```sh
python run_all.py
```

This reproduces the implemented Parts 1, 2 and 3 Track A results, figures, workbook in `outputs/`. The default data directory is `ml-1m/`. To use other locations:

```sh
python run_all.py --data-dir /path/to/ml-1m --output-dir /path/to/outputs
```

To run the tests:

```sh
python -m unittest -v
```

## Submission

All required submission documents are in the [deliverables](./deliverables/) folder.
