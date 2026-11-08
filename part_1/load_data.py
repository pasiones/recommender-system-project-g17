"""Read MovieLens users, movies and ratings."""

from pathlib import Path
from typing import Tuple
import pandas as pd


def load_users(path: Path) -> pd.DataFrame:
    """Load users.dat and preserve ZIP codes as strings."""
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        dtype={
            "user_id": "int64",
            "gender": "string",
            "age": "int64",
            "occupation": "int64",
            "zip_code": "string",
        },
    )


def load_movies(path: Path) -> pd.DataFrame:
    """Load movies.dat with the title and genre columns as strings."""
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
        dtype={
            "movie_id": "int64",
            "title": "string",
            "genres": "string",
        },
    )


def load_ratings(path: Path) -> pd.DataFrame:
    """Load ratings.dat with numeric identifiers, ratings, and timestamps."""
    return pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={
            "user_id": "int64",
            "movie_id": "int64",
            "rating": "float64",
            "timestamp": "int64",
        },
    )


def load_data(dir: str | Path = "ml-1m") -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """Load users, movies, and ratings from one archive directory."""
    data_path = Path(dir)
    return (
        load_users(data_path / "users.dat"),
        load_movies(data_path / "movies.dat"),
        load_ratings(data_path / "ratings.dat"),
    )
