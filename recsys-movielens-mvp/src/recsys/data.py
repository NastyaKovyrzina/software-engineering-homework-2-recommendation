"""Загрузка MovieLens-1M"""

from __future__ import annotations

import io
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"

AGES = (1, 18, 25, 35, 45, 50, 56)

def download_movielens(data_dir: Path = DATA_DIR) -> Path:
    """Скачивает и распаковывает ml-1m.zip"""
    target = data_dir / "ml-1m"
    if all((target / name).exists() for name in ("ratings.dat", "users.dat", "movies.dat")):
        return target
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Скачиваю {ML1M_URL} ...")
    with urllib.request.urlopen(ML1M_URL, timeout=60) as resp:
        payload = resp.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        for name in ("ml-1m/ratings.dat", "ml-1m/users.dat", "ml-1m/movies.dat", "ml-1m/README"):
            out = data_dir / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(zf.read(name))
    return target


def _read_dat(path: Path, names: list[str]) -> pd.DataFrame:
    return pd.read_csv(path, sep="::", engine="python", names=names, encoding="latin-1")


@dataclass
class MovieLens:

    ratings: pd.DataFrame 
    users: pd.DataFrame  
    items: pd.DataFrame 
    genres: list[str]
    train: pd.DataFrame = field(repr=False, default=None)
    val: pd.DataFrame = field(repr=False, default=None)
    test: pd.DataFrame = field(repr=False, default=None)

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_items(self) -> int:
        return len(self.items)

    
    def split(self, n_val: int = 5, n_test: int = 10) -> "MovieLens":
        df = self.ratings.sort_values(["user", "order"]).copy()
        pos_from_end = df.groupby("user").cumcount(ascending=False)
        self.test = df[pos_from_end < n_test].reset_index(drop=True)
        self.val = df[(pos_from_end >= n_test) & (pos_from_end < n_test + n_val)].reset_index(drop=True)
        self.train = df[pos_from_end >= n_test + n_val].reset_index(drop=True)
        return self

    def stage(self, stage: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        if stage == "val":
            return self.train, self.val
        if stage == "test":
            hist = pd.concat([self.train, self.val]).sort_values(["user", "order"])
            return hist.reset_index(drop=True), self.test
        raise ValueError(stage)

    def matrix(self, df: pd.DataFrame) -> sp.csr_matrix:
        """Разреженная матрица"""
        data = np.ones(len(df), dtype=np.float32)
        return sp.csr_matrix(
            (data, (df["user"].to_numpy(), df["item"].to_numpy())), shape=(self.n_users, self.n_items)
        )

    def genre_matrix(self) -> np.ndarray:
        """Multi-hot матрица item × genre."""
        m = np.zeros((self.n_items, len(self.genres)), dtype=np.float32)
        gidx = {g: i for i, g in enumerate(self.genres)}
        for item, gs in zip(self.items.index, self.items["genres"]):
            for g in gs:
                m[item, gidx[g]] = 1.0
        return m


def load_movielens(data_dir: Path = DATA_DIR, n_val: int = 5, n_test: int = 10, seed: int = 42) -> MovieLens:
    root = download_movielens(data_dir)
    ratings = _read_dat(root / "ratings.dat", ["user_id", "movie_id", "rating", "ts"])
    users = _read_dat(root / "users.dat", ["user_id", "gender", "age", "occupation", "zip"])
    movies = _read_dat(root / "movies.dat", ["movie_id", "title", "genres"])

    # Оставляем только фильмы, у которых есть хотя бы одна оценка
    movies = movies[movies["movie_id"].isin(ratings["movie_id"].unique())].reset_index(drop=True)
    users = users.sort_values("user_id").reset_index(drop=True)

    user_map = pd.Series(np.arange(len(users)), index=users["user_id"])
    item_map = pd.Series(np.arange(len(movies)), index=movies["movie_id"])
    ratings = pd.DataFrame({
        "user": user_map.loc[ratings["user_id"]].to_numpy(),
        "item": item_map.loc[ratings["movie_id"]].to_numpy(),
        "rating": ratings["rating"].to_numpy(),
        "ts": ratings["ts"].to_numpy(),
    })


    tiebreak = np.random.default_rng(seed).random(len(ratings))
    ratings["order"] = np.lexsort((tiebreak, ratings["ts"].to_numpy(), ratings["user"].to_numpy())).argsort()

    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype(int)
    movies["clean_title"] = movies["title"].str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    movies["genres"] = movies["genres"].str.split("|")
    genres = sorted({g for gs in movies["genres"] for g in gs})

    users["age_idx"] = users["age"].map({a: i for i, a in enumerate(AGES)})
    users["gender_idx"] = (users["gender"] == "F").astype(int)
    users.index.name = "user"
    movies.index.name = "item"

    return MovieLens(ratings=ratings, users=users, items=movies, genres=genres).split(n_val, n_test)


def tokenize_title(title: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", title.lower()))
