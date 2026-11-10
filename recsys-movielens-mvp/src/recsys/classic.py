import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from .base import Recommender
from .data import tokenize_title


class ContentBasedRecommender(Recommender):
    name = "ContentBased"

    def _fit(self, history):
        items = self.data.items
        genres = items["genres"].map(
            lambda values: " ".join(g.replace("'", "").replace("-", "") for g in values)
        )
        decades = (items["year"] // 10 * 10).astype(str).map(lambda d: f"d{d}")
        titles = items["clean_title"].map(tokenize_title)
        blocks = []
        for documents, weight, options in [
            (genres, 1.0, {}),
            (decades, 0.5, {}),
            (titles, 0.3, {"min_df": 2, "stop_words": "english"}),
        ]:
            features = TfidfVectorizer(
                token_pattern=r"[A-Za-z0-9]+", **options
            ).fit_transform(documents)
            blocks.append(weight * normalize(features))
        self.item_features = normalize(sp.hstack(blocks).tocsr())
        popularity = np.bincount(history["item"], minlength=self.data.n_items)
        self.popularity_prior = np.log1p(popularity.astype(np.float32)) ** 2

    def _set_history(self, history):
        recent = history.groupby("user", sort=False).tail(5)
        self.profiles = normalize(self.data.matrix(recent) @ self.item_features)

    def score(self, users):
        scores = (self.profiles[users] @ self.item_features.T).toarray()
        return scores * self.popularity_prior


class ItemKNNRecommender(Recommender):
    name = "ItemKNN"

    def _fit(self, history):
        matrix = self.data.matrix(history)
        cooccurrence = (matrix.T @ matrix).toarray().astype(np.float32)
        norms = np.sqrt(np.diag(cooccurrence))
        similarity = cooccurrence / (np.outer(norms, norms) + 10.0)
        np.fill_diagonal(similarity, 0)
        k = min(1000, self.data.n_items - 1)
        if k < 1:
            raise ValueError("Для ItemKNN нужны хотя бы два фильма")
        neighbors = np.argpartition(-similarity, k - 1, axis=1)[:, :k]
        rows = np.repeat(np.arange(self.data.n_items), k)
        values = np.take_along_axis(similarity, neighbors, axis=1).ravel()
        self.similarity = sp.csr_matrix(
            (values, (rows, neighbors.ravel())), shape=similarity.shape
        ).T.tocsr()

    def _set_history(self, history):
        recent = history.groupby("user", sort=False).tail(3)
        self.recent = self.data.matrix(recent)

    def score(self, users):
        return (self.recent[users] @ self.similarity).toarray()
