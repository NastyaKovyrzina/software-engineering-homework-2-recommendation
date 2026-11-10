import numpy as np

from .base import Recommender


class PopularRecommender(Recommender):
    name = "Popular"

    def _fit(self, history):
        self.popularity = np.bincount(
            history["item"], minlength=self.data.n_items
        ).astype(np.float32)

    def score(self, users):
        return np.tile(self.popularity, (len(users), 1))


class AlsoWatchedRecommender(Recommender):
    name = "AlsoWatched"

    def _fit(self, history):
        matrix = self.data.matrix(history)
        self.cooccurrence = (matrix.T @ matrix).toarray().astype(np.float32)
        np.fill_diagonal(self.cooccurrence, 0)

    def _set_history(self, history):
        recent = history.groupby("user", sort=False).tail(1)
        self.recent = self.data.matrix(recent)

    def score(self, users):
        return np.asarray(self.recent[users] @ self.cooccurrence)
