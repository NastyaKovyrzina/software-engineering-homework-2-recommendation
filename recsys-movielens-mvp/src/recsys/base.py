from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def top_k(scores: np.ndarray, seen: sp.csr_matrix, k: int) -> np.ndarray:
    s = np.array(scores, dtype=np.float32, copy=True)
    if not 1 <= k <= s.shape[1]:
        raise ValueError("k должно быть от 1 до размера каталога")
    rows, cols = seen.nonzero()
    s[rows, cols] = -np.inf
    if np.any(np.isfinite(s).sum(axis=1) < k):
        raise ValueError("Недостаточно неизвестных пользователю фильмов для top-k")
    candidates = np.argpartition(-s, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, candidates, axis=1), axis=1, kind="stable")
    return np.take_along_axis(candidates, order, axis=1)


class Recommender:
    name = "base"

    def fit(self, data, history: pd.DataFrame):
        self.data = data
        self._fit(history)
        return self.set_history(history)

    def _fit(self, history: pd.DataFrame):
        pass

    def set_history(self, history: pd.DataFrame):
        self.seen = self.data.matrix(history)
        self._set_history(history)
        return self

    def _set_history(self, history: pd.DataFrame):
        pass

    def score(self, users: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, users, k: int = 10, batch_size: int = 512) -> np.ndarray:
        users = np.asarray(users, dtype=np.int64)
        if not len(users):
            return np.empty((0, k), dtype=np.int64)
        recommendations = []
        for start in range(0, len(users), batch_size):
            batch = users[start:start + batch_size]
            recommendations.append(top_k(self.score(batch), self.seen[batch], k))
        return np.vstack(recommendations)
