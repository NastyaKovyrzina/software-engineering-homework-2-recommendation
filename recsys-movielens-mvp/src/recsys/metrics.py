"""Основные метрики"""
from __future__ import annotations

import time

import numpy as np


def ranking_metrics(recs, users, target, n_items: int) -> dict[str, float]:
    k = recs.shape[1]
    relevance = np.zeros(recs.shape, dtype=bool)
    counts = np.diff(target[users].indptr)
    if not len(users) or np.any(counts == 0):
        raise ValueError("Для оценки нужны пользователи с тестовыми взаимодействиями")
    for row, user in enumerate(users):
        relevance[row] = np.isin(recs[row], target[user].indices)
    hits = relevance.sum(axis=1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    ideal = np.array([discounts[:min(count, k)].sum() for count in counts])
    return {
        f"ndcg@{k}": float(((relevance * discounts).sum(axis=1) / ideal).mean()),
        f"recall@{k}": float((hits / counts).mean()),
        f"hitrate@{k}": float((hits > 0).mean()),
        f"coverage@{k}": len(np.unique(recs)) / n_items,
    }


def evaluate(model, data, stage: str = "test", k: int = 10, fit: bool = True):
    history, target = data.stage(stage)
    started = time.perf_counter()
    if fit:
        model.fit(data, history)
    fit_time = time.perf_counter() - started
    users = np.unique(target["user"].to_numpy())
    started = time.perf_counter()
    recs = model.recommend(users, k)
    elapsed = time.perf_counter() - started
    metrics = ranking_metrics(recs, users, data.matrix(target), data.n_items)
    metrics["fit_time_s"] = fit_time
    # Среднее время пакетной выдачи
    metrics["batch_ms_per_user"] = 1000 * elapsed / len(users)
    return metrics
