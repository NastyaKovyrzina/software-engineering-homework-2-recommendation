from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base import Recommender


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TwoTowerNet(nn.Module):
    def __init__(self, item_genres, item_year_bucket, dim=128, short_len=5):
        super().__init__()
        n_items, n_genres = item_genres.shape
        self.pad = n_items - 1
        self.short_len = short_len
        self.item_id = nn.Embedding(n_items, dim, padding_idx=self.pad)
        self.genre = nn.Linear(n_genres, dim, bias=False)
        self.year = nn.Embedding(int(item_year_bucket.max()) + 1, dim)
        self.item_mlp = nn.Sequential(
            nn.Linear(3 * dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim)
        )
        self.register_buffer("item_genres", torch.tensor(item_genres, dtype=torch.float32))
        self.register_buffer("item_year", torch.tensor(item_year_bucket, dtype=torch.long))
        self.hist_emb = nn.Embedding(n_items, dim, padding_idx=self.pad)
        self.gender = nn.Embedding(2, 16)
        self.age = nn.Embedding(7, 16)
        self.occupation = nn.Embedding(21, 16)
        self.user_mlp = nn.Sequential(
            nn.Linear(2 * dim + 48, 2 * dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(2 * dim, dim)
        )

    def item_vectors(self, items):
        features = torch.cat([
            self.item_id(items),
            self.genre(self.item_genres[items]),
            self.year(self.item_year[items]),
        ], dim=-1)
        return F.normalize(self.item_mlp(features), dim=-1)

    def user_vectors(self, history, gender, age, occupation):
        embeddings = self.hist_emb(history)
        mask = (history != self.pad).unsqueeze(-1).float()
        long_term = (embeddings * mask).sum(1) / mask.sum(1).clamp(min=1)
        short_embeddings = embeddings[:, -self.short_len:]
        short_mask = mask[:, -self.short_len:]
        short_term = (short_embeddings * short_mask).sum(1) / short_mask.sum(1).clamp(min=1)
        features = torch.cat([
            long_term, short_term, self.gender(gender), self.age(age),
            self.occupation(occupation)
        ], dim=-1)
        return F.normalize(self.user_mlp(features), dim=-1)


class TwoTowerRecommender(Recommender):
    name = "TwoTower"

    def __init__(self, epochs=18, seed=42, device=None):
        self.epochs, self.seed = epochs, seed
        self.hist_len, self.batch_size = 50, 1024
        self.device = torch.device(device) if device else pick_device()

    def _user_tensors(self, users):
        return tuple(
            torch.as_tensor(self.data.users[column].to_numpy()[users], device=self.device)
            for column in ("gender_idx", "age_idx", "occupation")
        )

    def _sequences(self, history):
        history = history.sort_values(["user", "order"])
        items = history["item"].to_numpy()
        users = history["user"].to_numpy()
        counts = np.bincount(users, minlength=self.data.n_users)
        return items, users, np.concatenate([[0], counts.cumsum()])

    def _gather_history(self, items, end, start):
        indices = end[:, None] - self.hist_len + np.arange(self.hist_len)[None, :]
        padded = np.append(items, self.net.pad)
        return padded[np.where(indices >= start[:, None], indices, len(items))]

    def _build_net(self):
        years = self.data.items["year"].to_numpy()
        buckets = np.concatenate([(years - years.min()) // 5, [0]])
        genres = self.data.genre_matrix()
        genres = np.vstack([genres, np.zeros((1, genres.shape[1]), np.float32)])
        self.net = TwoTowerNet(genres, buckets).to(self.device)

    @classmethod
    def from_pretrained(cls, path, data, history):
        model = cls()
        model.data = data
        model._build_net()
        state = torch.load(path, map_location=model.device, weights_only=True)
        model.net.load_state_dict(state)
        return model.set_history(history)

    def _fit(self, history):
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        self._build_net()
        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=0.001, weight_decay=1e-6
        )
        items, users, offsets = self._sequences(history)
        counts = np.bincount(items, minlength=self.data.n_items).astype(np.float64)
        with torch.no_grad():
            unknown = torch.as_tensor(np.flatnonzero(counts == 0), device=self.device)
            self.net.item_id.weight[unknown] = 0
            self.net.hist_emb.weight[unknown] = 0
        log_q = torch.as_tensor(
            np.log(np.maximum(counts, 1) / counts.sum()),
            dtype=torch.float32, device=self.device
        )
        for epoch in range(1, self.epochs + 1):
            started, total_loss = time.perf_counter(), 0.0
            self.net.train()
            permutation = rng.permutation(len(items))
            for start in range(0, len(items), self.batch_size):
                positions = permutation[start:start + self.batch_size]
                batch_users = users[positions]
                hist = self._gather_history(items, positions, offsets[batch_users])
                positives = torch.as_tensor(items[positions], device=self.device)
                user_vectors = self.net.user_vectors(
                    torch.as_tensor(hist, device=self.device),
                    *self._user_tensors(batch_users)
                )
                item_vectors = self.net.item_vectors(positives)
                logits = user_vectors @ item_vectors.T / 0.05
                logits = logits - log_q[positives][None, :]
                duplicates = positives[:, None] == positives[None, :]
                duplicates.fill_diagonal_(False)
                logits = logits.masked_fill(duplicates, float("-inf"))
                labels = torch.arange(len(positions), device=self.device)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(positions)
            print(f"  epoch={epoch} loss={total_loss / len(items):.4f} "
                  f"time_s={time.perf_counter() - started:.1f}")

    @torch.no_grad()
    def _set_history(self, history):
        self.net.eval()
        items, _, offsets = self._sequences(history)
        user_histories = self._gather_history(items, offsets[1:], offsets[:-1])
        item_ids = torch.arange(self.data.n_items, device=self.device)
        self.item_vectors = self.net.item_vectors(item_ids).cpu().numpy()
        vectors = []
        for start in range(0, self.data.n_users, 2048):
            users = np.arange(start, min(start + 2048, self.data.n_users))
            hist = torch.as_tensor(user_histories[users], device=self.device)
            vectors.append(
                self.net.user_vectors(hist, *self._user_tensors(users)).cpu().numpy()
            )
        self.user_vectors = np.vstack(vectors)

    def score(self, users):
        return self.user_vectors[users] @ self.item_vectors.T
