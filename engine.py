import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import time
import pickle
import os


class ALSRecommender:
    """
    Algorithme Alternating Least Squares (ALS) pour la factorisation matricielle.

    Optimise alternativement les matrices de facteurs latents utilisateurs et films
    en minimisant l'erreur quadratique regularisee.
    """

    def __init__(self, n_factors=20, n_iterations=15, reg_param=0.1):
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.reg_param = reg_param

        self.user_factors = None
        self.item_factors = None
        self.user_biases = None
        self.item_biases = None
        self.global_mean = None

        self.user_id_map = {}
        self.item_id_map = {}
        self.inverse_user_map = {}
        self.inverse_item_map = {}

        self.rating_matrix = None
        self.training_loss_history = []

        # Statistiques pour l'affichage
        self.n_users = 0
        self.n_items = 0
        self.n_ratings = 0
        self.sparsity = 0.0
        self.rmse_final = None

    def _create_id_mappings(self, user_ids, item_ids):
        unique_users = np.unique(user_ids)
        unique_items = np.unique(item_ids)

        self.user_id_map = {int(uid): idx for idx, uid in enumerate(unique_users)}
        self.item_id_map = {int(iid): idx for idx, iid in enumerate(unique_items)}
        self.inverse_user_map = {idx: int(uid) for uid, idx in self.user_id_map.items()}
        self.inverse_item_map = {idx: int(iid) for iid, idx in self.item_id_map.items()}

        return len(unique_users), len(unique_items)

    def fit(self, ratings_df):
        """Entraîne le modèle ALS sur le DataFrame de notations."""
        print(f"Démarrage entraînement ALS — facteurs={self.n_factors}, iter={self.n_iterations}, λ={self.reg_param}")

        user_ids = ratings_df["user_id"].values
        item_ids = ratings_df["movie_id"].values
        ratings = ratings_df["rating"].values.astype(np.float64)

        self.n_ratings = len(ratings)
        n_users, n_items = self._create_id_mappings(user_ids, item_ids)
        self.n_users = n_users
        self.n_items = n_items
        self.sparsity = 1 - (self.n_ratings / (n_users * n_items))

        user_indices = np.array([self.user_id_map[int(u)] for u in user_ids])
        item_indices = np.array([self.item_id_map[int(i)] for i in item_ids])

        # Initialisation
        rng = np.random.default_rng(42)
        self.user_factors = rng.normal(0, 0.01, (n_users, self.n_factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, self.n_factors))
        self.global_mean = float(ratings.mean())
        self.user_biases = np.zeros(n_users)
        self.item_biases = np.zeros(n_items)

        self.rating_matrix = csr_matrix(
            (ratings, (user_indices, item_indices)), shape=(n_users, n_items)
        )

        self.training_loss_history = []

        for iteration in range(self.n_iterations):
            t0 = time.time()
            self._update_user_factors(user_indices, item_indices, ratings)
            self._update_item_factors(user_indices, item_indices, ratings)
            loss = self._compute_rmse(user_indices, item_indices, ratings)
            self.training_loss_history.append(loss)
            elapsed = time.time() - t0
            print(f"  Iter {iteration + 1:2d}/{self.n_iterations} | RMSE={loss:.4f} | {elapsed:.1f}s")

        self.rmse_final = self.training_loss_history[-1]
        print(f"Entraînement terminé. RMSE final : {self.rmse_final:.4f}")

    def _update_user_factors(self, user_indices, item_indices, ratings):
        I = self.reg_param * np.eye(self.n_factors)
        for u in range(self.n_users):
            mask = user_indices == u
            if not mask.any():
                continue
            items = item_indices[mask]
            r = ratings[mask] - self.global_mean - self.item_biases[items]
            X = self.item_factors[items]
            A = X.T @ X + I
            b = X.T @ r
            try:
                self.user_factors[u] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                self.user_factors[u] = np.linalg.lstsq(A, b, rcond=None)[0]

    def _update_item_factors(self, user_indices, item_indices, ratings):
        I = self.reg_param * np.eye(self.n_factors)
        for i in range(self.n_items):
            mask = item_indices == i
            if not mask.any():
                continue
            users = user_indices[mask]
            r = ratings[mask] - self.global_mean - self.user_biases[users]
            X = self.user_factors[users]
            A = X.T @ X + I
            b = X.T @ r
            try:
                self.item_factors[i] = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                self.item_factors[i] = np.linalg.lstsq(A, b, rcond=None)[0]

    def _compute_rmse(self, user_indices, item_indices, ratings):
        preds = np.array(
            [self.predict_internal(u, i) for u, i in zip(user_indices, item_indices)]
        )
        return float(np.sqrt(np.mean((ratings - preds) ** 2)))

    def predict_internal(self, user_idx, item_idx):
        pred = (
            self.global_mean
            + self.user_biases[user_idx]
            + self.item_biases[item_idx]
            + np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
        )
        return float(np.clip(pred, 1.0, 5.0))

    def predict(self, user_id, movie_id):
        uid = int(user_id)
        mid = int(movie_id)
        if uid not in self.user_id_map or mid not in self.item_id_map:
            return self.global_mean
        return self.predict_internal(self.user_id_map[uid], self.item_id_map[mid])

    def recommend_top_k(self, user_id, k=10, exclude_rated=True):
        uid = int(user_id)
        if uid not in self.user_id_map:
            return self._get_popular_items(k)

        user_idx = self.user_id_map[uid]

        # Vectorised prediction
        preds = (
            self.global_mean
            + self.user_biases[user_idx]
            + self.item_biases
            + self.item_factors @ self.user_factors[user_idx]
        )
        preds = np.clip(preds, 1.0, 5.0)

        if exclude_rated:
            rated_indices = set(self.rating_matrix[user_idx].nonzero()[1])
            preds[list(rated_indices)] = -np.inf

        top_indices = np.argpartition(preds, -k)[-k:]
        top_indices = top_indices[np.argsort(preds[top_indices])[::-1]]

        return [(self.inverse_item_map[idx], float(preds[idx])) for idx in top_indices]

    def _get_popular_items(self, k):
        col_counts = np.diff(self.rating_matrix.tocsc().indptr)
        col_sums = np.array(self.rating_matrix.sum(axis=0)).flatten()
        with np.errstate(invalid="ignore"):
            means = np.where(col_counts > 0, col_sums / col_counts, self.global_mean)
        top = np.argsort(means)[::-1][:k]
        return [(self.inverse_item_map[int(i)], float(means[i])) for i in top]

    # ------------------------------------------------------------------ #
    #  Persistence                                                          #
    # ------------------------------------------------------------------ #
    def save(self, path="als_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Modèle sauvegardé → {path}")

    @staticmethod
    def load(path="als_model.pkl"):
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"Modèle chargé depuis {path}")
        return model
