import pandas as pd
import numpy as np
import os


def load_ratings(path="ml-1m"):
    """Charge le dataset MovieLens 1M ratings.dat"""
    filepath = os.path.join(path, "ratings.dat")
    ratings = pd.read_csv(
        filepath,
        sep="::",
        engine="python",
        names=["user_id", "movie_id", "rating", "timestamp"],
    )
    ratings["user_id"] = ratings["user_id"].astype(int)
    ratings["movie_id"] = ratings["movie_id"].astype(int)
    ratings["rating"] = ratings["rating"].astype(float)
    return ratings


def load_movies(path="ml-1m"):
    """Charge les informations des films."""
    filepath = os.path.join(path, "movies.dat")
    movies = pd.read_csv(
        filepath,
        sep="::",
        engine="python",
        names=["movie_id", "title", "genres"],
        encoding="latin-1",
    )
    movies["movie_id"] = movies["movie_id"].astype(int)
    return movies


def calculate_sparsity(ratings_df):
    """Calcule le taux de sparsité de la matrice utilisateur-film."""
    n_users = ratings_df["user_id"].nunique()
    n_movies = ratings_df["movie_id"].nunique()
    n_ratings = len(ratings_df)
    return 1 - (n_ratings / (n_users * n_movies))


def get_dataset_stats(ratings_df):
    """Retourne un dictionnaire de statistiques sur le dataset."""
    return {
        "n_users": int(ratings_df["user_id"].nunique()),
        "n_movies": int(ratings_df["movie_id"].nunique()),
        "n_ratings": int(len(ratings_df)),
        "sparsity": calculate_sparsity(ratings_df),
        "mean_rating": float(ratings_df["rating"].mean()),
        "min_rating": float(ratings_df["rating"].min()),
        "max_rating": float(ratings_df["rating"].max()),
    }
