#!/usr/bin/env python3
"""
Test de l'algorithme ALS sur un sous-ensemble du dataset MovieLens 1M.
"""

import pandas as pd
import numpy as np
import time
from engine import ALSRecommender
from utils import load_ratings, load_movies, get_dataset_stats


def main():
    print("=" * 70)
    print("  TEST ALS — The Streaming War")
    print("=" * 70)

    # ── Chargement ───────────────────────────────
    print("\n[1/4] Chargement des données…")
    ratings_df = load_ratings("ml-1m")
    movies_df = load_movies("ml-1m")
    stats = get_dataset_stats(ratings_df)

    print(f"  • Utilisateurs   : {stats['n_users']:,}")
    print(f"  • Films          : {stats['n_movies']:,}")
    print(f"  • Notations      : {stats['n_ratings']:,}")
    print(f"  • Sparsité       : {stats['sparsity']*100:.2f}%")
    print(f"  • Note moyenne   : {stats['mean_rating']:.2f}")

    # ── Sous-ensemble de test ─────────────────────
    print("\n[2/4] Création du sous-ensemble (50 000 notations)…")
    sample = ratings_df.sample(n=50_000, random_state=42)

    # ── Entraînement ─────────────────────────────
    print("\n[3/4] Entraînement ALS…")
    t0 = time.time()
    model = ALSRecommender(n_factors=10, n_iterations=5, reg_param=0.1)
    model.fit(sample)
    print(f"  Terminé en {time.time() - t0:.1f}s — RMSE final : {model.rmse_final:.4f}")

    # ── Tests ────────────────────────────────────
    print("\n[4/4] Tests fonctionnels")

    # Test 1 : utilisateur existant
    uid = 1
    print(f"\n  ▶ Top-5 pour l'utilisateur {uid}")
    t0 = time.time()
    recs = model.recommend_top_k(uid, k=5)
    print(f"    Généré en {(time.time()-t0)*1000:.1f} ms")
    for rank, (mid, pred) in enumerate(recs, 1):
        title = movies_df.loc[movies_df.movie_id == mid, "title"]
        t = title.values[0] if len(title) > 0 else f"Film #{mid}"
        print(f"    {rank}. {t} — {pred:.2f}/5")

    # Test 2 : prédiction unitaire
    print(f"\n  ▶ Prédiction u={uid}, film=1")
    pred = model.predict(uid, 1)
    title = movies_df.loc[movies_df.movie_id == 1, "title"].values
    print(f"    {title[0] if len(title) > 0 else 'Film #1'} → {pred:.2f}/5")

    # Test 3 : cold start
    print("\n  ▶ Cold start (utilisateur inconnu)")
    recs_cs = model.recommend_top_k(99999, k=5)
    for rank, (mid, avg) in enumerate(recs_cs, 1):
        title = movies_df.loc[movies_df.movie_id == mid, "title"]
        t = title.values[0] if len(title) > 0 else f"Film #{mid}"
        print(f"    {rank}. {t} — avg {avg:.2f}/5")

    # Historique de perte
    print("\n  ▶ Historique RMSE")
    for i, loss in enumerate(model.training_loss_history, 1):
        print(f"    Iter {i}: {loss:.4f}")

    # Sauvegarde / chargement
    model.save("/tmp/test_model.pkl")
    loaded = ALSRecommender.load("/tmp/test_model.pkl")
    assert abs(loaded.predict(uid, 1) - pred) < 1e-6, "Erreur de cohérence après chargement"

    print("\n" + "=" * 70)
    print("  ✅ TOUS LES TESTS PASSÉS")
    print("=" * 70)


if __name__ == "__main__":
    main()
