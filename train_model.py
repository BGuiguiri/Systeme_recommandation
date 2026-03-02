#!/usr/bin/env python3
"""
Script d'entraînement préalable du modèle ALS.
À exécuter UNE SEULE FOIS avant de lancer l'application Streamlit.

Usage :
    python train_model.py
    python train_model.py --path ml-1m --factors 20 --iters 15 --reg 0.1
"""

import argparse
import os
from engine import ALSRecommender
from utils import load_ratings, get_dataset_stats


def main():
    parser = argparse.ArgumentParser(description="Entraîne et sauvegarde le modèle ALS")
    parser.add_argument("--path", default="ml-1m", help="Chemin du dataset MovieLens 1M")
    parser.add_argument("--factors", type=int, default=50, help="Nombre de facteurs latents")
    parser.add_argument("--iters", type=int, default=15, help="Nombre d'itérations ALS")
    parser.add_argument("--reg", type=float, default=0.1, help="Coefficient de régularisation")
    parser.add_argument("--output", default="als_model.pkl", help="Fichier de sauvegarde")
    args = parser.parse_args()

    print("=" * 60)
    print("  ENTRAÎNEMENT DU MODÈLE ALS — The Streaming War")
    print("=" * 60)

    # Chargement des données
    print(f"\n[1/3] Chargement des données depuis '{args.path}'...")
    ratings_df = load_ratings(args.path)
    stats = get_dataset_stats(ratings_df)

    print(f"  • Utilisateurs  : {stats['n_users']:,}")
    print(f"  • Films         : {stats['n_movies']:,}")
    print(f"  • Notations     : {stats['n_ratings']:,}")
    print(f"  • Sparsité      : {stats['sparsity']:.4f} ({stats['sparsity']*100:.2f}%)")
    print(f"  • Note moyenne  : {stats['mean_rating']:.2f}")

    # Entraînement
    print(f"\n[2/3] Entraînement (facteurs={args.factors}, iter={args.iters}, λ={args.reg})...")
    model = ALSRecommender(
        n_factors=args.factors,
        n_iterations=args.iters,
        reg_param=args.reg,
    )
    model.fit(ratings_df)

    # Sauvegarde
    print(f"\n[3/3] Sauvegarde du modèle → {args.output}")
    model.save(args.output)

    print("\n✅ Modèle prêt. Lancez l'application avec : streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
