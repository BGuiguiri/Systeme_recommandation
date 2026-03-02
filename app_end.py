"""
Digui Stream — Application Streamlit avec ALS Recommender
=============================================================
Structure :
  • Page 1 : Inscription / Connexion
  • Page 2 : Interface principale
      – Sidebar   : statistiques du modèle + dataset
      – Main      : carrousel de films + onglets recommandations / recherche
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import time
import hashlib
import json
import base64
import requests
from io import BytesIO
from engine import ALSRecommender
from utils import load_movies, load_ratings, get_dataset_stats

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────
DATASET_PATH = "ml-1m"
MODEL_PATH = "als_model.pkl"
USERS_FILE = "users.json"
OMDB_API_KEY = os.getenv("OMDB_API_KEY", "")   # optionnel — https://www.omdbapi.com/apikey.aspx

st.set_page_config(
    page_title="Digui Stream",
    page_icon="▶",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CSS global
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
/* Palette blanche et bleue — Digui Stream */
:root {
    --bg: #f0f6ff;
    --bg2: #ffffff;
    --card: #ffffff;
    --accent: #1a56db;
    --accent2: #1e40af;
    --accent-light: #dbeafe;
    --text: #1e293b;
    --sub: #64748b;
    --border: #e2e8f0;
}

/* Fond général */
.stApp { background-color: var(--bg); color: var(--text); }

/* Sidebar bleue */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e40af 0%, #1a56db 100%) !important;
}
section[data-testid="stSidebar"] .stMarkdown,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span { color: #ffffff !important; }
section[data-testid="stSidebar"] .stat-box {
    background: rgba(255,255,255,0.15) !important;
    border-left: 4px solid #93c5fd !important;
    color: #ffffff !important;
}
section[data-testid="stSidebar"] .stat-box .val {
    color: #bfdbfe !important;
}

/* Titre principal */
.big-title {
    font-size: 3.2rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(135deg, #1a56db, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
    letter-spacing: -1px;
}
.sub-title {
    text-align: center;
    color: var(--sub);
    font-size: 1.1rem;
    margin-bottom: 1.5rem;
}

/* Cards de film */
.film-card {
    background: var(--card);
    border-radius: 14px;
    padding: 12px;
    text-align: center;
    border: 1px solid var(--border);
    box-shadow: 0 2px 12px rgba(26,86,219,0.08);
    transition: transform .2s, box-shadow .2s;
}
.film-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 24px rgba(26,86,219,0.18);
}
.film-card .title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text);
    margin-top: 8px;
}
.film-card .rating { color: #f59e0b; font-size: 0.8rem; }

/* Metric boxes */
.stat-box {
    background: rgba(255,255,255,0.18);
    border-left: 4px solid #93c5fd;
    border-radius: 8px;
    padding: 10px 14px;
    margin-bottom: 10px;
    font-size: 0.92rem;
}
.stat-box .val { font-size: 1.3rem; font-weight: 700; color: #bfdbfe; }

/* Boutons */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: background .2s !important;
}
.stButton > button:hover { background: var(--accent2) !important; }

/* Onglets */
.stTabs [data-baseweb="tab"] { font-weight: 600; color: var(--sub); }
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 3px solid var(--accent) !important;
}

/* Inputs */
.stTextInput > div > div > input,
.stNumberInput input {
    background: var(--bg2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}

/* Séparateurs */
hr { border-color: var(--border) !important; }

/* Login card */
.login-container {
    max-width: 420px;
    margin: 60px auto;
    background: var(--card);
    padding: 40px;
    border-radius: 18px;
    border: 1px solid var(--border);
    box-shadow: 0 8px 40px rgba(26,86,219,0.12);
}
.login-title {
    font-size: 2.2rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(135deg, #1a56db, #0ea5e9);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
    letter-spacing: -1px;
}
.login-sub { text-align: center; color: var(--sub); margin-bottom: 24px; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  Animation : oiseaux en arrière-plan
# ─────────────────────────────────────────────
st.markdown("""
<div id="birds-container" style="
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
"></div>

<style>
@keyframes fly {
    0%   { transform: translateX(-120px) translateY(0px) scaleX(1); }
    25%  { transform: translateX(25vw)   translateY(-40px) scaleX(1); }
    50%  { transform: translateX(50vw)   translateY(20px) scaleX(1); }
    75%  { transform: translateX(75vw)   translateY(-30px) scaleX(1); }
    100% { transform: translateX(110vw)  translateY(0px) scaleX(1); }
}
@keyframes wingFlap {
    0%, 100% { d: path("M0,8 Q-12,-6 -24,0 Q-12,4 0,8 Q12,4 24,0 Q12,-6 0,8"); }
    50%       { d: path("M0,8 Q-12,6  -24,0 Q-12,-4 0,8 Q12,-4 24,0 Q12,6  0,8"); }
}
.bird {
    position: absolute;
    opacity: 0.55;
    animation: fly linear infinite;
}
.bird svg path.wings {
    animation: wingFlap 0.45s ease-in-out infinite;
}
</style>

<script>
(function() {
    const container = document.getElementById('birds-container');
    if (!container) return;

    // SVG d'un oiseau stylisé (deux ailes + corps)
    function makeBird(size, color) {
        return `
        <svg width="${size}" height="${size * 0.6}" viewBox="-30 -15 60 30" fill="none" xmlns="http://www.w3.org/2000/svg">
          <!-- Corps -->
          <ellipse cx="0" cy="4" rx="7" ry="3.5" fill="${color}" opacity="0.9"/>
          <!-- Tête -->
          <circle cx="9" cy="1" r="4" fill="${color}" opacity="0.9"/>
          <!-- Bec -->
          <polygon points="13,1 17,-1 13,3" fill="${color}"/>
          <!-- Queue -->
          <polygon points="-7,4 -14,0 -14,8" fill="${color}" opacity="0.8"/>
          <!-- Aile gauche (haut) -->
          <path class="wings" d="M0,2 Q-10,-10 -22,-4 Q-10,2 0,2" fill="${color}" opacity="0.85"/>
          <!-- Aile droite (bas) -->
          <path class="wings" d="M0,6 Q-10,14 -22,10 Q-10,6 0,6" fill="${color}" opacity="0.6"/>
        </svg>`;
    }

    const configs = [
        { size: 38, color: '#1a56db', top: '12%',  duration: '18s', delay: '0s'   },
        { size: 28, color: '#3b82f6', top: '28%',  duration: '24s', delay: '4s'   },
        { size: 44, color: '#1e40af', top: '8%',   duration: '20s', delay: '9s'   },
        { size: 22, color: '#60a5fa', top: '42%',  duration: '28s', delay: '2s'   },
        { size: 34, color: '#1a56db', top: '55%',  duration: '22s', delay: '14s'  },
        { size: 26, color: '#93c5fd', top: '70%',  duration: '30s', delay: '7s'   },
        { size: 18, color: '#3b82f6', top: '20%',  duration: '26s', delay: '17s'  },
        { size: 40, color: '#1e40af', top: '80%',  duration: '19s', delay: '5s'   },
    ];

    configs.forEach(cfg => {
        const div = document.createElement('div');
        div.className = 'bird';
        div.innerHTML = makeBird(cfg.size, cfg.color);
        div.style.top = cfg.top;
        div.style.left = '-120px';
        div.style.animationDuration = cfg.duration;
        div.style.animationDelay = cfg.delay;
        container.appendChild(div);
    });
})();
</script>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Gestion des utilisateurs (JSON local)
# ─────────────────────────────────────────────

def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


def _load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            return json.load(f)
    return {}


def _save_users(users: dict):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def register_user(username: str, password: str) -> bool:
    users = _load_users()
    if username in users:
        return False
    users[username] = {"password": _hash(password), "ratings": {}}
    _save_users(users)
    return True


def login_user(username: str, password: str) -> bool:
    users = _load_users()
    return username in users and users[username]["password"] == _hash(password)


def get_user_ratings(username: str) -> dict:
    users = _load_users()
    return users.get(username, {}).get("ratings", {})


def save_user_rating(username: str, movie_id: int, rating: float):
    users = _load_users()
    if username not in users:
        return
    users[username]["ratings"][str(movie_id)] = rating
    _save_users(users)

# ─────────────────────────────────────────────
#  Chargement des ressources
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Chargement du modèle ALS…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Modèle introuvable ({MODEL_PATH}). Lancez d'abord : python train_model.py")
        st.stop()
    return ALSRecommender.load(MODEL_PATH)


@st.cache_resource(show_spinner="Chargement des films…")
def load_movies_cached():
    return load_movies(DATASET_PATH)


@st.cache_resource(show_spinner="Chargement des notations…")
def load_ratings_cached():
    return load_ratings(DATASET_PATH)


@st.cache_data(show_spinner=False, ttl=86400)
def get_poster_url(title: str, omdb_key: str = "") -> str:
    """
    Retourne l'URL de l'affiche d'un film via l'API OMDB.

    Stratégie :
      1. Extrait le titre propre et l'année depuis le format MovieLens ("Titre (AAAA)")
      2. Interroge OMDB avec titre + année pour un matching précis
      3. Si OMDB renvoie "N/A" ou échoue → fallback sur un placeholder stylisé
    Le décorateur @st.cache_data évite de ré-interroger l'API pour le même film
    et préserve le quota journalier (1 000 req/jour sur le plan gratuit).
    """
    # ── Nettoyage du titre MovieLens ──────────────────────────────────────
    year = ""
    clean = title.strip()
    if clean.endswith(")") and "(" in clean:
        year = clean[clean.rfind("(") + 1 : -1]
        clean = clean[: clean.rfind("(")].strip()

    # ── Appel OMDB ────────────────────────────────────────────────────────
    if omdb_key:
        try:
            params = {"t": clean, "apikey": omdb_key}
            if year:
                params["y"] = year          # précision accrue → moins de faux positifs
            r = requests.get("http://www.omdbapi.com/", params=params, timeout=4)
            r.raise_for_status()
            data = r.json()
            poster = data.get("Poster", "N/A")
            if poster and poster != "N/A":
                return poster               # URL directe fournie par OMDB
        except Exception:
            pass  # quota dépassé, réseau indisponible, titre introuvable → fallback

    # ── Fallback : placeholder élégant aux couleurs de l'app ─────────────
    # Couleur unique et reproductible basée sur le hash du titre
    hue = abs(hash(clean)) % 360
    # On utilise un service de placeholder avec couleur HSL simulée en hex
    palette = [
        ("1a1a2e", "e94560"),  # sombre / rouge
        ("0f3460", "f5a623"),  # bleu / orange
        ("16213e", "4ecdc4"),  # marine / turquoise
        ("2d132c", "ee4540"),  # bordeaux / rose
    ]
    bg, fg = palette[abs(hash(clean)) % len(palette)]
    label = clean[:18].replace(" ", "+")
    return f"https://placehold.co/300x450/{bg}/{fg}?text={label}&font=playfair-display"

# ─────────────────────────────────────────────
#  Session state initialisation
# ─────────────────────────────────────────────
defaults = {
    "logged_in": False,
    "username": "",
    "auth_tab": "login",
    "carousel_index": 0,
    "last_carousel_tick": 0.0,
    "user_ratings": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
#  PAGE 1 : Authentification
# ─────────────────────────────────────────────
if not st.session_state.logged_in:
    st.markdown(
        '<div class="login-title">Digui Stream</div>'
        '<div class="login-sub">Votre plateforme de recommandation cinéma</div>',
        unsafe_allow_html=True,
    )

    col_center = st.columns([1, 2, 1])[1]
    with col_center:
        login_tab, register_tab = st.tabs(["Connexion", "Inscription"])

        with login_tab:
            username_in = st.text_input("Nom d'utilisateur", key="li_user")
            password_in = st.text_input("Mot de passe", type="password", key="li_pass")
            if st.button("Se connecter", use_container_width=True, key="li_btn"):
                if login_user(username_in, password_in):
                    st.session_state.logged_in = True
                    st.session_state.username = username_in
                    st.session_state.user_ratings = get_user_ratings(username_in)
                    st.rerun()
                else:
                    st.error("Identifiants incorrects.")

        with register_tab:
            new_user = st.text_input("Choisissez un nom d'utilisateur", key="reg_user")
            new_pass = st.text_input("Choisissez un mot de passe", type="password", key="reg_pass")
            new_pass2 = st.text_input("Confirmez le mot de passe", type="password", key="reg_pass2")
            if st.button("S'inscrire", use_container_width=True, key="reg_btn"):
                if new_pass != new_pass2:
                    st.error("Les mots de passe ne correspondent pas.")
                elif len(new_user) < 3:
                    st.error("Nom d'utilisateur trop court (min 3 caractères).")
                elif register_user(new_user, new_pass):
                    st.success("Compte créé ! Vous pouvez maintenant vous connecter.")
                else:
                    st.error("Ce nom d'utilisateur est déjà pris.")
    st.stop()

# ─────────────────────────────────────────────
#  PAGE 2 : Interface principale
# ─────────────────────────────────────────────
model = load_model()
movies_df = load_movies_cached()
ratings_df = load_ratings_cached()
stats = get_dataset_stats(ratings_df)

# Synchroniser les notations de l'utilisateur courant
if not st.session_state.user_ratings:
    st.session_state.user_ratings = get_user_ratings(st.session_state.username)

# ── Sidebar ────────────────────────────────────
with st.sidebar:
    st.markdown(f"### {st.session_state.username}")
    if st.button("Déconnexion"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.session_state.user_ratings = {}
        st.rerun()

    st.markdown("---")
    st.markdown("### Statistiques Dataset")

    def stat_box(label, value):
        st.markdown(
            f'<div class="stat-box"><div>{label}</div><div class="val">{value}</div></div>',
            unsafe_allow_html=True,
        )

    stat_box("Utilisateurs", f"{stats['n_users']:,}")
    stat_box("Films", f"{stats['n_movies']:,}")
    stat_box("Notations totales", f"{stats['n_ratings']:,}")
    stat_box("Sparsité", f"{stats['sparsity']*100:.2f}%")
    stat_box("RMSE du modèle", f"{model.rmse_final:.4f}" if model.rmse_final else "N/A")
    stat_box("Facteurs latents", str(model.n_factors))
    stat_box("Itérations ALS", str(model.n_iterations))
    stat_box("λ Régularisation", str(model.reg_param))
    stat_box("Note moyenne", f"{stats['mean_rating']:.2f} / 5")

    st.markdown("---")
    st.markdown("### Convergence ALS")
    if model.training_loss_history:
        loss_df = pd.DataFrame(
            {
                "Itération": range(1, len(model.training_loss_history) + 1),
                "RMSE": model.training_loss_history,
            }
        ).set_index("Itération")
        st.line_chart(loss_df, color="#1a56db")

    st.markdown("---")
    st.caption(
        "**Pourquoi ALS ?**  \n"
        "ALS est naturellement parallélisable : chaque mise à jour utilisateur/film est indépendante. "
        "Contrairement au SGD, il se prête au calcul distribué (Spark, etc.) et converge plus "
        "stablement sur des matrices creuses comme MovieLens."
    )

# ── En-tête ────────────────────────────────────
st.markdown('<div class="big-title">Digui Stream</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Moteur de recommandation · MovieLens 1M</div>',
    unsafe_allow_html=True,
)

# ── Carrousel automatique (toutes les 5 s) ─────
CAROUSEL_MOVIES = movies_df.sample(20, random_state=99).reset_index(drop=True)

now = time.time()
if now - st.session_state.last_carousel_tick > 5:
    st.session_state.carousel_index = (st.session_state.carousel_index + 1) % len(CAROUSEL_MOVIES)
    st.session_state.last_carousel_tick = now

current_films = CAROUSEL_MOVIES.iloc[
    st.session_state.carousel_index : st.session_state.carousel_index + 5
]
if len(current_films) < 5:
    current_films = pd.concat(
        [current_films, CAROUSEL_MOVIES.iloc[: 5 - len(current_films)]]
    )

carousel_cols = st.columns(5)
for col, (_, row) in zip(carousel_cols, current_films.iterrows()):
    img_url = get_poster_url(row["title"], OMDB_API_KEY)
    col.markdown(
        f"""
        <div class="film-card">
            <img src="{img_url}" style="width:100%;height:200px;object-fit:cover;border-radius:8px;">
            <div class="title">{row['title'][:40]}</div>
            <div class="rating">{row['genres'].split('|')[0]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Auto-refresh léger du carrousel
st.markdown(
    """
<script>
setTimeout(function(){ window.location.reload(); }, 5000);
</script>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# ─────────────────────────────────────────────
#  Onglets principaux
# ─────────────────────────────────────────────
tab_reco, tab_search, tab_profile = st.tabs(
    ["Mes Recommandations", "Rechercher & Noter", "Mon Profil"]
)

# ── Helper : récupérer les recommandations de l'utilisateur connecté ──
def get_recommendations(k=12):
    """
    Génère les recommandations personnalisées pour l'utilisateur connecté.

    Logique :
      1. L'utilisateur n'a aucune note → films les plus populaires du modèle.
      2. L'utilisateur a au moins 2 notes → on calcule son vecteur de goûts
         à la volée (une étape ALS) sans ré-entraîner le modèle, puis on
         prédit ses notes sur tous les films et on retourne le top-k.
      3. L'utilisateur a un username numérique ET existe dans le modèle →
         on utilise directement son vecteur appris (utilisateurs MovieLens).
    """
    user_ratings = st.session_state.user_ratings

    # ── Cas 1 : utilisateur MovieLens connu dans le modèle ────────────────
    try:
        uid = int(st.session_state.username)
        if uid in model.user_id_map:
            return model.recommend_top_k(uid, k=k)
    except ValueError:
        pass

    # ── Cas 2 : aucune note → films populaires ───────────────────────────
    if not user_ratings:
        return model._get_popular_items(k)

    # ── Cas 3 : nouvel utilisateur avec des notes ────────────────────────
    # On garde uniquement les films que le modèle connaît
    known = [
        (int(mid), float(r))
        for mid, r in user_ratings.items()
        if int(mid) in model.item_id_map
    ]

    # Moins de 2 notes connues → pas assez pour calculer un profil fiable
    if len(known) < 2:
        return model._get_popular_items(k)

    # Indices internes des films notés
    item_indices = np.array([model.item_id_map[mid] for mid, _ in known])
    ratings_arr  = np.array([r for _, r in known], dtype=np.float64)

    # Calcul du vecteur utilisateur par une étape ALS :
    # On résout (X^T X + λI) · u = X^T · y
    # où X = facteurs des films notés, y = notes centrées
    X = model.item_factors[item_indices]
    y = ratings_arr - model.global_mean - model.item_biases[item_indices]
    A = X.T @ X + model.reg_param * np.eye(model.n_factors)
    user_vector = np.linalg.solve(A, X.T @ y)

    # Prédiction sur tous les films (vectorisée)
    preds = np.clip(
        model.global_mean + model.item_biases + model.item_factors @ user_vector,
        1.0, 5.0
    )

    # Exclure les films déjà notés par l'utilisateur
    for mid, _ in known:
        preds[model.item_id_map[mid]] = -np.inf

    # Retourner le top-k
    top_indices = np.argsort(preds)[::-1][:k]
    return [(model.inverse_item_map[int(i)], float(preds[i])) for i in top_indices]


# ── TAB 1 : Recommandations ────────────────────
with tab_reco:
    st.subheader("Films recommandés pour vous")

    k_reco = st.slider("Nombre de recommandations", 6, 24, 12, key="k_reco")

    recs = get_recommendations(k=k_reco)

    if not recs:
        st.info("Aucune recommandation disponible. Notez quelques films d'abord !")
    else:
        cols_per_row = 4
        for row_start in range(0, len(recs), cols_per_row):
            row_recs = recs[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, (mid, pred) in zip(cols, row_recs):
                title_row = movies_df[movies_df["movie_id"] == mid]
                title = title_row["title"].values[0] if len(title_row) > 0 else f"Film #{mid}"
                genres = title_row["genres"].values[0] if len(title_row) > 0 else ""
                img_url = get_poster_url(title, OMDB_API_KEY)
                stars = "★" * round(pred)
                col.markdown(
                    f"""
                    <div class="film-card">
                        <img src="{img_url}" style="width:100%;height:180px;object-fit:cover;border-radius:8px;">
                        <div class="title">{title[:35]}</div>
                        <div class="rating">{stars} {pred:.1f}/5</div>
                        <div style="font-size:0.75rem;color:#a0a0b0;">{genres.split('|')[0] if genres else ''}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

# ── TAB 2 : Recherche & Notation ───────────────
with tab_search:
    st.subheader("Recherchez un film et notez-le")

    search_query = st.text_input(
        "Tapez le titre d'un film…",
        placeholder="ex: Toy Story, Matrix, Titanic…",
        key="search_input",
    )

    if search_query.strip():
        mask = movies_df["title"].str.contains(search_query, case=False, na=False)
        results = movies_df[mask].head(12)

        if results.empty:
            st.warning("Aucun film trouvé pour cette recherche.")
        else:
            st.markdown(f"**{len(results)} résultat(s) trouvé(s)**")
            cols_per_row = 4
            for row_start in range(0, len(results), cols_per_row):
                chunk = results.iloc[row_start : row_start + cols_per_row]
                cols = st.columns(cols_per_row)
                for col, (_, row) in zip(cols, chunk.iterrows()):
                    img_url = get_poster_url(row["title"], OMDB_API_KEY)
                    already = st.session_state.user_ratings.get(str(row["movie_id"]))
                    already_label = f"Votre note : {already}/5" if already else "Non noté"
                    col.markdown(
                        f"""
                        <div class="film-card">
                            <img src="{img_url}" style="width:100%;height:160px;object-fit:cover;border-radius:8px;">
                            <div class="title">{row['title'][:35]}</div>
                            <div class="rating" style="font-size:0.78rem;">{already_label}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    with col:
                        new_rating = st.select_slider(
                            f"Note",
                            options=[1, 2, 3, 4, 5],
                            value=int(already) if already else 3,
                            key=f"rate_{row['movie_id']}",
                            label_visibility="collapsed",
                        )
                        if st.button("Enregistrer", key=f"save_{row['movie_id']}", use_container_width=True):
                            mid = int(row["movie_id"])
                            st.session_state.user_ratings[str(mid)] = new_rating
                            save_user_rating(st.session_state.username, mid, new_rating)
                            st.success(f"'{row['title'][:25]}' noté {new_rating}/5")
                            st.rerun()
    else:
        st.info("Commencez à taper pour rechercher un film. Les recommandations se mettront à jour après chaque notation.")

# ── TAB 3 : Profil ─────────────────────────────
with tab_profile:
    st.subheader(f"Profil de {st.session_state.username}")

    user_ratings = st.session_state.user_ratings
    if not user_ratings:
        st.info("Vous n'avez encore noté aucun film. Utilisez l'onglet **Rechercher & Noter** !")
    else:
        st.markdown(f"**Vous avez noté {len(user_ratings)} film(s)**")
        rated_data = []
        for mid_str, rating in user_ratings.items():
            mid = int(mid_str)
            row = movies_df[movies_df["movie_id"] == mid]
            title = row["title"].values[0] if len(row) > 0 else f"Film #{mid}"
            genres = row["genres"].values[0] if len(row) > 0 else ""
            rated_data.append({"Titre": title, "Genre": genres.split("|")[0], "Votre note": rating})

        df_rated = pd.DataFrame(rated_data).sort_values("Votre note", ascending=False)
        st.dataframe(df_rated, use_container_width=True, hide_index=True)

        # Distribution des notes
        st.markdown("#### Distribution de vos notes")
        counts = pd.Series(list(user_ratings.values())).value_counts().sort_index()
        st.bar_chart(counts)
