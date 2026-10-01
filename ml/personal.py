"""
personal.py — Personnalisation des recommandations par l'historique du joueur.

Le modèle global mesure la force d'un champion « joué par ceux qui le
choisissent ». Ton niveau sur ce champion peut être très différent. Pour
chaque partie de ton historique, le modèle donne la proba de victoire p
attendue POUR CETTE DRAFT ; le résidu (victoire − p) isole donc ton apport
personnel, indépendamment de la qualité des drafts que tu as eues.

Pour un champion c joué n fois, ton décalage de logit est estimé par :

    δ_c = Σ (y − p) / (Σ p(1 − p) + 1/τ²)

C'est l'estimation a posteriori (un pas de Newton depuis 0) d'un décalage
propre à toi avec un a priori N(0, τ²) : avec peu de parties, δ reste près
de 0 ; avec beaucoup, il tend vers ton écart réel. τ, l'écart typique réel
entre joueurs sur un même champion, est mesuré sur toute la base
(estimate_tau) à chaque entraînement du modèle.
"""
import math
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH, RANKED_SOLO_QUEUE
from ml.additive_model import AdditiveDraftModel
from ml.draft_data import load_matches, load_player_slots

DEFAULT_TAU: float = 0.3   # Utilisé si le modèle n'embarque pas de τ mesuré
TAU_MIN_GAMES: int = 5     # Paires (joueur, champion) retenues pour mesurer τ


# ── Résidus joueur × champion ─────────────────────────────────────────────────

def residuals(model: AdditiveDraftModel, drafts: pd.DataFrame, slots: pd.DataFrame) -> pd.DataFrame:
    """
    Pour chaque (joueur, match) : y = victoire de son équipe, p = proba prédite
    pour son équipe d'après la draft.

    Returns:
        DataFrame puuid, match_id, champion_id, position, side, y, p.
    """
    p_blue = pd.Series(model.predict_proba(drafts), index=drafts.index)
    res = slots[slots.match_id.isin(drafts.index)].copy()
    blue = res.side == "blue"
    res["p"] = np.where(blue, res.match_id.map(p_blue), 1 - res.match_id.map(p_blue))
    blue_win = res.match_id.map(drafts.blue_win)
    res["y"] = np.where(blue, blue_win, 1 - blue_win)
    return res


def estimate_tau(res: pd.DataFrame, min_games: int = TAU_MIN_GAMES) -> float:
    """
    Méthode des moments sur les paires (joueur, champion) d'au moins
    `min_games` parties. Avec S = Σ(y − p) et V = Σ p(1 − p) :
        E[S²] ≈ V + τ² V²   ⇒   τ² = Σ(S² − V) / Σ V²
    Le terme V est le bruit binomial attendu ; l'excédent mesure les vrais
    écarts de niveau entre joueurs sur un même champion.
    """
    res = res.assign(r=res.y - res.p, v=res.p * (1 - res.p))
    g = res.groupby(["puuid", "champion_id"]).agg(S=("r", "sum"), V=("v", "sum"), n=("r", "size"))
    g = g[g.n >= min_games]
    if g.empty:
        return DEFAULT_TAU
    tau2 = ((g.S ** 2 - g.V).sum()) / (g.V ** 2).sum()
    return math.sqrt(max(tau2, 1e-4))


# ── Profil d'un joueur ────────────────────────────────────────────────────────

@dataclass
class ChampionProfile:
    champion_id: int
    games: int
    wins: int
    expected_wins: float                 # Victoires attendues d'après les drafts jouées
    effect: float                        # δ (logit), déjà atténué
    games_by_role: dict[str, int] = field(default_factory=dict)


def player_profile(
    model: AdditiveDraftModel,
    puuid: str,
    db_path: Path = DB_PATH,
    tau: Optional[float] = None,
) -> dict[int, ChampionProfile]:
    """Profil par champion du joueur, à partir de ses parties présentes en base."""
    tau = tau or model.meta.get("personal_tau", DEFAULT_TAU)
    drafts = load_matches(db_path, puuid=puuid)
    if drafts.empty:
        return {}
    return champion_profiles(residuals(model, drafts, load_player_slots(db_path, puuid=puuid)), tau)


def champion_profiles(res: pd.DataFrame, tau: float) -> dict[int, ChampionProfile]:
    """Profils par champion à partir des résidus (y, p) d'un seul joueur."""
    profiles = {}
    for champ, grp in res.groupby("champion_id"):
        s = float((grp.y - grp.p).sum())
        v = float((grp.p * (1 - grp.p)).sum())
        profiles[int(champ)] = ChampionProfile(
            champion_id=int(champ),
            games=len(grp),
            wins=int(grp.y.sum()),
            expected_wins=float(grp.p.sum()),
            effect=s / (v + 1 / tau ** 2),
            games_by_role=grp.position.value_counts().to_dict(),
        )
    return profiles


# ── Collecte de l'historique ──────────────────────────────────────────────────

def fetch_history(
    conn: sqlite3.Connection,
    client,
    puuid: str,
    limit: int = 100,
    progress: Optional[Callable[[int, int], None]] = None,
) -> int:
    """
    Récupère les `limit` dernières parties Ranked Solo du joueur et insère en
    base celles qui manquent. `progress(fait, total)` est appelé après chaque match.

    Returns:
        Nombre de matchs nouvellement insérés.
    """
    from api.crawler import process_match  # Import local : dépendance API seulement ici
    from db.repository import match_exists

    match_ids: list[str] = []
    while len(match_ids) < limit:
        page = client.get_match_ids_by_puuid(
            puuid, queue=RANKED_SOLO_QUEUE, count=min(100, limit - len(match_ids)), start=len(match_ids)
        )
        if not page:
            break
        match_ids += page

    inserted = 0
    for i, match_id in enumerate(match_ids, 1):
        if not match_exists(conn, match_id) and process_match(conn, client, match_id):
            inserted += 1
        if progress:
            progress(i, len(match_ids))
    return inserted
