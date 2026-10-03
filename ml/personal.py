"""
personal.py — Personnalisation des recommandations par l'historique du joueur.

Le modèle global mesure la force d'un champion « joué par ceux qui le
choisissent ». Ton niveau sur ce champion peut être très différent. Pour
chaque partie de ton historique, le modèle donne la proba de victoire p
attendue POUR CETTE DRAFT ; le résidu (victoire − p) isole donc ton apport
personnel, indépendamment de la qualité des drafts que tu as eues.

A priori : le coût (ou le gain) de familiarité. La force d'un champion est
mesurée sur ceux qui le choisissent, souvent des habitués. Le modèle mesure à
chaque entraînement (estimate_familiarity), à joueur égal, l'écart entre ce
qu'un joueur obtient et ce que la draft laissait attendre selon le nombre de
parties qu'il avait déjà sur ce champion : aucune, 1-2, 3 et plus. Cet écart
est plus fort pour les picks de niche (un rôle qui représente moins de 10 % des
parties du champion : Karthus ADC...), joués surtout par des spécialistes.
Il sert de moyenne a priori μ du décalage personnel.

Pour un champion c joué n fois, ton décalage de logit est alors :

    δ_c = μ + (Σ (y − p) − μ · Σ p(1 − p)) / (Σ p(1 − p) + 1/τ²)

Estimation a posteriori (un pas de Newton depuis μ) avec un a priori N(μ, τ²) :
sans partie, δ = μ (le coût d'un champion jamais joué) ; avec beaucoup de
parties, δ tend vers ton écart réel. τ, l'écart typique réel entre joueurs sur
un même champion, est lui aussi mesuré sur la base (estimate_tau).
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

FAMILIARITY_BUCKETS: tuple[str, ...] = ("none", "few", "regular")   # 0, 1-2, 3+ parties antérieures
NICHE_ROLE_SHARE: float = 0.10       # Pick de niche : ce rôle fait moins de 10 % des parties du champion
FAMILIARITY_MIN_HISTORY: int = 10    # Joueurs dont la base contient au moins 10 parties antérieures


def familiarity_bucket(n_games: int) -> str:
    return "none" if n_games == 0 else ("few" if n_games <= 2 else "regular")


def role_shares(model: AdditiveDraftModel) -> dict[tuple[int, str], float]:
    """(champion, rôle) → part des parties du champion jouées à ce rôle."""
    totals: dict[int, int] = {}
    for role_games in model.games.values():
        for c, n in role_games.items():
            totals[int(c)] = totals.get(int(c), 0) + n
    return {
        (int(c), role): n / totals[int(c)]
        for role, role_games in model.games.items() for c, n in role_games.items()
    }


def is_niche(shares: dict[tuple[int, str], float], champ: int, role: str) -> bool:
    return shares.get((champ, role), 0.0) < NICHE_ROLE_SHARE


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


def estimate_familiarity(res: pd.DataFrame, shares: dict[tuple[int, str], float]) -> dict:
    """
    Écart (logit) entre résultat réel et prédiction de la draft selon la
    familiarité du joueur avec le champion, à joueur égal.

    `res` : résidus par (joueur, match) avec une colonne `t` (date). Seules les
    parties ANTÉRIEURES comptent pour la familiarité (sinon biais de survie :
    après une défaite sur un nouveau champion on le rejoue moins), et seuls les
    joueurs dont la base contient au moins FAMILIARITY_MIN_HISTORY parties
    antérieures sont retenus. Le résidu moyen de chaque joueur est retiré (son
    niveau général), puis converti en logit (÷ p(1−p) moyen).

    Returns:
        {"common" | "niche": {bucket: {"effect", "ci95", "n"}}}
    """
    res = res.sort_values("t")
    res = res.assign(
        r=res.y - res.p,
        prior_total=res.groupby("puuid").cumcount(),
        prior_same=res.groupby(["puuid", "champion_id"]).cumcount(),
    )
    h = res[res.prior_total >= FAMILIARITY_MIN_HISTORY].copy()
    h["r_within"] = h.r - h.groupby("puuid").r.transform("mean")
    h["kind"] = ["niche" if is_niche(shares, c, pos) else "common" for c, pos in zip(h.champion_id, h.position)]
    h["bucket"] = h.prior_same.map(familiarity_bucket)
    out: dict = {"common": {}, "niche": {}}
    for (kind, bucket), g in h.groupby(["kind", "bucket"]):
        scale = float((g.p * (1 - g.p)).mean())
        out[kind][bucket] = {
            "effect": float(g.r_within.mean() / scale),
            "ci95": float(1.96 * g.r_within.std(ddof=1) / np.sqrt(len(g)) / scale),
            "n": int(len(g)),
        }
    return out


# ── Profil d'un joueur ────────────────────────────────────────────────────────

@dataclass
class ChampionProfile:
    champion_id: int
    games: int
    wins: int
    expected_wins: float                 # Victoires attendues d'après les drafts jouées
    score: float                         # Σ (y − p) : victoires au-delà de l'attendu
    information: float                   # Σ p(1 − p)
    games_by_role: dict[str, int] = field(default_factory=dict)


def player_profile(
    model: AdditiveDraftModel,
    puuid: str,
    db_path: Path = DB_PATH,
) -> dict[int, ChampionProfile]:
    """Profil par champion du joueur, à partir de ses parties présentes en base."""
    drafts = load_matches(db_path, puuid=puuid)
    if drafts.empty:
        return {}
    return champion_profiles(residuals(model, drafts, load_player_slots(db_path, puuid=puuid)))


def champion_profiles(res: pd.DataFrame) -> dict[int, ChampionProfile]:
    """Statistiques suffisantes par champion à partir des résidus (y, p) d'un seul joueur."""
    return {
        int(champ): ChampionProfile(
            champion_id=int(champ),
            games=len(grp),
            wins=int(grp.y.sum()),
            expected_wins=float(grp.p.sum()),
            score=float((grp.y - grp.p).sum()),
            information=float((grp.p * (1 - grp.p)).sum()),
            games_by_role=grp.position.value_counts().to_dict(),
        )
        for champ, grp in res.groupby("champion_id")
    }


def prior_mean(model: AdditiveDraftModel, shares, champ: int, role: str, n_games: int) -> float:
    """Coût / gain de familiarité mesuré (logit) ; 0 si le modèle n'en embarque pas."""
    fam = model.meta.get("familiarity", {})
    kind = "niche" if is_niche(shares, champ, role) else "common"
    return float(fam.get(kind, {}).get(familiarity_bucket(n_games), {}).get("effect", 0.0))


def personal_offsets(
    model: AdditiveDraftModel,
    role: str,
    profiles: Optional[dict[int, ChampionProfile]] = None,
    tau: Optional[float] = None,
) -> dict[int, float]:
    """
    Décalage personnel δ (logit) de chaque champion jouable à `role`.

    Sans profil (profiles=None), tous les champions sont traités comme « jamais
    joués » : δ = coût d'inexpérience mesuré, ce que vaut le pick pour
    quelqu'un qui ne le joue pas.
    """
    tau = tau or model.meta.get("personal_tau", DEFAULT_TAU)
    shares = role_shares(model)
    offsets = {}
    for c in model.games.get(role, {}):
        champ = int(c)
        prof = (profiles or {}).get(champ)
        mu = prior_mean(model, shares, champ, role, prof.games if prof else 0)
        if prof is None:
            offsets[champ] = mu
        else:
            offsets[champ] = mu + (prof.score - mu * prof.information) / (prof.information + 1 / tau ** 2)
    return offsets


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
