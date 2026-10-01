"""
additive_model.py — Modèle additif régularisé de victoire en draft.

    logit P(bleu gagne) = avantage côté bleu
                        + Σ_rôles  [main(bleu, rôle) − main(rouge, rôle)]
                        + Σ_rôles   lane(champion bleu vs champion rouge, rôle)
                        + Σ_duos   [duo(bleu) − duo(rouge)]
                        + Σ_champions bleus comp(c) · z(AD rouge) − Σ_rouges comp(c) · z(AD bleu)

  - main : force propre d'un champion à un rôle.
  - lane : avantage d'un champion sur son vis-à-vis direct (matchup).
  - duo  : synergie de deux champions alliés (bot, jungle-mid, jungle-top).
  - comp : sensibilité d'un champion au profil de dégâts de l'équipe adverse.
           z(AD) = part de dégâts physiques de l'équipe adverse, centrée-réduite.
           Malphite a un comp positif (meilleur contre une équipe AD), Kassadin
           un comp négatif. Un seul paramètre par champion, alimenté par toutes
           les parties : estimable bien avant les matchups paire par paire.
  - bal  : équilibre des dégâts de chaque équipe, b_lin·z + b_sq·z² sur la part AD
           de l'équipe (bleu − rouge). Une équipe full AD ou full AP perd ~5 pts.

L'encodage est antisymétrique (bleu +1, rouge −1) : échanger les équipes inverse
exactement la prédiction, hors avantage de côté.

Régularisation : une L2 unique (paramètre C) appliquée à des colonnes mises à
l'échelle par groupe. Multiplier les colonnes d'un groupe par s revient à lui
donner un a priori s fois plus large : un groupe à s=0.25 est 4x plus « tiré
vers zéro » que les effets principaux. C'est un shrinkage hiérarchique simple,
choisi par validation, qui s'ajuste tout seul au volume de données.

Les effets exportés sont déjà multipliés par leur échelle : un consommateur
(app, recommandation) n'a besoin que de `effects` et `intercept`.
"""
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from ml.draft_data import ROLES

DUO_PAIRS: tuple[tuple[str, str], ...] = (("BOTTOM", "UTILITY"), ("JUNGLE", "MIDDLE"), ("JUNGLE", "TOP"))
GROUPS: tuple[str, ...] = ("main", "lane", "duo", "comp", "bal")
BALANCE_KEYS: tuple[str, str] = ("b|lin", "b|sq")


# ── Clés de features ──────────────────────────────────────────────────────────
# Toutes les fonctions (entraînement, évaluation, recommandation) passent par
# ces trois fonctions : une seule définition des features.

def main_key(role: str, champ: int) -> str:
    return f"m|{role}|{champ}"


def lane_key(role: str, a: int, b: int) -> tuple[str, float]:
    """Clé canonique (plus petit id en premier) et signe : +1 si `a` est le premier."""
    lo, hi = (a, b) if a < b else (b, a)
    return f"l|{role}|{lo}|{hi}", (1.0 if a == lo else -1.0)


def duo_key(r1: str, r2: str, c1: int, c2: int) -> str:
    return f"d|{r1}|{r2}|{c1}|{c2}"


def comp_key(champ: int) -> str:
    return f"a|{champ}"


@dataclass
class DamageProfile:
    """
    Part de dégâts physiques par champion, et normalisation de la part d'une équipe.
    La part d'équipe est bornée aux quantiles 0.5 % / 99.5 % observés à
    l'entraînement : au-delà, les données plafonnent (~43 % de victoires) et un
    terme quadratique extrapolerait des pénalités irréalistes.
    """
    ad_share: dict[int, float]
    mean: float = 0.5
    std: float = 0.1
    lo: Optional[float] = None
    hi: Optional[float] = None

    def share(self, champ: int) -> float:
        return self.ad_share.get(champ, self.mean)

    def team_ad(self, df: pd.DataFrame, side: str) -> np.ndarray:
        return np.mean([df[f"{side}_{r}"].astype(int).map(self.share).to_numpy() for r in ROLES], axis=0)

    def z(self, team_ad):
        if self.lo is not None:
            team_ad = np.clip(team_ad, self.lo, self.hi)
        return (team_ad - self.mean) / self.std

    @classmethod
    def fit(cls, df: pd.DataFrame, ad_share: dict[int, float]) -> "DamageProfile":
        prof = cls(ad_share)
        teams = np.concatenate([prof.team_ad(df, "blue"), prof.team_ad(df, "red")])
        prof.lo, prof.hi = (float(q) for q in np.quantile(teams, [0.005, 0.995]))
        clipped = np.clip(teams, prof.lo, prof.hi)
        prof.mean, prof.std = float(clipped.mean()), float(clipped.std())
        return prof


def draft_terms(
    df: pd.DataFrame, damage: Optional[DamageProfile] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Décompose chaque draft complète en termes (ligne, clé, valeur, groupe).
    Les colonnes attendues sont blue_<ROLE> / red_<ROLE>. Les termes « comp »
    ne sont produits que si un profil de dégâts est fourni.
    """
    n = len(df)
    rows_idx = np.arange(n)
    rows, keys, vals, groups = [], [], [], []

    def add(keys_arr, vals_arr, group):
        rows.append(rows_idx)
        keys.append(np.asarray(keys_arr, dtype=object))
        vals.append(np.asarray(vals_arr, dtype=float))
        groups.append(np.full(n, group, dtype=object))

    for role in ROLES:
        blue = df[f"blue_{role}"].astype(int).to_numpy()
        red = df[f"red_{role}"].astype(int).to_numpy()
        add([main_key(role, c) for c in blue], np.ones(n), "main")
        add([main_key(role, c) for c in red], -np.ones(n), "main")
        lanes = [lane_key(role, a, b) for a, b in zip(blue, red)]
        add([k for k, _ in lanes], [s for _, s in lanes], "lane")

    for r1, r2 in DUO_PAIRS:
        for side, sign in (("blue", 1.0), ("red", -1.0)):
            c1 = df[f"{side}_{r1}"].astype(int).to_numpy()
            c2 = df[f"{side}_{r2}"].astype(int).to_numpy()
            add([duo_key(r1, r2, a, b) for a, b in zip(c1, c2)], np.full(n, sign), "duo")

    if damage is not None:
        z_blue, z_red = damage.z(damage.team_ad(df, "blue")), damage.z(damage.team_ad(df, "red"))
        for role in ROLES:
            add([comp_key(c) for c in df[f"blue_{role}"].astype(int)], z_red, "comp")
            add([comp_key(c) for c in df[f"red_{role}"].astype(int)], -z_blue, "comp")
        add(np.full(n, BALANCE_KEYS[0], dtype=object), z_blue - z_red, "bal")
        add(np.full(n, BALANCE_KEYS[1], dtype=object), z_blue ** 2 - z_red ** 2, "bal")

    return (np.concatenate(rows), np.concatenate(keys), np.concatenate(vals), np.concatenate(groups))


def recency_weights(game_creation_ms: pd.Series, half_life_days: float | None) -> np.ndarray:
    """Poids 0.5^(âge / demi-vie), normalisés à une moyenne de 1."""
    if not half_life_days:
        return np.ones(len(game_creation_ms))
    age_days = (game_creation_ms.max() - game_creation_ms.to_numpy()) / 86_400_000
    w = 0.5 ** (age_days / half_life_days)
    return w / w.mean()


# ── Modèle ────────────────────────────────────────────────────────────────────

@dataclass
class Hyperparams:
    C: float = 0.01
    lane_scale: float = 0.0
    duo_scale: float = 0.0
    comp_scale: float = 0.0
    bal_scale: float = 0.0
    half_life_days: float | None = None

    def scales(self) -> dict[str, float]:
        return {"main": 1.0, "lane": self.lane_scale, "duo": self.duo_scale,
                "comp": self.comp_scale, "bal": self.bal_scale}


@dataclass
class AdditiveDraftModel:
    intercept: float
    effects: dict[str, float]
    hyperparams: Hyperparams
    games: dict[str, dict[str, int]] = field(default_factory=dict)          # rôle → champion → matchs
    pick_rate: dict[str, dict[str, float]] = field(default_factory=dict)    # rôle → champion → part
    champion_names: dict[str, str] = field(default_factory=dict)
    meta: dict = field(default_factory=dict)
    damage: Optional[DamageProfile] = None

    # ── Entraînement ──────────────────────────────────────────────────────────

    @classmethod
    def fit(cls, df: pd.DataFrame, hp: Hyperparams,
            ad_share: Optional[dict[int, float]] = None) -> "AdditiveDraftModel":
        # Imports lourds réservés à l'entraînement (l'app et la CLI n'en ont pas besoin)
        from scipy import sparse
        from sklearn.linear_model import LogisticRegression

        uses_damage = hp.comp_scale > 0 or hp.bal_scale > 0
        damage = DamageProfile.fit(df, ad_share) if (ad_share and uses_damage) else None
        rows, keys, vals, groups = draft_terms(df, damage)
        scales = hp.scales()
        keep = np.array([scales[g] > 0 for g in groups])
        rows, keys, vals, groups = rows[keep], keys[keep], vals[keep], groups[keep]

        vocab, cols = np.unique(keys, return_inverse=True)
        col_scale = np.zeros(len(vocab))
        col_scale[cols] = [scales[g] for g in groups]
        X = sparse.csr_matrix((vals * col_scale[cols], (rows, cols)), shape=(len(df), len(vocab)))

        clf = LogisticRegression(C=hp.C, solver="lbfgs", max_iter=5000)
        clf.fit(X, df.blue_win.to_numpy(), sample_weight=recency_weights(df.game_creation, hp.half_life_days))

        effects = clf.coef_[0] * col_scale
        return cls(
            intercept=float(clf.intercept_[0]),
            effects={k: float(e) for k, e in zip(vocab, effects) if e != 0.0},
            hyperparams=hp,
            damage=damage,
        )

    # ── Prédiction sur drafts complètes ───────────────────────────────────────

    def predict_logit(self, df: pd.DataFrame) -> np.ndarray:
        rows, keys, vals, _ = draft_terms(df, self.damage)
        contrib = vals * np.array([self.effects.get(k, 0.0) for k in keys])
        return self.intercept + np.bincount(rows, weights=contrib, minlength=len(df))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.predict_logit(df)))

    # ── Statistiques descriptives (filtres de rareté, distribution des picks) ─

    def attach_stats(self, df: pd.DataFrame, names: dict[int, str]) -> None:
        """
        games : nombre de VRAIS matchs (pas de lignes augmentées) par champion et rôle.
        pick_rate : part des picks du rôle, pondérée par récence (sert d'a priori
        sur ce que l'adversaire / l'allié va jouer dans un slot encore vide).
        """
        w = recency_weights(df.game_creation, self.hyperparams.half_life_days)
        self.games, self.pick_rate = {}, {}
        for role in ROLES:
            champs = pd.concat([df[f"blue_{role}"], df[f"red_{role}"]]).astype(int)
            weights = np.concatenate([w, w])
            self.games[role] = {str(c): int(n) for c, n in champs.value_counts().items()}
            share = pd.Series(weights, index=champs.to_numpy()).groupby(level=0).sum()
            self.pick_rate[role] = {str(c): float(v) for c, v in (share / share.sum()).items()}
        self.champion_names = {str(cid): name for cid, name in names.items()}

    # ── Sérialisation ─────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        payload = {
            "meta": self.meta,
            "hyperparams": self.hyperparams.__dict__,
            "intercept": self.intercept,
            "effects": self.effects,
            "games": self.games,
            "pick_rate": self.pick_rate,
            "champion_names": self.champion_names,
            "damage": None if self.damage is None else {
                "ad_share": {str(c): v for c, v in self.damage.ad_share.items()},
                "mean": self.damage.mean,
                "std": self.damage.std,
                "lo": self.damage.lo,
                "hi": self.damage.hi,
            },
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "AdditiveDraftModel":
        d = json.loads(path.read_text(encoding="utf-8"))
        dmg = d.get("damage")
        return cls(
            intercept=d["intercept"],
            effects=d["effects"],
            hyperparams=Hyperparams(**d["hyperparams"]),
            games=d["games"],
            pick_rate=d["pick_rate"],
            champion_names=d["champion_names"],
            meta=d["meta"],
            damage=None if dmg is None else DamageProfile(
                {int(c): v for c, v in dmg["ad_share"].items()}, dmg["mean"], dmg["std"],
                dmg.get("lo"), dmg.get("hi"),
            ),
        )


# ── Sensibilités au profil adverse : sélection parcimonieuse ──────────────────

def fit_sparse_comp(
    model: AdditiveDraftModel, df: pd.DataFrame, fdr: float = 0.1,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Estime, pour chaque champion, sa sensibilité au profil de dégâts adverse
    (effet « comp ») à partir des résidus du modèle, et ne garde que les effets
    statistiquement établis.

    Pourquoi pas la L2 du groupe « comp » : la plupart des champions n'ont aucune
    sensibilité de ce type, quelques-uns en ont une forte (Malphite, Kassadin...).
    Une pénalité L2 commune suppose des effets petits et répartis : elle écrase
    les vrais effets forts. Ici :
      1. pente par champion = Σ résidu·z / Σ p(1−p)·z²  (un pas de Newton depuis 0),
         erreur type = 1/√(Σ p(1−p)·z²) ;
      2. sélection par Benjamini-Hochberg au taux de fausses découvertes `fdr` ;
      3. atténuation James-Stein des pentes retenues : b·(1 − se²/b²).

    Returns:
        (effets à ajouter au modèle, tableau de diagnostic par champion)
    """
    from scipy.stats import norm

    dmg = model.damage
    p_blue = model.predict_proba(df)
    z_blue, z_red = dmg.z(dmg.team_ad(df, "blue")), dmg.z(dmg.team_ad(df, "red"))
    res_blue = df.blue_win.to_numpy() - p_blue
    info_w = p_blue * (1 - p_blue)

    parts = []
    for role in ROLES:
        # Un champion bleu fait face au profil rouge, et inversement (résidu vu de son équipe)
        parts.append(pd.DataFrame({"c": df[f"blue_{role}"].to_numpy(), "s": res_blue * z_red, "i": info_w * z_red ** 2}))
        parts.append(pd.DataFrame({"c": df[f"red_{role}"].to_numpy(), "s": -res_blue * z_blue, "i": info_w * z_blue ** 2}))
    stats = pd.concat(parts).groupby("c").agg(S=("s", "sum"), I=("i", "sum"), n=("s", "size"))
    stats = stats[stats.I > 0]
    stats["slope"] = stats.S / stats.I
    stats["se"] = 1 / np.sqrt(stats.I)
    stats["p_value"] = 2 * norm.sf(np.abs(stats.slope / stats.se))

    # Benjamini-Hochberg : plus grand rang k tel que p_(k) ≤ k/m · fdr
    ranked = stats.sort_values("p_value")
    m = len(ranked)
    passing = np.nonzero(ranked.p_value.to_numpy() <= np.arange(1, m + 1) / m * fdr)[0]
    selected = set(ranked.index[: passing.max() + 1]) if len(passing) else set()

    stats["selected"] = stats.index.isin(selected)
    stats["effect"] = np.where(
        stats.selected, stats.slope * np.clip(1 - stats.se ** 2 / stats.slope ** 2, 0, 1), 0.0
    )
    effects = {comp_key(int(c)): float(e) for c, e in stats.effect.items() if e != 0.0}
    return effects, stats.sort_values("p_value")
