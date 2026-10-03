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
  - tier : écart de force champion × rôle propre à une tranche d'ELO (Gold-Platine,
           Émeraude-Diamant, Master+), ajouté à « main » pour les matchs dont le tier
           est connu. Fortement atténué vers 0 : il ne s'écarte de l'effet global que
           si la tranche fournit assez de matchs pour le justifier.

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
from ml.draft_data import ROLES, TIER_TO_BUCKET

DUO_PAIRS: tuple[tuple[str, str], ...] = (("BOTTOM", "UTILITY"), ("JUNGLE", "MIDDLE"), ("JUNGLE", "TOP"))
GROUPS: tuple[str, ...] = ("main", "lane", "duo", "comp", "bal", "tier")
BALANCE_KEYS: tuple[str, str] = ("b|lin", "b|sq")


# ── Clés de features ──────────────────────────────────────────────────────────
# Toutes les fonctions (entraînement, évaluation, recommandation) passent par
# ces trois fonctions : une seule définition des features.

def main_key(role: str, champ: int) -> str:
    return f"m|{role}|{champ}"


def tier_key(bucket: str, role: str, champ: int) -> str:
    return f"t|{bucket}|{role}|{champ}"


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

    if "source_tier" in df.columns:
        buckets = df.source_tier.map(TIER_TO_BUCKET).to_numpy()
        tagged = pd.notna(buckets)
        if tagged.any():
            rows_t = rows_idx[tagged]
            for role in ROLES:
                for side, sign in (("blue", 1.0), ("red", -1.0)):
                    champs = df[f"{side}_{role}"].astype(int).to_numpy()[tagged]
                    rows.append(rows_t)
                    keys.append(np.array([tier_key(b, role, c) for b, c in zip(buckets[tagged], champs)], dtype=object))
                    vals.append(np.full(len(rows_t), sign))
                    groups.append(np.full(len(rows_t), "tier", dtype=object))

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
    tier_scale: float = 0.0
    half_life_days: float | None = None

    def scales(self) -> dict[str, float]:
        return {"main": 1.0, "lane": self.lane_scale, "duo": self.duo_scale,
                "comp": self.comp_scale, "bal": self.bal_scale, "tier": self.tier_scale}


@dataclass
class AdditiveDraftModel:
    intercept: float
    effects: dict[str, float]
    hyperparams: Hyperparams
    games: dict[str, dict[str, int]] = field(default_factory=dict)          # rôle → champion → matchs
    pick_rate: dict[str, dict[str, float]] = field(default_factory=dict)    # rôle → champion → part
    pick_rate_by_tier: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)  # tranche → rôle → …
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
        pick_rate_by_tier : la même chose par tranche d'ELO (les champions joués
        diffèrent d'un niveau à l'autre ; les matchs de tier inconnu sont exclus).
        """
        w = recency_weights(df.game_creation, self.hyperparams.half_life_days)

        def shares(sub: pd.DataFrame, weights: np.ndarray, role: str) -> dict[str, float]:
            champs = pd.concat([sub[f"blue_{role}"], sub[f"red_{role}"]]).astype(int)
            share = pd.Series(np.concatenate([weights, weights]), index=champs.to_numpy()).groupby(level=0).sum()
            return {str(c): float(v) for c, v in (share / share.sum()).items()}

        self.games, self.pick_rate = {}, {}
        for role in ROLES:
            champs = pd.concat([df[f"blue_{role}"], df[f"red_{role}"]]).astype(int)
            self.games[role] = {str(c): int(n) for c, n in champs.value_counts().items()}
            self.pick_rate[role] = shares(df, w, role)

        self.pick_rate_by_tier = {}
        if "source_tier" in df.columns:
            buckets = df.source_tier.map(TIER_TO_BUCKET)
            for bucket in buckets.dropna().unique():
                mask = (buckets == bucket).to_numpy()
                self.pick_rate_by_tier[bucket] = {role: shares(df[mask], w[mask], role) for role in ROLES}
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
            "pick_rate_by_tier": self.pick_rate_by_tier,
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
            pick_rate_by_tier=d.get("pick_rate_by_tier", {}),
            champion_names=d["champion_names"],
            meta=d["meta"],
            damage=None if dmg is None else DamageProfile(
                {int(c): v for c, v in dmg["ad_share"].items()}, dmg["mean"], dmg["std"],
                dmg.get("lo"), dmg.get("hi"),
            ),
        )


# ── Sensibilités au profil adverse : lissage bayésien empirique ───────────────

def comp_slopes_by_period(model: AdditiveDraftModel, df: pd.DataFrame, n_periods: int = 4) -> pd.DataFrame:
    """
    Pente de chaque champion (sensibilité au profil de dégâts adverse) estimée
    séparément sur `n_periods` blocs chronologiques de taille égale.

    Pente = Σ résidu·z / Σ p(1−p)·z² (un pas de Newton depuis 0), variance = 1/Σ p(1−p)·z².

    Returns:
        DataFrame indexé par (champion, période) : colonnes S, I (somme du score
        et information de Fisher), n.
    """
    dmg = model.damage
    p_blue = model.predict_proba(df)
    z_blue, z_red = dmg.z(dmg.team_ad(df, "blue")), dmg.z(dmg.team_ad(df, "red"))
    res_blue = df.blue_win.to_numpy() - p_blue
    info_w = p_blue * (1 - p_blue)
    order = np.argsort(df.game_creation.to_numpy(), kind="stable")
    period = np.empty(len(df), dtype=int)
    for k, idx in enumerate(np.array_split(order, n_periods)):
        period[idx] = k

    parts = []
    for role in ROLES:
        # Un champion bleu fait face au profil rouge, et inversement (résidu vu de son équipe)
        parts.append(pd.DataFrame({"c": df[f"blue_{role}"].to_numpy(), "t": period,
                                   "s": res_blue * z_red, "i": info_w * z_red ** 2}))
        parts.append(pd.DataFrame({"c": df[f"red_{role}"].to_numpy(), "t": period,
                                   "s": -res_blue * z_blue, "i": info_w * z_blue ** 2}))
    per = pd.concat(parts).groupby(["c", "t"]).agg(S=("s", "sum"), I=("i", "sum"), n=("s", "size"))
    return per[per.I > 0]


def fit_comp_effects(
    model: AdditiveDraftModel, df: pd.DataFrame, n_periods: int = 4,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Sensibilité de chaque champion au profil de dégâts adverse, lissée selon la
    force des preuves. Aucun seuil, aucun choix par champion : trois étapes
    calculées sur les 173 champions à la fois.

    1. Stabilité dans le temps (méta-analyse à effets aléatoires, DerSimonian-Laird) :
       les pentes par période b_ct sont combinées en supposant qu'un vrai effet
       peut dériver d'une période à l'autre avec une variance ω², mesurée sur
       l'ensemble des champions. Une dérive fréquente augmente l'incertitude de
       tous ; un champion dont l'effet change de signe obtient une pente
       combinée faible et incertaine.
    2. Lissage bayésien empirique (mélange « pas d'effet / vrai effet ») :
          vrai effet θ_c = 0 avec probabilité 1 − π, θ_c ~ N(0, τ²) sinon.
       π et τ sont estimés par maximum de vraisemblance sur toutes les pentes.
    3. Effet retenu = poids × b, avec poids = P(vrai effet | données) · τ² / (τ² + se²)
       (moyenne a posteriori) : les preuves fortes passent presque entières,
       les faibles partiellement, le bruit tombe à ~0. Le poids reste bien
       défini même quand les données ne montrent aucun effet (τ → 0, poids → 0),
       contrairement à P(vrai effet) seule, qui devient alors arbitraire.

    Returns:
        (effets à ajouter au modèle, tableau par champion trié par |effet| ;
        attrs : pi, tau, omega)
    """
    from scipy.optimize import minimize
    from scipy.stats import norm

    per = comp_slopes_by_period(model, df, n_periods)
    per["b"] = per.S / per.I

    # 1. Variance de dérive ω² entre périodes, poolée sur tous les champions
    g = per.groupby(level="c")
    w_sum = g.I.sum()
    b_fixed = g.S.sum() / w_sum                      # Σ w·b / Σ w, avec w = I et w·b = S
    q = (per.I * (per.b - b_fixed.reindex(per.index, level="c")) ** 2).groupby(level="c").sum()
    dof = g.size() - 1
    c_term = w_sum - (per.I ** 2).groupby(level="c").sum() / w_sum
    omega2 = max(0.0, float((q.sum() - dof.sum()) / c_term[dof > 0].sum())) if c_term[dof > 0].sum() > 0 else 0.0

    w_re = 1 / (1 / per.I + omega2)
    stats = pd.DataFrame({
        "slope": (w_re * per.b).groupby(level="c").sum() / w_re.groupby(level="c").sum(),
        "se": 1 / np.sqrt(w_re.groupby(level="c").sum()),
        "n": g.n.sum(),
    })

    # 2. Mélange : maximum de vraisemblance sur (π, τ)
    b, s2 = stats.slope.to_numpy(), stats.se.to_numpy() ** 2

    def neg_loglik(params):
        pi, tau2 = 1 / (1 + np.exp(-params[0])), np.exp(params[1])
        like = (1 - pi) * norm.pdf(b, 0, np.sqrt(s2)) + pi * norm.pdf(b, 0, np.sqrt(tau2 + s2))
        return -np.sum(np.log(like + 1e-300))

    best = min(
        (minimize(neg_loglik, x0, method="Nelder-Mead") for x0 in ([-2.0, np.log(0.01)], [0.0, np.log(0.003)])),
        key=lambda r: r.fun,
    )
    pi, tau2 = float(1 / (1 + np.exp(-best.x[0]))), float(np.exp(best.x[1]))

    # 3. Moyenne a posteriori
    real = pi * norm.pdf(b, 0, np.sqrt(tau2 + s2))
    null = (1 - pi) * norm.pdf(b, 0, np.sqrt(s2))
    stats["weight"] = real / (real + null) * tau2 / (tau2 + s2)
    stats["effect"] = stats.weight * b
    stats.attrs.update(pi=pi, tau=float(np.sqrt(tau2)), omega=float(np.sqrt(omega2)))

    effects = {comp_key(int(c)): float(e) for c, e in stats.effect.items() if abs(e) > 1e-6}
    return effects, stats.reindex(stats.effect.abs().sort_values(ascending=False).index)
