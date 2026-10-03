"""
lane_stats.py — Matchups appris sur les statistiques de lane (tâche auxiliaire).

Le résultat d'une partie (1 bit très bruité) ne suffit pas à estimer les
~15 000 matchups par rôle. On les apprend sur un signal continu mesuré à chaque
partie : l'écart de **part d'or dans l'équipe** entre les deux laners,
    y = log(or bleu / or équipe bleue) − log(or rouge / or équipe rouge).
Rapporté au total de chaque équipe, il est quasi indépendant du résultat
(corrélation ~0 avec la victoire) : il mesure la domination de lane.

Étape 1 : par rôle, régression ridge de y sur les forces champion × rôle et
les matchups (encodage antisymétrique, mêmes clés que le modèle additif).
Étape 2 : le score de matchups d'une draft (somme des 5 lanes) est ajouté au
logit du modèle de victoire avec un coefficient β estimé sur des scores
CROSS-FITTÉS (chaque match prédit par des modèles de lane entraînés sans lui).
Comme le score est une somme de termes de paire, β·score se réécrit en effets
de lane du modèle additif : le recommandeur les traite sans modification.

Validé hors échantillon (experiments/lane_stats_matchups.py) : +0,0009 à
+0,0010 de log-loss sur deux fenêtres de parties futures (IC ±0,0005).
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
from ml.draft_data import ROLES, connect_read_only

ALPHA_GRID: tuple[float, ...] = (3.0, 10.0, 30.0, 100.0)
PAIR_SCALE_GRID: tuple[float, ...] = (0.0, 0.3, 1.0, 3.0)
Settings = dict[str, tuple[float, float]]   # rôle → (alpha, échelle des matchups)


def load_lane_gold(df: pd.DataFrame, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Or de chaque slot, colonnes (side, role), aligné sur df.index."""
    conn = connect_read_only(db_path)
    p = pd.read_sql_query(
        "SELECT match_id, team_id, position, gold_earned AS g FROM participants "
        "WHERE position IN ('TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY')", conn)
    conn.close()
    p["side"] = p.team_id.map({100: "blue", 200: "red"})
    return p.pivot_table(index="match_id", columns=["side", "position"], values="g", aggfunc="first") \
            .reindex(df.index).clip(lower=1)


def lane_targets(gold: pd.DataFrame, kind: str = "share") -> pd.DataFrame:
    """
    Cible par match et par rôle.
      kind="share" : écart de part d'or dans l'équipe (domination de lane, indépendant du résultat) ;
      kind="raw"   : log du rapport d'or entre les deux laners (lié au résultat).
    """
    if kind == "raw":
        out = {r: np.log(gold[("blue", r)] / gold[("red", r)]) for r in ROLES}
    else:
        team = {s: sum(gold[(s, r)] for r in ROLES) for s in ("blue", "red")}
        out = {r: np.log(gold[("blue", r)] / team["blue"]) - np.log(gold[("red", r)] / team["red"]) for r in ROLES}
    return pd.DataFrame(out).fillna(0.0)


class LaneModel:
    """Ridge par rôle : cible ~ force(bleu) − force(rouge) + matchup(bleu, rouge)."""

    def __init__(self, role: str, alpha: float, pair_scale: float) -> None:
        self.role, self.alpha, self.pair_scale = role, alpha, pair_scale

    def _design(self, df: pd.DataFrame, fit: bool = False) -> sparse.csr_matrix:
        from ml.additive_model import lane_key, main_key

        blue = df[f"blue_{self.role}"].astype(int).to_numpy()
        red = df[f"red_{self.role}"].astype(int).to_numpy()
        lanes = [lane_key(self.role, a, b) for a, b in zip(blue, red)]
        keys = [main_key(self.role, c) for c in blue] + [main_key(self.role, c) for c in red] + [k for k, _ in lanes]
        vals = np.r_[np.ones(len(df)), -np.ones(len(df)), [s * self.pair_scale for _, s in lanes]]
        rows = np.r_[np.arange(len(df)), np.arange(len(df)), np.arange(len(df))]
        if fit:
            self.vocab = {k: i for i, k in enumerate(sorted(set(keys)))}
        cols = np.array([self.vocab.get(k, -1) for k in keys])
        keep = cols >= 0
        return sparse.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(len(df), len(self.vocab)))

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "LaneModel":
        from sklearn.linear_model import Ridge

        self.ridge = Ridge(alpha=self.alpha, solver="sparse_cg").fit(self._design(df, fit=True), y)
        is_pair = np.array([k.startswith("l|") for k in sorted(self.vocab, key=self.vocab.get)])
        self.coef_main = np.where(is_pair, 0.0, self.ridge.coef_)
        self.coef_pair = np.where(is_pair, self.ridge.coef_, 0.0)
        return self

    def predict_parts(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        X = self._design(df)
        return X @ self.coef_main, X @ self.coef_pair

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        main, pair = self.predict_parts(df)
        return self.ridge.intercept_ + main + pair

    def pair_effects(self) -> dict[str, float]:
        """Contribution de chaque matchup à la cible, par clé de lane (signe : + = avantage au 1er id)."""
        return {k: float(self.coef_pair[i] * self.pair_scale) for k, i in self.vocab.items()
                if k.startswith("l|") and self.coef_pair[i] != 0}


def tune_lane_settings(train: pd.DataFrame, targets: pd.DataFrame, val_frac: float = 0.15) -> Settings:
    """(alpha, échelle des matchups) par rôle, réglés sur les matchs les plus récents de `train`."""
    cut = int(len(train) * (1 - val_frac))
    inner_tr, inner_va = train.iloc[:cut], train.iloc[cut:]
    settings = {}
    for role in ROLES:
        y_tr, y_va = targets.loc[inner_tr.index, role].to_numpy(), targets.loc[inner_va.index, role].to_numpy()
        settings[role] = min(
            ((np.mean((LaneModel(role, a, s).fit(inner_tr, y_tr).predict(inner_va) - y_va) ** 2), a, s)
             for a in ALPHA_GRID for s in PAIR_SCALE_GRID),
            key=lambda t: t[0],
        )[1:]
    return settings


def fit_lane_models(train: pd.DataFrame, targets: pd.DataFrame, settings: Settings) -> dict[str, LaneModel]:
    return {r: LaneModel(r, *settings[r]).fit(train, targets.loc[train.index, r].to_numpy()) for r in ROLES}


def pair_score(models: dict[str, LaneModel], df: pd.DataFrame) -> np.ndarray:
    """Score de matchups d'une draft : somme des contributions de paire des 5 lanes (vue bleue)."""
    return np.sum([models[r].predict_parts(df)[1] for r in ROLES], axis=0)


def crossfit_pair_score(train: pd.DataFrame, targets: pd.DataFrame, settings: Settings,
                        n_folds: int = 5) -> np.ndarray:
    """Score de matchups de chaque match de `train`, prédit par des modèles entraînés sans lui."""
    from sklearn.model_selection import KFold

    out = np.zeros(len(train))
    for fit_idx, pred_idx in KFold(n_folds, shuffle=True, random_state=0).split(train):
        out[pred_idx] = pair_score(fit_lane_models(train.iloc[fit_idx], targets, settings), train.iloc[pred_idx])
    return out


def fit_transfer(offset: np.ndarray, score: np.ndarray, y: np.ndarray) -> float:
    """β tel que logit = offset + β·score maximise la vraisemblance (régression logistique à offset)."""
    nll = lambda b: np.sum(np.logaddexp(0, offset + b[0] * score) - y * (offset + b[0] * score))
    return float(minimize(nll, np.zeros(1), method="BFGS").x[0])


def transferred_effects(models: dict[str, LaneModel], beta: float) -> dict[str, float]:
    """Effets de lane à ajouter au modèle additif : β × contribution de chaque matchup."""
    effects: dict[str, float] = {}
    for model in models.values():
        for key, value in model.pair_effects().items():
            effects[key] = effects.get(key, 0.0) + beta * value
    return effects


def lane_stat_layer(
    train: pd.DataFrame, targets: pd.DataFrame, offset: np.ndarray,
    settings: Optional[Settings] = None, n_folds: int = 5,
) -> tuple[dict[str, float], float, Settings]:
    """
    Couche complète sur `train` : réglage (si besoin), β par cross-fitting, puis
    effets transférés calculés avec les modèles de lane entraînés sur tout `train`.

    Returns:
        (effets de lane à ajouter, β, réglages)
    """
    settings = settings or tune_lane_settings(train, targets)
    beta = fit_transfer(offset, crossfit_pair_score(train, targets, settings, n_folds), train.blue_win.to_numpy())
    return transferred_effects(fit_lane_models(train, targets, settings), beta), beta, settings
