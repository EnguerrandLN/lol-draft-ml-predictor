"""
lane_stats_extended.py — Prolongements de la piste 1 : d'autres cibles
auxiliaires et les synergies de duo.

La couche actuelle (ml/lane_stats.py) apprend les matchups sur l'écart de part
d'or entre les deux laners. Deux prolongements, même méthode :
  1. D'autres statistiques comme cibles de lane : part du farm (sbires tués),
     part des dégâts aux champions, part des morts de l'équipe. Chacune donne
     un score de matchups par draft ; tous les coefficients de transfert sont
     estimés ensemble.
  2. Les synergies de duo apprises sur la part d'or du duo, par exemple
     (ADC + support) / équipe, bleu − rouge. Ridge : forces des deux champions
     + effet propre de la paire ; transfert vers les effets de duo du modèle
     additif (ADC-support, jungle-mid, jungle-top).

Toutes les cibles sont des parts de l'équipe : quasi indépendantes du
résultat (une équipe qui gagne a plus de tout, pas une plus grosse part).

Protocole (comme lane_stats_matchups.py) : deux fenêtres de parties futures
(les 15 % les plus récents, les 15 % d'avant) ; réglages ridge par validation
interne sur la cible ; scores d'entraînement cross-fittés (chaque match prédit
par des modèles entraînés sans lui) ; coefficients de transfert par régression
logistique à offset. Gain de log-loss par rapport à la référence = modèle
actuel avec ses matchups appris sur la part d'or. IC apparié.

Usage : python experiments/lane_stats_extended.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR
from ml.additive_model import DUO_PAIRS, AdditiveDraftModel, duo_key, main_key
from ml.draft_data import ROLES, connect_read_only, load_champion_ad_share, load_champion_names, load_matches
from ml.lane_stats import ALPHA_GRID, PAIR_SCALE_GRID, LaneModel
from ml.train import log_losses, with_comp

# Statistique → (colonne de la base, lissage ajouté avant le log)
STATS = {
    "or": ("gold_earned", 1.0),
    "farm": ("total_minions_killed", 1.0),
    "dégâts": ("total_damage_dealt_to_champions", 100.0),
    "morts": ("deaths", 1.0),
}
DUOS = {f"{r1}+{r2}": (r1, r2) for r1, r2 in DUO_PAIRS}


# ── Cibles ────────────────────────────────────────────────────────────────────

def load_slot_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Par statistique : valeur de chaque slot, colonnes (side, role), alignée sur df.index."""
    conn = connect_read_only()
    cols = ", ".join(col for col, _ in STATS.values())
    p = pd.read_sql_query(
        f"SELECT match_id, team_id, position, {cols} FROM participants "
        "WHERE position IN ('TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY')", conn)
    conn.close()
    p["side"] = p.team_id.map({100: "blue", 200: "red"})
    return {name: p.pivot_table(index="match_id", columns=["side", "position"], values=col, aggfunc="first")
                   .reindex(df.index).clip(lower=0)
            for name, (col, _) in STATS.items()}


def share_targets(stat: pd.DataFrame, groups: dict[str, tuple[str, ...]], smooth: float) -> pd.DataFrame:
    """Par groupe de rôles : log(part de l'équipe) bleu − rouge."""
    team = {s: sum(stat[(s, r)] for r in ROLES) + len(ROLES) * smooth for s in ("blue", "red")}
    share = lambda s, roles: np.log((sum(stat[(s, r)] for r in roles) + len(roles) * smooth) / team[s])
    return pd.DataFrame({g: share("blue", roles) - share("red", roles) for g, roles in groups.items()}).fillna(0.0)


# ── Modèle de duo ─────────────────────────────────────────────────────────────

class DuoModel:
    """Ridge : cible ~ forces des deux champions de chaque duo + effet propre de la paire (bleu − rouge)."""

    def __init__(self, duo: str, alpha: float, pair_scale: float) -> None:
        self.duo, self.alpha, self.pair_scale = duo, alpha, pair_scale
        self.r1, self.r2 = DUOS[duo]

    def _design(self, df: pd.DataFrame, fit: bool = False) -> sparse.csr_matrix:
        n = len(df)
        keys, vals = [], []
        for side, sign in (("blue", 1.0), ("red", -1.0)):
            c1, c2 = df[f"{side}_{self.r1}"].astype(int).to_numpy(), df[f"{side}_{self.r2}"].astype(int).to_numpy()
            keys += [main_key(self.r1, c) for c in c1] + [main_key(self.r2, c) for c in c2]
            keys += [duo_key(self.r1, self.r2, a, b) for a, b in zip(c1, c2)]
            vals += [np.full(2 * n, sign), np.full(n, sign * self.pair_scale)]
        vals = np.concatenate(vals)
        rows = np.tile(np.arange(n), 6)
        if fit:
            self.vocab = {k: i for i, k in enumerate(sorted(set(keys)))}
        cols = np.array([self.vocab.get(k, -1) for k in keys])
        keep = cols >= 0
        return sparse.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, len(self.vocab)))

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> "DuoModel":
        from sklearn.linear_model import Ridge

        self.ridge = Ridge(alpha=self.alpha, solver="sparse_cg").fit(self._design(df, fit=True), y)
        is_pair = np.array([k.startswith("d|") for k in sorted(self.vocab, key=self.vocab.get)])
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
        return {k: float(self.coef_pair[i] * self.pair_scale) for k, i in self.vocab.items()
                if k.startswith("d|") and self.coef_pair[i] != 0}


# ── Couche générique : réglage, scores cross-fittés, transfert ────────────────

def tune(factory, unit: str, train: pd.DataFrame, y: pd.Series, val_frac: float = 0.15) -> tuple[float, float]:
    """(alpha, échelle de paire) minimisant l'erreur sur la cible, validation sur les matchs récents de `train`."""
    cut = int(len(train) * (1 - val_frac))
    tr, va = train.iloc[:cut], train.iloc[cut:]
    y_tr, y_va = y.loc[tr.index].to_numpy(), y.loc[va.index].to_numpy()
    return min(((np.mean((factory(unit, a, s).fit(tr, y_tr).predict(va) - y_va) ** 2), a, s)
                for a in ALPHA_GRID for s in PAIR_SCALE_GRID), key=lambda t: t[0])[1:]


class Layer:
    """Un score de paires par draft (somme sur plusieurs rôles ou un duo), appris sur une cible auxiliaire."""

    def __init__(self, factory, units: list[str], targets: pd.DataFrame) -> None:
        self.factory, self.units, self.targets = factory, units, targets

    def fit_models(self, train: pd.DataFrame) -> dict:
        return {u: self.factory(u, *self.settings[u]).fit(train, self.targets.loc[train.index, u].to_numpy())
                for u in self.units}

    def score(self, models: dict, df: pd.DataFrame) -> np.ndarray:
        return np.sum([m.predict_parts(df)[1] for m in models.values()], axis=0)

    def run(self, train: pd.DataFrame, test: pd.DataFrame, n_folds: int = 5) -> tuple[np.ndarray, np.ndarray]:
        """(score cross-fitté sur train, score sur test), réduits par l'écart-type du score d'entraînement."""
        from sklearn.model_selection import KFold

        self.settings = {u: tune(self.factory, u, train, self.targets[u]) for u in self.units}
        s_tr = np.zeros(len(train))
        for fit_idx, pred_idx in KFold(n_folds, shuffle=True, random_state=0).split(train):
            s_tr[pred_idx] = self.score(self.fit_models(train.iloc[fit_idx]), train.iloc[pred_idx])
        self.models = self.fit_models(train)
        s_te = self.score(self.models, test)
        sd = s_tr.std() or 1.0
        return s_tr / sd, s_te / sd


def fit_offset_logit(off: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize

    nll = lambda b: np.sum(np.logaddexp(0, off + x @ b) - y * (off + x @ b))
    return minimize(nll, np.zeros(x.shape[1]), method="BFGS").x


# ── Évaluation ────────────────────────────────────────────────────────────────

def evaluate_window(train, test, hp, ad_share, targets, names) -> None:
    base = AdditiveDraftModel.fit(train, hp, ad_share)
    base = with_comp(base, train) if base.damage is not None else base
    off_tr, off_te = base.predict_logit(train), base.predict_logit(test)
    y_tr, y_te = train.blue_win.to_numpy(), test.blue_win.to_numpy()
    print(f"  entraînement {len(train)} matchs → test {len(test)} matchs")

    scores, layers = {}, {}
    for stat in STATS:
        layers[f"lane {stat}"] = Layer(LaneModel, list(ROLES), targets[("lane", stat)])
    for duo in DUOS:
        layers[f"duo {duo}"] = Layer(DuoModel, [duo], targets[("duo", "or")])
    for label, layer in layers.items():
        scores[label] = layer.run(train, test)
        settings = ", ".join(f"{u[:3]} α={a:g} s={s:g}" for u, (a, s) in layer.settings.items())
        print(f"    {label:<22} ({settings})", flush=True)

    def gain(labels):
        x_tr = np.column_stack([scores[l][0] for l in labels])
        x_te = np.column_stack([scores[l][1] for l in labels])
        beta = fit_offset_logit(off_tr, x_tr, y_tr)
        return log_losses(y_te, 1 / (1 + np.exp(-(off_te + x_te @ beta)))), beta

    ref, beta_ref = gain(["lane or"])
    print(f"    référence (matchups sur la part d'or) : coef {np.round(beta_ref, 3)}, gain vs modèle sans couche "
          f"{(log_losses(y_te, 1 / (1 + np.exp(-off_te))) - ref).mean():+.5f}")
    others = [l for l in layers if l != "lane or"]
    variants = {f"+ {l}": ["lane or", l] for l in others}
    variants["+ autres cibles de lane"] = ["lane or"] + [l for l in others if l.startswith("lane")]
    variants["+ tous les duos"] = ["lane or"] + [l for l in others if l.startswith("duo")]
    variants["+ tout"] = list(layers)
    for label, labels in variants.items():
        ll, beta = gain(labels)
        d = ref - ll
        print(f"    {label:<28} gain {d.mean():+.5f} ± {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.5f}   "
              f"coef {np.round(beta, 3)}")
    varying = [l for l in layers if scores[l][0].std() > 0]   # Score nul si l'effet de paire est réglé à 0
    corr = np.corrcoef(np.column_stack([scores[l][0] for l in varying]).T)
    print("    corrélations des scores (entraînement) : " + ", ".join(
        f"{a}/{b} {corr[i, j]:+.2f}" for i, a in enumerate(varying) for j, b in enumerate(varying) if i < j
        and abs(corr[i, j]) > 0.3))
    for duo in DUOS:
        effects = sorted(layers[f"duo {duo}"].models[duo].pair_effects().items(), key=lambda kv: -kv[1])
        fmt = lambda kv: f"{names.get(int(kv[0].split('|')[3]), '?')}+{names.get(int(kv[0].split('|')[4]), '?')} {kv[1]:+.3f}"
        print(f"    duo {duo} : meilleures paires {', '.join(map(fmt, effects[:5]))} ; "
              f"pires {', '.join(map(fmt, effects[-3:]))}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    hp = AdditiveDraftModel.load(DATA_DIR / "additive_model.json").hyperparams
    ad_share = load_champion_ad_share()
    names = load_champion_names()
    df = load_matches()
    stats = load_slot_stats(df)
    targets = {("lane", s): share_targets(stats[s], {r: (r,) for r in ROLES}, STATS[s][1]) for s in STATS}
    targets[("duo", "or")] = share_targets(stats["or"], {d: rs for d, rs in DUOS.items()}, STATS["or"][1])
    for s in STATS:
        corr = targets[("lane", s)].corrwith(df.blue_win.astype(float))
        print(f"Cible « {s} » : corrélation avec la victoire par rôle {np.round(corr.to_numpy(), 2)}")
    print(f"Duos : corrélation avec la victoire {np.round(targets[('duo', 'or')].corrwith(df.blue_win.astype(float)).to_numpy(), 2)}")

    n = len(df)
    print(f"\n{n} matchs. Gain de log-loss par rapport à la référence (+ = mieux) :")
    for label, (start, end) in {"Fenêtre récente (15 % les plus récents)": (int(n * 0.85), n),
                                "Fenêtre précédente (15 % d'avant)": (int(n * 0.70), int(n * 0.85))}.items():
        print(f"\n{label}")
        evaluate_window(df.iloc[:start], df.iloc[start:end], hp, ad_share, targets, names)


if __name__ == "__main__":
    main()
