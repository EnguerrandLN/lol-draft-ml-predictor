"""
lane_stats_matchups.py — Apprendre les matchups sur les statistiques de lane (tâche auxiliaire).

Le résultat d'une partie (1 bit, très bruité) ne suffit pas à estimer les
matchups : ~15 000 paires par rôle. L'écart d'or entre les deux laners est un
signal continu, mesuré à chaque partie. Deux cibles par lane :
  - « écart d'or »  : log(or du laner bleu / or du laner rouge). Fortement lié
    au résultat (corrélation ~0,55) : c'est un écart au score, plus informatif
    qu'une victoire (comme l'écart de points en sport) ;
  - « part d'or »   : log(part de l'or de son équipe) bleu − rouge. Quasi
    indépendant du résultat (corrélation ~0) : domination de lane « pure ».

Étape 1 (auxiliaire) : par rôle, régression ridge de la cible sur les forces
champion × rôle et les matchups (encodage antisymétrique), réglée par
validation temporelle interne. Étape 2 (transfert) : les prédictions de
l'étape 1, sommées sur les 5 lanes (score de matchups, score de forces), sont
ajoutées au modèle de victoire actuel, coefficients ajustés sur l'entraînement.

Cross-fitting : pour l'étape 2, les scores des matchs d'entraînement viennent
de modèles de lane entraînés SANS ces matchs (5 folds). Sinon, avec une cible
liée au résultat comme l'écart d'or, les prédictions en échantillon
« connaissent » l'issue de la partie et l'étape 2 sur-pondère le score
(constaté : coefficient 0,24, perte de −0,002 au test).
Évaluation : gain de log-loss sur deux fenêtres de parties futures, IC apparié.

Le code de production (cible « part d'or ») est dans ml/lane_stats.py.

Usage : python experiments/lane_stats_matchups.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR
from ml.additive_model import AdditiveDraftModel
from ml.draft_data import ROLES, load_champion_ad_share, load_champion_names, load_matches
from ml.lane_stats import fit_lane_models, lane_targets, load_lane_gold, tune_lane_settings
from ml.train import log_losses, with_comp

TARGETS = {"écart d'or": "raw", "part d'or": "share"}


def lane_scores(models, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(score de forces, score de matchups) d'une draft, sommés sur les 5 lanes."""
    mains, pairs = zip(*(models[r].predict_parts(df) for r in ROLES))
    return np.sum(mains, axis=0), np.sum(pairs, axis=0)


def crossfit_scores(train: pd.DataFrame, targets: pd.DataFrame, settings, n_folds: int = 5):
    """Scores (forces, matchups) de chaque match d'entraînement, prédits sans ce match."""
    from sklearn.model_selection import KFold

    main, pair = np.zeros(len(train)), np.zeros(len(train))
    for fit_idx, pred_idx in KFold(n_folds, shuffle=True, random_state=0).split(train):
        models = fit_lane_models(train.iloc[fit_idx], targets, settings)
        main[pred_idx], pair[pred_idx] = lane_scores(models, train.iloc[pred_idx])
    return main, pair


def fit_offset_logit(off, x, y):
    nll = lambda b: np.sum(np.logaddexp(0, off + x @ b) - y * (off + x @ b))
    return minimize(nll, np.zeros(x.shape[1]), method="BFGS").x


def evaluate_window(train, test, hp, ad_share, targets, names) -> None:
    base = AdditiveDraftModel.fit(train, hp, ad_share)
    base = with_comp(base, train) if base.damage is not None else base
    off_tr, off_te = base.predict_logit(train), base.predict_logit(test)
    y_tr, y_te = train.blue_win.to_numpy(), test.blue_win.to_numpy()
    ll_base = log_losses(y_te, 1 / (1 + np.exp(-off_te)))
    print(f"  entraînement {len(train)} matchs → test {len(test)} matchs")

    pair_scores = {}
    for target in TARGETS:
        lane_settings = tune_lane_settings(train, targets[target])
        models = fit_lane_models(train, targets[target], lane_settings)
        settings = ", ".join(f"{r[:3]} α={a:g} s={sc:g}" for r, (a, sc) in lane_settings.items())
        main_tr, pair_tr = crossfit_scores(train, targets[target], lane_settings)
        main_te, pair_te = lane_scores(models, test)
        sd_m, sd_p = main_tr.std() or 1.0, pair_tr.std() or 1.0
        pair_scores[target] = (pair_tr / sd_p, pair_te / sd_p)
        print(f"    cible « {target} » ({settings})")
        for label, cols_tr, cols_te in (
            ("+ matchups", [pair_tr / sd_p], [pair_te / sd_p]),
            ("+ forces", [main_tr / sd_m], [main_te / sd_m]),
            ("+ les deux", [pair_tr / sd_p, main_tr / sd_m], [pair_te / sd_p, main_te / sd_m]),
        ):
            x_tr, x_te = np.column_stack(cols_tr), np.column_stack(cols_te)
            beta = fit_offset_logit(off_tr, x_tr, y_tr)
            d = ll_base - log_losses(y_te, 1 / (1 + np.exp(-(off_te + x_te @ beta))))
            print(f"      {label:<12} coef {np.round(beta, 3)}  | gain TEST {d.mean():+.5f} ± "
                  f"{1.96 * d.std(ddof=1) / np.sqrt(len(d)):.5f}")
        show_top_pairs(models, names)

    x_tr = np.column_stack([pair_scores[t][0] for t in TARGETS])
    x_te = np.column_stack([pair_scores[t][1] for t in TARGETS])
    beta = fit_offset_logit(off_tr, x_tr, y_tr)
    d = ll_base - log_losses(y_te, 1 / (1 + np.exp(-(off_te + x_te @ beta))))
    print(f"    matchups des DEUX cibles  coef {np.round(beta, 3)}  | gain TEST {d.mean():+.5f} ± "
          f"{1.96 * d.std(ddof=1) / np.sqrt(len(d)):.5f}  (corrélation des scores "
          f"{np.corrcoef(x_tr[:, 0], x_tr[:, 1])[0, 1]:.2f})")


def show_top_pairs(models, names, top=4) -> None:
    for role, m in models.items():
        effects = sorted(m.pair_effects().items(), key=lambda kv: -abs(kv[1]))[:top]
        if effects:
            print(f"        {role:<8} matchups (+ = avantage au 1er) : " + ", ".join(
                f"{names.get(int(k.split('|')[2]), '?')} vs {names.get(int(k.split('|')[3]), '?')} {v:+.3f}"
                for k, v in effects))


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    hp = AdditiveDraftModel.load(DATA_DIR / "additive_model.json").hyperparams
    ad_share = load_champion_ad_share()
    names = load_champion_names()
    df = load_matches()
    gold = load_lane_gold(df)
    targets = {label: lane_targets(gold, kind) for label, kind in TARGETS.items()}
    n = len(df)
    print(f"{n} matchs. Gain de log-loss par rapport au modèle actuel (+ = mieux) :")
    for label, (start, end) in {"Fenêtre récente (15 % les plus récents)": (int(n * 0.85), n),
                                "Fenêtre précédente (15 % d'avant)": (int(n * 0.70), int(n * 0.85))}.items():
        print(f"\n{label}")
        evaluate_window(df.iloc[:start], df.iloc[start:end], hp, ad_share, targets, names)


if __name__ == "__main__":
    main()
