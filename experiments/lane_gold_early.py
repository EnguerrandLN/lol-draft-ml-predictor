"""
lane_gold_early.py — Matchups appris sur l'or à 10-15 minutes (prolongement de la piste 1).

La couche actuelle (ml/lane_stats.py) apprend les matchups sur la part d'or
de FIN de partie : elle mélange la phase de lane et tout ce qui suit (un
champion de scaling prend sa part d'or tard, quel que soit son matchup).
L'or à 10 ou 15 minutes, lu dans les timelines (fetch_timelines.py), isole la
phase de lane. Cibles par rôle, bleu − rouge :
  - « part d'or à M min » : log(part de l'or de l'équipe), comme la couche actuelle ;
  - « écart d'or à M min » : log(or bleu / or rouge) du duel de lane.

Les timelines ne couvrent qu'un échantillon des matchs. Les modèles de lane
(ridge, mêmes réglages que la couche actuelle) sont entraînés sur les matchs
d'entraînement qui en ont une ; leurs scores de matchups s'appliquent ensuite
à toutes les drafts (ils ne dépendent que des champions). Les matchs de
l'échantillon reçoivent des scores cross-fittés, les autres le score du
modèle complet (qui ne les a pas vus) : le transfert vers la victoire est
estimé sur tous les matchs d'entraînement sans fuite.

Comparaisons (gain de log-loss sur deux fenêtres de parties futures) :
  - référence : modèle actuel avec ses matchups appris sur la part d'or de
    fin de partie (tous les matchs d'entraînement) ;
  - + matchups appris sur l'or à M minutes ;
  - contrôle d'efficacité : part d'or de fin de partie apprise sur le MÊME
    échantillon que l'or à M minutes. Si l'or précoce fait mieux à volume
    égal, c'est un meilleur signal par match : en collecter plus vaut le coût.

Usage : python experiments/lane_gold_early.py [--minutes 10 15]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR
from experiments.lane_stats_extended import fit_offset_logit, tune
from ml.additive_model import AdditiveDraftModel
from ml.draft_data import ROLES, connect_read_only, load_champion_ad_share, load_champion_names, load_matches
from ml.lane_stats import LaneModel, lane_targets, load_lane_gold, pair_score
from ml.train import log_losses, with_comp

MIN_SUBSET: int = 2000   # En dessous, pas assez de timelines dans l'entraînement d'une fenêtre


def load_frame_gold(df: pd.DataFrame, minute: int) -> pd.DataFrame:
    """Or de chaque slot à `minute`, colonnes (side, role), pour les matchs de df qui ont une timeline."""
    conn = connect_read_only()
    g = pd.read_sql_query(
        """SELECT f.match_id, p.team_id, p.position, f.total_gold AS g
           FROM timeline_frames f JOIN participants p ON p.match_id = f.match_id AND p.puuid = f.puuid
           WHERE f.minute = ? AND p.position IN ('TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY')""",
        conn, params=(minute,))
    conn.close()
    g["side"] = g.team_id.map({100: "blue", 200: "red"})
    wide = g.pivot_table(index="match_id", columns=["side", "position"], values="g", aggfunc="first")
    wide = wide[wide.notna().all(axis=1)]
    return wide.loc[wide.index.intersection(df.index)].clip(lower=1)


def subset_scores(train: pd.DataFrame, test: pd.DataFrame, targets: pd.DataFrame, n_folds: int = 5):
    """
    Scores de matchups (train, test) appris sur les matchs de `train` présents dans `targets`.
    Retourne (score train, score test, réglages, modèles), scores réduits par l'écart-type d'entraînement.
    """
    from sklearn.model_selection import KFold

    sub = train[train.index.isin(targets.index)]
    settings = {r: tune(LaneModel, r, sub, targets[r]) for r in ROLES}
    fit = lambda d: {r: LaneModel(r, *settings[r]).fit(d, targets.loc[d.index, r].to_numpy()) for r in ROLES}
    models = fit(sub)
    s_tr = pair_score(models, train)
    cross = np.zeros(len(sub))
    for fit_idx, pred_idx in KFold(n_folds, shuffle=True, random_state=0).split(sub):
        cross[pred_idx] = pair_score(fit(sub.iloc[fit_idx]), sub.iloc[pred_idx])
    s_tr[train.index.get_indexer(sub.index)] = cross
    sd = s_tr.std() or 1.0
    return s_tr / sd, pair_score(models, test) / sd, settings, models


def show_top_pairs(models, names, top: int = 4) -> None:
    for role, m in models.items():
        effects = sorted(m.pair_effects().items(), key=lambda kv: -abs(kv[1]))[:top]
        if effects:
            print(f"        {role:<8} " + ", ".join(
                f"{names.get(int(k.split('|')[2]), '?')} vs {names.get(int(k.split('|')[3]), '?')} {v:+.3f}"
                for k, v in effects))


def evaluate_window(train, test, hp, ad_share, end_share, early: dict[str, pd.DataFrame], names) -> None:
    base = AdditiveDraftModel.fit(train, hp, ad_share)
    base = with_comp(base, train) if base.damage is not None else base
    off_tr, off_te = base.predict_logit(train), base.predict_logit(test)
    y_tr, y_te = train.blue_win.to_numpy(), test.blue_win.to_numpy()
    n_sub = int(train.index.isin(next(iter(early.values())).index).sum())
    print(f"  entraînement {len(train)} matchs (dont {n_sub} avec timeline) → test {len(test)} matchs")
    if n_sub < MIN_SUBSET:
        print(f"  trop peu de timelines dans l'entraînement (< {MIN_SUBSET}) : fenêtre ignorée")
        return

    scores = {"or fin de partie (tous)": subset_scores(train, test, end_share)[:2]}
    sub_index = next(iter(early.values())).index
    scores["or fin de partie (échantillon)"] = subset_scores(train, test, end_share.loc[end_share.index.intersection(sub_index)])[:2]
    for label, targets in early.items():
        s_tr, s_te, settings, models = subset_scores(train, test, targets)
        scores[label] = (s_tr, s_te)
        print(f"    {label} : " + ", ".join(f"{r[:3]} α={a:g} s={s:g}" for r, (a, s) in settings.items()))
        show_top_pairs(models, names)

    def ll(labels):
        x_tr = np.column_stack([scores[l][0] for l in labels])
        x_te = np.column_stack([scores[l][1] for l in labels])
        beta = fit_offset_logit(off_tr, x_tr, y_tr)
        return log_losses(y_te, 1 / (1 + np.exp(-(off_te + x_te @ beta)))), beta

    ll_none = log_losses(y_te, 1 / (1 + np.exp(-off_te)))
    ref, _ = ll(["or fin de partie (tous)"])
    ci = lambda d: f"{d.mean():+.5f} ± {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.5f}"
    print(f"    Gain vs modèle sans matchups appris (même échantillon pour comparer les signaux) :")
    for label in scores:
        if label != "or fin de partie (tous)":
            print(f"      {label:<32} {ci(ll_none - ll([label])[0])}")
    print(f"      {'or fin de partie (tous)':<32} {ci(ll_none - ref)}   ← référence actuelle")
    print(f"    Gain en plus de la référence :")
    for label in early:
        d, beta = ll(["or fin de partie (tous)", label])
        print(f"      + {label:<30} {ci(ref - d)}   coef {np.round(beta, 3)}")
    corr = np.corrcoef([scores[l][0] for l in scores])
    print("    corrélation des scores avec la référence : " + ", ".join(
        f"{l} {corr[0, i]:+.2f}" for i, l in enumerate(scores) if i))


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=int, nargs="+", default=[10, 15])
    args = ap.parse_args()

    hp = AdditiveDraftModel.load(DATA_DIR / "additive_model.json").hyperparams
    ad_share = load_champion_ad_share()
    names = load_champion_names()
    df = load_matches()
    end_share = lane_targets(load_lane_gold(df), "share")
    early = {}
    for minute in args.minutes:
        gold = load_frame_gold(df, minute)
        early[f"part d'or à {minute} min"] = lane_targets(gold, "share")
        early[f"écart d'or à {minute} min"] = lane_targets(gold, "raw")
    first = next(iter(early.values()))
    print(f"{len(df)} matchs, dont {len(first)} avec timeline. Corrélation des cibles avec la victoire, par rôle :")
    for label, t in {"part d'or fin de partie": end_share.loc[first.index], **early}.items():
        print(f"  {label:<26} {np.round(t.corrwith(df.blue_win.loc[t.index].astype(float)).to_numpy(), 2)}")

    n = len(df)
    for label, (start, end) in {"Fenêtre récente (15 % les plus récents)": (int(n * 0.85), n),
                                "Fenêtre précédente (15 % d'avant)": (int(n * 0.70), int(n * 0.85))}.items():
        print(f"\n{label}")
        evaluate_window(df.iloc[:start], df.iloc[start:end], hp, ad_share, end_share, early, names)


if __name__ == "__main__":
    main()
