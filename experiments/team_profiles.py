"""
team_profiles.py — Contrôle, soutien et vraie frontline : effets d'équipe ?

Profils par champion mesurés avec les champs ajoutés au crawler (timeCCingOthers,
soins, boucliers, dégâts atténués) sur les matchs qui les contiennent :
  - contrôle   : temps de CC infligé par minute ;
  - soutien    : (soins + boucliers sur les alliés) par minute ;
  - frontline  : part des dégâts atténués (damageSelfMitigated) dans son équipe.
Ces profils sont des statistiques en jeu moyennées par champion, pas des
résultats de partie : ils peuvent être appliqués à tous les matchs.

Comme pour l'AD/AP, seul un effet d'équipe NON linéaire peut apporter quelque
chose au-delà de la force champion × rôle. On mesure donc le résidu du modèle
(victoires réelles − prédites ; modèle entraîné sur la 1re moitié, résidus sur
la 2de) selon l'indice d'équipe, puis on teste hors échantillon un terme
b_lin·z + b_sq·z² (bleu − rouge) sur deux fenêtres de parties futures.

Usage : python experiments/team_profiles.py
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
from ml.draft_data import ROLES, connect_read_only, load_champion_ad_share, load_champion_names, load_matches
from ml.train import log_losses, temporal_split, with_comp

MIN_GAMES_PROFILE = 50


def champion_profiles() -> pd.DataFrame:
    conn = connect_read_only()
    p = pd.read_sql_query(
        """
        SELECT p.match_id, p.team_id, p.champion_id, m.game_duration AS dur,
               p.time_ccing_others AS cc, p.total_heal AS heal,
               p.total_damage_shielded_on_teammates AS shield, p.damage_self_mitigated AS mitig
        FROM participants p JOIN matches m USING (match_id)
        WHERE p.time_ccing_others IS NOT NULL AND m.game_duration >= 900
        """,
        conn,
    )
    conn.close()
    minutes = p.dur / 60
    p["cc_rate"] = p.cc / minutes
    p["sustain_rate"] = (p.heal + p.shield) / minutes
    team_mitig = p.groupby(["match_id", "team_id"]).mitig.transform("sum")
    p["front_share"] = p.mitig / team_mitig.where(team_mitig > 0)
    prof = p.groupby("champion_id").agg(
        contrôle=("cc_rate", "mean"), soutien=("sustain_rate", "mean"), frontline=("front_share", "mean"),
        n=("cc_rate", "size"),
    )
    print(f"Profils calculés sur {p.match_id.nunique()} matchs ({len(prof)} champions)")
    return prof[prof.n >= MIN_GAMES_PROFILE]


def team_index(df: pd.DataFrame, side: str, profile: dict[int, float], fill: float) -> np.ndarray:
    return np.sum([df[f"{side}_{r}"].map(profile).fillna(fill).to_numpy() for r in ROLES], axis=0)


def base_model(train, hp, ad_share):
    model = AdditiveDraftModel.fit(train, hp, ad_share)
    return with_comp(model, train) if model.damage is not None else model


def residual_bins(name, df, base, profile, fill) -> None:
    p_blue = base.predict_proba(df)
    res = np.r_[df.blue_win - p_blue, (1 - df.blue_win) - (1 - p_blue)]
    own = np.r_[team_index(df, "blue", profile, fill), team_index(df, "red", profile, fill)]
    opp = np.r_[team_index(df, "red", profile, fill), team_index(df, "blue", profile, fill)]
    z = lambda x: (x - x.mean()) / x.std()
    for label, values in (("de son équipe", z(own)), ("(son équipe − adverse)", z(own - opp))):
        print(f"\n{name.upper()} — résidu selon l'indice {label} (écarts-types) :")
        for b, g in pd.Series(res).groupby(pd.cut(values, [-9, -2, -1, -0.3, 0.3, 1, 2, 9]), observed=True):
            print(f"  {str(b):<16} {100 * g.mean():+.2f} pts ± {196 * g.std(ddof=1) / np.sqrt(len(g)):.2f}  (n={len(g)})")


def oos_test(name, df, hp, ad_share, profile, fill) -> None:
    n = len(df)
    print(f"\n{name.upper()} — test hors échantillon d'un terme d'équipe b_lin·z + b_sq·z² :")
    for label, (start, end) in {"fenêtre récente": (int(n * 0.85), n),
                                "fenêtre précédente": (int(n * 0.70), int(n * 0.85))}.items():
        train, test = df.iloc[:start], df.iloc[start:end]
        base = base_model(train, hp, ad_share)
        teams = np.r_[team_index(train, "blue", profile, fill), team_index(train, "red", profile, fill)]
        mu, sd = teams.mean(), teams.std()

        def features(d):
            zb = (team_index(d, "blue", profile, fill) - mu) / sd
            zr = (team_index(d, "red", profile, fill) - mu) / sd
            return np.stack([zb - zr, zb ** 2 - zr ** 2], 1)

        x, off, y = features(train), base.predict_logit(train), train.blue_win.to_numpy()
        beta = minimize(lambda b: np.sum(np.logaddexp(0, off + x @ b) - y * (off + x @ b)), np.zeros(2), method="BFGS").x
        y_te, off_te = test.blue_win.to_numpy(), base.predict_logit(test)
        d = log_losses(y_te, 1 / (1 + np.exp(-off_te))) - log_losses(y_te, 1 / (1 + np.exp(-(off_te + features(test) @ beta))))
        print(f"  {label:<20} coefficients lin {beta[0]:+.3f}, quad {beta[1]:+.3f}  "
              f"| gain TEST {d.mean():+.5f} ± {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.5f}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    hp = AdditiveDraftModel.load(DATA_DIR / "additive_model.json").hyperparams
    ad_share = load_champion_ad_share()
    names = load_champion_names()
    prof = champion_profiles()
    for col in ("contrôle", "soutien", "frontline"):
        ranked = prof[col].sort_values()
        print(f"  {col:<10} plus fort : {', '.join(names.get(c, c) for c in ranked.index[::-1][:6])}"
              f" | plus faible : {', '.join(names.get(c, c) for c in ranked.index[:4])}")
    print("  corrélations entre profils :", prof[["contrôle", "soutien", "frontline"]].corr().round(2).to_dict())

    df = load_matches()
    first, second = temporal_split(df, 0.5)
    base = base_model(first, hp, ad_share)
    for col in ("contrôle", "soutien", "frontline"):
        profile = prof[col].to_dict()
        residual_bins(col, second, base, profile, prof[col].mean())
    for col in ("contrôle", "soutien", "frontline"):
        oos_test(col, df, hp, ad_share, prof[col].to_dict(), prof[col].mean())


if __name__ == "__main__":
    main()
