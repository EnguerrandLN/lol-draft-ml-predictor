"""
range_and_scaling.py — Portée et scaling des champions : effets d'équipe ?

Un effet LINÉAIRE d'une caractéristique de champion (« chaque corps-à-corps
ajoute x ») est déjà absorbé par la force champion × rôle du modèle. Seuls les
effets non linéaires à l'échelle de l'équipe peuvent apporter quelque chose,
comme l'équilibre AD/AP : « trop de corps-à-corps », « que du late game »...
On mesure donc le résidu du modèle actuel (victoires réelles − prédites) selon
la composition.

Caractéristiques :
  - portée : attackrange de Data Dragon (statique) ; corps-à-corps si ≤ 250.
  - scaling : pente du résidu d'un champion selon la durée de la partie
    (positive = gagne plus que prévu dans les parties longues), mesurée sur les
    données. Comme ce profil utilise des résultats de parties, il est calculé
    sur la PREMIÈRE moitié chronologique et testé sur la SECONDE (le modèle de
    base est lui aussi entraîné sur la première moitié) : aucun effet ne peut
    apparaître par construction.

Usage : python experiments/range_and_scaling.py          (analyse des résidus)
        python experiments/range_and_scaling.py --oos    (test hors échantillon de la portée)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR
from ml.additive_model import AdditiveDraftModel
from ml.display_names import DDRAGON
from ml.draft_data import ROLES, connect_read_only, load_champion_ad_share, load_champion_names, load_matches
from ml.train import temporal_split, with_comp

MELEE_MAX_RANGE = 250
MIN_GAMES_PROFILE = 100


def attack_ranges() -> dict[int, float]:
    version = requests.get(f"{DDRAGON}/api/versions.json", timeout=10).json()[0]
    data = requests.get(f"{DDRAGON}/cdn/{version}/data/en_US/champion.json", timeout=10).json()["data"]
    return {int(c["key"]): float(c["stats"]["attackrange"]) for c in data.values()}


def residual_frame(model: AdditiveDraftModel, df: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (match, équipe) : résidu vu de l'équipe et ses 5 champions + les 5 adverses."""
    p_blue = model.predict_proba(df)
    rows = []
    for side, opp, sign in (("blue", "red", 1), ("red", "blue", -1)):
        y = df.blue_win.to_numpy() if side == "blue" else 1 - df.blue_win.to_numpy()
        p = p_blue if side == "blue" else 1 - p_blue
        frame = pd.DataFrame({"res": y - p, "p": p, "y": y, "dur": df.game_duration.to_numpy()})
        for i, r in enumerate(ROLES):
            frame[f"own{i}"] = df[f"{side}_{r}"].to_numpy()
            frame[f"opp{i}"] = df[f"{opp}_{r}"].to_numpy()
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def scaling_profile(model: AdditiveDraftModel, df: pd.DataFrame) -> dict[int, float]:
    """Pente du résidu selon la durée (en écarts-types), par champion, atténuée vers 0."""
    fr = residual_frame(model, df)
    dz = (fr.dur - fr.dur.mean()) / fr.dur.std()
    parts = [pd.DataFrame({"c": fr[f"own{i}"], "s": fr.res * dz, "i": fr.p * (1 - fr.p) * dz ** 2}) for i in range(5)]
    st = pd.concat(parts).groupby("c").agg(S=("s", "sum"), I=("i", "sum"), n=("s", "size"))
    st = st[st.n >= MIN_GAMES_PROFILE]
    slope, var = st.S / st.I, 1 / st.I
    tau2 = max(float(np.var(slope) - var.mean()), 1e-6)            # vraie dispersion entre champions
    return (slope * tau2 / (tau2 + var)).to_dict()                   # moyenne a posteriori


def binned(label: str, values: pd.Series, res: pd.Series, bins) -> None:
    print(f"\n{label}")
    for b, g in res.groupby(pd.cut(values, bins), observed=True):
        print(f"  {str(b):<16} résidu {100 * g.mean():+.2f} pts ± {196 * g.std(ddof=1) / np.sqrt(len(g)):.2f}  (n={len(g)})")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    hp = AdditiveDraftModel.load(DATA_DIR / "additive_model.json").hyperparams
    ad_share = load_champion_ad_share()
    names = load_champion_names()
    conn = connect_read_only()
    durations = pd.read_sql_query("SELECT match_id, game_duration FROM matches", conn).set_index("match_id").game_duration
    conn.close()
    df = load_matches().join(durations)

    first, second = temporal_split(df, 0.5)
    base = AdditiveDraftModel.fit(first, hp, ad_share)
    base = with_comp(base, first) if base.damage is not None else base
    print(f"Profils et modèle de base : {len(first)} matchs (1re moitié) ; tests : {len(second)} matchs (2de moitié)")
    test = residual_frame(base, second)

    # ── Portée ────────────────────────────────────────────────────────────────
    ranges = attack_ranges()
    melee = {c: r <= MELEE_MAX_RANGE for c, r in ranges.items()}
    test["own_melee"] = sum(test[f"own{i}"].map(melee).fillna(False).astype(int) for i in range(5))
    test["opp_melee"] = sum(test[f"opp{i}"].map(melee).fillna(False).astype(int) for i in range(5))
    print("\nRépartition du nombre de corps-à-corps par équipe :", test.own_melee.value_counts().sort_index().to_dict())
    binned("PORTÉE — résidu selon le nombre de corps-à-corps de SON équipe :",
           test.own_melee, test.res, [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5])
    binned("PORTÉE — résidu selon (corps-à-corps de son équipe − de l'équipe adverse) :",
           test.own_melee - test.opp_melee, test.res, [-5.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 5.5])

    # ── Scaling ───────────────────────────────────────────────────────────────
    scaling = scaling_profile(base, first)
    ranked = sorted(scaling.items(), key=lambda kv: kv[1])
    print("\nSCALING (profil mesuré sur la 1re moitié, + = gagne plus en partie longue) :")
    print("  plus « late game » :", ", ".join(f"{names.get(c, c)} {s:+.3f}" for c, s in ranked[::-1][:8]))
    print("  plus « early game » :", ", ".join(f"{names.get(c, c)} {s:+.3f}" for c, s in ranked[:8]))
    own = sum(test[f"own{i}"].map(scaling).fillna(0) for i in range(5))
    opp = sum(test[f"opp{i}"].map(scaling).fillna(0) for i in range(5))
    z = lambda x: (x - x.mean()) / x.std()
    binned("SCALING — résidu selon le scaling de SON équipe (écarts-types) :",
           z(own), test.res, [-9, -2, -1, -0.3, 0.3, 1, 2, 9])
    binned("SCALING — résidu selon (scaling de son équipe − adverse) (écarts-types) :",
           z(own - opp), test.res, [-9, -2, -1, -0.3, 0.3, 1, 2, 9])


def melee_oos_test() -> None:
    """
    Test hors échantillon : le modèle actuel + un terme d'équipe sur le nombre de
    corps-à-corps (b_lin·z + b_sq·z², bleu − rouge, comme l'équilibre AD/AP)
    prédit-il mieux des parties futures ? Les 2 coefficients sont ajustés sur
    l'entraînement (le logit du modèle servant de base fixe), évalués sur deux
    fenêtres de test.
    """
    from scipy.optimize import minimize
    from ml.train import log_losses

    hp = AdditiveDraftModel.load(DATA_DIR / "additive_model.json").hyperparams
    ad_share = load_champion_ad_share()
    melee = {c: r <= MELEE_MAX_RANGE for c, r in attack_ranges().items()}
    df = load_matches()
    n = len(df)
    count = lambda d, side: sum(d[f"{side}_{r}"].map(melee).fillna(False).astype(int) for r in ROLES).to_numpy()

    print("\nTEST HORS ÉCHANTILLON — terme « nombre de corps-à-corps » ajouté au modèle actuel :")
    for label, (start, end) in {"fenêtre récente": (int(n * 0.85), n),
                                "fenêtre précédente": (int(n * 0.70), int(n * 0.85))}.items():
        train, test = df.iloc[:start], df.iloc[start:end]
        base = AdditiveDraftModel.fit(train, hp, ad_share)
        base = with_comp(base, train) if base.damage is not None else base
        mu = np.concatenate([count(train, "blue"), count(train, "red")]).mean()

        def features(d):
            zb, zr = count(d, "blue") - mu, count(d, "red") - mu
            return np.stack([zb - zr, zb ** 2 - zr ** 2], 1)

        x_tr, off_tr, y_tr = features(train), base.predict_logit(train), train.blue_win.to_numpy()

        def nll(beta):
            logit = off_tr + x_tr @ beta
            return np.sum(np.logaddexp(0, logit) - y_tr * logit)

        beta = minimize(nll, np.zeros(2), method="BFGS").x
        y = test.blue_win.to_numpy()
        off_te = base.predict_logit(test)
        p0, p1 = 1 / (1 + np.exp(-off_te)), 1 / (1 + np.exp(-(off_te + features(test) @ beta)))
        d = log_losses(y, p0) - log_losses(y, p1)
        print(f"  {label:<20} coefficients lin {beta[0]:+.3f}, quad {beta[1]:+.3f} (optimum à {mu - beta[0] / (2 * beta[1]):.1f} "
              f"corps-à-corps)  | gain TEST {d.mean():+.5f} ± {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.5f}")


if __name__ == "__main__":
    if "--oos" in sys.argv:
        for stream in (sys.stdout, sys.stderr):
            stream.reconfigure(encoding="utf-8", errors="replace")
        melee_oos_test()
    else:
        main()
