"""
frontline_balance.py — Une équipe sans frontline perd-elle plus ? (expérience)

Indice de frontline d'un champion : sa part moyenne des dégâts encaissés par
son équipe (0.20 = moyenne ; Dr. Mundo 0.36, Yuumi 0.08). Indice d'une équipe :
somme des indices de ses 5 champions.

Résultat (2026-10-01, 53k matchs) — NÉGATIF, non intégré au modèle :
  - Winrate plat selon la frontline de son équipe (49,6 % à 50,7 % sur les
    déciles) et selon l'écart avec l'adversaire. Seul le 1 % d'équipes les
    plus fragiles montre −3,6 ± 3,0 pts : à la limite du bruit.
    (Contraste : l'équilibre AD/AP, lui, vaut ~7 pts et est dans le modèle.)
  - Sensibilité par champion à la frontline adverse : 2 champions passent la
    correction de Benjamini-Hochberg (Maokai, Nautilus), sans les « anti-tanks »
    attendus (Vayne, Fiora). Pas assez pour justifier une dimension de plus.
  À retester avec plus de données, ou avec une meilleure mesure de la tankiness
  (dégâts atténués, temps de CC : non collectés aujourd'hui).

Usage : python experiments/frontline_balance.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR
from ml.additive_model import AdditiveDraftModel
from ml.draft_data import ROLES, connect_read_only, load_matches


def champion_frontline() -> dict[int, float]:
    conn = connect_read_only()
    p = pd.read_sql_query("SELECT match_id, team_id, champion_id, total_damage_taken AS dt FROM participants", conn)
    conn.close()
    p = p[p.dt > 0]
    p["share"] = p.dt / p.groupby(["match_id", "team_id"]).dt.transform("sum")
    return p.groupby("champion_id").share.mean().to_dict()


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    tank = champion_frontline()
    model = AdditiveDraftModel.load(DATA_DIR / "additive_model.json")
    names = {int(k): v for k, v in model.champion_names.items()}
    df = load_matches()
    front = {s: np.sum([df[f"{s}_{r}"].map(tank).fillna(0.2) for r in ROLES], axis=0) for s in ("blue", "red")}

    # 1. Winrate selon la frontline de son équipe
    teams = pd.concat([
        pd.DataFrame({"f": front["blue"], "win": df.blue_win}),
        pd.DataFrame({"f": front["red"], "win": 1 - df.blue_win}),
    ])
    print("Winrate selon l'indice de frontline de son équipe (déciles) :")
    for b, g in teams.groupby(pd.qcut(teams.f, 10), observed=True):
        print(f"  {str(b):<18} {g.win.mean():.1%} ±{1.96 * np.sqrt(0.25 / len(g)):.1%}")

    # 2. Sensibilité par champion à la frontline adverse, sur les résidus du modèle
    pooled = np.r_[front["blue"], front["red"]]
    z = {s: (front[s] - pooled.mean()) / pooled.std() for s in front}
    p_blue = model.predict_proba(df)
    res, w = df.blue_win.to_numpy() - p_blue, p_blue * (1 - p_blue)
    parts = []
    for r in ROLES:
        parts.append(pd.DataFrame({"c": df[f"blue_{r}"].to_numpy(), "s": res * z["red"], "i": w * z["red"] ** 2}))
        parts.append(pd.DataFrame({"c": df[f"red_{r}"].to_numpy(), "s": -res * z["blue"], "i": w * z["blue"] ** 2}))
    st = pd.concat(parts).groupby("c").agg(S=("s", "sum"), I=("i", "sum"))
    st["slope"] = st.S / st.I
    st["p"] = 2 * norm.sf(np.abs(st.slope * np.sqrt(st.I)))
    st = st.sort_values("p")
    k = len(st)
    passing = np.nonzero(st.p.to_numpy() <= np.arange(1, k + 1) / k * 0.1)[0]
    print(f"\nSensibilité à la frontline adverse : {0 if len(passing) == 0 else passing.max() + 1} "
          f"champion(s) significatif(s) après correction BH (FDR 10 %)")
    for c, row in st.head(8).iterrows():
        print(f"  {names.get(c, c):<12} {row.slope:+.3f}  p = {row.p:.4f}")
    print(f"  p < 0.05 : {(st.p < 0.05).sum()} champions sur {k} (attendu par hasard : {0.05 * k:.1f})")


if __name__ == "__main__":
    main()
