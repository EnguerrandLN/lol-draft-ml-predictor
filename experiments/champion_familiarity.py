"""
champion_familiarity.py — Combien coûte un champion hors de son pool ? (expérience)

Pour chaque partie d'un joueur dont la base contient l'historique récent, on
compare son résultat à la probabilité que la draft laissait attendre (résidu
du modèle), selon le nombre de parties ANTÉRIEURES du joueur sur ce champion.

Deux pièges évités :
  - Compter aussi les parties postérieures crée un biais de survie : après une
    défaite sur un nouveau champion on le rejoue moins, donc cette défaite reste
    seule dans la case « jamais joué ». Ce biais faisait apparaître −5,9 pts.
  - Comparer des joueurs différents mélange l'effet du champion et le niveau du
    joueur. L'écart « à joueur égal » retire le résidu moyen de chaque joueur.

Résultat (2026-10-01, 3 323 joueurs, 35k parties) :
    parties antérieures sur le champion   brut     à joueur égal
    aucune (fenêtre récente)              −3,3     −0,7 ± 1,0
    6 et plus                             +0,9     +0,7 ± 0,9
  - Un champion absent de tes ~20 dernières parties coûte ~1 à 1,5 pt par
    rapport à ton main : faible, à la limite du significatif.
  - Le gros écart brut (~4 pts) est surtout un écart ENTRE joueurs : ceux qui
    ont un pool stable sont meilleurs que ceux qui jouent de tout.
  - « Absent de la fenêtre » ≠ « toute première partie » : le vrai coût d'une
    première partie n'est pas mesurable sans données de maîtrise.

Usage : python experiments/champion_familiarity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR
from ml.additive_model import AdditiveDraftModel
from ml.draft_data import load_matches, load_player_slots
from ml.personal import residuals


def familiarity_table(res: pd.DataFrame, min_prior: int) -> None:
    h = res[res.prior_total >= min_prior].copy()
    h["bucket"] = pd.cut(h.prior_same, [-1, 0, 2, 5, 1000], labels=["aucune (fenêtre)", "1-2", "3-5", "6+"])
    h["r_within"] = h.r - h.groupby("puuid").r.transform("mean")
    print(f"\nJoueurs avec ≥ {min_prior} parties antérieures en base — {len(h)} parties, {h.puuid.nunique()} joueurs")
    print(f"  {'parties antérieures sur ce champion':<36} {'brut':>7} {'à joueur égal':>14} {'±IC95':>6} {'n':>7}")
    for b, g in h.groupby("bucket", observed=True):
        print(f"  {b:<36} {100 * g.r.mean():>+6.1f}  {100 * g.r_within.mean():>+12.1f}  "
              f"{196 * np.sqrt(0.25 / len(g)):>5.1f} {len(g):>7}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")

    model = AdditiveDraftModel.load(DATA_DIR / "additive_model.json")
    df = load_matches()
    res = residuals(model, df, load_player_slots())
    res["r"] = res.y - res.p
    res["t"] = res.match_id.map(df.game_creation)
    res = res.sort_values("t")

    # Uniquement les parties antérieures : le résultat d'une partie n'influence pas son classement
    res["prior_total"] = res.groupby("puuid").cumcount()
    res["prior_same"] = res.groupby(["puuid", "champion_id"]).cumcount()

    familiarity_table(res, 10)
    familiarity_table(res, 20)


if __name__ == "__main__":
    main()
