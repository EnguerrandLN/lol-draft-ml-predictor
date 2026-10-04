"""
patch_transition.py — Le modèle s'adapte-t-il à la sortie d'un patch ?

À chaque patch, des champions sont buffés ou nerfés. Le modèle actuel mélange
tous les patchs : il reste en retard sur ces changements. Deux remèdes :
  - récence : pondérer les parties récentes (demi-vie 14 j) ; pénalise tout
    l'historique, y compris pour les champions qui n'ont pas changé ;
  - marche aléatoire sur les patchs (groupe « patch » du modèle) : la force de
    chaque champion évolue par incréments régularisés ; seuls ceux que les
    données montrent modifiés bougent.

Protocole : pour les transitions 16.17 → 16.18 (10/09) et 16.18 → 16.19 (23/09),
on se place au jour de sortie, puis 1 et 3 jours après (données du nouveau
patch déjà disponibles), on entraîne sur tout ce qui précède et on teste sur
les 5 jours suivants du nouveau patch. Toutes les variantes sont affichées
(pas de sélection a posteriori sur le test). Modèle de base sans les couches
additionnelles (sensibilités, matchups appris sur l'or) pour isoler l'effet.

Usage : python experiments/patch_transition.py
"""
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR
from ml.additive_model import AdditiveDraftModel
from ml.draft_data import load_champion_ad_share, load_matches
from ml.train import log_losses

TRANSITIONS = {"16.18": "2026-09-10", "16.19": "2026-09-23"}
DELAYS_DAYS = (0, 1, 3)
TEST_DAYS = 5


def ms(day: str, plus_days: float = 0) -> int:
    return int((pd.Timestamp(day, tz="UTC") + pd.Timedelta(days=plus_days)).value // 1_000_000)


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    hp = replace(AdditiveDraftModel.load(DATA_DIR / "additive_model.json").hyperparams, patch_scale=0.0)
    variants = {
        "récence (demi-vie 14 j)": replace(hp, half_life_days=14),
        "patch, échelle 0.3": replace(hp, patch_scale=0.3),
        "patch, échelle 0.5": replace(hp, patch_scale=0.5),
        "patch, échelle 1.0": replace(hp, patch_scale=1.0),
    }
    df = load_matches()
    print(f"Base : {hp}\nGain de log-loss par rapport au modèle actuel (+ = mieux), test = {TEST_DAYS} jours suivants :")

    for patch, start in TRANSITIONS.items():
        for delay in DELAYS_DAYS:
            cutoff = ms(start, delay)
            train = df[df.game_creation < cutoff]
            test = df[(df.game_creation >= cutoff) & (df.game_creation < ms(start, delay + TEST_DAYS))
                      & (df.patch == patch)]
            ad_share = load_champion_ad_share(before_ms=cutoff)
            y = test.blue_win.to_numpy()
            current = AdditiveDraftModel.fit(train, hp, ad_share)
            ll_current = log_losses(y, current.predict_proba(test))
            print(f"\nPatch {patch}, J+{delay} : {len(train)} matchs d'entraînement "
                  f"(dont {int((train.patch == patch).sum())} du nouveau patch) → test {len(test)} matchs")
            for label, params in variants.items():
                d = ll_current - log_losses(y, AdditiveDraftModel.fit(train, params, ad_share).predict_proba(test))
                print(f"  {label:<24} {d.mean():+.5f} ± {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.5f}", flush=True)


if __name__ == "__main__":
    main()
