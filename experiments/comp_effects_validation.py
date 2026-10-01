"""
comp_effects_validation.py — Quelle méthode pour les sensibilités au profil adverse ?

Compare, sur des parties FUTURES (jamais vues), trois façons de traiter la
sensibilité de chaque champion au profil de dégâts de l'équipe adverse :
  - aucune          : pas d'effet par champion ;
  - tout ou rien    : sélection Benjamini-Hochberg (FDR 10 %), effets retenus
                      entiers, les autres à 0 (ancienne méthode) ;
  - lissage         : stabilité temporelle (effets aléatoires) + mélange bayésien
                      empirique, effet × probabilité qu'il soit réel (nouvelle méthode).

Deux fenêtres de test : les 15 % de matchs les plus récents, et les 15 %
précédents (le modèle étant alors entraîné sur tout ce qui les précède).
Gain mesuré en log-loss, sur tous les matchs et sur ceux où un champion à effet
notable (|effet| > 0.01 dans la méthode évaluée) est présent ; IC 95 % apparié.

Usage : python experiments/comp_effects_validation.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR
from ml.additive_model import AdditiveDraftModel, comp_key, fit_comp_effects
from ml.draft_data import ROLES, load_champion_ad_share, load_matches
from ml.train import log_losses


# Ancienne méthode, conservée comme point de comparaison.
def fit_sparse_comp(
    model: AdditiveDraftModel, df: pd.DataFrame, fdr: float = 0.1,
) -> tuple[dict[str, float], pd.DataFrame]:
    """
    Estime, pour chaque champion, sa sensibilité au profil de dégâts adverse
    (effet « comp ») à partir des résidus du modèle, et ne garde que les effets
    statistiquement établis.

    Pourquoi pas la L2 du groupe « comp » : la plupart des champions n'ont aucune
    sensibilité de ce type, quelques-uns en ont une forte (Malphite, Kassadin...).
    Une pénalité L2 commune suppose des effets petits et répartis : elle écrase
    les vrais effets forts. Ici :
      1. pente par champion = Σ résidu·z / Σ p(1−p)·z²  (un pas de Newton depuis 0),
         erreur type = 1/√(Σ p(1−p)·z²) ;
      2. sélection par Benjamini-Hochberg au taux de fausses découvertes `fdr` ;
      3. atténuation James-Stein des pentes retenues : b·(1 − se²/b²).

    Returns:
        (effets à ajouter au modèle, tableau de diagnostic par champion)
    """
    from scipy.stats import norm

    dmg = model.damage
    p_blue = model.predict_proba(df)
    z_blue, z_red = dmg.z(dmg.team_ad(df, "blue")), dmg.z(dmg.team_ad(df, "red"))
    res_blue = df.blue_win.to_numpy() - p_blue
    info_w = p_blue * (1 - p_blue)

    parts = []
    for role in ROLES:
        # Un champion bleu fait face au profil rouge, et inversement (résidu vu de son équipe)
        parts.append(pd.DataFrame({"c": df[f"blue_{role}"].to_numpy(), "s": res_blue * z_red, "i": info_w * z_red ** 2}))
        parts.append(pd.DataFrame({"c": df[f"red_{role}"].to_numpy(), "s": -res_blue * z_blue, "i": info_w * z_blue ** 2}))
    stats = pd.concat(parts).groupby("c").agg(S=("s", "sum"), I=("i", "sum"), n=("s", "size"))
    stats = stats[stats.I > 0]
    stats["slope"] = stats.S / stats.I
    stats["se"] = 1 / np.sqrt(stats.I)
    stats["p_value"] = 2 * norm.sf(np.abs(stats.slope / stats.se))

    # Benjamini-Hochberg : plus grand rang k tel que p_(k) ≤ k/m · fdr
    ranked = stats.sort_values("p_value")
    m = len(ranked)
    passing = np.nonzero(ranked.p_value.to_numpy() <= np.arange(1, m + 1) / m * fdr)[0]
    selected = set(ranked.index[: passing.max() + 1]) if len(passing) else set()

    stats["selected"] = stats.index.isin(selected)
    stats["effect"] = np.where(
        stats.selected, stats.slope * np.clip(1 - stats.se ** 2 / stats.slope ** 2, 0, 1), 0.0
    )
    effects = {comp_key(int(c)): float(e) for c, e in stats.effect.items() if e != 0.0}
    return effects, stats.sort_values("p_value")


METHODS = {"tout ou rien": fit_sparse_comp, "lissage": fit_comp_effects}


def evaluate(train, test, hp, ad_share) -> None:
    base = AdditiveDraftModel.fit(train, hp, ad_share)
    y = test.blue_win.to_numpy()
    ll_base = log_losses(y, base.predict_proba(test))
    print(f"  entraînement {len(train)} matchs → test {len(test)} matchs")
    for label, method in METHODS.items():
        effects, _ = method(base, train)
        model = AdditiveDraftModel(base.intercept, {**base.effects, **effects}, hp, damage=base.damage)
        diff = ll_base - log_losses(y, model.predict_proba(test))
        notable = {int(k.split("|")[1]) for k, e in effects.items() if abs(e) > 0.01}
        mask = np.any([test[f"{s}_{r}"].isin(notable) for s in ("blue", "red") for r in ROLES], axis=0)
        sub = diff[mask] if mask.any() else np.zeros(1)
        print(f"    {label:<14} tous : {diff.mean():+.5f} ± {1.96 * diff.std(ddof=1) / np.sqrt(len(diff)):.5f}"
              f"   | matchs concernés ({mask.sum():>5}) : {sub.mean():+.5f} ± {1.96 * sub.std(ddof=1) / np.sqrt(len(sub)):.5f}"
              f"   ({len(notable)} champions à effet notable)")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    hp = AdditiveDraftModel.load(DATA_DIR / "additive_model.json").hyperparams
    ad_share = load_champion_ad_share()
    df = load_matches()
    n = len(df)
    windows = {
        "Fenêtre récente (15 % les plus récents)": (int(n * 0.85), n),
        "Fenêtre précédente (15 % d'avant)": (int(n * 0.70), int(n * 0.85)),
    }
    print("Gain de log-loss par rapport au modèle SANS sensibilités par champion (+ = mieux) :")
    for label, (start, end) in windows.items():
        print(f"\n{label}")
        evaluate(df.iloc[:start], df.iloc[start:end], hp, ad_share)


if __name__ == "__main__":
    main()
