"""
learning_curve.py — Combien de matchs faut-il pour chaque composant du modèle ?

Protocole :
  - Test fixe : les 15 % de matchs les plus récents (jamais utilisés autrement).
  - Pour chaque taille N, `--seeds` sous-échantillons tirés dans le reste.
  - Sur chaque sous-échantillon, les composants sont ajoutés un par un, chacun
    avec sa force de régularisation re-réglée par validation temporelle interne
    (le bon niveau de shrinkage dépend du volume) :
        main  → + équilibre AD/AP → + matchups de lane → + synergies → + sensibilités par champion
  - Gain de chaque composant = baisse de log-loss sur le test par rapport à
    l'étape précédente, avec IC 95 % apparié (mêmes matchs de test).

Sorties : tableau console, experiments/results/learning_curve.csv, et
l'évolution de la part retenue de la sensibilité « Malphite contre une équipe AD ».

Usage : python experiments/learning_curve.py [--sizes 8000 15000 25000 35000] [--seeds 3]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from ml.additive_model import AdditiveDraftModel, Hyperparams, fit_comp_effects
from ml.draft_data import load_champion_ad_share, load_champion_names, load_matches
from ml.train import log_losses, temporal_split

OUT_DIR = Path(__file__).parent / "results"
STAGES = ("main", "+ équilibre", "+ lane", "+ duo", "+ comp")
TRACKED = ("Malphite", "Pyke", "Kassadin", "DrMundo")


def tune_stage(train: pd.DataFrame, base: Hyperparams, param: str, grid, ad_share) -> Hyperparams:
    """Choisit `param` dans `grid` par validation temporelle interne (15 % les plus récents)."""
    inner_train, inner_val = temporal_split(train, 0.15)
    best, best_ll = base, float("inf")
    for value in grid:
        hp = Hyperparams(**{**base.__dict__, param: value})
        p = AdditiveDraftModel.fit(inner_train, hp, ad_share).predict_proba(inner_val)
        ll = log_losses(inner_val.blue_win.to_numpy(), p).mean()
        if ll < best_ll:
            best, best_ll = hp, ll
    return best


def run_subsample(train: pd.DataFrame, test: pd.DataFrame, ad_share, names_to_id) -> dict:
    y = test.blue_win.to_numpy()
    out = {}

    hp = tune_stage(train, Hyperparams(), "C", (0.003, 0.01, 0.03), ad_share)
    stage_hps = {"main": hp}
    hp = tune_stage(train, hp, "bal_scale", (0.3, 1.0), ad_share)
    stage_hps["+ équilibre"] = hp
    hp = tune_stage(train, hp, "lane_scale", (0.0, 0.15, 0.3, 0.5, 1.0), ad_share)
    stage_hps["+ lane"] = hp
    hp = tune_stage(train, hp, "duo_scale", (0.0, 0.15, 0.3, 0.5), ad_share)
    stage_hps["+ duo"] = hp

    losses = {}
    for stage in STAGES[:-1]:
        model = AdditiveDraftModel.fit(train, stage_hps[stage], ad_share)
        losses[stage] = log_losses(y, model.predict_proba(test))
        out[f"hp {stage}"] = stage_hps[stage]

    # Dernière étape : sensibilités par champion sélectionnées sur ce sous-échantillon
    comp_effects, comp_stats = fit_comp_effects(model, train)
    model.effects.update(comp_effects)
    losses["+ comp"] = log_losses(y, model.predict_proba(test))
    out["n_comp_selected"] = int((comp_stats.weight > 0.5).sum())   # Pente retenue à plus de moitié
    for name in TRACKED:
        cid = names_to_id.get(name)
        out[f"p {name}"] = float(comp_stats.weight.get(cid, np.nan))

    const = log_losses(y, np.full(len(y), train.blue_win.mean()))
    out["gain total"] = float((const - losses["+ comp"]).mean())
    previous = const
    for stage in STAGES:
        diff = previous - losses[stage]
        out[f"gain {stage}"] = float(diff.mean())
        out[f"ci {stage}"] = float(1.96 * diff.std(ddof=1) / np.sqrt(len(diff)))
        previous = losses[stage]
    out["lane_scale"] = stage_hps["+ lane"].lane_scale
    out["duo_scale"] = stage_hps["+ duo"].duo_scale
    return out


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[8000, 15000, 25000, 35000])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    df = load_matches()
    ad_share = load_champion_ad_share()
    names_to_id = {v: k for k, v in load_champion_names().items()}
    pool, test = temporal_split(df, 0.15)
    sizes = [n for n in args.sizes if n < len(pool)] + [len(pool)]
    print(f"{len(df)} matchs : test fixe = {len(test)} plus récents, pool d'entraînement = {len(pool)}")

    rows = []
    for n in sizes:
        for seed in range(args.seeds if n < len(pool) else 1):
            sub = pool.sample(n=n, random_state=seed).sort_values("game_creation")
            r = run_subsample(sub, test, ad_share, names_to_id)
            r.update(n=n, seed=seed)
            rows.append(r)
            print(f"  N={n:>6} seed={seed}  gain total {r['gain total']:+.5f}  "
                  + "  ".join(f"{s} {r['gain ' + s]:+.5f}" for s in STAGES)
                  + f"  | comp > 50 % {r['n_comp_selected']}, part retenue Malphite {r['p Malphite']:.2f}"
                  + f"  [{time.time() - t0:.0f}s]", flush=True)

    res = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)
    res.drop(columns=[c for c in res.columns if c.startswith("hp ")]).to_csv(OUT_DIR / "learning_curve.csv", index=False)

    print(f"\n{'═' * 96}\nMOYENNE PAR TAILLE (gain de log-loss sur le test ; IC 95 % apparié moyen entre parenthèses)\n{'═' * 96}")
    agg = res.groupby("n")
    header = f"{'N':>7} {'total':>9} " + " ".join(f"{s:>17}" for s in STAGES) + f" {'comp >50%':>10} {'Malphite':>9}"
    print(header)
    for n, g in agg:
        cells = " ".join(f"{g['gain ' + s].mean():>+8.5f} ({g['ci ' + s].mean():.4f})" for s in STAGES)
        print(f"{n:>7} {g['gain total'].mean():>+9.5f} {cells} {g['n_comp_selected'].mean():>10.1f} {g['p Malphite'].median():>9.2f}")
    print(f"\nTerminé en {time.time() - t0:.0f}s → {OUT_DIR / 'learning_curve.csv'}")


if __name__ == "__main__":
    main()
