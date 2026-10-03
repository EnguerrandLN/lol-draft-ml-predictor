"""
evaluate.py — Évaluation scellée d'un modèle sauvegardé sur une période donnée.

Toutes les décisions de conception du modèle ont été prises en regardant les
fenêtres de test de ml/train.py (les 15 % de parties les plus récents) : ces
chiffres sont donc légèrement optimistes. Pour une mesure honnête, on entraîne
sur les parties d'avant une date, puis on évalue UNE fois sur les parties
d'après, jamais regardées (idéalement le patch suivant) :

  python ml/train.py --until 2026-10-08 --out data/model_sealed.json
  python ml/evaluate.py --model data/model_sealed.json --since 2026-10-08

Le modèle complet (avec sensibilités, calibration par tranche, etc.) est évalué
tel que l'app l'utilise, au global et par tranche d'ELO.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR
from ml.additive_model import AdditiveDraftModel
from ml.draft_data import TIER_BUCKET_LABELS, TIER_TO_BUCKET, load_matches


def log_losses(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def summarize(label: str, y: np.ndarray, p: np.ndarray, p_const: float) -> None:
    gain = log_losses(y, np.full(len(y), p_const)) - log_losses(y, p)
    acc = ((p > 0.5) == y).mean()
    print(f"  {label:<24} n={len(y):>6}  gain log-loss {gain.mean():+.5f} ± "
          f"{1.96 * gain.std(ddof=1) / np.sqrt(len(y)):.5f}  précision {acc:.1%}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DATA_DIR / "additive_model.json"))
    ap.add_argument("--since", required=True, metavar="AAAA-MM-JJ", help="Début de la période évaluée (UTC).")
    ap.add_argument("--until", default=None, metavar="AAAA-MM-JJ", help="Fin de la période évaluée (UTC).")
    args = ap.parse_args()

    model = AdditiveDraftModel.load(Path(args.model))
    trained_until = model.meta["date_range"][1]
    since = pd.Timestamp(args.since, tz="UTC").value // 1_000_000
    df = load_matches()
    df = df[df.game_creation >= since]
    if args.until:
        df = df[df.game_creation < pd.Timestamp(args.until, tz="UTC").value // 1_000_000]
    if since <= trained_until:
        print("⚠ La période évaluée chevauche les données d'entraînement du modèle : "
              "l'évaluation n'est pas scellée (utilise ml/train.py --until).")
    if df.empty:
        raise SystemExit("Aucune partie dans la période demandée.")

    y = df.blue_win.to_numpy()
    # Comme dans l'app : logit de la draft multiplié par le facteur de calibration de la tranche
    scale = df.source_tier.map(TIER_TO_BUCKET).map(model.meta.get("tier_calibration", {})).fillna(1.0).to_numpy()
    p = 1 / (1 + np.exp(-scale * model.predict_logit(df)))
    # Constante de référence : winrate bleu connu du modèle (pas celui de la période évaluée)
    p_const = float(model.meta.get("blue_winrate", 1 / (1 + np.exp(-model.intercept))))
    print(f"Modèle {args.model} (entraîné jusqu'au {pd.to_datetime(trained_until, unit='ms').date()}), "
          f"évalué sur {len(df)} parties ({df.patch.value_counts().to_dict()}) :")
    summarize("Toutes", y, p, p_const)
    buckets = df.source_tier.map(TIER_TO_BUCKET)
    for bucket, label in TIER_BUCKET_LABELS.items():
        mask = (buckets == bucket).to_numpy()
        if mask.sum() >= 200:
            summarize(label, y[mask], p[mask], p_const)

    print("\nCalibration (déciles de probabilité prédite) :")
    for _, g in pd.DataFrame({"p": p, "y": y}).groupby(pd.qcut(p, 10, duplicates="drop"), observed=True):
        print(f"  prédit {g.p.mean():6.1%}   observé {g.y.mean():6.1%} ± {1.96 * np.sqrt(0.25 / len(g)):.1%}")


if __name__ == "__main__":
    main()
