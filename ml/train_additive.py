"""
train_additive.py — Réglage, évaluation et export du modèle additif.

Protocole (aucune fuite possible : une ligne = un match, découpage temporel) :
  1. Test     = les 15 % de matchs les plus récents. Jamais vus avant l'étape 3.
  2. Réglage  = sur le reste, validation glissante (3 blocs de 10 % en fin de période).
                Recherche coordonnée : C, puis échelle lane, puis échelle duo,
                puis demi-vie de récence. Critère : log-loss.
  3. Rapport  = réentraînement sur tout le reste, évaluation sur le test :
                log-loss vs constante (IC 95 %), Brier, accuracy, calibration.
  4. Export   = réentraînement sur TOUTES les données → data/additive_model.json.

Usage :
  python ml/train_additive.py
  python ml/train_additive.py --min-tier EMERALD     # matchs étiquetés Émeraude+ uniquement
"""
import argparse
import logging
import sys
import time
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR
from ml.additive_model import AdditiveDraftModel, Hyperparams, fit_sparse_comp
from ml.draft_data import ROLES, load_champion_ad_share, load_champion_names, load_matches, load_player_slots
from ml.personal import estimate_tau, residuals

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TIER_ORDER = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]

C_GRID = (0.003, 0.01, 0.03, 0.1)
SCALE_GRID = (0.0, 0.15, 0.3, 0.5, 1.0)
HALF_LIFE_GRID = (None, 60, 30, 14)


# ── Métriques ─────────────────────────────────────────────────────────────────

def log_losses(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def temporal_split(df: pd.DataFrame, frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """df est trié par date : les `frac` derniers matchs forment la partie évaluée."""
    cut = int(len(df) * (1 - frac))
    return df.iloc[:cut], df.iloc[cut:]


def rolling_folds(df: pd.DataFrame, n_folds: int, block: float) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Validation glissante : chaque fold s'entraîne sur tout ce qui précède un bloc
    de `block` % des matchs, et est évalué sur ce bloc. Les blocs couvrent la
    fin de la période. Une seule fenêtre de validation est trop bruitée pour
    départager des réglages proches (vu en pratique sur les matchups).
    """
    folds = []
    for k in range(n_folds, 0, -1):
        start = int(len(df) * (1 - k * block))
        end = int(len(df) * (1 - (k - 1) * block))
        folds.append((df.iloc[:start], df.iloc[start:end]))
    return folds


def cv_log_loss(folds, hp: Hyperparams, ad_share: dict[int, float]) -> float:
    losses = [
        log_losses(val.blue_win.to_numpy(), AdditiveDraftModel.fit(train, hp, ad_share).predict_proba(val))
        for train, val in folds
    ]
    return float(np.concatenate(losses).mean())


# ── Réglage ───────────────────────────────────────────────────────────────────

def tune(df: pd.DataFrame, ad_share: dict[int, float], n_folds: int = 3, block: float = 0.1) -> Hyperparams:
    folds = rolling_folds(df, n_folds, block)
    y_val = np.concatenate([val.blue_win.to_numpy() for _, val in folds])
    p_const = np.concatenate([np.full(len(val), train.blue_win.mean()) for train, val in folds])
    const = float(log_losses(y_val, p_const).mean())
    log.info("Réglage : %d folds glissants, %d matchs de validation au total (constante : %.5f)",
             n_folds, len(y_val), const)

    best = Hyperparams()
    best_ll = float("inf")

    def try_grid(param: str, grid) -> None:
        nonlocal best, best_ll
        for value in grid:
            hp = Hyperparams(**{**best.__dict__, param: value})
            ll = cv_log_loss(folds, hp, ad_share)
            marker = ""
            if ll < best_ll - 1e-6:
                best, best_ll, marker = hp, ll, "  ★"
            log.info("  %-15s = %-6s → log-loss %.5f%s", param, value, ll, marker)

    try_grid("C", C_GRID)
    try_grid("bal_scale", SCALE_GRID)
    # Les sensibilités par champion (« comp ») ne passent pas par la L2 : elles sont
    # sélectionnées après coup par fit_sparse_comp (voir sa docstring).
    try_grid("lane_scale", SCALE_GRID)
    try_grid("duo_scale", SCALE_GRID)
    try_grid("half_life_days", HALF_LIFE_GRID)
    try_grid("C", C_GRID)  # Re-vérifier C une fois les autres groupes fixés

    log.info("Retenu : %s (log-loss val %.5f, gain vs constante %.5f)", best, best_ll, const - best_ll)
    return best


# ── Rapport sur le test ───────────────────────────────────────────────────────

def report(train: pd.DataFrame, test: pd.DataFrame, hp: Hyperparams, ad_share: dict[int, float]) -> dict:
    y = test.blue_win.to_numpy()
    n = len(y)
    p_const = np.full(n, train.blue_win.mean())
    ll_const = log_losses(y, p_const)

    results = {}
    print(f"\n{'═' * 78}\nTEST : {n} matchs les plus récents "
          f"({pd.to_datetime(test.game_creation.min(), unit='ms').date()} → "
          f"{pd.to_datetime(test.game_creation.max(), unit='ms').date()})\n{'═' * 78}")
    print(f"{'Modèle':<34} {'Log-loss':>9} {'Gain vs const (IC 95%)':>24} {'Brier':>7} {'Acc':>13}")

    def fitted(params: Hyperparams, sparse_comp: bool = False) -> AdditiveDraftModel:
        model = AdditiveDraftModel.fit(train, params, ad_share)
        if sparse_comp and model.damage is not None:
            model.effects.update(fit_sparse_comp(model, train)[0])
        return model

    candidates = {
        "Constante (winrate bleu)": None,
        "Effets champion×rôle seuls": lambda: fitted(Hyperparams(C=hp.C, half_life_days=hp.half_life_days)),
        "Sans sensibilités par champion": lambda: fitted(hp),
        "Modèle retenu": lambda: fitted(hp, sparse_comp=True),
    }
    for label, build in candidates.items():
        p = p_const if build is None else build().predict_proba(test)
        ll = log_losses(y, p)
        gain = ll_const - ll
        half_ci = 1.96 * gain.std(ddof=1) / np.sqrt(n)
        acc = ((p > 0.5) == y).mean()
        acc_ci = 1.96 * np.sqrt(acc * (1 - acc) / n)
        print(f"{label:<34} {ll.mean():>9.5f} {gain.mean():>+12.5f} ± {half_ci:.5f}   "
              f"{((p - y) ** 2).mean():>7.4f} {acc:>7.2%} ±{acc_ci:.1%}")
        results[label] = {"log_loss": float(ll.mean()), "gain": float(gain.mean()),
                          "gain_ci95": float(half_ci), "accuracy": float(acc)}
        if label == "Modèle retenu":
            p_model = p

    # Calibration : quand le modèle annonce X %, l'équipe gagne-t-elle X % du temps ?
    print("\nCalibration du modèle retenu (déciles de probabilité prédite) :")
    print(f"  {'Prédit':>8} {'Observé':>8} {'± IC95':>7} {'N':>6}")
    bins = pd.qcut(p_model, 10, duplicates="drop")
    for _, grp in pd.DataFrame({"p": p_model, "y": y}).groupby(bins, observed=True):
        obs = grp.y.mean()
        print(f"  {grp.p.mean():>8.1%} {obs:>8.1%} {1.96 * np.sqrt(obs * (1 - obs) / len(grp)):>7.1%} {len(grp):>6}")
    results["prediction_spread"] = {"p05": float(np.quantile(p_model, 0.05)), "p95": float(np.quantile(p_model, 0.95))}
    return results


def print_top_effects(model: AdditiveDraftModel, min_games: int = 200) -> None:
    names = model.champion_names
    print(f"\n{'═' * 78}\nAPERÇU DES EFFETS (modèle final, champions ≥ {min_games} matchs au rôle)\n{'═' * 78}")
    for role in ROLES:
        rows = [
            (names.get(c, c), model.effects.get(f"m|{role}|{c}", 0.0), n)
            for c, n in model.games[role].items() if n >= min_games
        ]
        rows.sort(key=lambda r: r[1])
        fmt = lambda r: f"{r[0]} {r[1]:+.2f} ({r[2]})"
        print(f"{role:<8} ▲ {', '.join(fmt(r) for r in rows[::-1][:4])}")
        print(f"{'':<8} ▼ {', '.join(fmt(r) for r in rows[:4])}")

    candidates = model.meta.get("comp_candidates", [])
    if candidates:
        print("\nSensibilité au profil de dégâts adverse (pente en logit par écart-type de part AD,")
        print("+ = meilleur contre une équipe AD). Retenus = significatifs après correction (FDR 10 %) :")
        for c in candidates[:10]:
            mark = "✔ retenu" if c["selected"] else ""
            print(f"  {names.get(str(c['champion_id']), c['champion_id']):<12} {c['slope']:+.3f}  "
                  f"p = {c['p_value']:.4f}  {mark}")

    b_lin, b_sq = (model.effects.get(k, 0.0) for k in ("b|lin", "b|sq"))
    if b_lin or b_sq:
        print(f"\nÉquilibre des dégâts de l'équipe : {b_lin:+.3f}·z {b_sq:+.3f}·z²  (z = part AD centrée-réduite)")

    lanes = sorted(
        ((k, e) for k, e in model.effects.items() if k.startswith("l|")), key=lambda kv: -abs(kv[1])
    )[:8]
    if lanes:
        print("\nMatchups de lane les plus marqués (logit, + = avantage au premier) :")
        for k, e in lanes:
            _, role, a, b = k.split("|")
            print(f"  {role:<8} {names.get(a, a)} vs {names.get(b, b)} : {e:+.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--min-tier", choices=TIER_ORDER, default=None,
                   help="Ne garder que les matchs étiquetés à ce tier ou plus (exclut les matchs sans tier).")
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--out", default=str(DATA_DIR / "additive_model.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    df = load_matches()
    if args.min_tier:
        allowed = TIER_ORDER[TIER_ORDER.index(args.min_tier):]
        df = df[df.source_tier.isin(allowed)]
    log.info("%d matchs chargés (patchs : %s)", len(df), df.patch.value_counts().head(4).to_dict())

    train_full, test = temporal_split(df, args.test_frac)

    ad_share = load_champion_ad_share()
    hp = tune(train_full, ad_share)
    results = report(train_full, test, hp, ad_share)

    log.info("Entraînement final sur les %d matchs...", len(df))
    model = AdditiveDraftModel.fit(df, hp, ad_share)
    comp_stats = None
    if model.damage is not None:
        comp_effects, comp_stats = fit_sparse_comp(model, df)
        model.effects.update(comp_effects)
    model.attach_stats(df, load_champion_names())
    # Écart réel entre joueurs sur un même champion : a priori de la personnalisation
    personal_tau = estimate_tau(residuals(model, df, load_player_slots()))
    log.info("τ personnel mesuré : %.3f logit", personal_tau)

    model.meta = {
        "n_matches": len(df),
        "min_tier": args.min_tier,
        "date_range": [int(df.game_creation.min()), int(df.game_creation.max())],
        "patches": df.patch.value_counts().to_dict(),
        "test_results": results,
        "personal_tau": personal_tau,
        # Sensibilités au profil adverse les plus marquées, retenues ou non (transparence)
        "comp_candidates": [] if comp_stats is None else [
            {"champion_id": int(c), "slope": float(r.slope), "p_value": float(r.p_value),
             "selected": bool(r.selected)}
            for c, r in comp_stats.head(15).iterrows()
        ],
    }
    model.save(Path(args.out))
    print_top_effects(model)
    log.info("Modèle exporté → %s (%d effets, %.0fs)", args.out, len(model.effects), time.time() - t0)


if __name__ == "__main__":
    main()
