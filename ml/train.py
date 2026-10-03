"""
train.py — Réglage, évaluation et export du modèle additif.

Protocole (aucune fuite possible : une ligne = un match, découpage temporel) :
  1. Test     = les 15 % de matchs les plus récents. Jamais vus avant l'étape 3.
  2. Réglage  = sur le reste, validation glissante (3 blocs de 10 % en fin de période).
                Recherche coordonnée : C, puis échelle lane, puis échelle duo,
                puis demi-vie de récence. Critère : log-loss.
  3. Rapport  = réentraînement sur tout le reste, évaluation sur le test :
                log-loss vs constante (IC 95 %), Brier, accuracy, calibration.
  4. Export   = réentraînement sur TOUTES les données → data/additive_model.json.

Usage :
  python ml/train.py
  python ml/train.py --min-tier EMERALD     # matchs étiquetés Émeraude+ uniquement
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
from ml.additive_model import AdditiveDraftModel, Hyperparams, fit_comp_effects
from ml.draft_data import (
    ROLES, TIER_BUCKET_LABELS, TIER_TO_BUCKET, load_champion_ad_share, load_champion_names, load_matches,
    load_player_slots,
)
from ml.personal import estimate_familiarity, estimate_tau, residuals, role_shares

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TIER_ORDER = ["IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER", "CHALLENGER"]

C_GRID = (0.003, 0.01, 0.03, 0.1)
SCALE_GRID = (0.0, 0.15, 0.3, 0.5, 1.0)
HALF_LIFE_GRID = (None, 60, 30, 14)
# Amélioration minimale de log-loss de validation pour ajouter de la complexité :
# en dessous, l'écart est du bruit de sélection (vu sur les écarts par ELO : 0.00001).
MIN_IMPROVEMENT = 5e-5


# ── Métriques ─────────────────────────────────────────────────────────────────

def estimate_tier_calibration(
    df: pd.DataFrame, hp: Hyperparams, ad_share: dict[int, float],
    n_folds: int = 5, prior_sd: float = 0.5, min_matches: int = 300,
) -> dict[str, dict[str, float]]:
    """
    Pente de calibration par tranche d'ELO, mesurée hors échantillon.

    Chaque fold de matchs étiquetés est prédit par un modèle entraîné sur tout
    le reste (non étiquetés + autres folds). Par tranche, la régression
    logistique de l'issue sur le logit prédit donne une pente ŝ (1 = effets de
    draft bien dosés ; 0.5 = deux fois trop forts à ce niveau), d'erreur type se.
    Elle est atténuée vers 1 avec un a priori N(1, prior_sd²) :
        s = 1 + (ŝ − 1) · prior_sd² / (prior_sd² + se²)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold

    tagged = df[df.source_tier.map(TIER_TO_BUCKET).notna()]
    untagged = df.drop(tagged.index)
    if len(tagged) < min_matches:
        return {}
    oos_logit = pd.Series(np.nan, index=tagged.index)
    for train_idx, test_idx in KFold(n_folds, shuffle=True, random_state=0).split(tagged):
        train = pd.concat([untagged, tagged.iloc[train_idx]]).sort_values("game_creation")
        fold = tagged.iloc[test_idx]
        oos_logit.loc[fold.index] = AdditiveDraftModel.fit(train, hp, ad_share).predict_logit(fold)

    out = {}
    for bucket, g in tagged.groupby(tagged.source_tier.map(TIER_TO_BUCKET)):
        if len(g) < min_matches:
            continue
        x = oos_logit.loc[g.index].to_numpy()
        clf = LogisticRegression(C=1e6).fit(x.reshape(-1, 1), g.blue_win.to_numpy())
        slope = float(clf.coef_[0][0])
        p = clf.predict_proba(x.reshape(-1, 1))[:, 1]
        se = float(1 / np.sqrt(np.sum(p * (1 - p) * x ** 2)))
        weight = prior_sd ** 2 / (prior_sd ** 2 + se ** 2)
        out[bucket] = {"scale": 1 + (slope - 1) * weight, "raw_slope": slope, "se": se, "n": len(g)}
    return out


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

    def try_grid(param: str, grid, min_gain: float = MIN_IMPROVEMENT) -> None:
        """Parcimonie : les grilles commencent par la valeur la plus simple (0 / None), et
        une autre valeur n'est retenue que si elle améliore la validation d'au moins `min_gain`."""
        nonlocal best, best_ll
        for value in grid:
            hp = Hyperparams(**{**best.__dict__, param: value})
            ll = cv_log_loss(folds, hp, ad_share)
            marker = ""
            if ll < best_ll - min_gain:
                best, best_ll, marker = hp, ll, "  ★"
            log.info("  %-15s = %-6s → log-loss %.5f%s", param, value, ll, marker)

    try_grid("C", C_GRID, min_gain=1e-6)
    try_grid("bal_scale", SCALE_GRID)
    try_grid("tier_scale", SCALE_GRID)
    # Les sensibilités par champion (« comp ») ne passent pas par la L2 : elles sont
    # lissées après coup par fit_comp_effects, puis adoptées ou non par decide_comp.
    try_grid("lane_scale", SCALE_GRID)
    try_grid("duo_scale", SCALE_GRID)
    try_grid("half_life_days", HALF_LIFE_GRID)
    try_grid("C", C_GRID, min_gain=1e-6)  # Re-vérifier C une fois les autres groupes fixés

    log.info("Retenu : %s (log-loss val %.5f, gain vs constante %.5f)", best, best_ll, const - best_ll)
    return best


def with_comp(model: AdditiveDraftModel, train: pd.DataFrame) -> AdditiveDraftModel:
    """Copie du modèle enrichie des sensibilités par champion lissées, estimées sur `train`."""
    effects, _ = fit_comp_effects(model, train)
    return AdditiveDraftModel(model.intercept, {**model.effects, **effects}, model.hyperparams, damage=model.damage)


def decide_comp(df: pd.DataFrame, hp: Hyperparams, ad_share: dict[int, float],
                n_folds: int = 3, block: float = 0.1) -> dict:
    """
    Garde-fou des sensibilités par champion, sur la même validation glissante
    que le réglage. Elles sont estimées par un lissage bayésien qui se règle
    déjà sur la force des preuves : la validation sert à détecter un NUISIBLE
    avéré, pas à départager le bruit. Elles ne sont donc rejetées que si elles
    dégradent significativement les prédictions (gain + IC 95 % < 0).

    (Règle précédente : adoptées si gain > 0. Avec des gains de ±0,00001 pour
    un IC de ±0,00015, la décision basculait au hasard d'un entraînement à
    l'autre : adoptées à 75k matchs, rejetées à 100k, adoptées à 155k.)
    """
    if hp.bal_scale <= 0 and hp.comp_scale <= 0:
        return {"adopted": False, "gain": 0.0, "ci95": 0.0}
    diffs = []
    for train, val in rolling_folds(df, n_folds, block):
        base = AdditiveDraftModel.fit(train, hp, ad_share)
        y = val.blue_win.to_numpy()
        diffs.append(log_losses(y, base.predict_proba(val)) - log_losses(y, with_comp(base, train).predict_proba(val)))
    d = np.concatenate(diffs)
    gain, ci = float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))
    adopted = gain + ci >= 0
    log.info("Sensibilités par champion en validation glissante : gain %+.5f ± %.5f → %s",
             gain, ci, "adoptées" if adopted else "rejetées (nuisibles)")
    return {"adopted": adopted, "gain": gain, "ci95": ci}


# ── Rapport sur le test ───────────────────────────────────────────────────────

def report(train: pd.DataFrame, test: pd.DataFrame, hp: Hyperparams, ad_share: dict[int, float],
           use_comp: bool) -> dict:
    y = test.blue_win.to_numpy()
    n = len(y)
    p_const = np.full(n, train.blue_win.mean())
    ll_const = log_losses(y, p_const)

    results = {}
    print(f"\n{'═' * 78}\nTEST : {n} matchs les plus récents "
          f"({pd.to_datetime(test.game_creation.min(), unit='ms').date()} → "
          f"{pd.to_datetime(test.game_creation.max(), unit='ms').date()})\n{'═' * 78}")
    print(f"{'Modèle':<34} {'Log-loss':>9} {'Gain vs const (IC 95%)':>24} {'Brier':>7} {'Acc':>13}")

    base = AdditiveDraftModel.fit(train, hp, ad_share)
    candidates = {
        "Constante (winrate bleu)": None,
        "Effets champion×rôle seuls": lambda: AdditiveDraftModel.fit(
            train, Hyperparams(C=hp.C, half_life_days=hp.half_life_days), ad_share),
        "Sans sensibilités par champion": lambda: base,
    }
    if base.damage is not None:
        candidates["Avec sensibilités lissées"] = lambda: with_comp(base, train)
    retained = "Avec sensibilités lissées" if use_comp else "Sans sensibilités par champion"

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
        if label == retained:
            p_model = p
            results["Modèle retenu"] = results[label]
    print(f"→ Modèle retenu : « {retained} »")

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

    comp = model.meta.get("comp", {})
    if comp.get("candidates"):
        prior = comp["prior"]
        print(f"\nSensibilité au profil de dégâts adverse ({'ADOPTÉE' if comp['adopted'] else 'non adoptée'} : "
              f"gain en validation {comp['validation_gain']:+.5f}).")
        print(f"Part estimée de champions à effet réel {prior['pi']:.0%}, ampleur typique {prior['tau']:.3f}, "
              f"dérive entre périodes {prior['omega']:.3f}. Pente en logit par écart-type de part AD")
        print("(+ = meilleur contre une équipe AD) ; effet = pente × part retenue (preuve × fiabilité) :")
        for c in comp["candidates"][:12]:
            print(f"  {names.get(str(c['champion_id']), c['champion_id']):<12} pente {c['slope']:+.3f} ± "
                  f"{1.96 * c['se']:.3f}  part retenue {c['weight']:.0%}  → effet {c['effect']:+.3f}")

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
    p.add_argument("--until", default=None, metavar="AAAA-MM-JJ",
                   help="N'utiliser que les parties commencées avant cette date (UTC). Les parties "
                        "postérieures restent scellées pour une évaluation finale avec ml/evaluate.py.")
    p.add_argument("--out", default=str(DATA_DIR / "additive_model.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()

    df = load_matches()
    if args.until:
        df = df[df.game_creation < pd.Timestamp(args.until, tz="UTC").value // 1_000_000]
    if args.min_tier:
        allowed = TIER_ORDER[TIER_ORDER.index(args.min_tier):]
        df = df[df.source_tier.isin(allowed)]
    log.info("%d matchs chargés (patchs : %s)", len(df), df.patch.value_counts().head(4).to_dict())

    train_full, test = temporal_split(df, args.test_frac)

    # Évaluation : profils de dégâts mesurés sans la période de test
    eval_ad_share = load_champion_ad_share(before_ms=int(test.game_creation.min()))
    hp = tune(train_full, eval_ad_share)
    comp_decision = decide_comp(train_full, hp, eval_ad_share)
    results = report(train_full, test, hp, eval_ad_share, use_comp=comp_decision["adopted"])
    ad_share = load_champion_ad_share()

    log.info("Entraînement final sur les %d matchs...", len(df))
    model = AdditiveDraftModel.fit(df, hp, ad_share)
    comp_stats = None
    if model.damage is not None:
        comp_effects, comp_stats = fit_comp_effects(model, df)
        if comp_decision["adopted"]:
            model.effects.update(comp_effects)
    model.attach_stats(df, load_champion_names())
    # Écart réel entre joueurs sur un même champion : a priori de la personnalisation
    res = residuals(model, df, load_player_slots())
    personal_tau = estimate_tau(res)
    log.info("τ personnel mesuré : %.3f logit", personal_tau)
    # Coût d'inexpérience sur un champion, selon pick courant / de niche (a priori personnel)
    familiarity = estimate_familiarity(res.assign(t=res.match_id.map(df.game_creation)), role_shares(model))
    for kind, buckets in familiarity.items():
        log.info("  familiarité, pick %-6s : %s", kind, ", ".join(
            f"{b} {v['effect']:+.3f} ± {v['ci95']:.3f} (n={v['n']})" for b, v in sorted(buckets.items())))

    log.info("Calibration par tranche d'ELO (validation croisée sur les matchs étiquetés)...")
    tier_calibration = estimate_tier_calibration(df, hp, ad_share)
    for bucket, c in tier_calibration.items():
        log.info("  %-22s pente hors échantillon %.2f ± %.2f (n=%d) → facteur retenu %.2f",
                 TIER_BUCKET_LABELS[bucket], c["raw_slope"], 1.96 * c["se"], c["n"], c["scale"])

    model.meta = {
        "n_matches": len(df),
        "blue_winrate": float(df.blue_win.mean()),
        "min_tier": args.min_tier,
        "date_range": [int(df.game_creation.min()), int(df.game_creation.max())],
        "patches": df.patch.value_counts().to_dict(),
        "test_results": results,
        "personal_tau": personal_tau,
        "familiarity": familiarity,
        "tier_calibration": {b: c["scale"] for b, c in tier_calibration.items()},
        "tier_calibration_detail": tier_calibration,
        # Sensibilités au profil adverse : décision, a priori estimé et champions les mieux établis
        "comp": {} if comp_stats is None else {
            "adopted": comp_decision["adopted"],
            "validation_gain": comp_decision["gain"],
            "validation_ci95": comp_decision["ci95"],
            "prior": dict(comp_stats.attrs),
            "candidates": [
                {"champion_id": int(c), "slope": float(r.slope), "se": float(r.se),
                 "weight": float(r.weight), "effect": float(r.effect)}
                for c, r in comp_stats.head(20).iterrows()
            ],
        },
    }
    model.save(Path(args.out))
    print_top_effects(model)
    log.info("Modèle exporté → %s (%d effets, %.0fs)", args.out, len(model.effects), time.time() - t0)


if __name__ == "__main__":
    main()
