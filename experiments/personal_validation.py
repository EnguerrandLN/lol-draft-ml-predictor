"""
personal_validation.py — Le décalage personnel prédit-il les parties FUTURES d'un joueur ?

L'app ajoute au logit de chaque candidat un décalage personnel (ml/personal.py) :
    δ = μ + (S − μ·V) / (V + 1/τ²)
μ : coût / gain de familiarité mesuré (aucune, 1-2, 3+ parties antérieures sur
le champion ; pick courant ou de niche), S = Σ(y − p) et V = Σ p(1 − p) sur
l'historique du joueur avec ce champion, τ : écart réel entre joueurs.
Ce calcul n'avait jamais été validé hors échantillon.

Protocole (comme l'usage réel : le profil est figé à l'import) :
  - modèle de draft, τ et μ estimés sur les parties d'avant la date de coupure ;
  - profils des joueurs construits sur leurs parties d'avant ;
  - test : leurs parties d'après, pour les joueurs ayant au moins 10 parties
    avant (comme le profil importé dans l'app, et comme l'estimation de μ) ;
  - on compare la log-loss de leur résultat sans décalage, avec μ seul, avec
    δ complet, et avec δ sans μ. IC groupé par joueur (ses lignes partagent δ).
  - Pente de δ : régression du résultat sur le logit de la draft (offset) + b·δ.
    b = 1 : décalage bien dosé ; b > 1 : trop prudent ; b < 1 : trop fort.

Usage : python experiments/personal_validation.py [--cutoff 2026-09-30]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR
from ml.additive_model import AdditiveDraftModel
from ml.draft_data import load_champion_ad_share, load_champion_names, load_matches, load_player_slots
from ml.personal import (
    FAMILIARITY_MIN_HISTORY, estimate_familiarity, estimate_tau, familiarity_bucket, is_niche, residuals,
    role_shares,
)
from ml.train import log_losses

PRIOR_GROUPS = {"champion jamais joué avant": (0, 0), "1-2 parties avant": (1, 2),
                "3-9 parties avant": (3, 9), "10+ parties avant": (10, 10 ** 6)}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clustered(d: pd.Series, groups: pd.Series) -> str:
    """Moyenne de d ± IC 95 % groupé : chaque groupe (joueur) compte pour une unité."""
    by = pd.DataFrame({"d": d.to_numpy(), "g": groups.to_numpy()}).groupby("g").d.agg(["sum", "size"])
    n = by["size"].sum()
    mean = by["sum"].sum() / n
    # Variance de la moyenne par linéarisation (estimateur « sandwich » pour un ratio de sommes)
    resid = by["sum"] - mean * by["size"]
    se = np.sqrt((resid ** 2).sum()) / n
    return f"{mean:+.5f} ± {1.96 * se:.5f}"


def offset_slope(logit: np.ndarray, delta: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """b et son erreur type dans logit P = logit_draft + b·δ."""
    nll = lambda b: np.sum(np.logaddexp(0, logit + b[0] * delta) - y * (logit + b[0] * delta))
    b = float(minimize(nll, np.ones(1), method="BFGS").x[0])
    p = sigmoid(logit + b * delta)
    return b, float(1 / np.sqrt(np.sum(p * (1 - p) * delta ** 2)))


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2026-09-30")
    args = ap.parse_args()
    cutoff = pd.Timestamp(args.cutoff, tz="UTC").value // 1_000_000

    df = load_matches()
    train, test = df[df.game_creation < cutoff], df[df.game_creation >= cutoff]
    hp = AdditiveDraftModel.load(DATA_DIR / "additive_model.json").hyperparams
    model = AdditiveDraftModel.fit(train, hp, load_champion_ad_share(before_ms=cutoff))
    model.attach_stats(train, load_champion_names())
    shares = role_shares(model)
    slots = load_player_slots()

    res_tr = residuals(model, train, slots)
    res_tr["t"] = res_tr.match_id.map(train.game_creation)
    tau = estimate_tau(res_tr)
    fam = estimate_familiarity(res_tr, shares)
    print(f"Coupure {args.cutoff} : modèle sur {len(train)} matchs, test sur {len(test)} matchs.")
    print(f"τ = {tau:.3f} ; μ (logit) : " + " ; ".join(
        f"{k} {b} {v['effect']:+.3f}" for k, buckets in fam.items() for b, v in sorted(buckets.items())))

    hist = res_tr.assign(r=res_tr.y - res_tr.p, v=res_tr.p * (1 - res_tr.p)) \
                 .groupby(["puuid", "champion_id"]).agg(S=("r", "sum"), V=("v", "sum"), n=("r", "size"))
    n_total = res_tr.groupby("puuid").size()
    eligible = n_total[n_total >= FAMILIARITY_MIN_HISTORY].index

    te = residuals(model, test, slots)
    te = te[te.puuid.isin(eligible)].join(hist, on=["puuid", "champion_id"]).fillna({"S": 0.0, "V": 0.0, "n": 0})
    te["n"] = te.n.astype(int)
    te["kind"] = ["niche" if is_niche(shares, c, pos) else "common" for c, pos in zip(te.champion_id, te.position)]
    te["mu"] = [fam.get(k, {}).get(familiarity_bucket(n), {}).get("effect", 0.0) for k, n in zip(te.kind, te.n)]
    shrink = 1 / (te.V + 1 / tau ** 2)
    te["delta"] = np.where(te.n > 0, te.mu + (te.S - te.mu * te.V) * shrink, te.mu)
    te["delta_no_mu"] = te.S * shrink
    logit = np.log(te.p / (1 - te.p)).to_numpy()
    y = te.y.to_numpy().astype(float)
    print(f"{te.puuid.nunique()} joueurs (≥ {FAMILIARITY_MIN_HISTORY} parties avant), {len(te)} parties de test ; "
          f"champion déjà joué avant : {(te.n > 0).mean():.0%} des parties.")

    ll = {"sans décalage": log_losses(y, te.p.to_numpy())}
    for label, col in (("μ seul (familiarité)", "mu"), ("δ complet (app)", "delta"), ("δ sans μ", "delta_no_mu")):
        ll[label] = log_losses(y, sigmoid(logit + te[col].to_numpy()))

    print("\nGain de log-loss par partie par rapport à « sans décalage » (IC groupé par joueur) :")
    groups = {"toutes": np.ones(len(te), dtype=bool)}
    groups |= {k: ((te.n >= lo) & (te.n <= hi)).to_numpy() for k, (lo, hi) in PRIOR_GROUPS.items()}
    groups |= {"pick de niche": (te.kind == "niche").to_numpy()}
    names = [k for k in ll if k != "sans décalage"]
    print(f"{'':<38} {'n':>7} " + " ".join(f"{k:>22}" for k in names))
    for label, m in groups.items():
        cells = [clustered(pd.Series(ll["sans décalage"][m] - ll[k][m]), te.puuid[m]) for k in names]
        print(f"{label:<38} {m.sum():>7} " + " ".join(f"{c:>22}" for c in cells))

    print("\nPente du décalage (1 = bien dosé) :")
    for col in ("mu", "delta", "delta_no_mu"):
        d = te[col].to_numpy()
        if np.any(d != 0):
            b, se = offset_slope(logit, d, y)
            print(f"  {col:<12} b = {b:.2f} ± {1.96 * se:.2f}   (écart-type du décalage {d.std():.3f} logit)")
    # À joueur égal : on retire la moyenne de chaque joueur (son niveau général). C'est ce qui
    # compte pour classer TES champions entre eux ; la pente brute mélange l'écart entre joueurs
    # (ceux qui jouent souvent des champions nouveaux sont aussi, en moyenne, moins forts).
    print("Pente à joueur égal (écarts au sein de chaque joueur) :")
    w = (te.p * (1 - te.p)).to_numpy()
    r = y - te.p.to_numpy()
    for col in ("mu", "delta", "delta_no_mu"):
        d = te[col] - te.groupby("puuid")[col].transform("mean")
        rc = r - pd.Series(r, index=te.index).groupby(te.puuid).transform("mean").to_numpy()
        info = float(np.sum(w * d.to_numpy() ** 2))
        if info > 0:
            print(f"  {col:<12} b = {np.sum(rc * d.to_numpy()) / info:.2f} ± {1.96 / np.sqrt(info):.2f}")
    known = (te.n > 0).to_numpy()
    b, se = offset_slope(logit[known], (te.delta - te.mu).to_numpy()[known], y[known])
    print(f"  partie « historique » de δ seule (δ − μ), champions déjà joués : b = {b:.2f} ± {1.96 * se:.2f}")


if __name__ == "__main__":
    main()
