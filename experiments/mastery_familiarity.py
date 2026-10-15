"""
mastery_familiarity.py — La maîtrise de champion améliore-t-elle le décalage personnel ?

L'app classe un champion « jamais joué » s'il est absent de l'historique
analysé, et lui applique le coût d'inexpérience. Un champion beaucoup joué
autrefois (forte maîtrise) mais absent des parties récentes est donc pénalisé
à tort, si la maîtrise compte.

Protocole prospectif (fetch_masteries.py) : la maîtrise de chaque joueur du
ladder est photographiée à une date ; on n'évalue que ses parties jouées APRÈS
la photo, que la maîtrise ne contient donc pas. Le modèle de draft est celui de
l'app (entraîné avant ces parties) ; le décalage personnel δ est calculé comme
dans l'app, sur l'historique en base d'avant la photo.

  A. Résidu après δ selon l'historique (jamais / 1-2 / 3+ parties avant la
     photo) et la maîtrise : la maîtrise distingue-t-elle les « jamais joués » ?
  B. Gain de prédiction : δ + effets de classes de maîtrise (régression
     logistique à offset, légère L2), estimés sur 4/5 des joueurs et testés sur
     le 1/5 restant (validation croisée groupée par joueur : de nouveaux joueurs,
     comme un utilisateur de l'app).

Usage : python experiments/mastery_familiarity.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR
from ml.additive_model import AdditiveDraftModel
from ml.draft_data import TIER_TO_BUCKET, connect_read_only, load_matches, load_player_slots
from ml.personal import familiarity_bucket, is_niche, role_shares
from ml.train import log_losses

MASTERY_BINS = [-1, 0, 10_000, 50_000, 150_000, np.inf]
MASTERY_LABELS = ["aucune", "≤ 10k", "10-50k", "50-150k", "> 150k"]
HISTORY_LABELS = {"none": "jamais (historique)", "few": "1-2 parties", "regular": "3+ parties"}
L2 = 1.0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def clustered_mean(d: np.ndarray, groups: pd.Series) -> str:
    by = pd.DataFrame({"d": d, "g": groups.to_numpy()}).groupby("g").d.agg(["sum", "size"])
    n = by["size"].sum()
    mean = by["sum"].sum() / n
    se = np.sqrt(((by["sum"] - mean * by["size"]) ** 2).sum()) / n
    return f"{mean:+.5f} ± {1.96 * se:.5f}"


def fit_offset(x: np.ndarray, offset: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Coefficients b de logit = offset + x·b (L2 légère : peu de parties par classe)."""
    def obj(b):
        z = offset + x @ b
        return np.sum(np.logaddexp(0, z) - y * z) + 0.5 * L2 * b @ b
    def grad(b):
        return x.T @ (sigmoid(offset + x @ b) - y) + L2 * b
    return minimize(obj, np.zeros(x.shape[1]), jac=grad, method="L-BFGS-B").x


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    model = AdditiveDraftModel.load(DATA_DIR / "additive_model.json")
    trained_until = model.meta["date_range"][1]
    tau = model.meta["personal_tau"]
    fam = model.meta["familiarity"]
    calib = model.meta.get("tier_calibration", {})
    shares = role_shares(model)

    conn = connect_read_only()
    snaps = pd.read_sql_query("SELECT puuid, fetched_at FROM mastery_snapshots", conn).set_index("puuid").fetched_at
    mastery = pd.read_sql_query("SELECT puuid, champion_id, points, last_play_time FROM champion_mastery", conn)
    conn.close()

    df = load_matches()
    slots = load_player_slots()
    slots = slots[slots.puuid.isin(snaps.index) & slots.match_id.isin(df.index)].copy()
    slots["t"] = slots.match_id.map(df.game_creation)
    slots["snap"] = slots.puuid.map(snaps)

    # Probabilité de l'équipe du joueur : logit de la draft × calibration de la tranche (comme l'app)
    logit_blue = pd.Series(model.predict_logit(df), index=df.index)
    scale = df.source_tier.map(TIER_TO_BUCKET).map(calib).fillna(1.0)
    sign = np.where(slots.side == "blue", 1.0, -1.0)
    slots["logit"] = sign * (slots.match_id.map(logit_blue) * slots.match_id.map(scale)).to_numpy()
    blue_win = slots.match_id.map(df.blue_win).to_numpy()
    slots["y"] = np.where(slots.side == "blue", blue_win, 1 - blue_win).astype(float)
    slots["p"] = sigmoid(slots.logit)

    # Historique d'avant la photo (comme le profil importé dans l'app)
    before = slots[slots.t < slots.snap]
    hist = before.assign(r=before.y - before.p, v=before.p * (1 - before.p)) \
                 .groupby(["puuid", "champion_id"]).agg(S=("r", "sum"), V=("v", "sum"), n=("r", "size"))
    test = slots[(slots.t > slots.snap) & (slots.t > trained_until)]
    test = test.join(hist, on=["puuid", "champion_id"]).fillna({"S": 0.0, "V": 0.0, "n": 0})
    test = test.merge(mastery, on=["puuid", "champion_id"], how="left").fillna({"points": 0})
    test["n"] = test.n.astype(int)
    test["hbucket"] = test.n.map(familiarity_bucket)
    test["kind"] = ["niche" if is_niche(shares, c, pos) else "common" for c, pos in zip(test.champion_id, test.position)]
    test["mu"] = [fam.get(k, {}).get(h, {}).get("effect", 0.0) for k, h in zip(test.kind, test.hbucket)]
    test["delta"] = np.where(test.n > 0, test.mu + (test.S - test.mu * test.V) / (test.V + 1 / tau ** 2), test.mu)
    test["mastery"] = pd.cut(test.points, MASTERY_BINS, labels=MASTERY_LABELS)
    off = (test.logit + test.delta).to_numpy()
    y = test.y.to_numpy()
    print(f"{len(test)} parties de {test.puuid.nunique()} joueurs, jouées après leur photo de maîtrise "
          f"et après l'entraînement du modèle ({pd.to_datetime(trained_until, unit='ms').date()}).")
    print(f"Gain de δ (app) vs sans décalage : "
          f"{clustered_mean(log_losses(y, test.p.to_numpy()) - log_losses(y, sigmoid(off)), test.puuid)}")

    # A. Résidu après δ, en points de probabilité, à joueur égal (moyenne du joueur retirée)
    r = y - sigmoid(off)
    test["r_within"] = r - pd.Series(r).groupby(test.puuid.to_numpy()).transform("mean").to_numpy()
    print("\nA. Résultat réel − prédiction avec δ (points), à joueur égal, selon historique × maîtrise :")
    tab = test.groupby(["hbucket", "mastery"], observed=True).agg(
        n=("r_within", "size"), ecart=("r_within", "mean"), sd=("r_within", "std"))
    tab["ic"] = 1.96 * tab.sd / np.sqrt(tab.n)
    for (h, m), row in tab.iterrows():
        print(f"  {HISTORY_LABELS[h]:<22} maîtrise {m:<8} n={row.n:>6}  {100 * row.ecart:+.1f} ± {100 * row.ic:.1f} pts")

    # B. Gain de prédiction des classes de maîtrise, validation croisée groupée par joueur
    cells = (test.hbucket.astype(str) + "|" + test.mastery.astype(str))
    x = pd.get_dummies(cells).astype(float).to_numpy()
    players = test.puuid.unique()
    rng = np.random.default_rng(0)
    fold_of = dict(zip(players, rng.integers(0, 5, len(players))))
    fold = test.puuid.map(fold_of).to_numpy()
    pred = np.empty(len(test))
    for k in range(5):
        tr, te = fold != k, fold == k
        b = fit_offset(x[tr], off[tr], y[tr])
        pred[te] = sigmoid(off[te] + x[te] @ b)
    gain = log_losses(y, sigmoid(off)) - log_losses(y, pred)
    print("\nB. Gain de log-loss en ajoutant la maîtrise à δ (nouveaux joueurs, validation croisée groupée) :")
    print(f"  toutes : {clustered_mean(gain, test.puuid)}")
    never = (test.n == 0).to_numpy()
    print(f"  champions absents de l'historique : {clustered_mean(gain[never], test.puuid[never])}")
    # Variante économe : une pente sur log(1 + points), centrée, séparée selon que le champion est
    # dans l'historique ou non (2 paramètres au lieu d'une case par classe)
    lp = np.log1p(test.points.to_numpy())
    lp = lp - lp.mean()
    x2 = np.column_stack([lp * never, lp * ~never])
    pred2 = np.empty(len(test))
    for k in range(5):
        tr, te = fold != k, fold == k
        pred2[te] = sigmoid(off[te] + x2[te] @ fit_offset(x2[tr], off[tr], y[tr]))
    gain2 = log_losses(y, sigmoid(off)) - log_losses(y, pred2)
    b2 = fit_offset(x2, off, y)
    info = [np.sum(sigmoid(off + x2 @ b2) * (1 - sigmoid(off + x2 @ b2)) * x2[:, j] ** 2) for j in range(2)]
    print(f"  pente sur log(points), 2 paramètres : toutes {clustered_mean(gain2, test.puuid)} ; "
          f"absents de l'historique {clustered_mean(gain2[never], test.puuid[never])}")
    print(f"    pente (logit par unité de log-points) : absents {b2[0]:+.3f} ± {1.96 / np.sqrt(info[0]):.3f}, "
          f"présents {b2[1]:+.3f} ± {1.96 / np.sqrt(info[1]):.3f}")
    b = fit_offset(x, off, y)
    print("  effets estimés sur toutes les parties (logit, en plus de δ) :")
    for cell, coef in sorted(zip(pd.get_dummies(cells).columns, b)):
        print(f"    {cell:<30} {coef:+.3f}  (n={int((cells == cell).sum())})")


if __name__ == "__main__":
    main()
