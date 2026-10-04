"""
pick_model_win.py — Les picks prédits rendent-ils plus justes les probabilités
de victoire d'une draft incomplète ? (piste 3, étape 2)

Le recommandeur calcule l'espérance du logit sur les slots encore vides. Deux
façons de décrire un slot vide :
  - pick rates de la tranche d'ELO (actuel) ;
  - modèle de picks (experiments/pick_model.py) : distribution du champion
    sachant les slots déjà visibles. Chaque slot vide est conditionné sur les
    slots visibles, indépendamment des autres slots vides : l'espérance du
    logit additif reste exacte sous cette hypothèse.

Pour chaque match de test : une équipe au hasard (« alliée ») et k slots
visibles parmi 10 (k uniforme de 1 à 9). On compare :
  - l'écart au logit de la draft complète, (E[logit | visible] − logit)².
    L'espérance idéale est la meilleure prédiction de ce que dira le modèle une
    fois la draft finie : mesure sensible, sans le bruit du résultat ;
  - la log-loss du résultat réel (mesure finale, mais bruitée).

Modèle de victoire : modèle additif + matchups appris sur la part d'or, avec
les réglages du modèle actuel, entraîné sur les parties antérieures au test du
modèle de picks. Puis exemple : un premier pick top à l'aveugle.

Usage : python experiments/pick_model_win.py [--model experiments/results/pick_model_eval.pt]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR
from experiments.pick_model import DraftBERT, Encoding, load_bans, masked_log_probs
from ml.additive_model import AdditiveDraftModel
from ml.draft_data import DRAFT_COLS, ROLES, TIER_TO_BUCKET, load_champion_ad_share, load_champion_names, load_matches
from ml.lane_stats import lane_stat_layer, lane_targets, load_lane_gold
from ml.recommend import DraftRecommender, DraftState, sigmoid
from ml.train import log_losses, with_extra_effects

MIN_PROB = 1e-4  # Champions plus improbables retirés des distributions prédites (vitesse)
K_GROUPS = {"1-4 visibles": (1, 4), "5-8 visibles": (5, 8), "9 visibles": (9, 9)}


class PickPredictor:
    """Modèle de picks sauvé → distributions des slots vides d'une draft partielle."""

    def __init__(self, path: Path) -> None:
        ckpt = torch.load(path, weights_only=False)
        self.enc = Encoding.__new__(Encoding)
        self.enc.lookup, self.enc.patches = ckpt["lookup"], ckpt["patches"]
        self.enc.n_champs = int(ckpt["lookup"].max()) + 1
        self.model = DraftBERT(self.enc.n_champs, len(self.enc.patches), ckpt["args"]["dim"], ckpt["args"]["layers"])
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()
        known = np.nonzero(self.enc.lookup)[0]
        self.ids = np.zeros(self.enc.n_champs, dtype=int)
        self.ids[self.enc.lookup[known]] = known
        self.test_start = ckpt.get("test_start")

    @torch.no_grad()
    def log_probs(self, champs: np.ndarray, bans: np.ndarray, tier: np.ndarray, patch: np.ndarray,
                  chunk: int = 4096) -> np.ndarray:
        """(B, 10, K) log-probabilités par slot ; `champs` : ids des champions, 0 = slot vide."""
        out = []
        for s in range(0, len(champs), chunk):
            sl = slice(s, s + chunk)
            out.append(masked_log_probs(
                self.model, torch.from_numpy(self.enc.champs(champs[sl])), torch.from_numpy(self.enc.champs(bans[sl])),
                torch.from_numpy(tier[sl]), torch.from_numpy(patch[sl])).numpy())
        return np.concatenate(out)

    def dist(self, logp: np.ndarray) -> dict[int, float]:
        p = np.exp(logp)
        keep = np.nonzero(p >= MIN_PROB)[0]
        total = p[keep].sum()
        return {int(self.ids[i]): float(p[i] / total) for i in keep}


def slot_dists(ids: np.ndarray, visible: np.ndarray, logp: np.ndarray | None, pp: PickPredictor,
               ally_side: str) -> dict[tuple[str, str], dict[int, float]]:
    """Distributions (équipe, rôle) : masse 1 si visible, sinon modèle de picks (logp fourni)."""
    offsets = {"ally": 0 if ally_side == "blue" else 5, "enemy": 5 if ally_side == "blue" else 0}
    out = {}
    for team, off in offsets.items():
        for r, role in enumerate(ROLES):
            j = off + r
            out[(team, role)] = {int(ids[j]): 1.0} if visible[j] else pp.dist(logp[j])
    return out


def sigmoid_arr(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def paired(a: np.ndarray, b: np.ndarray) -> str:
    d = a - b
    return f"{d.mean():+.5f} ± {1.96 * d.std(ddof=1) / np.sqrt(len(d)):.5f}"


def blind_pick_demo(rec: DraftRecommender, pp: PickPredictor, role: str = "TOP", tier: str = "MID",
                    min_games: int = 300) -> None:
    """Premier pick à l'aveugle : classement avec pick rates vs avec le modèle de picks."""
    names = rec.names
    candidates = [int(c) for c, n in rec.model.games[role].items() if n >= min_games]
    r = ROLES.index(role)
    champs = np.zeros((len(candidates), 10), dtype=int)
    champs[:, r] = candidates
    tier_idx = np.full(len(candidates), ("LOW", "MID", "HIGH").index(tier) + 1)
    patch_idx = np.full(len(candidates), len(pp.enc.patches))
    logp = pp.log_probs(champs, np.zeros_like(champs), tier_idx, patch_idx)

    rows = []
    for i, c in enumerate(candidates):
        visible = np.zeros(10, dtype=bool)
        visible[r] = True
        d_pm = slot_dists(champs[i], visible, logp[i], pp, "blue")
        d_pr = rec._slot_dists(DraftState(ally={role: c}, ally_side="blue", tier=tier))
        p_pr = sigmoid(rec.expected_logit(d_pr, "blue", tier=tier))
        p_pm = sigmoid(rec.expected_logit(d_pm, "blue", tier=tier))
        enemy = d_pm[("enemy", role)]
        likely = sorted(enemy, key=lambda e: -enemy[e])[:3]
        rows.append((c, p_pr, p_pm, likely, enemy, d_pr[("enemy", role)]))

    print(f"\n{'═' * 78}\nEXEMPLE : premier pick {role} à l'aveugle (tier {tier}, côté bleu, rien d'autre de connu)\n{'═' * 78}")
    for label, idx in (("pick rates", 1), ("modèle de picks", 2)):
        top = sorted(rows, key=lambda x: -x[idx])[:10]
        print(f"Top 10 ({label}) : " + ", ".join(f"{names.get(x[0], x[0])} {x[idx]:.1%}" for x in top))
    rows.sort(key=lambda x: x[2] - x[1])
    for title, sel in (("Plus fortes baisses", rows[:8]), ("Plus fortes hausses", rows[::-1][:5])):
        print(f"\n{title} (écart modèle de picks − pick rates ; vis-à-vis le plus probable selon le modèle,"
              f" entre parenthèses son pick rate) :")
        for c, p_pr, p_pm, likely, enemy, base in sel:
            resp = ", ".join(f"{names.get(e, e)} {enemy[e]:.0%} ({base.get(e, 0):.0%})" for e in likely)
            print(f"  {names.get(c, c):<12} {p_pr:.1%} → {p_pm:.1%} ({(p_pm - p_pr) * 100:+.1f} pt)   face à : {resp}")


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "experiments" / "results" / "pick_model_eval.pt"))
    ap.add_argument("--since", default=None, help="Début du test (UTC), si le modèle sauvé ne le précise pas.")
    ap.add_argument("--max-matches", type=int, default=None)
    args = ap.parse_args()
    torch.set_num_threads(8)

    pp = PickPredictor(Path(args.model))
    if args.since:
        since = max(pd.Timestamp(args.since, tz="UTC").value // 1_000_000, pp.test_start or 0)
    elif pp.test_start:
        since = pp.test_start
    else:
        raise SystemExit("Début du test inconnu : préciser --since (après la fin de l'entraînement du modèle de picks).")
    df = load_matches()
    train, test = df[df.game_creation < since], df[df.game_creation >= since]
    if args.max_matches:
        test = test.iloc[:args.max_matches]
    print(f"Entraînement {len(train)} matchs, test {len(test)} (depuis {pd.to_datetime(since, unit='ms'):%d/%m %H:%M})")

    current = AdditiveDraftModel.load(DATA_DIR / "additive_model.json")
    model = AdditiveDraftModel.fit(train, current.hyperparams, load_champion_ad_share(before_ms=since))
    extra, beta, _ = lane_stat_layer(train, lane_targets(load_lane_gold(df)), model.predict_logit(train),
                                     current.meta["lane_stats"]["settings"])
    model = with_extra_effects(model, extra)
    model.attach_stats(train, load_champion_names())
    rec = DraftRecommender(model)
    print(f"Modèle de victoire prêt (matchups sur la part d'or, β = {beta:.2f})")

    rng = np.random.default_rng(0)
    n = len(test)
    ids = test[DRAFT_COLS].to_numpy().astype(int)
    bans = load_bans(test.index)
    sides = np.where(rng.integers(0, 2, n) == 0, "blue", "red")
    k = rng.integers(1, 10, n)
    visible = rng.random((n, 10)).argsort(1).argsort(1) < k[:, None]
    logp = pp.log_probs(np.where(visible, ids, 0), bans, pp.enc.tiers(test), pp.enc.patch_ids(test))
    y = np.where(sides == "blue", test.blue_win.to_numpy(), 1 - test.blue_win.to_numpy())
    tiers = test.source_tier.map(TIER_TO_BUCKET).to_numpy()

    e_pr, e_pm, full = np.empty(n), np.empty(n), np.empty(n)
    t0 = time.time()
    for i in range(n):
        side, tier = sides[i], (tiers[i] if isinstance(tiers[i], str) else None)
        own, opp = (0, 5) if side == "blue" else (5, 0)
        state = DraftState(
            ally={role: int(ids[i, own + r]) for r, role in enumerate(ROLES) if visible[i, own + r]},
            enemy={role: int(ids[i, opp + r]) for r, role in enumerate(ROLES) if visible[i, opp + r]},
            bans={int(b) for b in bans[i] if b > 0}, ally_side=side, tier=tier,
        )
        e_pr[i] = rec.expected_logit(rec._slot_dists(state), side, tier=tier)
        e_pm[i] = rec.expected_logit(slot_dists(ids[i], visible[i], logp[i], pp, side), side, tier=tier)
        full[i] = rec.expected_logit(slot_dists(ids[i], np.ones(10, dtype=bool), None, pp, side), side, tier=tier)
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1}/{n} drafts ({time.time() - t0:.0f} s)", flush=True)

    sq_pr, sq_pm = (e_pr - full) ** 2, (e_pm - full) ** 2
    ll_pr, ll_pm = log_losses(y, sigmoid_arr(e_pr)), log_losses(y, sigmoid_arr(e_pm))
    ll_const = log_losses(y, np.full(n, y.mean()))
    open_lane = np.array([
        any(visible[i, (0 if sides[i] == "blue" else 5) + r] and not visible[i, (5 if sides[i] == "blue" else 0) + r]
            for r in range(5))
        for i in range(n)
    ])
    groups = {"toutes": np.ones(n, dtype=bool)}
    groups |= {label: (k >= lo) & (k <= hi) for label, (lo, hi) in K_GROUPS.items()}
    groups |= {"un vis-à-vis allié caché": open_lane, "aucun": ~open_lane}

    print(f"\nÉcart au logit de la draft complète (moyenne de l'écart², + = le modèle de picks est plus proche) "
          f"et log-loss du résultat :")
    print(f"{'':<26} {'n':>6} {'écart² pick rates':>18} {'gain écart²':>22} {'gain log-loss':>22} {'réf. vs constante':>18}")
    for label, m in groups.items():
        print(f"{label:<26} {m.sum():>6} {sq_pr[m].mean():>18.5f} {paired(sq_pr[m], sq_pm[m]):>22} "
              f"{paired(ll_pr[m], ll_pm[m]):>22} {(ll_const[m] - ll_pr[m]).mean():>+18.5f}")
    print(f"(log-loss de la draft complète vs constante : {(ll_const - log_losses(y, sigmoid_arr(full))).mean():+.5f})")

    blind_pick_demo(rec, pp)


if __name__ == "__main__":
    main()
