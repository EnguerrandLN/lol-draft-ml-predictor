"""
decision_value.py — Suivre l'outil fait-il gagner ?

Jusqu'ici on a validé des PRÉDICTIONS (la proba d'une draft complète), pas des
CONSEILS. Ici, pour chaque partie d'un joueur jouée après l'entraînement du
modèle de l'app, on reconstitue ce que l'outil lui aurait affiché au moment de
son pick : les 9 autres champions connus (on n'a pas l'ordre des picks : c'est
la situation d'un dernier pick), les bans, son côté, son niveau, et son profil
personnel construit sur ses parties d'avant.

Le score que l'outil donne au champion réellement joué se décompose :
    logit = B  +  A_liste  +  A_contexte  +  δ
  B          : valeur de la draft avec un pick moyen (ne dépend pas du joueur) ;
  A_liste    : ce que vaut ce champion à ce rôle sans rien connaître de la
               draft (une simple « tier list ») ;
  A_contexte : ce que la draft change pour ce champion (matchup, synergies,
               composition) : c'est l'apport propre d'un conseiller de draft ;
  δ          : décalage personnel (historique et coût d'inexpérience).

  1. Validité : à joueur égal (effets fixes joueur, modèle linéaire de
     probabilité, erreurs groupées par joueur), le résultat suit-il chaque
     composante ? Pente 1 = quand l'outil annonce +1 point, le joueur gagne
     +1 point de plus ; pente 0 = le conseil ne vaut rien. Erreurs types
     groupées par joueur et par match.
  2. Marge : écart entre le meilleur champion du pool du joueur selon l'outil
     et celui qu'il a joué (« regret »). Gain attendu en suivant l'outil =
     regret moyen × pente mesurée.

Limites : données d'observation (le joueur choisit ce qu'il sent bien) ; les 9
autres picks sont supposés connus ; pool = champions joués ≥ 3 fois à ce rôle
dans l'historique en base.

Usage : python experiments/decision_value.py [--workers 6] [--max-rows N]
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR
from ml.additive_model import AdditiveDraftModel
from ml.draft_data import ROLES, TIER_TO_BUCKET, connect_read_only, load_matches, load_player_slots
from ml.personal import ChampionProfile, personal_offsets, residuals
from ml.recommend import DraftRecommender, DraftState

MODEL_PATH = DATA_DIR / "additive_model.json"
POOL_MIN_GAMES = 3       # Comme l'option « Seulement ceux que je joue » de l'app
MIN_ROLE_GAMES = 50      # Seuil par défaut de l'app pour proposer un champion
MIN_TEST_GAMES = 5       # Parties de test par joueur (comparaison à joueur égal)
_G: dict = {}


def logit(p: float) -> float:
    return float(np.log(p / (1 - p)))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ── Calcul par partie (processus de travail) ──────────────────────────────────

def _init() -> None:
    model = AdditiveDraftModel.load(MODEL_PATH)
    rec = DraftRecommender(model)
    tier_list = {}
    for tier in (None, "LOW", "MID", "HIGH"):
        for role in ROLES:
            res = rec.recommend(DraftState(tier=tier), role, min_games=0)
            avg = logit(res[0].win_prob - res[0].vs_average)
            tier_list[(tier, role)] = {r.champion_id: r.score - avg for r in res}
    _G.update(model=model, rec=rec, tier_list=tier_list)


def _process(chunk: list[tuple]) -> list[tuple]:
    model, rec, tier_list = _G["model"], _G["rec"], _G["tier_list"]
    out = []
    for idx, role, side, tier, ally, enemy, bans, champ, prof in chunk:
        profiles = {c: ChampionProfile(c, n, 0, 0.0, s, v, by_role) for c, (n, s, v, by_role) in prof.items()}
        offsets = personal_offsets(model, role, profiles)
        res = rec.recommend(DraftState(ally=ally, enemy=enemy, bans=bans, ally_side=side, tier=tier),
                            role, min_games=0, personal=offsets)
        by = {r.champion_id: r for r in res}
        if champ not in by or champ not in tier_list[(tier, role)]:
            continue
        me = by[champ]
        delta = offsets.get(champ, 0.0)
        base = logit(me.win_prob - me.vs_average)                       # Draft avec un pick moyen
        pool = {c for c, (_, _, _, by_role) in prof.items() if by_role.get(role, 0) >= POOL_MIN_GAMES and c in by}
        pool.add(champ)
        best_pool = max(by[c].score for c in pool)
        best_all = max(r.score for r in res if r.games >= MIN_ROLE_GAMES or r.champion_id in pool)
        known = prof.get(champ, (0, 0, 0, {}))[3].get(role, 0) >= POOL_MIN_GAMES
        out.append((idx, base, me.score - delta, delta, tier_list[(tier, role)][champ], len(pool), best_pool, best_all, known))
    return out


# ── Statistiques ──────────────────────────────────────────────────────────────

def within(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return (df[cols] - df.groupby("puuid")[cols].transform("mean")).to_numpy()


def ols_clustered(x: np.ndarray, y: np.ndarray, players: np.ndarray, matches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Moindres carrés et erreurs types groupées à deux niveaux (sandwich) : par joueur
    (ses parties se ressemblent) et par match (plusieurs joueurs suivis partagent
    le même résultat). V = V_joueur + V_match − V_ligne.
    """
    xtx_inv = np.linalg.inv(x.T @ x)
    b = xtx_inv @ x.T @ y
    scores = pd.DataFrame(x * (y - x @ b)[:, None])

    def meat(groups) -> np.ndarray:
        g = scores.groupby(groups).sum().to_numpy()
        return g.T @ g

    v = xtx_inv @ (meat(players) + meat(matches) - meat(np.arange(len(y)))) @ xtx_inv
    return b, np.sqrt(np.maximum(np.diag(v), 0))


def slopes(d: pd.DataFrame, cols: list[str], labels: list[str]) -> np.ndarray:
    x = within(d, cols)
    y = within(d, ["y"])[:, 0]
    b, se = ols_clustered(x, y, d.puuid.to_numpy(), d.match_id.to_numpy())
    scale = float((d.p * (1 - d.p)).mean())      # Probabilité → logit
    for label, col, bi, si in zip(labels, cols, b, se):
        print(f"    {label:<34} pente {bi / scale:+.2f} ± {1.96 * si / scale:.2f}   "
              f"(écart-type à joueur égal : {100 * scale * x[:, cols.index(col)].std():.2f} pt)")
    return b / scale


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-rows", type=int, default=0)
    args = ap.parse_args()

    model = AdditiveDraftModel.load(MODEL_PATH)
    until = model.meta["date_range"][1]
    df = load_matches()
    hist_df, test_df = df[df.game_creation <= until], df[df.game_creation > until]
    conn = connect_read_only()
    ladder = {r[0] for r in conn.execute("SELECT puuid FROM ladder_players")}
    bans = pd.read_sql_query("SELECT match_id, champion_id FROM bans WHERE champion_id > 0", conn)
    conn.close()
    bans = bans[bans.match_id.isin(test_df.index)].groupby("match_id").champion_id.agg(set).to_dict()

    slots = load_player_slots()
    slots = slots[slots.puuid.isin(ladder) & slots.position.isin(ROLES)]
    test = slots[slots.match_id.isin(test_df.index)]
    counts = test.groupby("puuid").size()
    test = test[test.puuid.isin(counts[counts >= MIN_TEST_GAMES].index)].reset_index(drop=True)
    if args.max_rows:
        test = test.iloc[:args.max_rows]

    # Profil de chaque joueur sur ses parties d'avant (comme le profil importé dans l'app)
    res = residuals(model, hist_df, slots[slots.puuid.isin(test.puuid.unique())])
    res = res.assign(r=res.y - res.p, v=res.p * (1 - res.p))
    agg = res.groupby(["puuid", "champion_id"]).agg(n=("r", "size"), s=("r", "sum"), v=("v", "sum"))
    by_role = res.groupby(["puuid", "champion_id", "position"]).size()
    roles_of: dict = {}
    for (puuid, champ, pos), n in by_role.items():
        roles_of.setdefault((puuid, champ), {})[pos] = int(n)
    profiles: dict = {}
    for (puuid, champ), row in agg.iterrows():
        profiles.setdefault(puuid, {})[int(champ)] = (int(row.n), float(row.s), float(row.v), roles_of[(puuid, champ)])

    tasks = []
    for i, r in enumerate(test.itertuples(index=False)):
        m = test_df.loc[r.match_id]
        other = "red" if r.side == "blue" else "blue"
        tasks.append((
            i, r.position, r.side, TIER_TO_BUCKET.get(m.source_tier),
            {role: int(m[f"{r.side}_{role}"]) for role in ROLES if role != r.position},
            {role: int(m[f"{other}_{role}"]) for role in ROLES},
            bans.get(r.match_id, set()), int(r.champion_id), profiles.get(r.puuid, {}),
        ))
    print(f"Modèle entraîné jusqu'au {pd.to_datetime(until, unit='ms').date()} ; {len(tasks)} parties de "
          f"{test.puuid.nunique()} joueurs du ladder (≥ {MIN_TEST_GAMES} parties de test chacun).", flush=True)

    chunks = [tasks[i:i + 500] for i in range(0, len(tasks), 500)]
    with ProcessPoolExecutor(args.workers, initializer=_init) as pool:
        rows = [row for part in pool.map(_process, chunks) for row in part]
    d = pd.DataFrame(rows, columns=["i", "B", "draft", "delta", "A_list", "pool", "best_pool", "best_all", "known"]).set_index("i")
    d = d.join(test[["puuid", "match_id", "side"]])
    blue_win = d.match_id.map(test_df.blue_win)
    d["y"] = np.where(d.side == "blue", blue_win, 1 - blue_win).astype(float)
    d["A_draft"] = d.draft - d.B
    d["A_ctx"] = d.A_draft - d.A_list
    d["score"] = d.draft + d.delta
    d["p"] = sigmoid(d.score)
    d = d[d.groupby("puuid").y.transform("size") >= 2]
    print(f"{len(d)} parties analysées ; {d.match_id.duplicated().mean():.0%} partagent un match avec une autre ligne.")

    print("\n1. Le résultat suit-il le score de l'outil ? (à joueur égal ; pente 1 = bien dosé)")
    print("  a) score du pick en un bloc :")
    d["A_total"] = d.A_draft + d.delta
    total = slopes(d, ["B", "A_total"], ["reste de la draft (B)", "pick joué (A_liste + A_contexte + δ)"])
    print("  b) par composante :")
    slopes(d, ["B", "A_list", "A_ctx", "delta"],
           ["reste de la draft (B)", "tier list (A_liste)", "apport de la draft (A_contexte)", "décalage personnel (δ)"])

    print("\n2. Marge de progression : meilleur pick selon l'outil − pick joué")
    multi = d[d["pool"] >= 2]
    regret = sigmoid(multi.best_pool) - multi.p
    followed = regret <= 1e-9
    print(f"  Parties où le joueur avait au moins 2 champions dans son pool : {len(multi)} ({len(multi) / len(d):.0%}), "
          f"pool médian {int(multi['pool'].median())} champions.")
    print(f"  Il a joué le pick n°1 de l'outil dans {followed.mean():.0%} des cas ; sinon regret moyen "
          f"{100 * regret[~followed].mean():.2f} pt.")
    print(f"  Regret moyen dans son pool : {100 * regret.mean():.2f} pt de proba prédite "
          f"→ gain attendu en suivant l'outil ≈ {100 * regret.mean() * total[1]:.2f} pt (regret × pente du pick).")
    known = multi[multi.known]
    regret_known = sigmoid(known.best_pool) - known.p
    print(f"  Quand le pick joué est lui-même dans son pool ({len(known) / len(multi):.0%} des cas ; seule la draft "
          f"départage) : pick n°1 joué {(regret_known <= 1e-9).mean():.0%}, regret moyen {100 * regret_known.mean():.2f} pt "
          f"→ ≈ {100 * regret_known.mean() * total[1]:.2f} pt.")
    regret_all = sigmoid(d.best_all) - d.p
    print(f"  Parmi tous les champions (coût d'inexpérience compris) : regret moyen {100 * regret_all.mean():.2f} pt "
          f"→ ≈ {100 * regret_all.mean() * total[1]:.2f} pt.")

    print("\n3. Résultat réel selon l'écart au conseil (pool ≥ 2 ; à joueur égal ; en points, relatif au reste de la draft) :")
    m2 = multi.assign(regret=regret, pred=multi.p - sigmoid(multi.B), obs=multi.y - sigmoid(multi.B))
    dm = m2[["pred", "obs"]] - m2.groupby("puuid")[["pred", "obs"]].transform("mean")
    classes = pd.cut(m2.regret, [-1, 1e-9, 0.01, 0.02, 1], labels=["pick n°1 joué", "regret ≤ 1 pt", "1-2 pts", "> 2 pts"])
    for label, g in dm.groupby(classes, observed=True):
        print(f"  {label:<14} n={len(g):>6}  prédit {100 * g.pred.mean():+.2f}  observé {100 * g.obs.mean():+.2f} "
              f"± {196 * g.obs.std() / np.sqrt(len(g)):.2f}")


if __name__ == "__main__":
    main()
