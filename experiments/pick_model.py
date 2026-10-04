"""
pick_model.py — Prédire les picks à partir du reste de la draft (piste 3, étape 1).

Le recommandeur remplace chaque slot encore vide par les pick rates du rôle
(ceux de la tranche d'ELO si elle est connue), comme si les joueurs
choisissaient au hasard. Or un top adverse qui picke après le tien choisit
souvent un counter, un support s'accorde à son ADC, une équipe déjà full AD
complète en AP. Question : un modèle qui prédit le champion d'un slot à partir
des autres prédit-il les vrais picks mieux que les pick rates ?

Modèle : petit Transformer de type BERT. Une draft = 10 jetons (champion +
position côté × rôle) plus un jeton de contexte (tranche d'ELO, patch). On
masque une partie des slots au hasard et le modèle apprend à les retrouver.
L'API ne donne pas l'ordre des picks : il apprend quels champions vont
ensemble dans une partie, pas qui a contré qui.

Protocole :
  - Découpage temporel : test = 15 % des matchs les plus récents ; réglages
    (arrêt précoce, lissage des références) sur les 15 % les plus récents du
    reste, puis réentraînement sur tout le reste avec ces réglages.
  - Pour chaque match de test et chaque slot cible, k des 9 autres slots sont
    visibles (k uniforme de 0 à 9, slots au hasard, graine fixe) : toutes les
    situations, de la première pick à la dernière.
  - Les champions visibles et bannis sont exclus, comme dans le recommandeur.
  - Critère : log-vraisemblance négative du vrai champion (NLL, en nats).
    exp(NLL) = perplexité : nombre de champions équiprobables équivalent.
  - Références, toutes sans contexte de draft :
      pick rates globaux    (le recommandeur quand le tier est inconnu)
      pick rates par tier   (le recommandeur quand il est connu)
      pick rates du patch par tier, lissés (meilleur a priori sans contexte)

Usage : python experiments/pick_model.py [--dim 64] [--layers 2]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from ml.additive_model import patch_order, tracked_patches
from ml.draft_data import DRAFT_COLS, ROLES, TIER_TO_BUCKET, connect_read_only, load_matches
from ml.train import temporal_split

TIERS = ("LOW", "MID", "HIGH")       # Index 1..3 ; 0 = tier inconnu
ALPHAS = (100, 300, 1000, 3000, 10000)    # Lissage des pick rates hiérarchiques (réglé sur la validation)
K_GROUPS = {"0 visible": (0, 0), "1-4 visibles": (1, 4), "5-8 visibles": (5, 8), "9 visibles": (9, 9)}


# ── Données ───────────────────────────────────────────────────────────────────

def load_bans(index: pd.Index) -> np.ndarray:
    """(n_matchs, 10) champions bannis (−1 = pas de ban), dans l'ordre de `index`."""
    conn = connect_read_only()
    bans = pd.read_sql_query("SELECT match_id, team_id, pick_turn, champion_id FROM bans", conn)
    conn.close()
    bans["slot"] = (bans.team_id == 200) * 5 + (bans.pick_turn - 1) % 5
    wide = bans.pivot_table(index="match_id", columns="slot", values="champion_id", aggfunc="first")
    return wide.reindex(index=index, columns=range(10)).fillna(-1).astype(int).to_numpy()


class Encoding:
    """Champions → index 1..K (0 = masqué), tiers → 0..3, patchs suivis → 1..P (0 = autre)."""

    def __init__(self, df: pd.DataFrame, train: pd.DataFrame) -> None:
        champs = np.unique(df[DRAFT_COLS].to_numpy())
        self.lookup = np.zeros(int(champs.max()) + 1, dtype=np.int64)
        self.lookup[champs] = np.arange(1, len(champs) + 1)
        self.n_champs = len(champs) + 1
        self.patches = tracked_patches(train)

    def champs(self, ids: np.ndarray) -> np.ndarray:
        out = np.zeros(ids.shape, dtype=np.int64)
        known = (ids > 0) & (ids < len(self.lookup))
        out[known] = self.lookup[ids[known]]
        return out

    def tiers(self, df: pd.DataFrame) -> np.ndarray:
        return df.source_tier.map(TIER_TO_BUCKET).map({t: i + 1 for i, t in enumerate(TIERS)}).fillna(0).astype(int).to_numpy().copy()

    def patch_ids(self, df: pd.DataFrame) -> np.ndarray:
        """Patch inconnu à l'entraînement (plus récent) → dernier patch suivi."""
        index = {p: i + 1 for i, p in enumerate(self.patches)}
        newest = patch_order(self.patches[-1])
        return np.array([index.get(p, len(self.patches) if patch_order(p) > newest else 0) for p in df.patch],
                        dtype=np.int64)

    def tensors(self, df: pd.DataFrame, bans: np.ndarray) -> dict[str, torch.Tensor]:
        tokens, bans = self.champs(df[DRAFT_COLS].to_numpy().astype(int)), self.champs(bans)
        for j in range(bans.shape[1]):  # Ban incohérent (champion aussi joué) : ignoré
            bans[(bans[:, [j]] == tokens).any(1), j] = 0
        return {
            "tokens": torch.from_numpy(tokens),
            "bans": torch.from_numpy(bans),
            "tier": torch.from_numpy(self.tiers(df)),
            "patch": torch.from_numpy(self.patch_ids(df)),
        }


def eval_set(n_matches: int, seed: int) -> dict[str, np.ndarray]:
    """Une ligne par (match, slot cible) : k des 9 autres slots visibles, k uniforme de 0 à 9."""
    rng = np.random.default_rng(seed)
    match = np.repeat(np.arange(n_matches), 10)
    target = np.tile(np.arange(10), n_matches)
    k = rng.integers(0, 10, size=len(match))
    scores = rng.random((len(match), 10))
    scores[np.arange(len(match)), target] = 2.0  # La cible passe en dernier : jamais visible
    rank = scores.argsort(1).argsort(1)
    return {"match": match, "target": target, "k": k, "visible": rank < k[:, None]}


def unavailable_mask(visible_tokens: torch.Tensor, bans: torch.Tensor, n_champs: int) -> torch.Tensor:
    """Champions impossibles pour un slot vide : visibles, bannis, et l'index « masqué »."""
    mask = torch.zeros(len(visible_tokens), n_champs, dtype=torch.bool)
    mask.scatter_(1, visible_tokens, True)
    mask.scatter_(1, bans, True)
    mask[:, 0] = True
    return mask


# ── Références : pick rates ───────────────────────────────────────────────────

def counts_by(tokens: np.ndarray, n_champs: int) -> np.ndarray:
    """(5, n_champs) nombre de picks par rôle (deux équipes confondues)."""
    out = np.zeros((5, n_champs))
    for r in range(5):
        np.add.at(out[r], np.concatenate([tokens[:, r], tokens[:, r + 5]]), 1)
    return out


def pick_rate_tables(t: dict[str, torch.Tensor], n_champs: int, alpha: float) -> dict[str, np.ndarray]:
    """
    Tables (patch, tier, rôle, champion) de probabilités pour trois références.
    Lissage hiérarchique : global (Laplace 0,5) → patch → patch × tier, et
    global → tier ; `alpha` = poids en matchs de l'a priori.
    """
    tokens, tier, patch = t["tokens"].numpy(), t["tier"].numpy(), t["patch"].numpy()
    n_p, n_t = int(patch.max()) + 1, 4
    glob = counts_by(tokens, n_champs) + 0.5
    glob[:, 0] = 0
    glob /= glob.sum(1, keepdims=True)

    def smooth(counts, prior):
        return (counts + alpha * prior) / (counts.sum(1, keepdims=True) + alpha)

    by_tier = np.stack([glob if b == 0 else smooth(counts_by(tokens[tier == b], n_champs), glob) for b in range(n_t)])
    by_patch_tier = np.zeros((n_p, n_t, 5, n_champs))
    for p in range(n_p):
        p_prior = smooth(counts_by(tokens[patch == p], n_champs), glob)
        for b in range(n_t):
            sub = tokens[(patch == p) & (tier == b)] if b else tokens[patch == p]
            by_patch_tier[p, b] = smooth(counts_by(sub, n_champs), p_prior) if b else p_prior
    return {
        "pick rates globaux": np.broadcast_to(glob, (n_p, n_t, 5, n_champs)),
        "pick rates par tier": np.broadcast_to(by_tier, (n_p, n_t, 5, n_champs)),
        "pick rates patch × tier": by_patch_tier,
    }


def baseline_nll(table: np.ndarray, t: dict[str, torch.Tensor], ev: dict[str, np.ndarray], chunk: int = 50_000) -> np.ndarray:
    """NLL du vrai champion, probabilité renormalisée sur les champions disponibles."""
    tokens = t["tokens"].numpy()
    out = np.empty(len(ev["match"]))
    for s in range(0, len(out), chunk):
        sl = slice(s, s + chunk)
        m, tg = ev["match"][sl], ev["target"][sl]
        probs = table[t["patch"].numpy()[m], t["tier"].numpy()[m], tg % 5]
        visible = np.where(ev["visible"][sl], tokens[m], 0)
        unavail = unavailable_mask(torch.from_numpy(visible), t["bans"][m], table.shape[-1]).numpy()
        avail_mass = (probs * ~unavail).sum(1)
        out[sl] = -np.log(probs[np.arange(len(m)), tokens[m, tg]] / avail_mass)
    return out


# ── Modèle ────────────────────────────────────────────────────────────────────

class DraftBERT(nn.Module):

    def __init__(self, n_champs: int, n_patches: int, dim: int, layers: int, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.champ = nn.Embedding(n_champs, dim)
        self.slot = nn.Parameter(torch.randn(10, dim) * 0.02)
        self.tier = nn.Embedding(len(TIERS) + 1, dim)
        self.patch = nn.Embedding(n_patches + 1, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, n_champs)

    def forward(self, tokens, tier, patch) -> torch.Tensor:
        """Logits (B, 10, n_champs) pour chaque slot (les slots visibles sont ignorés en aval)."""
        x = self.champ(tokens) + self.slot
        context = (self.tier(tier) + self.patch(patch)).unsqueeze(1)
        h = self.encoder(torch.cat([context, x], 1))[:, 1:]
        return self.out(self.norm(h))


def masked_log_probs(model: DraftBERT, tokens_in, bans, tier, patch) -> torch.Tensor:
    logits = model(tokens_in, tier, patch)
    unavail = unavailable_mask(tokens_in, bans, logits.shape[-1])
    return torch.log_softmax(logits.masked_fill(unavail[:, None, :], float("-inf")), -1)


def train_epoch(model, opt, t: dict[str, torch.Tensor], batch: int, gen: torch.Generator) -> float:
    """Une passe : chaque match une fois, k ~ U{0..9} slots visibles tirés à nouveau."""
    model.train()
    n = len(t["tokens"])
    perm = torch.randperm(n, generator=gen)
    total = 0.0
    for s in range(0, n, batch):
        idx = perm[s:s + batch]
        tokens = t["tokens"][idx]
        k = torch.randint(0, 10, (len(idx), 1), generator=gen)
        visible = torch.rand(len(idx), 10, generator=gen).argsort(1).argsort(1) < k
        logp = masked_log_probs(model, tokens.where(visible, 0), t["bans"][idx], t["tier"][idx], t["patch"][idx])
        nll = -logp.gather(2, tokens.unsqueeze(2)).squeeze(2)
        # Chaque draft pèse autant, quel que soit son nombre de slots masqués
        loss = torch.where(visible, 0.0, nll).sum(1).div((~visible).sum(1)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(idx)
    return total / n


@torch.no_grad()
def model_nll(model, t: dict[str, torch.Tensor], ev: dict[str, np.ndarray], chunk: int = 8192) -> np.ndarray:
    model.eval()
    out = np.empty(len(ev["match"]))
    for s in range(0, len(out), chunk):
        sl = slice(s, s + chunk)
        m, tg = torch.from_numpy(ev["match"][sl]), torch.from_numpy(ev["target"][sl])
        tokens = t["tokens"][m]
        visible = torch.from_numpy(ev["visible"][sl])
        logp = masked_log_probs(model, tokens.where(visible, 0), t["bans"][m], t["tier"][m], t["patch"][m])
        rows = torch.arange(len(m))
        out[sl] = -logp[rows, tg, tokens[rows, tg]].numpy()
    return out


def fit_model(t, n_champs, n_patches, args, epochs: int | None = None,
              val: tuple | None = None) -> tuple[DraftBERT, int]:
    """Entraîne ; avec `val`, arrêt précoce (patience 3) et retour du meilleur nombre d'époques."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    model = DraftBERT(n_champs, n_patches, args.dim, args.layers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    best, best_state, best_epoch, epoch = np.inf, None, 0, 0
    while epoch < (epochs or args.max_epochs):
        epoch += 1
        t0 = time.time()
        loss = train_epoch(model, opt, t, args.batch, gen)
        msg = f"  époque {epoch:>2} : perte {loss:.4f}"
        if val is not None:
            v = model_nll(model, *val).mean()
            msg += f", validation {v:.4f}"
            if v < best - 1e-4:
                best, best_epoch = v, epoch
                best_state = {k: x.clone() for k, x in model.state_dict().items()}
            elif epoch - best_epoch >= 3:
                print(msg + f" ({time.time() - t0:.0f} s) → arrêt", flush=True)
                break
        print(msg + f" ({time.time() - t0:.0f} s)", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch or epoch


# ── Rapport ───────────────────────────────────────────────────────────────────

def paired_gain(base: np.ndarray, new: np.ndarray, match: np.ndarray) -> tuple[float, float]:
    """Gain moyen de NLL par pick, IC 95 % calculé par match (les 10 slots d'un match sont liés)."""
    d = pd.Series(base - new).groupby(match).mean()
    return float(d.mean()), float(1.96 * d.std(ddof=1) / np.sqrt(len(d)))


def report(results: dict[str, np.ndarray], ev: dict[str, np.ndarray], reference: str, title: str) -> None:
    print(f"\n{title} — {len(ev['match']) // 10} matchs, {len(ev['match'])} picks")
    print(f"{'':<26} {'NLL':>7} {'perplexité':>11} {'gain vs ' + reference:>36}")
    for name, nll in results.items():
        gain = "" if name == reference else "{:+.4f} ± {:.4f}".format(*paired_gain(results[reference], nll, ev["match"]))
        print(f"{name:<26} {nll.mean():>7.4f} {np.exp(nll.mean()):>11.1f} {gain:>36}")

    opposing_visible = ev["visible"][np.arange(len(ev["match"])), (ev["target"] + 5) % 10]
    groups = {label: (ev["k"] >= lo) & (ev["k"] <= hi) for label, (lo, hi) in K_GROUPS.items()}
    groups |= {"vis-à-vis visible": opposing_visible, "vis-à-vis caché": ~opposing_visible}
    groups |= {f"rôle {r}": ev["target"] % 5 == i for i, r in enumerate(ROLES)}
    names = [n for n in results if n != reference]
    print(f"\nGain par situation (NLL de « {reference} » − NLL de chaque méthode) :")
    print(f"{'':<20} {'NLL réf.':>9} " + " ".join(f"{n[:22]:>24}" for n in names))
    for label, mask in groups.items():
        cells = ["{:+.4f} ± {:.4f}".format(*paired_gain(results[reference][mask], results[n][mask], ev["match"][mask]))
                 for n in names]
        print(f"{label:<20} {results[reference][mask].mean():>9.4f} " + " ".join(f"{c:>24}" for c in cells))


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--max-epochs", type=int, default=60)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--save", default=None, help="Chemin où sauver le modèle final (entraîné hors test).")
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    df = load_matches()
    bans = load_bans(df.index)
    train, test = temporal_split(df, 0.15)
    fit, val = temporal_split(train, 0.15)
    print(f"{len(df)} matchs : réglage sur {len(fit)} → validation {len(val)} ; test {len(test)} "
          f"(depuis {pd.to_datetime(test.game_creation.min(), unit='ms'):%d/%m %H:%M})")
    print(f"Modèle : dim {args.dim}, {args.layers} couches, lr {args.lr}, lots de {args.batch}")

    def tensors(enc, part):
        return enc.tensors(part, bans[df.index.get_indexer(part.index)])

    # 1. Réglages sur la validation
    enc = Encoding(df, fit)
    t_fit, t_val = tensors(enc, fit), tensors(enc, val)
    ev_val = eval_set(len(val), seed=1)
    alpha_nll = {a: baseline_nll(pick_rate_tables(t_fit, enc.n_champs, a)["pick rates patch × tier"], t_val, ev_val).mean()
                 for a in ALPHAS}
    alpha = min(alpha_nll, key=alpha_nll.get)
    print("Lissage des pick rates (validation) : " + ", ".join(f"α={a} → {v:.4f}" for a, v in alpha_nll.items())
          + f" ; retenu α={alpha}")
    print("Entraînement avec arrêt précoce :")
    t0 = time.time()
    _, epochs = fit_model(t_fit, enc.n_champs, len(enc.patches), args, val=(t_val, ev_val))
    print(f"Meilleur nombre d'époques : {epochs} ({time.time() - t0:.0f} s)")

    # 2. Réentraînement sur tout le reste, évaluation sur le test
    enc = Encoding(df, train)
    t_train, t_test = tensors(enc, train), tensors(enc, test)
    ev = eval_set(len(test), seed=2)
    tables = pick_rate_tables(t_train, enc.n_champs, alpha)
    results = {name: baseline_nll(table, t_test, ev) for name, table in tables.items()}
    print(f"\nRéentraînement sur {len(train)} matchs, {epochs} époques :")
    model, _ = fit_model(t_train, enc.n_champs, len(enc.patches), args, epochs=epochs)
    results["Transformer (draft)"] = model_nll(model, t_test, ev)
    report(results, ev, "pick rates patch × tier", "Test")

    if args.save:
        torch.save({"state": model.state_dict(), "lookup": enc.lookup, "patches": enc.patches,
                    "args": vars(args), "epochs": epochs, "test_start": int(test.game_creation.min())}, args.save)
        print(f"\nModèle sauvé : {args.save}")


if __name__ == "__main__":
    main()
