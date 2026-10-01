"""
factorization_machine.py — Les synergies et counters « par ressemblance » apportent-ils quelque chose ?

Le modèle additif estime les interactions paire par paire (matchups de lane,
duos) : ~15 000 paires par rôle, ~3 parties chacune, aucun gain mesurable.
Une factorization machine donne à chaque champion de petits vecteurs appris
(k dimensions) et déduit toutes les interactions de ces vecteurs :
  - synergie de deux alliés a, b       = <v_a, v_b>
  - avantage de a sur un adversaire b   = <u_a, w_b> − <u_b, w_a>   (antisymétrique)
Deux champions au style proche (mêmes vecteurs) partagent ainsi leurs données :
~170 × 3k paramètres au lieu d'une valeur par paire. C'est l'idée centrale d'un
réseau de neurones (des embeddings de champions), contrainte et régularisée.

Protocole : le modèle actuel (additif, avec ses hyperparamètres) sert de base
FIXE ; la FM apprend ce qu'il reste à expliquer, sur les mêmes matchs
d'entraînement. Taille k et régularisation λ réglées sur une validation interne
(base ré-entraînée sans ces matchs) avec arrêt précoce. Évaluation sur deux
fenêtres de parties futures : gain de log-loss par rapport à la base seule,
IC 95 % apparié. Variantes : synergies seules, counters seuls, les deux.

Usage : python experiments/factorization_machine.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR
from ml.additive_model import AdditiveDraftModel
from ml.draft_data import ROLES, load_champion_ad_share, load_champion_names, load_matches
from ml.train import log_losses, temporal_split, with_comp

K_GRID = (2, 4, 8)
LAMBDA_GRID = (1e-5, 1e-4, 1e-3)
MAX_EPOCHS = 400
PATIENCE = 30
VARIANTS = ("synergies", "counters", "les deux")


class FactorizationMachine(torch.nn.Module):
    """Interactions de faible rang entre champions, ajoutées à un logit de base fixe."""

    def __init__(self, n_champions: int, k: int, synergy: bool, counter: bool) -> None:
        super().__init__()
        init = lambda: torch.nn.Parameter(0.01 * torch.randn(n_champions, k))
        self.synergy, self.counter = synergy, counter
        if synergy:
            self.v = init()
        if counter:
            self.u, self.w = init(), init()

    def forward(self, blue: torch.Tensor, red: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        logit = offset.clone()
        if self.synergy:
            def team_synergy(team):  # Σ_{i<j} <v_i, v_j> = (‖Σv‖² − Σ‖v‖²) / 2
                v = self.v[team]
                return 0.5 * (v.sum(1).pow(2).sum(1) - v.pow(2).sum((1, 2)))
            logit = logit + team_synergy(blue) - team_synergy(red)
        if self.counter:
            ub, wb, ur, wr = self.u[blue].sum(1), self.w[blue].sum(1), self.u[red].sum(1), self.w[red].sum(1)
            logit = logit + (ub * wr).sum(1) - (ur * wb).sum(1)
        return logit

    def penalty(self) -> torch.Tensor:
        return sum(p.pow(2).sum() for p in self.parameters())


def encode(df: pd.DataFrame, index: dict[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    enc = lambda side: torch.tensor(np.stack([df[f"{side}_{r}"].map(index).fillna(0).astype(int) for r in ROLES], 1))
    return enc("blue"), enc("red")


def train_fm(data, k, lam, variant, n_champ, epochs=None, val=None, seed=0):
    """Entraîne la FM ; avec `val`, arrêt précoce et retour du meilleur nombre d'époques."""
    torch.manual_seed(seed)
    fm = FactorizationMachine(n_champ, k, variant != "counters", variant != "synergies")
    opt = torch.optim.Adam(fm.parameters(), lr=0.01)
    blue, red, off, y = data
    best = (float("inf"), 0)
    for epoch in range(1, (epochs or MAX_EPOCHS) + 1):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(fm(blue, red, off), y) + lam * fm.penalty()
        loss.backward()
        opt.step()
        if val is not None:
            with torch.no_grad():
                vb, vr, voff, vy = val
                vloss = torch.nn.functional.binary_cross_entropy_with_logits(fm(vb, vr, voff), vy).item()
            if vloss < best[0] - 1e-7:
                best = (vloss, epoch)
            elif epoch - best[1] > PATIENCE:
                break
    return fm, best


def tensors(df, offset, index):
    blue, red = encode(df, index)
    return blue, red, torch.tensor(offset, dtype=torch.float32), torch.tensor(df.blue_win.to_numpy(), dtype=torch.float32)


def base_model(train, hp, ad_share):
    model = AdditiveDraftModel.fit(train, hp, ad_share)
    return with_comp(model, train) if model.damage is not None else model


def evaluate_window(train, test, hp, ad_share, index, names) -> None:
    n_champ = len(index) + 1
    t0 = time.time()

    # Réglage : base ré-entraînée sans la validation interne, puis grille (k, λ) avec arrêt précoce
    inner_train, inner_val = temporal_split(train, 0.15)
    inner_base = base_model(inner_train, hp, ad_share)
    tr = tensors(inner_train, inner_base.predict_logit(inner_train), index)
    va = tensors(inner_val, inner_base.predict_logit(inner_val), index)
    base_val = torch.nn.functional.binary_cross_entropy_with_logits(va[2], va[3]).item()

    base = base_model(train, hp, ad_share)
    full = tensors(train, base.predict_logit(train), index)
    te = tensors(test, base.predict_logit(test), index)
    y = test.blue_win.to_numpy()
    ll_base = log_losses(y, 1 / (1 + np.exp(-te[2].numpy())))

    print(f"  entraînement {len(train)} matchs → test {len(test)} matchs")
    for variant in VARIANTS:
        grid = [(train_fm(tr, k, lam, variant, n_champ, val=va)[1], k, lam) for k in K_GRID for lam in LAMBDA_GRID]
        (val_loss, epochs), k, lam = min(grid, key=lambda g: g[0][0])
        fm, _ = train_fm(full, k, lam, variant, n_champ, epochs=max(epochs, 1))
        with torch.no_grad():
            p = torch.sigmoid(fm(te[0], te[1], te[2])).numpy()
        diff = ll_base - log_losses(y, p)
        print(f"    {variant:<10} k={k} λ={lam:g} époques={epochs:<4} validation {base_val - val_loss:+.5f}"
              f"  | TEST {diff.mean():+.5f} ± {1.96 * diff.std(ddof=1) / np.sqrt(len(diff)):.5f}"
              f"  [{time.time() - t0:.0f}s]", flush=True)
        if variant == "les deux":
            show_top_interactions(fm, index, names)


def show_top_interactions(fm, index, names, top=6) -> None:
    ids = {i: c for c, i in index.items()}
    with torch.no_grad():
        if fm.synergy:
            syn = (fm.v @ fm.v.T).numpy()
            np.fill_diagonal(syn, 0)
            pairs = sorted(((syn[i, j], i, j) for i in range(1, len(syn)) for j in range(i + 1, len(syn))), reverse=True)
            print("      synergies les plus fortes :", ", ".join(
                f"{names.get(ids[i], i)}+{names.get(ids[j], j)} {s:+.3f}" for s, i, j in pairs[:top]))
        if fm.counter:
            adv = (fm.u @ fm.w.T - fm.w @ fm.u.T).numpy()   # avantage de i sur j
            pairs = sorted(((adv[i, j], i, j) for i in range(1, len(adv)) for j in range(1, len(adv)) if i != j), reverse=True)
            print("      counters les plus forts   :", ", ".join(
                f"{names.get(ids[i], i)}>{names.get(ids[j], j)} {s:+.3f}" for s, i, j in pairs[:top]))


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    torch.set_num_threads(max(1, torch.get_num_threads()))
    hp = AdditiveDraftModel.load(DATA_DIR / "additive_model.json").hyperparams
    ad_share = load_champion_ad_share()
    names = load_champion_names()
    df = load_matches()
    champs = sorted(set(np.unique(df[[f"{s}_{r}" for s in ("blue", "red") for r in ROLES]].to_numpy())))
    index = {int(c): i + 1 for i, c in enumerate(champs)}   # 0 réservé (champion absent de la base)
    n = len(df)
    print(f"{n} matchs, {len(index)} champions. Gain de log-loss par rapport au modèle additif seul (+ = mieux) :")
    for label, (start, end) in {
        "Fenêtre récente (15 % les plus récents)": (int(n * 0.85), n),
        "Fenêtre précédente (15 % d'avant)": (int(n * 0.70), int(n * 0.85)),
    }.items():
        print(f"\n{label}")
        evaluate_window(df.iloc[:start], df.iloc[start:end], hp, ad_share, index, names)


if __name__ == "__main__":
    main()
