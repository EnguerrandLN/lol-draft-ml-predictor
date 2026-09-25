"""
train_model.py — Architecture PyTorch pour l'estimateur de draft (MLM).

Principe :
  Pour chaque ligne du dataset, le modèle reçoit en entrée le contexte de draft
  (alliés, ennemis, bans) et doit prédire le champion masqué (target_champion_id).

  Tous les champion IDs passent dans une Embedding partagée. Le slot masqué (-1)
  est traité comme un token de padding (index 0 → vecteur nul) : il n'apporte
  aucune information au modèle, ce qui implémente le masquage MLM.

Architecture :
  Input (20 IDs) → Embedding partagée (dim=32) → Flatten (640)
  → FC(640→256) + ReLU + Dropout → FC(256→128) + ReLU + Dropout
  → FC(128→N_champions)  [logits, CrossEntropyLoss]

Usage :
  python train_model.py
  python train_model.py --epochs 20 --batch-size 512 --embed-dim 64
  python train_model.py --dataset data/mlm_draft_dataset.csv --device cuda
"""
import argparse
import logging
import sqlite3
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH, DATA_DIR

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Colonnes du dataset ───────────────────────────────────────────────────────

ALLY_COLS  = ["ally_top", "ally_jungle", "ally_mid", "ally_adc", "ally_supp"]
ENEMY_COLS = ["enemy_top", "enemy_jungle", "enemy_mid", "enemy_adc", "enemy_supp"]
BAN_COLS   = [f"ban_{i}" for i in range(1, 11)]
FEATURE_COLS = ALLY_COLS + ENEMY_COLS + BAN_COLS   # 5 + 5 + 10 = 20 colonnes

# Index du token spécial "aucun champion" (masqué ou absent)
PAD_IDX = 0


# ── Noms des champions ────────────────────────────────────────────────────────

def load_champion_names(db_path=DB_PATH) -> dict[int, str]:
    """
    Charge le mapping {champion_id: champion_name} depuis la base SQLite.
    Utilisé pour afficher des noms lisibles à la place des IDs bruts.
    """
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT DISTINCT champion_id, champion_name FROM participants "
            "WHERE champion_name IS NOT NULL AND champion_id > 0"
        ).fetchall()
        conn.close()
        names = {int(cid): name for cid, name in rows if name}
        log.info("Noms chargés depuis la DB : %d champions", len(names))
        return names
    except Exception as exc:
        log.warning("Impossible de charger les noms depuis la DB : %s", exc)
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────────────────────

class ChampionEncoder:
    """
    Encode les champion IDs Riot (discontinus) vers des indices continus [1, N].

    L'index 0 est réservé au token de padding (champion masqué -1 ou absent 0).
    Ainsi la couche nn.Embedding peut utiliser padding_idx=0.

    Exemple :
      IDs Riot : [1, 51, 235, 800, ...]  (non contigus)
      Encodés  : {1→1, 51→2, 235→3, 800→4, ...}
      -1 ou 0  : → 0  (padding)
    """

    def __init__(self, champion_ids: set[int]) -> None:
        sorted_ids = sorted(champion_ids)
        # idx 0 réservé au padding → champions commencent à 1
        self.id_to_idx: dict[int, int] = {
            cid: idx + 1 for idx, cid in enumerate(sorted_ids)
        }
        self.idx_to_id: dict[int, int] = {v: k for k, v in self.id_to_idx.items()}
        self.n_champions: int = len(sorted_ids)     # Sans le padding
        self.vocab_size:  int = len(sorted_ids) + 1 # +1 pour idx 0
        # Noms (peuplé via attach_names)
        self.id_to_name:  dict[int, str] = {}

    def attach_names(self, names: dict[int, str]) -> None:
        """Attache le mapping champion_id → nom lisible."""
        self.id_to_name = names

    def encode(self, champion_id: int) -> int:
        """Convertit un ID Riot en index continu. -1 et 0 → PAD_IDX."""
        if champion_id <= 0:
            return PAD_IDX
        return self.id_to_idx.get(champion_id, PAD_IDX)

    def decode(self, idx: int) -> int:
        """Reconvertit un index en ID Riot."""
        return self.idx_to_id.get(idx, 0)

    def decode_name(self, idx: int) -> str:
        """
        Reconvertit un index en nom de champion lisible.
        Retourne 'Unknown (id=X)' si le nom n'est pas disponible.
        """
        riot_id = self.decode(idx)
        if riot_id == 0:
            return "[masqué]"
        name = self.id_to_name.get(riot_id)
        return name if name else f"id={riot_id}"

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, cols: list[str]) -> "ChampionEncoder":
        """Construit l'encodeur à partir de toutes les colonnes pertinentes."""
        all_ids: set[int] = set()
        for col in cols:
            all_ids.update(df[col].dropna().astype(int).unique())
        # Retire les valeurs spéciales (padding / masque)
        all_ids.discard(0)
        all_ids.discard(-1)
        log.info("ChampionEncoder : %d champions uniques détectés", len(all_ids))
        return cls(all_ids)


class DraftDataset(Dataset):
    """
    Dataset PyTorch pour le draft MLM.

    Chaque sample est un dict contenant :
      - features : LongTensor (20,) — IDs encodés [ally×5, enemy×5, ban×10]
      - target   : LongTensor (1,)  — index encodé du champion cible
      - win      : FloatTensor (1,) — 1.0 = victoire (optionnel, pour pondération)

    Args:
        csv_path : Chemin vers mlm_draft_dataset.csv.
        encoder  : ChampionEncoder pré-construit (ou None pour le créer ici).
    """

    def __init__(
        self,
        csv_path: str | Path,
        encoder: ChampionEncoder | None = None,
    ) -> None:
        log.info("Chargement du dataset : %s", csv_path)
        df = pd.read_csv(csv_path)
        log.info("  %d lignes × %d colonnes", len(df), len(df.columns))

        # ── Construction ou réutilisation de l'encodeur ──────────────────────
        all_id_cols = FEATURE_COLS + ["target_champion_id"]
        if encoder is None:
            self.encoder = ChampionEncoder.from_dataframe(df, all_id_cols)
        else:
            self.encoder = encoder

        log.info(
            "  Vocabulaire : %d champions | vocab_size = %d (+ 1 padding)",
            self.encoder.n_champions, self.encoder.vocab_size,
        )

        # ── Encodage des features (20 colonnes) ──────────────────────────────
        # -1 (masqué) et 0 (absent) → PAD_IDX = 0 via encoder.encode()
        feature_matrix = df[FEATURE_COLS].fillna(0).astype(int)
        # .map() remplace .applymap() depuis pandas 2.1
        encoded_features = feature_matrix.map(self.encoder.encode).values
        self.features = torch.tensor(encoded_features, dtype=torch.long)

        # ── Encodage de la cible ──────────────────────────────────────────────
        targets_raw = df["target_champion_id"].astype(int).values
        targets_enc = np.vectorize(self.encoder.encode)(targets_raw)
        self.targets = torch.tensor(targets_enc, dtype=torch.long)

        # ── Variable win (conditionnement) ────────────────────────────────────
        self.wins = torch.tensor(df["target_win"].values, dtype=torch.float32).unsqueeze(1)

        # ── Position (Embedding de Rôle) ──────────────────────────────────────
        ROLE_TO_IDX = {"TOP": 0, "JUNGLE": 1, "MIDDLE": 2, "BOTTOM": 3, "UTILITY": 4}
        positions_idx = df["target_position"].map(ROLE_TO_IDX).fillna(0).astype(int).values
        self.positions = torch.tensor(positions_idx, dtype=torch.long)

        log.info("  Dataset prêt : %d samples", len(self))

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],   # (20,) LongTensor
            "target":   self.targets[idx],     # () LongTensor  (scalar)
            "win":      self.wins[idx],        # () FloatTensor (scalar)
            "position": self.positions[idx],   # () LongTensor  (scalar)
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODÈLE
# ─────────────────────────────────────────────────────────────────────────────

class DraftModel(nn.Module):
    """
    Réseau de prédiction de champion pour la phase de draft (avec Self-Attention).

    Toutes les entrées (alliés, ennemis, bans) passent dans une Embedding partagée.
    Le Transformer permet aux champions d'interagir entre eux dans la séquence pour
    comprendre les synergies et les counters de la composition d'équipe.
    """

    def __init__(
        self,
        vocab_size:    int,
        embedding_dim: int = 32,
        hidden_dim:    int = 256,
        dropout:       float = 0.4,
        n_inputs:      int = 20,
    ) -> None:
        super().__init__()

        self.vocab_size    = vocab_size
        self.embedding_dim = embedding_dim
        self.n_inputs      = n_inputs
        flat_dim = n_inputs * embedding_dim    # 20 × 32 = 640

        # ── Embedding partagée ────────────────────────────────────────────────
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=PAD_IDX,
        )

        # ── Self-Attention (Transformer) ──────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # ── Embedding de rôle ─────────────────────────────────────────────────
        self.role_embedding = nn.Embedding(5, 8)

        # ── Réseau Feed-Forward ───────────────────────────────────────────────
        self.network = nn.Sequential(
            nn.Linear(flat_dim + 1 + 8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Couche de sortie : logits sur tout le vocabulaire
            nn.Linear(hidden_dim // 2, vocab_size),
        )

    def forward(self, x: torch.Tensor, target_win: torch.Tensor, target_position: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : LongTensor (B, 20) — IDs encodés des champions en contexte.
            target_win : FloatTensor (B, 1) — condition de victoire.
            target_position: LongTensor (B,) — entier du rôle cible.

        Returns:
            logits : FloatTensor (B, vocab_size) — score brut par champion.
        """
        # 1. Obtenir les embeddings de la séquence (B, 20, 32)
        embedded = self.embedding(x)
        
        # 2. Permettre aux champions de s'observer (Self-Attention)
        attended = self.transformer(embedded) # (B, 20, 32)
        
        # 3. Aplatir le contexte enrichi
        embedded_flat = attended.view(attended.size(0), -1) # (B, 640)
        
        # 4. Calculer le rôle
        role_embed = self.role_embedding(target_position) # (B, 8)
        
        # 5. Injection globale et prise de décision
        combined = torch.cat((embedded_flat, target_win, role_embed), dim=1) # (B, 649)
        logits   = self.network(combined)   # (B, vocab_size)
        return logits

    def predict_top_k(
        self, x: torch.Tensor, target_win: torch.Tensor, target_position: torch.Tensor, k: int = 5
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retourne les k champions les plus probables pour chaque sample.
        """
        with torch.no_grad():
            logits = self.forward(x, target_win, target_position)
            probs  = torch.softmax(logits, dim=-1)
        return torch.topk(probs, k=k, dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. BOUCLE D'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model:     DraftModel,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    """
    Effectue une epoch d'entraînement complète.

    Returns:
        (avg_loss, accuracy) sur l'ensemble d'entraînement.
    """
    model.train()
    total_loss  = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        features  = batch["features"].to(device)   # (B, 20)
        wins      = batch["win"].to(device)        # (B, 1)
        positions = batch["position"].to(device)   # (B,)
        targets   = batch["target"].to(device)     # (B,)

        optimizer.zero_grad()
        logits = model(features, wins, positions)                   # (B, vocab_size)
        loss   = criterion(logits, targets)
        loss.backward()
        # Gradient clipping : stabilise l'entraînement
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss    += loss.item() * len(targets)
        preds          = logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += len(targets)

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate(
    model:     DraftModel,
    loader:    DataLoader,
    criterion: nn.Module,
    device:    torch.device,
) -> tuple[float, float]:
    """Évalue le modèle sans mise à jour des gradients."""
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        features  = batch["features"].to(device)
        wins      = batch["win"].to(device)
        positions = batch["position"].to(device)
        targets   = batch["target"].to(device)
        logits    = model(features, wins, positions)
        loss     = criterion(logits, targets)

        total_loss    += loss.item() * len(targets)
        preds          = logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += len(targets)

    return total_loss / total_samples, total_correct / total_samples


# ─────────────────────────────────────────────────────────────────────────────
# 4. ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Entraîne le DraftModel MLM sur mlm_draft_dataset.csv",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",    default=str(DATA_DIR / "mlm_draft_dataset.csv"))
    p.add_argument("--model-out",  default=str(DATA_DIR / "draft_model.pt"),
                   help="Chemin de sauvegarde du modèle entraîné.")
    p.add_argument("--epochs",     type=int,   default=10)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--embed-dim",  type=int,   default=32)
    p.add_argument("--hidden-dim", type=int,   default=256)
    p.add_argument("--dropout",    type=float, default=0.3)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--val-split",  type=float, default=0.15,
                   help="Fraction des données pour la validation.")
    p.add_argument("--device",     default="auto",
                   choices=["auto", "cpu", "cuda", "mps"],
                   help="Device PyTorch. 'auto' choisit le meilleur disponible.")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    log.info("Device : %s", device)

    # ── Reproductibilité ──────────────────────────────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = DraftDataset(args.dataset)
    encoder = dataset.encoder

    # Attache les noms lisibles à l'encodeur
    encoder.attach_names(load_champion_names())

    n_val   = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    log.info("Split : %d train | %d val", n_train, n_val)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Modèle ────────────────────────────────────────────────────────────────
    model = DraftModel(
        vocab_size    = encoder.vocab_size,
        embedding_dim = args.embed_dim,
        hidden_dim    = args.hidden_dim,
        dropout       = args.dropout,
        n_inputs      = len(FEATURE_COLS),  # 20
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Modèle initialisé : vocab=%d | embed_dim=%d | hidden=%d | params=%s",
        encoder.vocab_size, args.embed_dim, args.hidden_dim,
        f"{n_params:,}",
    )

    # ── Calcul des Poids de Classes (Class Weights) ───────────────────────────
    # Corrige le déséquilibre (champions très populaires vs très rares)
    log.info("Calcul des class weights pour rééquilibrer la Loss...")
    frequencies = torch.bincount(dataset.targets, minlength=encoder.vocab_size).float()
    
    # Lissage par racine carrée (Square Root Smoothing) pour éviter la surcompensation
    weights = 1.0 / torch.sqrt(frequencies + 1e-5)
    weights[PAD_IDX] = 0.0 # PAD token n'a pas besoin de poids
    
    # Normalisation pour une moyenne à 1.0
    valid_weights = weights[1:]
    weights[1:] = valid_weights / valid_weights.mean()
    
    weights_tensor = weights.to(device)

    # ── Loss & Optimizer ──────────────────────────────────────────────────────
    # ignore_index=PAD_IDX : si un champion cible est 0 (inconnu), on l'ignore
    criterion = nn.CrossEntropyLoss(weight=weights_tensor, ignore_index=PAD_IDX, label_smoothing=0.1)
    
    # AdamW est recommandé pour les Transformers avec un fort weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    
    # CosineAnnealingLR : décroissance douce sur toute la durée de l'entraînement.
    # Évite de tuer le LR prématurément (problème de ReduceLROnPlateau avec patience=2).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 1e-3,
    )

    # ── Validation du forward pass avant de lancer l'entraînement ────────────
    log.info("Vérification du forward pass…")
    sample_batch = next(iter(train_loader))
    with torch.no_grad():
        sample_out = model(
            sample_batch["features"].to(device),
            sample_batch["win"].to(device),
            sample_batch["position"].to(device)
        )
    log.info(
        "  ✓ Forward pass OK : input %s → output %s",
        tuple(sample_batch["features"].shape),
        tuple(sample_out.shape),
    )

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    log.info("Démarrage de l'entraînement (%d epochs)…", args.epochs)
    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>9} | {'Val Acc':>8} | {'LR':>8} | {'Time':>6}")
    print("-" * 75)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed    = time.time() - t0

        print(
            f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2%} | "
            f"{val_loss:>9.4f} | {val_acc:>7.2%} | {current_lr:>8.2e} | {elapsed:>5.1f}s"
        )

        # Sauvegarde du meilleur modèle
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            out_path = Path(args.model_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch":        epoch,
                    "model_state":  model.state_dict(),
                    "vocab_size":   encoder.vocab_size,
                    "embed_dim":    args.embed_dim,
                    "hidden_dim":   args.hidden_dim,
                    "n_inputs":     len(FEATURE_COLS),
                    "id_to_idx":    encoder.id_to_idx,
                    "idx_to_id":    encoder.idx_to_id,
                    "id_to_name":   encoder.id_to_name,
                    "val_loss":     val_loss,
                },
                out_path,
            )
            log.info("  ★ Meilleur modèle sauvegardé (val_loss=%.4f) → %s", val_loss, out_path)

    print("-" * 75)
    log.info("Entraînement terminé. Meilleure val_loss : %.4f", best_val_loss)

    # ── Exemple de prédiction finale ──────────────────────────────────────────
    log.info("Exemple de prédiction top-5 sur le premier batch de validation…")
    model.load_state_dict(
        torch.load(args.model_out, map_location=device, weights_only=False)["model_state"]
    )
    sample = next(iter(val_loader))
    probs, idxs = model.predict_top_k(
        sample["features"].to(device),
        sample["win"].to(device),
        sample["position"].to(device),
        k=5
    )

    print("\n  Top-5 prédictions (premier sample du val set)")
    print(f"  Cible réelle : {encoder.decode_name(sample['target'][0].item())}")
    print("  Prédictions  :")
    for rank, (prob, idx) in enumerate(
        zip(probs[0].cpu().tolist(), idxs[0].cpu().tolist()), start=1
    ):
        print(f"    #{rank}  {encoder.decode_name(idx):<20}  prob={prob:.2%}")


if __name__ == "__main__":
    main()
