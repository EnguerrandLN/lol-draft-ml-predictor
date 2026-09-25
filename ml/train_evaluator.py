"""
train_evaluator_model.py — Modèle Évaluateur de Draft (Winrate Predictor).

Le modèle reçoit une draft (complète ou partielle) et prédit la probabilité 
de victoire de l'équipe alliée (target_win = 1.0 ou 0.0).
"""
import argparse
import logging
import sqlite3
import sys
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

# ── Constantes ────────────────────────────────────────────────────────────────
ALLY_COLS  = ["ally_top", "ally_jungle", "ally_mid", "ally_adc", "ally_supp"]
ENEMY_COLS = ["enemy_top", "enemy_jungle", "enemy_mid", "enemy_adc", "enemy_supp"]
FEATURE_COLS = ALLY_COLS + ENEMY_COLS
PAD_IDX = 0

def load_champion_names(db_path=DB_PATH) -> dict[int, str]:
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT DISTINCT champion_id, champion_name FROM participants "
            "WHERE champion_name IS NOT NULL AND champion_id > 0"
        ).fetchall()
        conn.close()
        return {int(cid): name for cid, name in rows if name}
    except Exception:
        return {}

class ChampionEncoder:
    def __init__(self, champion_ids: set[int]) -> None:
        sorted_ids = sorted(champion_ids)
        self.id_to_idx = {cid: idx + 1 for idx, cid in enumerate(sorted_ids)}
        self.idx_to_id = {v: k for k, v in self.id_to_idx.items()}
        self.n_champions = len(sorted_ids)
        self.vocab_size  = len(sorted_ids) + 1
        self.id_to_name  = {}

    def attach_names(self, names: dict[int, str]) -> None:
        self.id_to_name = names

    def encode(self, champion_id: int) -> int:
        if champion_id <= 0: return PAD_IDX
        return self.id_to_idx.get(champion_id, PAD_IDX)

    def decode(self, idx: int) -> int:
        return self.idx_to_id.get(idx, 0)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame, cols: list[str]) -> "ChampionEncoder":
        all_ids = set()
        for col in cols:
            all_ids.update(df[col].dropna().astype(int).unique())
        all_ids.discard(0)
        all_ids.discard(-1)
        return cls(all_ids)


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────────────────────────────────────
class DraftEvaluatorDataset(Dataset):
    def __init__(self, csv_path: str | Path, encoder: ChampionEncoder | None = None) -> None:
        log.info("Chargement du dataset : %s", csv_path)
        df = pd.read_csv(csv_path)
        
        all_id_cols = FEATURE_COLS
        if encoder is None:
            self.encoder = ChampionEncoder.from_dataframe(df, all_id_cols)
        else:
            self.encoder = encoder
            
        feature_matrix = df[FEATURE_COLS].fillna(0).astype(int)
        encoded_features = feature_matrix.map(self.encoder.encode).values
        features_tensor = torch.tensor(encoded_features, dtype=torch.long)
        
        # Filtre anti-NaN : on exclut les lignes où TOUS les picks sont vides (masqués)
        valid_rows = (features_tensor != PAD_IDX).any(dim=1)
        self.features = features_tensor[valid_rows]
        
        # Cible mathématique : probabilité de victoire (0.0 ou 1.0)
        wins_tensor = torch.tensor(df["target_win"].values, dtype=torch.float32)
        self.wins = wins_tensor[valid_rows]
        
        log.info("Dataset Evaluator prêt : %d samples (dont %d exclus car vides)", len(self), (~valid_rows).sum().item())

    def __len__(self) -> int:
        return len(self.wins)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.wins[idx]


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
class WinPredictorModel(nn.Module):
    def __init__(
        self,
        vocab_size:    int,
        embedding_dim: int = 32,
        hidden_dim:    int = 256,
        dropout:       float = 0.4,
        n_inputs:      int = 10,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=PAD_IDX)
        
        # ── Positional Embedding ──────────────────────────────────────────────
        self.pos_embedding = nn.Embedding(n_inputs, embedding_dim)
        self.pos_dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        flat_dim = n_inputs * embedding_dim
        
        self.network = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, 1) # 1 seule sortie brute (sans Sigmoid)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Base Champion Embedding
        embedded = self.embedding(x)
        
        # 2. Positional Embedding (indispensable pour l'ordre des rôles)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand_as(x)
        embedded = embedded + self.pos_embedding(positions)
        embedded = self.pos_dropout(embedded)
        
        # 3. Création du masque d'attention pour ignorer les slots vides (PAD_IDX)
        padding_mask = (x == PAD_IDX)
        
        # 4. Self-Attention
        attended = self.transformer(embedded, src_key_padding_mask=padding_mask)
        
        # 5. Aplatissement et Décision finale
        flat = attended.flatten(start_dim=1)
        return self.network(flat).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# 3. BOUCLE D'ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        
        optimizer.zero_grad()
        logits = model(features)
        loss = criterion(logits, targets.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(targets)
        # Calcul Précision (Sigmoid pour ramener entre 0 et 1)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == targets.float()).sum().item()
        total_samples += len(targets)
        
    return total_loss / total_samples, total_correct / total_samples

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        logits = model(features)
        loss = criterion(logits, targets.float())
        
        total_loss += loss.item() * len(targets)
        preds = (torch.sigmoid(logits) > 0.5).float()
        total_correct += (preds == targets.float()).sum().item()
        total_samples += len(targets)
    return total_loss / total_samples, total_correct / total_samples


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset",    default=str(DATA_DIR / "evaluator_draft_dataset.csv"))
    p.add_argument("--model-out",  default=str(DATA_DIR / "win_predictor.pt"))
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch-size", type=int,   default=512)
    p.add_argument("--embed-dim",  type=int,   default=64)
    p.add_argument("--hidden-dim", type=int,   default=512)
    p.add_argument("--dropout",    type=float, default=0.4)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--val-split",  type=float, default=0.15)
    p.add_argument("--device",     default="auto")
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    log.info("Device : %s", device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = DraftEvaluatorDataset(args.dataset)
    encoder = dataset.encoder
    encoder.attach_names(load_champion_names())
    
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False)
    
    model = WinPredictorModel(
        vocab_size=encoder.vocab_size,
        embedding_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        n_inputs=10
    ).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_loss = float("inf")
    print(f"\n{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>9} | {'Val Acc':>8} | {'LR':>8}")
    print("-" * 75)
    
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2%} | {val_loss:>9.4f} | {val_acc:>7.2%} | {optimizer.param_groups[0]['lr']:>8.2e}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            out_path = Path(args.model_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "vocab_size": encoder.vocab_size,
                "embed_dim": args.embed_dim,
                "hidden_dim": args.hidden_dim,
                "n_inputs": 10,
                "id_to_idx": encoder.id_to_idx,
                "idx_to_id": encoder.idx_to_id,
                "id_to_name": encoder.id_to_name,
            }, out_path)
            log.info("  ★ Meilleur modèle sauvegardé (val_loss=%.4f) → %s", val_loss, out_path)

if __name__ == "__main__":
    main()
