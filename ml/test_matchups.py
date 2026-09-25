import sys
import torch
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR
from ml.train_evaluator import PAD_IDX, WinPredictorModel

def test():
    model_path = DATA_DIR / "win_predictor.pt"
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    model = WinPredictorModel(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        n_inputs=checkpoint["n_inputs"],
        num_features=11
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    id_to_idx = checkpoint["id_to_idx"]
    id_to_name = checkpoint["id_to_name"]
    name_to_id = {v: k for k, v in id_to_name.items() if v}
    
    # Load features
    with open(DATA_DIR / "champion_features.json", "r") as f:
        feats_dict = json.load(f)
    num_features = len(next(iter(feats_dict.values())))
    
    def get_feat(cid):
        if cid <= 0: return [0.0] * num_features
        return feats_dict.get(str(cid), [0.0] * num_features)
        
    def encode(cid):
        if cid <= 0: return PAD_IDX
        return id_to_idx.get(cid, PAD_IDX)

    valid_cids = [cid for cid in id_to_name.keys() if cid > 0]
    
    def eval_draft(name, enemies):
        print(f"\n{'='*50}\n📊 {name} : {enemies}\n{'='*50}")
        # Allies are empty, target is Top (index 0)
        draft = [-1, -1, -1, -1, -1] + [name_to_id[e] for e in enemies]
        
        batch_size = len(valid_cids)
        batch_ids = torch.zeros((batch_size, 10), dtype=torch.long)
        batch_feats = torch.zeros((batch_size, 10, num_features), dtype=torch.float32)
        
        for i, test_cid in enumerate(valid_cids):
            draft_copy = list(draft)
            draft_copy[0] = test_cid # Target Ally Top
            batch_ids[i] = torch.tensor([encode(c) for c in draft_copy], dtype=torch.long)
            batch_feats[i] = torch.tensor([get_feat(c) for c in draft_copy], dtype=torch.float32)
            
        with torch.no_grad():
            logits = model(batch_ids, batch_feats)
            probs = torch.sigmoid(logits)
            
        results = []
        for i, test_cid in enumerate(valid_cids):
            results.append((id_to_name[test_cid], probs[i].item()))
            
        results.sort(key=lambda x: x[1], reverse=True)
        print("🏆 TOP 5 :")
        for rank in range(5):
            print(f"  #{rank+1} {results[rank][0]:<15} {results[rank][1]*100:.2f}%")
        print("\n💀 FLOP 5 :")
        for rank in range(5):
            print(f"  #{rank+1} {results[-(rank+1)][0]:<15} {results[-(rank+1)][1]*100:.2f}%")
            
    eval_draft("Draft A (Test Armure - 100% AD)", ["Tryndamere", "LeeSin", "Zed", "Draven", "Pyke"])
    eval_draft("Draft B (Test Mobilité - No Dash)", ["Garen", "Udyr", "Sett", "Mordekaiser", "Taric"])

if __name__ == "__main__":
    test()
