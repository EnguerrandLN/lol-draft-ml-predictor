import sys
from pathlib import Path

import streamlit as st
import torch

# Injecter la racine dans le PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR
from ml.train_evaluator import PAD_IDX, WinPredictorModel

# ─── CONFIGURATION DE LA PAGE ─────────────────────────────────────────────────
st.set_page_config(page_title="LoL Draft Evaluator", layout="wide", page_icon="⚖️")

st.title("⚖️ LoL Draft Evaluator")
st.markdown("Simulateur de Winrate par substitution itérative (Modèle Binaire).")

# ─── CHARGEMENT DU MODÈLE ET DES ROLES ────────────────────────────────────────
@st.cache_resource
def load_evaluator():
    """Charge le modèle évaluateur (BCE) depuis le checkpoint PyTorch."""
    model_path = DATA_DIR / "win_predictor.pt"
    if not model_path.exists():
        st.error(f"Modèle introuvable à : {model_path}. Veuillez entraîner le modèle d'abord.")
        st.stop()
        
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    model = WinPredictorModel(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        n_inputs=checkpoint["n_inputs"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()  # Très important pour désactiver le Dropout
    
    return model, checkpoint["id_to_idx"], checkpoint["idx_to_id"], checkpoint["id_to_name"]

model, id_to_idx, idx_to_id, id_to_name = load_evaluator()

@st.cache_data
def get_champion_role_counts() -> dict[int, dict[str, int]]:
    """Récupère la fréquence absolue (N) de chaque champion pour chaque rôle depuis le cache JSON."""
    try:
        import json
        with open(DATA_DIR / "role_counts.json", "r") as f:
            raw_counts = json.load(f)
            # Convert string keys back to int
            return {int(k): v for k, v in raw_counts.items()}
    except Exception as e:
        return {}

@st.cache_resource
def load_champion_features():
    import json
    feature_file = DATA_DIR / "champion_features.json"
    with open(feature_file, "r") as f:
        return json.load(f)

champ_features_dict = load_champion_features()
num_features = len(next(iter(champ_features_dict.values())))

# ─── PRÉPARATION DES LISTES POUR L'UI ─────────────────────────────────────────
EMPTY_CHOICE = "--- Vide ---"

valid_champions = {cid: name for cid, name in id_to_name.items() if name}
sorted_names = sorted(valid_champions.values())
options = [EMPTY_CHOICE] + sorted_names
name_to_id = {name: cid for cid, name in valid_champions.items()}

# ─── INTERFACE GRAPHIQUE ──────────────────────────────────────────────────────
st.subheader("🎯 Rôle Cible de la simulation")
roles = ["Top", "Jungle", "Mid", "ADC", "Support"]
target_role = st.selectbox("Sélectionnez le rôle allié à optimiser :", roles)
target_idx_in_tensor = roles.index(target_role)

st.divider()

col1, col2, col3 = st.columns(3)

def champion_selectbox(label: str, key: str, disabled: bool = False) -> int:
    choice = st.selectbox(label, options, key=key, disabled=disabled)
    if choice == EMPTY_CHOICE:
        return -1
    return name_to_id[choice]

ally_selections = []
enemy_selections = []
ban_selections = []

with col1:
    st.header("🔵 Équipe Alliée")
    for role in roles:
        is_target = (role == target_role)
        cid = champion_selectbox(f"Allié {role}", key=f"ally_{role}", disabled=is_target)
        ally_selections.append(cid)

with col2:
    st.header("🔴 Équipe Ennemie")
    for role in roles:
        cid = champion_selectbox(f"Ennemi {role}", key=f"enemy_{role}")
        enemy_selections.append(cid)

with col3:
    st.header("🚫 Bans")
    for i in range(1, 11):
        cid = champion_selectbox(f"Ban {i}", key=f"ban_{i}")
        ban_selections.append(cid)

st.divider()

st.sidebar.header("⚙️ Paramètres")
C = st.sidebar.slider(
    "Facteur de Confiance (C)", 
    min_value=10, max_value=500, value=50, step=10, 
    help="Shrinkage paramétrique : tire la probabilité vers 50% si le champion est très rare à ce poste."
)

# ─── LOGIQUE D'ÉVALUATION (FORCE BRUTE) ───────────────────────────────────────
if st.button("🚀 Évaluer toutes les combinaisons possibles", use_container_width=True, type="primary"):
    
    # 1. Construction du vecteur de base (10 cases sans les bans)
    features_raw = ally_selections + enemy_selections
    
    # Sécurité absolue : S'assurer que le rôle cible est VIDE avant de tester
    features_raw[target_idx_in_tensor] = -1
    
    # ── Règle 4 : Limite de Trous (Garde-fou OOD) ──
    selected_count = sum(1 for cid in features_raw if cid > 0)
    if selected_count < 4:  # < 4 lockés + 1 cible = < 5 champions totaux
        st.warning("⚠️ **Attention (OOD)** : Vous évaluez une draft avec moins de 5 champions lockés. Les Transformers gèrent mal les séquences majoritairement vides (Winrates aberrants probables).")
        
    # Les bans ne vont pas dans le tenseur, mais bloquent les choix possibles
    used_ids = set([cid for cid in features_raw + ban_selections if cid > 0])
    
    def encode(cid: int) -> int:
        if cid <= 0: return PAD_IDX
        return id_to_idx.get(cid, PAD_IDX)
        
    features_encoded = [encode(cid) for cid in features_raw]
    
    # 2. Liste des champions testables (tous sauf ceux déjà lock/ban)
    valid_cids = [cid for cid in id_to_name.keys() if cid > 0 and cid not in used_ids]
    
    if not valid_cids:
        st.warning("Aucun champion disponible ou valide à tester pour ce poste.")
        st.stop()
        
    # 3. Création du Tenseur Batch
    batch_size = len(valid_cids)
    batch_tensor = torch.zeros((batch_size, 10), dtype=torch.long)
    batch_feats = torch.zeros((batch_size, 10, num_features), dtype=torch.float32)
    
    def get_feat(c: int) -> list:
        if c <= 0: return [0.0] * num_features
        return champ_features_dict.get(str(c), [0.0] * num_features)
    
    for i, cid in enumerate(valid_cids):
        draft_copy = list(features_raw)
        # Insertion du champion testé (Riot ID)
        draft_copy[target_idx_in_tensor] = cid
        
        batch_tensor[i] = torch.tensor([encode(c) for c in draft_copy], dtype=torch.long)
        batch_feats[i] = torch.tensor([get_feat(c) for c in draft_copy], dtype=torch.float32)
        
    # 4. Inférence (Sécurisée)
    model.eval()
    with torch.no_grad():
        logits = model(batch_tensor, batch_feats)   # Shape: (Batch,) grace au squeeze(-1)
        probs = torch.sigmoid(logits)               # Shape: (Batch,)
        
    # 5. Application du Shrinkage Paramétrique
    role_counts_db = get_champion_role_counts()
    results = []
    
    for i, cid in enumerate(valid_cids):
        prob_brute = probs[i].item()
        
        N = role_counts_db.get(cid, {}).get(target_role, 0)
        
        prob_finale = (N * prob_brute + C * 0.5) / (N + C)
        
        results.append({
            "cid": cid,
            "prob_finale": prob_finale,
            "prob_brute": prob_brute,
            "N": N
        })
        
    # Tri par probabilité finale (décroissant)
    results.sort(key=lambda x: x["prob_finale"], reverse=True)
    top_10 = results[:10]
    
    # 6. Affichage (Arrondi à 2 décimales)
    st.subheader(f"🏆 Top 10 des meilleurs picks {target_role}")
    
    for rank, res in enumerate(top_10, 1):
        champ_name = valid_champions.get(res["cid"], f"Inconnu (id={res['cid']})")
        prob_pct = res["prob_finale"] * 100
        
        # Info transparence
        penalite_text = f" (N = {res['N']} parties)"
        
        if rank == 1:
            st.success(f"**#{rank}. {champ_name}** — {prob_pct:.2f}% chance 🎯" + penalite_text)
        elif rank <= 3:
            st.info(f"**#{rank}. {champ_name}** — {prob_pct:.2f}% chance" + penalite_text)
        else:
            st.write(f"**#{rank}.** {champ_name} — {prob_pct:.2f}%" + penalite_text)
