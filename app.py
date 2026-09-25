import sqlite3
import sys
from pathlib import Path

import streamlit as st
import torch

# Injecter la racine dans le PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR, DB_PATH
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

# ─── LOGIQUE D'ÉVALUATION (FORCE BRUTE) ───────────────────────────────────────
if st.button("🚀 Évaluer toutes les combinaisons possibles", use_container_width=True, type="primary"):
    
    # 1. Construction du vecteur de base (10 cases sans les bans)
    features_raw = ally_selections + enemy_selections
    
    # Sécurité absolue : S'assurer que le rôle cible est VIDE avant de tester
    features_raw[target_idx_in_tensor] = -1
    
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
    
    for i, cid in enumerate(valid_cids):
        draft_copy = list(features_encoded)
        # Insertion du champion testé
        draft_copy[target_idx_in_tensor] = encode(cid)
        batch_tensor[i] = torch.tensor(draft_copy, dtype=torch.long)
        
    # 4. Inférence (Sécurisée)
    model.eval()
    with torch.no_grad():
        logits = model(batch_tensor)                # Shape: (Batch,) grace au squeeze(-1)
        probs = torch.sigmoid(logits)               # Shape: (Batch,)
        
    # 5. Tri des résultats
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    
    top_probs = sorted_probs[:10].tolist()
    top_indices = sorted_indices[:10].tolist()
    
    # 6. Affichage (Arrondi à 2 décimales)
    st.subheader(f"🏆 Top 10 des meilleurs picks {target_role}")
    
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
        best_cid = valid_cids[idx]
        champ_name = valid_champions.get(best_cid, f"Inconnu (id={best_cid})")
        
        prob_percent = prob * 100
        
        if rank == 1:
            st.success(f"**#{rank}. {champ_name}** — {prob_percent:.2f}% de chance de victoire 🎯")
        elif rank <= 3:
            st.info(f"**#{rank}. {champ_name}** — {prob_percent:.2f}% de chance de victoire")
        else:
            st.write(f"**#{rank}.** {champ_name} — {prob_percent:.2f}%")
