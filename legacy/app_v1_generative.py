import sys
from pathlib import Path

import streamlit as st
import torch

# Injecter la racine dans le PYTHONPATH pour permettre l'import du dossier ml/ et config.py
sys.path.insert(0, str(Path(__file__).parent))

from config import DATA_DIR
from ml.train_model import PAD_IDX, DraftModel

# ─── CONFIGURATION DE LA PAGE ─────────────────────────────────────────────────
st.set_page_config(page_title="LoL Draft ML Predictor", layout="wide", page_icon="🏆")

st.title("🔮 LoL Draft Predictor")
st.markdown("Interface d'inférence conditionnée sur la victoire (`target_win = 1.0`).")

# ─── CHARGEMENT DU MODÈLE ET DU VOCABULAIRE ───────────────────────────────────
@st.cache_resource
def load_predictor():
    """Charge le modèle et ses dictionnaires depuis le checkpoint PyTorch."""
    model_path = DATA_DIR / "draft_model.pt"
    if not model_path.exists():
        st.error(f"Modèle introuvable à : {model_path}. Veuillez entraîner le modèle d'abord.")
        st.stop()
        
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    
    model = DraftModel(
        vocab_size=checkpoint["vocab_size"],
        embedding_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        n_inputs=checkpoint["n_inputs"]
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    
    return model, checkpoint["id_to_idx"], checkpoint["idx_to_id"], checkpoint["id_to_name"]

model, id_to_idx, idx_to_id, id_to_name = load_predictor()

# ─── PRÉPARATION DES LISTES POUR L'UI ─────────────────────────────────────────
EMPTY_CHOICE = "--- Vide ---"

# On trie les noms des champions alphabétiquement pour les menus déroulants
valid_champions = {cid: name for cid, name in id_to_name.items() if name}
sorted_names = sorted(valid_champions.values())
options = [EMPTY_CHOICE] + sorted_names

# Dictionnaire inverse: Nom -> ID Riot
name_to_id = {name: cid for cid, name in valid_champions.items()}

# ─── INTERFACE GRAPHIQUE ──────────────────────────────────────────────────────
st.subheader("🎯 Cible de la prédiction")
roles = ["Top", "Jungle", "Mid", "ADC", "Support"]
target_role = st.selectbox("Rôle à prédire (sera masqué au modèle)", roles)
target_idx_in_tensor = roles.index(target_role)

st.divider()

col1, col2, col3 = st.columns(3)

def champion_selectbox(label: str, key: str, disabled: bool = False) -> int:
    """Affiche un selectbox et retourne l'ID Riot correspondant (ou -1 si vide)."""
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
        # Griser le menu si c'est le rôle que l'on souhaite prédire
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

# ─── LOGIQUE D'INFÉRENCE ──────────────────────────────────────────────────────
if st.button("🚀 Prédire le meilleur pick (Win Condition = 1.0)", use_container_width=True, type="primary"):
    
    # 1. Construction du vecteur d'entrée brut (taille 20)
    features_raw = ally_selections + enemy_selections + ban_selections
    
    # 2. Forcer impérativement la cible à -1 (masqué) au cas où
    features_raw[target_idx_in_tensor] = -1
    
    # 3. Encodage continu pour le modèle
    def encode(cid: int) -> int:
        if cid <= 0: return PAD_IDX
        return id_to_idx.get(cid, PAD_IDX)
        
    features_encoded = [encode(cid) for cid in features_raw]
    
    # 4. Conversion en tenseurs PyTorch [Batch=1, Features=20], [Batch=1, 1], et [Batch=1]
    features_tensor = torch.tensor([features_encoded], dtype=torch.long)
    target_win_tensor = torch.tensor([[1.0]], dtype=torch.float32)
    target_position_tensor = torch.tensor([target_idx_in_tensor], dtype=torch.long)
    
    # 5. Inférence
    with torch.no_grad():
        logits = model(features_tensor, target_win_tensor, target_position_tensor)
        probs = torch.softmax(logits, dim=-1)
        
        # Récupération d'un top large pour pouvoir filtrer les champions déjà pris
        top_probs, top_indices = torch.topk(probs, k=25, dim=-1)
        
        top_probs = top_probs[0].tolist()
        top_indices = top_indices[0].tolist()
        
        st.subheader(f"🏆 Top 5 Recommandations pour {target_role}")
        
        # Set des champions déjà présents dans la draft pour ne pas les conseiller
        used_ids = set([cid for cid in features_raw if cid > 0])
        
        count = 0
        for prob, idx in zip(top_probs, top_indices):
            cid = idx_to_id.get(idx, 0)
            
            if cid == 0: continue # Token de Padding (ignoré)
            if cid in used_ids: continue # Champion déjà pick/ban
            
            count += 1
            champ_name = valid_champions.get(cid, f"Inconnu (id={cid})")
            
            # Affichage élégant
            if count == 1:
                st.success(f"**#{count} - {champ_name}** ({prob:.2%}) 👑")
            else:
                st.info(f"**#{count} - {champ_name}** ({prob:.2%})")
                
            if count == 5:
                break
