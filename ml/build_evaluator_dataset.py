import logging
import random
import sqlite3
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR, DB_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

def extract_role_counts():
    import json
    log.info("Extraction des fréquences de rôles (role_counts.json)...")
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT champion_id, position, count(*) FROM participants WHERE champion_id > 0 AND position IN ('TOP', 'JUNGLE', 'MIDDLE', 'BOTTOM', 'UTILITY') GROUP BY champion_id, position").fetchall()
    
    counts = {}
    for cid, pos, count in rows:
        if cid not in counts:
            counts[cid] = {'TOP':0, 'JUNGLE':0, 'MIDDLE':0, 'BOTTOM':0, 'UTILITY':0}
        counts[cid][pos] = count
        
    riot_to_role = {'TOP': 'Top', 'JUNGLE': 'Jungle', 'MIDDLE': 'Mid', 'BOTTOM': 'ADC', 'UTILITY': 'Support'}
    final_counts = {str(cid): {riot_to_role[k]: v for k, v in pos_counts.items()} for cid, pos_counts in counts.items()}
    
    with open(DATA_DIR / "role_counts.json", "w") as f:
        json.dump(final_counts, f, indent=4)
    conn.close()
    log.info(f"-> {len(final_counts)} champions sauvegardés dans role_counts.json.")

def build_evaluator_dataset():
    extract_role_counts()
    
    conn = sqlite3.connect(DB_PATH)
    
    log.info("Récupération des matchs complets (10 joueurs)...")
    matches = conn.execute("""
        SELECT match_id 
        FROM participants 
        GROUP BY match_id 
        HAVING count(*) = 10
    """).fetchall()
    
    valid_match_ids = [m[0] for m in matches]
    log.info(f"{len(valid_match_ids)} matchs trouvés.")

    query = f"""
        SELECT match_id, team_id, position, champion_id, win
        FROM participants
        WHERE match_id IN ({','.join(['?']*len(valid_match_ids))})
    """
    
    log.info("Chargement des données en mémoire...")
    df_participants = pd.read_sql_query(query, conn, params=valid_match_ids)
    conn.close()

    dataset_rows = []
    
    # On groupe par match
    grouped = df_participants.groupby('match_id')
    
    log.info("Génération du dataset Evaluator (avec Data Augmentation)...")
    for match_id, group in tqdm(grouped, total=len(valid_match_ids)):
        
        # Séparation Alliés (team 100) / Ennemis (team 200)
        # (Techniquement, le bleu et rouge s'inversent d'une game à l'autre, on double la donnée en inversant la perspective)
        
        team_100 = group[group['team_id'] == 100]
        team_200 = group[group['team_id'] == 200]
        
        def get_champ(team_df, pos):
            row = team_df[team_df['position'] == pos]
            return int(row['champion_id'].iloc[0]) if not row.empty else -1
            
        positions = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        
        # Draft complète du point de vue de l'équipe 100
        draft_100 = [get_champ(team_100, p) for p in positions] + [get_champ(team_200, p) for p in positions]
        win_100 = int(team_100['win'].iloc[0])
        
        # Draft complète du point de vue de l'équipe 200 (les rôles sont inversés : 200 devient "Allié")
        draft_200 = [get_champ(team_200, p) for p in positions] + [get_champ(team_100, p) for p in positions]
        win_200 = int(team_200['win'].iloc[0])
        
        # Fonction de Data Augmentation : créer des drafts partielles
        def augment_draft(base_draft, win_status):
            rows = []
            # 1. Ajouter la draft complète (fin de game)
            rows.append(base_draft + [win_status])
            
            # 2. Ajouter 3 drafts partielles aléatoires pour que l'IA apprenne à évaluer en cours de sélection
            for _ in range(3):
                partial = list(base_draft)
                # On masque entre 1 et 9 champions aléatoirement
                n_mask = random.randint(1, 9)
                indices_to_mask = random.sample(range(10), n_mask)
                for idx in indices_to_mask:
                    partial[idx] = -1
                rows.append(partial + [win_status])
            return rows

        dataset_rows.extend(augment_draft(draft_100, win_100))
        dataset_rows.extend(augment_draft(draft_200, win_200))

    columns = [
        "ally_top", "ally_jungle", "ally_mid", "ally_adc", "ally_supp",
        "enemy_top", "enemy_jungle", "enemy_mid", "enemy_adc", "enemy_supp",
        "target_win"
    ]
    
    final_df = pd.DataFrame(dataset_rows, columns=columns)
    
    out_path = DATA_DIR / "evaluator_draft_dataset.csv"
    final_df.to_csv(out_path, index=False)
    log.info(f"Terminé ! CSV généré : {out_path} ({len(final_df)} lignes).")

if __name__ == "__main__":
    build_evaluator_dataset()
