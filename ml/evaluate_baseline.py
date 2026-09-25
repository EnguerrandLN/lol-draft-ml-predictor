import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR
from ml.train_evaluator import ChampionEncoder, FEATURE_COLS

def evaluate_baseline():
    dataset_path = DATA_DIR / "evaluator_draft_dataset.csv"
    print(f"Chargement du dataset : {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # 1. Nettoyage et encodage (même logique que l'évaluateur PyTorch)
    encoder = ChampionEncoder.from_dataframe(df, FEATURE_COLS)
    feature_matrix = df[FEATURE_COLS].fillna(0).astype(int)
    encoded_features = feature_matrix.map(encoder.encode).values
    
    # Filtre anti-NaN (lignes vides)
    valid_rows = (encoded_features != 0).any(axis=1)
    encoded_features = encoded_features[valid_rows]
    wins = df["target_win"].values[valid_rows]
    
    print(f"Échantillons valides : {len(encoded_features)}")
    
    # 2. Conversion en Bag of Words (One-Hot) Séparé
    n_samples = len(encoded_features)
    vocab_size = encoder.vocab_size
    
    # [Batch, vocab_size] pour alliés, [Batch, vocab_size] pour ennemis
    X_ally = np.zeros((n_samples, vocab_size), dtype=np.float32)
    X_enemy = np.zeros((n_samples, vocab_size), dtype=np.float32)
    
    # Les 5 premiers sont alliés, les 5 derniers sont ennemis
    for i in range(n_samples):
        for j in range(5):
            ally_id = encoded_features[i, j]
            if ally_id > 0:
                X_ally[i, ally_id] = 1.0
                
        for j in range(5, 10):
            enemy_id = encoded_features[i, j]
            if enemy_id > 0:
                X_enemy[i, enemy_id] = 1.0
                
    X = np.concatenate([X_ally, X_enemy], axis=1)
    y = wins
    
    # 3. Séparation Train/Val
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42)
    
    print(f"Entraînement de la Régression Logistique sur {len(X_train)} samples...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # 4. Évaluation
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    
    acc = accuracy_score(y_val, y_pred)
    brier = brier_score_loss(y_val, y_prob)
    
    print("\n" + "="*40)
    print("🏆 RÉSULTATS BASELINE (LogisticRegression)")
    print("="*40)
    print(f"Validation Accuracy  : {acc*100:.2f}%")
    print(f"Brier Score Loss     : {brier:.4f}")
    print("="*40)

if __name__ == "__main__":
    evaluate_baseline()
