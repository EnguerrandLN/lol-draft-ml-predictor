"""
build_mlm_dataset.py — Génération du dataset MLM pour l'estimateur de draft.

Principe Masked Language Modeling :
  Pour chaque match, on génère 10 lignes d'entraînement (une par joueur).
  Dans chaque ligne, un joueur est la "Cible" (target à prédire) ;
  les 9 autres forment le "Contexte" (features). Le slot de la cible
  dans les colonnes ally_* est masqué à -1 pour éviter tout data leakage.

Schéma de sortie (mlm_draft_dataset.csv) :
  match_id | target_position | target_champion_id | target_win
  ally_top | ally_jungle | ally_mid | ally_adc | ally_supp      (alliés)
  enemy_top | enemy_jungle | enemy_mid | enemy_adc | enemy_supp  (ennemis)
  ban_1 … ban_10                                                 (bans triés)

Usage :
  python build_mlm_dataset.py
  python build_mlm_dataset.py --output data/mlm_draft_dataset.csv
  python build_mlm_dataset.py --wins-only   # N'exporte que les drafts gagnantes
  python build_mlm_dataset.py --min-matches 5 # Filtre les matchs incomplets

Prérequis : pip install pandas
"""
import argparse
import logging
import random
import sqlite3
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

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

# Mapping teamPosition (API) → nom de colonne court
POSITION_MAP: dict[str, str] = {
    "TOP":     "top",
    "JUNGLE":  "jungle",
    "MIDDLE":  "mid",
    "BOTTOM":  "adc",
    "UTILITY": "supp",
}
POSITIONS: list[str] = ["top", "jungle", "mid", "adc", "supp"]

# Ordre chronologique des bans en SoloQ (phase 1 : turns 1-3 croisés ; phase 2 : turns 4-5 inversés)
# Format : (team_id, pick_turn)
BAN_DRAFT_ORDER: list[tuple[int, int]] = [
    (100, 1), (200, 1),
    (100, 2), (200, 2),
    (100, 3), (200, 3),
    (200, 4), (100, 4),
    (200, 5), (100, 5),
]
MASK_VALUE: int = -1   # Valeur utilisée pour masquer le slot cible


# ── Chargement ────────────────────────────────────────────────────────────────

def load_data(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Charge et pré-traite les tables participants et bans depuis SQLite.

    Returns:
        (participants_df, bans_df)
    """
    log.info("Chargement des participants…")
    participants = pd.read_sql_query(
        """
        SELECT
            p.match_id,
            p.puuid,
            p.team_id,
            p.position,          -- teamPosition : TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY
            p.champion_id,
            p.win
        FROM participants p
        JOIN matches m ON m.match_id = p.match_id
        WHERE p.position IS NOT NULL AND p.position != ''
        """,
        conn,
    )

    log.info("Chargement des bans…")
    bans = pd.read_sql_query(
        """
        SELECT match_id, team_id, pick_turn, champion_id
        FROM bans
        WHERE champion_id != -1
        """,
        conn,
    )

    log.info(
        "Chargé : %d participants sur %d matchs | %d bans",
        len(participants),
        participants["match_id"].nunique(),
        len(bans),
    )
    return participants, bans


# ── Pivot des bans ────────────────────────────────────────────────────────────

def build_ban_lookup(bans: pd.DataFrame) -> pd.DataFrame:
    """
    Pivote les bans en une ligne par match avec colonnes ban_1…ban_10.
    L'ordre respecte la chronologie réelle de la draft (BAN_DRAFT_ORDER).

    Returns:
        DataFrame indexé par match_id avec colonnes ban_1…ban_10.
    """
    # Clé composite pour le tri et la fusion
    bans = bans.copy()
    bans["ban_key"] = list(zip(bans["team_id"], bans["pick_turn"]))

    # Rang global selon BAN_DRAFT_ORDER (1 à 10)
    order_map = {key: rank + 1 for rank, key in enumerate(BAN_DRAFT_ORDER)}
    bans["ban_rank"] = bans["ban_key"].map(order_map)

    # Supprime les bans sans rang reconnu (ne devrait pas arriver en SoloQ)
    bans = bans.dropna(subset=["ban_rank"])
    bans["ban_rank"] = bans["ban_rank"].astype(int)

    # Pivot : une ligne par match, une colonne par rang de ban
    pivot = bans.pivot_table(
        index="match_id",
        columns="ban_rank",
        values="champion_id",
        aggfunc="first",
    )
    pivot.columns = [f"ban_{c}" for c in pivot.columns]

    # S'assure que les 10 colonnes existent (comble avec 0 si absentes)
    for i in range(1, 11):
        col = f"ban_{i}"
        if col not in pivot.columns:
            pivot[col] = 0

    pivot = pivot[[f"ban_{i}" for i in range(1, 11)]]
    pivot = pivot.fillna(0).astype(int).reset_index()

    log.info("Ban lookup : %d matchs avec bans pivotés", len(pivot))
    return pivot


# ── Pivot des picks ───────────────────────────────────────────────────────────

def build_pick_matrix(participants: pd.DataFrame) -> pd.DataFrame:
    """
    Pivote les picks en une ligne par (match_id, team_id) avec une colonne
    par rôle : champion_top, champion_jungle, champion_mid, champion_adc, champion_supp.

    Returns:
        DataFrame avec index (match_id, team_id) et 5 colonnes de champions.
    """
    df = participants.copy()
    df["pos_short"] = df["position"].map(POSITION_MAP)

    # Ignore les positions non reconnues
    df = df.dropna(subset=["pos_short"])

    # Pivot : une ligne par (match, team), une colonne par position
    matrix = df.pivot_table(
        index=["match_id", "team_id"],
        columns="pos_short",
        values="champion_id",
        aggfunc="first",
    )

    # Renomme et s'assure que les 5 colonnes existent
    matrix.columns = [f"champion_{c}" for c in matrix.columns]
    for pos in POSITIONS:
        col = f"champion_{pos}"
        if col not in matrix.columns:
            matrix[col] = pd.NA

    matrix = matrix[[f"champion_{p}" for p in POSITIONS]].reset_index()
    return matrix


# ── Construction du dataset MLM ───────────────────────────────────────────────

def build_mlm_dataset(
    participants: pd.DataFrame,
    ban_lookup: pd.DataFrame,
    pick_matrix: pd.DataFrame,
    wins_only: bool = False,
) -> pd.DataFrame:
    """
    Génère les 10 lignes MLM par match.

    Pour chaque participant :
      - target_champion_id  : son champion (variable Y)
      - target_position     : son rôle
      - ally_*              : champions alliés (-1 au slot masqué)
      - enemy_*             : champions ennemis
      - ban_1…ban_10        : bans du match

    Args:
        participants: DataFrame des participants (brut, nettoyé).
        ban_lookup:   DataFrame pivoté des bans (une ligne par match).
        pick_matrix:  DataFrame pivoté des picks par (match_id, team_id).
        wins_only:    Si True, ne garde que les lignes où target_win == 1.

    Returns:
        DataFrame final du dataset MLM.
    """
    df = participants.copy()
    df["pos_short"] = df["position"].map(POSITION_MAP)
    df = df.dropna(subset=["pos_short"])

    rows: list[dict] = []

    # Groupe par match : on traite les 10 joueurs ensemble
    for match_id, group in df.groupby("match_id"):

        # Récupère la matrice des picks pour ce match
        match_matrix = pick_matrix[pick_matrix["match_id"] == match_id].set_index("team_id")
        if len(match_matrix) < 2:
            continue  # Match incomplet, on skip

        team_ids = group["team_id"].unique()
        if len(team_ids) != 2:
            continue

        # Récupère les bans
        ban_row = ban_lookup[ban_lookup["match_id"] == match_id]
        if ban_row.empty:
            ban_dict = {f"ban_{i}": 0 for i in range(1, 11)}
        else:
            ban_dict = ban_row.iloc[0].drop("match_id").to_dict()

        # Pour chaque joueur du match → une ligne MLM
        for _, player in group.iterrows():
            pos     = player["pos_short"]          # ex: "top"
            team    = player["team_id"]             # 100 ou 200
            enemy_team = [t for t in team_ids if t != team][0]

            # Vérifie que les deux équipes ont une entrée dans la matrice
            if team not in match_matrix.index or enemy_team not in match_matrix.index:
                continue

            ally_picks  = match_matrix.loc[team]
            enemy_picks = match_matrix.loc[enemy_team]

            # Construction de ally_* avec masquage du slot cible
            ally_cols: dict[str, int] = {}
            for p in POSITIONS:
                col = f"champion_{p}"
                val = ally_picks.get(col, pd.NA)
                if p == pos:
                    ally_cols[f"ally_{p}"] = MASK_VALUE   # ← Masque pour éviter le leakage
                elif pd.isna(val):
                    ally_cols[f"ally_{p}"] = 0
                else:
                    ally_cols[f"ally_{p}"] = int(val)

            # Construction de enemy_*
            enemy_cols: dict[str, int] = {}
            for p in POSITIONS:
                col = f"champion_{p}"
                val = enemy_picks.get(col, pd.NA)
                enemy_cols[f"enemy_{p}"] = 0 if pd.isna(val) else int(val)

            # ── Masquage Dynamique (Data Augmentation) ────────────────────
            # Simule une draft partielle en masquant N autres choix (0 à 9)
            other_slots = [f"ally_{p}" for p in POSITIONS if p != pos] + [f"enemy_{p}" for p in POSITIONS]
            n_to_mask = random.randint(0, 9)
            
            if n_to_mask > 0:
                slots_to_mask = random.sample(other_slots, n_to_mask)
                for slot in slots_to_mask:
                    if slot.startswith("ally_"):
                        ally_cols[slot] = MASK_VALUE
                    else:
                        enemy_cols[slot] = MASK_VALUE

            row = {
                "match_id":           match_id,
                "target_position":    player["position"],   # Forme longue pour lisibilité
                "target_champion_id": int(player["champion_id"]),
                "target_win":         int(player["win"]),
                **ally_cols,
                **enemy_cols,
                **ban_dict,
            }
            rows.append(row)

    dataset = pd.DataFrame(rows)

    # Ordonne les colonnes de façon logique
    col_order = (
        ["match_id", "target_position", "target_champion_id", "target_win"]
        + [f"ally_{p}"  for p in POSITIONS]
        + [f"enemy_{p}" for p in POSITIONS]
        + [f"ban_{i}"   for i in range(1, 11)]
    )
    # Ne garde que les colonnes qui existent
    col_order = [c for c in col_order if c in dataset.columns]
    dataset = dataset[col_order]

    if wins_only:
        before = len(dataset)
        dataset = dataset[dataset["target_win"] == 1].reset_index(drop=True)
        log.info("Filtre wins_only : %d → %d lignes", before, len(dataset))

    return dataset


# ── Validation ────────────────────────────────────────────────────────────────

def validate(dataset: pd.DataFrame) -> None:
    """Vérifie les invariants clés du dataset."""
    log.info("Validation du dataset…")

    # 1. Pas de leakage : ally_{pos} == MASK_VALUE là où c'est la cible
    pos_to_ally = {v: f"ally_{v}" for v in POSITION_MAP.values()}
    leakage_count = 0
    for pos_long, pos_short in POSITION_MAP.items():
        ally_col = pos_to_ally[pos_short]
        if ally_col not in dataset.columns:
            continue
        mask = dataset["target_position"] == pos_long
        non_masked = dataset.loc[mask, ally_col] != MASK_VALUE
        leakage_count += non_masked.sum()

    if leakage_count > 0:
        log.error("DATA LEAKAGE DÉTECTÉ : %d lignes où ally non masqué !", leakage_count)
    else:
        log.info("  ✓ Aucun data leakage détecté (ally masqué = %d partout)", MASK_VALUE)

    # 2. Pas de valeurs nulles dans les colonnes clés
    key_cols = ["target_champion_id", "target_position", "target_win"]
    nulls = dataset[key_cols].isnull().sum()
    if nulls.any():
        log.warning("  Valeurs nulles dans les colonnes clés :\n%s", nulls[nulls > 0])
    else:
        log.info("  ✓ Aucune valeur nulle dans les colonnes clés")

    # 3. target_champion_id jamais 0
    zero_champs = (dataset["target_champion_id"] == 0).sum()
    if zero_champs:
        log.warning("  %d lignes avec target_champion_id == 0 (champion inconnu)", zero_champs)
    else:
        log.info("  ✓ Aucun champion cible à 0")

    # 4. Statistiques rapides
    n_matches = dataset["match_id"].nunique()
    n_rows    = len(dataset)
    n_champs  = dataset["target_champion_id"].nunique()
    wr        = dataset["target_win"].mean() * 100

    log.info("  Matchs uniques    : %d", n_matches)
    log.info("  Lignes totales    : %d  (attendu ~%d)", n_rows, n_matches * 10)
    log.info("  Champions uniques : %d", n_champs)
    log.info("  Winrate moyen     : %.1f%%  (attendu ~50%%)", wr)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Génère le dataset MLM draft depuis la base SQLite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o",
        default=str(DATA_DIR / "mlm_draft_dataset.csv"),
        help="Chemin du fichier CSV de sortie.",
    )
    parser.add_argument(
        "--wins-only",
        action="store_true",
        help="N'exporte que les lignes où target_win == 1.",
    )
    parser.add_argument(
        "--min-participants",
        type=int,
        default=10,
        metavar="N",
        help="Nombre minimum de participants avec position connue pour inclure un match.",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Connexion à la base : %s", DB_PATH)
    conn = sqlite3.connect(str(DB_PATH))

    try:
        # ── Chargement ──────────────────────────────────────────────────────
        participants, bans = load_data(conn)

        # ── Filtrage : matchs avec les 10 positions complètes ───────────────
        participants["pos_short"] = participants["position"].map(POSITION_MAP)
        valid_positions = participants.dropna(subset=["pos_short"])
        pos_per_match = valid_positions.groupby("match_id")["pos_short"].count()
        complete_matches = pos_per_match[pos_per_match >= args.min_participants].index

        before = participants["match_id"].nunique()
        participants = participants[participants["match_id"].isin(complete_matches)]
        after = participants["match_id"].nunique()
        log.info(
            "Matchs filtrés (>= %d positions connues) : %d/%d matchs conservés",
            args.min_participants, after, before,
        )

        # ── Pivots ──────────────────────────────────────────────────────────
        ban_lookup   = build_ban_lookup(bans[bans["match_id"].isin(complete_matches)])
        pick_matrix  = build_pick_matrix(participants)

        # ── Construction MLM ────────────────────────────────────────────────
        log.info("Construction du dataset MLM…")
        dataset = build_mlm_dataset(
            participants, ban_lookup, pick_matrix,
            wins_only=args.wins_only,
        )

        # ── Validation ──────────────────────────────────────────────────────
        validate(dataset)

        # ── Export CSV ──────────────────────────────────────────────────────
        dataset.to_csv(output_path, index=False)
        log.info("Dataset exporté : %s  (%d lignes × %d colonnes)",
                 output_path, len(dataset), len(dataset.columns))

        # ── Aperçu ──────────────────────────────────────────────────────────
        print("\n" + "─" * 80)
        print("  APERÇU (5 premières lignes)")
        print("─" * 80)
        with pd.option_context("display.max_columns", 30, "display.width", 200):
            print(dataset.head())
        print("─" * 80)
        print(f"\n  Colonnes ({len(dataset.columns)}) :")
        print("  " + " | ".join(dataset.columns.tolist()))
        print()

    finally:
        conn.close()


if __name__ == "__main__":
    main()
