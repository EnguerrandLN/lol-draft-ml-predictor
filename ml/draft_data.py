"""
draft_data.py — Chargement des drafts complètes depuis la base SQLite.

Une ligne par match (pas d'augmentation) : les 10 champions par équipe et par
rôle, le résultat, la date, le patch et le tier source.
"""
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH

ROLES: tuple[str, ...] = ("TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
SIDES: tuple[str, ...] = ("blue", "red")
TEAM_TO_SIDE: dict[int, str] = {100: "blue", 200: "red"}
DRAFT_COLS: list[str] = [f"{side}_{role}" for side in SIDES for role in ROLES]


def connect_read_only(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Connexion en lecture seule : sûre pendant que le crawler écrit."""
    return sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)


def load_matches(
    db_path: Path = DB_PATH,
    include_remakes: bool = False,
    puuid: Optional[str] = None,
) -> pd.DataFrame:
    """
    Args:
        puuid: Si fourni, seulement les matchs joués par ce joueur.

    Returns:
        DataFrame indexé par match_id avec les colonnes DRAFT_COLS (champion_id),
        blue_win (0/1), game_creation (ms), patch ("16.18"), source_tier.
        Seuls les matchs avec exactement un joueur par rôle et par équipe sont gardés.
    """
    conn = connect_read_only(db_path)
    remake_filter = "" if include_remakes else "AND COALESCE(m.ended_early, 0) = 0"
    player_filter = "AND p.match_id IN (SELECT match_id FROM participants WHERE puuid = :puuid)" if puuid else ""
    participants = pd.read_sql_query(
        f"""
        SELECT p.match_id, p.team_id, p.position, p.champion_id, p.win
        FROM participants p JOIN matches m USING (match_id)
        WHERE m.queue_id = 420 {remake_filter} {player_filter}
        """,
        conn,
        params={"puuid": puuid},
    )
    matches = pd.read_sql_query(
        "SELECT match_id, game_creation, game_version, source_tier FROM matches",
        conn,
    ).set_index("match_id")
    conn.close()

    participants = participants[participants.position.isin(ROLES)]
    participants["slot"] = participants.team_id.map(TEAM_TO_SIDE) + "_" + participants.position

    # Un match n'est gardé que si ses 10 slots sont remplis exactement une fois
    slot_counts = participants.groupby("match_id").slot.agg(["size", "nunique"])
    complete = slot_counts[(slot_counts["size"] == 10) & (slot_counts["nunique"] == 10)].index
    participants = participants[participants.match_id.isin(complete)]

    drafts = participants.pivot(index="match_id", columns="slot", values="champion_id")[DRAFT_COLS]
    blue_win = participants[participants.team_id == 100].groupby("match_id").win.first()

    df = drafts.join(blue_win.rename("blue_win")).join(matches, how="inner")
    df["patch"] = df.game_version.str.split(".").str[:2].str.join(".")
    return df.drop(columns="game_version").sort_values("game_creation")


def load_player_slots(db_path: Path = DB_PATH, puuid: Optional[str] = None) -> pd.DataFrame:
    """
    Une ligne par (joueur, match) : puuid, match_id, side ("blue"/"red"),
    position, champion_id. Restreint à un joueur si `puuid` est fourni.
    """
    conn = connect_read_only(db_path)
    slots = pd.read_sql_query(
        "SELECT puuid, match_id, team_id, position, champion_id FROM participants"
        + (" WHERE puuid = :puuid" if puuid else ""),
        conn,
        params={"puuid": puuid},
    )
    conn.close()
    slots["side"] = slots.pop("team_id").map(TEAM_TO_SIDE)
    return slots


def load_champion_ad_share(db_path: Path = DB_PATH) -> dict[int, float]:
    """
    Part des dégâts physiques dans les dégâts infligés aux champions, par
    champion (tous rôles), mesurée sur les vraies parties : 0.01 pour Karthus,
    0.98 pour Draven. Sert à décrire le profil de dégâts d'une équipe.
    """
    conn = connect_read_only(db_path)
    rows = conn.execute(
        """
        SELECT champion_id, SUM(physical_damage_dealt_to_champions) * 1.0 / SUM(total_damage_dealt_to_champions)
        FROM participants
        WHERE champion_id > 0 AND total_damage_dealt_to_champions > 0
        GROUP BY champion_id
        """
    ).fetchall()
    conn.close()
    return {int(cid): float(share) for cid, share in rows}


def load_champion_names(db_path: Path = DB_PATH) -> dict[int, str]:
    conn = connect_read_only(db_path)
    rows = conn.execute(
        "SELECT champion_id, champion_name FROM participants "
        "WHERE champion_name IS NOT NULL AND champion_id > 0 GROUP BY champion_id"
    ).fetchall()
    conn.close()
    return {int(cid): name for cid, name in rows}
