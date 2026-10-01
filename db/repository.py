"""
db/repository.py — Couche d'accès aux données (DAL).

Toutes les fonctions reçoivent une connexion sqlite3.Connection et opèrent
en mode "INSERT OR IGNORE" / "ON CONFLICT DO UPDATE" pour être idempotentes
(sécurisé contre les doubles insertions lors des relances du crawler).
"""
import json
import logging
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)


# ── Matches ───────────────────────────────────────────────────────────────────

def match_exists(conn: sqlite3.Connection, match_id: str) -> bool:
    """Retourne True si le match est déjà en base."""
    cur = conn.execute("SELECT 1 FROM matches WHERE match_id = ?", (match_id,))
    return cur.fetchone() is not None


def upsert_match(
    conn: sqlite3.Connection,
    match_data: dict,
    source_tier: Optional[str] = None,
    source_division: Optional[str] = None,
) -> None:
    """
    Insère les métadonnées d'un match.
    Ignore silencieusement les doublons (idempotent).

    Args:
        conn: Connexion SQLite active.
        match_data: Réponse brute de l'endpoint Match-V5.
        source_tier: Tier du joueur du ladder par lequel le match a été trouvé.
        source_division: Division de ce joueur (I..IV).
    """
    info: dict = match_data["info"]
    winning_team: Optional[int] = next(
        (t["teamId"] for t in info.get("teams", []) if t.get("win")), None
    )
    ended_early: int = int(
        any(p.get("gameEndedInEarlySurrender") for p in info.get("participants", []))
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO matches
            (match_id, game_version, queue_id, game_duration, platform_id, game_creation,
             winning_team, ended_early, source_tier, source_division)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            match_data["metadata"]["matchId"],
            info.get("gameVersion"),
            info.get("queueId"),
            info.get("gameDuration"),
            info.get("platformId"),
            info.get("gameCreation"),
            winning_team,
            ended_early,
            source_tier,
            source_division,
        ),
    )


# ── Bans ──────────────────────────────────────────────────────────────────────

def upsert_bans(conn: sqlite3.Connection, match_id: str, teams: list[dict]) -> None:
    """
    Insère les bans des deux équipes pour un match donné.

    Args:
        conn: Connexion SQLite active.
        match_id: Identifiant du match (ex: EUW1_7123456789).
        teams: Liste des deux objets 'team' de la réponse Match-V5.
    """
    for team in teams:
        for ban in team.get("bans", []):
            conn.execute(
                """
                INSERT OR IGNORE INTO bans (match_id, team_id, pick_turn, champion_id)
                VALUES (?, ?, ?, ?)
                """,
                (match_id, team["teamId"], ban["pickTurn"], ban["championId"]),
            )


# ── Participants ──────────────────────────────────────────────────────────────

def upsert_participant(conn: sqlite3.Connection, match_id: str, p: dict) -> None:
    """
    Insère les statistiques d'un participant.

    Champs de rôle (trois niveaux de granularité) :
      - position     (teamPosition) : TOP | JUNGLE | MIDDLE | BOTTOM | UTILITY
      - lane         (lane)         : TOP_LANE | MID_LANE | BOT_LANE | JUNGLE | NONE
      - role         (role)         : CARRY | SUPPORT | SOLO | NONE

    Les items sont sérialisés en JSON (liste de 7 entiers : item0..item6).

    Args:
        conn: Connexion SQLite active.
        match_id: Identifiant du match.
        p: Objet participant brut de la réponse Match-V5.
    """
    items: str = json.dumps([p.get(f"item{i}", 0) for i in range(7)])
    conn.execute(
        """
        INSERT OR IGNORE INTO participants (
            match_id, puuid, summoner_name, summoner_id, team_id,
            position, lane, role,
            champion_id, champion_name, win,
            kills, deaths, assists,
            total_minions_killed, gold_earned,
            total_damage_dealt_to_champions,
            physical_damage_dealt_to_champions,
            magic_damage_dealt_to_champions,
            true_damage_dealt_to_champions,
            total_damage_taken,
            vision_score, wards_placed, items
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            match_id,
            p.get("puuid"),
            p.get("summonerName"),
            p.get("summonerId"),
            p.get("teamId"),
            p.get("teamPosition"),                          # TOP | JUNGLE | MIDDLE | BOTTOM | UTILITY
            p.get("lane"),                                  # TOP_LANE | MID_LANE | BOT_LANE | JUNGLE | NONE
            p.get("role"),                                  # CARRY | SUPPORT | SOLO | NONE
            p.get("championId"),
            p.get("championName"),
            1 if p.get("win") else 0,
            p.get("kills", 0),
            p.get("deaths", 0),
            p.get("assists", 0),
            p.get("totalMinionsKilled", 0),
            p.get("goldEarned", 0),
            p.get("totalDamageDealtToChampions", 0),
            p.get("physicalDamageDealtToChampions", 0),
            p.get("magicDamageDealtToChampions", 0),
            p.get("trueDamageDealtToChampions", 0),
            p.get("totalDamageTaken", 0),
            p.get("visionScore", 0),
            p.get("wardsPlaced", 0),
            items,
        ),
    )


# ── Ladder ────────────────────────────────────────────────────────────────────

def upsert_ladder_players(conn: sqlite3.Connection, players: list[dict]) -> None:
    """
    Insère / rafraîchit des joueurs du ladder et les remet en 'pending'.
    `last_crawled_at` est conservé : un joueur déjà vu ne sera interrogé que
    sur ses parties jouées depuis.

    Args:
        players: dicts avec puuid, tier, division, league_points, priority.
    """
    conn.executemany(
        """
        INSERT INTO ladder_players (puuid, tier, division, league_points, status, priority)
        VALUES (:puuid, :tier, :division, :league_points, 'pending', :priority)
        ON CONFLICT(puuid) DO UPDATE SET
            tier          = excluded.tier,
            division      = excluded.division,
            league_points = excluded.league_points,
            status        = 'pending',
            priority      = excluded.priority,
            snapshot_at   = CURRENT_TIMESTAMP
        """,
        players,
    )


def get_next_ladder_player(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    """Prochain joueur 'pending' (ordre aléatoire fixé à l'enfilement), ou None."""
    return conn.execute(
        """
        SELECT puuid, tier, division, last_crawled_at FROM ladder_players
        WHERE status = 'pending' ORDER BY priority LIMIT 1
        """
    ).fetchone()


def mark_ladder_player_done(conn: sqlite3.Connection, puuid: str, crawled_at: int) -> None:
    """Marque un joueur traité ; `crawled_at` (epoch s) borne la prochaine passe."""
    conn.execute(
        "UPDATE ladder_players SET status = 'done', last_crawled_at = ? WHERE puuid = ?",
        (crawled_at, puuid),
    )


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_match_count(conn: sqlite3.Connection) -> int:
    """Retourne le nombre total de matchs en base."""
    cur = conn.execute("SELECT COUNT(*) FROM matches")
    return cur.fetchone()[0]


def get_ladder_stats(conn: sqlite3.Connection) -> dict[str, int]:
    """Retourne les compteurs de ladder_players par statut."""
    cur = conn.execute(
        "SELECT status, COUNT(*) AS cnt FROM ladder_players GROUP BY status"
    )
    return {row["status"]: row["cnt"] for row in cur.fetchall()}
