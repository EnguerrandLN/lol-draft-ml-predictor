"""
database/repository.py — Couche d'accès aux données (DAL).

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


def upsert_match(conn: sqlite3.Connection, match_data: dict) -> None:
    """
    Insère les métadonnées d'un match.
    Ignore silencieusement les doublons (idempotent).

    Args:
        conn: Connexion SQLite active.
        match_data: Réponse brute de l'endpoint Match-V5.
    """
    info: dict = match_data["info"]
    winning_team: Optional[int] = next(
        (t["teamId"] for t in info.get("teams", []) if t.get("win")), None
    )
    conn.execute(
        """
        INSERT OR IGNORE INTO matches
            (match_id, game_version, queue_id, game_duration, platform_id, game_creation, winning_team)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            match_data["metadata"]["matchId"],
            info.get("gameVersion"),
            info.get("queueId"),
            info.get("gameDuration"),
            info.get("platformId"),
            info.get("gameCreation"),
            winning_team,
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


# ── Summoner cache ────────────────────────────────────────────────────────────

def upsert_summoner(
    conn: sqlite3.Connection,
    puuid: str,
    summoner_id: str,
    summoner_name: str,
    region: str,
) -> None:
    """
    Met à jour le cache d'un invocateur (upsert).

    Args:
        conn: Connexion SQLite active.
        puuid: PUUID Riot global.
        summoner_id: ID spécifique à la plateforme.
        summoner_name: Nom affiché de l'invocateur.
        region: Région de la plateforme (ex: EUW1).
    """
    conn.execute(
        """
        INSERT INTO summoner_cache (puuid, summoner_id, summoner_name, region)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(puuid) DO UPDATE SET
            summoner_id     = excluded.summoner_id,
            summoner_name   = excluded.summoner_name,
            last_crawled_at = CURRENT_TIMESTAMP
        """,
        (puuid, summoner_id, summoner_name, region),
    )


# ── Crawl Queue ───────────────────────────────────────────────────────────────

def enqueue_puuids(conn: sqlite3.Connection, puuids: list[str]) -> int:
    """
    Ajoute des PUUIDs à la file d'attente BFS (ignore les doublons).

    Returns:
        Nombre de nouveaux PUUIDs effectivement ajoutés.
    """
    added = 0
    for puuid in puuids:
        cur = conn.execute(
            "INSERT OR IGNORE INTO crawl_queue (puuid, status) VALUES (?, 'pending')",
            (puuid,),
        )
        added += cur.rowcount
    return added


def get_pending_puuid(conn: sqlite3.Connection) -> Optional[str]:
    """
    Dépile le prochain PUUID 'pending' par ordre d'insertion (FIFO).

    Returns:
        Un PUUID ou None si la file est vide.
    """
    cur = conn.execute(
        "SELECT puuid FROM crawl_queue WHERE status = 'pending' ORDER BY enqueued_at LIMIT 1"
    )
    row = cur.fetchone()
    return row["puuid"] if row else None


def mark_puuid_done(conn: sqlite3.Connection, puuid: str) -> None:
    """Marque un PUUID comme traité avec succès."""
    conn.execute(
        "UPDATE crawl_queue SET status = 'done', processed_at = CURRENT_TIMESTAMP WHERE puuid = ?",
        (puuid,),
    )


def mark_puuid_error(conn: sqlite3.Connection, puuid: str) -> None:
    """Marque un PUUID en erreur (ne sera pas re-traité automatiquement)."""
    conn.execute(
        "UPDATE crawl_queue SET status = 'error', processed_at = CURRENT_TIMESTAMP WHERE puuid = ?",
        (puuid,),
    )


# ── Stats ─────────────────────────────────────────────────────────────────────

def get_match_count(conn: sqlite3.Connection) -> int:
    """Retourne le nombre total de matchs en base."""
    cur = conn.execute("SELECT COUNT(*) FROM matches")
    return cur.fetchone()[0]


def get_queue_stats(conn: sqlite3.Connection) -> dict[str, int]:
    """Retourne les compteurs de la crawl_queue par statut."""
    cur = conn.execute(
        "SELECT status, COUNT(*) AS cnt FROM crawl_queue GROUP BY status"
    )
    return {row["status"]: row["cnt"] for row in cur.fetchall()}
