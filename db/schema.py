"""
db/schema.py — Création du schéma SQLite.

  matches (centre) ← bans, participants (faits)
  ladder_players    (file du crawler : joueurs échantillonnés par tier)
"""
import logging
import sqlite3

from config import DB_PATH

logger = logging.getLogger(__name__)

# ── DDL ──────────────────────────────────────────────────────────────────────

_CREATE_MATCHES: str = """
CREATE TABLE IF NOT EXISTS matches (
    match_id        TEXT    PRIMARY KEY,
    game_version    TEXT,
    queue_id        INTEGER,
    game_duration   INTEGER,   -- secondes
    platform_id     TEXT,
    game_creation   INTEGER,   -- epoch ms
    winning_team    INTEGER,   -- 100 ou 200
    crawled_at      TIMESTAMP  DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_BANS: str = """
CREATE TABLE IF NOT EXISTS bans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id    TEXT    NOT NULL,
    team_id     INTEGER NOT NULL,   -- 100 ou 200
    pick_turn   INTEGER NOT NULL,   -- 1-5 par équipe
    champion_id INTEGER NOT NULL,   -- -1 = pas de ban
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
    UNIQUE (match_id, team_id, pick_turn)
);
"""

_CREATE_PARTICIPANTS: str = """
CREATE TABLE IF NOT EXISTS participants (
    id                                      INTEGER PRIMARY KEY AUTOINCREMENT,
    match_id                                TEXT    NOT NULL,
    puuid                                   TEXT    NOT NULL,
    summoner_name                           TEXT,
    summoner_id                             TEXT,
    team_id                                 INTEGER NOT NULL,  -- 100 ou 200
    position                                TEXT,   -- teamPosition : TOP/JUNGLE/MIDDLE/BOTTOM/UTILITY
    lane                                    TEXT,   -- lane Riot   : TOP_LANE/MID_LANE/BOT_LANE/JUNGLE/NONE
    role                                    TEXT,   -- role Riot   : CARRY/SUPPORT/SOLO/NONE
    champion_id                             INTEGER NOT NULL,
    champion_name                           TEXT,
    win                                     INTEGER NOT NULL,  -- 1 = victoire, 0 = défaite
    kills                                   INTEGER DEFAULT 0,
    deaths                                  INTEGER DEFAULT 0,
    assists                                 INTEGER DEFAULT 0,
    total_minions_killed                    INTEGER DEFAULT 0,
    gold_earned                             INTEGER DEFAULT 0,
    -- Dégâts infligés aux champions (décomposition physique / magique / vrai)
    total_damage_dealt_to_champions         INTEGER DEFAULT 0,
    physical_damage_dealt_to_champions      INTEGER DEFAULT 0,
    magic_damage_dealt_to_champions         INTEGER DEFAULT 0,
    true_damage_dealt_to_champions          INTEGER DEFAULT 0,
    -- Dégâts reçus
    total_damage_taken                      INTEGER DEFAULT 0,
    damage_self_mitigated                   INTEGER,   -- dégâts absorbés (armure, RM, boucliers)
    -- Contrôle et soutien (profils de champions : engage, soins, protection)
    time_ccing_others                       INTEGER,   -- score Riot de temps de CC infligé
    total_time_cc_dealt                     INTEGER,   -- durée cumulée des CC infligés (s)
    total_heal                              INTEGER,
    total_heals_on_teammates                INTEGER,
    total_damage_shielded_on_teammates      INTEGER,
    vision_score                            INTEGER DEFAULT 0,
    wards_placed                            INTEGER DEFAULT 0,
    items                                   TEXT,   -- JSON: [item0..item6]
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
    UNIQUE (match_id, puuid)
);
"""

_CREATE_LADDER_PLAYERS: str = """
CREATE TABLE IF NOT EXISTS ladder_players (
    puuid           TEXT    PRIMARY KEY,
    tier            TEXT    NOT NULL,   -- GOLD ... CHALLENGER
    division        TEXT,               -- I..IV (I pour les tiers apex)
    league_points   INTEGER,
    status          TEXT    NOT NULL DEFAULT 'pending',  -- pending | done
    priority        REAL    NOT NULL,   -- aléatoire : mélange les tiers dans la file
    last_crawled_at INTEGER,            -- epoch s du dernier historique récupéré
    snapshot_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_INDICES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_bans_match         ON bans(match_id);",
    "CREATE INDEX IF NOT EXISTS idx_participants_match  ON participants(match_id);",
    "CREATE INDEX IF NOT EXISTS idx_participants_puuid  ON participants(puuid);",
    "CREATE INDEX IF NOT EXISTS idx_matches_version     ON matches(game_version);",
    "CREATE INDEX IF NOT EXISTS idx_ladder_next         ON ladder_players(status, priority);",
]


# ── Nouvelles colonnes à ajouter sur une DB existante ────────────────────────
# ALTER TABLE ignore les colonnes déjà présentes grâce au try/except.

_MIGRATIONS: list[str] = [
    "ALTER TABLE participants ADD COLUMN lane                               TEXT;",
    "ALTER TABLE participants ADD COLUMN role                               TEXT;",
    "ALTER TABLE participants ADD COLUMN physical_damage_dealt_to_champions INTEGER DEFAULT 0;",
    "ALTER TABLE participants ADD COLUMN magic_damage_dealt_to_champions    INTEGER DEFAULT 0;",
    "ALTER TABLE participants ADD COLUMN true_damage_dealt_to_champions     INTEGER DEFAULT 0;",
    "ALTER TABLE participants ADD COLUMN total_damage_taken                 INTEGER DEFAULT 0;",
    # NULL pour les matchs collectés avant leur ajout (et non 0, qui serait une vraie valeur)
    "ALTER TABLE participants ADD COLUMN damage_self_mitigated              INTEGER;",
    "ALTER TABLE participants ADD COLUMN time_ccing_others                  INTEGER;",
    "ALTER TABLE participants ADD COLUMN total_time_cc_dealt                INTEGER;",
    "ALTER TABLE participants ADD COLUMN total_heal                         INTEGER;",
    "ALTER TABLE participants ADD COLUMN total_heals_on_teammates           INTEGER;",
    "ALTER TABLE participants ADD COLUMN total_damage_shielded_on_teammates INTEGER;",
    # Remake (gameEndedInEarlySurrender) : le résultat ne dépend pas de la draft
    "ALTER TABLE matches ADD COLUMN ended_early     INTEGER;",
    # Rang du joueur du ladder par lequel le match a été trouvé (proxy de l'ELO du match)
    "ALTER TABLE matches ADD COLUMN source_tier     TEXT;",
    "ALTER TABLE matches ADD COLUMN source_division TEXT;",
]

# Matchs collectés avant l'ajout de ended_early : la durée < 5 min sert de proxy.
_BACKFILLS: list[str] = [
    "UPDATE matches SET ended_early = (game_duration < 300) WHERE ended_early IS NULL;",
]


def _migrate_db(conn: sqlite3.Connection) -> None:
    """
    Applique les migrations ALTER TABLE sur une base existante.
    Chaque instruction est tentée individuellement ; si la colonne existe déjà
    SQLite lève une OperationalError qui est silencieusement ignorée.
    """
    for stmt in _MIGRATIONS:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass  # Colonne déjà présente — pas d'action requise
    for stmt in _BACKFILLS:
        conn.execute(stmt)
    conn.commit()
    logger.debug("Migrations appliquées.")


# ── Init ─────────────────────────────────────────────────────────────────────

def init_db() -> sqlite3.Connection:
    """
    Crée (ou ouvre) la base de données SQLite, applique le schéma et retourne
    une connexion prête à l'emploi.

    Returns:
        sqlite3.Connection: Connexion active avec foreign_keys et WAL activés.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA foreign_keys = ON;")
    # Write-Ahead Logging : meilleure concurrence et performances en écriture
    conn.execute("PRAGMA journal_mode = WAL;")
    # Synchronisation moins stricte (ok pour un collecteur de données)
    conn.execute("PRAGMA synchronous = NORMAL;")

    for stmt in (
        _CREATE_MATCHES,
        _CREATE_BANS,
        _CREATE_PARTICIPANTS,
        _CREATE_LADDER_PLAYERS,
    ):
        conn.execute(stmt)

    # Migration des colonnes ajoutées après la création initiale
    _migrate_db(conn)

    for idx in _INDICES:
        conn.execute(idx)

    conn.commit()
    logger.info("Base de données initialisée : %s", DB_PATH)
    return conn
