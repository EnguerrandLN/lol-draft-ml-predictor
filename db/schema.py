"""
database/schema.py — Création du schéma SQLite optimisé pour le ML.

Schéma en étoile :
  matches (centre) ← bans, participants (faits)
  summoner_cache    (dimension joueur)
  crawl_queue       (état du BFS)

Jointure ML : une seule requête suffit pour reconstruire le vecteur
  (bans, picks alliés, picks ennemis, position, version patch) → champion cible.
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
    vision_score                            INTEGER DEFAULT 0,
    wards_placed                            INTEGER DEFAULT 0,
    items                                   TEXT,   -- JSON: [item0..item6]
    FOREIGN KEY (match_id) REFERENCES matches(match_id) ON DELETE CASCADE,
    UNIQUE (match_id, puuid)
);
"""

_CREATE_SUMMONER_CACHE: str = """
CREATE TABLE IF NOT EXISTS summoner_cache (
    puuid           TEXT PRIMARY KEY,
    summoner_id     TEXT,
    summoner_name   TEXT,
    region          TEXT,
    last_crawled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_CRAWL_QUEUE: str = """
CREATE TABLE IF NOT EXISTS crawl_queue (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    puuid        TEXT    NOT NULL UNIQUE,
    status       TEXT    NOT NULL DEFAULT 'pending',  -- pending | done | error
    enqueued_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);
"""

_INDICES: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_bans_match         ON bans(match_id);",
    "CREATE INDEX IF NOT EXISTS idx_participants_match  ON participants(match_id);",
    "CREATE INDEX IF NOT EXISTS idx_participants_puuid  ON participants(puuid);",
    "CREATE INDEX IF NOT EXISTS idx_crawl_status        ON crawl_queue(status);",
    "CREATE INDEX IF NOT EXISTS idx_matches_version     ON matches(game_version);",
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
        _CREATE_SUMMONER_CACHE,
        _CREATE_CRAWL_QUEUE,
    ):
        conn.execute(stmt)

    for idx in _INDICES:
        conn.execute(idx)

    # Migration des colonnes ajoutées après la création initiale
    _migrate_db(conn)

    conn.commit()
    logger.info("Base de données initialisée : %s", DB_PATH)
    return conn
