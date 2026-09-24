"""
fetch_player.py — Récupère l'intégralité de l'historique ranked solo d'un joueur.

Ce script est distinct du crawler BFS : il cible un seul joueur et exhauste
tous ses matchs Ranked Solo/Duo via pagination automatique de l'API.

Utile pour :
  - Construire un profil de joueur complet (historique personnel).
  - Personnaliser le prédicteur selon les habitudes d'un joueur donné.
  - Ré-indexer un joueur spécifique sans relancer le crawler général.

Usage :
  python fetch_player.py --riot-id KeytedLN#EUW
  python fetch_player.py --riot-id KeytedLN#EUW --limit 50
  python fetch_player.py --puuid <PUUID>
  python fetch_player.py --riot-id KeytedLN#EUW --log-level DEBUG
"""
import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from api.client import RiotApiClient
from config import RANKED_SOLO_QUEUE
from api.crawler import process_match
from db.repository import match_exists, get_match_count
from db.schema import init_db


# ── Logging ───────────────────────────────────────────────────────────────────

def setup_logging(level: str = "INFO") -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("fetch_player.log", encoding="utf-8"),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch all ranked solo matches for a single player.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--riot-id",
        metavar="NAME#TAG",
        help="Riot ID of the player (e.g. KeytedLN#EUW).",
    )
    group.add_argument(
        "--puuid",
        metavar="PUUID",
        help="PUUID of the player (skips the Riot ID resolution step).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Max number of matches to fetch. 0 = no limit (fetch all).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# ── Core ──────────────────────────────────────────────────────────────────────

def resolve_puuid(client: RiotApiClient, riot_id: str) -> str:
    """
    Resolves a Riot ID (GameName#TAG) to a PUUID.

    Args:
        client: Initialized RiotApiClient.
        riot_id: Riot ID string in the format "GameName#TAG".

    Returns:
        PUUID string.

    Raises:
        SystemExit: If the Riot ID cannot be resolved.
    """
    logger = logging.getLogger(__name__)
    try:
        game_name, tag_line = riot_id.split("#", 1)
    except ValueError:
        logger.error("Invalid Riot ID format '%s'. Expected: GameName#TAG", riot_id)
        sys.exit(1)

    logger.info("Resolving %s#%s ...", game_name, tag_line)
    account = client.get_account_by_riot_id(game_name, tag_line)
    if not account:
        logger.error("Could not resolve '%s#%s'. Check the Riot ID and region.", riot_id, "")
        sys.exit(1)

    puuid: str = account["puuid"]
    logger.info(
        "Resolved -> PUUID: %s... (gameName=%s, tagLine=%s)",
        puuid[:20],
        account.get("gameName"),
        account.get("tagLine"),
    )
    return puuid


def fetch_all_matches(
    conn: sqlite3.Connection,
    client: RiotApiClient,
    puuid: str,
    limit: int = 0,
) -> dict[str, int]:
    """
    Fetches and stores all ranked solo matches for a player.

    Paginates through the complete match history (100 IDs per page),
    skips matches already in the DB, and inserts new ones.

    Args:
        conn: Active SQLite connection.
        client: Initialized RiotApiClient.
        puuid: Player's PUUID.
        limit: Maximum number of matches to fetch (0 = unlimited).

    Returns:
        Summary dict with keys: total_found, already_in_db, newly_inserted, errors.
    """
    logger = logging.getLogger(__name__)

    # ── Step 1: Collect all match IDs ────────────────────────────────────────
    logger.info("Fetching complete match history (paginating 100/page)...")
    all_ids: list[str] = client.get_all_match_ids_by_puuid(
        puuid=puuid,
        queue=RANKED_SOLO_QUEUE,
    )

    total_found = len(all_ids)
    if total_found == 0:
        logger.warning("No ranked solo matches found for this player.")
        return {"total_found": 0, "already_in_db": 0, "newly_inserted": 0, "errors": 0}

    logger.info("Found %d ranked solo matches in history.", total_found)

    # Apply optional limit
    if limit > 0:
        all_ids = all_ids[:limit]
        logger.info("Limiting to %d most recent matches (--limit %d).", limit, limit)

    # ── Step 2: Process each match ────────────────────────────────────────────
    already_in_db = 0
    newly_inserted = 0
    errors = 0

    for i, match_id in enumerate(all_ids, start=1):
        prefix = f"[{i:>4}/{len(all_ids)}]"

        if match_exists(conn, match_id):
            logger.debug("%s %s already in DB, skipping.", prefix, match_id)
            already_in_db += 1
            continue

        try:
            result = process_match(conn, client, match_id)
            if result:
                newly_inserted += 1
                logger.info(
                    "%s %s inserted (+%d PUUIDs discovered)",
                    prefix, match_id, len(result),
                )
            else:
                # process_match returned [] — match fetched but not stored
                # (e.g. wrong queue filter or already existed mid-run)
                already_in_db += 1

        except Exception as exc:
            logger.error("%s Error on %s: %s", prefix, match_id, exc, exc_info=True)
            errors += 1

    return {
        "total_found": total_found,
        "already_in_db": already_in_db,
        "newly_inserted": newly_inserted,
        "errors": errors,
    }


def print_summary(
    riot_id_or_puuid: str,
    puuid: str,
    stats: dict[str, int],
    conn: sqlite3.Connection,
) -> None:
    """Prints a clean run summary to the logger."""
    logger = logging.getLogger(__name__)

    total_matches_in_db = get_match_count(conn)
    total_participants = conn.execute("SELECT COUNT(*) FROM participants").fetchone()[0]

    sep = "+" + "-" * 46 + "+"
    logger.info(sep)
    logger.info("| %-44s |", "PLAYER FETCH SUMMARY")
    logger.info(sep)
    logger.info("| Player : %-35s |", riot_id_or_puuid)
    logger.info("| PUUID  : %-35s |", puuid[:35] + "...")
    logger.info(sep)
    logger.info("| Matches found in history   : %14d |", stats["total_found"])
    logger.info("| Already in DB (skipped)    : %14d |", stats["already_in_db"])
    logger.info("| Newly inserted             : %14d |", stats["newly_inserted"])
    logger.info("| Errors                     : %14d |", stats["errors"])
    logger.info(sep)
    logger.info("| Total matches in DB        : %14d |", total_matches_in_db)
    logger.info("| Total participants in DB   : %14d |", total_participants)
    logger.info(sep)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    api_key: str = os.getenv("RIOT_API_KEY", "").strip()
    if not api_key or not api_key.startswith("RGAPI-"):
        logger.error(
            "RIOT_API_KEY not found or invalid. "
            "Create a .env file with: RIOT_API_KEY=RGAPI-..."
        )
        sys.exit(1)

    logger.info("Initializing database...")
    conn: sqlite3.Connection = init_db()
    client: RiotApiClient = RiotApiClient(api_key=api_key)

    # Resolve identity
    if args.riot_id:
        puuid = resolve_puuid(client, args.riot_id)
        label = args.riot_id
    else:
        puuid = args.puuid
        label = puuid[:20] + "..."
        logger.info("Using PUUID directly: %s...", puuid[:20])

    # Fetch
    stats = fetch_all_matches(
        conn=conn,
        client=client,
        puuid=puuid,
        limit=args.limit,
    )

    print_summary(label, puuid, stats, conn)
    conn.close()


if __name__ == "__main__":
    main()
