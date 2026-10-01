"""
crawl.py — Point d'entrée CLI du crawler de matchs Ranked Solo.

Le crawler échantillonne des joueurs dans le ladder (par tier/division) et
collecte leurs parties récentes. Il tourne en continu par passes successives
et reprend automatiquement là où il s'était arrêté (état en base).

Usage :
  # Crawl continu avec les paramètres par défaut (config.py)
  python crawl.py

  # Uniquement le haut elo, sur les 14 derniers jours
  python crawl.py --tiers EMERALD DIAMOND MASTER GRANDMASTER CHALLENGER --days-back 14

  # S'arrêter à 200 000 matchs en base
  python crawl.py --max-matches 200000

Clé API : lue dans .env (RIOT_API_KEY=RGAPI-...). Si elle expire en cours de
route, le crawler se met en pause et reprend dès que .env contient une
nouvelle clé — inutile de le relancer.
"""
import argparse
import logging
import os
import sqlite3
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values, load_dotenv

# Assurer que le répertoire courant est dans le PATH Python
sys.path.insert(0, str(Path(__file__).parent))

ENV_PATH: Path = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)

from api.client import RiotApiClient
from api.crawler import ALL_TIERS, LadderCrawler
from config import (
    CRAWL_DAYS_BACK,
    CRAWL_TIERS,
    LADDER_PAGES_PER_DIVISION,
    MAX_MATCHES_PER_SUMMONER,
)
from db.schema import init_db


def setup_logging(level: str = "INFO") -> None:
    """
    Configure le logging vers la console ET crawler.log (rotation 10 Mo x 3).

    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR).
    """
    log_level: int = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            "crawler.log", maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
        ),
    ]

    logging.basicConfig(level=log_level, format=fmt, datefmt=date_fmt, handlers=handlers)

    # Réduire le bruit des bibliothèques externes (les retries transport
    # d'urllib3 sont normaux ; seuls les échecs persistants sont loggés)
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("requests").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Riot Draft Data Crawler — Collecte de matchs LoL Ranked Solo par ladder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=list(CRAWL_TIERS),
        type=str.upper,
        choices=ALL_TIERS,
        metavar="TIER",
        help=f"Tiers à échantillonner parmi : {' '.join(ALL_TIERS)}.",
    )
    parser.add_argument(
        "--pages-per-division",
        type=int,
        default=LADDER_PAGES_PER_DIVISION,
        metavar="N",
        help="Pages League-V4 (~205 joueurs) lues par division à chaque passe.",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=CRAWL_DAYS_BACK,
        metavar="N",
        help="Ne collecter que les parties des N derniers jours.",
    )
    parser.add_argument(
        "--matches-per-player",
        type=int,
        default=MAX_MATCHES_PER_SUMMONER,
        metavar="N",
        help="Nombre max de parties demandées par joueur et par passe (max 100).",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=0,
        metavar="N",
        help="S'arrêter quand la base contient N matchs (0 = crawl continu).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Niveau de verbosité des logs.",
    )
    return parser.parse_args()


def read_api_key_from_env_file() -> Optional[str]:
    """Relit .env sur disque (la clé peut avoir été remplacée pendant le crawl)."""
    key: str = (dotenv_values(ENV_PATH).get("RIOT_API_KEY") or "").strip()
    return key if key.startswith("RGAPI-") else None


def print_db_summary(conn: sqlite3.Connection) -> None:
    """Affiche un resume structure du contenu de la base."""
    logger = logging.getLogger(__name__)

    queries = {
        "Matchs":            "SELECT COUNT(*) FROM matches",
        "Matchs (remakes)":  "SELECT COUNT(*) FROM matches WHERE ended_early = 1",
        "Matchs avec tier":  "SELECT COUNT(*) FROM matches WHERE source_tier IS NOT NULL",
        "Ladder pending":    "SELECT COUNT(*) FROM ladder_players WHERE status='pending'",
        "Ladder done":       "SELECT COUNT(*) FROM ladder_players WHERE status='done'",
    }

    sep = "+" + "-" * 38 + "+"
    logger.info(sep)
    logger.info("| %-36s |", "ETAT DE LA BASE DE DONNEES")
    logger.info(sep)
    for label, query in queries.items():
        count = conn.execute(query).fetchone()[0]
        logger.info("| %-22s : %10d |", label, count)
    logger.info(sep)


def main() -> None:
    # Force UTF-8 sur stdout/stderr pour eviter les erreurs cp1252 sous Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    # ── Vérification de la clé API ────────────────────────────────────────
    api_key: str = os.getenv("RIOT_API_KEY", "").strip()
    if not api_key or not api_key.startswith("RGAPI-"):
        logger.error(
            "RIOT_API_KEY introuvable ou invalide. "
            "Créez un fichier .env avec : RIOT_API_KEY=RGAPI-..."
        )
        sys.exit(1)

    # ── Initialisation ────────────────────────────────────────────────────
    logger.info("Initialisation de la base de données...")
    conn: sqlite3.Connection = init_db()

    client: RiotApiClient = RiotApiClient(api_key=api_key)
    crawler = LadderCrawler(
        conn=conn,
        client=client,
        tiers=args.tiers,
        pages_per_division=args.pages_per_division,
        days_back=args.days_back,
        matches_per_player=args.matches_per_player,
        reload_api_key=read_api_key_from_env_file,
    )

    # ── Lancement du crawler ──────────────────────────────────────────────
    crawler.run(max_matches=args.max_matches)

    # ── Résumé final ──────────────────────────────────────────────────────
    print_db_summary(conn)
    conn.close()


if __name__ == "__main__":
    main()
