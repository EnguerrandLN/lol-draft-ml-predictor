"""
main.py — Point d'entrée CLI du pipeline de collecte Riot Draft.

Usage :
  # Résolution d'un Riot ID en PUUID puis crawl
  python main.py --riot-id KeytedLN#EUW --max-matches 5

  # Avec un PUUID direct (si déjà connu)
  python main.py --puuid <PUUID> --max-matches 100

  # Mode debug avec logs détaillés
  python main.py --riot-id KeytedLN#EUW --max-matches 5 --log-level DEBUG
"""
import argparse
import io
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

# Assurer que le répertoire courant est dans le PATH Python
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv()

from api.client import RiotApiClient
from api.crawler import DraftCrawler
from db.schema import init_db


def setup_logging(level: str = "INFO") -> None:
    """
    Configure le logging vers la console ET un fichier crawler.log.

    Args:
        level: Niveau de log (DEBUG, INFO, WARNING, ERROR).
    """
    log_level: int = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s - %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    # Force UTF-8 sur stdout/stderr pour eviter les erreurs cp1252 sous Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("crawler.log", encoding="utf-8"),
    ]

    logging.basicConfig(level=log_level, format=fmt, datefmt=date_fmt, handlers=handlers)

    # Réduire le bruit des bibliothèques externes
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Riot Draft Data Crawler — Collecte de matchs LoL Ranked Solo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--riot-id",
        type=str,
        metavar="NAME#TAG",
        help="Riot ID du joueur seed (ex: KeytedLN#EUW). Résolu via Account-V1.",
    )
    group.add_argument(
        "--puuid",
        type=str,
        metavar="PUUID",
        help="PUUID du joueur seed (alternative à --riot-id).",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=100,
        metavar="N",
        help="Nombre cible de matchs à collecter.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Reprend le crawl depuis la queue existante sans résoudre de nouveau joueur seed. "
            "Utile après une interruption (clé expirée, Ctrl+C...)."
        ),
    )
    parser.add_argument(
        "--months-back",
        type=int,
        default=6,
        metavar="N",
        help="Ne collecter que les matchs des N derniers mois (0 = pas de limite).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Niveau de verbosité des logs.",
    )
    return parser.parse_args()


def print_db_summary(conn: sqlite3.Connection) -> None:
    """Affiche un resume structure du contenu de la base."""
    logger = logging.getLogger(__name__)

    queries = {
        "Matchs":        "SELECT COUNT(*) FROM matches",
        "Participants":  "SELECT COUNT(*) FROM participants",
        "Bans":          "SELECT COUNT(*) FROM bans",
        "Joueurs vus":   "SELECT COUNT(*) FROM summoner_cache",
        "Queue done":    "SELECT COUNT(*) FROM crawl_queue WHERE status='done'",
        "Queue error":   "SELECT COUNT(*) FROM crawl_queue WHERE status='error'",
        "Queue pending": "SELECT COUNT(*) FROM crawl_queue WHERE status='pending'",
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

    # ── Calcul du filtre temporel ─────────────────────────────────────────
    start_time: int | None = None
    if args.months_back > 0:
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=args.months_back * 30)
        start_time = int(cutoff_dt.timestamp())
        logger.info(
            "Filtre temporel : matchs depuis le %s (%d mois en arrière)",
            cutoff_dt.strftime("%Y-%m-%d"),
            args.months_back,
        )

    # ── Initialisation ────────────────────────────────────────────────────
    logger.info("Initialisation de la base de données...")
    conn: sqlite3.Connection = init_db()

    client: RiotApiClient = RiotApiClient(api_key=api_key)
    crawler: DraftCrawler = DraftCrawler(conn=conn, client=client, start_time=start_time)

    # ── Résolution du joueur seed (ignorée en mode --resume) ─────────────
    if args.resume:
        pending = conn.execute(
            "SELECT COUNT(*) FROM crawl_queue WHERE status='pending'"
        ).fetchone()[0]
        done = conn.execute(
            "SELECT COUNT(*) FROM crawl_queue WHERE status='done'"
        ).fetchone()[0]
        if pending == 0:
            logger.error(
                "Mode --resume : aucun joueur en attente dans la queue. "
                "Lancez d'abord avec --riot-id pour initialiser le seed."
            )
            sys.exit(1)
        logger.info(
            "Mode --resume : reprise depuis la queue existante "
            "(%d pending | %d done).", pending, done
        )
    else:
        seed_puuid: str = args.puuid or ""

        if not seed_puuid and args.riot_id:
            try:
                game_name, tag_line = args.riot_id.split("#", 1)
            except ValueError:
                logger.error(
                    "Format Riot ID invalide : '%s'. Attendu : GameName#TAG", args.riot_id
                )
                sys.exit(1)

            logger.info("Résolution du Riot ID : %s#%s ...", game_name, tag_line)
            account: dict | None = client.get_account_by_riot_id(game_name, tag_line)
            if not account:
                logger.error(
                    "Impossible de résoudre '%s#%s'. Vérifiez le Riot ID et la région.",
                    game_name, tag_line,
                )
                sys.exit(1)

            seed_puuid = account["puuid"]
            logger.info("PUUID résolu : %s", seed_puuid)

        if not seed_puuid:
            logger.error("Fournissez --riot-id, --puuid, ou --resume pour démarrer le crawl.")
            sys.exit(1)

        crawler.seed(seed_puuid)

    # ── Lancement du crawler ──────────────────────────────────────────────
    crawler.run(max_matches=args.max_matches)

    # ── Résumé final ──────────────────────────────────────────────────────
    print_db_summary(conn)
    conn.close()


if __name__ == "__main__":
    main()
