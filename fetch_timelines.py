"""
fetch_timelines.py — Récupère la chronologie (timeline) d'un échantillon de matchs déjà en base.

La timeline Match-V5 donne l'état de chaque joueur minute par minute. On en
garde quelques images (10, 15 et 20 minutes : or, XP, farm, dégâts) pour
mesurer la phase de lane seule, sans le reste de la partie.

Une requête par match. Le crawler utilise la même clé et donc les mêmes
quotas : lancés ensemble, chacun va deux fois moins vite.

Les matchs sont pris au hasard parmi ceux des N derniers jours (hors remakes).
La progression est enregistrée en base (table timelines) : relancer reprend là
où on s'était arrêté. Clé expirée : pause jusqu'à ce que .env contienne une
nouvelle clé, comme le crawler.

Usage :
  python fetch_timelines.py                  # sans limite, matchs des 30 derniers jours
  python fetch_timelines.py --limit 20000
  python fetch_timelines.py --days-back 60
"""
import argparse
import logging
import os
import sqlite3
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

from dotenv import dotenv_values, load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

ENV_PATH: Path = Path(__file__).parent / ".env"
load_dotenv(ENV_PATH)

from api.client import ApiKeyError, ApiUnavailableError, RiotApiClient
from config import RANKED_SOLO_QUEUE
from db.repository import get_matches_without_timeline, get_timeline_count, upsert_timeline
from db.schema import init_db

MINUTES: tuple[int, ...] = (10, 15, 20)
BATCH_SIZE: int = 200
KEY_POLL_INTERVAL_S: int = 60
UNAVAILABLE_PAUSE_S: int = 60
PROGRESS_LOG_EVERY: int = 100

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Logs vers la console ET fetch_timelines.log (rotation 10 Mo x 3)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            RotatingFileHandler("fetch_timelines.log", maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("requests").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Récupère les timelines d'un échantillon de matchs en base.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--limit", type=int, default=0, help="Nombre max de timelines à récupérer (0 = sans limite).")
    p.add_argument("--days-back", type=int, default=30, help="Ne prendre que les matchs des N derniers jours.")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def read_api_key_from_env_file() -> Optional[str]:
    """Relit .env sur disque (la clé peut avoir été remplacée en cours de route)."""
    key: str = (dotenv_values(ENV_PATH).get("RIOT_API_KEY") or "").strip()
    return key if key.startswith("RGAPI-") else None


def wait_for_new_key(client: RiotApiClient) -> None:
    """Bloque jusqu'à ce qu'une clé différente de la clé refusée soit dans .env."""
    logger.warning(
        "Clé API refusée (expirée ?). Pause : régénère une clé sur https://developer.riotgames.com "
        "et colle-la dans .env (RIOT_API_KEY=...). Reprise automatique sous %ds.", KEY_POLL_INTERVAL_S,
    )
    rejected: str = client.api_key
    while True:
        time.sleep(KEY_POLL_INTERVAL_S)
        new_key = read_api_key_from_env_file()
        if new_key and new_key != rejected:
            client.set_api_key(new_key)
            logger.info("Nouvelle clé API détectée — reprise.")
            return


def fetch(conn: sqlite3.Connection, client: RiotApiClient, since_ms: int, limit: int) -> int:
    """Récupère les timelines manquantes jusqu'à `limit` (0 = toutes). Retourne le nombre traité."""
    done, start = 0, time.monotonic()
    while not limit or done < limit:
        batch = get_matches_without_timeline(conn, since_ms, RANKED_SOLO_QUEUE, BATCH_SIZE)
        if not batch:
            logger.info("Plus aucun match sans timeline sur la période.")
            break
        for match_id in batch:
            if limit and done >= limit:
                break
            try:
                timeline = client.get_match_timeline(match_id)
            except ApiKeyError:
                wait_for_new_key(client)
                continue  # Le match reste sans timeline : il sera repris dans un prochain lot
            except ApiUnavailableError as exc:
                logger.warning("%s — nouvelle tentative dans %ds.", exc, UNAVAILABLE_PAUSE_S)
                time.sleep(UNAVAILABLE_PAUSE_S)
                continue
            try:
                upsert_timeline(conn, match_id, timeline, MINUTES)
                conn.commit()
            except Exception as exc:
                # Données inattendues : on saute ce match, la collecte continue
                conn.rollback()
                logger.error("Erreur sur la timeline %s : %s", match_id, exc, exc_info=True)
                continue
            done += 1
            if done % PROGRESS_LOG_EVERY == 0:
                hours = max((time.monotonic() - start) / 3600, 1e-9)
                logger.info("%d timelines en base | +%d cette session (%.0f/h)",
                            get_timeline_count(conn), done, done / hours)
    return done


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    api_key: str = os.getenv("RIOT_API_KEY", "").strip()
    if not api_key.startswith("RGAPI-"):
        logger.error("RIOT_API_KEY introuvable ou invalide. Créez un fichier .env avec : RIOT_API_KEY=RGAPI-...")
        sys.exit(1)

    conn = init_db()
    client = RiotApiClient(api_key=api_key)
    since_ms = int((time.time() - args.days_back * 86400) * 1000)
    logger.info("═══ Démarrage | %d timelines en base | matchs des %d derniers jours | cible : %s ═══",
                get_timeline_count(conn), args.days_back, args.limit or "illimitée")
    done = 0
    try:
        done = fetch(conn, client, since_ms, args.limit)
    except KeyboardInterrupt:
        logger.info("Interruption utilisateur (Ctrl+C) — arrêt propre.")
    finally:
        logger.info("RÉSUMÉ — %d timelines en base", get_timeline_count(conn))
        conn.close()


if __name__ == "__main__":
    main()
