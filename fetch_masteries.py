"""
fetch_masteries.py — Photographie la maîtrise de champions des joueurs du ladder.

La maîtrise (Champion-Mastery-V4) donne, pour chaque champion d'un joueur, ses
points cumulés, son niveau et la date de sa dernière partie, toutes files
confondues. Elle sert à mesurer le coût d'un champion peu joué au-delà de ce
que montre l'historique en base (~20 parties par joueur).

Elle est mesurée à un instant donné : elle inclut toutes les parties d'avant,
y compris celles déjà en base. Pour éviter de prédire le passé avec de
l'information future, elle ne sera utilisée que pour les parties jouées APRÈS
la photo (que le crawler collecte ensuite) : c'est aussi l'usage de l'app,
qui lit la maîtrise à l'import puis conseille les parties suivantes.

Une requête par joueur sur le routage plateforme (euw1), dont le quota est
distinct de celui de Match-V5 (europe) : il peut tourner en même temps que le
crawler. Reprend où il s'est arrêté ; clé expirée : pause jusqu'à ce que .env
contienne une nouvelle clé.

Usage :
  python fetch_masteries.py               # tous les joueurs du ladder sans photo
  python fetch_masteries.py --limit 2000
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
from db.repository import get_mastery_count, get_players_without_mastery, upsert_masteries
from db.schema import init_db

BATCH_SIZE: int = 200
KEY_POLL_INTERVAL_S: int = 60
UNAVAILABLE_PAUSE_S: int = 60
PROGRESS_LOG_EVERY: int = 200

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    """Logs vers la console ET fetch_masteries.log (rotation 10 Mo x 3)."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)-8s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            RotatingFileHandler("fetch_masteries.log", maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"),
        ],
    )
    logging.getLogger("urllib3").setLevel(logging.ERROR)
    logging.getLogger("requests").setLevel(logging.WARNING)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Photographie la maîtrise des joueurs du ladder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--limit", type=int, default=0, help="Nombre max de joueurs (0 = tous).")
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


def fetch(conn: sqlite3.Connection, client: RiotApiClient, limit: int) -> int:
    """Photographie les joueurs sans maîtrise jusqu'à `limit` (0 = tous). Retourne le nombre traité."""
    done, start = 0, time.monotonic()
    while not limit or done < limit:
        batch = get_players_without_mastery(conn, BATCH_SIZE)
        if not batch:
            logger.info("Tous les joueurs du ladder ont leur maîtrise.")
            break
        for puuid in batch:
            if limit and done >= limit:
                break
            try:
                masteries = client.get_champion_masteries(puuid)
            except ApiKeyError:
                wait_for_new_key(client)
                continue  # Le joueur reste sans photo : il sera repris dans un prochain lot
            except ApiUnavailableError as exc:
                logger.warning("%s — nouvelle tentative dans %ds.", exc, UNAVAILABLE_PAUSE_S)
                time.sleep(UNAVAILABLE_PAUSE_S)
                continue
            upsert_masteries(conn, puuid, masteries, int(time.time() * 1000))
            conn.commit()
            done += 1
            if done % PROGRESS_LOG_EVERY == 0:
                hours = max((time.monotonic() - start) / 3600, 1e-9)
                logger.info("%d joueurs en base | +%d cette session (%.0f/h)", get_mastery_count(conn), done, done / hours)
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
    logger.info("═══ Démarrage | %d joueurs avec maîtrise | cible : %s ═══",
                get_mastery_count(conn), args.limit or "tous les joueurs du ladder")
    try:
        fetch(conn, client, args.limit)
    except KeyboardInterrupt:
        logger.info("Interruption utilisateur (Ctrl+C) — arrêt propre.")
    finally:
        logger.info("RÉSUMÉ — %d joueurs avec maîtrise en base", get_mastery_count(conn))
        conn.close()


if __name__ == "__main__":
    main()
