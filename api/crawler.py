"""
crawler/crawler.py — Logique d'exploration BFS et d'extraction de matchs.

Deux composants :
  1. process_match()   : Prend un match_id, appelle Match-V5, insère en DB,
                         retourne les 10 PUUIDs des participants.
  2. DraftCrawler      : Boucle BFS principale. Part d'un joueur seed,
                         récupère ses matchs, extrait les participants,
                         les réenfile, recommence jusqu'à max_matches.

Résilience :
  - Erreurs d'un match individuel → logged + skip, le crawler continue.
  - Erreurs d'un joueur → marqué 'error' dans crawl_queue, le crawler passe au suivant.
  - Interruption clavier (Ctrl+C) → arrêt propre avec affichage des stats.
"""
import logging
import sqlite3
from typing import Optional

from api.client import RiotApiClient
from config import MAX_MATCHES_PER_SUMMONER, RANKED_SOLO_QUEUE
from db.repository import (
    enqueue_puuids,
    get_match_count,
    get_pending_puuid,
    get_queue_stats,
    mark_puuid_done,
    mark_puuid_error,
    match_exists,
    upsert_bans,
    upsert_match,
    upsert_participant,
)

logger = logging.getLogger(__name__)


# ── Étape 3 : Extraction d'un match ──────────────────────────────────────────

def process_match(
    conn: sqlite3.Connection,
    client: RiotApiClient,
    match_id: str,
) -> list[str]:
    """
    Récupère un match via l'API et insère ses données en base.

    Pipeline :
      1. Vérifie si le match est déjà en DB (skip si oui).
      2. Appelle GET /lol/match/v5/matches/{matchId}.
      3. Insère dans : matches → bans → participants (dans cet ordre, FK-safe).
      4. Commit et retourne la liste des 10 PUUIDs participants.

    Args:
        conn: Connexion SQLite active.
        client: Instance du client API Riot.
        match_id: Identifiant du match (ex: "EUW1_7123456789").

    Returns:
        Liste des PUUIDs participants (10 joueurs), vide si erreur ou déjà traité.
    """
    if match_exists(conn, match_id):
        logger.debug("Match %s déjà en base, skip.", match_id)
        return []

    match_data: Optional[dict] = client.get_match(match_id)
    if match_data is None:
        logger.warning("Impossible de récupérer le match %s.", match_id)
        return []

    # Vérification de cohérence : queue_id attendu
    info: dict = match_data.get("info", {})
    queue_id: int = info.get("queueId", -1)
    if queue_id != RANKED_SOLO_QUEUE:
        logger.debug(
            "Match %s ignoré (queue_id=%d, attendu %d).",
            match_id, queue_id, RANKED_SOLO_QUEUE,
        )
        return []

    api_match_id: str = match_data["metadata"]["matchId"]
    participants_data: list[dict] = info.get("participants", [])
    teams_data: list[dict] = info.get("teams", [])

    # ── Insertion en DB (ordre FK : matches d'abord) ──────────────────────
    upsert_match(conn, match_data)
    upsert_bans(conn, api_match_id, teams_data)

    puuids: list[str] = []
    for participant in participants_data:
        upsert_participant(conn, api_match_id, participant)
        puuid: Optional[str] = participant.get("puuid")
        if puuid:
            puuids.append(puuid)

    conn.commit()

    total_bans: int = sum(len(t.get("bans", [])) for t in teams_data)
    logger.info(
        "✓ Match %s | %d participants | %d bans | v%s | durée %ds",
        api_match_id,
        len(participants_data),
        total_bans,
        info.get("gameVersion", "?"),
        info.get("gameDuration", 0),
    )

    return puuids


# ── Étape 4 : Crawler BFS ─────────────────────────────────────────────────────

class DraftCrawler:
    """
    Crawler BFS pour la collecte de matchs LoL Ranked Solo.

    Algorithme :
      seed(puuid)         → insère le joueur de départ dans crawl_queue
      run(max_matches)    → boucle BFS jusqu'à max_matches matchs collectés

    À chaque itération :
      1. Dépile un PUUID 'pending' (FIFO).
      2. Récupère jusqu'à MAX_MATCHES_PER_SUMMONER match_ids via Match-V5.
      3. Pour chaque match_id : process_match() → insère + retourne 10 PUUIDs.
      4. Enfile les nouveaux PUUIDs (INSERT OR IGNORE → pas de doublon).
      5. Marque le PUUID courant 'done'.

    Args:
        conn: Connexion SQLite active.
        client: Client API Riot initialisé.
        queue_id: Queue à cibler (défaut : Ranked Solo = 420).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        client: RiotApiClient,
        queue_id: int = RANKED_SOLO_QUEUE,
        start_time: int | None = None,
    ) -> None:
        self.conn: sqlite3.Connection = conn
        self.client: RiotApiClient = client
        self.queue_id: int = queue_id
        self.start_time: int | None = start_time  # Epoch Unix en secondes

    def seed(self, puuid: str) -> None:
        """
        Insère le PUUID de départ dans la crawl_queue.

        Args:
            puuid: PUUID Riot du joueur seed.
        """
        added: int = enqueue_puuids(self.conn, [puuid])
        self.conn.commit()
        if added:
            logger.info("Seed enfilé : %s", puuid)
        else:
            logger.info("Seed déjà en queue : %s", puuid)

    def run(self, max_matches: int = 100) -> None:
        """
        Lance la boucle BFS jusqu'à collecter max_matches matchs.

        La boucle s'arrête si :
          - max_matches est atteint.
          - La crawl_queue est vide (plus de joueurs à explorer).
          - L'utilisateur interrompt avec Ctrl+C.

        Args:
            max_matches: Nombre cible de matchs à collecter.
        """
        logger.info(
            "═══ Démarrage du crawler | Cible : %d matchs | Queue : %d ═══",
            max_matches,
            self.queue_id,
        )

        try:
            self._crawl_loop(max_matches)
        except KeyboardInterrupt:
            logger.info("Interruption utilisateur (Ctrl+C) — arrêt propre.")
        finally:
            self._log_final_stats()

    def _crawl_loop(self, max_matches: int) -> None:
        """Boucle principale BFS."""
        while True:
            current_count: int = get_match_count(self.conn)
            if current_count >= max_matches:
                logger.info(
                    "Objectif atteint : %d/%d matchs collectés.", current_count, max_matches
                )
                break

            puuid: Optional[str] = get_pending_puuid(self.conn)
            if puuid is None:
                logger.warning("File d'attente vide — plus de joueurs à explorer.")
                break

            try:
                self._process_player(puuid, max_matches)
            except ValueError as exc:
                # Clé API invalide/expirée (401) — arrêt immédiat
                logger.error("Arrêt du crawler : %s", exc)
                logger.error(
                    "Obtenez une nouvelle clé sur https://developer.riotgames.com "
                    "et mettez à jour le fichier .env"
                )
                break

    def _process_player(self, puuid: str, max_matches: int) -> None:
        """
        Traite un joueur : récupère ses match_ids et les process un par un.

        Args:
            puuid: PUUID du joueur à traiter.
            max_matches: Limite globale de matchs.
        """
        short_id: str = puuid[:16] + "…"
        current_count: int = get_match_count(self.conn)
        logger.info(
            "─── Joueur %s | %d/%d matchs collectés",
            short_id, current_count, max_matches,
        )

        try:
            match_ids: list[str] = self.client.get_match_ids_by_puuid(
                puuid=puuid,
                queue=self.queue_id,
                count=MAX_MATCHES_PER_SUMMONER,
                start_time=self.start_time,
            )

            if not match_ids:
                logger.info("Aucun match ranked solo trouvé pour %s.", short_id)
                mark_puuid_done(self.conn, puuid)
                self.conn.commit()
                return

            logger.info(
                "%d match IDs récupérés pour %s.", len(match_ids), short_id
            )

            for match_id in match_ids:
                if get_match_count(self.conn) >= max_matches:
                    break

                try:
                    new_puuids: list[str] = process_match(
                        self.conn, self.client, match_id
                    )
                    if new_puuids:
                        added: int = enqueue_puuids(self.conn, new_puuids)
                        self.conn.commit()
                        logger.debug(
                            "%d nouveaux joueurs enfilés depuis %s.", added, match_id
                        )
                except Exception as match_exc:
                    logger.error(
                        "Erreur sur le match %s : %s", match_id, match_exc, exc_info=True
                    )
                    # On continue avec le match suivant — résilience

            mark_puuid_done(self.conn, puuid)
            self.conn.commit()

        except Exception as player_exc:
            logger.error(
                "Erreur fatale sur le joueur %s : %s", short_id, player_exc, exc_info=True
            )
            mark_puuid_error(self.conn, puuid)
            self.conn.commit()

    def _log_final_stats(self) -> None:
        """Affiche un résumé final des stats de collecte."""
        match_count: int = get_match_count(self.conn)
        queue_stats: dict[str, int] = get_queue_stats(self.conn)

        logger.info("═" * 50)
        logger.info("RÉSUMÉ FINAL")
        logger.info("  Matchs collectés : %d", match_count)
        logger.info("  Queue — pending: %d | done: %d | error: %d",
                    queue_stats.get("pending", 0),
                    queue_stats.get("done", 0),
                    queue_stats.get("error", 0))
        logger.info("═" * 50)
