"""
api/crawler.py — Collecte de matchs Ranked Solo par échantillonnage du ladder.

Deux composants :
  1. process_match()   : Prend un match_id, appelle Match-V5, insère en DB,
                         retourne les 10 PUUIDs des participants.
  2. LadderCrawler     : Boucle principale. Échantillonne des joueurs dans
                         chaque tier/division du ladder (League-V4), puis
                         récupère leurs parties récentes.

Pourquoi le ladder plutôt qu'un BFS :
  - Chaque match est étiqueté avec le rang du joueur par lequel il a été
    trouvé (source_tier / source_division). Le matchmaking regroupant des
    joueurs de MMR proche, c'est un proxy correct de l'ELO du match.
  - La répartition par ELO est contrôlée (même nombre de joueurs par
    division, ordre de traitement aléatoire) au lieu de dériver au hasard.

Fonctionnement en continu :
  - Quand tous les joueurs ont été traités, une nouvelle passe commence :
    le ladder est re-échantillonné et chaque joueur n'est interrogé que sur
    les parties jouées depuis la passe précédente.
  - Clé API expirée (401/403) : le crawler se met en pause et relit le
    fichier .env jusqu'à ce qu'une nouvelle clé y soit écrite.
  - API / réseau indisponible : pause puis reprise sur le même joueur.
  - Un joueur n'est marqué 'done' qu'après succès : une erreur ne « brûle »
    jamais la file.
"""
import logging
import random
import sqlite3
import time
from typing import Callable, Optional

from api.client import APEX_TIERS, ApiKeyError, ApiUnavailableError, RiotApiClient
from config import RANKED_SOLO_QUEUE
from db.repository import (
    get_ladder_stats,
    get_match_count,
    get_next_ladder_player,
    mark_ladder_player_done,
    match_exists,
    upsert_bans,
    upsert_ladder_players,
    upsert_match,
    upsert_participant,
)

logger = logging.getLogger(__name__)

DIVISION_TIERS: tuple[str, ...] = ("IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND")
ALL_TIERS: tuple[str, ...] = DIVISION_TIERS + tuple(APEX_TIERS)
DIVISIONS: tuple[str, ...] = ("I", "II", "III", "IV")
LADDER_PAGE_SIZE: int = 205  # Taille d'une page League-V4

KEY_POLL_INTERVAL_S: int = 60
UNAVAILABLE_PAUSE_S: int = 60
PROGRESS_LOG_EVERY: int = 100


# ── Extraction d'un match ─────────────────────────────────────────────────────

def process_match(
    conn: sqlite3.Connection,
    client: RiotApiClient,
    match_id: str,
    source_tier: Optional[str] = None,
    source_division: Optional[str] = None,
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
        source_tier: Tier du joueur par lequel le match a été trouvé.
        source_division: Division de ce joueur.

    Returns:
        Liste des PUUIDs participants (10 joueurs), vide si erreur ou déjà traité.

    Raises:
        ApiKeyError, ApiUnavailableError: propagées à l'appelant.
    """
    if match_exists(conn, match_id):
        logger.debug("Match %s déjà en base, skip.", match_id)
        return []

    match_data: Optional[dict] = client.get_match(match_id)
    if match_data is None:
        logger.warning("Match %s introuvable (404).", match_id)
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
    upsert_match(conn, match_data, source_tier, source_division)
    upsert_bans(conn, api_match_id, teams_data)

    puuids: list[str] = []
    for participant in participants_data:
        upsert_participant(conn, api_match_id, participant)
        puuid: Optional[str] = participant.get("puuid")
        if puuid:
            puuids.append(puuid)

    conn.commit()

    logger.debug(
        "✓ Match %s | %s %s | v%s | durée %ds",
        api_match_id,
        source_tier or "?",
        source_division or "",
        info.get("gameVersion", "?"),
        info.get("gameDuration", 0),
    )

    return puuids


# ── Crawler par ladder ────────────────────────────────────────────────────────

class LadderCrawler:
    """
    Crawler de matchs Ranked Solo échantillonnés par tier.

    Args:
        conn: Connexion SQLite active.
        client: Client API Riot initialisé.
        tiers: Tiers à échantillonner (ex : ["GOLD", ..., "CHALLENGER"]).
        pages_per_division: Pages League-V4 (~205 joueurs) lues par division.
            Les tiers apex sont plafonnés au même nombre de joueurs.
        days_back: Ne collecter que les parties des N derniers jours.
        matches_per_player: Nombre max de match IDs demandés par joueur et par passe.
        reload_api_key: Renvoie la clé actuellement écrite dans .env
            (appelé quand la clé en cours est refusée).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        client: RiotApiClient,
        tiers: list[str],
        pages_per_division: int,
        days_back: int,
        matches_per_player: int,
        reload_api_key: Callable[[], Optional[str]],
    ) -> None:
        self.conn = conn
        self.client = client
        self.tiers = tiers
        self.pages_per_division = pages_per_division
        self.days_back = days_back
        self.matches_per_player = matches_per_player
        self.reload_api_key = reload_api_key

        self._match_count: int = 0
        self._session_new: int = 0
        self._session_start: float = time.monotonic()

    # ── Boucle principale ────────────────────────────────────────────────────

    def run(self, max_matches: int = 0) -> None:
        """
        Collecte jusqu'à max_matches matchs en base (0 = sans limite).
        S'arrête proprement sur Ctrl+C.
        """
        self._match_count = get_match_count(self.conn)
        logger.info(
            "═══ Démarrage | %d matchs en base | cible : %s | tiers : %s ═══",
            self._match_count,
            max_matches or "illimitée",
            ", ".join(self.tiers),
        )
        try:
            self._crawl_loop(max_matches)
        except KeyboardInterrupt:
            logger.info("Interruption utilisateur (Ctrl+C) — arrêt propre.")
        finally:
            self._log_progress(final=True)

    def _crawl_loop(self, max_matches: int) -> None:
        while not max_matches or self._match_count < max_matches:
            try:
                player = get_next_ladder_player(self.conn)
                if player is None:
                    if self._snapshot_ladder() == 0:
                        logger.error("Aucun joueur récupéré depuis le ladder — arrêt.")
                        return
                    continue
                self._process_player(player, max_matches)
            except ApiKeyError:
                self._wait_for_new_key()
            except ApiUnavailableError as exc:
                logger.warning("%s — nouvelle tentative dans %ds.", exc, UNAVAILABLE_PAUSE_S)
                time.sleep(UNAVAILABLE_PAUSE_S)

        logger.info("Objectif atteint : %d matchs en base.", self._match_count)

    def _process_player(self, player: sqlite3.Row, max_matches: int) -> None:
        """Récupère les parties récentes d'un joueur, puis le marque 'done'."""
        now: int = int(time.time())
        start_time: int = max(now - self.days_back * 86400, player["last_crawled_at"] or 0)

        match_ids: list[str] = self.client.get_match_ids_by_puuid(
            puuid=player["puuid"],
            count=self.matches_per_player,
            start_time=start_time,
        )

        for match_id in match_ids:
            if max_matches and self._match_count >= max_matches:
                return  # Joueur laissé 'pending' : il sera repris au prochain lancement
            try:
                if process_match(
                    self.conn, self.client, match_id, player["tier"], player["division"]
                ):
                    self._on_new_match()
            except (ApiKeyError, ApiUnavailableError):
                raise
            except Exception as exc:
                # Données inattendues sur un match : on le saute, le crawl continue
                logger.error("Erreur sur le match %s : %s", match_id, exc, exc_info=True)

        mark_ladder_player_done(self.conn, player["puuid"], now)
        self.conn.commit()

    # ── Échantillonnage du ladder ────────────────────────────────────────────

    def _snapshot_ladder(self) -> int:
        """
        Échantillonne les joueurs de chaque tier/division et les remet en file
        avec un ordre aléatoire (les tiers sont ainsi mélangés tout au long
        de la passe). Retourne le nombre de joueurs enfilés.
        """
        logger.info("Nouvelle passe : échantillonnage du ladder...")
        players_per_bucket: int = self.pages_per_division * LADDER_PAGE_SIZE
        players: list[dict] = []

        for tier in self.tiers:
            if tier in APEX_TIERS:
                entries = self.client.get_apex_league(tier)
                if len(entries) > players_per_bucket:
                    entries = random.sample(entries, players_per_bucket)
                players += self._to_players(entries, tier, "I")
                continue

            for division in DIVISIONS:
                for page in range(1, self.pages_per_division + 1):
                    entries = self.client.get_league_entries(tier, division, page)
                    if not entries:
                        break  # Fin de la division
                    players += self._to_players(entries, tier, division)

        upsert_ladder_players(self.conn, players)
        self.conn.commit()

        counts: dict[str, int] = {}
        for p in players:
            counts[p["tier"]] = counts.get(p["tier"], 0) + 1
        logger.info(
            "Ladder échantillonné : %d joueurs (%s)",
            len(players),
            ", ".join(f"{t} {n}" for t, n in counts.items()),
        )
        return len(players)

    @staticmethod
    def _to_players(entries: list[dict], tier: str, division: str) -> list[dict]:
        players = []
        for entry in entries:
            puuid: Optional[str] = entry.get("puuid")
            if not puuid or entry.get("inactive"):
                continue
            players.append({
                "puuid": puuid,
                "tier": tier,
                "division": entry.get("rank", division),
                "league_points": entry.get("leaguePoints"),
                "priority": random.random(),
            })
        if entries and not any(entry.get("puuid") for entry in entries):
            logger.error(
                "Les entrées League-V4 de %s %s ne contiennent pas de puuid.", tier, division
            )
        return players

    # ── Clé API ──────────────────────────────────────────────────────────────

    def _wait_for_new_key(self) -> None:
        """Bloque jusqu'à ce qu'une clé différente de la clé refusée soit dans .env."""
        logger.warning(
            "Clé API refusée (expirée ?). Crawler en pause : régénère une clé sur "
            "https://developer.riotgames.com et colle-la dans .env (RIOT_API_KEY=...). "
            "Reprise automatique sous %ds après la mise à jour.",
            KEY_POLL_INTERVAL_S,
        )
        rejected_key: str = self.client.api_key
        while True:
            time.sleep(KEY_POLL_INTERVAL_S)
            new_key: Optional[str] = self.reload_api_key()
            if new_key and new_key != rejected_key:
                self.client.set_api_key(new_key)
                logger.info("Nouvelle clé API détectée — reprise du crawl.")
                return

    # ── Suivi ────────────────────────────────────────────────────────────────

    def _on_new_match(self) -> None:
        self._match_count += 1
        self._session_new += 1
        if self._session_new % PROGRESS_LOG_EVERY == 0:
            self._log_progress()

    def _log_progress(self, final: bool = False) -> None:
        hours: float = max((time.monotonic() - self._session_start) / 3600, 1e-9)
        stats: dict[str, int] = get_ladder_stats(self.conn)
        logger.info(
            "%s%d matchs en base | +%d cette session (%.0f/h) | joueurs : %d en attente, %d traités",
            "RÉSUMÉ FINAL — " if final else "",
            self._match_count,
            self._session_new,
            self._session_new / hours,
            stats.get("pending", 0),
            stats.get("done", 0),
        )
