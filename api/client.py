"""
api/client.py — Client HTTP sécurisé pour l'API Riot Games.

Fonctionnalités clés :
  - Rate limiting proactif par host (europe / euw1 ont des quotas séparés) :
    les fenêtres sont apprises depuis le header X-App-Rate-Limit, et chaque
    requête attend qu'il reste de la place dans toutes les fenêtres.
  - Gestion des 429 résiduels (limites de méthode / service) : sleep(Retry-After)
    puis retry automatique.
  - Exponential backoff sur les erreurs 5xx (jusqu'à MAX_RETRIES tentatives).
  - Les 404 retournent None silencieusement (ressource inconnue, non fatal).
  - Les 401/403 lèvent ApiKeyError (clé invalide/expirée) : l'appelant ne doit
    JAMAIS interpréter une clé expirée comme « aucun résultat ».
  - Session requests réutilisée pour efficacité (keep-alive TCP).
"""
import logging
import time
from collections import deque
from typing import Optional
from urllib.parse import urlparse

import requests
from requests import Response, Session
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import (
    INITIAL_BACKOFF_S,
    MAX_RETRIES,
    PLATFORM_URL,
    RANKED_SOLO_QUEUE,
    REGION_URL,
    REQUEST_TIMEOUT_S,
)

logger = logging.getLogger(__name__)

# Limites d'une clé de développement / personnelle, utilisées tant qu'aucun
# header X-App-Rate-Limit n'a été reçu pour un host.
DEFAULT_APP_LIMITS: list[tuple[int, float]] = [(20, 1.0), (100, 120.0)]

APEX_TIERS: dict[str, str] = {
    "MASTER": "masterleagues",
    "GRANDMASTER": "grandmasterleagues",
    "CHALLENGER": "challengerleagues",
}


class ApiKeyError(ValueError):
    """Clé API refusée (401/403) : invalide ou expirée."""


class ApiUnavailableError(RuntimeError):
    """API ou réseau toujours en erreur après MAX_RETRIES tentatives."""


class RateLimiter:
    """
    Fenêtres glissantes par host. `acquire()` bloque jusqu'à ce que toutes les
    fenêtres aient de la place ; `update_limits()` applique les limites réelles
    annoncées par l'API (ex : "20:1,100:120").
    """

    SAFETY_MARGIN_S: float = 0.05

    def __init__(self) -> None:
        self._limits: dict[str, list[tuple[int, float]]] = {}
        self._history: dict[str, deque[float]] = {}

    def acquire(self, host: str) -> None:
        limits = self._limits.get(host, DEFAULT_APP_LIMITS)
        history = self._history.setdefault(host, deque())
        longest = max(window for _, window in limits)

        while True:
            now = time.monotonic()
            while history and history[0] <= now - longest:
                history.popleft()

            wait = 0.0
            for count, window in limits:
                in_window = [t for t in history if t > now - window]
                if len(in_window) >= count:
                    # Attendre que la plus ancienne requête « en trop » sorte de la fenêtre
                    oldest = in_window[len(in_window) - count]
                    wait = max(wait, oldest + window - now + self.SAFETY_MARGIN_S)

            if wait <= 0:
                history.append(now)
                return
            time.sleep(wait)

    def update_limits(self, host: str, header_value: Optional[str]) -> None:
        if not header_value:
            return
        try:
            limits = [
                (int(count), float(window))
                for count, window in (part.split(":") for part in header_value.split(","))
            ]
        except ValueError:
            return
        if limits and limits != self._limits.get(host):
            logger.info("Limites de taux pour %s : %s", host, header_value)
            self._limits[host] = limits


class RiotApiClient:
    """
    Client HTTP pour l'API Riot Games.

    Gère automatiquement :
    - L'authentification via le header X-Riot-Token.
    - Le rate limiting (lecture dynamique des headers de réponse).
    - Le retry avec exponential backoff.

    Args:
        api_key: Clé API Riot Games (RGAPI-...).
        platform_url: URL de routing plateforme (ex: https://euw1.api.riotgames.com).
        region_url: URL de routing régional (ex: https://europe.api.riotgames.com).
    """

    def __init__(
        self,
        api_key: str,
        platform_url: str = PLATFORM_URL,
        region_url: str = REGION_URL,
    ) -> None:
        self.api_key: str = api_key
        self.platform_url: str = platform_url
        self.region_url: str = region_url

        self._session: Session = requests.Session()
        self._session.headers.update({"X-Riot-Token": api_key})
        # L'edge Riot coupe régulièrement des connexions sans répondre
        # (RemoteDisconnected, ~3 % des requêtes, y compris sur des connexions
        # neuves). Ces coupures sont retentées tout de suite sur une connexion
        # fraîche, au niveau transport ; 429/5xx restent gérés par _request.
        transport_retry = Retry(
            total=3, connect=3, read=3, status=0, other=0,
            backoff_factor=0.2,
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        self._session.mount("https://", HTTPAdapter(max_retries=transport_retry))
        self._rate_limiter = RateLimiter()

    def set_api_key(self, api_key: str) -> None:
        """Remplace la clé à chaud (ex : clé de dev régénérée)."""
        self.api_key = api_key
        self._session.headers.update({"X-Riot-Token": api_key})

    # ── Méthode centrale ──────────────────────────────────────────────────────

    def _request(self, url: str) -> Optional[dict]:
        """
        Effectue une requête GET avec gestion complète des erreurs.

        Stratégie de retry :
          - 429 → sleep(Retry-After, ou backoff si absent) puis retry sans limite.
          - 5xx / réseau → exponential backoff, puis ApiUnavailableError après
                  MAX_RETRIES tentatives.
          - 404 → retourne None (non fatal).
          - 401/403 → lève ApiKeyError immédiatement.

        Args:
            url: URL complète à appeler.

        Returns:
            dict JSON désérialisé, ou None si la ressource est introuvable.

        Raises:
            ApiKeyError: Si la clé API est invalide ou expirée (401/403).
            ApiUnavailableError: Si l'API ou le réseau reste en erreur.
        """
        host: str = urlparse(url).netloc
        backoff: float = INITIAL_BACKOFF_S
        server_error_attempts: int = 0
        network_error_attempts: int = 0

        while True:
            self._rate_limiter.acquire(host)
            try:
                response: Response = self._session.get(url, timeout=REQUEST_TIMEOUT_S)
                # Corps tronqué / JSON invalide traités comme une erreur réseau
                payload = response.json() if response.status_code == 200 else None
            except (requests.exceptions.RequestException, ValueError) as exc:
                # Timeout, connexion coupée, réponse tronquée (ChunkedEncodingError)...
                network_error_attempts += 1
                if network_error_attempts > MAX_RETRIES:
                    raise ApiUnavailableError(
                        f"Erreur réseau persistante après {MAX_RETRIES} tentatives : {url}"
                    )
                logger.warning(
                    "Erreur réseau (%s) sur %s. Backoff %.1fs (tentative %d/%d).",
                    type(exc).__name__, url, backoff, network_error_attempts, MAX_RETRIES,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue

            # ── Log dynamique des rate limits ─────────────────────────────
            self._log_rate_limit_headers(response)
            self._rate_limiter.update_limits(host, response.headers.get("X-App-Rate-Limit"))

            status: int = response.status_code

            if status == 200:
                return payload

            elif status == 429:
                # Retry-After (secondes) est absent pour les limites de type "service"
                limit_type: str = response.headers.get("X-Rate-Limit-Type", "service")
                retry_after_header: Optional[str] = response.headers.get("Retry-After")
                if retry_after_header is not None:
                    retry_after: float = int(retry_after_header)
                else:
                    retry_after = backoff
                    backoff = min(backoff * 2, 60.0)
                logger.warning(
                    "Rate limit atteint [type=%s]. Pause de %gs...",
                    limit_type,
                    retry_after,
                )
                time.sleep(retry_after)
                # Pas d'incrémentation de server_error_attempts : c'est normal

            elif status == 404:
                logger.debug("Ressource introuvable (404) : %s", url)
                return None

            elif status in (401, 403):
                logger.error(
                    "Accès refusé (%d) — Clé API invalide ou expirée. URL: %s", status, url
                )
                raise ApiKeyError(f"Clé API Riot invalide ou expirée (HTTP {status}).")

            elif status in (500, 502, 503, 504):
                server_error_attempts += 1
                if server_error_attempts > MAX_RETRIES:
                    raise ApiUnavailableError(
                        f"Erreur serveur {status} persistante après {MAX_RETRIES} tentatives : {url}"
                    )
                logger.warning(
                    "Erreur serveur %d. Backoff %.1fs (tentative %d/%d).",
                    status,
                    backoff,
                    server_error_attempts,
                    MAX_RETRIES,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)

            else:
                logger.error(
                    "Statut HTTP inattendu %d pour %s — abandon.", status, url
                )
                return None

    def _log_rate_limit_headers(self, response: Response) -> None:
        """
        Trace les headers de rate limit reçus pour monitoring dynamique.
        Évite tout hardcoding des limites.
        """
        count: str = response.headers.get("X-App-Rate-Limit-Count", "–")
        limit: str = response.headers.get("X-App-Rate-Limit", "–")
        method_count: str = response.headers.get("X-Method-Rate-Limit-Count", "–")
        method_limit: str = response.headers.get("X-Method-Rate-Limit", "–")
        logger.debug(
            "Rate limits — App: %s / %s | Method: %s / %s",
            count, limit, method_count, method_limit,
        )

    # ── Endpoints ─────────────────────────────────────────────────────────────

    def get_account_by_riot_id(self, game_name: str, tag_line: str) -> Optional[dict]:
        """
        Résout un Riot ID (GameName#TAG) en PUUID via Account-V1.

        Args:
            game_name: Partie avant le # (ex: "KeytedLN").
            tag_line: Partie après le # (ex: "EUW").

        Returns:
            dict contenant 'puuid', 'gameName', 'tagLine' ou None.
        """
        url = f"{self.region_url}/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}"
        logger.info("Résolution Riot ID : %s#%s", game_name, tag_line)
        return self._request(url)

    def get_match_ids_by_puuid(
        self,
        puuid: str,
        queue: int = RANKED_SOLO_QUEUE,
        count: int = 20,
        start: int = 0,
        start_time: int | None = None,
    ) -> list[str]:
        """
        Récupère une page de match IDs d'un joueur via Match-V5.

        Args:
            puuid: PUUID du joueur.
            queue: ID de la queue (420 = Ranked Solo).
            count: Nombre de matchs à récupérer (max 100).
            start: Index de départ pour la pagination.
            start_time: Timestamp Unix (secondes) — ignore les matchs plus anciens.

        Returns:
            Liste de match IDs (peut être vide si aucun match trouvé).
        """
        url = (
            f"{self.region_url}/lol/match/v5/matches/by-puuid/{puuid}/ids"
            f"?queue={queue}&count={count}&start={start}"
        )
        if start_time is not None:
            url += f"&startTime={start_time}"
        result: Optional[list] = self._request(url)
        return result if isinstance(result, list) else []

    def get_all_match_ids_by_puuid(
        self,
        puuid: str,
        queue: int = RANKED_SOLO_QUEUE,
        page_size: int = 100,
    ) -> list[str]:
        """
        Récupère TOUS les match IDs d'un joueur en paginant automatiquement.

        L'API Riot limite chaque requête à 100 résultats maximum.
        Cette méthode boucle avec un offset croissant jusqu'à ce qu'une
        page vide soit retournée, signalant la fin de l'historique.

        Args:
            puuid: PUUID du joueur.
            queue: ID de la queue (420 = Ranked Solo).
            page_size: Taille de chaque page (max 100, défaut 100).

        Returns:
            Liste complète et ordonnée de tous les match IDs du joueur.
        """
        all_ids: list[str] = []
        start: int = 0

        while True:
            page: list[str] = self.get_match_ids_by_puuid(
                puuid=puuid,
                queue=queue,
                count=page_size,
                start=start,
            )
            if not page:
                break  # Page vide = fin de l'historique

            all_ids.extend(page)
            logger.info(
                "Page %d fetched : %d match IDs (total: %d)",
                start // page_size + 1,
                len(page),
                len(all_ids),
            )

            if len(page) < page_size:
                break  # Dernière page partielle = fin de l'historique

            start += page_size

        return all_ids

    def get_match(self, match_id: str) -> Optional[dict]:
        """
        Récupère les données complètes d'un match via Match-V5.

        Args:
            match_id: Identifiant du match (ex: "EUW1_7123456789").

        Returns:
            dict complet du match ou None si introuvable.
        """
        url = f"{self.region_url}/lol/match/v5/matches/{match_id}"
        return self._request(url)

    def get_summoner_by_puuid(self, puuid: str) -> Optional[dict]:
        """
        Récupère les infos d'un invocateur par PUUID via Summoner-V4.

        Note: Utilise le routing plateforme (euw1), pas régional (europe).

        Args:
            puuid: PUUID Riot du joueur.

        Returns:
            dict Summoner ou None si introuvable.
        """
        url = f"{self.platform_url}/lol/summoner/v4/summoners/by-puuid/{puuid}"
        return self._request(url)

    def get_league_entries(
        self,
        tier: str,
        division: str,
        page: int = 1,
        queue: str = "RANKED_SOLO_5x5",
    ) -> list[dict]:
        """
        Une page (~205 joueurs) d'une division classée via League-V4.
        Pour MASTER / GRANDMASTER / CHALLENGER, utiliser get_apex_league().

        Returns:
            Liste de LeagueEntryDTO (puuid, tier, rank, leaguePoints...).
            Vide au-delà de la dernière page.
        """
        url = (
            f"{self.platform_url}/lol/league/v4/entries/{queue}/{tier}/{division}"
            f"?page={page}"
        )
        result: Optional[list] = self._request(url)
        return result if isinstance(result, list) else []

    def get_apex_league(self, tier: str, queue: str = "RANKED_SOLO_5x5") -> list[dict]:
        """
        Tous les joueurs d'un tier apex (MASTER / GRANDMASTER / CHALLENGER).

        Returns:
            Liste de LeagueItemDTO (puuid, rank, leaguePoints...).
        """
        url = f"{self.platform_url}/lol/league/v4/{APEX_TIERS[tier]}/by-queue/{queue}"
        result: Optional[dict] = self._request(url)
        return result.get("entries", []) if isinstance(result, dict) else []

    def get_champion_masteries(self, puuid: str) -> list[dict]:
        """
        Maîtrise de tous les champions d'un joueur via Champion-Mastery-V4.

        Returns:
            Liste de ChampionMasteryDTO (championId, championPoints, championLevel, lastPlayTime...).
        """
        url = f"{self.platform_url}/lol/champion-mastery/v4/champion-masteries/by-puuid/{puuid}"
        result: Optional[list] = self._request(url)
        return result if isinstance(result, list) else []
