"""
api_client/client.py — Client HTTP sécurisé pour l'API Riot Games.

Fonctionnalités clés :
  - Rate limiting 100% dynamique : lecture de X-App-Rate-Limit-Count et
    Retry-After à chaque réponse (aucune valeur codée en dur).
  - Gestion des 429 : time.sleep(Retry-After) puis retry automatique.
  - Exponential backoff sur les erreurs 5xx (jusqu'à MAX_RETRIES tentatives).
  - Les 404 retournent None silencieusement (ressource inconnue, non fatal).
  - Les 403 lèvent une ValueError immédiate (clé invalide/expirée).
  - Session requests réutilisée pour efficacité (keep-alive TCP).
"""
import logging
import time
from typing import Optional

import requests
from requests import Response, Session

from config import (
    INITIAL_BACKOFF_S,
    MAX_RETRIES,
    PLATFORM_URL,
    RANKED_SOLO_QUEUE,
    REGION_URL,
    REQUEST_TIMEOUT_S,
)

logger = logging.getLogger(__name__)


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

    # ── Méthode centrale ──────────────────────────────────────────────────────

    def _request(self, url: str) -> Optional[dict]:
        """
        Effectue une requête GET avec gestion complète des erreurs.

        Stratégie de retry :
          - 429 → sleep(Retry-After depuis le header) puis retry sans limite
                  (la limite de taux est gérée par l'API elle-même).
          - 5xx → exponential backoff, jusqu'à MAX_RETRIES tentatives.
          - 404 → retourne None (non fatal).
          - 403 → lève ValueError immédiatement.

        Args:
            url: URL complète à appeler.

        Returns:
            dict JSON désérialisé, ou None si la ressource est introuvable.

        Raises:
            ValueError: Si la clé API est invalide ou expirée (403).
        """
        backoff: float = INITIAL_BACKOFF_S
        server_error_attempts: int = 0
        network_error_attempts: int = 0

        while True:
            try:
                response: Response = self._session.get(url, timeout=REQUEST_TIMEOUT_S)
            except requests.exceptions.Timeout:
                network_error_attempts += 1
                if network_error_attempts > MAX_RETRIES:
                    logger.error(
                        "Timeout persistant après %d tentatives, abandon : %s",
                        MAX_RETRIES, url,
                    )
                    return None
                logger.warning(
                    "Timeout sur %s. Backoff %.1fs (tentative %d/%d).",
                    url, backoff, network_error_attempts, MAX_RETRIES,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue
            except requests.exceptions.ConnectionError as exc:
                network_error_attempts += 1
                if network_error_attempts > MAX_RETRIES:
                    logger.error(
                        "Erreur réseau persistante après %d tentatives, abandon : %s",
                        MAX_RETRIES, url,
                    )
                    return None
                logger.warning(
                    "Erreur réseau : %s. Backoff %.1fs (tentative %d/%d).",
                    exc, backoff, network_error_attempts, MAX_RETRIES,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
                continue

            # ── Log dynamique des rate limits ─────────────────────────────
            self._log_rate_limit_headers(response)

            status: int = response.status_code

            if status == 200:
                return response.json()

            elif status == 429:
                # Le header Retry-After est en secondes (entier)
                retry_after: int = int(response.headers.get("Retry-After", "1"))
                limit_type: str = response.headers.get("X-Rate-Limit-Type", "unknown")
                logger.warning(
                    "Rate limit atteint [type=%s]. Pause de %ds...",
                    limit_type,
                    retry_after,
                )
                time.sleep(retry_after)
                # Pas d'incrémentation de server_error_attempts : c'est normal

            elif status == 404:
                logger.debug("Ressource introuvable (404) : %s", url)
                return None

            elif status == 403:
                logger.error(
                    "Accès refusé (403) — Clé API invalide ou expirée. URL: %s", url
                )
                raise ValueError("Clé API Riot invalide ou expirée (HTTP 403).")

            elif status in (500, 502, 503, 504):
                server_error_attempts += 1
                if server_error_attempts > MAX_RETRIES:
                    logger.error(
                        "Erreur serveur %d persistante après %d tentatives : %s",
                        status,
                        MAX_RETRIES,
                        url,
                    )
                    return None
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
