"""
tests/test_client.py — Tests unitaires du client API Riot avec mocks.

Couverture :
  - Réponse 200 → retourne le JSON désérialisé.
  - Réponse 429 → sleep(Retry-After), puis retry et succès.
  - Réponse 404 → retourne None silencieusement.
  - Réponse 401/403 → lève ApiKeyError immédiatement.
  - Réponse 503 → backoff puis ApiUnavailableError après MAX_RETRIES.
  - Timeout réseau → backoff puis retry.
  - RateLimiter → attend quand une fenêtre est pleine, apprend les limites des headers.
  - get_account_by_riot_id → construit la bonne URL.
  - get_match_ids_by_puuid → retourne [] si réponse None.
"""
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

# Assurer que le package racine est dans sys.path pour les imports relatifs
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests

from api.client import ApiKeyError, ApiUnavailableError, RateLimiter, RiotApiClient


def make_response(status_code: int, json_data=None, headers: dict | None = None) -> MagicMock:
    """Helper : crée un mock de requests.Response."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_data or {}
    mock_resp.headers = headers or {}
    return mock_resp


class TestRiotApiClientRequest(unittest.TestCase):
    """Tests de la méthode centrale _request()."""

    def setUp(self) -> None:
        self.client = RiotApiClient(api_key="RGAPI-test-key")
        self.url = "https://europe.api.riotgames.com/test/endpoint"

    @patch("api.client.requests.Session.get")
    def test_200_returns_json(self, mock_get: MagicMock) -> None:
        """Un 200 doit retourner le contenu JSON désérialisé."""
        expected = {"key": "value"}
        mock_get.return_value = make_response(200, expected)

        result = self.client._request(self.url)

        self.assertEqual(result, expected)
        mock_get.assert_called_once()

    @patch("api.client.time.sleep")
    @patch("api.client.requests.Session.get")
    def test_429_sleeps_retry_after_then_retries(
        self, mock_get: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """
        Un 429 doit appeler time.sleep avec la valeur du header Retry-After,
        puis retenter et retourner le JSON si le retry réussit.
        """
        retry_after = 3
        rate_limit_resp = make_response(
            429, headers={"Retry-After": str(retry_after), "X-Rate-Limit-Type": "application"}
        )
        success_resp = make_response(200, {"result": "ok"})
        mock_get.side_effect = [rate_limit_resp, success_resp]

        result = self.client._request(self.url)

        self.assertEqual(result, {"result": "ok"})
        mock_sleep.assert_called_once_with(retry_after)
        self.assertEqual(mock_get.call_count, 2)

    @patch("api.client.requests.Session.get")
    def test_404_returns_none(self, mock_get: MagicMock) -> None:
        """Un 404 doit retourner None silencieusement (non fatal)."""
        mock_get.return_value = make_response(404)

        result = self.client._request(self.url)

        self.assertIsNone(result)

    @patch("api.client.requests.Session.get")
    def test_403_raises_api_key_error(self, mock_get: MagicMock) -> None:
        """Un 403 doit lever ApiKeyError immédiatement (clé invalide)."""
        mock_get.return_value = make_response(403)

        with self.assertRaises(ApiKeyError) as ctx:
            self.client._request(self.url)

        self.assertIn("403", str(ctx.exception))

    @patch("api.client.requests.Session.get")
    def test_401_expired_key_raises_api_key_error(self, mock_get: MagicMock) -> None:
        """Une clé de dev expirée renvoie 401 « Unknown apikey » : ce n'est PAS un résultat vide."""
        mock_get.return_value = make_response(401)

        with self.assertRaises(ApiKeyError):
            self.client._request(self.url)

    @patch("api.client.time.sleep")
    @patch("api.client.requests.Session.get")
    def test_503_backoff_raises_after_max_retries(
        self, mock_get: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """
        Des erreurs 503 répétées doivent déclencher le backoff exponentiel
        puis lever ApiUnavailableError après MAX_RETRIES tentatives.
        """
        from config import MAX_RETRIES
        mock_get.return_value = make_response(503)

        with self.assertRaises(ApiUnavailableError):
            self.client._request(self.url)

        # Vérifie que sleep a été appelé MAX_RETRIES fois
        self.assertEqual(mock_sleep.call_count, MAX_RETRIES)
        # Vérifie le backoff exponentiel (1.0, 2.0, 4.0...)
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        for i in range(1, len(sleep_calls)):
            self.assertAlmostEqual(sleep_calls[i], sleep_calls[i - 1] * 2, places=5)

    @patch("api.client.time.sleep")
    @patch("api.client.requests.Session.get")
    def test_timeout_triggers_backoff_and_retry(
        self, mock_get: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """Un Timeout réseau doit déclencher le backoff et retenter."""
        success_resp = make_response(200, {"data": "ok"})
        mock_get.side_effect = [
            requests.exceptions.Timeout(),
            success_resp,
        ]

        result = self.client._request(self.url)

        self.assertEqual(result, {"data": "ok"})
        mock_sleep.assert_called_once()  # Un seul sleep (avant le retry réussi)


class TestRiotApiClientTruncatedResponse(unittest.TestCase):

    @patch("api.client.time.sleep")
    @patch("api.client.requests.Session.get")
    def test_truncated_response_is_retried(self, mock_get: MagicMock, mock_sleep: MagicMock) -> None:
        """Une réponse coupée en plein transfert (vu en conditions réelles) doit être retentée."""
        mock_get.side_effect = [
            requests.exceptions.ChunkedEncodingError("Response ended prematurely"),
            make_response(200, {"data": "ok"}),
        ]

        result = RiotApiClient(api_key="RGAPI-test-key")._request("https://euw1.api.riotgames.com/x")

        self.assertEqual(result, {"data": "ok"})
        mock_sleep.assert_called_once()


class TestTransportRetries(unittest.TestCase):

    def test_dropped_connections_are_retried_by_transport(self) -> None:
        """Les coupures « Remote end closed connection » doivent être retentées par urllib3."""
        client = RiotApiClient(api_key="RGAPI-test-key")
        retry = client._session.get_adapter("https://europe.api.riotgames.com").max_retries

        self.assertGreaterEqual(retry.read, 1)     # RemoteDisconnected = erreur de lecture
        self.assertGreaterEqual(retry.connect, 1)
        self.assertTrue(retry.is_retry("GET", 503) is False)  # 5xx/429 restent gérés par _request
        self.assertIn("GET", retry.allowed_methods)


class TestRateLimiter(unittest.TestCase):
    """Tests du rate limiting proactif."""

    @patch("api.client.time.sleep")
    @patch("api.client.time.monotonic")
    def test_waits_when_window_is_full(self, mock_monotonic: MagicMock, mock_sleep: MagicMock) -> None:
        """La 3e requête dans une fenêtre 2:10 doit attendre ~10s."""
        clock = [100.0]
        mock_monotonic.side_effect = lambda: clock[0]
        mock_sleep.side_effect = lambda s: clock.__setitem__(0, clock[0] + s)

        limiter = RateLimiter()
        limiter.update_limits("host", "2:10")
        limiter.acquire("host")
        limiter.acquire("host")
        mock_sleep.assert_not_called()

        limiter.acquire("host")
        mock_sleep.assert_called_once()
        self.assertAlmostEqual(mock_sleep.call_args.args[0], 10.0, delta=0.1)

    def test_update_limits_parses_header(self) -> None:
        limiter = RateLimiter()
        limiter.update_limits("host", "20:1,100:120")
        self.assertEqual(limiter._limits["host"], [(20, 1.0), (100, 120.0)])

    def test_hosts_are_limited_separately(self) -> None:
        limiter = RateLimiter()
        limiter.update_limits("europe", "1:100")
        limiter.acquire("europe")
        with patch("api.client.time.sleep") as mock_sleep:
            limiter.acquire("euw1")  # Quota distinct : pas d'attente
            mock_sleep.assert_not_called()


class TestRiotApiClientEndpoints(unittest.TestCase):
    """Tests des méthodes d'endpoint spécifiques."""

    def setUp(self) -> None:
        self.client = RiotApiClient(
            api_key="RGAPI-test-key",
            platform_url="https://euw1.api.riotgames.com",
            region_url="https://europe.api.riotgames.com",
        )

    @patch.object(RiotApiClient, "_request")
    def test_get_account_by_riot_id_builds_correct_url(
        self, mock_request: MagicMock
    ) -> None:
        """get_account_by_riot_id doit appeler _request avec l'URL Account-V1 correcte."""
        mock_request.return_value = {
            "puuid": "test-puuid-123",
            "gameName": "KeytedLN",
            "tagLine": "EUW",
        }

        result = self.client.get_account_by_riot_id("KeytedLN", "EUW")

        expected_url = (
            "https://europe.api.riotgames.com"
            "/riot/account/v1/accounts/by-riot-id/KeytedLN/EUW"
        )
        mock_request.assert_called_once_with(expected_url)
        self.assertEqual(result["puuid"], "test-puuid-123")

    @patch.object(RiotApiClient, "_request")
    def test_get_match_ids_returns_empty_list_on_none(
        self, mock_request: MagicMock
    ) -> None:
        """get_match_ids_by_puuid doit retourner [] si _request retourne None (404)."""
        mock_request.return_value = None

        result = self.client.get_match_ids_by_puuid("fake-puuid")

        self.assertEqual(result, [])

    @patch.object(RiotApiClient, "_request")
    def test_get_match_ids_includes_queue_param(
        self, mock_request: MagicMock
    ) -> None:
        """L'URL de get_match_ids_by_puuid doit inclure queue=420."""
        mock_request.return_value = ["EUW1_111", "EUW1_222"]
        puuid = "test-puuid-abc"

        self.client.get_match_ids_by_puuid(puuid, queue=420, count=5)

        called_url: str = mock_request.call_args.args[0]
        self.assertIn("queue=420", called_url)
        self.assertIn("count=5", called_url)
        self.assertIn(puuid, called_url)

    @patch.object(RiotApiClient, "_request")
    def test_get_match_uses_region_url(self, mock_request: MagicMock) -> None:
        """get_match doit utiliser le routing régional (europe), pas plateforme (euw1)."""
        mock_request.return_value = {"metadata": {}, "info": {}}
        match_id = "EUW1_9999999999"

        self.client.get_match(match_id)

        called_url: str = mock_request.call_args.args[0]
        self.assertIn("europe.api.riotgames.com", called_url)
        self.assertIn(match_id, called_url)

    @patch.object(RiotApiClient, "_request")
    def test_get_summoner_uses_platform_url(self, mock_request: MagicMock) -> None:
        """get_summoner_by_puuid doit utiliser le routing plateforme (euw1)."""
        mock_request.return_value = {"id": "summoner-id", "puuid": "test-puuid"}
        puuid = "test-puuid"

        self.client.get_summoner_by_puuid(puuid)

        called_url: str = mock_request.call_args.args[0]
        self.assertIn("euw1.api.riotgames.com", called_url)


if __name__ == "__main__":
    unittest.main(verbosity=2)
