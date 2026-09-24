"""
tests/test_client.py — Tests unitaires du client API Riot avec mocks.

Couverture :
  - Réponse 200 → retourne le JSON désérialisé.
  - Réponse 429 → sleep(Retry-After), puis retry et succès.
  - Réponse 404 → retourne None silencieusement.
  - Réponse 403 → lève ValueError immédiatement.
  - Réponse 503 → backoff puis retourne None après MAX_RETRIES.
  - Timeout réseau → backoff puis retry.
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

from api.client import RiotApiClient


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

    @patch("api_client.client.requests.Session.get")
    def test_200_returns_json(self, mock_get: MagicMock) -> None:
        """Un 200 doit retourner le contenu JSON désérialisé."""
        expected = {"key": "value"}
        mock_get.return_value = make_response(200, expected)

        result = self.client._request(self.url)

        self.assertEqual(result, expected)
        mock_get.assert_called_once()

    @patch("api_client.client.time.sleep")
    @patch("api_client.client.requests.Session.get")
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

    @patch("api_client.client.requests.Session.get")
    def test_404_returns_none(self, mock_get: MagicMock) -> None:
        """Un 404 doit retourner None silencieusement (non fatal)."""
        mock_get.return_value = make_response(404)

        result = self.client._request(self.url)

        self.assertIsNone(result)

    @patch("api_client.client.requests.Session.get")
    def test_403_raises_value_error(self, mock_get: MagicMock) -> None:
        """Un 403 doit lever ValueError immédiatement (clé invalide)."""
        mock_get.return_value = make_response(403)

        with self.assertRaises(ValueError) as ctx:
            self.client._request(self.url)

        self.assertIn("403", str(ctx.exception))

    @patch("api_client.client.time.sleep")
    @patch("api_client.client.requests.Session.get")
    def test_503_backoff_returns_none_after_max_retries(
        self, mock_get: MagicMock, mock_sleep: MagicMock
    ) -> None:
        """
        Des erreurs 503 répétées doivent déclencher le backoff exponentiel
        et retourner None après MAX_RETRIES tentatives.
        """
        from config import MAX_RETRIES
        mock_get.return_value = make_response(503)

        result = self.client._request(self.url)

        self.assertIsNone(result)
        # Vérifie que sleep a été appelé MAX_RETRIES fois
        self.assertEqual(mock_sleep.call_count, MAX_RETRIES)
        # Vérifie le backoff exponentiel (1.0, 2.0, 4.0...)
        sleep_calls = [c.args[0] for c in mock_sleep.call_args_list]
        for i in range(1, len(sleep_calls)):
            self.assertAlmostEqual(sleep_calls[i], sleep_calls[i - 1] * 2, places=5)

    @patch("api_client.client.time.sleep")
    @patch("api_client.client.requests.Session.get")
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
