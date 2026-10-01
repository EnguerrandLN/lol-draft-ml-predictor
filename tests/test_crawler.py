"""
tests/test_crawler.py — Tests du LadderCrawler sur une base SQLite temporaire.

Couverture :
  - Échantillonnage du ladder → joueurs enfilés avec leur tier.
  - Matchs stockés avec source_tier / source_division / ended_early.
  - Clé expirée sur l'historique d'un joueur → le joueur n'est PAS marqué
    'done', le crawler attend une nouvelle clé puis reprend ce joueur.
  - API indisponible → même garantie.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.client import ApiKeyError, ApiUnavailableError
from api.crawler import LadderCrawler
from db.schema import init_db


def make_match(match_id: str, early: bool = False) -> dict:
    participants = [
        {
            "puuid": f"{match_id}-p{i}",
            "teamId": 100 if i < 5 else 200,
            "teamPosition": ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"][i % 5],
            "championId": i + 1,
            "win": i < 5,
            "gameEndedInEarlySurrender": early,
            "damageSelfMitigated": 1000 + i,
            "timeCCingOthers": 20 + i,
            "totalTimeCCDealt": 100 + i,
            "totalHeal": 500 + i,
            "totalHealsOnTeammates": 50 + i,
            "totalDamageShieldedOnTeammates": 30 + i,
        }
        for i in range(10)
    ]
    return {
        "metadata": {"matchId": match_id},
        "info": {
            "queueId": 420,
            "gameVersion": "16.19.1",
            "gameDuration": 200 if early else 1800,
            "participants": participants,
            "teams": [{"teamId": 100, "win": True, "bans": []}, {"teamId": 200, "win": False, "bans": []}],
        },
    }


class TestLadderCrawler(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        with patch("db.schema.DB_PATH", Path(self.tmp.name) / "test.db"):
            self.conn = init_db()

        self.client = MagicMock()
        self.client.api_key = "RGAPI-old"
        self.client.get_league_entries.side_effect = (
            lambda tier, division, page: [{"puuid": "gold-player", "rank": "II", "leaguePoints": 50}]
            if (division, page) == ("II", 1) else []
        )
        self.client.get_apex_league.return_value = [{"puuid": "chall-player", "rank": "I", "leaguePoints": 900}]
        self.client.get_match_ids_by_puuid.side_effect = lambda puuid, **kw: [f"EUW1_{puuid}"]
        self.client.get_match.side_effect = lambda match_id: make_match(
            match_id, early=match_id.startswith("EUW1_chall")
        )

        self.reload_key = MagicMock(return_value="RGAPI-new")
        self.crawler = LadderCrawler(
            conn=self.conn,
            client=self.client,
            tiers=["GOLD", "CHALLENGER"],
            pages_per_division=1,
            days_back=30,
            matches_per_player=20,
            reload_api_key=self.reload_key,
        )

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_matches_are_tagged_with_source_tier(self) -> None:
        self.crawler.run(max_matches=2)

        rows = dict(
            (r["match_id"], (r["source_tier"], r["source_division"], r["ended_early"]))
            for r in self.conn.execute("SELECT match_id, source_tier, source_division, ended_early FROM matches")
        )
        self.assertEqual(rows["EUW1_gold-player"], ("GOLD", "II", 0))
        self.assertEqual(rows["EUW1_chall-player"], ("CHALLENGER", "I", 1))
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM participants").fetchone()[0], 20)
        support_stats = self.conn.execute(
            """SELECT damage_self_mitigated, time_ccing_others, total_time_cc_dealt, total_heal,
                      total_heals_on_teammates, total_damage_shielded_on_teammates
               FROM participants WHERE puuid = 'EUW1_gold-player-p3'"""
        ).fetchone()
        self.assertEqual(tuple(support_stats), (1003, 23, 103, 503, 53, 33))

    @patch("api.crawler.time.sleep")
    def test_expired_key_does_not_burn_player(self, mock_sleep: MagicMock) -> None:
        """Clé refusée → pause, rechargement depuis .env, puis le même joueur est retraité."""
        calls = {"n": 0}

        def matchlist(puuid, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise ApiKeyError("expired")
            return [f"EUW1_{puuid}"]

        self.client.get_match_ids_by_puuid.side_effect = matchlist

        self.crawler.run(max_matches=2)

        self.client.set_api_key.assert_called_once_with("RGAPI-new")
        self.assertEqual(self.conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0], 2)

    @patch("api.crawler.time.sleep")
    def test_unavailable_api_keeps_player_pending(self, mock_sleep: MagicMock) -> None:
        self.client.get_match.side_effect = ApiUnavailableError("down")
        mock_sleep.side_effect = KeyboardInterrupt  # Arrêter le crawler pendant la pause

        self.crawler.run()

        statuses = [r[0] for r in self.conn.execute("SELECT status FROM ladder_players")]
        self.assertEqual(statuses, ["pending", "pending"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
