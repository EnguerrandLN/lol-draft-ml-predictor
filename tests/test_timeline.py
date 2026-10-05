"""
tests/test_timeline.py — Timelines Match-V5 : extraction, stockage, reprise.

  - Les images retenues sont celles des minutes demandées, par joueur.
  - Une minute au-delà de la fin de la partie est ignorée.
  - Une timeline introuvable est mémorisée : le match n'est plus redemandé.
  - Les images se joignent aux participants (équipe, poste) par le puuid.
  - Le client appelle l'endpoint timeline sur le routing régional.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.client import RiotApiClient
from db.repository import (
    get_matches_without_timeline, get_timeline_count, timeline_frames, upsert_match, upsert_participant,
    upsert_timeline,
)
from db.schema import init_db
from tests.test_crawler import make_match


def make_timeline(match_id: str, n_frames: int) -> dict:
    """Or total = 100 × minute + participantId, pour retrouver chaque valeur."""
    frames = [
        {"timestamp": m * 60_000, "participantFrames": {
            str(pid): {"participantId": pid, "totalGold": 100 * m + pid, "xp": 50 * m, "level": 1 + m // 3,
                       "minionsKilled": 7 * m, "jungleMinionsKilled": pid % 2 * m,
                       "damageStats": {"totalDamageDoneToChampions": 30 * m}}
            for pid in range(1, 11)}}
        for m in range(n_frames)
    ]
    participants = [{"participantId": i + 1, "puuid": f"{match_id}-p{i}"} for i in range(10)]
    return {"metadata": {"matchId": match_id}, "info": {"frameInterval": 60_000, "frames": frames,
                                                        "participants": participants}}


class TestTimelineFrames(unittest.TestCase):

    def test_requested_minutes_per_player(self) -> None:
        rows = timeline_frames(make_timeline("M", 25), (10, 15))
        self.assertEqual(len(rows), 20)
        by_key = {(r[0], r[1]): r for r in rows}
        self.assertEqual(by_key[("M-p0", 15)][2], 1501)        # Or : participantId 1 à 15 min
        self.assertEqual(by_key[("M-p9", 10)][7], 300)         # Dégâts aux champions à 10 min

    def test_minutes_after_game_end_are_skipped(self) -> None:
        rows = timeline_frames(make_timeline("M", 18), (10, 15, 20))
        self.assertEqual(sorted({r[1] for r in rows}), [10, 15])

    def test_puuids_from_metadata_when_info_has_none(self) -> None:
        timeline = make_timeline("M", 12)
        del timeline["info"]["participants"]
        timeline["metadata"]["participants"] = [f"M-p{i}" for i in range(10)]
        self.assertEqual({r[0] for r in timeline_frames(timeline, (10,))}, {f"M-p{i}" for i in range(10)})


class TestTimelineStorage(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        with patch("db.schema.DB_PATH", Path(self.tmp.name) / "test.db"):
            self.conn = init_db()
        for match_id in ("EUW1_1", "EUW1_2"):
            match = make_match(match_id)
            match["info"]["gameCreation"] = 1_000_000
            upsert_match(self.conn, match)
            for p in match["info"]["participants"]:
                upsert_participant(self.conn, match_id, p)
        self.conn.commit()

    def tearDown(self) -> None:
        self.conn.close()
        self.tmp.cleanup()

    def test_resume_skips_fetched_and_missing_timelines(self) -> None:
        self.assertEqual(sorted(get_matches_without_timeline(self.conn, 0, 420, 10)), ["EUW1_1", "EUW1_2"])
        self.assertEqual(upsert_timeline(self.conn, "EUW1_1", make_timeline("EUW1_1", 30), (10, 15, 20)), 30)
        upsert_timeline(self.conn, "EUW1_2", None, (10, 15, 20))   # Introuvable (404)
        self.assertEqual(get_matches_without_timeline(self.conn, 0, 420, 10), [])
        self.assertEqual(get_timeline_count(self.conn), 1)

    def test_frames_join_participants(self) -> None:
        upsert_timeline(self.conn, "EUW1_1", make_timeline("EUW1_1", 16), (15,))
        rows = self.conn.execute(
            """SELECT p.team_id, p.position, f.total_gold FROM timeline_frames f
               JOIN participants p ON p.match_id = f.match_id AND p.puuid = f.puuid
               WHERE f.minute = 15 ORDER BY f.total_gold""").fetchall()
        self.assertEqual(len(rows), 10)
        self.assertEqual(tuple(rows[0]), (100, "TOP", 1501))


class TestTimelineEndpoint(unittest.TestCase):

    @patch.object(RiotApiClient, "_request")
    def test_timeline_url(self, mock_request: MagicMock) -> None:
        RiotApiClient(api_key="RGAPI-test").get_match_timeline("EUW1_42")
        url = mock_request.call_args[0][0]
        self.assertTrue(url.endswith("/lol/match/v5/matches/EUW1_42/timeline"))
        self.assertIn("europe", url)


if __name__ == "__main__":
    unittest.main(verbosity=2)
