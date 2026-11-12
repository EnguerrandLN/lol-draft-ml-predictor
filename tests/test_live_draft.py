"""
tests/test_live_draft.py — Lecture de la sélection de champions du client du jeu.

  - Identifiants : lockfile et ligne de commande du client.
  - Session du client → draft : seuls les picks VERROUILLÉS comptent (un champion
    survolé est ignoré), bans, côté, rôle, picks adverses restants.
  - Attribution des rôles adverses : la plus probable globalement, un champion
    par rôle, y compris pour un champion jouable à plusieurs rôles.
  - Client fermé ou hors sélection : statut explicite, pas d'exception.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from api import lcu
from ml.live_draft import assign_roles, parse_champ_select, resolve

# Parties par rôle : 1 = top pur, 2 = jungle pur, 3 = mid (un peu top), 4 = ADC pur,
# 5 = support pur, 6 = jouable mid ET support, 7 = mid pur
GAMES = {
    "TOP": {"1": 900, "3": 150},
    "JUNGLE": {"2": 800},
    "MIDDLE": {"3": 850, "6": 500, "7": 900},
    "BOTTOM": {"4": 1000},
    "UTILITY": {"5": 700, "6": 450},
}


def action(cell: int, champ: int, kind: str, completed: bool = True) -> dict:
    return {"actorCellId": cell, "championId": champ, "type": kind, "completed": completed}


def make_session() -> dict:
    """Joueur côté rouge (cellule 7), jungle. Deux alliés et trois adversaires ont verrouillé."""
    roles = ["top", "jungle", "middle", "bottom", "utility"]
    return {
        "localPlayerCellId": 6,
        "myTeam": [{"cellId": 5 + i, "team": 2, "assignedPosition": roles[i], "championId": 0} for i in range(5)],
        "theirTeam": [{"cellId": i, "team": 1, "assignedPosition": "", "championId": 0} for i in range(5)],
        "actions": [
            [action(c, 100 + c, "ban") for c in range(10)],
            [action(0, 4, "pick")],
            [action(5, 1, "pick"), action(6, 0, "pick", completed=False)],          # Le joueur n'a pas encore choisi
            [action(1, 5, "pick"), action(2, 6, "pick")],
            [action(7, 3, "pick"), action(8, 44, "pick", completed=False)],         # 44 : seulement survolé
            [action(3, 0, "pick", completed=False), action(4, 0, "pick", completed=False)],
        ],
    }


class TestCredentials(unittest.TestCase):

    def test_lockfile(self) -> None:
        creds = lcu.parse_lockfile("LeagueClient:1234:51234:s3cret-Tok_en:https\n")
        self.assertEqual((creds.port, creds.token, creds.base_url), (51234, "s3cret-Tok_en", "https://127.0.0.1:51234"))
        self.assertIsNone(lcu.parse_lockfile("n'importe quoi"))

    def test_command_line(self) -> None:
        cmd = '"C:/Riot Games/League of Legends/LeagueClientUx.exe" --remoting-auth-token=abc-DEF_1 --app-port=60123 --x'
        creds = lcu.parse_command_line(cmd)
        self.assertEqual((creds.port, creds.token), (60123, "abc-DEF_1"))
        self.assertIsNone(lcu.parse_command_line(""))


class TestFindCredentials(unittest.TestCase):

    def test_process_lookup_is_throttled(self) -> None:
        """Sans lockfile, la liste des processus n'est relue que toutes les 15 s (l'app interroge toutes les 2 s)."""
        found = lcu.Credentials(port=2, token="t")
        with patch.object(lcu, "DEFAULT_LOCKFILES", ()), patch.dict(lcu.os.environ, {}, clear=False),                 patch.object(lcu, "_process_probe", [float("-inf"), None]),                 patch.object(lcu, "_credentials_from_process", return_value=found) as probe:
            lcu.os.environ.pop(lcu.LOCKFILE_ENV, None)
            self.assertEqual([lcu.find_credentials(), lcu.find_credentials()], [found, found])
            self.assertEqual(probe.call_count, 1)

    def test_lockfile_path_from_environment(self) -> None:
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lockfile"
            path.write_text("LeagueClient:1:4242:tok:https", encoding="utf-8")
            with patch.dict(lcu.os.environ, {lcu.LOCKFILE_ENV: str(path)}):
                self.assertEqual(lcu.find_credentials().port, 4242)


class TestSessionStatus(unittest.TestCase):

    CREDS = lcu.Credentials(port=1, token="t")

    def test_no_client(self) -> None:
        with patch.object(lcu, "find_credentials", return_value=None):
            self.assertEqual(lcu.champ_select_session(), (lcu.NO_CLIENT, None))
        with patch.object(lcu.requests, "get", side_effect=requests.exceptions.ConnectionError):
            self.assertEqual(lcu.champ_select_session(self.CREDS), (lcu.NO_CLIENT, None))

    def test_not_in_champ_select(self) -> None:
        with patch.object(lcu.requests, "get", return_value=MagicMock(status_code=404)):
            self.assertEqual(lcu.champ_select_session(self.CREDS), (lcu.NO_CHAMP_SELECT, None))

    def test_session_is_returned(self) -> None:
        response = MagicMock(status_code=200)
        response.json.return_value = make_session()
        with patch.object(lcu.requests, "get", return_value=response) as get:
            status, session = lcu.champ_select_session(self.CREDS)
        self.assertEqual(status, lcu.OK)
        self.assertEqual(session["localPlayerCellId"], 6)
        self.assertTrue(get.call_args[0][0].startswith("https://127.0.0.1:1/"))   # Jamais ailleurs que sur la machine


class TestParse(unittest.TestCase):

    def test_only_locked_picks_count(self) -> None:
        live = parse_champ_select(make_session())
        self.assertEqual((live.side, live.role, live.my_champion), ("red", "JUNGLE", None))
        self.assertEqual(live.ally, {"TOP": 1, "MIDDLE": 3})          # Le survol (44) du bot est ignoré
        self.assertEqual(live.enemy, [4, 5, 6])
        self.assertEqual(live.bans, {100 + c for c in range(10)})
        self.assertEqual(live.enemy_picks_left, 2)

    def test_signature_changes_with_the_draft(self) -> None:
        session = make_session()
        before = parse_champ_select(session).signature()
        session["actions"][4][1]["completed"] = True                  # Le bot verrouille
        self.assertNotEqual(parse_champ_select(session).signature(), before)

    def test_empty_session(self) -> None:
        live = parse_champ_select({})
        self.assertEqual((live.ally, live.enemy, live.bans, live.role), ({}, [], set(), None))


class TestAssignRoles(unittest.TestCase):

    def test_obvious_roles(self) -> None:
        self.assertEqual(assign_roles([4, 5, 2], GAMES), {"BOTTOM": 4, "UTILITY": 5, "JUNGLE": 2})

    def test_flex_pick_takes_the_remaining_role(self) -> None:
        """6 est surtout joué mid, mais 7 ne se joue que mid : 6 passe support."""
        self.assertEqual(assign_roles([6, 7], GAMES), {"UTILITY": 6, "MIDDLE": 7})
        self.assertEqual(assign_roles([6], GAMES), {"MIDDLE": 6})

    def test_full_team_and_unknown_champion(self) -> None:
        full = assign_roles([7, 999, 4, 2, 5], GAMES)                 # 999 : inconnu du modèle → le rôle restant
        self.assertEqual(full, {"MIDDLE": 7, "TOP": 999, "BOTTOM": 4, "JUNGLE": 2, "UTILITY": 5})
        self.assertEqual(assign_roles([], GAMES), {})

    def test_resolve_keeps_player_role_free(self) -> None:
        session = make_session()
        for player in session["myTeam"]:
            player["assignedPosition"] = ""                           # File sans rôles attribués
        live = parse_champ_select(session)
        live.role = "JUNGLE"                                          # Rôle choisi à la main dans l'app
        ally, enemy = resolve(live, GAMES)
        self.assertEqual(ally, {"TOP": 1, "MIDDLE": 3})
        self.assertEqual(enemy, {"BOTTOM": 4, "UTILITY": 5, "MIDDLE": 6})

    def test_manual_role_correction_is_kept(self) -> None:
        """Le joueur a corrigé : 6 est en fait support. 5 (support pur) prend alors le rôle restant le plus probable."""
        live = parse_champ_select(make_session())
        _, enemy = resolve(live, GAMES, enemy_pinned={6: "UTILITY", 12345: "TOP"})   # 12345 : plus dans la draft
        self.assertEqual(enemy["UTILITY"], 6)
        self.assertEqual(enemy["BOTTOM"], 4)
        self.assertEqual(sorted(enemy.values()), [4, 5, 6])


if __name__ == "__main__":
    unittest.main(verbosity=2)
