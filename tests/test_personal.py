"""
tests/test_personal.py — Personnalisation par l'historique du joueur.

  - estimate_tau retrouve l'écart réel simulé entre joueurs (et ~0 sans écart).
  - Le décalage personnel est atténué avec peu de parties, et suit le vrai
    écart avec beaucoup de parties.
  - Le recommandeur ajoute ce décalage au bon candidat et le reclasse.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.personal import champion_profiles, estimate_tau
from ml.recommend import DraftRecommender, DraftState
from tests.test_additive import hand_model


def simulate(n_players: int, n_champs: int, games: int, tau: float, rng) -> pd.DataFrame:
    """Résidus simulés : chaque paire (joueur, champion) a un vrai décalage N(0, τ²)."""
    rows = []
    for player in range(n_players):
        for champ in range(n_champs):
            delta = rng.normal(0, tau)
            p = rng.uniform(0.4, 0.6, games)
            logit = np.log(p / (1 - p)) + delta
            y = (rng.random(games) < 1 / (1 + np.exp(-logit))).astype(int)
            rows.append(pd.DataFrame({"puuid": player, "champion_id": champ, "p": p, "y": y, "position": "TOP"}))
    return pd.concat(rows, ignore_index=True)


class TestPersonal(unittest.TestCase):

    def test_estimate_tau_recovers_true_spread(self) -> None:
        rng = np.random.default_rng(0)
        res = simulate(n_players=400, n_champs=5, games=20, tau=0.4, rng=rng)
        self.assertAlmostEqual(estimate_tau(res), 0.4, delta=0.07)

    def test_estimate_tau_is_near_zero_without_spread(self) -> None:
        rng = np.random.default_rng(1)
        res = simulate(n_players=400, n_champs=5, games=20, tau=0.0, rng=rng)
        self.assertLess(estimate_tau(res), 0.08)

    def test_effect_is_shrunk_with_few_games(self) -> None:
        # 3 victoires sur 3 parties attendues à 50 % : écart brut énorme, effet atténué
        few = pd.DataFrame({"champion_id": 1, "p": [0.5] * 3, "y": [1] * 3, "position": "TOP"})
        effect = champion_profiles(few, tau=0.15)[1].effect
        self.assertGreater(effect, 0)
        self.assertLess(effect, 0.05)

    def test_effect_follows_true_gap_with_many_games(self) -> None:
        # 60 % de victoires sur 2000 parties attendues à 50 % : logit réel ≈ 0.405
        many = pd.DataFrame({"champion_id": 1, "p": [0.5] * 2000, "y": [1] * 1200 + [0] * 800, "position": "TOP"})
        profile = champion_profiles(many, tau=0.4)[1]
        self.assertEqual((profile.games, profile.wins), (2000, 1200))
        self.assertAlmostEqual(profile.effect, math.log(0.6 / 0.4), delta=0.05)


class TestRecommenderPersonal(unittest.TestCase):

    def test_personal_offset_reranks_candidate(self) -> None:
        rec = DraftRecommender(hand_model())
        base = {r.champion_id: r for r in rec.recommend(DraftState(), "TOP", min_games=0)}
        boosted = rec.recommend(DraftState(), "TOP", min_games=0, personal={4: 1.0})

        self.assertEqual(boosted[0].champion_id, 4)
        self.assertGreater(boosted[0].win_prob, base[4].win_prob)
        self.assertAlmostEqual(boosted[0].win_prob - base[4].win_prob, boosted[0].personal_effect, places=10)
        self.assertEqual(base[4].personal_effect, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
