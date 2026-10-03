"""
tests/test_personal.py — Personnalisation par l'historique du joueur.

  - estimate_tau retrouve l'écart réel simulé entre joueurs (et ~0 sans écart).
  - estimate_familiarity retrouve un coût d'inexpérience simulé.
  - Le décalage personnel part du coût d'inexpérience mesuré (a priori), est
    atténué avec peu de parties et suit le vrai écart avec beaucoup de parties ;
    un pick de niche reçoit l'a priori « niche ».
  - Le recommandeur ajoute ce décalage au bon candidat et le reclasse.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.personal import champion_profiles, estimate_familiarity, estimate_tau, personal_offsets
from ml.recommend import DraftRecommender, DraftState
from tests.test_additive import hand_model

FAMILIARITY = {
    "common": {"none": {"effect": -0.05}, "few": {"effect": 0.0}, "regular": {"effect": 0.03}},
    "niche": {"none": {"effect": -0.13}, "few": {"effect": -0.08}, "regular": {"effect": 0.02}},
}


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


def model_with_familiarity():
    model = hand_model()
    model.meta["familiarity"] = FAMILIARITY
    model.meta["personal_tau"] = 0.15
    model.games["MIDDLE"]["1"] = 20      # Champion 1 : Top (500 parties), rarement Mid → niche au Mid
    return model


class TestPersonal(unittest.TestCase):

    def test_estimate_tau_recovers_true_spread(self) -> None:
        rng = np.random.default_rng(0)
        res = simulate(n_players=400, n_champs=5, games=20, tau=0.4, rng=rng)
        self.assertAlmostEqual(estimate_tau(res), 0.4, delta=0.07)

    def test_estimate_tau_is_near_zero_without_spread(self) -> None:
        rng = np.random.default_rng(1)
        res = simulate(n_players=400, n_champs=5, games=20, tau=0.0, rng=rng)
        self.assertLess(estimate_tau(res), 0.08)

    def test_estimate_familiarity_recovers_inexperience_cost(self) -> None:
        """Joueurs avec un main et des picks occasionnels ; −0.4 de logit sur un champion jamais joué."""
        rng = np.random.default_rng(2)
        rows = []
        for player in range(600):
            main, seen, t = int(rng.integers(0, 10)), {}, 0
            for _ in range(40):
                champ = main if rng.random() < 0.6 else int(rng.integers(10, 60))
                prior = seen.get(champ, 0)
                effect = -0.4 if prior == 0 else (0.0 if prior <= 2 else 0.1)
                p = 0.5
                y = int(rng.random() < 1 / (1 + math.exp(-effect)))
                rows.append({"puuid": player, "champion_id": champ, "position": "TOP", "p": p, "y": y, "t": t})
                seen[champ] = prior + 1
                t += 1
        fam = estimate_familiarity(pd.DataFrame(rows), shares={})
        common = fam["niche"]   # shares vide → tout est « niche » ; seule la différence compte
        self.assertAlmostEqual(common["none"]["effect"] - common["regular"]["effect"], -0.5, delta=0.15)

    def test_no_profile_gives_inexperience_prior(self) -> None:
        offsets = personal_offsets(model_with_familiarity(), "TOP")
        self.assertTrue(all(o == FAMILIARITY["common"]["none"]["effect"] for o in offsets.values()))

    def test_niche_pick_gets_niche_prior(self) -> None:
        offsets = personal_offsets(model_with_familiarity(), "MIDDLE")
        self.assertEqual(offsets[1], FAMILIARITY["niche"]["none"]["effect"])     # Champion 1 au Mid : niche
        self.assertEqual(offsets[13], FAMILIARITY["common"]["none"]["effect"])   # Champion 13 : son rôle

    def test_effect_is_shrunk_toward_prior_with_few_games(self) -> None:
        # 3 victoires sur 3 parties attendues à 50 % : écart brut énorme, effet proche de l'a priori « few/regular »
        few = champion_profiles(pd.DataFrame({"champion_id": 1, "p": [0.5] * 3, "y": [1] * 3, "position": "TOP"}))
        delta = personal_offsets(model_with_familiarity(), "TOP", few)[1]
        prior = FAMILIARITY["common"]["regular"]["effect"]
        self.assertGreater(delta, prior)
        self.assertLess(delta - prior, 0.05)

    def test_effect_follows_true_gap_with_many_games(self) -> None:
        # 60 % de victoires sur 2000 parties attendues à 50 % : logit réel ≈ 0.405, quel que soit l'a priori
        many = champion_profiles(pd.DataFrame(
            {"champion_id": 1, "p": [0.5] * 2000, "y": [1] * 1200 + [0] * 800, "position": "TOP"}))
        self.assertEqual((many[1].games, many[1].wins), (2000, 1200))
        delta = personal_offsets(model_with_familiarity(), "TOP", many, tau=0.4)[1]
        self.assertAlmostEqual(delta, math.log(0.6 / 0.4), delta=0.05)


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
