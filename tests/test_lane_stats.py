"""
tests/test_lane_stats.py — Matchups appris sur les statistiques de lane.

  - Un matchup planté dans la cible de lane est retrouvé avec le bon signe.
  - Les effets transférés dans le modèle additif reproduisent exactement
    β × score de matchups (conventions de signe et de clés cohérentes).
  - Antisymétrie : échanger les équipes inverse le score.
  - La décision d'adoption et le rapport tournent sur des données simulées.
"""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import ml.lane_stats as lane_stats
import ml.train as train
from ml.additive_model import lane_key
from ml.draft_data import ROLES
from ml.lane_stats import fit_lane_models, pair_score, transferred_effects
from tests.test_additive import hand_model, random_drafts
from tests.test_training import SHARES, simulated_matches

SETTINGS = {r: (10.0, 1.0) for r in ROLES}


def planted_targets(df: pd.DataFrame, effect: float, seed: int) -> pd.DataFrame:
    """Part d'or : le Top 1 domine le Top 3 (et inversement), bruit ailleurs."""
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({r: rng.normal(0, 0.3, len(df)) for r in ROLES}, index=df.index)
    t["TOP"] += effect * ((df.blue_TOP == 1) & (df.red_TOP == 3)) - effect * ((df.blue_TOP == 3) & (df.red_TOP == 1))
    return t


def swap(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: c.replace("blue_", "tmp_").replace("red_", "blue_").replace("tmp_", "red_"))


class TestLaneStats(unittest.TestCase):

    def setUp(self) -> None:
        self.df = random_drafts(20_000, np.random.default_rng(3))
        self.targets = planted_targets(self.df, effect=0.5, seed=4)
        self.models = fit_lane_models(self.df, self.targets, SETTINGS)

    def test_planted_matchup_is_recovered(self) -> None:
        key, sign = lane_key("TOP", 1, 3)
        effect = sign * self.models["TOP"].pair_effects()[key]      # avantage de 1 sur 3
        self.assertGreater(effect, 0.25)
        others = [abs(v) for k, v in self.models["TOP"].pair_effects().items() if k != key]
        self.assertGreater(effect, 3 * max(others))

    def test_transferred_effects_reproduce_scaled_score(self) -> None:
        model = hand_model()
        beta = 0.7
        extra = transferred_effects(self.models, beta)
        updated = train.with_extra_effects(model, extra)
        sample = self.df.iloc[:500]
        np.testing.assert_allclose(
            updated.predict_logit(sample) - model.predict_logit(sample),
            beta * pair_score(self.models, sample), atol=1e-9,
        )

    def test_score_is_antisymmetric(self) -> None:
        sample = self.df.iloc[:500]
        np.testing.assert_allclose(pair_score(self.models, swap(sample)), -pair_score(self.models, sample), atol=1e-9)


class TestLaneStatsPipeline(unittest.TestCase):

    @patch.object(train, "C_GRID", (0.1,))
    @patch.object(train, "SCALE_GRID", (0.0, 0.5))
    @patch.object(train, "HALF_LIFE_GRID", (None,))
    @patch.object(lane_stats, "ALPHA_GRID", (10.0,))
    @patch.object(lane_stats, "PAIR_SCALE_GRID", (0.0, 1.0))
    def test_decision_and_report_run(self) -> None:
        df = simulated_matches(6000, seed=12)
        targets = planted_targets(df, effect=0.5, seed=13)
        train_full, test = train.temporal_split(df, 0.15)
        hp = train.tune(train_full, SHARES)
        decision = train.decide_lane_stats(train_full, hp, SHARES, use_comp=False, targets=targets)
        self.assertEqual(set(decision), {"adopted", "gain", "ci95"})
        results = train.report(train_full, test, hp, SHARES, use_comp=False,
                               lane_targets_df=targets, use_lane=decision["adopted"])
        self.assertIn("+ matchups appris sur l'or", results)
        self.assertIn("Modèle retenu", results)


if __name__ == "__main__":
    unittest.main(verbosity=2)
