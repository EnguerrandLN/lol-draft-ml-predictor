"""
tests/test_training.py — Pipeline d'entraînement et sérialisation du modèle.

  - Aller-retour JSON : un modèle complet (profil de dégâts, sensibilités,
    calibration par tranche, pick rates par tranche, familiarité) rechargé
    donne exactement les mêmes recommandations.
  - Pick rates par tranche : un slot vide est estimé avec les picks du niveau.
  - Pipeline : réglage, garde-fou des sensibilités, rapport et calibration par
    tranche tournent de bout en bout sur des données simulées.
"""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import ml.train as train
from ml.additive_model import AdditiveDraftModel, Hyperparams
from ml.recommend import DraftRecommender, DraftState
from tests.test_additive import ROLE_CHAMPS, hand_model, hand_model_with_comp, random_drafts

SHARES = {c: (0.9 if c % 2 else 0.1) for cs in ROLE_CHAMPS.values() for c in cs}
TIERS = ["GOLD", "EMERALD", "MASTER", None]


def simulated_matches(n: int, seed: int) -> pd.DataFrame:
    """Drafts aléatoires dont l'issue suit un modèle connu, avec dates et tiers."""
    rng = np.random.default_rng(seed)
    df = random_drafts(n, rng)
    truth = hand_model_with_comp()
    p = 1 / (1 + np.exp(-truth.predict_logit(df)))
    df["blue_win"] = (rng.random(n) < p).astype(int)
    df["source_tier"] = [TIERS[i % len(TIERS)] for i in range(n)]
    df.index = [f"M{i}" for i in range(n)]
    return df


class TestSerialization(unittest.TestCase):

    def test_full_model_roundtrip(self) -> None:
        model = hand_model_with_comp()
        model.meta.update(tier_calibration={"HIGH": 0.6}, familiarity={"common": {"none": {"effect": -0.05}}})
        model.pick_rate_by_tier = {"HIGH": {r: {str(c): 1 / 6 for c in cs} for r, cs in ROLE_CHAMPS.items()}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            model.save(path)
            loaded = AdditiveDraftModel.load(path)

        state = DraftState(enemy={"MIDDLE": 13, "BOTTOM": 19}, ally={"UTILITY": 25}, ally_side="red", tier="HIGH")
        before = [(r.champion_id, r.win_prob) for r in DraftRecommender(model).recommend(state, "TOP", min_games=0)]
        after = [(r.champion_id, r.win_prob) for r in DraftRecommender(loaded).recommend(state, "TOP", min_games=0)]
        self.assertEqual([c for c, _ in before], [c for c, _ in after])
        np.testing.assert_allclose([p for _, p in before], [p for _, p in after], rtol=0, atol=1e-12)
        self.assertEqual(loaded.damage.lo, model.damage.lo)


class TestTierPickRates(unittest.TestCase):

    def test_empty_slot_uses_tier_pick_rates(self) -> None:
        """En Master+, tous les Tops adverses jouent 3 : vis-à-vis inconnu ≡ vis-à-vis 3 connu."""
        model = hand_model()
        high = {r: {str(c): 1 / 6 for c in cs} for r, cs in ROLE_CHAMPS.items()}
        high["TOP"] = {"3": 1.0}
        model.pick_rate_by_tier = {"HIGH": high}
        rec = DraftRecommender(model)
        unknown = {r.champion_id: r.win_prob for r in rec.recommend(DraftState(tier="HIGH"), "TOP", min_games=0)}
        known = {r.champion_id: r.win_prob
                 for r in rec.recommend(DraftState(enemy={"TOP": 3}, tier="HIGH"), "TOP", min_games=0)}
        self.assertAlmostEqual(unknown[1], known[1], places=10)
        # Sans niveau : pick rates globaux, le vis-à-vis 3 n'est qu'une possibilité parmi 6
        self.assertGreater(rec.recommend(DraftState(), "TOP", min_games=0, pool={1})[0].win_prob, unknown[1])


class TestTrainingPipeline(unittest.TestCase):

    @patch.object(train, "C_GRID", (0.1,))
    @patch.object(train, "SCALE_GRID", (0.0, 0.5))
    @patch.object(train, "HALF_LIFE_GRID", (None,))
    def test_pipeline_runs_end_to_end(self) -> None:
        df = simulated_matches(6000, seed=11)
        train_full, test = train.temporal_split(df, 0.15)

        hp = train.tune(train_full, SHARES)
        self.assertIsInstance(hp, Hyperparams)
        self.assertGreater(hp.bal_scale, 0)          # Effet d'équilibre présent dans la vérité simulée

        decision = train.decide_comp(train_full, hp, SHARES)
        self.assertEqual(set(decision), {"adopted", "gain", "ci95"})

        results = train.report(train_full, test, hp, SHARES, use_comp=decision["adopted"])
        self.assertIn("Modèle retenu", results)
        self.assertGreater(results["Modèle retenu"]["gain"], 0)   # Mieux que la constante

        calibration = train.estimate_tier_calibration(df, hp, SHARES)
        self.assertEqual(set(calibration), {"LOW", "MID", "HIGH"})
        for c in calibration.values():
            self.assertGreater(c["scale"], 0.3)
            self.assertLess(c["scale"], 1.7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
