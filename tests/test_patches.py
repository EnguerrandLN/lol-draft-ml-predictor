"""
tests/test_patches.py — Évolution des forces d'un patch à l'autre (groupe « patch »).

  - Un nerf simulé est retrouvé : la force principale (« main ») suit le DERNIER
    patch, pas la moyenne de tous les patchs.
  - Une draft d'un patch futur est prédite avec la force du dernier patch.
  - Sans le groupe (patch_scale = 0), aucun terme « patch » n'est produit.
  - Le recommandeur reste exact pour une partie du patch courant.
  - La liste des patchs suivis survit à la sérialisation.
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.additive_model import AdditiveDraftModel, Hyperparams, main_key, tracked_patches
from ml.draft_data import ROLES
from ml.recommend import DraftRecommender, DraftState
from tests.test_additive import random_drafts


def nerf_data(n_per_patch: int, seed: int) -> pd.DataFrame:
    """Le Top 1 est très fort au patch 16.1 (+0.6), puis nerfé au 16.2 (−0.2)."""
    rng = np.random.default_rng(seed)
    parts = []
    for i, (patch, strength) in enumerate((("16.1", 0.6), ("16.2", -0.2))):
        d = random_drafts(n_per_patch, rng)
        logit = strength * ((d.blue_TOP == 1).astype(float) - (d.red_TOP == 1).astype(float))
        d["blue_win"] = (rng.random(len(d)) < 1 / (1 + np.exp(-logit))).astype(int)
        d["patch"] = patch
        d["game_creation"] = (i * n_per_patch + np.arange(n_per_patch)) * 60_000
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df.index = [f"M{i}" for i in range(len(df))]
    return df


class TestPatchLayer(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.df = nerf_data(15_000, seed=1)
        cls.with_patch = AdditiveDraftModel.fit(cls.df, Hyperparams(C=1.0, patch_scale=1.0))
        cls.without = AdditiveDraftModel.fit(cls.df, Hyperparams(C=1.0))

    def strength(self, model, champ=1) -> float:
        """Écart de force du Top 1 par rapport à la moyenne des autres Tops."""
        top = [model.effects.get(main_key("TOP", c), 0.0) for c in range(1, 7)]
        return top[champ - 1] - np.mean(top[:champ - 1] + top[champ:])

    def test_main_effect_follows_latest_patch(self) -> None:
        self.assertEqual(self.with_patch.patches, ["16.1", "16.2"])
        self.assertLess(self.strength(self.with_patch), 0.0)          # Nerfé : faible au dernier patch
        self.assertGreater(self.strength(self.without), 0.1)          # Sans le groupe : moyenne des deux patchs

    def test_future_patch_uses_latest_strength(self) -> None:
        future = self.df.iloc[:200].assign(patch="16.3")
        latest = self.df.iloc[:200].assign(patch="16.2")
        np.testing.assert_allclose(self.with_patch.predict_logit(future), self.with_patch.predict_logit(latest))

    def test_no_patch_terms_without_group(self) -> None:
        self.assertEqual(self.without.patches, [])
        self.assertFalse(any(k.startswith("p|") for k in self.without.effects))

    def test_recommender_exact_at_current_patch(self) -> None:
        model = self.with_patch
        model.games = {r: {str(c): 500 for c in range(1 + 6 * i, 7 + 6 * i)} for i, r in enumerate(ROLES)}
        model.pick_rate = {r: {str(c): 1 / 6 for c in range(1 + 6 * i, 7 + 6 * i)} for i, r in enumerate(ROLES)}
        model.champion_names = {str(c): f"C{c}" for c in range(1, 31)}
        rec = DraftRecommender(model)
        row = self.df.iloc[-1]
        state = DraftState(ally={r: int(row[f"blue_{r}"]) for r in ROLES if r != "TOP"},
                           enemy={r: int(row[f"red_{r}"]) for r in ROLES}, ally_side="blue")
        for r in rec.recommend(state, "TOP", min_games=0):
            full = row.copy()
            full["blue_TOP"] = r.champion_id
            self.assertAlmostEqual(r.win_prob, model.predict_proba(pd.DataFrame([full]))[0], places=10)

    def test_patches_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            self.with_patch.save(path)
            loaded = AdditiveDraftModel.load(path)
        self.assertEqual(loaded.patches, self.with_patch.patches)
        sample = self.df.iloc[:300]
        np.testing.assert_allclose(loaded.predict_logit(sample), self.with_patch.predict_logit(sample))

    def test_small_patches_are_not_tracked(self) -> None:
        small = self.df.iloc[:1500].assign(patch="16.0")
        self.assertEqual(tracked_patches(pd.concat([small, self.df])), ["16.1", "16.2"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
