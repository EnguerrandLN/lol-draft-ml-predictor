"""
tests/test_additive.py — Garanties du modèle additif et du recommandeur.

  - Antisymétrie : échanger les équipes inverse le logit (hors avantage de côté).
  - Draft complète : le recommandeur donne exactement la même proba que le modèle.
  - Contre : un matchup défavorable connu ressort comme « pire réponse ».
  - Données synthétiques : un champion réellement plus fort est retrouvé.
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.additive_model import (
    AdditiveDraftModel, DamageProfile, Hyperparams, comp_key, duo_key, fit_comp_effects, lane_key, main_key, tier_key,
)
from ml.draft_data import ROLES
from ml.recommend import DraftRecommender, DraftState

# Champions 1..30 : 6 par rôle
ROLE_CHAMPS = {role: list(range(1 + 6 * i, 7 + 6 * i)) for i, role in enumerate(ROLES)}


def random_drafts(n: int, rng: np.random.Generator) -> pd.DataFrame:
    data = {}
    for role, champs in ROLE_CHAMPS.items():
        picks = np.array([rng.choice(champs, 2, replace=False) for _ in range(n)])
        data[f"blue_{role}"], data[f"red_{role}"] = picks[:, 0], picks[:, 1]
    df = pd.DataFrame(data)
    df["game_creation"] = np.arange(n) * 60_000
    return df


def hand_model() -> AdditiveDraftModel:
    """Petit modèle aux effets choisis, avec pick rates uniformes."""
    effects = {main_key("TOP", 1): 0.3, main_key("TOP", 2): -0.2, main_key("MIDDLE", 13): 0.1}
    k, s = lane_key("TOP", 1, 3)
    effects[k] = s * -0.5               # 1 perd nettement contre 3
    effects[duo_key("BOTTOM", "UTILITY", 19, 25)] = 0.25
    effects[duo_key("JUNGLE", "TOP", 7, 1)] = 0.15
    model = AdditiveDraftModel(intercept=0.04, effects=effects, hyperparams=Hyperparams())
    model.games = {r: {str(c): 500 for c in cs} for r, cs in ROLE_CHAMPS.items()}
    model.pick_rate = {r: {str(c): 1 / len(cs) for c in cs} for r, cs in ROLE_CHAMPS.items()}
    model.champion_names = {str(c): f"C{c}" for cs in ROLE_CHAMPS.values() for c in cs}
    return model


def hand_model_with_comp() -> AdditiveDraftModel:
    """hand_model + profils de dégâts (ids impairs AD, pairs AP) et sensibilités « comp »."""
    model = hand_model()
    all_champs = [c for cs in ROLE_CHAMPS.values() for c in cs]
    model.damage = DamageProfile({c: (0.9 if c % 2 else 0.1) for c in all_champs}, mean=0.5, std=0.15)
    model.effects.update({comp_key(1): 0.4, comp_key(3): -0.3, comp_key(13): 0.2, comp_key(20): 0.25})
    model.effects.update({"b|lin": 0.05, "b|sq": -0.3})   # Équipe déséquilibrée pénalisée
    return model


class TestAdditiveModel(unittest.TestCase):

    def test_swapping_teams_flips_logit(self) -> None:
        model = hand_model()
        df = random_drafts(50, np.random.default_rng(0))
        swapped = df.rename(columns=lambda c: c.replace("blue_", "tmp_").replace("red_", "blue_").replace("tmp_", "red_"))
        lg, lg_swapped = model.predict_logit(df), model.predict_logit(swapped)
        np.testing.assert_allclose(lg - model.intercept, -(lg_swapped - model.intercept), atol=1e-12)

    def test_fit_recovers_strong_champion(self) -> None:
        rng = np.random.default_rng(1)
        df = random_drafts(20_000, rng)
        true_logit = 0.6 * ((df.blue_TOP == 1).astype(float) - (df.red_TOP == 1).astype(float))
        df["blue_win"] = (rng.random(len(df)) < 1 / (1 + np.exp(-true_logit))).astype(int)

        model = AdditiveDraftModel.fit(df, Hyperparams(C=1.0))

        strongest = max(ROLE_CHAMPS["TOP"], key=lambda c: model.effects.get(main_key("TOP", c), 0.0))
        self.assertEqual(strongest, 1)
        self.assertAlmostEqual(model.effects[main_key("TOP", 1)] - model.effects[main_key("TOP", 2)], 0.6, delta=0.15)


class TestCompEffects(unittest.TestCase):
    """Sensibilités au profil adverse lissées (fit_comp_effects)."""

    SHARES = {c: (0.9 if c % 2 else 0.1) for cs in ROLE_CHAMPS.values() for c in cs}

    def simulate(self, effect_early: float, effect_late: float, seed: int) -> pd.DataFrame:
        """Champion 1 (Top) gagne d'autant plus que l'équipe adverse est AD ; l'effet peut
        différer entre la première et la seconde moitié de la période (dérive)."""
        rng = np.random.default_rng(seed)
        df = random_drafts(40_000, rng)
        prof = DamageProfile.fit(df, self.SHARES)
        z_blue, z_red = prof.z(prof.team_ad(df, "blue")), prof.z(prof.team_ad(df, "red"))
        effect = np.where(np.arange(len(df)) < len(df) // 2, effect_early, effect_late)
        logit = effect * ((df.blue_TOP == 1) * z_red - (df.red_TOP == 1) * z_blue)
        df["blue_win"] = (rng.random(len(df)) < 1 / (1 + np.exp(-logit))).astype(int)
        return df

    def fit(self, df: pd.DataFrame):
        model = AdditiveDraftModel.fit(df, Hyperparams(C=1.0, bal_scale=1.0), self.SHARES)
        return fit_comp_effects(model, df)

    def test_planted_effect_is_recovered_with_right_sign(self) -> None:
        effects, stats = self.fit(self.simulate(0.4, 0.4, seed=5))
        self.assertGreater(stats.loc[1, "weight"], 0.6)
        self.assertGreater(effects[comp_key(1)], 0.2)
        self.assertLess(effects[comp_key(1)], 0.4)   # Atténué, jamais amplifié

    def test_no_effect_gives_negligible_effects(self) -> None:
        effects, stats = self.fit(self.simulate(0.0, 0.0, seed=6))
        self.assertLess(max((abs(e) for e in effects.values()), default=0.0), 0.03)
        self.assertLess(stats.weight.max(), 0.2)

    def test_effect_that_flips_over_time_is_neutralized(self) -> None:
        """Un effet fort mais qui change de signe à mi-parcours ne doit pas être retenu (cas « Pyke »)."""
        effects, stats = self.fit(self.simulate(0.4, -0.4, seed=7))
        self.assertLess(abs(effects.get(comp_key(1), 0.0)), 0.05)


class TestRecommender(unittest.TestCase):

    def setUp(self) -> None:
        self.model = hand_model()
        self.rec = DraftRecommender(self.model)

    def test_full_draft_matches_model(self) -> None:
        """Toute la draft connue sauf le Top allié : la proba doit égaler celle du modèle."""
        self._check_full_draft(self.model, self.rec)

    def test_full_draft_matches_model_with_comp(self) -> None:
        """Même garantie avec les termes de composition (croisés entre le candidat et son vis-à-vis)."""
        model = hand_model_with_comp()
        self._check_full_draft(model, DraftRecommender(model))

    def test_comp_matches_enumeration_with_unknown_laner(self) -> None:
        """Vis-à-vis inconnu : la proba attendue doit égaler la moyenne exacte sur ses picks possibles."""
        model = hand_model_with_comp()
        rec = DraftRecommender(model)
        df = random_drafts(5, np.random.default_rng(3))
        for _, row in df.iterrows():
            ally = {r: int(row[f"blue_{r}"]) for r in ROLES if r != "TOP"}
            enemy = {r: int(row[f"red_{r}"]) for r in ROLES if r != "TOP"}
            got = {r.champion_id: r for r in rec.recommend(DraftState(ally=ally, enemy=enemy, ally_side="blue"), "TOP", min_games=0)}
            for cand in ROLE_CHAMPS["TOP"]:
                options = [e for e in ROLE_CHAMPS["TOP"] if e != cand]   # Pick rates uniformes
                logits = []
                for e in options:
                    full = row.copy()
                    full["blue_TOP"], full["red_TOP"] = cand, e
                    logits.append(model.predict_logit(pd.DataFrame([full]))[0])
                expected = 1 / (1 + math.exp(-np.mean(logits)))
                self.assertAlmostEqual(got[cand].win_prob, expected, places=10)

    def test_full_draft_matches_model_with_tier(self) -> None:
        """Écarts par tranche d'ELO : la proba du recommandeur (tranche HIGH) = celle du modèle sur un match HIGH."""
        model = hand_model_with_comp()
        model.effects.update({tier_key("HIGH", "TOP", 1): -0.5, tier_key("HIGH", "MIDDLE", 13): 0.3})
        rec = DraftRecommender(model)
        df = random_drafts(10, np.random.default_rng(7)).assign(source_tier="MASTER")
        for _, row in df.iterrows():
            state = DraftState(
                ally={r: int(row[f"blue_{r}"]) for r in ROLES if r != "TOP"},
                enemy={r: int(row[f"red_{r}"]) for r in ROLES},
                ally_side="blue", tier="HIGH",
            )
            for r in rec.recommend(state, "TOP", min_games=0):
                full = row.copy()
                full["blue_TOP"] = r.champion_id
                self.assertAlmostEqual(r.win_prob, model.predict_proba(pd.DataFrame([full]))[0], places=10)

    def test_tier_deviation_changes_ranking_only_for_that_tier(self) -> None:
        model = hand_model()
        model.effects[tier_key("HIGH", "TOP", 1)] = -1.0   # Champion 1 : fort en général, faible en Master+
        rec = DraftRecommender(model)
        self.assertEqual(rec.recommend(DraftState(), "TOP", min_games=0)[0].champion_id, 1)
        self.assertNotEqual(rec.recommend(DraftState(tier="HIGH"), "TOP", min_games=0)[0].champion_id, 1)
        self.assertEqual(rec.recommend(DraftState(tier="LOW"), "TOP", min_games=0)[0].champion_id, 1)

    def test_tier_calibration_scales_logit(self) -> None:
        model = hand_model()
        model.meta["tier_calibration"] = {"HIGH": 0.5}
        rec = DraftRecommender(model)
        state = DraftState(enemy={"TOP": 3}, ally_side="blue")
        base = {r.champion_id: r.win_prob for r in rec.recommend(state, "TOP", min_games=0)}
        high = {r.champion_id: r.win_prob for r in rec.recommend(DraftState(enemy={"TOP": 3}, ally_side="blue", tier="HIGH"), "TOP", min_games=0)}
        logit = lambda p: math.log(p / (1 - p))
        for c in base:
            self.assertAlmostEqual(logit(high[c]), 0.5 * logit(base[c]), places=10)

    def test_draft_estimate_is_exact_with_two_unknown_slots(self) -> None:
        """Espérance (variance comprise pour l'équilibre) = moyenne exacte sur les 36 drafts possibles."""
        model = hand_model_with_comp()
        rec = DraftRecommender(model)
        row = random_drafts(1, np.random.default_rng(4)).iloc[0]
        ally = {r: int(row[f"blue_{r}"]) for r in ROLES if r != "TOP"}
        enemy = {r: int(row[f"red_{r}"]) for r in ROLES if r != "MIDDLE"}
        state = DraftState(ally=ally, enemy=enemy, ally_side="blue")

        logits = []
        for a in ROLE_CHAMPS["TOP"]:
            if a == enemy["TOP"]:
                continue
            for e in ROLE_CHAMPS["MIDDLE"]:
                if e == ally["MIDDLE"]:
                    continue
                full = row.copy()
                full["blue_TOP"], full["red_MIDDLE"] = a, e
                logits.append(model.predict_logit(pd.DataFrame([full]))[0])
        got = rec.expected_logit(rec._slot_dists(state), "blue")
        self.assertAlmostEqual(got, float(np.mean(logits)), places=10)

    def test_unbalanced_team_pushes_other_damage_type(self) -> None:
        """Équipe alliée full AD (ids impairs) : un ADC AP (id pair) doit être favorisé, et inversement."""
        rec = DraftRecommender(hand_model_with_comp())
        full_ad = {r: next(c for c in cs if c % 2) for r, cs in ROLE_CHAMPS.items() if r != "BOTTOM"}
        full_ap = {r: next(c for c in cs if c % 2 == 0) for r, cs in ROLE_CHAMPS.items() if r != "BOTTOM"}
        with_ad = {r.champion_id: r.win_prob for r in rec.recommend(DraftState(ally=full_ad), "BOTTOM", min_games=0)}
        with_ap = {r.champion_id: r.win_prob for r in rec.recommend(DraftState(ally=full_ap), "BOTTOM", min_games=0)}
        ap_adc, ad_adc = 20, 19
        self.assertGreater(with_ad[ap_adc] - with_ad[ad_adc], 0.02)
        self.assertGreater(with_ap[ad_adc] - with_ap[ap_adc], 0.02)

    def test_ad_heavy_enemy_favors_armor_champion(self) -> None:
        """Champion 1 (comp +0.4) doit gagner plus contre une équipe AD que contre une équipe AP."""
        rec = DraftRecommender(hand_model_with_comp())
        ad_team = {r: next(c for c in cs if c % 2) for r, cs in ROLE_CHAMPS.items() if r != "TOP"}
        ap_team = {r: next(c for c in cs if c % 2 == 0) for r, cs in ROLE_CHAMPS.items() if r != "TOP"}
        vs_ad = {r.champion_id: r.win_prob for r in rec.recommend(DraftState(enemy=ad_team), "TOP", min_games=0)}
        vs_ap = {r.champion_id: r.win_prob for r in rec.recommend(DraftState(enemy=ap_team), "TOP", min_games=0)}
        self.assertGreater(vs_ad[1] - vs_ap[1], 0.1)
        self.assertLess(vs_ad[3] - vs_ap[3], 0)   # Champion 3 (comp −0.3) : l'inverse

    def _check_full_draft(self, model, rec) -> None:
        df = random_drafts(20, np.random.default_rng(2))
        for _, row in df.iterrows():
            state = DraftState(
                ally={r: int(row[f"blue_{r}"]) for r in ROLES if r != "TOP"},
                enemy={r: int(row[f"red_{r}"]) for r in ROLES},
                ally_side="blue",
            )
            by_champ = {r.champion_id: r.win_prob for r in rec.recommend(state, "TOP", min_games=0)}
            for cand, prob in by_champ.items():
                full = row.copy()
                full["blue_TOP"] = cand
                expected = model.predict_proba(pd.DataFrame([full]))[0]
                self.assertAlmostEqual(prob, expected, places=10)

    def test_expected_logit_with_empty_draft_is_average(self) -> None:
        """Draft vide, côté inconnu : l'espérance est la moyenne exacte sur toutes les drafts."""
        dists = self.rec._slot_dists(DraftState())
        lg = self.rec.expected_logit(dists, None)
        # Termes non nuls en moyenne : main TOP (0.3−0.2)/6 côté allié moins côté ennemi = 0
        self.assertAlmostEqual(lg, 0.0, places=10)

    def test_counter_pick_is_detected(self) -> None:
        """Champion 1 est fort mais perd contre 3 : en blind, 3 est sa pire réponse."""
        results = {r.champion_id: r for r in self.rec.recommend(DraftState(), "TOP", min_games=0)}
        self.assertEqual(results[1].worst_response, "C3")
        self.assertLess(results[1].win_prob_if_countered, results[1].win_prob)
        # Si l'ennemi a déjà pické 3, la proba de 1 tombe à la valeur « contrée »
        known = {r.champion_id: r for r in self.rec.recommend(DraftState(enemy={"TOP": 3}), "TOP", min_games=0)}
        self.assertLess(known[1].win_prob, results[1].win_prob)

    def test_synergy_raises_partner(self) -> None:
        with_braum = {r.champion_id: r.win_prob for r in self.rec.recommend(DraftState(ally={"UTILITY": 25}), "BOTTOM", min_games=0)}
        without = {r.champion_id: r.win_prob for r in self.rec.recommend(DraftState(ally={"UTILITY": 26}), "BOTTOM", min_games=0)}
        self.assertGreater(with_braum[19] - without[19], 0.05)

    def test_unavailable_champions_are_excluded(self) -> None:
        state = DraftState(enemy={"MIDDLE": 13}, bans={2})
        ids = {r.champion_id for r in self.rec.recommend(state, "TOP", pool={1, 2, 3}, min_games=0)}
        self.assertEqual(ids, {1, 3})


if __name__ == "__main__":
    unittest.main(verbosity=2)
