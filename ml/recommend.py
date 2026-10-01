"""
recommend.py — Recommandation de pick à n'importe quel moment de la draft.

Principe : le modèle additif donne le logit d'une draft COMPLÈTE. Pour une draft
partielle, chaque slot encore vide est remplacé par sa distribution probable
(pick rates du rôle, hors champions déjà pris ou bannis), et on calcule
l'espérance du logit. Le modèle étant linéaire, cette espérance est exacte
pour chaque terme (sous l'hypothèse que les slots vides sont indépendants).

Pour un candidat `c` au rôle r, les termes qui le lient au slot ennemi e du
même rôle sont le matchup de lane et l'interaction de composition (le profil
de dégâts de chaque équipe inclut c et e). On décompose donc exactement :

    logit(c, e) = B + A(c) + E(e) + P(c, e)

    B      : tout ce qui ne dépend ni de c ni de e
    A(c)   : force de c + synergies + réaction de c (et des ennemis) au profil des équipes
    E(e)   : idem pour le vis-à-vis e (signe négatif : c'est l'ennemi)
    P(c,e) : matchup de lane + interaction de composition entre c et e

  - Probabilité attendue : moyenne sur e selon sa distribution probable.
  - Si contré : parmi les réponses plausibles du vis-à-vis, celle dont le
    terme de paire P(c, e) est le pire pour c remplace sa valeur moyenne. La force
    propre E(e) reste à sa moyenne : sinon la « pire réponse » serait juste
    le champion le plus fort du rôle, identique pour tous les candidats, et
    ne mesurerait pas l'exposition de c aux counters. Pertinent quand tu
    picks avant ton vis-à-vis.

Usage :
  python ml/recommend.py --role TOP --enemy MIDDLE=Zed JUNGLE=LeeSin --bans Yasuo
  python ml/recommend.py --role BOTTOM --ally UTILITY=Braum --pool Jinx Caitlyn Ezreal --risk 0.5
"""
import argparse
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATA_DIR
from ml.additive_model import BALANCE_KEYS, DUO_PAIRS, AdditiveDraftModel
from ml.draft_data import ROLES

ROLE_ALIASES: dict[str, str] = {
    "TOP": "TOP", "JUNGLE": "JUNGLE", "JGL": "JUNGLE", "JG": "JUNGLE",
    "MID": "MIDDLE", "MIDDLE": "MIDDLE",
    "ADC": "BOTTOM", "BOT": "BOTTOM", "BOTTOM": "BOTTOM",
    "SUPPORT": "UTILITY", "SUP": "UTILITY", "SUPP": "UTILITY", "UTILITY": "UTILITY",
}

MIN_RESPONSE_SHARE: float = 0.01   # Réponse ennemie « plausible » : ≥ 1 % des picks du rôle
MAX_RESPONSES: int = 15

Dist = dict[int, float]  # champion → probabilité


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@dataclass
class DraftState:
    """Draft vue de ton équipe. Rôles au format Riot (TOP, JUNGLE, MIDDLE, BOTTOM, UTILITY)."""
    ally: dict[str, int] = field(default_factory=dict)
    enemy: dict[str, int] = field(default_factory=dict)
    bans: set[int] = field(default_factory=set)
    ally_side: Optional[str] = None  # "blue" / "red" / None si inconnu

    def unavailable(self) -> set[int]:
        return set(self.ally.values()) | set(self.enemy.values()) | self.bans


@dataclass
class Recommendation:
    champion_id: int
    name: str
    games: int
    win_prob: float              # Espérance sur les slots encore vides
    vs_average: float            # Écart avec un pick moyen à ce rôle (points de proba)
    win_prob_if_countered: float # Si le vis-à-vis choisit ton pire matchup plausible
    worst_response: Optional[str] # Ce pire matchup (None si aucun matchup défavorable connu)
    worst_response_id: Optional[int]
    score: float                 # Critère de tri (mélange attendu / contré selon risk)
    personal_effect: float = 0.0 # Part de win_prob due à ton historique sur ce champion (points de proba)


class DraftRecommender:

    def __init__(self, model: AdditiveDraftModel) -> None:
        self.model = model
        self.names: dict[int, str] = {int(k): v for k, v in model.champion_names.items()}

        # Index des effets par champion pour des sommes rapides
        self.main: dict[str, dict[int, float]] = {r: {} for r in ROLES}
        self.lane: dict[str, dict[int, dict[int, float]]] = {r: {} for r in ROLES}  # a → {b: avantage de a sur b}
        self.duo: dict[tuple[str, str], dict[int, dict[int, float]]] = {}            # (r1,r2) → c1 → {c2: effet}
        for (r1, r2) in DUO_PAIRS:
            self.duo[(r1, r2)] = {}
            self.duo[(r2, r1)] = {}  # Même effet, indexé depuis l'autre rôle

        for key, eff in model.effects.items():
            parts = key.split("|")
            if parts[0] == "m":
                self.main[parts[1]][int(parts[2])] = eff
            elif parts[0] == "l":
                role, lo, hi = parts[1], int(parts[2]), int(parts[3])
                self.lane[role].setdefault(lo, {})[hi] = eff
                self.lane[role].setdefault(hi, {})[lo] = -eff
            elif parts[0] == "d":
                r1, r2, c1, c2 = parts[1], parts[2], int(parts[3]), int(parts[4])
                self.duo[(r1, r2)].setdefault(c1, {})[c2] = eff
                self.duo[(r2, r1)].setdefault(c2, {})[c1] = eff

        # Sensibilité de chaque champion au profil de dégâts adverse (groupe « comp »)
        self.damage = model.damage
        self.comp: dict[int, float] = {
            int(k.split("|")[1]): eff for k, eff in model.effects.items() if k.startswith("a|")
        }
        self.bal_lin, self.bal_sq = (model.effects.get(k, 0.0) for k in BALANCE_KEYS)

        self.pick_rate: dict[str, Dist] = {
            r: {int(c): p for c, p in model.pick_rate[r].items()} for r in ROLES
        }

    # ── Distributions des slots ──────────────────────────────────────────────

    def _slot_dists(self, state: DraftState) -> dict[tuple[str, str], Dist]:
        """(équipe, rôle) → distribution : masse 1 si connu, sinon pick rates disponibles."""
        unavailable = state.unavailable()
        dists = {}
        for team, picks in (("ally", state.ally), ("enemy", state.enemy)):
            for role in ROLES:
                if role in picks:
                    dists[(team, role)] = {picks[role]: 1.0}
                else:
                    avail = {c: p for c, p in self.pick_rate[role].items() if c not in unavailable}
                    total = sum(avail.values())
                    dists[(team, role)] = {c: p / total for c, p in avail.items()} if total else {}
        return dists

    # ── Termes du logit ──────────────────────────────────────────────────────

    def _main_term(self, role: str, dist: Dist) -> float:
        return sum(p * self.main[role].get(c, 0.0) for c, p in dist.items())

    def _lane_term(self, role: str, ally: Dist, enemy: Dist) -> float:
        total = 0.0
        for a, pa in ally.items():
            row = self.lane[role].get(a)
            if row:
                total += pa * sum(eff * enemy.get(b, 0.0) for b, eff in row.items())
        return total

    def _duo_term(self, r1: str, r2: str, d1: Dist, d2: Dist) -> float:
        total = 0.0
        for c1, p1 in d1.items():
            row = self.duo[(r1, r2)].get(c1)
            if row:
                total += p1 * sum(eff * d2.get(c2, 0.0) for c2, eff in row.items())
        return total

    def _share(self, champ: int) -> float:
        return self.damage.share(champ) if self.damage else 0.0

    def _team_sums(self, dists, team: str) -> tuple[float, float]:
        """(Σ part AD attendue, Σ sensibilité comp attendue) sur les slots de l'équipe.
        Une distribution vide ne contribue pas (slot traité à part)."""
        ad = sum(p * self._share(c) for role in ROLES for c, p in dists[(team, role)].items())
        comp = sum(p * self.comp.get(c, 0.0) for role in ROLES for c, p in dists[(team, role)].items())
        return ad, comp

    def _z(self, ad_sum: float) -> float:
        """Part AD d'équipe (somme sur 5 slots) bornée puis centrée-réduite, comme à l'entraînement.
        Avec des slots inconnus, la borne s'applique à la part attendue (approximation
        qui n'intervient qu'aux compositions extrêmes)."""
        return float(self.damage.z(ad_sum / 5))

    def _comp_term(self, dists) -> float:
        if not self.damage:
            return 0.0
        ad_ally, comp_ally = self._team_sums(dists, "ally")
        ad_enemy, comp_enemy = self._team_sums(dists, "enemy")
        return comp_ally * self._z(ad_enemy) - comp_enemy * self._z(ad_ally)

    def _ad_moments(self, dists, team: str) -> tuple[float, float]:
        """Espérance et variance de la somme des parts AD de l'équipe (slots indépendants)."""
        mean = var = 0.0
        for role in ROLES:
            dist = dists[(team, role)]
            m = sum(p * self._share(c) for c, p in dist.items())
            mean += m
            var += sum(p * self._share(c) ** 2 for c, p in dist.items()) - m * m
        return mean, var

    def _balance(self, ad_sum_mean: float, ad_sum_var: float) -> float:
        """E[b_lin·z + b_sq·z²] pour une équipe : E[z²] = E[z]² + Var(z)."""
        z = self._z(ad_sum_mean)
        var_z = ad_sum_var / (25 * self.damage.std ** 2)
        return self.bal_lin * z + self.bal_sq * (z * z + var_z)

    def _balance_term(self, dists) -> float:
        if not self.damage:
            return 0.0
        return self._balance(*self._ad_moments(dists, "ally")) - self._balance(*self._ad_moments(dists, "enemy"))

    def expected_logit(self, dists: dict[tuple[str, str], Dist], ally_side: Optional[str],
                       include_balance: bool = True) -> float:
        """
        Espérance du logit (point de vue allié). Une distribution vide annule ses
        termes linéaires. L'équilibre d'équipe n'étant pas linéaire, il n'a de sens
        que sur une draft dont chaque slot a une distribution : `include_balance=False`
        l'exclut quand des slots sont vidés pour être traités à part.
        """
        side = {"blue": 1.0, "red": -1.0}.get(ally_side or "", 0.0)
        logit = side * self.model.intercept
        for role in ROLES:
            logit += self._main_term(role, dists[("ally", role)])
            logit -= self._main_term(role, dists[("enemy", role)])
            logit += self._lane_term(role, dists[("ally", role)], dists[("enemy", role)])
        for r1, r2 in DUO_PAIRS:
            logit += self._duo_term(r1, r2, dists[("ally", r1)], dists[("ally", r2)])
            logit -= self._duo_term(r1, r2, dists[("enemy", r1)], dists[("enemy", r2)])
        logit += self._comp_term(dists)
        return logit + self._balance_term(dists) if include_balance else logit

    def _own_terms(self, team: str, role: str, champ: int, dists) -> float:
        """Force propre + synergies d'un champion placé dans (team, role), signe allié."""
        total = self.main[role].get(champ, 0.0)
        for r1, r2 in DUO_PAIRS:
            if role in (r1, r2):
                partner = r2 if role == r1 else r1
                row = self.duo[(role, partner)].get(champ, {})
                total += sum(eff * dists[(team, partner)].get(x, 0.0) for x, eff in row.items())
        return total if team == "ally" else -total

    def draft_win_prob(self, state: DraftState) -> float:
        """Probabilité de victoire de ton équipe pour la draft telle quelle (slots vides en espérance)."""
        return sigmoid(self.expected_logit(self._slot_dists(state), state.ally_side))

    # ── Recommandation ───────────────────────────────────────────────────────

    def recommend(
        self,
        state: DraftState,
        role: str,
        pool: Optional[set[int]] = None,
        min_games: int = 50,
        risk: float = 0.0,
        personal: Optional[dict[int, float]] = None,
    ) -> list[Recommendation]:
        """
        Classe les champions jouables au rôle `role` pour ton équipe.

        Args:
            pool: Si fourni, ne considérer que ces champions (ton champion pool).
            min_games: Nombre minimum de matchs RÉELS du champion à ce rôle.
            risk: 0 = trier sur la proba attendue ; 1 = trier sur la proba si
                contré ; entre les deux = mélange. Utile quand tu picks avant
                ton vis-à-vis.
            personal: Décalages de logit propres au joueur par champion
                (ml.personal), ajoutés au candidat. Le « pick moyen » de
                référence reste celui d'un joueur moyen.
        """
        state = DraftState(
            ally={r: c for r, c in state.ally.items() if r != role},
            enemy=dict(state.enemy), bans=set(state.bans), ally_side=state.ally_side,
        )
        dists = self._slot_dists(state)
        enemy_dist: Dist = dists[("enemy", role)]
        population: Dist = dists[("ally", role)]

        # B : draft sans le slot allié ni le slot ennemi du rôle
        base_dists = {**dists, ("ally", role): {}, ("enemy", role): {}}
        B = self.expected_logit(base_dists, state.ally_side, include_balance=False)

        # Termes de composition : sommes sur les 4 autres slots de chaque équipe.
        # L'équilibre de ton équipe ne dépend que de c (et des autres slots) : il va
        # entièrement dans A(c) ; celui de l'équipe adverse entièrement dans E(e).
        if self.damage:
            ad_a, th_a = self._team_sums(base_dists, "ally")
            ad_e, th_e = self._team_sums(base_dists, "enemy")
            mom_a, mom_e = self._ad_moments(base_dists, "ally"), self._ad_moments(base_dists, "enemy")
            k = 1 / (5 * self.damage.std)
            comp_a = lambda c: (self.comp.get(c, 0.0) * self._z(ad_e) - th_e * self._share(c) * k
                                + self._balance(mom_a[0] + self._share(c), mom_a[1]))
            comp_e = lambda e: (th_a * self._share(e) * k - self.comp.get(e, 0.0) * self._z(ad_a)
                                - self._balance(mom_e[0] + self._share(e), mom_e[1]))
            cross = lambda c, e: (self.comp.get(c, 0.0) * self._share(e) - self.comp.get(e, 0.0) * self._share(c)) * k
        else:
            k = 0.0
            comp_a = comp_e = lambda c: 0.0
            cross = lambda c, e: 0.0

        # E(e) pour chaque vis-à-vis possible
        E = {e: self._own_terms("enemy", role, e, dists) + comp_e(e) for e in enemy_dist}
        E_mean = sum(p * E[e] for e, p in enemy_dist.items())
        share_mean = sum(p * self._share(e) for e, p in enemy_dist.items())
        comp_mean = sum(p * self.comp.get(e, 0.0) for e, p in enemy_dist.items())
        responses = sorted(
            (e for e, p in enemy_dist.items() if p >= MIN_RESPONSE_SHARE),
            key=lambda e: -enemy_dist[e],
        )[:MAX_RESPONSES]
        lane_row = self.lane[role]

        def logit_expected(c: int) -> tuple[float, float]:
            """(logit attendu, terme de paire P(c, e) attendu) pour le candidat c."""
            A = self._own_terms("ally", role, c, dists) + comp_a(c)
            # Le vis-à-vis ne peut pas être c : on retire c de sa distribution
            pc = enemy_dist.get(c, 0.0)
            norm = 1.0 - pc
            if norm <= 0:
                return B + A, 0.0
            e_part = (E_mean - pc * E.get(c, 0.0)) / norm
            row = lane_row.get(c, {})
            l_part = sum(eff * enemy_dist.get(e, 0.0) for e, eff in row.items() if e != c) / norm
            sh = (share_mean - pc * self._share(c)) / norm
            th = (comp_mean - pc * self.comp.get(c, 0.0)) / norm
            x_part = (self.comp.get(c, 0.0) * sh - th * self._share(c)) * k
            return B + A + e_part + l_part + x_part, l_part + x_part

        avg_logit = sum(p * logit_expected(c)[0] for c, p in population.items())

        games = self.model.games[role]
        unavailable = state.unavailable()
        candidates = [
            int(c) for c, n in games.items()
            if n >= min_games and int(c) not in unavailable and (pool is None or int(c) in pool)
        ]

        results = []
        for c in candidates:
            exp_logit, expected_pair = logit_expected(c)
            delta = (personal or {}).get(c, 0.0)
            exp_logit += delta
            row = lane_row.get(c, {})
            worst_pair, worst_e = expected_pair, None
            for e in responses:
                pair = row.get(e, 0.0) + cross(c, e)
                if e != c and pair < worst_pair:
                    worst_pair, worst_e = pair, e
            worst_logit = exp_logit - expected_pair + worst_pair
            score = (1 - risk) * exp_logit + risk * worst_logit
            results.append(Recommendation(
                champion_id=c,
                name=self.names.get(c, str(c)),
                games=games[str(c)],
                win_prob=sigmoid(exp_logit),
                vs_average=sigmoid(exp_logit) - sigmoid(avg_logit),
                win_prob_if_countered=sigmoid(worst_logit),
                worst_response=self.names.get(worst_e) if worst_e is not None else None,
                worst_response_id=worst_e,
                score=score,
                personal_effect=sigmoid(exp_logit) - sigmoid(exp_logit - delta),
            ))
        results.sort(key=lambda r: -r.score)
        return results


# ── CLI ───────────────────────────────────────────────────────────────────────

def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def resolve_champion(name: str, names: dict[int, str]) -> int:
    lookup = {_normalize(n): cid for cid, n in names.items()}
    key = _normalize(name)
    if key in lookup:
        return lookup[key]
    matches = [cid for n, cid in lookup.items() if n.startswith(key)]
    if len(matches) == 1:
        return matches[0]
    raise SystemExit(f"Champion inconnu ou ambigu : {name!r}")


def parse_slots(items: list[str], names: dict[int, str]) -> dict[str, int]:
    slots = {}
    for item in items:
        role, _, champ = item.partition("=")
        if role.upper() not in ROLE_ALIASES or not champ:
            raise SystemExit(f"Format attendu ROLE=Champion, reçu : {item!r}")
        slots[ROLE_ALIASES[role.upper()]] = resolve_champion(champ, names)
    return slots


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser(description="Recommandation de pick (modèle additif).")
    p.add_argument("--role", required=True, type=lambda r: ROLE_ALIASES[r.upper()],
                   help="Rôle à pourvoir (TOP, JUNGLE, MID, ADC, SUPPORT).")
    p.add_argument("--ally", nargs="*", default=[], metavar="ROLE=Champion")
    p.add_argument("--enemy", nargs="*", default=[], metavar="ROLE=Champion")
    p.add_argument("--bans", nargs="*", default=[], metavar="Champion")
    p.add_argument("--pool", nargs="*", default=None, metavar="Champion",
                   help="Limiter aux champions que tu joues.")
    p.add_argument("--side", choices=["blue", "red"], default=None)
    p.add_argument("--min-games", type=int, default=50)
    p.add_argument("--risk", type=float, default=0.0,
                   help="0 = proba attendue, 1 = proba si contré.")
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--model", default=str(DATA_DIR / "additive_model.json"))
    args = p.parse_args()

    model = AdditiveDraftModel.load(Path(args.model))
    rec = DraftRecommender(model)
    state = DraftState(
        ally=parse_slots(args.ally, rec.names),
        enemy=parse_slots(args.enemy, rec.names),
        bans={resolve_champion(b, rec.names) for b in args.bans},
        ally_side=args.side,
    )
    pool = {resolve_champion(c, rec.names) for c in args.pool} if args.pool else None

    results = rec.recommend(state, args.role, pool=pool, min_games=args.min_games, risk=args.risk)
    print(f"\nRôle {args.role} — {len(results)} candidats (≥ {args.min_games} matchs au rôle)\n")
    print(f"{'#':>3} {'Champion':<14} {'Victoire':>8} {'vs moyen':>10} {'Si contré':>10}  {'par':<12} {'Matchs':>6}")
    for i, r in enumerate(results[:args.top], 1):
        print(f"{i:>3} {r.name:<14} {r.win_prob:>8.1%} {r.vs_average * 100:>+6.1f} pts {r.win_prob_if_countered:>10.1%}  "
              f"{(r.worst_response or '—'):<12} {r.games:>6}")


if __name__ == "__main__":
    main()
