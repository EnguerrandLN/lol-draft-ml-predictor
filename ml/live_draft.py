"""
live_draft.py — De la sélection de champions du client du jeu à une draft exploitable.

Le client (api/lcu.py) donne, pour la partie en cours : le côté et le rôle du
joueur, les picks verrouillés de son équipe avec leur rôle, les picks
verrouillés adverses SANS leur rôle, et les bans. Deux étapes :

  1. parse_champ_select : ne garde que ce qui est acquis. Un champion seulement
     survolé (pas encore verrouillé) peut encore changer : il est ignoré.
  2. assign_roles : attribue un rôle aux champions adverses. Chaque champion a
     une répartition connue de ses parties entre les rôles (Jinx : ADC à
     ~100 %) ; on retient l'attribution globale la plus probable, un champion
     par rôle. Avec 5 champions il y a 120 attributions : on les énumère.
"""
import itertools
import math
from dataclasses import dataclass, field
from typing import Optional

from ml.draft_data import ROLES

CLIENT_ROLES: dict[str, str] = {
    "top": "TOP", "jungle": "JUNGLE", "middle": "MIDDLE", "bottom": "BOTTOM", "utility": "UTILITY",
}


@dataclass
class LiveDraft:
    side: Optional[str] = None                 # "blue" / "red"
    role: Optional[str] = None                 # Rôle attribué au joueur (None hors partie classée)
    ally: dict[str, int] = field(default_factory=dict)        # Rôle → champion verrouillé (coéquipiers)
    ally_unassigned: list[int] = field(default_factory=list)  # Coéquipiers verrouillés sans rôle attribué
    enemy: list[int] = field(default_factory=list)            # Champions adverses verrouillés (rôle inconnu)
    bans: set[int] = field(default_factory=set)
    my_champion: Optional[int] = None          # Pick du joueur s'il est déjà verrouillé
    enemy_picks_left: int = 0                  # Picks adverses encore à venir

    def signature(self) -> tuple:
        """Change dès que la draft change : sert à ne mettre l'affichage à jour qu'à ce moment."""
        return (self.side, self.role, tuple(sorted(self.ally.items())), tuple(sorted(self.ally_unassigned)),
                tuple(sorted(self.enemy)), tuple(sorted(self.bans)), self.my_champion, self.enemy_picks_left)


def parse_champ_select(session: dict) -> LiveDraft:
    """Session `/lol-champ-select/v1/session` du client → LiveDraft."""
    actions = [a for group in session.get("actions", []) for a in group]
    done = [a for a in actions if a.get("completed") and a.get("championId", 0) > 0]
    locked = {a["actorCellId"]: a["championId"] for a in done if a.get("type") == "pick"}
    my_cell = session.get("localPlayerCellId")
    my_team, their_team = session.get("myTeam", []), session.get("theirTeam", [])
    their_cells = {p.get("cellId") for p in their_team}

    live = LiveDraft(bans={a["championId"] for a in done if a.get("type") == "ban"})
    for player in my_team:
        cell, role = player.get("cellId"), CLIENT_ROLES.get(player.get("assignedPosition") or "")
        if cell == my_cell:
            live.role = role
            live.side = {1: "blue", 2: "red"}.get(player.get("team")) or ("blue" if (cell or 0) < 5 else "red")
            live.my_champion = locked.get(cell)
        elif cell in locked:
            if role:
                live.ally[role] = locked[cell]
            else:
                live.ally_unassigned.append(locked[cell])
    live.enemy = [locked[c] for c in sorted(their_cells) if c in locked]
    live.enemy_picks_left = sum(
        1 for a in actions
        if a.get("type") == "pick" and not a.get("completed") and a.get("actorCellId") in their_cells
    )
    return live


def assign_roles(
    champions: list[int], games: dict[str, dict[str, int]], roles: tuple[str, ...] = ROLES,
) -> dict[str, int]:
    """
    Attribution la plus probable d'un rôle à chaque champion (un champion par rôle).

    Args:
        champions: Champions à placer (au plus autant que de rôles).
        games: Rôle → champion → nombre de parties (statistiques du modèle).
        roles: Rôles encore libres.

    Returns:
        Rôle → champion, pour les champions donnés.
    """
    champions = list(champions)[:len(roles)]
    if not champions:
        return {}

    def log_share(champ: int, role: str) -> float:
        """log P(rôle | champion), lissé : un champion inconnu est réparti uniformément."""
        total = sum(games.get(r, {}).get(str(champ), 0) for r in ROLES)
        return math.log((games.get(role, {}).get(str(champ), 0) + 1) / (total + len(ROLES)))

    table = {(c, r): log_share(c, r) for c in champions for r in roles}
    best = max(itertools.permutations(roles, len(champions)),
               key=lambda perm: sum(table[(c, r)] for c, r in zip(champions, perm)))
    return dict(zip(best, champions))


def resolve(
    live: LiveDraft, games: dict[str, dict[str, int]], enemy_pinned: Optional[dict[int, str]] = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Équipes complètes par rôle : (alliés, adversaires). Les coéquipiers sans
    rôle attribué (files non classées) et les adversaires sont placés par
    assign_roles ; le rôle du joueur reste libre pour lui.

    Args:
        enemy_pinned: Champion adverse → rôle imposé (correction manuelle du
            joueur) ; les autres adversaires se répartissent les rôles restants.
    """
    ally = dict(live.ally)
    free = tuple(r for r in ROLES if r not in ally and r != live.role)
    ally.update(assign_roles(live.ally_unassigned, games, free))

    enemy: dict[str, int] = {}
    for champ in live.enemy:
        role = (enemy_pinned or {}).get(champ)
        if role in ROLES and role not in enemy:
            enemy[role] = champ
    rest = [c for c in live.enemy if c not in enemy.values()]
    enemy.update(assign_roles(rest, games, tuple(r for r in ROLES if r not in enemy)))
    return ally, enemy
