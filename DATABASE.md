# Base de données des matchs League of Legends

Base SQLite de parties **Ranked Solo/Duo (queue 420)**, serveur **EUW**,
collectée via l'API Riot (Match-V5 et League-V4) par le crawler du projet
`lol-draft-ml-predictor`. Le crawler tourne en continu et ajoute environ
2 600 matchs par heure.

État au 02/10/2026 : **~115 000 matchs**, 1,15 million de lignes joueurs,
~800 Mo. Parties d'octobre 2024 à octobre 2026, dont l'essentiel sur les
patchs 16.16 à 16.19.

## Accès

Fichier : `data/draft.db` dans le projet du crawler. La base est en mode
**WAL** : le crawler peut écrire pendant que d'autres programmes lisent.

**Lecture directe (recommandé si les deux projets sont sur la même machine)**,
en lecture seule pour ne jamais gêner ni abîmer la collecte :

```python
import sqlite3
DB = r"C:\Users\engue\Desktop\Projet Perso\lol-draft-ml-predictor\data\draft.db"
conn = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
```

**Copie figée** : ne pas copier `draft.db` seul pendant que le crawler tourne,
car les écritures récentes sont encore dans `draft.db-wal`. Utiliser la
sauvegarde SQLite, qui produit une copie cohérente :

```python
import sqlite3
src = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
dst = sqlite3.connect("draft_copy.db")
src.backup(dst)
dst.close(); src.close()
```

## Tables

### `matches` — une ligne par partie

| Colonne | Type | Signification |
|---|---|---|
| `match_id` | TEXT, clé | Identifiant Riot, ex. `EUW1_7993746972` |
| `game_version` | TEXT | Version complète, ex. `16.19.823.722` (patch = 2 premiers nombres : `16.19`) |
| `queue_id` | INTEGER | Toujours 420 (Ranked Solo/Duo) |
| `game_duration` | INTEGER | Durée en **secondes** |
| `platform_id` | TEXT | `EUW1` (quelques dizaines de matchs EUN1, TR1, RU, ME1 issus d'anciens crawls) |
| `game_creation` | INTEGER | Date de début, timestamp Unix en **millisecondes** |
| `winning_team` | INTEGER | 100 (bleu) ou 200 (rouge) ; NULL pour 10 parties sans vainqueur |
| `crawled_at` | TIMESTAMP | Date de collecte (UTC) |
| `ended_early` | INTEGER | 1 = **remake** uniquement (partie annulée vers 3 min à cause d'un absent) ; le résultat ne dépend pas de la draft. Les FF (abandon à 15/20 min) ne sont **pas** signalées : ce sont de vraies parties |
| `source_tier` | TEXT | Tier du joueur du ladder par lequel le match a été trouvé (GOLD … CHALLENGER) |
| `source_division` | TEXT | Division de ce joueur (I à IV ; I pour Master et au-dessus) |

### `participants` — une ligne par joueur et par partie (10 par match)

| Colonne | Signification |
|---|---|
| `id` | Clé technique |
| `match_id` | Partie (→ `matches`) |
| `puuid` | Identifiant Riot permanent du joueur. Un même joueur apparaît dans de nombreuses parties |
| `team_id` | 100 = bleu, 200 = rouge |
| `position` | Rôle : `TOP`, `JUNGLE`, `MIDDLE`, `BOTTOM`, `UTILITY` (support) ; vide pour 187 lignes |
| `lane`, `role` | Anciens champs Riot, peu fiables : utiliser `position` |
| `champion_id`, `champion_name` | Champion (id numérique Riot, nom interne : `MonkeyKing` pour Wukong, `KSante`…) |
| `win` | 1 = victoire, 0 = défaite |
| `kills`, `deaths`, `assists` | KDA |
| `total_minions_killed`, `gold_earned` | Farm et or |
| `total_damage_dealt_to_champions` | Dégâts aux champions, détaillés en `physical_…`, `magic_…`, `true_…` |
| `total_damage_taken` | Dégâts reçus |
| `damage_self_mitigated` | Dégâts absorbés (armure, résistance magique, boucliers sur soi) |
| `time_ccing_others` | Score Riot de contrôle infligé |
| `total_time_cc_dealt` | Durée cumulée des contrôles infligés (secondes) |
| `total_heal`, `total_heals_on_teammates` | Soins (total, et sur les alliés) |
| `total_damage_shielded_on_teammates` | Boucliers posés sur les alliés |
| `vision_score`, `wards_placed` | Vision |
| `items` | Objets en fin de partie : JSON de 7 ids, `[item0, …, item5, trinket]`, 0 = emplacement vide |
| `summoner_name`, `summoner_id` | Obsolètes côté Riot : presque toujours vides |

Contrainte : `(match_id, puuid)` est unique.

### `bans` — 10 lignes par partie

| Colonne | Signification |
|---|---|
| `match_id`, `team_id` | Partie et équipe qui bannit |
| `pick_turn` | Ordre du ban : 1 à 5 pour l'équipe 100, 6 à 10 pour l'équipe 200 |
| `champion_id` | Champion banni ; **-1 = pas de ban** |

### `ladder_players` — file de travail du crawler

Joueurs échantillonnés dans le classement (environ 600 par division, de Gold
à Challenger) : `puuid`, `tier`, `division`, `league_points`, `status`
(`pending` / `done`), `priority` (ordre aléatoire), `last_crawled_at` (epoch
en secondes). Utile pour connaître le rang de ces joueurs ; à ne pas modifier.

### `timelines` et `timeline_frames` — état des joueurs en cours de partie (échantillon)

Remplies par `fetch_timelines.py` pour un **échantillon** de matchs (pas
tous) : une requête d'API de plus par match, via l'endpoint *timeline* de
Match-V5.

- `timelines` : une ligne par match demandé (`match_id`, `frame_count` =
  nombre d'images d'une minute, **0 = timeline introuvable**, `fetched_at`).
- `timeline_frames` : une ligne par joueur et par minute retenue (**10, 15
  et 20** ; une minute après la fin de la partie est absente). Colonnes :
  `match_id`, `puuid`, `minute`, `total_gold`, `xp`, `level`,
  `minions_killed`, `jungle_minions_killed`, `damage_to_champions`
  (cumulés depuis le début de la partie).

Jointure avec les joueurs par `(match_id, puuid)` :

```sql
SELECT p.team_id, p.position, p.champion_name, f.total_gold
FROM timeline_frames f
JOIN participants p ON p.match_id = f.match_id AND p.puuid = f.puuid
WHERE f.minute = 15;
```

### Tables héritées

`crawl_queue` (ancien crawl en largeur, 285k puuids) et `summoner_cache`
(vide) ne sont plus utilisées.

## Pièges à connaître

- **Remakes** : exclure `ended_early = 1` (~2 % des parties) pour toute analyse
  de résultat. Pour les parties collectées avant octobre 2026, ce champ a été
  déduit d'une durée inférieure à 5 minutes.
- **FF (abandons)** : pas de colonne dédiée. Ce sont de vraies parties, perdues
  par l'équipe qui abandonne : les garder (les exclure retirerait surtout les
  parties déséquilibrées et fausserait les winrates). Elles se voient dans la
  distribution des durées (pics à 15 et 20 minutes), mais une durée courte ne
  suffit pas à identifier une FF de façon fiable.
- **Rang** : `source_tier` est le rang d'**un** joueur de la partie (celui par
  lequel elle a été trouvée). Le matchmaking réunit des joueurs de niveau
  proche, donc c'est une bonne approximation du niveau du match, mais pas le
  rang des 10 joueurs. Il est **NULL pour ~53 000 parties** collectées avant
  l'échantillonnage par ladder (niveau inconnu, probablement Gold-Émeraude).
- **Nouveaux champs** (`damage_self_mitigated`, `time_ccing_others`,
  `total_time_cc_dealt`, `total_heal`, `total_heals_on_teammates`,
  `total_damage_shielded_on_teammates`) : **NULL pour les parties collectées
  avant le 02/10/2026** (NULL = inconnu, ce n'est pas un vrai zéro).
- **Rôles** : se fier à `position`. Quelques lignes ont une position vide.
- **Statistiques de fin de partie** (dégâts, or, objets…) : elles dépendent du
  déroulement et de la durée. Les normaliser par minute, et ne pas les utiliser
  comme information « avant la partie ».
- **Patchs** : l'équilibrage change à chaque patch (toutes les 2 semaines).
  Filtrer ou pondérer par `game_version` selon l'usage.
- **Vainqueur** : `winning_team` et `participants.win` concordent ; 10 parties
  n'ont pas de vainqueur (NULL).
- **Noms de champions** : ce sont les noms internes de Riot. Les noms affichés
  (Wukong, K'Santé…) et les icônes sont sur Data Dragon :
  `https://ddragon.leagueoflegends.com/cdn/<version>/data/fr_FR/champion.json`.

## Requêtes utiles

**Drafts complètes, une ligne par match** (10 champions et vainqueur, sans remakes) :

```sql
SELECT m.match_id, m.game_creation, m.game_version, m.source_tier,
       MAX(CASE WHEN p.team_id = 100 AND p.position = 'TOP'     THEN p.champion_id END) AS blue_top,
       MAX(CASE WHEN p.team_id = 100 AND p.position = 'JUNGLE'  THEN p.champion_id END) AS blue_jungle,
       MAX(CASE WHEN p.team_id = 100 AND p.position = 'MIDDLE'  THEN p.champion_id END) AS blue_mid,
       MAX(CASE WHEN p.team_id = 100 AND p.position = 'BOTTOM'  THEN p.champion_id END) AS blue_adc,
       MAX(CASE WHEN p.team_id = 100 AND p.position = 'UTILITY' THEN p.champion_id END) AS blue_support,
       MAX(CASE WHEN p.team_id = 200 AND p.position = 'TOP'     THEN p.champion_id END) AS red_top,
       MAX(CASE WHEN p.team_id = 200 AND p.position = 'JUNGLE'  THEN p.champion_id END) AS red_jungle,
       MAX(CASE WHEN p.team_id = 200 AND p.position = 'MIDDLE'  THEN p.champion_id END) AS red_mid,
       MAX(CASE WHEN p.team_id = 200 AND p.position = 'BOTTOM'  THEN p.champion_id END) AS red_adc,
       MAX(CASE WHEN p.team_id = 200 AND p.position = 'UTILITY' THEN p.champion_id END) AS red_support,
       m.winning_team = 100 AS blue_win
FROM matches m JOIN participants p USING (match_id)
WHERE m.ended_early = 0 AND m.winning_team IS NOT NULL
GROUP BY m.match_id;
```

**Winrate par champion et par rôle sur un patch** :

```sql
SELECT p.champion_name, p.position, COUNT(*) AS games, AVG(p.win) AS winrate
FROM participants p JOIN matches m USING (match_id)
WHERE m.ended_early = 0 AND m.game_version LIKE '16.19.%'
GROUP BY p.champion_name, p.position
HAVING games >= 200
ORDER BY winrate DESC;
```

**Historique d'un joueur** (via son puuid) :

```sql
SELECT m.game_creation, p.champion_name, p.position, p.win, p.kills, p.deaths, p.assists
FROM participants p JOIN matches m USING (match_id)
WHERE p.puuid = ? ORDER BY m.game_creation DESC;
```

Avec pandas : `pd.read_sql_query(requête, conn)`.

## Confidentialité et conditions d'utilisation

Les données viennent de l'API Riot et sont soumises à ses conditions
d'utilisation. Les `puuid` identifient des joueurs réels : ne pas publier la
base ni la versionner dans un dépôt Git public (le projet d'origine l'exclut
via `.gitignore` : `*.db`, `*.db-wal`, `*.db-shm`).
