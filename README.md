# LoL Draft Advisor

Outil d'aide à la draft pour League of Legends : à n'importe quel moment de la
sélection des champions (first pick, last pick ou entre les deux), il classe
les champions jouables à ton rôle selon la probabilité de victoire qu'ils
donnent à ton équipe, compte tenu de ce qui est déjà connu de la draft.

Les données viennent de l'API Riot (Ranked Solo/Duo, EUW) ; le modèle est un
modèle additif régularisé, évalué sur des parties récentes jamais vues.

## Ce que le modèle sait faire (et pas encore)

Évaluation sur les 15 % de parties les plus récentes (~23 000 matchs, 152k au total) :

| Modèle | Log-loss | Précision |
|---|---|---|
| Constante (toujours ~50 %) | 0,6931 | 50 % |
| Force champion × rôle seule | 0,6901 | 53,3 % ± 0,6 |
| Modèle sans matchups appris sur l'or | 0,6886 | 54,1 % ± 0,6 |
| **Modèle complet** | **0,6876** | **54,4 % ± 0,6** |

- La draft seule explique peu du résultat en Ranked Solo (le reste, ce sont les
  joueurs) : ~54 % est proche du plafond connu pour ce type de prédiction. Un
  bon pick apporte typiquement 1 à 4 points de winrate. Les probabilités sont
  bien calibrées (prédit 41 % → observé 42 %, prédit 59 % → observé 58 %).
- **Bien estimé** : la force de chaque champion à chaque rôle, l'équilibre des
  dégâts d'une équipe (une composition full AD ou full AP perd ~7 points), les
  forces propres à chaque niveau (Gold-Platine, Émeraude-Diamant, Master+) et
  les synergies des duos fréquents (bot, jungle-mid, jungle-top).
- **Sensibilités propres à un champion** (« Malphite contre une équipe AD »,
  « Galio contre une équipe AP ») : pente de chaque champion pondérée par la
  force des preuves et leur stabilité dans le temps (lissage bayésien empirique,
  sans seuil ni réglage par champion).
- **Coût d'inexpérience** : la force d'un champion est mesurée sur ceux qui le
  choisissent. Pour quelqu'un qui ne l'a jamais joué, ~−1,3 pt sur un pick
  courant et ~−3 pts sur un pick de niche (rôle < 10 % des parties du champion).
  Appliqué automatiquement avec ton profil, affiché sinon (« Si jamais joué »).
- **Selon le niveau** : en Master+, la draft pèse moins (facteur 0,76) ; l'app
  applique les forces et la calibration de ton niveau, détecté depuis ton Riot ID.
- **Counters appris sur les statistiques de lane** : le résultat d'une partie
  (1 bit) ne suffit pas à estimer ~15 000 matchups par rôle. Ils sont appris
  sur un signal continu, l'écart de part d'or entre les deux laners (tâche
  auxiliaire), puis transférés au modèle de victoire (cross-fitting). Gain
  significatif sur des parties futures ; le modèle retrouve des counters connus
  (Sett et Warwick contre Irelia, Malphite et Teemo contre Vayne top…).
- **Pas d'interaction « par ressemblance »** : synergies et counters déduits
  d'embeddings de champions (factorization machine) testés jusqu'à 150k matchs
  sans gain, donc pas de réseau de neurones entraîné sur la victoire.

Détail des expériences et de l'audit : [`experiments/RESULTS.md`](experiments/RESULTS.md).
Description de la base de données : [`DATABASE.md`](DATABASE.md).

## Structure

```
app.py               Application Streamlit (conseiller de draft)
crawl.py             Collecte des matchs (crawler par ladder, tourne en continu)
fetch_player.py      Import de l'historique complet d'un joueur
config.py            Routing API, chemins, paramètres du crawler
api/
  client.py          Client HTTP Riot (rate limiting, retries, erreurs de clé)
  crawler.py         Échantillonnage du ladder et extraction des matchs
db/
  schema.py          Schéma SQLite (matches, participants, bans, ladder_players)
  repository.py      Accès aux données
ml/
  draft_data.py      Chargement des drafts (une ligne par match réel)
  additive_model.py  Modèle additif et sélection des sensibilités par champion
  train.py           Réglage, évaluation temporelle, export du modèle
  evaluate.py        Évaluation scellée d'un modèle sur une période jamais vue
  recommend.py       Recommandation sur une draft partielle (+ CLI)
  personal.py        Personnalisation par l'historique du joueur
  lane_stats.py      Matchups appris sur la part d'or des laners (tâche auxiliaire)
  display_names.py   Noms et icônes des champions (Data Dragon)
data/
  additive_model.json  Modèle entraîné (versionné : l'app fonctionne sans la base)
  draft.db             Base SQLite des matchs (non versionnée)
experiments/         Expériences reproductibles et leurs résultats (RESULTS.md)
tests/               Tests unitaires (pytest)
```

## Installation

```bash
pip install -r requirements.txt
```

Crée un fichier `.env` à la racine avec ta clé API Riot
([developer.riotgames.com](https://developer.riotgames.com)) :

```
RIOT_API_KEY=RGAPI-...
```

## Utilisation

**Conseiller de draft**

```bash
python -m streamlit run app.py
```

Renseigne les picks des deux équipes, les bans et le rôle à pourvoir. Ton
Riot ID importe ton historique et ton rang (le niveau est alors réglé
automatiquement) : chaque champion reçoit un décalage personnel, coût
d'inexpérience compris pour ceux que tu ne joues pas, et tu peux te limiter à
ton pool. Sans profil, précise ton niveau : la base sur-représente le haut ELO.

**Collecte des matchs**

```bash
python crawl.py
```

Échantillonne ~600 joueurs par division (Gold → Challenger par défaut) et
collecte leurs parties récentes ; chaque match est étiqueté avec le rang du
joueur source. Le crawl tourne par passes successives et reprend où il s'est
arrêté. Si la clé expire (24 h pour une clé de développement), il se met en
pause : colle la nouvelle clé dans `.env`, il repart seul. Débit plafonné par
la clé : ~2 500 matchs/heure.

**Entraînement**

```bash
python ml/train.py
python ml/train.py --min-tier EMERALD   # seulement les matchs Émeraude et plus
```

Règle les hyperparamètres par validation temporelle glissante, affiche les
performances sur la période la plus récente (avec intervalles de confiance et
calibration), puis exporte `data/additive_model.json`. L'app recharge le
nouveau modèle automatiquement. À relancer régulièrement pendant le crawl
(~20 min à 150k matchs).

**Évaluation scellée** : les fenêtres de test de `train.py` ont servi aux choix
de conception, leurs chiffres sont donc un peu optimistes. Pour une mesure
honnête, entraîner jusqu'à une date puis évaluer une seule fois après :

```bash
python ml/train.py --until 2026-10-08 --out data/model_sealed.json
python ml/evaluate.py --model data/model_sealed.json --since 2026-10-08
```

**Recommandation en ligne de commande**

```bash
python ml/recommend.py --role ADC --ally SUPPORT=Braum --enemy MID=Zed --bans Jinx --side red --tier MID
python ml/recommend.py --role TOP --tier MID --jamais-joue    # avec le coût d'inexpérience
```

**Tests**

```bash
python -m pytest
```

## Le modèle en bref

```
logit P(victoire) = avantage de côté
                  + force champion × rôle          (par équipe)
                  + matchups de lane                (atténués + appris sur la part d'or)
                  + synergies bot / jungle-mid / jungle-top
                  + équilibre des dégâts de l'équipe
                  + sensibilité de certains champions au profil adverse
                  + écarts de force par tranche d'ELO      (quand les données les établissent)
```

Chaque groupe d'effets a sa propre force de régularisation, choisie par
validation : les effets peu mesurables sont automatiquement ramenés vers zéro
et gagnent en poids à mesure que les données s'accumulent. Pour une draft
partielle, chaque slot encore vide est remplacé par la distribution des picks
habituels du rôle ; le modèle étant additif, l'espérance de victoire est
calculée exactement.
