# LoL Draft Advisor

Outil d'aide à la draft pour League of Legends : à n'importe quel moment de la
sélection des champions (first pick, last pick ou entre les deux), il classe
les champions jouables à ton rôle selon la probabilité de victoire qu'ils
donnent à ton équipe, compte tenu de ce qui est déjà connu de la draft.

Les données viennent de l'API Riot (Ranked Solo/Duo, EUW) ; le modèle est un
modèle additif régularisé, évalué sur des parties récentes jamais vues.

## Ce que le modèle sait faire (et pas encore)

Évaluation sur les 15 % de parties les plus récentes (~8 000 matchs) :

| Modèle | Log-loss | Précision |
|---|---|---|
| Constante (toujours ~50 %) | 0,6932 | 50 % |
| Force champion × rôle seule | 0,6885 | 54,2 % ± 1,1 |
| **Modèle complet** | **0,6876** | 54,2 % ± 1,1 |

- La draft seule explique peu du résultat en Ranked Solo (le reste, ce sont les
  joueurs) : ~54 % est proche du plafond connu pour ce type de prédiction. Un
  bon pick apporte typiquement 1 à 4 points de winrate.
- **Bien estimé** : la force de chaque champion à chaque rôle, et l'équilibre
  des dégâts d'une équipe (une composition full AD ou full AP perd ~7 points).
- **Encore faible** : les matchups de lane et les synergies (trop peu de
  parties par paire de champions), et les sensibilités propres à un champion
  (« Malphite contre une équipe AD ») : seules celles qui sont statistiquement
  établies sont retenues, et la liste s'allonge avec les données.

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
  recommend.py       Recommandation sur une draft partielle (+ CLI)
  personal.py        Personnalisation par l'historique du joueur
  display_names.py   Noms et icônes des champions (Data Dragon)
data/
  additive_model.json  Modèle entraîné (versionné : l'app fonctionne sans la base)
  draft.db             Base SQLite des matchs (non versionnée)
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

Renseigne les picks des deux équipes, les bans et le rôle à pourvoir. Dans la
barre latérale, ton Riot ID importe ton historique : les champions que tu joues
reçoivent un bonus ou malus personnel (fortement atténué tant que tu as peu de
parties), et tu peux te limiter à ton pool.

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
nouveau modèle automatiquement. À relancer régulièrement pendant le crawl.

**Recommandation en ligne de commande**

```bash
python ml/recommend.py --role ADC --ally SUPPORT=Braum --enemy MID=Zed --bans Jinx --side red
```

**Tests**

```bash
python -m pytest
```

## Le modèle en bref

```
logit P(victoire) = avantage de côté
                  + force champion × rôle          (par équipe)
                  + matchups de lane                (atténués)
                  + synergies bot / jungle-mid / jungle-top
                  + équilibre des dégâts de l'équipe
                  + sensibilité de certains champions au profil adverse
```

Chaque groupe d'effets a sa propre force de régularisation, choisie par
validation : les effets peu mesurables sont automatiquement ramenés vers zéro
et gagnent en poids à mesure que les données s'accumulent. Pour une draft
partielle, chaque slot encore vide est remplacé par la distribution des picks
habituels du rôle ; le modèle étant additif, l'espérance de victoire est
calculée exactement.
