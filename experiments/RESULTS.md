# Résultats des expériences

Synthèse des expériences menées sur le modèle additif (octobre 2026, base de
~58k matchs Ranked Solo EUW). Chaque script de ce dossier est reproductible :
`python experiments/<script>.py`.

## 1. Courbe d'apprentissage — combien de matchs faut-il ? (`learning_curve.py`)

Test fixe (15 % de matchs les plus récents), sous-échantillons croissants du
reste, 3 tirages par taille, régularisation re-réglée à chaque taille. Gain de
log-loss par rapport à une prédiction constante, composant par composant :

| Matchs d'entraînement | Total | Force champion × rôle | Équilibre AD/AP | Matchups de lane | Synergies | p-value Malphite |
|---|---|---|---|---|---|---|
| 8 000 | +0,0017 | +0,0012 | +0,0005 | 0 | 0 | 0,38 |
| 15 000 | +0,0022 | +0,0017 | +0,0005 | 0 | 0 | 0,28 |
| 25 000 | +0,0028 | +0,0023 | +0,0005 | 0 | 0 | 0,066 |
| 35 000 | +0,0034 | +0,0030 | +0,0005 | −0,0002 | 0 | 0,016 |
| 48 600 | +0,0038 | +0,0035 | +0,0005 | −0,0002 | 0 | 0,005 |

- **La force champion × rôle n'est pas saturée** : ~+0,0009 de gain par
  doublement des données. Continuer le crawl améliore directement le classement.
- **L'équilibre AD/AP est saturé dès 8k matchs** (2 paramètres).
- **Matchups de lane et synergies : aucun gain mesurable jusqu'à 50k.** Médiane
  de ~3 parties par paire de lane : il faudra un ordre de grandeur de plus.
- **Sensibilités par champion** : la p-value de Malphite décroît comme prévu
  (∝ √N). (Mesure faite avec l'ancienne sélection en tout ou rien ; remplacée
  depuis par le lissage de la section 5.)

Détail brut : `results/learning_curve.csv`.

## 2. Le modèle selon le niveau (calibration par tranche, dans `ml/train.py`)

Prédictions hors échantillon (validation croisée 5 folds) sur les matchs
étiquetés par le crawler, pente de calibration par tranche (1 = effets de
draft bien dosés à ce niveau) :

| Tranche | Matchs | Pente brute | Facteur retenu (atténué vers 1) |
|---|---|---|---|
| Gold-Platine et moins | ~2 000 | 0,85 ± 0,42 | 0,87 |
| Émeraude-Diamant | ~2 800 | 0,98 ± 0,37 | 0,99 |
| Master+ | ~1 600 | 0,41 ± 0,49 | 0,53 |

- **En Master+, les effets de draft appris sont environ deux fois trop forts.**
  Soit la draft y pèse moins, soit d'autres champions y sont forts. L'app et la
  CLI appliquent ce facteur quand on choisit son niveau (`--tier HIGH`) :
  les probabilités affichées restent honnêtes.
- **Écarts de force champion × rôle par tranche** : implémentés (groupe
  « tier »), mais pas encore retenus par la validation (gain < 0,00005).
  Ils s'activeront d'eux-mêmes avec plus de matchs étiquetés : tous les
  nouveaux matchs du crawler le sont.

## 3. Équilibre « frontline » (`frontline_balance.py`) — négatif

Indice = part moyenne des dégâts encaissés par le champion dans son équipe.
Winrate plat selon la frontline de l'équipe (49,6-50,7 % sur les déciles) ;
2 champions seulement significatifs côté « sensibilité aux tanks adverses »
(Maokai, Nautilus). Non intégré. À retester avec une meilleure mesure
(dégâts atténués, temps de CC : non collectés aujourd'hui).

## 4. Coût d'un champion hors de son pool (`champion_familiarity.py`)

À joueur égal et en ne comptant que les parties **antérieures** (sinon biais de
survie : la mesure naïve donnait −5,9 pts) :

| Parties antérieures sur ce champion | Écart à la prédiction, à joueur égal |
|---|---|
| Aucune dans l'historique récent | −0,7 pt (± 1,0) |
| 6 et plus | +0,7 pt (± 0,9) |

Un champion absent de ses ~20 dernières parties coûte ~1 à 1,5 pt par rapport
à son main. Le gros écart brut (~4 pts) est un écart entre joueurs (un pool
stable va avec un meilleur niveau), pas un effet du choix de champion.

## 5. Sensibilités par champion : tout ou rien vs lissage (`comp_effects_validation.py`)

Problème de l'ancienne méthode (sélection Benjamini-Hochberg, effets retenus
entiers) : falaise entre retenu et non retenu (Malphite 100 %, Galio 0 % à
p = 0,02), et un seul effet retenu peut dominer les classements. Pyke, retenu
à p = 0,001, faisait passer Zed de 6ᵉ à 1ᵉʳ en mid dès qu'il était en face ;
estimé sur la période ancienne, son effet **ne se reproduisait pas** sur les
parties récentes (−0,0009 de log-loss sur ses parties).

Nouvelle méthode (`fit_comp_effects`), entièrement calculée :
  1. pentes par période (4 blocs) combinées en méta-analyse à effets aléatoires :
     la dérive entre périodes (ω) est mesurée sur tous les champions ;
  2. mélange bayésien empirique « pas d'effet / vrai effet » dont la proportion
     (π ≈ 23 %) et l'ampleur (τ ≈ 0,034) sont estimées sur les 173 champions ;
  3. effet = pente × part retenue (probabilité d'effet réel × fiabilité).

Validation sur parties futures (gain de log-loss vs sans sensibilités) :

| | Fenêtre récente | Fenêtre précédente |
|---|---|---|
| Tout ou rien | +0,0002 ± 0,0002 | 0 (rien de retenu à 52k) |
| Lissage | +0,0002 ± 0,0002 | −0,0001 ± 0,0003 |

Les deux sont au niveau du bruit en prédiction ; le lissage est retenu pour son
comportement (pas de falaise, effets instables neutralisés). L'entraînement
décide seul de l'adopter : il ne l'est que si la validation glissante est
positive (aujourd'hui : +0,00001, adopté de justesse).

Comportement obtenu (rang dans un pool de 6-8 champions du rôle) :

| Scénario | Malphite (Top) | Galio (Mid) | Rammus (Jungle) |
|---|---|---|---|
| Aucun ennemi | 5ᵉ (+0,9 pt) | 3ᵉ (+0,9) | 3ᵉ (+2,0) |
| Ennemis full AP | 6ᵉ (−1,4) | **1ᵉʳ** (+1,8) | 3ᵉ (+1,8) |
| Ennemis full AD | **1ᵉʳ** (+5,5) | 4ᵉ (−0,2) | 2ᵉ (+2,3) |

Avec ou sans Pyke en face, les classements sont désormais quasi identiques.
Rammus ne bouge que légèrement : les données ne montrent pour lui qu'une faible
preuve d'effet anti-AD.

## 6. Note sur la fenêtre de test (02/10)

Le gain au test est passé de ~+0,0038 à ~+0,0013 après la nuit de crawl. Ce
n'est pas une régression du modèle : la fenêtre de test est désormais 100 %
patch 16.19 (l'entraînement est surtout en 16.18/16.17) et à 28 % Master+, où
la draft prédit peu (gain −0,001). En Émeraude-Diamant, le gain reste ~+0,003.

## 7. Synergies et counters « par ressemblance » (`factorization_machine.py`) — négatif

Question : faut-il passer à un réseau de neurones ? Son seul apport par rapport
au modèle additif est d'apprendre des interactions entre champions. Test de
l'idée centrale (des embeddings de champions) sous forme contrainte : une
factorization machine donne à chaque champion de petits vecteurs (k = 2 à 8)
dont on déduit toutes les synergies (⟨v_a, v_b⟩) et tous les counters
(⟨u_a, w_b⟩ − ⟨u_b, w_a⟩). Elle apprend sur les résidus du modèle actuel ;
k et la régularisation sont réglés en validation interne avec arrêt précoce.

Gain de log-loss sur des parties futures, par rapport au modèle additif seul (75k matchs) :

| Variante | Fenêtre récente | Fenêtre précédente |
|---|---|---|
| Synergies | −0,00003 ± 0,00003 | −0,00019 ± 0,00024 |
| Counters | +0,00003 ± 0,00040 | −0,00007 ± 0,00023 |
| Les deux | +0,00005 ± 0,00033 | −0,00002 ± 0,00041 |

- Aucun gain : les petits gains de validation (+0,0001 à +0,0003) ne se
  transfèrent pas, et l'arrêt précoce coupe l'apprentissage après 1 à 30 époques.
- Les paires apprises changent d'une fenêtre à l'autre (Jinx/Caitlyn d'un côté,
  Darius/Xayah de l'autre) : du bruit, pas une structure stable.
- **Conclusion : à ~75k matchs, il n'y a pas d'interaction exploitable au-delà
  de ce que le modèle capture déjà (dont l'équilibre AD/AP). Un réseau de
  neurones, plus flexible et moins régularisé, n'en trouverait pas davantage.**
  À refaire quand le volume aura nettement augmenté.

## 8. Portée et scaling des champions (`range_and_scaling.py`)

Un effet linéaire de ces caractéristiques est déjà absorbé par la force
champion × rôle : seuls des effets d'équipe non linéaires peuvent apporter
quelque chose. Profils et modèle de base sur la 1re moitié chronologique,
résidus testés sur la 2de (pas d'effet possible par construction).

**Scaling — négatif.** Profil cohérent (late game : Bel'Veth, Kayle, Kassadin,
Smolder ; early game : Diana, Irelia, LeBlanc), mais aucun effet d'équipe :
résidus dans les intervalles de confiance, que ce soit selon le scaling de
l'équipe ou selon l'écart avec l'adversaire.

**Portée — effet plausible, faible.** Résidu selon le nombre de corps-à-corps
(attackrange ≤ 250, Data Dragon) de son équipe :

| Corps-à-corps | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Résidu (pts) | −5,2 ± 5,0 | −1,3 ± 1,2 | −0,2 | +0,7 ± 0,6 | −0,3 | −4,0 ± 5,7 |
| Équipes | 371 | 6 651 | 30 148 | 29 865 | 8 326 | 293 |

Test hors échantillon d'un terme d'équipe b_lin·z + b_sq·z² (comme l'AD/AP) :

| Fenêtre | Coefficients | Optimum | Gain de log-loss au test |
|---|---|---|---|
| Récente | +0,007 / −0,021 | 2,7 | +0,00003 ± 0,00028 |
| Précédente | +0,007 / −0,020 | 2,7 | +0,00018 ± 0,00026 |

Coefficients quasi identiques sur deux périodes (signe d'un effet réel, à
l'inverse des synergies de la section 7), gain positif mais non significatif :
l'effet ne concerne que les compositions rares (~−1,5 pt avec un seul
corps-à-corps, ~−4 pts avec aucun). Non intégré à ce stade.

## 9. Point à 100k matchs (02/10)

**Réentraînement complet** (`ml/train.py`, 97,6k matchs exploitables) :
- Sensibilités par champion **non adoptées** : validation glissante −0,00008 ±
  0,00017. Avec plus de données, les effets estimés sont plus diffus et plus
  petits (π 23 % → 66 %, τ 0,034 → 0,017) ; Malphite : pente +0,095 → +0,072,
  part retenue 67 % → 37 %. Seul l'équilibre AD/AP d'équipe reste actif.
- Calibration Master+ confirmée sur 11k matchs : 0,58 ± 0,20 (facteur 0,59).
  Émeraude-Diamant 0,86, Gold-Platine 0,95.
- Écarts par ELO, synergies de duo et pondération de récence toujours rejetés ;
  matchups de lane conservés (échelle 0,5).
- Test (15 % les plus récents) : gain +0,0023 ± 0,0016, précision 53,3 %.

**Contrôle, soutien, vraie frontline** (`team_profiles.py`, profils sur les 22k
matchs collectés avec les nouveaux champs) : profils cohérents (contrôle :
Nocturne, Maokai, Nautilus ; soutien : Zac, Vladimir, Soraka ; frontline :
Sion, K'Santé, Rammus, Ornn) mais **aucun effet d'équipe** : gain hors
échantillon entre −0,00012 et +0,00008 (± ~0,0001) pour les trois.

**Corps-à-corps** (`range_and_scaling.py --oos`) : coefficients toujours stables
(quad −0,024 / −0,023, optimum 2,7) ; gain au test −0,00016 et +0,00022.
Toujours non prouvé, toujours non intégré.

**Factorization machine** (`factorization_machine.py`) : premier signal faible
sur la fenêtre récente (synergies + counters +0,00021 ± 0,00014) mais pas sur
la précédente (−0,00008 ± 0,00035), et paires apprises instables, dominées
par les champions les plus joués. Pas encore de quoi justifier un réseau de
neurones ; à refaire vers 150-200k matchs.

## Règles de méthode adoptées en cours de route

- **Parcimonie** dans le réglage : une complexité supplémentaire n'est retenue
  que si elle améliore la validation d'au moins 0,00005 (sinon, bruit de
  sélection — vu sur les écarts par ELO, retenus à tort sur 0,00001).
- **Pas de seuil sur les champions** : 173 champions testés à la fois → au lieu
  d'un seuil (tout ou rien), lissage bayésien empirique dont la force est
  estimée sur les 173 champions eux-mêmes (section 5).
- **Biais de survie** : toute mesure liée à l'historique d'un joueur n'utilise
  que les parties antérieures à celle évaluée.
