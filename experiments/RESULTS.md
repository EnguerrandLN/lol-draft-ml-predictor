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

## 10. Point à 155k matchs (03/10)

**Réentraînement complet** (151,7k matchs exploitables, 102k étiquetés par
tier) — la validation adopte d'elle-même trois couches rejetées jusqu'ici :

| Couche | 100k | 155k |
|---|---|---|
| Écarts de force par tranche d'ELO | rejetés | **adoptés** (échelle 0,3) |
| Synergies de duo (bot, jungle-mid, jungle-top) | rejetées | **adoptées** (échelle 0,3) |
| Sensibilités par champion au profil adverse | rejetées | **adoptées** (validation +0,00001) |
| Calibration Master+ | 0,59 | 0,76 |

Sensibilités : Malphite part retenue 77 % (+0,064), Galio 52 %, Kassadin 43 %.
Calibration : Gold-Platine 1,03, Émeraude-Diamant 0,94, Master+ 0,76 (les écarts
par ELO absorbent une partie de l'écart).

Test (22,7k matchs les plus récents) : gain **+0,0045 ± 0,0013**, précision
54,2 %, contre +0,0030 pour la force champion × rôle seule : les couches
supplémentaires apportent +0,0015, le plus gros apport mesuré à ce jour.

Comportement : Malphite 1er contre une composition full AD ; Galio puis
Kassadin en tête contre une composition full AP ; classements de jungle
différents selon le niveau (Briar 1re en Gold-Platine, RekSai et Sejuani devant
en Master+). Le biais des spécialistes reste visible en blind pick (Karthus,
Veigar, Yone en ADC ; Darius en jungle).

**Factorization machine** : le signal faible de 100k ne se confirme pas
(synergies + counters +0,00009 ± 0,00018 sur les deux fenêtres) ; paires
apprises toujours instables. Les interactions qui passent sont les duos
fréquents, estimés paire par paire : il n'y a toujours pas de structure de
« ressemblance » exploitable, donc pas de réseau de neurones.

**Corps-à-corps** : coefficients stables (optimum 2,7) mais l'effet rétrécit
(quad −0,024 → −0,016) et le gain au test est nul (−0,00004 / 0,00000 ±
0,00015). Toujours non intégré.

**Contrôle, soutien, frontline** (profils sur 75k matchs) : soutien et
frontline nuls ; **contrôle** légèrement positif sur les deux fenêtres
(+0,00007 et +0,00004 ± 0,00007, coefficients identiques : une équipe qui
contrôle plus gagne un peu plus). Pas significatif ; à revoir avec plus de
matchs collectés avec les nouveaux champs.

## 11. Audit du projet (03/10, 152k matchs)

**Hypothèses vérifiées et réfutées**
- *Les picks de niche (Karthus ADC…) viennent d'un ou deux one-tricks
  sur-représentés par le crawl* : **faux**. Karthus ADC = 1 197 parties de 646
  joueurs, le plus présent en fait 4 % ; seuls 9 couples champion × rôle sur
  564 dépendent à plus de 20 % d'un seul joueur.
- *Karthus et Veigar ADC sont surévalués pour un non-spécialiste* : **faux**.
  Ce ne sont pas des picks de niche (31 % et 29 % des parties de ces champions
  se jouent en bot) et leur coût d'inexpérience est celui d'un pick courant.

**Problèmes trouvés et corrigés**
1. **Coût d'inexpérience** (le plus important). À joueur égal et en ne comptant
   que les parties antérieures, l'écart « jamais joué » vs « 3+ parties » selon
   la part du rôle dans les parties du champion :

   | Part du rôle | < 5 % | 5-10 % | 10-20 % | 20-35 % | 35-60 % | > 60 % |
   |---|---|---|---|---|---|---|
   | Écart (pts) | −4,1 | −3,1 | −1,5 | −2,0 | −2,5 | −2,0 |

   Le surcoût est concentré sous 10 % (vrais picks de niche : Kog'Maw top,
   Galio top, Fiddlesticks top…). Mesuré à chaque entraînement
   (`estimate_familiarity`, pick courant / de niche × jamais / 1-2 / 3+) et
   utilisé comme moyenne a priori du décalage personnel ; sans profil, l'app
   affiche « Si jamais joué ». Effet : les picks de niche quittent le haut du
   classement pour qui ne les joue pas (Top : Kog'Maw 52,6 % → 49,4 %).
2. **Règle d'adoption des sensibilités par champion** : basculait au hasard
   (adoptées à 75k, rejetées à 100k, adoptées à 155k) sur des gains de ±0,00001
   pour un IC de ±0,00015. Le lissage bayésien se règle déjà sur la force des
   preuves : la validation ne les rejette plus que si elles nuisent
   significativement (gain + IC < 0).
3. **Fuite mineure** : le profil de dégâts des champions était mesuré en
   incluant la période de test. Il est désormais calculé sans elle pour
   l'évaluation (effet négligeable : Malphite 0,262 → 0,261).
4. **Représentativité des niveaux** : le crawl prend autant de joueurs par
   division, d'où 24 % de Master+ et 27 % de Diamant parmi les matchs étiquetés
   (contre ~1 % et ~3 % des joueurs). Le modèle « niveau non précisé » est donc
   tiré vers le haut ELO. Corrigé côté usage : rang détecté depuis le Riot ID
   (niveau réglé automatiquement), pick rates propres à chaque niveau pour les
   slots encore vides, avertissement dans l'app.
5. **Évaluation réutilisée** : toutes les décisions de conception ont été
   prises en regardant les mêmes fenêtres de test : les chiffres de test sont
   légèrement optimistes. Ajout de `train.py --until` et `ml/evaluate.py` pour
   une évaluation scellée sur une période jamais vue (ex. le patch 16.20).
6. **Tests** : le pipeline d'entraînement n'en avait aucun. Ajout d'un test de
   bout en bout sur données simulées, d'un aller-retour de sérialisation du
   modèle complet, des pick rates par niveau et de la mesure de familiarité
   (49 tests).

**Vérifié et sain** : calibration (prédit 41 % → observé 42 % ; 59 % → 58 %),
429 du crawler négligeables (~0,1 % des requêtes), recommandeur exact par
rapport au modèle (tests d'énumération).

## 12. Matchups appris sur les statistiques de lane (`lane_stats_matchups.py`, 152k matchs)

Le résultat d'une partie ne suffit pas à estimer les matchups (~15 000 paires
par rôle). Tâche auxiliaire : par lane, régression ridge d'un signal continu
mesuré à chaque partie sur les forces et les matchups, puis transfert au modèle
de victoire (score de matchups × β). Deux cibles :
- **écart d'or brut** entre les deux laners : très lié au résultat (corr. ~0,55) ;
- **écart de part d'or dans l'équipe** : quasi indépendant du résultat (corr. ~0),
  domination de lane « pure ».

Piège rencontré : sans cross-fitting, les prédictions en échantillon de la cible
« écart d'or » contiennent l'issue des parties d'entraînement → coefficient
gonflé (0,24) et perte au test (−0,002). Avec cross-fitting (scores de
l'entraînement prédits par des modèles entraînés sans ces matchs) :

| Matchups appris sur… | Fenêtre récente | Fenêtre précédente |
|---|---|---|
| Écart d'or brut | +0,0008 ± 0,0005 | +0,0007 ± 0,0005 |
| **Part d'or dans l'équipe** | **+0,0010 ± 0,0005** | **+0,0009 ± 0,0005** |
| Les deux (corrélées à 0,88) | +0,0009 | +0,0008 |
| Forces apprises sur l'or | ≈ 0 | ≈ 0 |

Premier gain d'interaction significatif et reproduit. Intégré
(`ml/lane_stats.py`) : la décision passe par la validation glissante (+0,00065 ±
0,00034, adopté), puis β·score est réécrit en effets de lane du modèle additif
(le recommandeur les traite sans modification). Le réglage écarte les matchups
de support (part d'or non pertinente pour ce rôle).

Modèle final (test, 15 % les plus récents) : gain **+0,0056 ± 0,0014** (contre
+0,0045 sans), précision 54,4 %. Counters retrouvés : contre Irelia, Warwick et
Sett (+6,5 pts vs pick moyen) ; contre Vayne top, Malphite, Teemo, Nasus ;
contre Yasuo mid, Malzahar, Lissandra, Vladimir.

Prolongements : l'or à 10-15 minutes (endpoint *timeline*, hors contamination
de fin de partie), le farm et les dégâts comme cibles auxiliaires
supplémentaires, et la même idée pour les synergies (bot lane : part d'or du duo).

## 13. Forces qui évoluent avec les patchs (`patch_transition.py`)

Méthode (groupe « patch » du modèle) : marche aléatoire sur les patchs. La force
de chaque champion évolue d'un patch à l'autre par incréments régularisés (L2) ;
paramétrage inversé pour que « main » soit la force au dernier patch, si bien que
le recommandeur raisonne directement au patch courant, sans modification.

Simulation de sortie de patch : entraînement jusqu'au jour de sortie + 0, 1 ou
3 jours, test sur les 5 jours suivants du nouveau patch. Gain de log-loss par
rapport au modèle actuel :

| | J+0 | J+1 | J+3 |
|---|---|---|---|
| 16.18, marche aléatoire (échelle 0,3) | 0,0000 | **+0,0002 ± 0,0001** | **+0,0002 ± 0,0001** |
| 16.18, marche aléatoire (échelle 1,0) | 0,0000 | +0,0004 ± 0,0006 | +0,0004 ± 0,0006 |
| 16.19, marche aléatoire (échelle 0,3) | 0,0000 | −0,0001 ± 0,0001 | −0,0001 ± 0,0002 |
| 16.19, marche aléatoire (échelle 1,0) | −0,0001 | **−0,0013 ± 0,0006** | **−0,0013 ± 0,0005** |
| Pondération de récence (demi-vie 14 j) | −0,0002 | −0,0003 | −0,0004 |

- La pondération de récence nuit toujours (cohérent avec la validation).
- La marche aléatoire aide à la sortie du 16.18 et nuit à celle du 16.19.
  Explication vérifiée : la transition 16.19 est faussée par un changement de
  **population** dans nos données. L'ancien crawl (BFS, sans tier) s'est arrêté
  le 26/09, en plein 16.19 : 46 % des parties des premiers jours du 16.19 en
  viennent, 0 % des jours de test, où le haut ELO passe de 15 % à 28 %. Les
  « évolutions » apprises captaient en partie ce changement de joueurs. La
  transition 16.18 (population stable : 35-40 % d'ancien crawl des deux côtés)
  est le test propre.
- Intégré au réglage automatique (`patch_scale`) ; la validation glissante le
  rejette aujourd'hui (échelle 0,3 : +0,00003, sous le seuil de parcimonie).
  **Le vrai test sera la sortie du 16.20**, avec des données homogènes (100 %
  crawl par ladder) : relancer `experiments/patch_transition.py` en ajoutant la
  transition, et l'entraînement décidera seul.

**Le test du 16.20 (14/10) : négatif.** Données homogènes (100 % crawl par
ladder), 23,5k parties du 16.20, test sur les 5 jours suivants :

| | J+0 | J+1 | J+3 |
|---|---|---|---|
| Marche aléatoire (échelle 0,3) | +0,00014 ± 0,00030 | −0,00041 ± 0,00040 | 0,00000 ± 0,00055 |
| Marche aléatoire (échelle 1,0) | +0,00007 ± 0,00048 | **−0,00534 ± 0,00141** | **−0,00306 ± 0,00155** |
| Récence (demi-vie 14 j) | −0,00031 | −0,00041 | −0,00016 |

Avec 1 à 3 jours du nouveau patch (2,4k à 8k parties), l'évolution propre à
chaque champion est noyée dans le bruit : la marche aléatoire n'aide pas, et
nuit nettement peu régularisée. Bilan sur trois sorties de patch : +0,0002
(16.18), test faussé (16.19), nul à négatif (16.20). La couche reste dans le
code, rejetée par la validation ; plus de suivi spécifique.

Piste liée : traiter l'ancien crawl (sans tier) comme une population à part
(écarts de force propres, comme une tranche d'ELO), pour que les changements de
source de données ne se confondent plus avec les changements d'équilibrage.

## 14. Prédire les picks à partir du reste de la draft (`pick_model.py`, `pick_model_win.py`) — négatif

Le recommandeur remplace chaque slot encore vide par les pick rates du rôle,
comme si le vis-à-vis choisissait sans regarder ton pick. Méthode : un petit
Transformer de type BERT (10 jetons champion × position, plus tranche d'ELO et
patch) apprend à retrouver des slots masqués, sur 130k drafts. L'API ne donne
pas l'ordre des picks : il apprend quels champions vont ensemble.

**Étape 1 : prédire les vrais picks** (test : 22 874 matchs futurs, k des 9
autres slots visibles, k uniforme de 0 à 9 ; log-vraisemblance du vrai
champion, champions visibles et bannis exclus) :

| Méthode | NLL (nats) | Perplexité | Gain |
|---|---|---|---|
| Pick rates globaux | 3,803 | 44,8 | −0,049 ± 0,002 |
| Pick rates par tier (recommandeur) | 3,753 | 42,7 | +0,000 ± 0,001 |
| Pick rates du patch par tier, lissés | 3,754 | 42,7 | référence |
| **Transformer** | **3,731** | **41,7** | **+0,023 ± 0,001** |

- Le tier compte beaucoup ; le patch courant n'apporte rien de plus.
- Le gain croît avec le nombre de picks visibles (+0,011 à 1-4, +0,050 à 9)
  et vient surtout du duo bot (ADC, support : +0,04). Faible au top (+0,007).

**Étape 2 : effet sur la proba de victoire d'une draft incomplète** (23 548
matchs futurs, une équipe, k slots visibles parmi 10, k uniforme de 1 à 9 ;
modèle de victoire actuel entraîné sur les mêmes parties) :

| | Gain |
|---|---|
| Écart² au logit de la draft complète (0,0275 avec les pick rates) | +0,00025 ± 0,00004 (−0,9 %) |
| Log-loss du résultat réel | −0,00002 ± 0,00006 |

Premier pick top à l'aveugle (tier MID) : classement quasi identique, écarts
≤ 0,2 point. Contrôle sans modèle, sur tous les matchs : avec la vraie
distribution des vis-à-vis de chaque champion au lieu des pick rates, ses
matchups changent de ± 0,19 pt au top (écart-type entre champions ; bruit
0,03), ± 0,07 au mid et ± 0,02 au bot. Les exceptions sont des champions eux-mêmes choisis en
counter-pick (Sylas top +1,2 pt, Kayle +0,2), pas des picks contrés.

- **Conclusion : en solo queue, les joueurs counter-pickent peu.
  L'hypothèse d'indépendance du recommandeur est juste à 0,2 point près.**
  Le modèle de picks apprend une structure réelle (accords ADC-support) mais
  elle ne change pas les recommandations. Non intégré (parcimonie).

## 15. Piste 1 prolongée : autres cibles et synergies de duo (`lane_stats_extended.py`, 164k matchs) — négatif

Même méthode que la section 12 (ridge sur une cible auxiliaire, scores
cross-fittés, transfert par régression logistique à offset), avec :
  - d'autres statistiques de lane comme cibles, toutes en part de l'équipe :
    farm (sbires tués), dégâts aux champions, morts ;
  - les synergies de duo apprises sur la part d'or du duo, par exemple
    (ADC + support) / équipe. Ridge sur les forces des deux champions et
    l'effet propre de la paire, transférable aux effets de duo du modèle.

Gain de log-loss par rapport au modèle actuel (avec ses matchups appris sur la
part d'or), coefficients de transfert estimés ensemble :

| Ajout | Fenêtre récente | Fenêtre précédente |
|---|---|---|
| Farm | −0,00003 ± 0,00005 | +0,00000 ± 0,00005 |
| Dégâts aux champions | +0,00024 ± 0,00024 | +0,00001 ± 0,00025 |
| Morts | −0,00000 ± 0,00001 | −0,00000 ± 0,00005 |
| Duo ADC-support | −0,00005 ± 0,00024 | +0,00013 ± 0,00028 |
| Duos jungle-mid, jungle-top | 0 (effet de paire réglé à 0) | 0 |
| Tout | +0,00015 ± 0,00035 | +0,00013 ± 0,00038 |

- Les autres statistiques sont des variantes de la même information : leurs
  scores de matchups sont corrélés à 0,74-0,80 avec celui de la part d'or
  (farm, dégâts). Avec les dégâts, le transfert se répartit entre les deux
  scores sans gain reproduit.
- Synergies ADC-support : des paires cohérentes et stables d'une fenêtre à
  l'autre (Samira + Nautilus, Lucian + Milio en tête des deux ; Yuumi avec les
  ADC peu mobiles Jhin, Jinx, Kai'Sa en queue), mais sans gain mesurable sur
  la prédiction des victoires.
- Jungle-mid, jungle-top : la part d'or du duo ne dépend pas de la paire
  (validation interne : effet de paire nul).
- **Conclusion : la part d'or de fin de partie contient déjà l'information
  utile des statistiques de fin de partie.** Rien d'intégré. Prolongement
  restant : l'or à 10-15 minutes (endpoint *timeline*), qui isole la phase de
  lane au lieu de la mélanger avec le reste de la partie ; il coûte une
  requête d'API de plus par match.

## 16. Point à 192k matchs (05/10)

**Réentraînement complet** (192k matchs exploitables, 141k étiquetés par tier).
Mêmes réglages retenus qu'à 152k (C 0,01 ; lane 0,5 ; duo 0,3 ; équilibre
0,15 ; tier 0,3) ; couche patch toujours rejetée (le vrai test reste le 16.20).

| | 152k | 192k |
|---|---|---|
| Matchups appris sur l'or (validation glissante) | +0,00065 ± 0,00034 | **+0,00090 ± 0,00034** |
| Sensibilités par champion (validation) | +0,00001 | +0,00007 ± 0,00011 |
| Calibration Master+ | 0,76 | 0,89 |
| Calibration Émeraude-Diamant / Gold-Platine | 0,94 / 1,03 | 1,01 / 0,98 |
| Test : gain vs constante | +0,0056 ± 0,0014 | +0,0053 ± 0,0013 |
| Test : précision | 54,4 % | 54,2 % |

Les fenêtres de test diffèrent (29/09-03/10 contre 01/10-05/10) : les deux
modèles sont équivalents au test, dans l'IC. Sur la nouvelle fenêtre, les
matchups appris sur l'or apportent +0,0008 (0,0045 → 0,0053). Les écarts de
calibration par ELO se resserrent vers 1 avec le volume (Master+ : 24k → 44k
matchs). Recommandation contre une composition full AD : Malphite 1er, puis
Kled et Cassiopeia.

## 17. Matchups appris sur l'or à 10 et 15 minutes (`lane_gold_early.py`, 20k timelines) — négatif

Données : timelines de 20 000 matchs tirés au hasard sur les 30 derniers
jours (`fetch_timelines.py`, ~1 760 par heure avec une clé de développement).
Cibles par rôle : part d'or de l'équipe et écart d'or du duel de lane, à 10 et
15 minutes. Même méthode que la section 12 ; les modèles de lane apprennent
sur les matchs d'entraînement qui ont une timeline (13 à 17k), leurs scores
s'appliquent à toutes les drafts (cross-fittés pour l'échantillon).

Gain de log-loss sur des parties futures (28,8k matchs par fenêtre) :

| Matchups appris sur… | Fenêtre récente | Fenêtre précédente |
|---|---|---|
| Or de fin de partie, même échantillon | +0,00022 ± 0,00019 | +0,00010 ± 0,00019 |
| Part d'or à 10 min | +0,00023 ± 0,00019 | +0,00014 ± 0,00018 |
| Écart d'or à 10 min | +0,00030 ± 0,00020 | +0,00012 ± 0,00019 |
| Part d'or à 15 min | +0,00021 ± 0,00018 | +0,00012 ± 0,00018 |
| Écart d'or à 15 min | +0,00030 ± 0,00024 | −0,00003 ± 0,00021 |
| **Or de fin de partie, tous les matchs (actuel)** | **+0,00082 ± 0,00045** | **+0,00092 ± 0,00044** |
| En plus de l'actuel : meilleure cible précoce | +0,00002 ± 0,00004 | −0,00000 ± 0,00003 |

- À volume égal, l'or précoce ne fait pas mieux que l'or de fin de partie
  (écarts de ±0,0001, dans le bruit) : ce n'est pas un signal plus riche par
  match.
- Le volume compte bien plus que la cible : la couche actuelle, apprise sur
  ~150k matchs, apporte 3 à 6 fois plus que n'importe quelle cible apprise
  sur 15k. Égaler ce volume en timelines demanderait ~85 h de collecte, pour
  un signal équivalent à celui qu'on a déjà gratuitement.
- Ajoutés à la couche actuelle, les matchups précoces n'apportent rien.
- Les matchups précoces les plus marqués sont plausibles (Fiora et Irelia
  contre Yone à 10 min, Katarina contre Syndra), mais bruités à ce volume.
- **Conclusion : on arrête la collecte de timelines pour ce modèle.** L'outil
  (`fetch_timelines.py`, tables `timelines` et `timeline_frames`) reste
  disponible.

## 18. Validation de la personnalisation (`personal_validation.py`)

Le décalage personnel de l'app (δ = μ + (S − μV)/(V + 1/τ²), μ = coût de
familiarité, S et V sur l'historique du joueur avec le champion) n'avait jamais
été testé hors échantillon. Protocole conforme à l'usage (profil figé à
l'import) : modèle, τ et μ estimés avant une date de coupure ; profils des
joueurs sur leurs parties d'avant ; test sur leurs parties d'après (joueurs
ayant ≥ 10 parties avant). IC groupé par joueur. Pente b dans
logit P = logit de la draft + b·décalage : 1 = bien dosé.

**Diagnostic.** Coupure 30/09 (16 902 joueurs, 201 931 parties de test) :
δ améliore la prédiction (+0,00047 ± 0,00011), surtout grâce à μ, mais μ est
sous-dosé : pente 1,63 ± 0,22, et 1,85 ± 0,28 à joueur égal. Cause : μ était
mesuré en retirant la moyenne de chaque joueur au résidu seulement ; les
joueurs qui changent souvent de champion ont aussi une moyenne plus basse, ce
qui ramène l'écart mesuré vers 0. Correction : régression à effets fixes joueur
(on centre aussi les indicatrices de classe), effets recentrés. Test unitaire
sur données simulées où l'ancienne méthode donne −0,35 au lieu de −0,5.

| Coupure | | Gain de δ (app) | Pente δ | Pente μ à joueur égal |
|---|---|---|---|---|
| 30/09 | avant correction | +0,00047 ± 0,00011 | 1,24 ± 0,17 | 1,85 ± 0,28 |
| 30/09 | après | **+0,00052 ± 0,00015** | **0,97 ± 0,13** | 1,31 ± 0,19 |
| 25/09 | après | +0,00050 ± 0,00014 | 1,01 ± 0,14 | 1,40 ± 0,19 |

- Par classe (30/09, après) : champion jamais joué +0,00129 ± 0,00036 ;
  1-2 parties −0,00006 ; 3-9 +0,00014 ; 10+ +0,00058 ± 0,00032.
- La partie « historique » de δ (au-delà de μ) apporte peu : +0,00007 ± 0,00006
  et +0,00003 ± 0,00006, pente 0,79 ± 0,34 et 0,57 ± 0,35. Avec ~20 à 60
  parties par joueur en base, l'historique d'un champion distingue mal le
  talent du hasard (τ ≈ 0,14) ; le gros de l'apport est le coût d'inexpérience.
- Reste à joueur égal une pente de 1,3-1,4 sur μ : le coût d'inexpérience est
  peut-être encore un peu sous-estimé pour classer ses propres champions.
- Limite : « jamais joué » = absent de l'historique analysé ; la maîtrise
  (Champion-Mastery-V4) est affichée dans l'app mais pas utilisée.

**Réentraînement à 211k matchs (13/10) avec la correction.** Coût de
familiarité (logit) : pick courant aucune partie −0,083, 1-2 −0,026, 3+ +0,047 ;
niche aucune −0,163, 1-2 −0,092, 3+ +0,039 (contre −0,059 et −0,125 « aucune »
à 192k avec l'ancienne méthode). τ = 0,142. Réglages re-choisis par la
validation : C 0,03 ; lane 0,3 ; duo 0,15 ; équilibre 0,5 ; tier 0,3 (variances
a priori des matchups et duos quasi inchangées : s²C ≈ 0,0027 et 0,0007).
Test (31 658 matchs, 03/10-07/10) : +0,00563 ± 0,00135, précision 54,5 %.
Facteurs de calibration 0,82 (Master+) à 0,87 (Gold-Platine) : les effets sont
un peu trop forts à tous les niveaux.

## 19. Évaluation scellée sur le patch 16.20 (14/10)

Préparée depuis l'audit (section 11) : toutes les décisions de conception ont
été prises en regardant les mêmes fenêtres de test, donc leurs chiffres
pouvaient être optimistes. Modèle entraîné et réglé uniquement sur les parties
d'avant le 07/10 (`train.py --until 2026-10-07`), évalué **une seule fois** sur
toutes les parties du 07/10 au 14/10 : 22 332 parties, dont 22 177 du 16.20,
un patch qu'aucune décision n'avait regardé.

| Tranche | Parties | Gain de log-loss vs constante | Précision |
|---|---|---|---|
| **Toutes** | 22 332 | **+0,00537 ± 0,00139** | **54,2 %** |
| Gold-Platine et moins | 6 552 | +0,00590 ± 0,00269 | 54,4 % |
| Émeraude-Diamant | 9 132 | +0,00491 ± 0,00226 | 53,8 % |
| Master+ | 6 648 | +0,00549 ± 0,00226 | 54,5 % |

- **Le chiffre honnête confirme les tests internes** : +0,0054 contre
  +0,0058 sur la fenêtre de test interne du même modèle (03/10-06/10). Pas
  d'optimisme mesurable, et le modèle tient sur un patch qu'il n'a jamais vu.
- Calibration : prédit 40,0 % → observé 40,2 % ; 50,0 % → 49,5 % ;
  58,7 % → 57,4 % (IC ± 2,1 pts par décile).
- Plus d'écart entre niveaux : la draft prédit autant en Master+ qu'ailleurs.
- Réserves : 482 parties du 07/10 figuraient dans la fenêtre de test du
  réentraînement du 13/10 (chiffre agrégé seulement) ; le test de la couche
  patch (section 13) a regardé le 16.20, mais cette couche n'est pas dans le
  modèle évalué.

## 20. La maîtrise de champion et le décalage personnel (`mastery_familiarity.py`) — premier point, non concluant

Question : l'app compte « jamais joué » un champion absent de l'historique
analysé ; un champion beaucoup joué autrefois est-il pénalisé à tort ?
Protocole prospectif : maîtrise des 15 741 joueurs du ladder photographiée
les 13-14/10 (`fetch_masteries.py`), évaluation sur leurs seules parties jouées
après la photo et après l'entraînement du modèle de l'app ; δ calculé comme
dans l'app sur l'historique d'avant la photo.

Premier point (16/10) : 23 676 parties de 4 516 joueurs. Le décalage de l'app
y fait +0,00032 ± 0,00046 (non significatif à ce volume).

Résultat − prédiction avec δ (points, à joueur égal), champions **absents de
l'historique** selon leur maîtrise : ≤ 10k points −4,4 ± 3,0 ; 10-50k
−0,4 ± 2,2 ; 50-150k +1,3 ± 2,4 ; > 150k +1,7 ± 3,4. Même tendance chez les
champions joués 3+ fois : 10-50k −2,2 ± 2,3 ; 50-150k −1,2 ± 1,7 ; > 150k
+0,7 ± 0,8. Maîtrise nulle : seulement 156 parties (presque tout champion
joué en partie classée a déjà de la maîtrise).

Gain de prédiction en ajoutant la maîtrise (validation croisée groupée par
joueur) : 14 classes +0,00001 ± 0,00042 ; une pente sur log(1 + points)
+0,00013 ± 0,00026. Pente : champions absents de l'historique +0,010 ± 0,019,
présents +0,026 ± 0,017.

- Tendance cohérente (plus de maîtrise, meilleur que prévu, même à historique
  égal) mais pas de gain de prédiction démontré : **non intégré**.
- À refaire quand le crawler aura fait un tour complet après la photo
  (volume environ doublé).

## Règles de méthode adoptées en cours de route

- **Parcimonie** dans le réglage : une complexité supplémentaire n'est retenue
  que si elle améliore la validation d'au moins 0,00005 (sinon, bruit de
  sélection — vu sur les écarts par ELO, retenus à tort sur 0,00001).
- **Pas de seuil sur les champions** : 173 champions testés à la fois → au lieu
  d'un seuil (tout ou rien), lissage bayésien empirique dont la force est
  estimée sur les 173 champions eux-mêmes (section 5).
- **Biais de survie** : toute mesure liée à l'historique d'un joueur n'utilise
  que les parties antérieures à celle évaluée.
