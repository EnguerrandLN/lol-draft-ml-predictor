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

## Règles de méthode adoptées en cours de route

- **Parcimonie** dans le réglage : une complexité supplémentaire n'est retenue
  que si elle améliore la validation d'au moins 0,00005 (sinon, bruit de
  sélection — vu sur les écarts par ELO, retenus à tort sur 0,00001).
- **Pas de seuil sur les champions** : 173 champions testés à la fois → au lieu
  d'un seuil (tout ou rien), lissage bayésien empirique dont la force est
  estimée sur les 173 champions eux-mêmes (section 5).
- **Biais de survie** : toute mesure liée à l'historique d'un joueur n'utilise
  que les parties antérieures à celle évaluée.
