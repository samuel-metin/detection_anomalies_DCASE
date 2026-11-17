# Rapport final : améliorer l'apprentissage de l'auto-encodeur (DCASE 2020 Task 2, pompe)

Date : 17 novembre 2026. Quatre sous-agents ont travaillé en parallèle. Chacun avait une piste et une branche
git. Le principe d'auto-encodeur est conservé, et l'`ae` du papier n'a pas changé : il sert de référence.

## En bref

- **Le choix du score compte plus que l'architecture.** On garde l'`ae` du papier, déjà entraîné, et on
  remplace l'erreur moyenne par une distance de Mahalanobis sur l'erreur de reconstruction. L'AUC passe alors de
  **72.9 à 85.3**.
- **Donner l'ID de la machine au décodeur aide** (`cae`). On gagne +3 AUC et +5 pAUC dès 30 epochs.
- IDNN gagne un peu (+1.5 AUC). `ae_plus` (normalisation, débruitage, réseau plus large) ne fait pas mieux que
  le bruit.
- **Limites :**
  - les nouveaux modèles n'ont été entraînés que 30 epochs, avec une seule seed ;
  - la variance entre seeds est d'environ ±1 point ;
  - les scores de la piste D ont été choisis en regardant le test.

## Résultats

AUC et pAUC en %, moyennes des 4 ID de la pompe. Sauf pour la piste D, le score est celui du papier : l'erreur
moyenne du son.

| Piste | Run | AUC | pAUC | id_00 | id_02 | id_04 | id_06 |
|---|---|---|---|---|---|---|---|
| Baseline officielle (papier) | — | 72.89 | 59.99 | 67.2 | 61.5 | 88.3 | 74.6 |
| Référence `ae`, 100 epochs | ae/pump | 72.85 | 59.41 | 66.1 | 61.1 | 87.3 | 77.0 |
| **D : score `maha`** (sur `ae/pump`) | — | **85.3** | 65.8 | 77.7 | 86.0 | 92.3 | 85.3 |
| D : score `knn` (sur `ae/pump`) | — | 83.3 | **73.2** | 82.7 | 72.4 | 98.5 | 79.5 |
| **C : `cae` concat** | cae/C-concat-30 | **75.77** | **64.90** | 68.9 | 61.2 | 96.9 | 76.1 |
| C : `cae` FiLM | cae/C-film-30 | 75.16 | 63.20 | 68.6 | 61.1 | 95.8 | 75.2 |
| B : `idnn` 128×3, goulot 16 | idnn/B-128x3-16-30 | 74.30 | 61.05 | 65.3 | 61.9 | 91.8 | 78.1 |
| A : `ae_plus` 256×4, goulot 16 | ae_plus/A-256-16-30ep | 73.36 | 61.02 | — | — | 90-92 | — |

Les colonnes id_00 à id_06 donnent l'AUC de chaque ID. Tous les runs sont dans `results/historique.md`.

## Détail par piste

### D : scores calculés après l'AE, sans réentraîner (branche `piste-D`)

Les paramètres de ces scores sont estimés sur les sons normaux de `train/`, jamais sur `test/`.

- **`maha`** : distance de Mahalanobis du vecteur d'erreur moyen d'un son (640 valeurs), avec une moyenne et une
  covariance (Ledoit-Wolf) par ID. Ce score est stable : entre 83.5 et 85.3 d'AUC selon les variantes, et
  85.7 sur le checkpoint de 15 epochs. Il ne s'applique qu'aux modèles qui reconstruisent l'entrée entière,
  donc pas à IDNN.
- **`knn`** : distance au plus proche son d'entraînement de la même machine, dans l'espace latent (code latent
  moyen du son). Il donne la meilleure pAUC, mais il est sensible à k : avec k = 5, l'AUC tombe à 80.3 ; avec
  k = 20, à 75.5.
- **Échecs** : les agrégations (quantiles, max) ne battent pas la moyenne. Le z-score par ID ne change pas
  l'AUC par ID.
- **Intégration** : les deux scores sont ajoutés au notebook, et `mse` reste le score par défaut. Le notebook lit
  maintenant aussi `train/`, ce qui prend environ 1 minute de plus.

### C : auto-encodeur conditionné par l'ID (`cae`, branche `piste-C`)

- Le décodeur reçoit l'ID de la machine, de deux façons au choix :
  - **FiLM** : une échelle et un décalage du code latent, propres à chaque ID ;
  - **concat** : le one-hot de l'ID est ajouté au code latent.
- Les deux variantes se valent : leur écart de 0.6 point reste dans le bruit.
- Le gain vient surtout de id_04 (de 87 à 96) et de id_00 (de 66 à 69). id_02 ne bouge pas.
- **Variante de Kapka : échec (28.9 AUC, scores inversés).** Pendant l'entraînement, le modèle reçoit un
  mauvais ID et sa sortie doit alors valoir 0. Causes probables : une cible 0 très éloignée des features
  (environ −30 dB), et des statistiques de BatchNorm faussées.
- **Données** : l'option `data.with_ids: true` (fausse par défaut) charge les ID avec les vecteurs. Le
  comportement de l'`ae` ne change pas.

### B : IDNN, interpolation de la trame centrale (`idnn`, branche `piste-B`)

- Le modèle prédit la trame centrale à partir des 4 autres. Le score est l'erreur sur cette trame.
- L'architecture du papier (64-32-16) ne fait pas mieux que l'AE (72.63). Des couches 128×3 avec un goulot
  de 16 donnent 74.30. Passer à 7 trames n'apporte rien (74.39).

### A : `ae_plus`, MLP amélioré (branche `piste-A`)

- Essais :
  - normalisation des entrées : 73.16 ;
  - avec débruitage, initialisation comme Keras et scheduler cosinus : 73.11 ;
  - couches 256×4 et goulot de 16 : 73.36.
- Aucun gain au-delà du bruit, sauf sur id_04 (90-92). Cette piste ajoute des options d'entraînement
  (AdamW, cosinus) sans bénéfice mesuré.

## Infrastructure ajoutée (branche `ameliorations`)

- **Interface commune des modèles** : `error(x, ids)` renvoie l'erreur par vecteur. La perte est sa moyenne, et
  le score d'un son en est aussi la moyenne. `fit_stats(x_train)` est facultatif.
- **`evaluate_run.py`** : à lancer après chaque entraînement. Il remplace `results/summary.md` (dernier run) et
  ajoute une ligne à `results/historique.md`.
- **Sous-agent** : `.claude/agents/experimentateur.md`, avec les règles d'économie de tokens. Les agents ne
  lisent que `summary.md` et `historique.md`.

## Runs interrompus

`ae_plus/A-norm-100ep` (epoch 1) et `idnn/B-128x3-16-100` (epoch 11) se sont arrêtés à la fin des agents, sans
`model.pt`. Ils sont conservés dans `outputs/`.

## Étapes réalisées ensuite (choix final)

- **Fusion** : seules les pistes C (`cae`) et D (scores `maha` et `knn`) sont gardées dans `ameliorations`.
  Les branches des pistes A et B, et les worktrees, sont supprimés. Les runs restent dans `outputs/`.
- **Correction du notebook** : il passe maintenant l'ID de chaque son au modèle. Sans cette correction, le
  `cae` aurait reconstruit tous les sons avec l'ID 0. Pour cela, `forward(x, ids=None)` est commun à tous
  les modèles.
- **Combinaison `cae` + `maha`** et **confirmation avec une 2e seed** (30 epochs) :

| Run | AUC mse | pAUC mse | AUC maha | pAUC maha | AUC knn | pAUC knn |
|---|---|---|---|---|---|---|
| cae concat, seed 0 | 75.77 | 64.90 | 87.04 | 71.33 | 77.51 | 60.95 |
| cae concat, seed 1 | 76.39 | 65.89 | 87.22 | 71.32 | 76.57 | 60.34 |
| cae FiLM, seed 0 | 75.16 | 63.20 | 86.60 | 69.98 | 82.52 | 68.74 |
| cae FiLM, seed 1 | 73.74 | 60.98 | 86.25 | 68.84 | 81.32 | 65.95 |
| ae baseline (100 ep) | 72.85 | 59.41 | 85.3 | 65.8 | 83.3 | 73.2 |

- **Choix final : `cae` concat + score `maha`, 87.1 AUC / 71.3 pAUC** (moyenne des 2 seeds). C'est le
  meilleur résultat, et il est stable entre seeds. FiLM est moins stable, et la variante de Kapka échoue. Ces
  deux variantes sont retirées du code : `CAE` n'a plus que le conditionnement par concaténation. Les runs
  FiLM et Kapka ne sont plus rechargeables, mais ils sont conservés dans `outputs/`.
- Pas de nouvel entraînement de 100 epochs, ni d'essai sur une autre machine (choix de l'utilisateur).
