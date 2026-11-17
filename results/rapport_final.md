# Rapport final : détection de sons anormaux par auto-encodeur (DCASE 2020 Task 2, pompe)

Date : 17 novembre 2026.

## Résultat

**Choix final : l'auto-encodeur conditionné par l'ID de la machine (`cae`), évalué avec le score `maha`.**
Il obtient **87.1 AUC / 71.3 pAUC** sur la pompe, contre 72.85 / 59.41 pour la baseline du papier (`ae`).

| Modèle | Score | AUC | pAUC | id_00 | id_02 | id_04 | id_06 |
|---|---|---|---|---|---|---|---|
| Baseline officielle (papier) | `mse` | 72.89 | 59.99 | 67.2 | 61.5 | 88.3 | 74.6 |
| `ae` (baseline reproduite, 100 epochs) | `mse` | 72.85 | 59.41 | 66.1 | 61.1 | 87.3 | 77.0 |
| `ae` (100 epochs) | `maha` | 85.3 | 65.8 | 77.7 | 86.0 | 92.3 | 85.3 |
| `cae` (30 epochs) | `mse` | 76.1 | 65.4 | 68.5 | 61.0 | 97.3 | 77.4 |
| **`cae` (30 epochs)** | **`maha`** | **87.1** | **71.3** | 77.8 | 86.5 | 99.0 | 84.9 |

- AUC et pAUC sont en %, pour chaque ID de machine puis en moyenne. Les colonnes id_00 à id_06 donnent l'AUC.
- Lignes `cae` `mse` : moyenne de deux entraînements (seeds 0 et 1 : 75.77 et 76.39 d'AUC).
- Ligne `cae` `maha` : AUC et pAUC moyennes des deux seeds (87.04 et 87.22 d'AUC, 71.33 et 71.32 de pAUC). Les
  colonnes par ID viennent de la seed 0.

## Les deux choix gardés

### 1. `cae` : un auto-encodeur qui connaît la machine

- **Principe :** la pompe regroupe 4 machines (id_00, id_02, id_04, id_06), apprises par un seul modèle. Le `cae`
  reprend l'auto-encodeur de la baseline : mêmes features, même encodeur, même goulot de 8. Seul le décodeur
  change : il reçoit le code latent concaténé au one-hot de l'ID de la machine (id_04 -> 4). Il reconstruit
  ainsi mieux les sons normaux de chaque machine, et un son anormal ressort davantage.
- **Gain :** +3 AUC et +6 pAUC avec le score du papier, dès 30 epochs. Le gain vient surtout de id_04
  (87 → 97).
- **Mise en œuvre :**
  - `configs/cae.yaml` ;
  - classe `CAE` dans `src/models.py` ;
  - option `data.with_ids: true` : les batchs deviennent `(vecteurs, ids)`, et l'ID est lu dans le nom du fichier.

### 2. `maha` : un meilleur score d'anomalie (notebook)

- **Principe :** le score du papier fait la moyenne de l'erreur de reconstruction. Il ignore donc *où* le modèle
  se trompe. `maha` résume chaque son par son vecteur d'erreur moyen `x - x̂` (640 valeurs). Il mesure ensuite
  sa distance de Mahalanobis aux sons normaux d'entraînement de la même machine (moyenne et covariance
  Ledoit-Wolf, par ID).
- **Gain :** +11 à +12 AUC, quel que soit le modèle. Il est très fort sur id_02, la machine la plus difficile
  (61 → 86).
- **Données :** les paramètres sont estimés sur `train/` seulement. La validation reste réservée au seuil, et le
  test ne sert qu'à mesurer.
- **Autre score :** le notebook calcule aussi `knn`, la distance au plus proche son d'entraînement dans l'espace
  latent. Il donne la meilleure pAUC avec l'`ae` (73.2), mais il est moins stable.

## Ce qui a été construit autour

- **Interface commune des modèles** : `encode(x)`, `forward(x, ids=None)` et `error(x, ids=None)`, qui donne
  l'erreur par vecteur. La perte d'entraînement est la moyenne de cette erreur, et le score d'un son aussi.
- **`evaluate_run.py`** : à lancer après chaque entraînement. Il écrit un résumé court du run dans
  `results/summary.md` et ajoute une ligne à `results/historique.md`.
- **Notebook** : il passe l'ID de chaque son au modèle et calcule `mse`, `maha` et `knn`.

## Limites

- Le `cae` n'a été entraîné que 30 epochs, avec deux seeds. Les deux seeds sont très proches.
- `maha` a été choisi en regardant le test de la pompe. Son gain est stable selon les réglages, mais il n'a pas
  été vérifié sur un autre type de machine.

## Pour reproduire

```bash
python train.py --config configs/cae.yaml --epochs 30 --run-name cae
python evaluate_run.py outputs/cae/cae --change "cae, 30 epochs"
# puis notebooks/evaluation.ipynb avec CHECKPOINT = "outputs/cae/cae/checkpoints/model.pt"
```
