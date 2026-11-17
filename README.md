# Détection de sons anormaux de machines par auto-encodeur (DCASE 2020 Task 2)

Un auto-encodeur entraîné uniquement sur des sons « normaux » d'une machine apprend à bien les reconstruire,
mais reconstruit mal ce qu'il n'a jamais entendu. L'erreur de reconstruction devient alors un **score
d'anomalie**. C'est la surveillance de l'état des machines par le son : détecter une panne sans en avoir jamais
enregistré.

Le projet reproduit la **baseline officielle du DCASE 2020 Challenge Task 2** (Koizumi et al., 2020,
*Description and Discussion on DCASE2020 Challenge Task2: Unsupervised Anomalous Sound Detection for Machine
Condition Monitoring*) :

1. `prepare_data.py` transforme les sons d'un type de machine (par défaut la **pompe**) en log-mel ;
2. `train.py` entraîne l'auto-encodeur (AE) de la baseline sur les sons normaux, et sauvegarde le modèle ;
3. `evaluate_run.py` écrit un résumé court du run (AUC / pAUC par machine) dans `results/summary.md` et une
   ligne dans `results/historique.md` ;
4. le notebook `notebooks/evaluation.ipynb` calcule les métriques du papier (AUC et pAUC par machine) sur le jeu
   de test, des scores plus performants (`maha`, `knn`), et permet d'écouter et d'examiner n'importe quel son.

En plus de la baseline (`ae`), le projet propose un auto-encodeur **conditionné par l'ID de la machine** (`cae`),
voir [Améliorations](#améliorations).

## Données

Jeu de développement du DCASE 2020 Task 2 (https://zenodo.org/records/3678171, licence CC BY-NC-SA 4.0),
tiré de MIMII et ToyADMOS : sons de 10 s, mono, 16 kHz. Pour la pompe (`dev_data_pump.zip`, 1 Go), 4 machines
(`id_00`, `id_02`, `id_04`, `id_06`) :

| Dossier | Contenu |
|---------|---------|
| `train/` | 3 349 sons normaux, toutes machines confondues |
| `test/` | 400 sons normaux (100 par machine) et 456 sons anormaux (fuite, contamination, colmatage…) |

Décompresser l'archive dans `data/dcase2020/` (on obtient `data/dcase2020/pump/train/` et `test/`), puis :

```bash
python prepare_data.py                    # data/pump/
python prepare_data.py --machine fan      # autre type de machine -> data/fan/
```

```
data/pump/
├── train/normal/      3 014 log-mel (90 % des sons d'entraînement)
├── val/normal/        335 log-mel (10 %, tirés au hasard avec --seed)
└── test/
    ├── normal/        400 log-mel
    └── anomalie/      456 log-mel
```

Chaque fichier `.npy` garde le nom du son d'origine (ex. `anomaly_id_02_00000012.npy`, qui contient l'ID de la
machine) et contient son log-mel `(128 bandes, 313 trames)`.

## Protocole du papier

| Étape | Choix de la baseline |
|-------|----------------------|
| Features | log-mel : 128 bandes, fenêtre de 1 024 échantillons (64 ms), pas de 512, puissance 2, `10 · log10(mel + ε)` |
| Entrée du modèle | 5 trames consécutives concaténées : vecteurs de 640 valeurs, environ 309 par son |
| Modèle | AE 640 → 128 → 128 → 128 → 128 → 8 → 128 → 128 → 128 → 128 → 640, Linear + BatchNorm + ReLU, sortie linéaire |
| Entraînement | un modèle par type de machine (toutes les machines ensemble), MSE, Adam (lr 0.001), 100 epochs, batchs de 512, 10 % de validation |
| Score d'un son | erreur quadratique moyenne de reconstruction sur tous ses vecteurs |
| Métriques | par machine : AUC et pAUC (taux de faux positifs ≤ 0.1), puis moyenne |

Écarts assumés : les log-mel sont calculés avec torchaudio, réglé comme librosa (fenêtres centrées, échelle et
normalisation « slaney ») ; la validation est un tirage aléatoire de 10 % des **sons** (la baseline Keras prend
les 10 derniers % des vecteurs, donc presque uniquement la dernière machine).

## Améliorations

| Modèle | Score | AUC | pAUC |
|--------|-------|-----|------|
| `ae` (baseline, 100 epochs) | `mse` (papier) | 72.85 | 59.41 |
| `ae` (baseline, 100 epochs) | `maha` | 85.3 | 65.8 |
| `cae` (30 epochs) | `mse` | 76.1 | 65.4 |
| **`cae` (30 epochs)** | **`maha`** | **87.1** | **71.3** |

Pompe, moyenne des 4 machines. Lignes `cae` : moyenne de deux entraînements (seeds 0 et 1), très proches
(`maha` : 87.0 et 87.2 d'AUC).

- **`cae`** (`configs/cae.yaml`) : l'AE de la baseline, dont le décodeur reçoit le code latent concaténé au
  one-hot de l'ID de la machine (id_04 -> 4). Un seul modèle pour toutes les machines, mais il sait de laquelle
  vient le son. Les ID sont lus dans le nom des fichiers (`data.with_ids: true`).
- **`maha`** (notebook) : distance de Mahalanobis du vecteur d'erreur moyen `x - x̂` d'un son à ceux des sons
  normaux d'entraînement de la même machine. **`knn`** : distance au plus proche son d'entraînement dans l'espace
  latent. Ces scores ont été choisis en regardant le test de la pompe : leur gain reste à confirmer ailleurs.
- Chaque run est résumé par `evaluate_run.py` dans `results/summary.md` et `results/historique.md`.

```bash
python train.py --config configs/cae.yaml --epochs 30 --run-name cae
python evaluate_run.py outputs/cae/cae --change "cae, 30 epochs"
```

## Installation

Python 3.9 ou plus récent.

```bash
python -m venv .venv
# Windows : .venv\Scripts\activate    |  Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
```

Pour utiliser un GPU, installer d'abord la version CUDA de `torch` / `torchaudio` en suivant
https://pytorch.org/get-started/locally/.

## Entraînement

```bash
python train.py --config configs/ae.yaml --run-name pump

# Autre type de machine (créé avec prepare_data.py)
python train.py --config configs/ae.yaml --data-root data/fan --run-name fan

# Essai rapide
python train.py --config configs/ae.yaml --epochs 1 --set trainer.limit_train_batches=0.1
```

N'importe quel paramètre peut être changé sans modifier le YAML,
avec `--set cle.sous_cle=valeur` (répétable, valeur lue en YAML). Raccourcis : `--epochs`, `--lr`, `--run-name`,
`--device` (`auto`, `cpu`, `cuda`, `cuda:0`…), `--seed`, `--data-root`, `--batch-size`, `--output-dir`.
La section `trainer:` des YAML est passée telle quelle à `lightning.pytorch.Trainer`.
`python train.py --help` liste toutes les options.

Les vecteurs (environ 930 000 en train pour la pompe, 2,4 Go) sont gardés en mémoire. Un epoch prend une
quinzaine de secondes sur une RTX 4060, soit environ 25 minutes pour les 100 epochs du papier.

### Suivi dans TensorBoard

Dans un second terminal, pendant ou après l'entraînement :

```bash
tensorboard --logdir outputs
# si Windows bloque tensorboard.exe (contrôle d'application) :
python -m tensorboard.main --logdir outputs
```

Puis ouvrir http://localhost:6006. Chaque run apparaît sous le nom `<model>/<run>`.

| Onglet  | Contenu |
|---------|---------|
| Scalars | `loss/train_step`, `loss/train_epoch`, `loss/val` |
| Images  | `reconstructions` : log-mel du début de la validation et sa reconstruction, à chaque epoch |
| HParams | config complète du run |

### Sorties

```
outputs/<model>/<run>/          # <run> : horodatage par défaut, ou --run-name
├── events.out.tfevents.*       # logs TensorBoard
├── hparams.yaml                # hyperparamètres (Lightning)
├── config.yaml                 # config effective (overrides compris)
└── checkpoints/
    ├── last.ckpt               # checkpoint Lightning du dernier epoch
    ├── best-epochNNN.ckpt      # meilleure perte de validation
    └── model.pt                # poids + config + pertes par epoch, lisible sans Lightning
```

`model.pt` se recharge avec `torch.load` :

```python
ckpt = torch.load("outputs/ae/<run>/checkpoints/model.pt", weights_only=True)
model = build_autoencoder(ckpt["config"]["model"])   # depuis src/models.py
model.load_state_dict(ckpt["model_state"])
```

## Évaluation

Le notebook `notebooks/evaluation.ipynb` charge un `model.pt` (par défaut le plus récent de `outputs/`), lit les
log-mel de `val/` et `test/` du jeu de données de sa config, et analyse la détection :

- scores par son : `mse`, erreur de reconstruction moyenne (score du papier), et `maha` / `knn`, estimés sur les
  sons normaux de `train/` par machine ;
- **AUC** et **pAUC** par machine et en moyenne, à côté des résultats de la baseline officielle ;
- courbes ROC, distribution des scores ; en complément, rappel et faux positifs au seuil (quantile des scores de
  validation) ;
- `show(...)` : log-mel, reconstruction et carte d'erreur d'un son, avec un lecteur audio pour l'écouter ; sons
  les plus suspects, anomalies manquées, fausses alertes ;
- tester ses propres sons : sons du test choisis par leur nom, sons modifiés (bruit, sifflement, chocs…), ou un
  fichier WAV quelconque ;
- espace latent (ACP) par machine.

Dans VS Code, choisir le noyau `.venv`.

## Structure du projet

```
.
├── prepare_data.py      # sons WAV -> log-mel (.npy) rangés en train / val / test
├── train.py             # script d'entraînement (CLI)
├── evaluate_run.py      # résumé d'un run -> results/summary.md, results/historique.md
├── configs/             # ae.yaml (baseline), cae.yaml (conditionné par l'ID)
├── src/
│   ├── data.py          # log-mel et vecteurs de 5 trames (réglages du papier), lecture des .npy
│   ├── models.py        # Encoder, Decoder, AE de la baseline, CAE
│   ├── lit/             # entraînement Lightning
│   │   ├── datamodule.py    # FeatureDataModule (train, validation)
│   │   ├── modules.py       # AutoencoderModule
│   │   ├── callbacks.py     # spectrogrammes TensorBoard, historique des pertes
│   │   └── training.py      # fit(cfg) : Trainer, TensorBoardLogger, checkpoints, model.pt
│   └── utils.py         # config YAML et overrides
├── tests/               # tests unitaires (pytest)
├── notebooks/
│   └── evaluation.ipynb # évaluation : métriques du papier, figures, écoute et test de sons
├── results/             # summary.md, historique.md, rapport_final.md
├── data/                # dcase2020/ (sons) et log-mel par machine (non versionnés)
└── outputs/             # runs d'entraînement (non versionnés)
```

## Tests

```bash
pytest
```

Les tests n'utilisent que des données factices (sons et log-mel aléatoires). Ils ne téléchargent rien et
tournent sur CPU en quelques secondes. Ils couvrent les features (log-mel, vecteurs de 5 trames), la
construction du jeu de données, l'architecture de la baseline et du `cae`, la config, l'entraînement Lightning
(logs TensorBoard, checkpoints, `model.pt`), `train.py` et `evaluate_run.py` de bout en bout.

## Pistes

- Les autres types de machines (`fan`, `slider`, `valve`, `ToyCar`, `ToyConveyor`), sur la même base.
- Méthodes de référence simples : ACP en reconstruction, plus proche voisin sur les log-mel.
- Approches plus récentes du challenge : classification des ID de machine (les sons mal classés sont suspects).
