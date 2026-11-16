"""Construit le jeu de données d'un type de machine du DCASE 2020 Task 2 : un log-mel (.npy) par son.

Protocole de la baseline (Koizumi et al., 2020, « Description and Discussion on DCASE2020 Challenge Task2 ») :
- un modèle par type de machine, entraîné sur les sons normaux de toutes ses machines (ID) ;
- 10 % des sons d'entraînement servent de validation ;
- le test est le jeu de test de développement complet (sons normaux et anormaux de chaque ID).

Entrée : data/dcase2020/<machine>/train/*.wav et test/*.wav, tels que dans dev_data_<machine>.zip
(https://zenodo.org/records/3678171). Sortie (défaut pour la pompe) :
    data/pump/
    ├── train/normal/      log-mel des sons normaux d'entraînement (90 %)
    ├── val/normal/        log-mel des sons normaux mis de côté (10 %)
    └── test/
        ├── normal/        log-mel des sons normaux de test
        └── anomalie/      log-mel des sons anormaux de test

Chaque fichier garde le nom du son d'origine (ex. normal_id_00_00000000.npy) : l'ID de la machine en fait
partie, pour l'évaluation par machine. Chaque log-mel a la forme (128 bandes, 313 trames).

Exemples :
    python prepare_data.py
    python prepare_data.py --machine fan
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data import load_wav, logmel  # noqa: E402


def split_files(train_files: list[Path], test_files: list[Path], val_ratio: float = 0.1,
                seed: int = 0) -> dict[str, list[Path]]:
    """Fichiers de chaque dossier : {"train/normal", "val/normal", "test/normal", "test/anomalie"}.

    La validation est un tirage aléatoire (graine fixe) de val_ratio des sons d'entraînement. Les sons de test
    sont classés d'après leur nom (normal_... ou anomaly_...).
    """
    if not 0 < val_ratio < 1:
        raise ValueError(f"--val-ratio={val_ratio} doit être entre 0 et 1")
    train_files = sorted(train_files)
    if any(not f.name.startswith("normal_") for f in train_files):
        raise ValueError("le dossier train/ ne doit contenir que des sons normaux (normal_*.wav)")
    n_val = max(1, round(len(train_files) * val_ratio))
    val_idx = set(np.random.default_rng(seed).choice(len(train_files), n_val, replace=False).tolist())
    return {
        "train/normal": [f for i, f in enumerate(train_files) if i not in val_idx],
        "val/normal": [f for i, f in enumerate(train_files) if i in val_idx],
        "test/normal": sorted(f for f in test_files if f.name.startswith("normal_")),
        "test/anomalie": sorted(f for f in test_files if f.name.startswith("anomaly_")),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculer les log-mel d'un type de machine du DCASE 2020 Task 2.")
    parser.add_argument("--machine", default="pump", help="Type de machine (défaut : pump)")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/dcase2020"), help="Sons WAV décompressés")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Part des sons d'entraînement en validation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, help="Dossier produit (défaut : data/<machine>)")
    parser.add_argument("--overwrite", action="store_true", help="Remplacer le dossier s'il existe déjà")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    raw = args.raw_dir / args.machine
    out = args.out or Path("data") / args.machine
    train_files, test_files = list((raw / "train").glob("*.wav")), list((raw / "test").glob("*.wav"))
    if not train_files or not test_files:
        parser.error(f"sons introuvables dans {raw}/train et {raw}/test : décompresser dev_data_{args.machine}.zip "
                     f"(https://zenodo.org/records/3678171) dans {args.raw_dir}")
    if out.exists():
        if not args.overwrite:
            parser.error(f"{out} existe déjà (--overwrite pour le remplacer)")
        shutil.rmtree(out)
    try:
        splits = split_files(train_files, test_files, args.val_ratio, args.seed)
    except ValueError as err:
        parser.error(str(err))

    for name, files in splits.items():
        folder = out / name
        folder.mkdir(parents=True)
        for f in files:
            np.save(folder / f"{f.stem}.npy", logmel(load_wav(f)))
        print(f"{folder} : {len(files)} sons")
    return 0


if __name__ == "__main__":
    sys.exit(main())
