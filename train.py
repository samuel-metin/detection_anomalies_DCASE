"""Entraînement de l'auto-encodeur (AE) de la baseline sur les sons normaux d'un type de machine.

Le modèle vient de model.name dans la config. Les log-mel sont lus dans <data.root>/train/normal/ et
<data.root>/val/normal/ (créés par prepare_data.py). Tout est écrit dans outputs/<model>/<run>/ : logs
TensorBoard, config et checkpoints. L'évaluation sur test/ se fait avec evaluate_run.py.

Exemples :
    python train.py --config configs/ae.yaml
    python train.py --config configs/ae.yaml --epochs 20 --run-name essai
    python train.py --config configs/ae.yaml --data-root data/fan --run-name fan

Suivi : tensorboard --logdir outputs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from lit import fit  # noqa: E402
from models import MODEL_NAMES  # noqa: E402
from utils import ConfigError, apply_overrides, load_config, parse_override  # noqa: E402

# argument CLI -> clé de configuration
OVERRIDE_FLAGS = {
    "device": "device",
    "seed": "seed",
    "output_dir": "output_dir",
    "data_root": "data.root",
    "batch_size": "data.batch_size",
    "epochs": "train.epochs",
    "lr": "train.lr",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Entraîner l'AE sur les sons normaux d'une machine.")
    parser.add_argument("--config", type=Path, required=True, help="Fichier YAML, ex. configs/ae.yaml")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--run-name", help="Nom du run (défaut : horodatage)")
    parser.add_argument("--device", help="auto | cpu | cuda | cuda:0 ...")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--data-root", help="Dossier du jeu de données, ex. data/pump")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output-dir", help="Racine des sorties (défaut : outputs)")
    parser.add_argument("--set", action="append", default=[], metavar="CLE=VALEUR",
                        help="Override générique, ex. --set model.latent_shape=4 (répétable)")
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    """Config du fichier --config, puis flags CLI, puis --set ; valide model.name."""
    cfg = load_config(args.config)
    overrides = {key: getattr(args, flag) for flag, key in OVERRIDE_FLAGS.items()}
    for item in args.set:
        key, value = parse_override(item)
        overrides[key] = value
    cfg = apply_overrides(cfg, overrides)

    name = cfg.get("model", {}).get("name")
    if name not in MODEL_NAMES:
        raise ConfigError(f"model.name invalide dans la config : {name!r} (choix : {MODEL_NAMES})")
    return cfg


def train(args: argparse.Namespace) -> Path:
    cfg = resolve_config(args)
    print(f"Modèle : {cfg['model']['name']} | données : {cfg['data']['root']}")
    run_dir = fit(cfg, args.run_name).run_dir
    print(f"Terminé. Checkpoints : {run_dir / 'checkpoints'}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        train(args)
    except ConfigError as err:
        parser.error(str(err))
    return 0


if __name__ == "__main__":
    sys.exit(main())
