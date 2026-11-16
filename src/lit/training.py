"""Construction du Trainer Lightning et lancement d'un entraînement depuis la config."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import lightning.pytorch as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from lit.callbacks import LossHistory, ReconstructionImages
from lit.datamodule import FeatureDataModule
from lit.modules import build_module
from utils import ConfigError, save_yaml


def device_to_trainer_args(device: str | None) -> tuple[str, int | list[int]]:
    """'auto' | 'cpu' | 'cuda' | 'cuda:N' | 'mps' -> (accelerator, devices) pour pl.Trainer."""
    if device in (None, "auto"):
        return "auto", 1
    if device == "cpu":
        return "cpu", 1
    if device == "mps":
        return "mps", 1
    if device.startswith("cuda"):
        if ":" in device:
            return "gpu", [int(device.split(":", 1)[1])]
        return "gpu", 1
    raise ConfigError(f"device inconnu : {device!r} (auto, cpu, cuda, cuda:N, mps)")


@dataclass
class FitResult:
    run_dir: Path
    module: pl.LightningModule
    history: dict[str, list[float]]


def fit(cfg: dict, run_name: str | None = None) -> FitResult:
    """Entraîne le modèle décrit par cfg. Tout est écrit dans <output_dir>/<model>/<run>/ :
    logs TensorBoard (events.out.tfevents.*, hparams.yaml), config.yaml et checkpoints/ :
    last.ckpt et best-epochNNN.ckpt (Lightning), model.pt (poids + config + pertes, lisible sans Lightning)."""
    if cfg.get("seed") is not None:
        pl.seed_everything(cfg["seed"], workers=True)

    name = cfg["model"]["name"]
    output_dir = cfg.get("output_dir", "outputs")
    version = run_name or time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_dir) / name / version
    save_yaml(cfg, run_dir / "config.yaml")

    train_cfg = cfg["train"]
    history = LossHistory({"train": "loss/train_epoch", "val": "loss/val"})
    callbacks = [
        history,
        ReconstructionImages(),
        ModelCheckpoint(dirpath=run_dir / "checkpoints", filename="best-epoch{epoch:03d}",
                        auto_insert_metric_name=False, monitor="loss/val", mode="min",
                        save_top_k=1, save_last=True),
    ]
    accelerator, devices = device_to_trainer_args(cfg.get("device", "auto"))
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise ConfigError(f"device={cfg.get('device')} demandé mais CUDA n'est pas disponible.")
    trainer_kwargs = dict(
        max_epochs=train_cfg.get("epochs", 5),
        accelerator=accelerator,
        devices=devices,
        logger=TensorBoardLogger(save_dir=output_dir, name=name, version=version, default_hp_metric=False),
        callbacks=callbacks,
        default_root_dir=str(run_dir),
    )
    trainer_kwargs.update(cfg.get("trainer") or {})  # section `trainer` du YAML, passée telle quelle

    module = build_module(cfg)
    datamodule = FeatureDataModule(cfg["data"])
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module, datamodule=datamodule)

    # Export pur PyTorch pour l'évaluation : torch.load(model.pt) suffit, sans Lightning.
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg, "history": history.history, "epochs": len(history.history["train"]),
                "model_state": {k: v.cpu() for k, v in module.model.state_dict().items()}},
               run_dir / "checkpoints" / "model.pt")
    return FitResult(run_dir, module, history.history)
