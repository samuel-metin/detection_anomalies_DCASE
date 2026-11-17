"""Callbacks de suivi : images dans TensorBoard et historique des pertes."""

from __future__ import annotations

import lightning.pytorch as pl
import torch


def _to_image(spectrogram: torch.Tensor) -> torch.Tensor:
    """Spectrogramme (H, W) en dB -> image (1, H, W) dans [0, 1] pour add_image."""
    s = spectrogram.detach().cpu()
    return ((s - s.min()) / (s.max() - s.min() + 1e-8)).unsqueeze(0)


def log_image(trainer: pl.Trainer, tag: str, grid: torch.Tensor, step: int) -> None:
    for logger in trainer.loggers:
        experiment = getattr(logger, "experiment", None)
        if hasattr(experiment, "add_image"):  # TensorBoard SummaryWriter
            experiment.add_image(tag, grid, global_step=step)


class ReconstructionImages(pl.Callback):
    """À chaque validation : spectrogramme log-mel du début de la validation (en haut) et sa reconstruction
    (en bas). Chaque colonne est la première trame d'un vecteur : on retrouve le spectrogramme du son.
    Si la validation contient les ids des machines, ils sont passés au modèle."""

    def __init__(self, n_vectors: int = 300, n_mels: int = 128):
        self.n_vectors = n_vectors
        self.n_mels = n_mels
        self.batch: list[torch.Tensor] | None = None

    def on_fit_start(self, trainer, pl_module):
        self.batch = [t[: self.n_vectors] for t in trainer.datamodule.val_ds.tensors]  # vecteurs (et ids)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking or self.batch is None:
            return
        x, *ids = [t.to(pl_module.device) for t in self.batch]
        with torch.no_grad():
            rec = pl_module(x, *ids)
        both = torch.cat([x[:, : self.n_mels].T.flip(0), rec[:, : self.n_mels].T.flip(0)])  # (2 * n_mels, T)
        log_image(trainer, "reconstructions", _to_image(both), trainer.current_epoch)


class LossHistory(pl.Callback):
    """Mémorise les pertes d'epoch, enregistrées dans model.pt. keys : {label: métrique loguée}."""

    def __init__(self, keys: dict[str, str]):
        self.keys = keys
        self.history: dict[str, list[float]] = {label: [] for label in keys}

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        for label, key in self.keys.items():
            if key in metrics:
                self.history[label].append(float(metrics[key]))
