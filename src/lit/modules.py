"""LightningModules : enveloppent les architectures de models sans les modifier."""

from __future__ import annotations

import lightning.pytorch as pl
import torch

from models import build_autoencoder


class AutoencoderModule(pl.LightningModule):
    """Auto-encodeur : perte = moyenne de `model.error` (MSE de reconstruction pour l'AE), Adam.

    Métriques TensorBoard : loss/train_step, loss/train_epoch, loss/val.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters()  # config complète -> hparams TensorBoard + checkpoint
        self.config = config
        self.model = build_autoencoder(config["model"])

    def forward(self, x: torch.Tensor, *ids: torch.Tensor) -> torch.Tensor:
        return self.model(x, *ids)  # ids éventuels : modèles conditionnés par l'ID de la machine

    def on_fit_start(self):
        # Statistiques calculées une fois sur tout le train (normalisation…), pour les modèles qui en ont besoin
        if hasattr(self.model, "fit_stats"):
            self.model.fit_stats(self.trainer.datamodule.train_ds.tensors[0].to(self.device))

    def _loss(self, batch, stage: str) -> tuple[torch.Tensor, int]:
        # batch = (vecteurs,) ou (vecteurs, ids) : error() renvoie l'erreur de chaque vecteur
        return self.model.error(*batch).mean(), batch[0].size(0)

    def training_step(self, batch, batch_idx):
        loss, bs = self._loss(batch, "train")
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=bs)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, bs = self._loss(batch, "val")
        self.log("loss/val", loss, on_epoch=True, prog_bar=True, batch_size=bs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config["train"].get("lr", 1e-3))


def build_module(config: dict) -> pl.LightningModule:
    return AutoencoderModule(config)
