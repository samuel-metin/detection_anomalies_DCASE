"""LightningDataModule : vecteurs log-mel des sons normaux de train/ et val/ (voir data.get_train_val)."""

from __future__ import annotations

import warnings

import lightning.pytorch as pl
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, TensorDataset

from data import get_train_val

# Lightning conseille des workers de chargement : inutile ici, les vecteurs sont déjà en mémoire.
warnings.filterwarnings("ignore", ".*does not have many workers.*")


class FeatureDataModule(pl.LightningDataModule):
    """Les vecteurs tiennent en mémoire : chaque batch est extrait d'un coup (tenseur[indices]), sans workers.

    Le dernier batch incomplet de l'entraînement est ignoré (un batch d'un seul vecteur ferait échouer le
    BatchNorm) ; avec environ un million de vecteurs par epoch, c'est négligeable.
    """

    def __init__(self, data_cfg: dict):
        super().__init__()
        self.cfg = dict(data_cfg)
        self.train_ds: TensorDataset | None = None
        self.val_ds: TensorDataset | None = None

    def setup(self, stage: str | None = None) -> None:
        if self.train_ds is None:
            self.train_ds, self.val_ds = get_train_val(self.cfg)

    def _loader(self, dataset: TensorDataset, shuffle: bool, drop_last: bool) -> DataLoader:
        order = RandomSampler(dataset) if shuffle else SequentialSampler(dataset)
        batches = BatchSampler(order, batch_size=self.cfg.get("batch_size", 512), drop_last=drop_last)
        return DataLoader(dataset, sampler=batches, batch_size=None, pin_memory=torch.cuda.is_available())

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=self.cfg.get("shuffle", True), drop_last=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False, drop_last=False)
