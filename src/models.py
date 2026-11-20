"""Auto-encodeur (AE) de la baseline DCASE 2020 Task 2.

Architecture de la baseline (Koizumi et al., 2020), sur des vecteurs de 640 valeurs (5 trames log-mel) :
    Encoder : 640 -> 128 -> 128 -> 128 -> 128      (Linear, BatchNorm, ReLU à chaque couche)
    code    : 128 -> 8                              (Linear, BatchNorm, ReLU)
    Decoder : 8 -> 128 -> 128 -> 128 -> 128 -> 640  (Linear, BatchNorm, ReLU ; sortie linéaire)

`NNAE` : même AE, dont le décodeur reçoit le code mélangé à celui du plus proche exemple normal de la même machine.

Interface (callbacks d'entraînement, notebook d'évaluation) :
- `encode(x)` renvoie le code latent ;
- `forward(x, ids=None)` vaut `decode(encode(x))` : la reconstruction ;
- `error(x, ids=None)` renvoie l'erreur de chaque vecteur, de forme (B,) : la perte d'entraînement est sa moyenne
  et le score d'anomalie d'un son est sa moyenne sur les vecteurs du son. `ids` (numéro de machine, id_04 -> 4)
  sert aux modèles qui en ont besoin (nnae : voisins de la même machine) ; l'AE l'ignore.

Un modèle peut aussi définir `fit_stats(x_train)`, appelé une fois avant l'entraînement avec tous les vecteurs
d'entraînement (par exemple pour des statistiques de normalisation rangées dans des buffers).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def mlp(sizes: Sequence[int]) -> nn.Sequential:
    """Couches Linear, BatchNorm, ReLU entre les tailles successives de `sizes` (blocs de la baseline)."""
    layers: list[nn.Module] = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(n_in, n_out), nn.BatchNorm1d(n_out), nn.ReLU()]
    return nn.Sequential(*layers)


class Encoder(nn.Module):
    """x -> représentation cachée de taille hidden_shapes[-1]."""

    def __init__(self, input_shape: int, hidden_shapes: Sequence[int]):
        super().__init__()
        self.net = mlp([input_shape, *hidden_shapes])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Decoder(nn.Module):
    """z -> reconstruction, en remontant les couches de l'Encoder dans l'ordre inverse ; sortie linéaire
    (les log-mel en dB ne sont pas bornés)."""

    def __init__(self, latent_shape: int, hidden_shapes: Sequence[int], output_shape: int):
        super().__init__()
        self.hidden = mlp([latent_shape, *reversed(hidden_shapes)])
        self.out = nn.Linear(hidden_shapes[0], output_shape)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(self.hidden(z))


class AE(nn.Module):
    def __init__(self, input_shape: int = 640, hidden_shapes: Sequence[int] = (128, 128, 128, 128),
                 latent_shape: int = 8):
        super().__init__()
        hidden_shapes = list(hidden_shapes)
        self.latent_shape = latent_shape
        self.encoder = Encoder(input_shape, hidden_shapes)
        self.to_latent = nn.Linear(hidden_shapes[-1], latent_shape)
        self.latent_act = nn.Sequential(nn.BatchNorm1d(latent_shape), nn.ReLU())  # code de la baseline
        self.decoder = Decoder(latent_shape, hidden_shapes, input_shape)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.latent_act(self.to_latent(self.encoder(x)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.decode(self.encode(x))  # ids ignorés : même interface que les modèles conditionnés

    def error(self, x: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        """MSE de reconstruction de chaque vecteur (B,), comme la baseline."""
        return ((self(x) - x) ** 2).mean(dim=1)


class NNAE(AE):
    """AE qui reconstruit en s'appuyant sur de vrais exemples normaux : le décodeur reçoit
    z_mix = alpha * z + (1 - alpha) * z_ppv, où z_ppv est le code le plus proche de z parmi des sons normaux de la
    même machine. Un son normal a un voisin proche (z_mix ≈ z) ; un son anormal non : sa reconstruction est tirée
    vers du normal et son erreur augmente.

    - entraînement : voisins pris dans le batch, parmi les vecteurs d'autres sons (`sounds`, config
      `data.with_sounds: true`) de la même machine ; le gradient passe aussi par le code du voisin ;
    - évaluation (mode eval) : voisins pris dans une mémoire de `memory_size` codes par machine (vecteurs
      d'entraînement tirés au hasard, toujours les mêmes), recalculée par `build_memory` avant chaque validation
      et à la fin de l'entraînement : elle est rangée dans model.pt. Machine absente de la mémoire : pas de mélange.
    Encodeur, goulot et décodeur sont ceux de l'AE de la baseline ; l'ID ne sert qu'à choisir les voisins (sans
    ids : ID 0, toutes les machines ensemble).
    """

    def __init__(self, input_shape: int = 640, hidden_shapes: Sequence[int] = (128, 128, 128, 128),
                 latent_shape: int = 8, n_ids: int = 8, alpha: float = 0.5, memory_size: int = 128):
        super().__init__(input_shape, hidden_shapes, latent_shape)
        self.alpha = alpha
        self.memory_size = memory_size
        self.register_buffer("memory", torch.zeros(n_ids, memory_size, latent_shape))
        self.register_buffer("memory_filled", torch.zeros(n_ids, dtype=torch.bool))

    @torch.no_grad()
    def build_memory(self, x_train: torch.Tensor, ids_train: torch.Tensor, seed: int = 0) -> None:
        """Mémoire : codes de `memory_size` vecteurs d'entraînement par machine, tirés au hasard (graine fixe)."""
        was_training = self.training
        self.eval()
        generator = torch.Generator().manual_seed(seed)
        ids_train = ids_train.cpu()
        self.memory_filled.zero_()
        for mid in ids_train.unique().tolist():
            rows = (ids_train == mid).nonzero().squeeze(1)
            rows = rows[torch.randperm(len(rows), generator=generator)[: self.memory_size]]
            codes = self.encode(x_train[rows.to(x_train.device)].to(self.memory.device))
            self.memory[mid] = codes[torch.arange(self.memory_size) % len(codes)]  # répété si trop peu de vecteurs
            self.memory_filled[mid] = True
        self.train(was_training)

    def neighbours(self, z: torch.Tensor, ids: torch.Tensor, sounds: torch.Tensor | None = None) -> torch.Tensor:
        """Code du plus proche exemple normal de la même machine, pour chaque code de z (z lui-même à défaut)."""
        if self.training:  # dans le batch, hors du son lui-même (hors du vecteur lui-même sans `sounds`)
            same = sounds if sounds is not None else torch.arange(len(z), device=z.device)
            invalid = (ids[:, None] != ids[None, :]) | (same[:, None] == same[None, :])
            dist = torch.cdist(z.detach(), z.detach()).masked_fill(invalid, float("inf"))
            best, j = dist.min(dim=1)
            return torch.where(torch.isinf(best)[:, None], z, z[j])
        memory = self.memory[ids]  # (B, memory_size, latent)
        j = ((memory - z[:, None]) ** 2).sum(dim=2).argmin(dim=1)
        return torch.where(self.memory_filled[ids][:, None], memory[torch.arange(len(z)), j], z)

    def forward(self, x: torch.Tensor, ids: torch.Tensor | None = None,
                sounds: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encode(x)
        if ids is None:
            ids = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        z_mix = self.alpha * z + (1 - self.alpha) * self.neighbours(z, ids, sounds)
        return self.decode(z_mix)

    def error(self, x: torch.Tensor, ids: torch.Tensor | None = None,
              sounds: torch.Tensor | None = None) -> torch.Tensor:
        """MSE de reconstruction de chaque vecteur (B,), après mélange avec le plus proche voisin normal."""
        return ((self(x, ids, sounds) - x) ** 2).mean(dim=1)


MODELS = {"ae": AE, "nnae": NNAE}
MODEL_NAMES = tuple(MODELS)


def build_autoencoder(model_cfg: dict) -> nn.Module:
    """Modèle décrit par la section `model` de la config : les autres clés que `name` sont passées au
    constructeur."""
    name = model_cfg["name"]
    if name not in MODELS:
        raise ValueError(f"Auto-encodeur inconnu : {name!r} (choix : {list(MODELS)})")
    return MODELS[name](**{k: v for k, v in model_cfg.items() if k != "name"})
