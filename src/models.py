"""Auto-encodeur (AE) de la baseline DCASE 2020 Task 2.

Architecture de la baseline (Koizumi et al., 2020), sur des vecteurs de 640 valeurs (5 trames log-mel) :
    Encoder : 640 -> 128 -> 128 -> 128 -> 128      (Linear, BatchNorm, ReLU à chaque couche)
    code    : 128 -> 8                              (Linear, BatchNorm, ReLU)
    Decoder : 8 -> 128 -> 128 -> 128 -> 128 -> 640  (Linear, BatchNorm, ReLU ; sortie linéaire)

Interface (callbacks d'entraînement, notebook d'évaluation) :
- `encode(x)` renvoie le code latent ;
- `forward(x)` vaut `decode(encode(x))` : la reconstruction ;
- `error(x, ids=None)` renvoie l'erreur de chaque vecteur, de forme (B,) : la perte d'entraînement est sa moyenne
  et le score d'anomalie d'un son est sa moyenne sur les vecteurs du son. `ids` (numéro de machine, id_04 -> 4)
  sert aux modèles qui en ont besoin ; l'AE l'ignore.

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def error(self, x: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        """MSE de reconstruction de chaque vecteur (B,), comme la baseline."""
        return ((self(x) - x) ** 2).mean(dim=1)


MODELS = {"ae": AE}
MODEL_NAMES = tuple(MODELS)


def build_autoencoder(model_cfg: dict) -> nn.Module:
    """Modèle décrit par la section `model` de la config : les autres clés que `name` sont passées au
    constructeur."""
    name = model_cfg["name"]
    if name not in MODELS:
        raise ValueError(f"Auto-encodeur inconnu : {name!r} (choix : {list(MODELS)})")
    return MODELS[name](**{k: v for k, v in model_cfg.items() if k != "name"})
