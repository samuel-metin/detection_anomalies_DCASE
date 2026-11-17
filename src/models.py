"""Auto-encodeur (AE) de la baseline DCASE 2020 Task 2.

Architecture de la baseline (Koizumi et al., 2020), sur des vecteurs de 640 valeurs (5 trames log-mel) :
    Encoder : 640 -> 128 -> 128 -> 128 -> 128      (Linear, BatchNorm, ReLU à chaque couche)
    code    : 128 -> 8                              (Linear, BatchNorm, ReLU)
    Decoder : 8 -> 128 -> 128 -> 128 -> 128 -> 640  (Linear, BatchNorm, ReLU ; sortie linéaire)

`CAE` : même AE, dont le décodeur reçoit aussi l'ID de la machine (config `data.with_ids: true`).

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


class CAE(AE):
    """Auto-encodeur conditionné par l'ID de la machine (idée de Kapka, 2020) : un seul modèle pour toutes les
    machines, mais le décodeur sait de quelle machine vient le son. Encodeur et goulot identiques à l'AE.

    Conditionnement du code latent z par l'ID (`conditioning`) :
    - "film" : z * échelle[id] + décalage[id] (Embedding, initialisés à 1 et 0) ;
    - "concat" : z concaténé au one-hot de l'ID, en entrée du décodeur.

    Variante de Kapka (`wrong_id_weight` > 0) : à l'entraînement, chaque vecteur est aussi décodé avec l'ID
    d'un autre vecteur du batch ; quand cet ID diffère du vrai, la sortie doit valoir `wrong_id_target`
    (constante). Le modèle doit donc vraiment utiliser l'ID. En évaluation, seul compte le vrai ID.
    Sans ids (callbacks, notebook), l'ID 0 est utilisé.
    """

    def __init__(self, input_shape: int = 640, hidden_shapes: Sequence[int] = (128, 128, 128, 128),
                 latent_shape: int = 8, n_ids: int = 8, conditioning: str = "film",
                 wrong_id_weight: float = 0.0, wrong_id_target: float = 0.0):
        super().__init__(input_shape, hidden_shapes, latent_shape)
        self.n_ids = n_ids
        self.conditioning = conditioning
        self.wrong_id_weight = wrong_id_weight
        self.wrong_id_target = wrong_id_target
        if conditioning == "film":
            self.scale = nn.Embedding(n_ids, latent_shape)
            self.shift = nn.Embedding(n_ids, latent_shape)
            nn.init.ones_(self.scale.weight)
            nn.init.zeros_(self.shift.weight)
        elif conditioning == "concat":
            self.decoder = Decoder(latent_shape + n_ids, list(hidden_shapes), input_shape)
        else:
            raise ValueError(f"conditioning inconnu : {conditioning!r} (film, concat)")

    def decode(self, z: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        if ids is None:
            ids = torch.zeros(z.size(0), dtype=torch.long, device=z.device)
        if self.conditioning == "film":
            z = z * self.scale(ids) + self.shift(ids)
        else:
            z = torch.cat([z, nn.functional.one_hot(ids, self.n_ids).to(z.dtype)], dim=1)
        return self.decoder(z)

    def forward(self, x: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        return self.decode(self.encode(x), ids)

    def error(self, x: torch.Tensor, ids: torch.Tensor | None = None) -> torch.Tensor:
        """MSE de reconstruction avec le vrai ID ; à l'entraînement, plus le terme des mauvais ID (Kapka)."""
        z = self.encode(x)
        err = ((self.decode(z, ids) - x) ** 2).mean(dim=1)
        if self.training and self.wrong_id_weight > 0 and ids is not None:
            wrong = ids[torch.randperm(ids.size(0), device=ids.device)]
            err_wrong = ((self.decode(z, wrong) - self.wrong_id_target) ** 2).mean(dim=1)
            err = err + self.wrong_id_weight * (wrong != ids) * err_wrong
        return err


MODELS = {"ae": AE, "cae": CAE}
MODEL_NAMES = tuple(MODELS)


def build_autoencoder(model_cfg: dict) -> nn.Module:
    """Modèle décrit par la section `model` de la config : les autres clés que `name` sont passées au
    constructeur."""
    name = model_cfg["name"]
    if name not in MODELS:
        raise ValueError(f"Auto-encodeur inconnu : {name!r} (choix : {list(MODELS)})")
    return MODELS[name](**{k: v for k, v in model_cfg.items() if k != "name"})
