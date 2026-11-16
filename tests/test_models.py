"""Architecture de l'AE de la baseline : formes et passes forward."""

import pytest
import torch

from models import AE, Decoder, Encoder, build_autoencoder

HIDDEN = [128, 128, 128, 128]


def test_encoder_decoder_shapes(batch):
    assert Encoder(640, HIDDEN)(batch).shape == (8, 128)
    assert Decoder(8, HIDDEN, 640)(torch.zeros(8, 8)).shape == (8, 640)


def test_baseline_architecture():
    """640 -> 128 x 4 -> 8 -> 128 x 4 -> 640, BatchNorm et ReLU sur chaque couche cachée, sortie linéaire."""
    model = AE(640, HIDDEN, 8)
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == \
        [(640, 128)] + [(128, 128)] * 3 + [(128, 8)] + [(8, 128)] + [(128, 128)] * 3 + [(128, 640)]
    assert sum(isinstance(m, torch.nn.BatchNorm1d) for m in model.modules()) == 9
    assert isinstance(list(model.decoder.modules())[-1], torch.nn.Linear)


def test_forward_shape_and_single_vector(batch):
    model = AE(640, HIDDEN, 8)
    assert model(batch).shape == (8, 640)
    with torch.no_grad():  # un seul vecteur : en mode eval (BatchNorm)
        assert model.eval()(batch[:1]).shape == (1, 640)


def test_encode_decode_matches_forward(batch):
    model = AE(640, [32, 16], 4).eval()
    with torch.no_grad():
        z = model.encode(batch)
        assert z.shape == (8, 4)
        assert torch.allclose(model.decode(z), model(batch))


def test_build_autoencoder():
    model = build_autoencoder({"name": "ae", "hidden_shapes": [64, 32], "latent_shape": 4})
    assert type(model) is AE and model.latent_shape == 4
    assert model(torch.zeros(2, 640)).shape == (2, 640)


def test_build_autoencoder_unknown_name():
    with pytest.raises(ValueError):
        build_autoencoder({"name": "inconnu"})

