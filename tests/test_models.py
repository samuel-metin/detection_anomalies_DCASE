"""Architecture de l'AE de la baseline et de l'AE à plus proche voisin (NNAE) : formes et passes forward."""

import pytest
import torch

from models import AE, NNAE, Decoder, Encoder, build_autoencoder

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


def test_nnae_neighbour_from_another_sound_of_same_machine():
    model = NNAE(640, [32, 16], 4).train()
    z = torch.tensor([[0.0, 0, 0, 0], [0.1, 0, 0, 0], [5.0, 0, 0, 0], [0.2, 0, 0, 0]])
    ids = torch.tensor([0, 0, 0, 2])
    sounds = torch.tensor([0, 0, 1, 2])
    # vecteurs 0 et 1 (même son) : seul le vecteur 2 est permis (le 3 est d'une autre machine) ; vecteur 3 : seul
    # de sa machine -> lui-même
    assert torch.equal(model.neighbours(z, ids, sounds), z[[2, 2, 1, 3]])
    assert torch.equal(model.neighbours(z, ids), z[[1, 0, 1, 3]])  # sans sons : seul le vecteur lui-même est exclu


def test_nnae_memory(batch):
    model = NNAE(640, [32, 16], 4, alpha=0.5, memory_size=4)
    model.build_memory(torch.randn(10, 640), torch.tensor([0] * 6 + [2] * 4))
    assert model.training and model.memory_filled.tolist() == [True, False, True] + [False] * 5
    model.eval()
    with torch.no_grad():
        ids = torch.tensor([0, 2, 4, 6, 0, 2, 4, 6])
        z = model.encode(batch)
        mixed = model.neighbours(z, ids)
        assert torch.equal(mixed[[2, 3]], z[[2, 3]])  # id 4 et 6 absents de la mémoire : pas de mélange
        assert all(any(torch.equal(mixed[i], m) for m in model.memory[ids[i]]) for i in (0, 1))
        assert model(batch[:1], ids[:1]).shape == (1, 640)
        assert torch.allclose(model.error(batch, ids), ((model(batch, ids) - batch) ** 2).mean(dim=1))
        model.alpha = 1.0  # sans mélange : l'AE
        assert torch.allclose(model(batch, ids), model.decode(z))


def test_build_nnae():
    model = build_autoencoder({"name": "nnae", "hidden_shapes": [64, 32], "latent_shape": 4})
    assert type(model) is NNAE and model.decoder.hidden[0].in_features == 4  # pas d'ID dans le décodeur
    assert model.memory.shape == (8, 128, 4)
