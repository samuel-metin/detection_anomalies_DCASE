"""Features log-mel de la baseline DCASE et lecture des dossiers .npy."""

import numpy as np
import pytest
import torch
from scipy.io import wavfile

from data import N_MELS, get_train_val, load_vectors, load_wav, logmel, machine_id, to_vectors
from utils import ConfigError


def test_load_wav_scales_int16(tmp_path):
    wavfile.write(tmp_path / "a.wav", 16000, np.array([0, 16384, -32768], dtype=np.int16))
    assert np.allclose(load_wav(tmp_path / "a.wav"), [0, 0.5, -1])
    wavfile.write(tmp_path / "b.wav", 8000, np.zeros(10, dtype=np.int16))
    with pytest.raises(ValueError):
        load_wav(tmp_path / "b.wav")


def test_logmel_shape_and_db():
    t = np.arange(16000) / 16000
    mel = logmel(0.5 * np.sin(2 * np.pi * 440 * t))
    assert mel.shape == (N_MELS, 1 + 16000 // 512)  # 1 s : 32 trames (fenêtres centrées)
    assert mel.dtype == np.float32
    assert len(set(mel[:, 1:-1].argmax(axis=0))) == 1  # son pur : même bande dominante (hors trames de bord)
    silence = logmel(np.zeros(16000))
    assert np.allclose(silence, 10 * np.log10(np.finfo(float).eps), atol=1e-3)  # log du epsilon, comme la baseline


def test_to_vectors_concatenates_consecutive_frames():
    mel = np.arange(N_MELS * 7, dtype=np.float32).reshape(N_MELS, 7)
    v = to_vectors(mel)
    assert v.shape == (3, N_MELS * 5)
    for i in range(3):
        for k in range(5):  # vecteur i = trames i, i+1, ..., i+4 mises bout à bout
            assert torch.equal(v[i, k * N_MELS:(k + 1) * N_MELS], torch.as_tensor(mel[:, i + k]))


def test_load_vectors_empty_folder(tmp_path):
    with pytest.raises(ConfigError, match="prepare_data"):
        load_vectors(tmp_path / "absent")


def test_get_train_val(fake_data):
    train, val = get_train_val({"root": str(fake_data)})
    assert (len(train), len(val)) == (64, 32)
    assert train[0][0].shape == (640,)


def test_machine_id():
    assert machine_id("normal_id_04_00000012") == 4
    with pytest.raises(ValueError):
        machine_id("sans_identifiant")


def test_get_train_val_with_ids(fake_data):
    train, val = get_train_val({"root": str(fake_data), "with_ids": True})
    assert (len(train), len(val)) == (64, 32)
    x, ids = train.tensors
    assert x.shape == (64, 640) and ids.dtype == torch.long
    assert ids.tolist() == [0] * 32 + [2] * 32  # 2 sons de chaque machine, dans l'ordre des fichiers
    assert torch.equal(x, get_train_val({"root": str(fake_data)})[0].tensors[0])


def test_get_train_val_with_sounds(fake_data):
    train, _ = get_train_val({"root": str(fake_data), "with_ids": True, "with_sounds": True})
    x, ids, sounds = train.tensors
    assert sounds.tolist() == [i for i in range(4) for _ in range(16)]  # 4 sons de 16 vecteurs
