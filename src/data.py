"""Features audio de la baseline DCASE 2020 Task 2 (Koizumi et al., 2020) et lecture du jeu de données.

- `logmel` : son -> spectrogramme log-mel (128 bandes, fenêtre 1024, pas 512, puissance 2), comme la baseline ;
- `to_vectors` : 5 trames consécutives concaténées -> vecteurs de 640 valeurs, l'entrée des modèles ;
- `get_train_val` : vecteurs de <root>/train/normal/*.npy et <root>/val/normal/*.npy (créés par
  prepare_data.py). Le dossier test/ sert à l'évaluation, dans notebooks/evaluation.ipynb.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
from torch.utils.data import TensorDataset

from utils import ConfigError

SAMPLE_RATE = 16000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 512
POWER = 2.0
N_FRAMES = 5  # trames concaténées par vecteur -> N_MELS * N_FRAMES = 640

# Mêmes réglages que librosa.feature.melspectrogram (fenêtre de Hann centrée, échelle et normalisation « slaney »)
_MEL = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
                                            n_mels=N_MELS, power=POWER, norm="slaney", mel_scale="slaney")


def load_wav(path: str | Path) -> np.ndarray:
    """Fichier WAV mono 16 kHz -> signal float32 dans [-1, 1]."""
    rate, signal = wavfile.read(path)
    if rate != SAMPLE_RATE:
        raise ValueError(f"{path} : {rate} Hz, {SAMPLE_RATE} Hz attendus")
    if signal.dtype == np.int16:
        signal = signal / 32768.0
    if signal.ndim > 1:  # stéréo -> premier canal
        signal = signal[:, 0]
    return signal.astype(np.float32)


def logmel(signal: np.ndarray) -> np.ndarray:
    """Signal -> log-mel (N_MELS, trames) en dB : 20 / POWER * log10(mel + epsilon), comme la baseline."""
    mel = _MEL(torch.as_tensor(signal, dtype=torch.float32)).numpy()
    return (20.0 / POWER * np.log10(mel + sys.float_info.epsilon)).astype(np.float32)


def to_vectors(mel: np.ndarray | torch.Tensor, n_frames: int = N_FRAMES) -> torch.Tensor:
    """Log-mel (N_MELS, T) -> (T - n_frames + 1, N_MELS * n_frames) : n_frames trames consécutives concaténées
    (trame 0 puis trame 1…), comme la baseline."""
    frames = torch.as_tensor(mel).T  # (T, N_MELS)
    return frames.unfold(0, n_frames, 1).transpose(1, 2).reshape(-1, frames.size(1) * n_frames)


def load_vectors(folder: str | Path, n_frames: int = N_FRAMES) -> torch.Tensor:
    """Vecteurs de tous les log-mel (.npy) d'un dossier, mis bout à bout, dans l'ordre des fichiers."""
    paths = sorted(Path(folder).glob("*.npy"))
    if not paths:
        raise ConfigError(f"aucun fichier .npy dans {folder} : créer le jeu de données avec python prepare_data.py")
    return torch.cat([to_vectors(np.load(p), n_frames) for p in paths])


def get_train_val(data_cfg: dict) -> tuple[TensorDataset, TensorDataset]:
    """(train, val) depuis la section `data` de la config : vecteurs de <root>/train/normal et <root>/val/normal."""
    root = Path(data_cfg.get("root", "data/pump"))
    n_frames = data_cfg.get("n_frames", N_FRAMES)
    return (TensorDataset(load_vectors(root / "train" / "normal", n_frames)),
            TensorDataset(load_vectors(root / "val" / "normal", n_frames)))
