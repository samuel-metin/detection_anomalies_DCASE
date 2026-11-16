"""Fixtures communes : petit jeu de données factice (log-mel aléatoires), sans DCASE."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs"
N_TRAMES = 20  # trames par log-mel factice -> 20 - 5 + 1 = 16 vecteurs par son


def write_logmels(folder: Path, n: int, seed: int = 0) -> None:
    """n log-mel factices (128, N_TRAMES) en dB dans folder."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        np.save(folder / f"normal_id_00_{i:08d}.npy", rng.normal(-30, 10, (128, N_TRAMES)).astype(np.float32))


@pytest.fixture
def batch() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(8, 640)


@pytest.fixture
def fake_data(tmp_path) -> Path:
    """Jeu factice dans tmp_path/data : 4 sons dans train/normal (64 vecteurs), 2 dans val/normal (32)."""
    root = tmp_path / "data"
    write_logmels(root / "train" / "normal", 4, seed=0)
    write_logmels(root / "val" / "normal", 2, seed=1)
    return root


def small_config(output_dir, epochs: int = 1) -> dict:
    """Config réelle allégée pour des tests rapides sur CPU, avec les données de la fixture fake_data."""
    from utils import load_config

    cfg = load_config(CONFIG_DIR / "ae.yaml")
    cfg["output_dir"] = str(output_dir)
    cfg["device"] = "cpu"
    cfg["data"].update(root=str(Path(output_dir) / "data"), batch_size=8)
    cfg["train"]["epochs"] = epochs
    cfg["trainer"] = {"log_every_n_steps": 1, "enable_progress_bar": False, "enable_model_summary": False}
    cfg["model"].update(hidden_shapes=[32, 16], latent_shape=4)
    return cfg
