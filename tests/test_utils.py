"""Configuration YAML et overrides."""

import pytest
import torch

from models import build_autoencoder
from utils import ConfigError, apply_overrides, load_config, parse_override

from conftest import CONFIG_DIR


@pytest.mark.parametrize("config_file", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.stem)
def test_configs_build_models(config_file):
    cfg = load_config(config_file)
    assert {"model", "data", "train", "trainer"} <= cfg.keys()
    assert cfg["data"]["root"] == "data/pump"
    assert cfg["model"]["input_shape"] == 128 * cfg["data"]["n_frames"]
    assert build_autoencoder(cfg["model"])(torch.zeros(2, 640)).shape == (2, 640)


def test_parse_override_and_apply():
    key, value = parse_override("train.epochs=3")
    assert (key, value) == ("train.epochs", 3)
    cfg = {"train": {"epochs": 5, "lr": 0.1}}
    new = apply_overrides(cfg, {key: value, "train.lr": None, "data.root": "x"})
    assert new == {"train": {"epochs": 3, "lr": 0.1}, "data": {"root": "x"}}
    assert cfg["train"]["epochs"] == 5  # l'original n'est pas modifié


def test_parse_override_invalid():
    with pytest.raises(ConfigError):
        parse_override("train.epochs")
