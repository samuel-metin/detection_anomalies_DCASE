"""Script train.py : résolution de la config et entraînement de bout en bout sur données factices."""

import pytest

import train
from utils import ConfigError

from conftest import CONFIG_DIR

FAST = ["--device", "cpu", "--batch-size", "8",
        "--set", "trainer.enable_progress_bar=false", "--set", "trainer.log_every_n_steps=1"]


def _args(*argv):
    return train.build_parser().parse_args(list(argv))


def test_config_and_overrides():
    cfg = train.resolve_config(_args("--config", str(CONFIG_DIR / "ae.yaml"), "--epochs", "3",
                                     "--set", "model.latent_shape=4", "--data-root", "data/fan"))
    assert cfg["model"]["name"] == "ae"
    assert cfg["train"]["epochs"] == 3
    assert cfg["model"]["latent_shape"] == 4
    assert cfg["data"]["root"] == "data/fan"


def test_config_is_required():
    with pytest.raises(SystemExit):
        _args()


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError):
        train.resolve_config(_args("--config", str(tmp_path / "absent.yaml")))


def test_invalid_model_name(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  name: inconnu\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        train.resolve_config(_args("--config", str(path)))


def test_train_end_to_end(tmp_path, fake_data):
    run_dir = train.train(_args("--config", str(CONFIG_DIR / "ae.yaml"), "--epochs", "1", "--run-name", "t",
                                "--output-dir", str(tmp_path), "--data-root", str(fake_data), *FAST))
    assert run_dir == tmp_path / "ae" / "t"
    assert list(run_dir.glob("events.out.tfevents.*"))
    for path in ["config.yaml", "checkpoints/model.pt", "checkpoints/last.ckpt"]:
        assert (run_dir / path).is_file(), path


def test_train_without_dataset(tmp_path):
    with pytest.raises(ConfigError, match="prepare_data"):
        train.train(_args("--config", str(CONFIG_DIR / "ae.yaml"), "--output-dir", str(tmp_path),
                          "--data-root", str(tmp_path / "absent"), *FAST))
