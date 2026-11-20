"""LightningModules, DataModule, callbacks TensorBoard et fit() sur données factices."""

import lightning.pytorch as pl
import pytest
import torch

from lit import AutoencoderModule, FeatureDataModule, build_module, device_to_trainer_args, fit
from models import build_autoencoder
from utils import ConfigError

from conftest import small_config


def _snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def _changed(before, module):
    return any(not torch.equal(b, p) for b, p in zip(before, module.parameters()))


def _bare_trainer():
    return pl.Trainer(max_epochs=1, limit_train_batches=1, limit_val_batches=0, accelerator="cpu",
                      devices=1, logger=False, enable_checkpointing=False, enable_progress_bar=False,
                      enable_model_summary=False)


def test_autoencoder_module_forward(tmp_path):
    module = build_module(small_config(tmp_path))
    assert type(module) is AutoencoderModule
    assert module(torch.zeros(4, 640)).shape == (4, 640)


def test_datamodule_batches(fake_data, tmp_path):
    dm = FeatureDataModule(small_config(tmp_path)["data"])
    dm.setup("fit")
    (vectors,) = next(iter(dm.train_dataloader()))
    assert vectors.shape == (8, 640)
    assert (len(dm.train_ds), len(dm.val_ds)) == (64, 32)
    assert (len(dm.train_dataloader()), len(dm.val_dataloader())) == (8, 4)


def test_one_training_step_updates_weights(fake_data, tmp_path):
    cfg = small_config(tmp_path)
    module = build_module(cfg)
    before = _snapshot(module.model)
    _bare_trainer().fit(module, datamodule=FeatureDataModule(cfg["data"]))
    assert _changed(before, module.model)


def _event_tags(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    acc = EventAccumulator(str(run_dir))
    acc.Reload()
    tags = acc.Tags()
    return set(tags.get("scalars", [])), set(tags.get("images", []))


def test_fit_writes_tensorboard_and_checkpoints(fake_data, tmp_path):
    result = fit(small_config(tmp_path, epochs=2), run_name="t")
    run_dir = result.run_dir
    assert run_dir == tmp_path / "ae" / "t"
    assert list(run_dir.glob("events.out.tfevents.*"))
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "checkpoints" / "last.ckpt").is_file()
    assert list((run_dir / "checkpoints").glob("best-epoch*.ckpt"))
    assert len(result.history["train"]) == 2 and len(result.history["val"]) == 2

    scalars, images = _event_tags(run_dir)
    assert {"loss/train_step", "loss/train_epoch", "loss/val"} <= scalars
    assert "reconstructions" in images


def test_model_pt_reloads_without_lightning(fake_data, tmp_path):
    result = fit(small_config(tmp_path), run_name="t")
    ckpt = torch.load(result.run_dir / "checkpoints" / "model.pt", weights_only=True)
    assert ckpt["config"]["model"]["name"] == "ae" and ckpt["epochs"] == 1
    assert len(ckpt["history"]["train"]) == 1
    model = build_autoencoder(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model_state"])
    x = torch.randn(2, 640)
    with torch.no_grad():
        assert torch.allclose(model.eval()(x), result.module.model.cpu().eval()(x))


def test_fit_nnae_saves_memory(fake_data, tmp_path):
    cfg = small_config(tmp_path, model="nnae")
    dm = FeatureDataModule(cfg["data"])
    dm.setup("fit")
    assert len(next(iter(dm.train_dataloader()))) == 3  # (vecteurs, ids, sons)
    result = fit(cfg, run_name="t")
    ckpt = torch.load(result.run_dir / "checkpoints" / "model.pt", weights_only=True)
    model = build_autoencoder(ckpt["config"]["model"])
    model.load_state_dict(ckpt["model_state"])
    assert model.memory_filled.tolist() == [True, False, True] + [False] * 5
    x, ids = torch.randn(2, 640), torch.tensor([0, 2])
    with torch.no_grad():
        assert torch.allclose(model.eval()(x, ids), result.module.model.cpu().eval()(x, ids))


@pytest.mark.parametrize("device, expected", [
    ("auto", ("auto", 1)), (None, ("auto", 1)), ("cpu", ("cpu", 1)),
    ("cuda", ("gpu", 1)), ("cuda:1", ("gpu", [1])), ("mps", ("mps", 1)),
])
def test_device_to_trainer_args(device, expected):
    assert device_to_trainer_args(device) == expected


def test_device_to_trainer_args_invalid():
    with pytest.raises(ConfigError):
        device_to_trainer_args("tpu")
