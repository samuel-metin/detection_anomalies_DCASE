"""Construction du jeu de données (log-mel .npy) à partir de sons WAV factices."""

import numpy as np
import pytest
from scipy.io import wavfile

import prepare_data
from prepare_data import split_files


def _wavs(folder, names, seed=0):
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for name in names:
        wavfile.write(folder / f"{name}.wav", 16000, rng.integers(-3000, 3000, 16000, dtype=np.int16))


@pytest.fixture
def raw(tmp_path):
    """Sons factices de 1 s : 10 normaux en train ; 3 normaux et 2 anormaux en test."""
    root = tmp_path / "dcase2020"
    _wavs(root / "pump" / "train", [f"normal_id_0{i % 2 * 2}_{i:08d}" for i in range(10)])
    _wavs(root / "pump" / "test", ["normal_id_00_00000000", "normal_id_00_00000001", "normal_id_02_00000000",
                                   "anomaly_id_00_00000000", "anomaly_id_02_00000000"])
    return root


def test_split_files(raw):
    train, test = list((raw / "pump" / "train").glob("*.wav")), list((raw / "pump" / "test").glob("*.wav"))
    s = split_files(train, test, val_ratio=0.2, seed=0)
    assert {k: len(v) for k, v in s.items()} == {"train/normal": 8, "val/normal": 2,
                                                 "test/normal": 3, "test/anomalie": 2}
    assert not set(s["train/normal"]) & set(s["val/normal"])
    assert all(f.name.startswith("anomaly_") for f in s["test/anomalie"])
    assert split_files(train, test, val_ratio=0.2, seed=0) == s  # reproductible
    with pytest.raises(ValueError):
        split_files(train, test, val_ratio=0)
    with pytest.raises(ValueError):  # une anomalie dans le train
        split_files(train + [raw / "anomaly_id_00_00000000.wav"], test)


def test_main_writes_logmels(raw, tmp_path):
    out = tmp_path / "pump"
    argv = ["--raw-dir", str(raw), "--out", str(out)]
    assert prepare_data.main(argv) == 0
    counts = {f: len(list((out / f).glob("*.npy"))) for f in ["train/normal", "val/normal", "test/normal", "test/anomalie"]}
    assert counts == {"train/normal": 9, "val/normal": 1, "test/normal": 3, "test/anomalie": 2}
    assert (out / "test" / "anomalie" / "anomaly_id_02_00000000.npy").is_file()  # nom d'origine (ID) conservé
    assert np.load(next((out / "train" / "normal").glob("*.npy"))).shape == (128, 32)
    with pytest.raises(SystemExit):  # le dossier existe déjà
        prepare_data.main(argv)
    assert prepare_data.main(argv + ["--overwrite"]) == 0
    with pytest.raises(SystemExit):  # machine absente
        prepare_data.main(["--raw-dir", str(raw), "--machine", "fan", "--out", str(tmp_path / "fan")])
