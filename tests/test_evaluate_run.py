"""Script evaluate_run.py : résumé et historique d'un run entraîné sur données factices."""

import evaluate_run
from lit import fit

from conftest import small_config, write_logmels


def test_summary_and_history(fake_data, tmp_path):
    write_logmels(fake_data / "test" / "normal", 3, seed=2)
    write_logmels(fake_data / "test" / "anomalie", 2, seed=3)
    for path in (fake_data / "test" / "anomalie").glob("*.npy"):  # normal_id_00_* -> anomaly_id_00_*
        path.rename(path.with_name(path.name.replace("normal", "anomaly")))
    run_dir = fit(small_config(tmp_path), run_name="t").run_dir
    results = tmp_path / "results"

    evaluate_run.main([str(run_dir), "--change", "essai", "--results", str(results), "--device", "cpu"])
    evaluate_run.main([str(run_dir), "--change", "même run", "--results", str(results), "--device", "cpu"])

    summary = (results / "summary.md").read_text(encoding="utf-8").splitlines()
    assert len(summary) <= 20 and summary[0] == "# Dernier run : ae/t"
    assert any("AUC" in line and "pAUC" in line for line in summary)
    rows = (results / "historique.md").read_text(encoding="utf-8").splitlines()[4:]
    assert len(rows) == 2
    assert rows[0].startswith("| ae/t | essai |") and "premier run" in rows[0]
    assert "équivalent" in rows[1]  # même modèle : même AUC que le meilleur run précédent
