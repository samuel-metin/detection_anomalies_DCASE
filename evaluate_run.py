"""Résumé court d'un run entraîné, pour comparer les essais sans relire les logs ni le notebook.

Lancé après train.py, sur le model.pt d'un run. Calcule le score du papier (moyenne de model.error sur les
vecteurs d'un son), l'AUC / pAUC par ID de machine sur test/, et le seuil de décision : quantile 95 % des scores
de validation. Écrit ensuite :
- results/summary.md : résumé du dernier run évalué (20 lignes au plus), remplacé à chaque appel ;
- results/historique.md : une ligne par run (run, changement testé, AUC, pAUC, conclusion), ajoutée à la fin.

L'analyse détaillée (figures, matrice de confusion, écoute) est dans notebooks/nnae_experience.ipynb.

Exemples :
    python evaluate_run.py outputs/ae/pump --change "baseline du papier"
    python evaluate_run.py outputs/nnae/nnae/checkpoints/model.pt --change "nnae, 10 epochs"
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data import to_vectors  # noqa: E402
from models import build_autoencoder  # noqa: E402

BASELINE_AUC = {"pump": (72.89, 59.99)}  # baseline officielle (moyenne AUC, pAUC en %), Koizumi et al., 2020
QUANTILE = 0.95  # seuil = quantile des scores de validation -> environ 5 % de fausses alertes
MAX_FPR = 0.1
HISTORY_HEADER = ("| run | changement testé | AUC | pAUC | conclusion |\n"
                  "|---|---|---|---|---|\n")


def machine_id(name: str) -> int:
    """normal_id_04_00000012 -> 4."""
    return int(re.search(r"id_(\d+)", name).group(1))


@torch.no_grad()
def score_folder(model, folder: Path, n_frames: int, device: str) -> list[tuple[str, float]]:
    """(nom, score) de chaque son d'un dossier de log-mel : moyenne de model.error sur ses vecteurs."""
    scores = []
    for path in sorted(folder.glob("*.npy")):
        x = to_vectors(np.load(path), n_frames).to(device)
        ids = torch.full((x.size(0),), machine_id(path.stem), dtype=torch.long, device=device)
        scores.append((path.stem, model.error(x, ids).mean().item()))
    return scores


def evaluate(checkpoint: Path, device: str) -> dict:
    """Métriques d'un model.pt : AUC / pAUC par ID et moyennes, seuil de validation, pertes finales."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
    cfg = ckpt["config"]
    model = build_autoencoder(cfg["model"])
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device).eval()
    data = Path(cfg["data"]["root"])
    data = data if data.is_absolute() else ROOT / data
    n_frames = cfg["data"].get("n_frames", 5)

    val = [s for _, s in score_folder(model, data / "val" / "normal", n_frames, device)]
    test = [(name, s, False) for name, s in score_folder(model, data / "test" / "normal", n_frames, device)]
    test += [(name, s, True) for name, s in score_folder(model, data / "test" / "anomalie", n_frames, device)]
    if not val or not test:
        raise SystemExit(f"val/ ou test/ vide dans {data} : lancer prepare_data.py")

    per_id = {}
    for mid in sorted({machine_id(n) for n, _, _ in test}):
        labels = [a for n, _, a in test if machine_id(n) == mid]
        scores = [s for n, s, _ in test if machine_id(n) == mid]
        per_id[f"id_{mid:02d}"] = (100 * roc_auc_score(labels, scores),
                                   100 * roc_auc_score(labels, scores, max_fpr=MAX_FPR))
    threshold = float(np.quantile(val, QUANTILE))
    pred = np.array([s > threshold for _, s, _ in test])
    anomaly = np.array([a for _, _, a in test])
    history = ckpt.get("history", {})
    return {
        "cfg": cfg, "epochs": ckpt.get("epochs"), "machine": data.name, "per_id": per_id,
        "auc": float(np.mean([v[0] for v in per_id.values()])),
        "pauc": float(np.mean([v[1] for v in per_id.values()])),
        "threshold": threshold,
        "recall": float(pred[anomaly].mean()), "false_alarms": float(pred[~anomaly].mean()),
        "loss_train": (history.get("train") or [float("nan")])[-1],
        "loss_val": (history.get("val") or [float("nan")])[-1],
    }


def best_previous(history_path: Path) -> tuple[str, float] | None:
    """(run, AUC) du meilleur run déjà inscrit dans l'historique."""
    if not history_path.is_file():
        return None
    best = None
    for line in history_path.read_text(encoding="utf-8").splitlines()[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        try:
            auc = float(cells[2])
        except (IndexError, ValueError):
            continue
        if best is None or auc > best[1]:
            best = (cells[0], auc)
    return best


def observations(m: dict, previous: tuple[str, float] | None) -> tuple[list[str], str]:
    """Deux lignes d'observations automatiques et une conclusion courte pour l'historique."""
    base = BASELINE_AUC.get(m["machine"])
    comparisons = []
    if base:
        comparisons.append(f"{m['auc'] - base[0]:+.2f} AUC vs baseline officielle ({base[0]})")
    if previous:
        comparisons.append(f"{m['auc'] - previous[1]:+.2f} vs meilleur run précédent ({previous[0]}, "
                           f"{previous[1]:.2f})")
    worst = min(m["per_id"], key=lambda k: m["per_id"][k][0])
    gap = m["loss_val"] / m["loss_train"] if m["loss_train"] else float("nan")
    lines = ["; ".join(comparisons) or "pas de référence",
             f"ID le plus faible : {worst} ({m['per_id'][worst][0]:.1f} AUC) ; perte val / train = {gap:.2f}"]
    if previous is None:
        conclusion = "premier run"
    elif m["auc"] > previous[1] + 1:
        conclusion = "meilleur (> +1 pt)"
    elif m["auc"] >= previous[1] - 1:
        conclusion = "équivalent (±1 pt, bruit entre seeds)"
    else:
        conclusion = "moins bon"
    return lines, conclusion


def write_results(run: str, change: str, m: dict, results: Path) -> None:
    results.mkdir(parents=True, exist_ok=True)
    history_path = results / "historique.md"
    obs, conclusion = observations(m, best_previous(history_path))
    model_cfg = m["cfg"]["model"]
    train_cfg = m["cfg"].get("train", {})
    summary = [
        f"# Dernier run : {run}",
        f"- changement testé : {change}",
        f"- modèle : {model_cfg}",
        f"- données : {m['machine']}, n_frames {m['cfg']['data'].get('n_frames', 5)}, "
        f"batch {m['cfg']['data'].get('batch_size')} ; entraînement : {train_cfg}, {m['epochs']} epochs, "
        f"seed {m['cfg'].get('seed')}",
        f"- perte finale : train {m['loss_train']:.4g}, val {m['loss_val']:.4g}",
        f"- **AUC {m['auc']:.2f} / pAUC {m['pauc']:.2f}** (moyenne par ID, score = erreur moyenne)",
        "- par ID (AUC / pAUC) : " + ", ".join(f"{k} {a:.1f}/{p:.1f}" for k, (a, p) in m["per_id"].items()),
        f"- seuil (quantile {QUANTILE:.0%} de val) : {m['threshold']:.4g} -> rappel {m['recall']:.1%}, "
        f"fausses alertes {m['false_alarms']:.1%} sur le test",
        "",
        "Observations :",
        f"- {obs[0]}",
        f"- {obs[1]}",
        f"- conclusion : {conclusion}",
    ]
    (results / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    if not history_path.is_file():
        history_path.write_text("# Historique des runs\n\n" + HISTORY_HEADER, encoding="utf-8")
    clean = change.replace("|", "/")
    with history_path.open("a", encoding="utf-8") as f:
        f.write(f"| {run} | {clean} | {m['auc']:.2f} | {m['pauc']:.2f} | {conclusion} |\n")


def find_checkpoint(path: Path) -> Path:
    """Dossier de run ou chemin de model.pt -> chemin de model.pt."""
    if path.is_dir():
        path = path / "checkpoints" / "model.pt"
    if not path.is_file():
        raise SystemExit(f"model.pt introuvable : {path}")
    return path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Résumé d'un run : AUC / pAUC par ID, seuil, historique.")
    parser.add_argument("run", type=Path, help="dossier du run (outputs/<modèle>/<run>) ou son model.pt")
    parser.add_argument("--change", required=True, help="changement testé par ce run, en quelques mots")
    parser.add_argument("--results", type=Path, default=ROOT / "results", help="dossier des résumés")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args(argv)

    checkpoint = find_checkpoint(args.run)
    run_dir = checkpoint.parent.parent
    run = f"{run_dir.parent.name}/{run_dir.name}"
    metrics = evaluate(checkpoint, args.device)
    write_results(run, args.change, metrics, args.results)
    print((args.results / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
