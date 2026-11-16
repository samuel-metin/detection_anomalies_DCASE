"""Configuration YAML et overrides en ligne de commande."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Configuration invalide ou incohérente."""


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Fichier de configuration introuvable : {path}")
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def parse_override(item: str) -> tuple[str, Any]:
    """'train.epochs=3' -> ('train.epochs', 3). La valeur est interprétée en YAML."""
    if "=" not in item:
        raise ConfigError(f"Override invalide '{item}' : format attendu cle.sous_cle=valeur")
    key, raw = item.split("=", 1)
    key = key.strip()
    if not key:
        raise ConfigError(f"Override invalide '{item}' : clé vide")
    return key, yaml.safe_load(raw)


def set_by_path(cfg: dict, dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
        if not isinstance(node, dict):
            raise ConfigError(f"Impossible d'écrire '{dotted_key}' : '{k}' n'est pas une section")
    node[keys[-1]] = value


def apply_overrides(cfg: dict, overrides: dict[str, Any]) -> dict:
    """Retourne une copie de cfg avec les overrides (les valeurs None sont ignorées)."""
    cfg = copy.deepcopy(cfg)
    for key, value in overrides.items():
        if value is not None:
            set_by_path(cfg, key, value)
    return cfg
