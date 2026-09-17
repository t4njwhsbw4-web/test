"""Lädt config/config.yaml an einer zentralen Stelle."""
from __future__ import annotations

import pathlib

import yaml

DEFAULT_CONFIG_PATH = pathlib.Path(__file__).resolve().parents[2] / "config" / "config.yaml"


def load_config(path: pathlib.Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
