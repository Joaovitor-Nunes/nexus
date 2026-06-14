# src/utils/io.py

import json
import pickle
from pathlib import Path
from typing import Any
import logging

logger = logging.getLogger(__name__)


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Salvo: {path}")


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Salvo: {path}")


def load_pickle(path: Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)