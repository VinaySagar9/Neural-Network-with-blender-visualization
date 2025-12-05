from __future__ import annotations

import json
import os
import pickle
from typing import Any

import numpy as np


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)  # make sure dir exists


def dump_pickle(obj: Any, path: str) -> None:
    parent = os.path.dirname(path) or "."  # parent folder
    ensure_dir(parent)  # ensure parent exists
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)  # save object


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)  # load object


def dump_json(obj: Any, path: str) -> None:
    parent = os.path.dirname(path) or "."  # parent folder
    ensure_dir(parent)  # ensure parent exists
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)  # pretty json


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)  # parse json


def minmax01(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    lo = float(np.min(x))  # min
    hi = float(np.max(x))  # max
    return (x - lo) / (hi - lo + eps)  # scale into [0,1]


def ema(prev: float, cur: float, decay: float) -> float:
    return prev * decay + cur * (1.0 - decay)  # smooth pulse


def clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)  # clamp to [0,1]
