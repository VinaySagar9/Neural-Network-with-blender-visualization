from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class SplitPack:
    x_train: np.ndarray
    x_val: np.ndarray
    x_all: np.ndarray
    ids_all: np.ndarray
    y_all: Optional[np.ndarray]


def _fill_missing(df: pd.DataFrame, feat_cols: List[str], how: str) -> pd.DataFrame:
    if how == "drop":
        return df.dropna(subset=feat_cols)  # drop rows that have missing feats
    if how == "mean":
        return df.fillna({c: df[c].mean() for c in feat_cols})  # fill with mean
    return df.fillna({c: df[c].median() for c in feat_cols})  # fill with median


def make_splits(
    df: pd.DataFrame,
    feat_cols: List[str],
    id_col: str,
    label_col: Optional[str],
    normal_label_value: int,
    fillna: str,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[SplitPack, StandardScaler]:
    df = df.copy()  # avoid mutating caller df
    df = _fill_missing(df, feat_cols, fillna)  # clean missing
    df = df.dropna(subset=feat_cols)  # drop any leftovers

    ids_all = df[id_col].to_numpy()  # keep ids
    x_all_raw = df[feat_cols].astype(float).to_numpy()  # features

    y_all = None
    if label_col is not None and label_col in df.columns:
        y_all = df[label_col].to_numpy()  # optional labels

    df_train = df
    if y_all is not None:
        df_train = df[df[label_col] == normal_label_value]  # train only on normal rows

    x_pool = df_train[feat_cols].astype(float).to_numpy()  # train pool
    _, x_hold = train_test_split(x_pool, test_size=test_size, random_state=seed)  # carve out holdout
    x_val_raw, x_train_raw = train_test_split(x_hold, test_size=val_size, random_state=seed)  # split holdout

    scaler = StandardScaler()  # z-score scaling
    x_train = scaler.fit_transform(x_train_raw)  # fit scaler on train
    x_val = scaler.transform(x_val_raw)  # transform val
    x_all = scaler.transform(x_all_raw)  # transform all rows

    return SplitPack(x_train=x_train, x_val=x_val, x_all=x_all, ids_all=ids_all, y_all=y_all), scaler
