from __future__ import annotations

import argparse
import os
from typing import Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.dbio import read_sqlite_df
from src.prepkit import make_splits
from src.net_ae import TinyAE
from src.utilbits import dump_json, dump_pickle, ensure_dir


def _row_mse(model: TinyAE, arr: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(arr).float().to(device)  # move data
        xh = model(xb)  # reconstruct
        e = torch.mean((xb - xh) ** 2, dim=1)  # per-row mse
    return e.detach().cpu().numpy()  # return to numpy


def _pick_cutoff(errs: np.ndarray, cfg: Dict) -> float:
    mode = cfg["threshold"]["method"]  # cutoff method
    if mode == "mean_std":
        k = float(cfg["threshold"]["std_k"])  # std multiplier
        return float(errs.mean() + k * errs.std())  # mean + k*std
    q = float(cfg["threshold"]["quantile"])  # quantile
    return float(np.quantile(errs, q))  # quantile cutoff


def _load_df(cfg: Dict):
    mode = cfg["data"]["mode"]  # data mode
    if mode == "sqlite":
        return read_sqlite_df(cfg["data"]["sqlite"]["db_path"], cfg["data"]["sqlite"]["sql_query"])  # read sqlite
    if mode == "csv":
        sep = cfg["data"]["csv"]["delimiter"]  # csv delimiter
        header = 0 if bool(cfg["data"]["csv"]["has_header"]) else None  # header row
        import pandas as pd
        return pd.read_csv(cfg["data"]["csv"]["path"], sep=sep, header=header)  # read csv
    raise ValueError("train needs sqlite or csv mode")  # manual mode can't train alone


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out_dir", default="artifacts")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)  # parse config

    df = _load_df(cfg)  # load data table
    feat_cols = list(cfg["data"]["features"])  # feature columns

    split, scaler = make_splits(
        df=df,
        feat_cols=feat_cols,
        id_col=cfg["data"]["id_col"],
        label_col=cfg["data"].get("label_col"),
        normal_label_value=int(cfg["data"]["normal_label_value"]),
        fillna=cfg["prep"]["fillna"],
        test_size=float(cfg["prep"]["test_size"]),
        val_size=float(cfg["prep"]["val_size"]),
        seed=int(cfg["prep"]["seed"]),
    )  # build normalized splits

    device = "cuda" if torch.cuda.is_available() else "cpu"  # pick device
    in_dim = split.x_train.shape[1]  # feature count

    model = TinyAE(in_dim=in_dim, mid_dim=int(cfg["net"]["hidden"]), lat_dim=int(cfg["net"]["latent"])).to(device)

    ds = TensorDataset(torch.from_numpy(split.x_train).float())  # train dataset
    dl = DataLoader(ds, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, drop_last=False)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )  # optimizer

    best_val = float("inf")
    hist = {"train_loss": [], "val_loss": []}

    for _ in range(int(cfg["train"]["epochs"])):
        model.train()
        run = 0.0
        for (xb,) in dl:
            xb = xb.to(device)  # move batch
            xh = model(xb)  # reconstruct
            loss = torch.mean((xb - xh) ** 2)  # mse loss
            opt.zero_grad(set_to_none=True)  # clear grads
            loss.backward()  # backprop
            opt.step()  # update params
            run += float(loss.detach().cpu().item()) * xb.shape[0]  # sum loss

        train_loss = run / max(1, len(ds))  # mean train loss
        val_errs = _row_mse(model, split.x_val, device)  # val errors
        val_loss = float(val_errs.mean())  # mean val loss

        hist["train_loss"].append(train_loss)
        hist["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            ensure_dir(args.out_dir)
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))  # save best model

    train_errs = _row_mse(model, split.x_train, device)  # errors on train
    cutoff = _pick_cutoff(train_errs, cfg)  # choose threshold

    ensure_dir(args.out_dir)
    dump_pickle(scaler, os.path.join(args.out_dir, "scaler.pkl"))  # save scaler
    dump_json(
        {"threshold": cutoff, "features": feat_cols, "net": cfg["net"], "prep": cfg["prep"]},
        os.path.join(args.out_dir, "meta.json"),
    )  # save meta
    dump_json(hist, os.path.join(args.out_dir, "metrics.json"))  # save metrics


if __name__ == "__main__":
    main()
