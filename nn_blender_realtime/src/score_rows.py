from __future__ import annotations

import argparse
import os

import pandas as pd
import torch
import yaml

from src.dbio import read_sqlite_df
from src.net_ae import TinyAE
from src.prepkit import make_splits
from src.utilbits import load_json, ensure_dir


def _load_df(cfg: dict) -> pd.DataFrame:
    mode = cfg["data"]["mode"]  # data mode
    if mode == "sqlite":
        return read_sqlite_df(cfg["data"]["sqlite"]["db_path"], cfg["data"]["sqlite"]["sql_query"])  # read sqlite
    if mode == "csv":
        sep = cfg["data"]["csv"]["delimiter"]  # csv delimiter
        header = 0 if bool(cfg["data"]["csv"]["has_header"]) else None  # header row
        return pd.read_csv(cfg["data"]["csv"]["path"], sep=sep, header=header)  # read csv
    raise ValueError("score needs sqlite or csv mode")  # manual mode needs a different path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--out_csv", default="artifacts/scores.csv")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)  # parse config

    df = _load_df(cfg)  # load rows
    feat_cols = list(cfg["data"]["features"])  # features

    split, _ = make_splits(
        df=df,
        feat_cols=feat_cols,
        id_col=cfg["data"]["id_col"],
        label_col=cfg["data"].get("label_col"),
        normal_label_value=int(cfg["data"]["normal_label_value"]),
        fillna=cfg["prep"]["fillna"],
        test_size=float(cfg["prep"]["test_size"]),
        val_size=float(cfg["prep"]["val_size"]),
        seed=int(cfg["prep"]["seed"]),
    )  # rebuild normalized all-rows matrix

    meta = load_json(os.path.join(args.artifacts, "meta.json"))  # load meta
    thr = float(meta["threshold"])  # threshold

    device = "cuda" if torch.cuda.is_available() else "cpu"  # pick device
    in_dim = split.x_all.shape[1]  # feature count

    model = TinyAE(in_dim=in_dim, mid_dim=int(cfg["net"]["hidden"]), lat_dim=int(cfg["net"]["latent"])).to(device)
    model.load_state_dict(torch.load(os.path.join(args.artifacts, "model.pt"), map_location=device))  # load weights
    model.eval()

    with torch.no_grad():
        xb = torch.from_numpy(split.x_all).float().to(device)  # move data
        xh = model(xb)  # reconstruct
        errs = torch.mean((xb - xh) ** 2, dim=1).detach().cpu().numpy()  # mse per row

    out = pd.DataFrame({cfg["data"]["id_col"]: split.ids_all, "recon_error": errs})  # results table
    out["is_anomaly_pred"] = (out["recon_error"] > thr).astype(int)  # label by threshold

    if split.y_all is not None:
        out["is_anomaly_true"] = split.y_all  # attach labels if available

    ensure_dir(os.path.dirname(args.out_csv) or ".")  # ensure out dir
    out.to_csv(args.out_csv, index=False)  # write csv


if __name__ == "__main__":
    main()
