from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import yaml

from src.dbio import read_sqlite_df
from src.net_ae import TinyAE
from src.prepkit import make_splits
from src.utilbits import dump_json, load_json, ensure_dir, minmax01, ema


def _load_df(cfg: dict):
    mode = cfg["data"]["mode"]  # data mode
    if mode == "sqlite":
        return read_sqlite_df(cfg["data"]["sqlite"]["db_path"], cfg["data"]["sqlite"]["sql_query"])  # read sqlite
    if mode == "csv":
        sep = cfg["data"]["csv"]["delimiter"]  # csv delimiter
        header = 0 if bool(cfg["data"]["csv"]["has_header"]) else None  # header row
        import pandas as pd
        return pd.read_csv(cfg["data"]["csv"]["path"], sep=sep, header=header)  # read csv
    raise ValueError("export needs sqlite or csv mode")  # manual mode doesn't batch export


def _pack_layer(tag: str, vec: np.ndarray) -> dict:
    vec = vec.reshape(-1)  # flatten
    norm = minmax01(vec)  # map into [0,1]
    return {f"n_{tag}_{i:03d}": float(norm[i]) for i in range(norm.shape[0])}  # object-name keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_samples", type=int, default=240)
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

    x_clip = split.x_all[: args.max_samples]  # cap row count
    ids_clip = split.ids_all[: args.max_samples]  # ids

    ensure_dir(args.out_dir)

    with torch.no_grad():
        xb = torch.from_numpy(x_clip).float().to(device)  # move data
        xh, glow = model(xb, want_glow=True)  # recon + activations
        errs = torch.mean((xb - xh) ** 2, dim=1).detach().cpu().numpy()  # mse per row

    err01 = minmax01(errs)  # normalized errors
    pulse = 0.0
    d = float(cfg["realtime"]["pulse_decay"])

    for i in range(x_clip.shape[0]):
        frame = {}
        frame.update(_pack_layer("enc0", glow.enc0[i].detach().cpu().numpy()))  # encoder glow
        frame.update(_pack_layer("lat", glow.lat[i].detach().cpu().numpy()))  # latent glow
        frame.update(_pack_layer("dec0", glow.dec0[i].detach().cpu().numpy()))  # decoder glow
        frame.update(_pack_layer("out", xh[i].detach().cpu().numpy()))  # output values

        pulse = ema(pulse, float(err01[i]), d)  # smoothed pulse

        frame["_meta"] = {
            "row_id": int(ids_clip[i]),
            "recon_error": float(errs[i]),
            "anomaly_score_01": float(err01[i]),
            "pulse_01": float(pulse),
            "threshold": float(thr),
        }  # frame metadata

        dump_json(frame, os.path.join(args.out_dir, f"frames_{i+1:04d}.json"))  # write frame json


if __name__ == "__main__":
    main()
