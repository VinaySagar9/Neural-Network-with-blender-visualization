from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import threading
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import yaml

from src.dbio import stream_sqlite_rows
from src.net_ae import TinyAE
from src.utilbits import load_json, load_pickle, minmax01, ema, clamp01


def _make_server(host: str, port: int) -> Tuple[socket.socket, "queue.Queue[socket.socket]"]:
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # tcp socket
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # reuse port
    srv.bind((host, port))  # bind host/port
    srv.listen(1)  # listen for 1 client
    out_q: "queue.Queue[socket.socket]" = queue.Queue()  # accepted clients

    def _accept_loop() -> None:
        while True:
            c, _ = srv.accept()  # accept client
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)  # low-latency packets
            out_q.put(c)  # store latest client

    t = threading.Thread(target=_accept_loop, daemon=True)  # background accept thread
    t.start()  # start accepting
    return srv, out_q


def _send_line(conn: socket.socket, payload: dict) -> None:
    msg = (json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8")  # compact json line
    conn.sendall(msg)  # send to blender


def _pack_layer(tag: str, vec: np.ndarray) -> Dict[str, float]:
    vec = vec.reshape(-1)  # flatten
    norm = minmax01(vec)  # [0,1]
    return {f"n_{tag}_{i:03d}": float(norm[i]) for i in range(norm.shape[0])}  # object keys


def _manual_loop(prompt: str, feat_dim: int) -> Iterable[Tuple[int, np.ndarray]]:
    idx = 0
    while True:
        raw = input(prompt)  # read user input
        raw = raw.strip()  # trim whitespace
        if raw.lower() in {"q", "quit", "exit"}:
            return  # stop
        parts = [p.strip() for p in raw.split(",")]  # split by commas
        if len(parts) != feat_dim:
            print(f"need {feat_dim} values")  # simple feedback
            continue
        try:
            vec = np.array([float(p) for p in parts], dtype=float)  # parse floats
        except ValueError:
            print("bad number in input")  # parse failed
            continue
        idx += 1
        yield idx, vec  # provide fake id + row


def _csv_tail_loop(path: str, has_header: bool, delim: str, feat_cols: List[str]) -> Iterable[Tuple[int, np.ndarray]]:
    last_pos = 0
    saw_header = False
    while True:
        if not os.path.exists(path):
            time.sleep(0.25)
            continue
        with open(path, "r", encoding="utf-8") as f:
            f.seek(last_pos)  # resume from last position
            chunk = f.read()  # read new content
            last_pos = f.tell()  # remember position

        if not chunk:
            time.sleep(0.25)
            continue

        lines = [ln for ln in chunk.splitlines() if ln.strip()]  # new non-empty lines
        if has_header and not saw_header:
            saw_header = True  # skip header once
            if lines:
                lines = lines[1:]  # drop header line

        for ln in lines:
            parts = [p.strip() for p in ln.split(delim)]  # split line
            if len(parts) < 1 + len(feat_cols):
                continue
            try:
                row_id = int(parts[0])  # assume id is first col
                vec = np.array([float(x) for x in parts[1:1 + len(feat_cols)]], dtype=float)  # feature slice
            except Exception:
                continue
            yield row_id, vec  # row yield


def _sqlite_poll_loop(cfg: dict, feat_cols: List[str], id_col: str) -> Iterable[Tuple[int, np.ndarray]]:
    last_id = 0
    while True:
        got = stream_sqlite_rows(
            db_path=cfg["data"]["sqlite"]["db_path"],
            stream_query=cfg["data"]["sqlite"]["stream_query"],
            last_id=last_id,
            limit=int(cfg["realtime"]["batch_limit"]),
        )  # fetch new rows since last id

        if got.empty:
            time.sleep(float(cfg["realtime"]["poll_ms"]) / 1000.0)
            continue

        for _, row in got.iterrows():
            try:
                row_id = int(row[id_col])  # id is needed for cursor
                vec = row[feat_cols].astype(float).to_numpy()  # feature vec
            except Exception:
                continue
            last_id = max(last_id, row_id)  # advance cursor
            yield row_id, vec  # yield new row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--artifacts", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)  # parse config

    meta = load_json(os.path.join(args.artifacts, "meta.json"))  # load meta
    feat_cols = list(meta["features"])  # feature columns
    thr = float(meta["threshold"])  # threshold

    scaler = load_pickle(os.path.join(args.artifacts, "scaler.pkl"))  # load scaler

    host = cfg["realtime"]["host"]
    port = int(cfg["realtime"]["port"])
    _, conn_q = _make_server(host, port)  # accept blender client

    device = "cuda" if torch.cuda.is_available() else "cpu"  # pick device
    in_dim = len(feat_cols)

    model = TinyAE(in_dim=in_dim, mid_dim=int(cfg["net"]["hidden"]), lat_dim=int(cfg["net"]["latent"])).to(device)
    model.load_state_dict(torch.load(os.path.join(args.artifacts, "model.pt"), map_location=device))  # load weights
    model.eval()

    mode = cfg["data"]["mode"]  # streaming mode
    if mode == "manual":
        feed = _manual_loop(cfg["data"]["manual"]["prompt"], in_dim)  # manual samples
    elif mode == "csv":
        feed = _csv_tail_loop(
            path=cfg["data"]["csv"]["path"],
            has_header=bool(cfg["data"]["csv"]["has_header"]),
            delim=cfg["data"]["csv"]["delimiter"],
            feat_cols=feat_cols,
        )  # csv tail stream
    else:
        feed = _sqlite_poll_loop(cfg, feat_cols=feat_cols, id_col=cfg["data"]["id_col"])  # sqlite poll stream

    glow_scale = float(cfg["realtime"]["glow_scale"])
    pulse_scale = float(cfg["realtime"]["pulse_scale"])
    pulse_decay = float(cfg["realtime"]["pulse_decay"])

    active_conn: Optional[socket.socket] = None
    pulse = 0.0

    print(f"listening for blender on {host}:{port}")  # server info

    for row_id, raw_vec in feed:
        while active_conn is None:
            try:
                active_conn = conn_q.get(timeout=0.1)  # grab next connected client
                print("blender connected")  # status
            except queue.Empty:
                pass

        norm_vec = scaler.transform(np.asarray(raw_vec, dtype=float).reshape(1, -1))  # apply scaling
        xb = torch.from_numpy(norm_vec).float().to(device)  # move to device

        with torch.no_grad():
            xh, glow = model(xb, want_glow=True)  # recon + activations
            err = torch.mean((xb - xh) ** 2, dim=1).item()  # mse score

        is_bad = 1 if err > thr else 0  # anomaly flag
        score01 = clamp01((err / (thr + 1e-12)) - 0.5)  # rough normalize
        pulse = ema(pulse, score01, pulse_decay)  # smooth pulse

        pkt = {}
        pkt.update({k: v * glow_scale for k, v in _pack_layer("enc0", glow.enc0[0].detach().cpu().numpy()).items()})  # scale enc glow
        pkt.update({k: v * glow_scale for k, v in _pack_layer("lat", glow.lat[0].detach().cpu().numpy()).items()})  # scale lat glow
        pkt.update({k: v * glow_scale for k, v in _pack_layer("dec0", glow.dec0[0].detach().cpu().numpy()).items()})  # scale dec glow
        pkt.update({k: v * glow_scale for k, v in _pack_layer("out", xh[0].detach().cpu().numpy()).items()})  # scale out glow
        pkt["_meta"] = {
            "row_id": int(row_id),
            "recon_error": float(err),
            "threshold": float(thr),
            "is_anomaly": int(is_bad),
            "pulse": float(pulse * pulse_scale),
        }  # attach meta

        try:
            _send_line(active_conn, pkt)  # send update packet
        except Exception:
            try:
                active_conn.close()  # close dead socket
            except Exception:
                pass
            active_conn = None  # wait for next connection


if __name__ == "__main__":
    main()
