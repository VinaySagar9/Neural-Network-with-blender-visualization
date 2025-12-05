import bpy
import json
import socket

FRAMES_HOST = "127.0.0.1"
FRAMES_PORT = 55888

EMISSION_NODE_NAME = "Emission"  # emission node name
BASE_STRENGTH = 1.0  # baseline glow

_sock = None
_buf = ""


def _ensure_socket():
    global _sock
    if _sock is not None:
        return
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # tcp socket
    s.connect((FRAMES_HOST, FRAMES_PORT))  # connect to python streamer
    s.setblocking(False)  # nonblocking reads
    _sock = s


def _set_emission(obj, strength):
    mat = obj.active_material
    if not mat or not mat.use_nodes:
        return
    nodes = mat.node_tree.nodes
    em = nodes.get(EMISSION_NODE_NAME)
    if em is None:
        return
    em.inputs["Strength"].default_value = float(strength)  # set emission strength


def _apply_packet(pkt):
    meta = pkt.get("_meta", {})  # pulse lives here
    pulse = float(meta.get("pulse", 0.0))  # pulse boost
    for key, val in pkt.items():
        if key == "_meta":
            continue
        obj = bpy.data.objects.get(key)
        if obj is None:
            continue
        _set_emission(obj, BASE_STRENGTH + float(val) + pulse)  # apply glow


def _poll():
    global _buf
    try:
        _ensure_socket()
    except Exception:
        return 0.2

    try:
        chunk = _sock.recv(65536).decode("utf-8")  # read bytes
    except BlockingIOError:
        return 0.05
    except Exception:
        return 0.2

    if not chunk:
        return 0.1

    _buf += chunk
    while "\n" in _buf:
        line, _buf = _buf.split("\n", 1)
        line = line.strip()
        if not line:
            continue
        try:
            pkt = json.loads(line)  # parse json packet
        except Exception:
            continue
        if isinstance(pkt, dict):
            _apply_packet(pkt)  # update glow in scene

    return 0.02


def start_live_glow():
    bpy.app.timers.register(_poll)  # start polling socket


start_live_glow()
