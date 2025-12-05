# autoencoder anomaly detector + realtime blender glow

this project:
- trains an autoencoder on "normal" rows
- scores new rows by reconstruction error
- streams neuron glow + anomaly pulse to blender over tcp in realtime


## quick start

### 1) setup
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

### 2) configure
edit config.yaml:
- data source (sqlite or csv) and feature columns
- id_col and optional label_col

### 3) train
python -m src.train_ae --config config.yaml --out_dir artifacts

### 4) run realtime stream (manual / csv tail / sqlite poll)
python -m src.live_stream --config config.yaml --artifacts artifacts

### 5) blender
- open blender
- scripting tab -> load blender/drive_live_glow.py
- set FRAMES_HOST + FRAMES_PORT if needed
- run script, then press play or just wait (updates are realtime)

blender objects:
- neuron spheres should be named like n_enc0_000 ... n_lat_015 ... n_dec0_063 ... n_out_006
- each neuron sphere needs a material with nodes enabled and an emission node named "Emission"
