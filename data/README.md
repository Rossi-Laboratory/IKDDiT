
# MPOM Dataset

This dataset contains photolithography overlay map data. Each sample includes:
- `overlay_prev.png`: Previous overlay map image
- `logs.json`: Equipment log information
- `id.txt`: Equipment ID

## How to Use

1. Run the download script:
```bash
bash download_mpom.sh
```

2. In `src/train.py` or `src/inference.py`, specify the data path:
```bash
--data_dir data/mpom
```

3. Directory structure after extraction:
```
data/mpom/
├── overlay_prev.png     # Overlay map image from previous lithography layer
├── logs.json            # Corresponding equipment logs
├── id.txt               # Equipment ID (categorical)
```

This dataset is designed for multimodal diffusion training and includes real-world metadata from photolithography tools.
