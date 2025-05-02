# MPOM Dataset

本资料集包含光刻 Overlay Map 数据，每个样本包括：
- `overlay_prev.png`: 上一轮 overlay map
- `logs.json`: 设备日志信息
- `id.txt`: 设备 ID

## 使用方法
1. 执行下载脚本：
   ```bash
   bash download_mpom.sh
   ```
2. 在 `src/train.py` 或 `src/inference.py` 中指定 `--data_dir data/mpom`。
