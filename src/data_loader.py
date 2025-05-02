import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class MPOMDataset(Dataset):
    def __init__(self, data_dir, train=True):
        self.data_dir = data_dir
        self.train = train
        self.overlays = sorted([f for f in os.listdir(data_dir) if f.startswith('overlay_prev') and f.endswith('.png')])

    def __len__(self):
        return len(self.overlays)

    def __getitem__(self, idx):
        overlay_name = self.overlays[idx]
        overlay = Image.open(os.path.join(self.data_dir, overlay_name)).convert('RGB')
        overlay = torch.tensor(np.array(overlay)).permute(2,0,1).float() / 255.0
        log_name = overlay_name.replace('overlay_prev', 'logs').replace('.png', '.json')
        logs = json.load(open(os.path.join(self.data_dir, log_name)))
        logs_tensor = torch.tensor(logs['features'], dtype=torch.float32)
        id_name = overlay_name.replace('overlay_prev', 'id').replace('.png', '.txt')
        with open(os.path.join(self.data_dir, id_name), 'r') as f:
            id_val = int(f.read().strip())
        return overlay, logs_tensor, id_val
