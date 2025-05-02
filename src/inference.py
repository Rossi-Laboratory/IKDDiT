import argparse
import torch
from torch.utils.data import DataLoader
from data_loader import MPOMDataset
from src.models.ikddit import IKDDiT
from src.utils.diffusion import LatentDiffusion
import yaml

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--config', type=str, default='configs/ikddit_s.yaml')
    parser.add_argument('--mask_ratio', type=float)
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MPOMDataset(config['data_dir'], train=False)
    loader = DataLoader(dataset, batch_size=1)
    model = IKDDiT(config).to(device)
    model.load_state_dict(torch.load(args.model))
    model.eval()
    diffusion = LatentDiffusion(config['diffusion'])
    for idx, batch in enumerate(loader):
        x, logs, ids = batch
        x, logs, ids = x.to(device), logs.to(device), ids.to(device)
        shape = x.shape
        recon = diffusion.sample_loop(shape, model.student_dec, logs, args.mask_ratio)
        torch.save(recon.cpu(), f"recon_{idx}.pt")
