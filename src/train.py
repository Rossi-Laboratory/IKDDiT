import argparse
import torch
from torch.utils.data import DataLoader
from data_loader import MPOMDataset
from src.models.ikddit import IKDDiT
from src.utils.diffusion import LatentDiffusion
from src.utils.contrastive import info_nce_loss
import yaml

def train(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = MPOMDataset(config['data_dir'])
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    model = IKDDiT(config).to(device)
    diffusion = LatentDiffusion(config['diffusion'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    for epoch in range(config['epochs']):
        for batch in loader:
            x, logs, ids = batch
            x, logs, ids = x.to(device), logs.to(device), ids.to(device)
            t = torch.randint(0, diffusion.T, (x.size(0),), device=device)
            z_t = diffusion.q_sample(x, t)
            recon, d_loss = model(z_t, logs, ids, config['mask_ratio'], t)
            l_dsm = ((recon - x)**2).mean()
            l_mae = torch.abs(recon - x).mean()
            # dummy teacher output for info_nce example
            _, z_teacher = model(z_t, logs, ids, 0.0, t)
            l_info = info_nce_loss(model.student_enc(z_t, logs), z_teacher)
            loss = l_dsm + config['lambda1'] * l_mae + config['lambda2'] * d_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}: Loss {loss.item():.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/ikddit_s.yaml')
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config))
    train(config)
