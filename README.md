
# Implicit Knowledge Distillation Diffusion Transformer (IKDDiT)
[Paper]() | [Project Page](https://rossi-laboratory.github.io/IKDDiT/) | [Video](https://www.youtube.com/watch?v=SyJvVZ12K4E) | [Code](https://github.com/Rossi-Laboratory/IKDDiT)

This project implements IKDDiT: a Teacher–Student Diffusion Transformer based on an implicit discriminator, designed for photolithography overlay map generation with computational acceleration advantages.

## 🎉 ICCV 2025 Accepted Paper
This paper has been accepted for presentation at the **International Conference on Computer Vision 2025, Honolulu, Hawaii.**

## Key Features
- **Implicit Discriminator**: During training, the Teacher DiT sees all image patches, while the Student DiT sees only visible patches (others are masked), with token alignment guided by an implicit discriminator.
- **Inference Acceleration**: Only the Student DiT is used during inference, denoising and reconstructing a small number of non-masked patches, significantly reducing inference time.
- **Unified Contrastive Embedding**: Embeds image data, equipment logs, and barcode IDs into a shared space, aligned via InfoNCE loss.
- **Gated Cross-Attention**: Fuses condition tokens with latent maps to improve reconstruction quality.

## Repository Structure
See the directory tree for full structure details.

## Installation
```bash
conda env create -f environment.yml
conda activate ikddit
```
or
```bash
pip install -r requirements.txt
```

## Training
1. Download the dataset:
   ```bash
   cd data && bash download_mpom.sh && cd ..
   ```
2. Train the model (Teacher + Student + Implicit Discriminator):
   ```bash
   python src/train.py --config configs/ikddit_s.yaml
   ```

## Hyperparameters
- `mask_ratio` (float): percentage of patches masked during Student encoding (default: 0.5).  
- Ablation study results for `mask_ratio` (FID-15k):

  | Mask Ratio | FID-15k |
  | ---------- | ------- |
  | 0%         | 27.46   |
  | 25%        | 26.06   |
  | 50%        | 24.66   |
  | 70%        | 123.85  |

  Optimal performance is achieved at a 50% mask ratio.

## Loss Function
- **Eq.8:** L_IKDDiT = L_DSM + λ1 * L_MAE + λ2 * L_D

Loss components:
1. DSM: Denoising Score Matching.
2. MAE: Mean Absolute Error (L1), for reconstruction error.
3. Discriminator Loss: Implicit discriminator-guided alignment.

Hyperparameters:
- `lambda1` (float): Weight for MAE.
- `lambda2` (float): Weight for discriminator.

All parameters can be set in `configs/ikddit_s.yaml`.

## Inference
Run only the Student DiT Encoder + Decoder:
```bash
python src/inference.py --model checkpoints/student_ikddit.pth --mask_ratio 0.5
```

## Visualization
- `notebooks/demo.ipynb`: Demonstrates alignment loss during training, σ heatmap during inference, and speed-up benchmarks.

## Citation
```bibtex
@inproceedings{anonymous2025ikddit,
  title={Photolithography Overlay Map Generation with Implicit Knowledge Distillation Diffusion Transformer},
  author={Anonymous},
  booktitle={ICCV},
  year={2025}
}
```
