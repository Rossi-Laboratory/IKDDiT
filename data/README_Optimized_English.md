
# IKDDiT: Implicit Knowledge Distillation Diffusion Transformer

This repository provides a full implementation of **IKDDiT**, a novel Teacher–Student Diffusion Transformer guided by an **Implicit Discriminator**, designed for overlay map generation in advanced semiconductor photolithography processes.

## 🔍 Key Features

- **Teacher–Student Framework with Masking**: During training, the Teacher DiT sees all image patches while the Student DiT sees only visible patches. An implicit discriminator ensures token alignment between teacher and student outputs.
- **Fast Inference**: During inference, only the Student DiT Encoder and Decoder are used, allowing masked patches to be reconstructed efficiently for faster runtime.
- **Unified Contrastive Embedding**: Logs and equipment IDs are embedded into a shared latent space, aligned with image data using InfoNCE loss.
- **Gated Cross-Attention**: Enhances fusion of condition tokens with visual features, improving reconstruction quality and semantic consistency.

## 🗂 Repository Structure

```
IKDDiT-Github-Repository/
├── configs/                 # Hyperparameters and model configuration
├── data/                    # MPOM dataset and download script
├── docs/                    # Mathematical methodology and loss function details
├── figures/                 # Architecture figures and visuals
├── notebooks/               # Demo notebook for visualization and experiments
├── src/                     # Main source code for models, training, utils
│   ├── models/              # Student, Teacher, Decoder, Discriminator modules
│   └── utils/               # Attention, diffusion, contrastive modules
├── README.md                # Project overview and usage instructions
├── LICENSE                  # MIT License
├── CITATION.cff            # BibTeX citation entry
├── requirements.txt         # Python dependencies (pip)
└── environment.yml          # Conda environment setup
```

## 🚀 Installation

Using Conda (recommended):

```bash
conda env create -f environment.yml
conda activate ikddit
```

Using pip:

```bash
pip install -r requirements.txt
```

## 📦 Dataset Setup

To download and prepare the MPOM dataset:

```bash
cd data && bash download_mpom.sh && cd ..
```

## 🏋️‍♂️ Training

Train the full IKDDiT model (Teacher, Student, Discriminator):

```bash
python src/train.py --config configs/ikddit_s.yaml
```

### 🔧 Hyperparameters

- `mask_ratio`: proportion of image patches masked in Student Encoder (default: `0.5`)
- `lambda1`, `lambda2`: weights for MAE and discriminator losses

#### 📊 Mask Ratio vs FID-15k (Ablation Study)

| Mask Ratio | FID-15k |
|------------|---------|
| 0.0        | 27.46   |
| 0.25       | 26.06   |
| 0.5        | 24.66   |
| 0.75       | 123.85  |

**Best performance at 50% masking.**

## 📐 Loss Function (Eq. 8)

```math
L_{IKDDiT} = L_{DSM} + λ₁ L_{MAE} + λ₂ L_{D}
```

- `L_DSM`: Denoising Score Matching loss
- `L_MAE`: Mean Absolute Error (L1)
- `L_D`: Discriminator loss for student-teacher alignment

All losses and parameters are configurable via `configs/ikddit_s.yaml`.

## 🔍 Inference

Fast inference using only the Student Encoder + Decoder:

```bash
python src/inference.py --model checkpoints/student_ikddit.pth --mask_ratio 0.5
```

## 📊 Visualization

Explore model behavior using the notebook:

```bash
notebooks/demo.ipynb
```

Includes:
- Training loss curves
- Discriminator alignment loss
- σ heatmap visualization
- Inference speed benchmarks

## 📖 Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{anonymous2025ikddit,
  title={Photolithography Overlay Map Generation with Implicit Knowledge Distillation Diffusion Transformer},
  author={Anonymous},
  booktitle={ICCV},
  year={2025}
}
```

## 📄 License

This project is licensed under the [MIT License](LICENSE).
