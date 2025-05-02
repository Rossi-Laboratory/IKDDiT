
# Methodology Details

## 1. Model Architecture
- **Teacher DiT Encoder**: Global self-attention + cross-attention.
- **Student DiT Encoder**: Self-attention + Gated Cross-Attention.
- **Student Decoder**: Denoising reconstruction with LayerNorm and Linear projection.

## 2. Diffusion Process
- **Forward (q_sample)**: Equation (1)
- **Reverse (p_sample)**: Equation (2)

## 3. Loss Function (Eq. 8)
L_IKDDiT = L_DSM + λ1 * L_MAE + λ2 * L_D

### Loss Terms:
- **L_DSM**: Denoising Score Matching loss.
- **L_MAE**: Mean Absolute Error between generated and ground truth patches.
- **L_D**: Discriminator loss to align student and teacher token representations.

## 4. Hyperparameter Settings
- `mask_ratio`: 0.5
- `lambda1`: 1.0
- `lambda2`: 0.1

## 5. Ablation Study
Refer to Table 7: Analyzes the effect of `mask_ratio` on FID-15k scores.

## 6. Algorithm Pseudocode

```pseudo
for epoch in epochs:
  for batch in dataloader:
    z_t = q_sample(x, t)
    recon, d_loss = model(z_t, context, mask_ratio, t)
    loss = l_dsm + λ1 * l_mae + λ2 * d_loss
    backpropagation...
```
