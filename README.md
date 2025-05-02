# Implicit Knowledge Distillation Diffusion Transformer (IKDDiT)

本專案實現 IKDDiT：基於隱式鑑別器的 Teacher–Student Diffusion Transformer，用於光刻 Overlay Map 生成，具備計算加速優勢。

## 核心特點
- **Implicit Discriminator**：訓練階段，Teacher DiT 查看完整 image patch；Student DiT 僅見可見 patch (其餘 masked)，通過隱式鑑別器對齊 token。
- **Inference 加速**：只運行 Student DiT，對少量 non-masked patch 進行去噪與重建，顯著縮短推理時間。
- **Unified Contrastive Embedding**：將影像、設備 log 與 ID 條碼嵌入同一空間，並以 InfoNCE loss 對齊。
- **Gated Cross-Attention**：融合條件 token 與 latent map，提高重建品質。

## Repository Structure
詳見目錄結構。

## Installation
```bash
conda env create -f environment.yml
conda activate ikddit
``` 
或
```bash
pip install -r requirements.txt
```

## Training
1. 下載資料：
   ```bash
   cd data && bash download_mpom.sh && cd ..
   ```
2. 訓練模型（Teacher + Student + Implicit Discriminator）：
   ```bash
   python src/train.py --config configs/ikddit_s.yaml
   ```

## Hyperparameters
- `mask_ratio` (float): percentage of patches to mask during student encoding (default: 0.5).  
- Ablation study on `mask_ratio` (FID-15k):

  | Mask Ratio | FID-15k |
  | ---------- | ------- |
  | 0%         | 27.46   |
  | 25%        | 26.06   |
  | 50%        | 24.66   |
  | 70%        | 123.85  |

  Optimal performance achieved at 50% mask ratio.

## Loss Function
- **Eq.8:** L_IKDDiT = L_DSM + λ1 * L_MAE + λ2 * L_D

三項 loss：
1. DSM: Denoising Score Matching。
2. MAE: Mean Absolute Error (L1)，負責重建誤差。
3. Discriminator Loss: 隱式鑑別器對齊訊號。

超參數：
- `lambda1` (float): MAE 權重。
- `lambda2` (float): Discriminator 權重。

請在 configs/ikddit_s.yaml 中設定以上參數。

## Inference
只啟動 Student DiT Encoder + Decoder：
```bash
python src/inference.py --model checkpoints/student_ikddit.pth --mask_ratio 0.5
```

## Visualization
- `notebooks/demo.ipynb`：顯示 training 中的 alignment loss、inference 中 σ heatmap 與加速比。

## Citation
```bibtex
@inproceedings{anonymous2025ikddit,
  title={Photolithography Overlay Map Generation with Implicit Knowledge Distillation Diffusion Transformer},
  author={Anonymous},
  booktitle={ICCV},
  year={2025}
}
```