# 方法論細節

## 1. 模型架構
- Teacher DiT Encoder: 全域 self-attn + cross-attn
- Student DiT Encoder: 自注意 + Gated Cross-Attention
- Student Decoder: 去噪重建 + LayerNorm + Linear

## 2. 擾動流程
- **Forward (q_sample)**: 方程式 Eq.1
- **Reverse (p_sample)**: 方程式 Eq.2

## 3. 損失函數 (Eq.8)
L_IKDDiT = L_DSM + λ1 L_MAE + λ2 L_D

## 4. 超參數設定
- mask_ratio: 0.5
- λ1: 1.0, λ2: 0.1

## 5. Ablation Study
見 Table 7: 探討 mask_ratio 對 FID-15k 的影響。

## 6. 算法流程圖
```pseudo
for epoch...
  for batch...
    z_t = q_sample(x, t)
    recon, d_loss = model(z_t, context, mask_ratio, t)
    loss = l_dsm + λ1*l_mae + λ2*d_loss
    backprop...
```
