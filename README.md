# seg_model

Input (3 × H × W)
│
▼
Encoder ──────────────────────────────────────────────────────────────
│  (0) Conv2d(3→16, k=3, s=2) + BN + Hardswish
│  (1) InvertedResidual(16→16, depthwise k=3, s=2, SE, ReLU)
│  (2) InvertedResidual(16→24, exp=72, depthwise k=3, s=2)
│  (3) InvertedResidual(24→24, exp=88, depthwise k=3, s=1)
│  (4) InvertedResidual(24→40, exp=96, depthwise k=5, s=2, SE)
│  (5) InvertedResidual(40→40, exp=240, depthwise k=5, s=1, SE)
│  (6) InvertedResidual(40→40, exp=240, depthwise k=5, s=1, SE)
│  (7) InvertedResidual(40→48, exp=120, depthwise k=5, s=1, SE)
│  (8) InvertedResidual(48→48, exp=144, depthwise k=5, s=1, SE)
│  (9) InvertedResidual(48→96, exp=288, depthwise k=5, s=2, SE)
│ (10) InvertedResidual(96→96, exp=576, depthwise k=5, s=1, SE)
│ (11) InvertedResidual(96→96, exp=576, depthwise k=5, s=1, SE)
│ (12) Conv2d(96→576, k=1, s=1) + BN + Hardswish
│
▼
Projection Layer: Conv2d(576→128, k=1, s=1)
│
▼
LinearAttentionBlock (128 channels)
│
▼
Decoder ──────────────────────────────────────────────────────────────
│  ConvTranspose2d(128→64, k=2, s=2) + BN + ReLU
│  ConvTranspose2d(64→32, k=2, s=2) + BN + ReLU
│  Conv2d(32→10, k=1, s=1)
│
▼
PrototypeAttention
│
▼
Output (10 × H_out × W_out)
