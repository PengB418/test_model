'''
DBAT（Dynamic Blink-Aware Temporal module）

Description:
Input frames/Feature map: T × H × W × 3   / T x N x C
Output Feature Map: T x N x C
With temporal aggregation: 1 x N x C

DBAT block combine with each backbone block, as pre-stage temporal attention.
Frame diff block is put at the beginning of T frames input, then concat with original RGB frames as backbone input.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

'''
Motion Encoder (learnable frame difference)
Instead of raw |I_t - I_{t-1}|, use learnable motion encoding M_t = Conv([I_t, I_{t-1}, |I_t - I_{t-1}|])
'''
class MotionEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_t, x_prev):
        # x_t, x_prev: [B, C, H, W]
        diff = torch.abs(x_t - x_prev)
        x = torch.cat([x_t, x_prev, diff], dim=1)
        return self.conv(x)

'''
Soft Noise Suppression Module
Original: Threshold filter D̃_t = β·(D_t>α)
Replaces hard threshold α with soft attention mask
'''
class SoftMask(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.mask_net = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        mask = self.mask_net(x)
        return x * mask

'''
Temporal Token Pooling
Convert spatial feature → token
'''
class TokenPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: [B, C, H, W]
        return x.mean(dim=[2, 3])  # [B, C]

'''
Short-term Temporal Attention (3–5 frames) 
输入一个时间序列特征 [B,T,C]，通过多头自注意力让每个时间步与所有时间步交互，输出同样大小的增强特征 [B,T,C]
batch size, token sequence length, feature dimension
'''
class ShortTermAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, x):
        # x: [B, T, C] output: [B, T, C]
        out, _ = self.attn(x, x, x)  #attn(query, key, value) -> self attention, output, attn_weights
        return out

'''
Long-term Frequency-aware Attention
inject sinusoidal bias (learnable periodic prior).
'''
class LongTermAttention(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.freq = nn.Parameter(torch.tensor(1.0))  # learnable frequency

    def temporal_bias(self, T, device):
        t = torch.arange(T, device=device).float()
        dist = t[None, :] - t[:, None]
        return torch.sin(self.freq * dist)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape

        bias = self.temporal_bias(T, x.device)  # [T, T]

        # NOTE: PyTorch MHA doesn't support bias directly,
        # so we approximate via additive embedding
        x = x + bias.mean(dim=1).unsqueeze(0).unsqueeze(-1)

        out, _ = self.attn(x, x, x)
        return out

'''
Cross-Attention Fusion (replaces GRU)
'''
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, short_feat, long_feat):
        # both: [B, T, C]

        attn_out, _ = self.attn(short_feat, long_feat, long_feat)

        gate = torch.sigmoid(self.gate(
            torch.cat([short_feat, long_feat], dim=-1)
        ))

        return gate * attn_out + (1 - gate) * short_feat

'''
Full DBAT++ Module
'''
class DBATPP(nn.Module):
    def __init__(self, backbone, dim):
        super().__init__()

        self.backbone = backbone  # MobileViT / UniFormer backbone

        self.motion_encoder = MotionEncoder(dim)
        self.soft_mask = SoftMask(dim)

        self.pool = TokenPool()

        self.short_attn = ShortTermAttention(dim)
        self.long_attn = LongTermAttention(dim)

        self.fusion = CrossAttentionFusion(dim)

        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, 3)  # ON / OFF / BLINK
        )

    def forward(self, video):
        """
        video: [B, T, C, H, W]
        """

        B, T, C, H, W = video.shape
        features = []

        # --- spatial encoding per frame ---
        for t in range(T):
            if t == 0:
                feat = self.backbone(video[:, t])
            else:
                feat = self.motion_encoder(
                    self.backbone(video[:, t]),
                    self.backbone(video[:, t - 1])
                )

            feat = self.soft_mask(feat)
            features.append(self.pool(feat))

        features = torch.stack(features, dim=1)  # [B, T, C]

        # --- temporal modeling ---
        short = self.short_attn(features[:, -3:])   # last 3 frames
        long = self.long_attn(features)

        fused = self.fusion(short, long)

        # --- prediction ---
        out = self.classifier(fused.mean(dim=1))

        return out