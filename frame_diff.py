'''
Frame Difference Block + RGB Concatenation Module

    Input:
        video : [B,T,3,H,W]

    Output:
        video_diff : [B,T,6,H,W]  (RGB + |I_t - I_{t-1}|)
'''

import torch
import torch.nn as nn

class FrameDiffConcat(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, video):
        """
        video: [B, T, 3, H, W]
        return: [B, T, 6, H, W]
        """

        B, T, C, H, W = video.shape

        # motion difference tensor
        diff = torch.zeros_like(video)

        # temporal frame difference
        diff[:, 1:] = torch.abs(video[:, 1:] - video[:, :-1])

        # concatenate RGB + motion
        out = torch.cat([video, diff], dim=2)

        return out