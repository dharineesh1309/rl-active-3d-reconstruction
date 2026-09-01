"""
reconstructor.py
────────────────
Model architecture for multi-view 3D reconstruction.
Contains only the network class definitions — no inference runner,
no visualisation, no data loading.

Imported by:
  env/view_recon_env.py  — to compute the IoU reward
  train.py               — (optionally) to verify the checkpoint loads before spawning workers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.InstanceNorm3d(channels),
            nn.ReLU(True),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.InstanceNorm3d(channels),
        )

    def forward(self, x):
        return F.relu(x + self.net(x))


class ViewFusion(nn.Module):
    """
    Attention-based fusion over V views.
    Accepts any number of views at inference time — the attention is
    purely feature-based (no positional dependency).
    """
    def __init__(self, n_views, feat_dim):
        super().__init__()
        self.attn = nn.Linear(feat_dim, 1)

    def forward(self, feats):
        # feats: (B, V, feat_dim)
        weights = self.attn(feats)                                   # (B, V, 1)
        weights = torch.softmax(weights.squeeze(-1), dim=1).unsqueeze(-1)  # (B, V, 1)
        return (feats * weights).sum(dim=1)                          # (B, feat_dim)


class Encoder(nn.Module):
    def __init__(self, feature_dim=512, img_size=256, n_views=24):
        super().__init__()
        effnet = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
        self.backbone = effnet.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self._inferred_feat_channels = self._infer_backbone_channels(img_size)
        self.fc = nn.Linear(self._inferred_feat_channels, feature_dim)

    def _infer_backbone_channels(self, img_size):
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            c = self.backbone(dummy).shape[1]
        return c

    def forward_single(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        return self.fc(torch.flatten(x, 1))

    def forward(self, views):
        return self.forward_single(views)


class Decoder(nn.Module):
    def __init__(self, feature_dim=512, voxel_size=64, start_channels=256):
        super().__init__()
        self.voxel_size = voxel_size
        self.start_size = 4
        self.start_ch   = start_channels

        self.fc = nn.Linear(feature_dim, self.start_ch * (self.start_size ** 3))

        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose3d(self.start_ch,      self.start_ch // 2,  4, 2, 1),
                nn.InstanceNorm3d(self.start_ch // 2),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.ConvTranspose3d(self.start_ch // 2, self.start_ch // 4,  4, 2, 1),
                nn.InstanceNorm3d(self.start_ch // 4),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.ConvTranspose3d(self.start_ch // 4, self.start_ch // 8,  4, 2, 1),
                nn.InstanceNorm3d(self.start_ch // 8),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.ConvTranspose3d(self.start_ch // 8, self.start_ch // 16, 4, 2, 1),
                nn.InstanceNorm3d(self.start_ch // 16),
                nn.ReLU(True),
            ),
        ])

        self.res_blocks = nn.ModuleList([
            ResBlock3D(self.start_ch // 2),
            ResBlock3D(self.start_ch // 4),
            ResBlock3D(self.start_ch // 8),
            ResBlock3D(self.start_ch // 16),
        ])

        final_channels = self.start_ch // 16
        self.final_conv = nn.Sequential(
            nn.Conv3d(final_channels, final_channels, 3, padding=1),
            nn.InstanceNorm3d(final_channels),
            nn.ReLU(True),
            nn.Conv3d(final_channels, 1, 1),
        )

    def forward(self, x):
        B = x.shape[0]
        x = self.fc(x).view(
            B, self.start_ch, self.start_size, self.start_size, self.start_size
        )
        for up, res in zip(self.up_blocks, self.res_blocks):
            x = res(up(x))
        return self.final_conv(x)


class EncoderDecoder(nn.Module):
    """
    Full reconstruction model.

    forward(views) → raw logits (B, 1, voxel_size, voxel_size, voxel_size)
    Apply torch.sigmoid() to get occupancy probabilities.
    """
    def __init__(self, feature_dim=512, voxel_size=64, img_size=256, n_views=24):
        super().__init__()
        self.encoder     = Encoder(feature_dim, img_size, n_views)
        self.view_fuser  = ViewFusion(n_views, feature_dim)
        self.decoder     = Decoder(feature_dim, voxel_size)

    def forward(self, views):
        # views: (B, V, C, H, W)
        B, V, C, H, W = views.shape
        feats = self.encoder(views.view(B * V, C, H, W)).view(B, V, -1)
        fused = self.view_fuser(feats)
        return self.decoder(fused)
