"""
Pix2Vox-F backbone (hzxie/Pix2Vox, F branch).

The three network classes below are the official F-branch encoder/decoder/merger,
with the two config values they read inlined:

    NETWORK.TCONV_USE_BIAS = False
    NETWORK.LEAKY_VALUE    = 0.2

Those are the official defaults, and TCONV_USE_BIAS in particular is load-bearing:
with bias=True the decoder gains five bias tensors the released checkpoint does not
contain, and load_state_dict(strict=True) fails. That is the single reason the
earlier hand-written port in build/pix2vox_f.py never loaded.

VGG16-BN is built with weights=None because every weight, VGG trunk included,
comes from the checkpoint.

Input is 224x224. The preprocessing chain reproduces Pix2Vox's own test-time
transform (core/test.py) exactly, so the reported IoU is comparable to the
published number:

    CenterCrop((224,224), (128,128))  ->  RandomBackground([[240,240]]*3)
    ->  Normalize(mean=.5, std=.5)    ->  ToTensor

Output is 32**3, which is already the scoring resolution, so no resampling.
"""

import os

import cv2
import numpy as np
import torch
import torchvision.models

from . import VOXEL_RES, Backbone

TCONV_USE_BIAS = False
LEAKY_VALUE = 0.2


# ── Official F-branch network ────────────────────────────────────────────────

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg16_bn = torchvision.models.vgg16_bn(weights=None)
        self.vgg = torch.nn.Sequential(*list(vgg16_bn.features.children()))[:27]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=4),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 128, kernel_size=3),
            torch.nn.BatchNorm2d(128),
            torch.nn.ELU(),
        )

    def forward(self, rendering_images):
        # (B, V, 3, 224, 224) -> (B, V, 128, 4, 4)
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []
        for img in rendering_images:
            features = self.vgg(img.squeeze(dim=0))
            features = self.layer1(features)
            features = self.layer2(features)
            features = self.layer3(features)
            image_features.append(features)
        return torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()


class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2,
                                     bias=TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(128), torch.nn.ReLU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2,
                                     bias=TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(64), torch.nn.ReLU(),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2,
                                     bias=TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(32), torch.nn.ReLU(),
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2,
                                     bias=TCONV_USE_BIAS, padding=1),
            torch.nn.BatchNorm3d(8), torch.nn.ReLU(),
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=TCONV_USE_BIAS),
            torch.nn.Sigmoid(),
        )

    def forward(self, image_features):
        # (B, V, 128, 4, 4) -> raw (B, V, 9, 32, 32, 32), gen (B, V, 32, 32, 32)
        image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
        image_features = torch.split(image_features, 1, dim=0)
        gen_voxels, raw_features = [], []

        for features in image_features:
            # 128 channels x 4 x 4 == 2048 == 256 x 2 x 2 x 2: the 2D -> 3D bridge
            gen_voxel = features.view(-1, 256, 2, 2, 2)
            gen_voxel = self.layer1(gen_voxel)
            gen_voxel = self.layer2(gen_voxel)
            gen_voxel = self.layer3(gen_voxel)
            gen_voxel = self.layer4(gen_voxel)
            raw_feature = gen_voxel
            gen_voxel = self.layer5(gen_voxel)
            raw_feature = torch.cat((raw_feature, gen_voxel), dim=1)

            gen_voxels.append(torch.squeeze(gen_voxel, dim=1))
            raw_features.append(raw_feature)

        gen_voxels = torch.stack(gen_voxels).permute(1, 0, 2, 3, 4).contiguous()
        raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        return raw_features, gen_voxels


class Merger(torch.nn.Module):
    def __init__(self):
        super().__init__()

        def block(cin, cout):
            return torch.nn.Sequential(
                torch.nn.Conv3d(cin, cout, kernel_size=3, padding=1),
                torch.nn.BatchNorm3d(cout),
                torch.nn.LeakyReLU(LEAKY_VALUE),
            )

        self.layer1 = block(9, 16)
        self.layer2 = block(16, 8)
        self.layer3 = block(8, 4)
        self.layer4 = block(4, 2)
        self.layer5 = block(2, 1)

    def forward(self, raw_features, coarse_volumes):
        # Score each view's volume, softmax across views, take the weighted sum.
        n_views = coarse_volumes.size(1)
        raw_features = torch.split(raw_features, 1, dim=1)
        volume_weights = []

        for i in range(n_views):
            volume_weight = torch.squeeze(raw_features[i], dim=1)
            volume_weight = self.layer1(volume_weight)
            volume_weight = self.layer2(volume_weight)
            volume_weight = self.layer3(volume_weight)
            volume_weight = self.layer4(volume_weight)
            volume_weight = self.layer5(volume_weight)
            volume_weights.append(torch.squeeze(volume_weight, dim=1))

        volume_weights = torch.stack(volume_weights).permute(1, 0, 2, 3, 4).contiguous()
        volume_weights = torch.softmax(volume_weights, dim=1)
        coarse_volumes = (coarse_volumes * volume_weights).sum(dim=1)
        return torch.clamp(coarse_volumes, min=0, max=1)


# ── Preprocessing: Pix2Vox's own test-time transform ─────────────────────────

IMG_SIZE = 224
CROP_SIZE = 128
BG_COLOR = 240 / 255.0      # TEST.RANDOM_BG_COLOR_RANGE is [[240,240]] * 3
MEAN = STD = 0.5


def _preprocess(images) -> torch.Tensor:
    """
    PIL images -> (1, V, 3, 224, 224) float32 tensor.

    Accepts RGBA (ShapeNet renderings, transparent background) or RGB. For RGBA
    the transparent pixels are composited onto the near-white background Pix2Vox
    tested against; for RGB the image is used as-is, since the background is
    already baked in.
    """
    processed = []
    for img in images:
        arr = np.asarray(img).astype(np.float32) / 255.0
        if arr.ndim == 2:                       # greyscale -> RGB
            arr = np.stack([arr] * 3, axis=-1)

        h, w = arr.shape[:2]
        if h > CROP_SIZE and w > CROP_SIZE:     # centre crop, then resize
            top = (h - CROP_SIZE) // 2
            left = (w - CROP_SIZE) // 2
            arr = arr[top:top + CROP_SIZE, left:left + CROP_SIZE]
        arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE))

        if arr.shape[2] == 4:                   # composite onto the background
            is_bg = (arr[:, :, 3:4] == 0).astype(np.float32)
            arr = is_bg * BG_COLOR + (1.0 - is_bg) * arr[:, :, :3]

        arr = (arr - MEAN) / STD
        processed.append(torch.from_numpy(arr.transpose(2, 0, 1).copy()))

    return torch.stack(processed).unsqueeze(0)


# ── Backbone wrapper ─────────────────────────────────────────────────────────

class Pix2VoxF(Backbone):
    """Pretrained Pix2Vox-F. Cheap and fast; the reference backbone."""

    name = "pix2vox_f"
    cost = 0.09    # measured 0.51s / 5 views (CPU): 1.26x occnet

    @staticmethod
    def _ckpt_path(cfg) -> str:
        return os.environ.get("PIX2VOX_F_CKPT", getattr(cfg, "pix2vox_f_ckpt", ""))

    @classmethod
    def available(cls, cfg) -> bool:
        return os.path.isfile(cls._ckpt_path(cfg))

    def __init__(self, cfg, device: str = "cpu"):
        self.device = device
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.merger = Merger()

        ckpt = torch.load(self._ckpt_path(cfg), map_location="cpu", weights_only=False)

        # Checkpoints saved from DataParallel carry a "module." prefix.
        def strip(sd):
            return {k.replace("module.", "", 1): v for k, v in sd.items()}

        # strict=True on purpose: a silent partial load here would poison the
        # reward with a half-initialised network and be very hard to notice.
        self.encoder.load_state_dict(strip(ckpt["encoder_state_dict"]), strict=True)
        self.decoder.load_state_dict(strip(ckpt["decoder_state_dict"]), strict=True)
        self.merger.load_state_dict(strip(ckpt["merger_state_dict"]), strict=True)

        for m in (self.encoder, self.decoder, self.merger):
            m.to(device).eval()
            for p in m.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def predict(self, images, cams=None) -> np.ndarray:
        views = _preprocess(images).to(self.device)
        features = self.encoder(views)
        raw_features, gen_voxels = self.decoder(features)
        volume = self.merger(raw_features, gen_voxels)
        out = volume.squeeze(0).cpu().numpy().astype(np.float32)
        assert out.shape == (VOXEL_RES,) * 3, f"unexpected output shape {out.shape}"
        return out
