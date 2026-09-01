"""
Pix2Vox-A backbone (hzxie/Pix2Vox, A branch).

The expensive sibling of Pix2Vox-F: a wider encoder, a much wider decoder, and
an extra Refiner stage. 457 MB against F's 29.8 MB, and correspondingly slower.
That contrast is the point — a cheap backbone and a dear one that is genuinely
better give the policy's model head a real decision to make, which is exactly
what the compute-cost penalty exists to arbitrate.

Network code is the official A-branch encoder/decoder/merger/refiner with the
two config values inlined (TCONV_USE_BIAS=False, LEAKY_VALUE=0.2), matching
pix2vox_f.py. The released checkpoint reports best_iou 0.6855 at epoch 153.

Pipeline, following the repo's own core/test.py:

    encoder -> decoder -> merger -> refiner

Both merger and refiner are enabled (USE_MERGER and USE_REFINER default True,
and epoch 153 is well past the epochs at which the repo starts using them).

Preprocessing is identical to Pix2Vox-F, so it is imported rather than repeated.
"""

import os

import numpy as np
import torch
import torchvision.models

from . import VOXEL_RES, Backbone
from .pix2vox_f import LEAKY_VALUE, TCONV_USE_BIAS, Merger, _preprocess


class Encoder(torch.nn.Module):
    """A-branch encoder: wider heads than F, output (B, V, 256, 8, 8)."""

    def __init__(self):
        super().__init__()
        vgg16_bn = torchvision.models.vgg16_bn(weights=None)
        self.vgg = torch.nn.Sequential(*list(vgg16_bn.features.children()))[:27]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=3),
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU(),
        )

    def forward(self, rendering_images):
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
    """A-branch decoder: the 2D->3D bridge is 256x8x8 == 2048x2x2x2."""

    def __init__(self):
        super().__init__()

        def up(cin, cout):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose3d(cin, cout, kernel_size=4, stride=2,
                                         bias=TCONV_USE_BIAS, padding=1),
                torch.nn.BatchNorm3d(cout), torch.nn.ReLU(),
            )

        self.layer1 = up(2048, 512)
        self.layer2 = up(512, 128)
        self.layer3 = up(128, 32)
        self.layer4 = up(32, 8)
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1, bias=TCONV_USE_BIAS),
            torch.nn.Sigmoid(),
        )

    def forward(self, image_features):
        image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
        image_features = torch.split(image_features, 1, dim=0)
        gen_volumes, raw_features = [], []

        for features in image_features:
            gen_volume = features.view(-1, 2048, 2, 2, 2)
            gen_volume = self.layer1(gen_volume)
            gen_volume = self.layer2(gen_volume)
            gen_volume = self.layer3(gen_volume)
            gen_volume = self.layer4(gen_volume)
            raw_feature = gen_volume
            gen_volume = self.layer5(gen_volume)
            raw_feature = torch.cat((raw_feature, gen_volume), dim=1)

            gen_volumes.append(torch.squeeze(gen_volume, dim=1))
            raw_features.append(raw_feature)

        gen_volumes = torch.stack(gen_volumes).permute(1, 0, 2, 3, 4).contiguous()
        raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        return raw_features, gen_volumes


class Refiner(torch.nn.Module):
    """U-net style residual refinement of the merged 32**3 volume."""

    def __init__(self):
        super().__init__()

        def down(cin, cout):
            return torch.nn.Sequential(
                torch.nn.Conv3d(cin, cout, kernel_size=4, padding=2),
                torch.nn.BatchNorm3d(cout),
                torch.nn.LeakyReLU(LEAKY_VALUE),
                torch.nn.MaxPool3d(kernel_size=2),
            )

        def up(cin, cout):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose3d(cin, cout, kernel_size=4, stride=2,
                                         bias=TCONV_USE_BIAS, padding=1),
                torch.nn.BatchNorm3d(cout), torch.nn.ReLU(),
            )

        self.layer1 = down(1, 32)
        self.layer2 = down(32, 64)
        self.layer3 = down(64, 128)
        self.layer4 = torch.nn.Sequential(torch.nn.Linear(8192, 2048), torch.nn.ReLU())
        self.layer5 = torch.nn.Sequential(torch.nn.Linear(2048, 8192), torch.nn.ReLU())
        self.layer6 = up(128, 64)
        self.layer7 = up(64, 32)
        self.layer8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2,
                                     bias=TCONV_USE_BIAS, padding=1),
            torch.nn.Sigmoid(),
        )

    def forward(self, coarse_volumes):
        v32_l = coarse_volumes.view(-1, 1, VOXEL_RES, VOXEL_RES, VOXEL_RES)
        v16_l = self.layer1(v32_l)
        v8_l = self.layer2(v16_l)
        v4_l = self.layer3(v8_l)

        flat = self.layer5(self.layer4(v4_l.view(-1, 8192)))

        v4_r = v4_l + flat.view(-1, 128, 4, 4, 4)
        v8_r = v8_l + self.layer6(v4_r)
        v16_r = v16_l + self.layer7(v8_r)
        v32_r = (v32_l + self.layer8(v16_r)) * 0.5

        return v32_r.view(-1, VOXEL_RES, VOXEL_RES, VOXEL_RES)


class Pix2VoxA(Backbone):
    """Pretrained Pix2Vox-A. The expensive, higher-ceiling voxel backbone."""

    name = "pix2vox_a"
    cost = 1.0     # measured 1.34s / 5 views (CPU): 3.88x occnet, the dearest

    @staticmethod
    def _ckpt_path(cfg) -> str:
        return os.environ.get("PIX2VOX_A_CKPT", getattr(cfg, "pix2vox_a_ckpt", ""))

    @classmethod
    def available(cls, cfg) -> bool:
        return os.path.isfile(cls._ckpt_path(cfg))

    def __init__(self, cfg, device: str = "cpu"):
        self.device = device
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.merger = Merger()
        self.refiner = Refiner()

        ckpt = torch.load(self._ckpt_path(cfg), map_location="cpu", weights_only=False)

        def strip(sd):
            return {k.replace("module.", "", 1): v for k, v in sd.items()}

        for mod, key in ((self.encoder, "encoder_state_dict"),
                         (self.decoder, "decoder_state_dict"),
                         (self.merger, "merger_state_dict"),
                         (self.refiner, "refiner_state_dict")):
            mod.load_state_dict(strip(ckpt[key]), strict=True)

        for m in (self.encoder, self.decoder, self.merger, self.refiner):
            m.to(device).eval()
            for p in m.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def predict(self, images, cams=None) -> np.ndarray:
        views = _preprocess(images).to(self.device)
        raw_features, gen_volumes = self.decoder(self.encoder(views))
        volume = self.merger(raw_features, gen_volumes)
        volume = self.refiner(volume)
        out = volume.squeeze(0).cpu().numpy().astype(np.float32)
        assert out.shape == (VOXEL_RES,) * 3, f"unexpected output shape {out.shape}"
        return out
