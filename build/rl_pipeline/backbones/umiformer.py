"""
UMIFormer backbone (GaryZhu1996/UMIFormer, ShapeNet).

The transformer member of the set. Where Pix2Vox fuses per-view CNN features by
attention over whole-image vectors, UMIFormer runs a ViT over every view and
then reasons across *tokens* from all views jointly ("decoupled intra-view and
inter-view" blocks), before a 3D-RETR style transformer decoder emits the
volume. Different inductive bias from both the CNN and the implicit-field
backbones, which is the point of including it.

Why this one rather than 3D-R2N2: it is PyTorch native, so the released
checkpoint loads directly instead of needing the model reimplemented from a
Theano parameter dump.

Architecture is pinned by the repo's config.py defaults, which the released
ShapeNet checkpoint was trained with — a DeiT-base distilled ViT encoder with
16 blocks in a [0,0,0,1]*4 intra/inter pattern, an STM merger, and a 3D-RETR
decoder at 32**3. See _vendor/umiformer/conf.py.

Preprocessing is *identical* to Pix2Vox's: CenterCrop((224,224),(128,128)),
composite the transparent background onto 240/255, then map [0,1] to [-1,1].
UMIFormer spells that last step `x * 2 - 1` where Pix2Vox writes
`(x - 0.5) / 0.5`, which is the same function, so `_preprocess` is imported
rather than duplicated.

Output is already 32**3 occupancy probability, so no resampling and no
threshold calibration.
"""

import os

import numpy as np
import torch

from . import VOXEL_RES, Backbone
from ._vendor.umiformer.conf import UMIFORMER as UMI_CFG
from ._vendor.umiformer.decoder import Decoder
from ._vendor.umiformer.encoder import Encoder
from ._vendor.umiformer.merger import Merger
from .pix2vox_f import _preprocess


class UMIFormer(Backbone):
    """Pretrained UMIFormer. Transformer multi-view voxel reconstruction."""

    name = "umiformer"
    cost = 1.28     # CPU seconds per 5-view prediction (config.py, cost v1)

    @staticmethod
    def _ckpt_path(cfg) -> str:
        return os.environ.get("UMIFORMER_CKPT", getattr(cfg, "umiformer_ckpt", ""))

    @classmethod
    def available(cls, cfg) -> bool:
        return os.path.isfile(cls._ckpt_path(cfg))

    def __init__(self, cfg, device: str = "cpu"):
        self.device = device
        self.encoder = Encoder(UMI_CFG)
        self.decoder = Decoder(UMI_CFG)
        self.merger = Merger(UMI_CFG)

        ckpt = torch.load(self._ckpt_path(cfg), map_location="cpu", weights_only=False)

        def strip(sd):
            return {k.replace("module.", "", 1): v for k, v in sd.items()}

        for mod, key in ((self.encoder, "encoder_state_dict"),
                         (self.decoder, "decoder_state_dict"),
                         (self.merger, "merger_state_dict")):
            mod.load_state_dict(strip(ckpt[key]), strict=True)

        for m in (self.encoder, self.decoder, self.merger):
            m.to(device).eval()
            for p in m.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def predict(self, images, cams=None) -> np.ndarray:
        views = _preprocess(images).to(self.device)     # (1, V, 3, 224, 224)
        features = self.encoder(views)                  # (1, V, P, D)
        context = self.merger(features)                 # (1, V*P -> P, D)
        volume = self.decoder(context)                  # sigmoid already applied

        out = volume.squeeze().cpu().numpy().astype(np.float32)
        assert out.shape == (VOXEL_RES,) * 3, f"unexpected output shape {out.shape}"
        return out
