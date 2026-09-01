"""
Occupancy Networks backbone (autonomousvision/occupancy_networks, img2mesh).

Network modules are vendored verbatim under _vendor/occnet (encoder, decoder,
layers) with only their `im2mesh.*` imports rewritten to local ones. The repo's
compiled extensions (libmesh, libmcubes, libsimplify) are deliberately NOT
vendored: those exist for mesh extraction, and we only need the occupancy field.

Architecture is fixed by configs/img/onet.yaml, which is what the released
checkpoint was trained with:

    encoder resnet18,  c_dim 256,  z_dim 0,  decoder cbatchnorm,  img 224

z_dim 0 means the model is deterministic — no latent sampling, just
encoder -> c -> decoder.

Why this one fits well
──────────────────────
OccNet's image model was trained on `img_choy2016` at 24 views: the very
renderings this project uses. So it is genuinely in-distribution, not
transferred.

It also predicts occupancy at *arbitrary* 3D points rather than on a fixed
grid, so producing our 32**3 scoring volume is a direct query rather than a
resample. Points follow the repo's own generation convention: a grid over
[-0.5, 0.5] scaled by box_size = 1 + padding = 1.1.

Multi-view
──────────
The released model is single-image conditioned; there is no official
multi-view variant. Views are fused by mean-pooling the per-view encodings
before decoding, which is the same trick the project's own ViewFusion uses.
This is a documented deviation from the published model, and it is why OccNet
here should not be expected to match its paper number exactly.

Coordinate frame
────────────────
OccNet's normalised mesh space and ShapeNetVox32's binvox space are not
guaranteed to share axis order or direction. A wrong convention produces a
perfectly-shaped volume that scores near zero IoU, so the mapping is an
explicit, measurable constant rather than an assumption — see AXES below and
`python -m backbones.bench --calibrate-axes`.
"""

import os

import numpy as np
import torch
from PIL import Image

from . import VOXEL_RES, Backbone
from ._vendor.occnet.decoder import DecoderCBatchNorm
from ._vendor.occnet.encoder_conv import Resnet18

C_DIM = 256
IMG_SIZE = 224
# Extent of the query grid in OccNet's normalised object space.
#
# im2mesh's mesh generation uses box_size = 1 + padding = 1.1, and copying that
# here was a mistake: the padding exists to give marching cubes room *around*
# the surface, but when scoring directly against a tight 32**3 voxel grid it
# shrinks the object inside the grid and throws away resolution. Measured on
# held-out chairs, 1.1 scored IoU 0.22 while the true optimum is a clean peak at
# 0.96 scoring 0.45 -- the single largest correctness fix to this backbone.
# Re-measure with a box sweep if the ground-truth voxelisation ever changes.
BOX_SIZE = 0.96
OCC_THRESHOLD = 0.2     # configs/img/onet.yaml test.threshold

# Axis mapping from OccNet's (x, y, z) query order to the binvox grid order,
# as (permutation, flips). Identity until measured; --calibrate-axes reports
# the permutation that maximises IoU on real data, and the value belongs here.
AXES = {"perm": (0, 1, 2), "flip": (False, False, False)}


def _occ_grid_points(res: int = VOXEL_RES) -> torch.Tensor:
    """
    The res**3 query points, in im2mesh's own ordering.

    make_3d_grid varies the first axis slowest and the last fastest, so the
    flat result reshapes directly to (res, res, res).
    """
    lin = torch.linspace(-0.5, 0.5, res)
    xs = lin.view(-1, 1, 1).expand(res, res, res)
    ys = lin.view(1, -1, 1).expand(res, res, res)
    zs = lin.view(1, 1, -1).expand(res, res, res)
    return BOX_SIZE * torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3)


def _apply_axes(volume: np.ndarray) -> np.ndarray:
    """Re-orient a predicted volume into the ground-truth grid convention."""
    out = np.transpose(volume, AXES["perm"])
    for axis, flip in enumerate(AXES["flip"]):
        if flip:
            out = np.flip(out, axis=axis)
    return np.ascontiguousarray(out)


def _preprocess(images) -> torch.Tensor:
    """
    PIL images -> (V, 3, 224, 224).

    Matches im2mesh's transform: Resize then ToTensor, with no explicit
    normalisation, because the Resnet18 encoder applies ImageNet normalisation
    internally. Alpha is dropped with .convert("RGB") exactly as the repo's
    ImagesField does, rather than composited, so transparent pixels keep
    whatever RGB the renderer left underneath.
    """
    tensors = []
    for img in images:
        img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        arr = np.asarray(img).astype(np.float32) / 255.0
        tensors.append(torch.from_numpy(arr.transpose(2, 0, 1).copy()))
    return torch.stack(tensors)


class OccNet(Backbone):
    """Pretrained Occupancy Networks (image-conditioned). Mid-cost, geometry-clean."""

    name = "occnet"
    cost = 0.0     # measured 0.35s / 5 views (CPU): the cheapest

    @staticmethod
    def _ckpt_path(cfg) -> str:
        return os.environ.get("OCCNET_CKPT", getattr(cfg, "occnet_ckpt", ""))

    @classmethod
    def available(cls, cfg) -> bool:
        return os.path.isfile(cls._ckpt_path(cfg))

    def __init__(self, cfg, device: str = "cpu"):
        self.device = device
        self.encoder = Resnet18(c_dim=C_DIM)
        self.decoder = DecoderCBatchNorm(dim=3, z_dim=0, c_dim=C_DIM)

        ckpt = torch.load(self._ckpt_path(cfg), map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt)

        def part(prefix):
            n = len(prefix)
            return {k[n:]: v for k, v in state.items() if k.startswith(prefix)}

        # strict=True: a partial load here yields a half-random network that
        # still returns plausible-looking volumes, which is far worse than a crash.
        self.encoder.load_state_dict(part("encoder."), strict=True)
        self.decoder.load_state_dict(part("decoder."), strict=True)

        for m in (self.encoder, self.decoder):
            m.to(device).eval()
            for p in m.parameters():
                p.requires_grad_(False)

        self._points = _occ_grid_points().to(device)

    @torch.no_grad()
    def predict(self, images, cams=None) -> np.ndarray:
        views = _preprocess(images).to(self.device)

        # Mean-pool per-view encodings: the decoder consumes a single latent.
        c = self.encoder(views).mean(dim=0, keepdim=True)

        logits = self.decoder(self._points.unsqueeze(0), None, c)
        probs = torch.sigmoid(logits).reshape(VOXEL_RES, VOXEL_RES, VOXEL_RES)

        out = _apply_axes(probs.cpu().numpy().astype(np.float32))
        assert out.shape == (VOXEL_RES,) * 3, f"unexpected output shape {out.shape}"
        return out
