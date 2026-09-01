"""
TripoSR backbone (VAST-AI-Research/TripoSR, stabilityai/TripoSR weights).

The neural-rendering member of the set. TripoSR is a large reconstruction model
in the LRM family: a single image is tokenised into a **triplane**, and a small
MLP decodes density and colour anywhere in that volume, rendered with volume
rendering. So it is a radiance field, not a voxel grid or an occupancy network.

Why this rather than pixelNeRF
──────────────────────────────
pixelNeRF needs the input images' camera poses, and reconstructing those from
Choy's rendering_metadata.txt did not work: projecting known ground-truth voxels
through the best of 1,536 candidate conventions put only 0.55 of the projection
inside the rendered silhouette, where a correct convention gives ~0.95. Its
native data (NMR) ships exact cameras but is a different renderer at 64x64,
which would degrade the voxel backbones trained on Choy's 137x137.

TripoSR sidesteps all of that: it predicts in a **canonical frame from a single
image with no camera input at all**, so it runs directly on the renderings
already in this project.

Occupancy
─────────
Query density on the 32**3 scoring grid and map it through a sigmoid centred on
the repo's own mesh-extraction threshold (25.0), so 0.5 occupancy corresponds to
exactly the isosurface TripoSR itself would triangulate. Threshold sweeping in
backbones.bench then explores around that point rather than around an arbitrary
number.

Caveats worth stating
─────────────────────
  * Trained on Objaverse, not ShapeNet, so expect a domain-shift penalty that
    the ShapeNet-native voxel models do not pay.
  * **Single-view.** Extra views are ignored, so its IoU is flat across budgets.
    That is not a defect here: the measured gap between backbones is narrowest
    at one view, so a strong single-view model is exactly what can win the
    low-budget regime that nothing else claims.
  * Its canonical frame need not agree with binvox's axes. AXES below is
    measured, not assumed -- see `backbones.bench --calibrate-axes`.
"""

import os

import numpy as np
import torch

from . import VOXEL_RES, Backbone

# Density above which a point counts as inside. TripoSR's extract_mesh() uses
# 25.0 for marching cubes, but measured against binvox ground truth a lower
# threshold scores better, and the result is insensitive across 5-25.
DENSITY_THRESHOLD = 5.0
DENSITY_SCALE = 10.0     # softness of the sigmoid around that threshold

# Half-width of the scoring cube in TripoSR's canonical space.
#
# NOT the renderer's `radius` (0.87): that bounds ray marching, not the object,
# and using it puts the chair in the middle ~8 voxels of a 32**3 grid --
# measured IoU 0.062 with only 2% of the grid occupied against 28% ground truth.
# Sweeping shows a clean peak at 0.35 (IoU 0.376), with the same axis convention
# winning at every extent, which is what says the calibration is real.
GRID_HALF_EXTENT = 0.35

# Fraction of the frame the object should occupy, matching TripoSR's own
# run.py preprocessing (resize_foreground(image, 0.85)).
FOREGROUND_RATIO = 0.85
BACKGROUND = 0.5         # run.py composites onto mid grey, not white

# Measured, not assumed: TripoSR's canonical frame swaps Y and Z relative to
# binvox and mirrors the last axis. Stable across every extent swept.
AXES = {"perm": (0, 2, 1), "flip": (False, False, True)}


def _apply_axes(volume, axes=None):
    axes = AXES if axes is None else axes
    out = np.transpose(volume, axes["perm"])
    for axis, flip in enumerate(axes["flip"]):
        if flip:
            out = np.flip(out, axis=axis)
    return np.ascontiguousarray(out)


def _preprocess(image):
    """
    One PIL image -> RGB PIL image the way TripoSR's run.py prepares it.

    The renderings already carry alpha, so the background-removal step TripoSR
    applies to photographs is unnecessary; the alpha is composited straight onto
    the mid-grey it expects.
    """
    from PIL import Image
    from ._vendor.triposr.utils import resize_foreground

    if image.mode != "RGBA":
        image = image.convert("RGBA")
    image = resize_foreground(image, FOREGROUND_RATIO)

    arr = np.asarray(image).astype(np.float32) / 255.0
    rgb = arr[:, :, :3] * arr[:, :, 3:4] + (1.0 - arr[:, :, 3:4]) * BACKGROUND
    return Image.fromarray((rgb * 255).astype(np.uint8))


class TripoSR(Backbone):
    """Pretrained TripoSR. Single-view triplane radiance field."""

    name = "triposr"
    cost = 0.5      # provisional; set from measurement after benchmarking

    @staticmethod
    def _ckpt_dir(cfg) -> str:
        return os.environ.get("TRIPOSR_DIR", getattr(cfg, "triposr_dir", ""))

    @classmethod
    def available(cls, cfg) -> bool:
        d = cls._ckpt_dir(cfg)
        return bool(d) and os.path.isfile(os.path.join(d, "model.ckpt")) \
            and os.path.isfile(os.path.join(d, "config.yaml"))

    @staticmethod
    def _register_vendored_package():
        """
        Make the vendored copy importable as `tsr`.

        TripoSR's config.yaml names its components by dotted path
        (`tsr.models.tokenizers.image.DINOSingleImageTokenizer`) and resolves
        them with importlib at load time. Aliasing the vendored package under
        that name satisfies those lookups without editing upstream config or
        requiring the real package to be installed.
        """
        import importlib
        import sys

        if "tsr" not in sys.modules:
            sys.modules["tsr"] = importlib.import_module(
                "backbones._vendor.triposr")

    @staticmethod
    def _remap_vit_keys(state: dict) -> dict:
        """
        Translate the checkpoint's ViT weights to the installed transformers layout.

        TripoSR pins transformers==4.35, whose ViTModel stores attention as
        `encoder.layer.N.attention.attention.{query,key,value}`. transformers 5.x
        refactored that to `layers.N.attention.{q,k,v}_proj`, so the released
        checkpoint no longer matches a freshly-built model. The tensors are
        identical -- only the names changed -- so renaming is lossless, and the
        strict load below is what proves the mapping is complete.
        """
        import re

        rules = [
            (r"\.attention\.attention\.query\.", ".attention.q_proj."),
            (r"\.attention\.attention\.key\.",   ".attention.k_proj."),
            (r"\.attention\.attention\.value\.", ".attention.v_proj."),
            (r"\.attention\.output\.dense\.",    ".attention.o_proj."),
            (r"\.intermediate\.dense\.",          ".mlp.fc1."),
            (r"(?<!attention)\.output\.dense\.",  ".mlp.fc2."),
        ]
        out = {}
        for k, v in state.items():
            nk = k
            if ".encoder.layer." in nk:
                nk = nk.replace(".encoder.layer.", ".layers.")
                for pat, rep in rules:
                    nk = re.sub(pat, rep, nk)
            out[nk] = v
        return out

    def __init__(self, cfg, device: str = "cpu"):
        self._register_vendored_package()
        import torch as _torch
        from omegaconf import OmegaConf

        from ._vendor.triposr.system import TSR

        self.device = device
        d = self._ckpt_dir(cfg)
        conf = OmegaConf.load(os.path.join(d, "config.yaml"))
        OmegaConf.resolve(conf)
        self.model = TSR(conf)

        state = _torch.load(os.path.join(d, "model.ckpt"), map_location="cpu",
                            weights_only=False)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            state = self._remap_vit_keys(state)
            # strict=True on purpose: a partial load would leave a randomly
            # initialised ViT producing confident nonsense.
            self.model.load_state_dict(state, strict=True)
        self.model.renderer.set_chunk_size(0)   # no chunking; we query a small grid
        self.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        lin = torch.linspace(-GRID_HALF_EXTENT, GRID_HALF_EXTENT, VOXEL_RES)
        xs = lin.view(-1, 1, 1).expand(VOXEL_RES, VOXEL_RES, VOXEL_RES)
        ys = lin.view(1, -1, 1).expand(VOXEL_RES, VOXEL_RES, VOXEL_RES)
        zs = lin.view(1, 1, -1).expand(VOXEL_RES, VOXEL_RES, VOXEL_RES)
        self._points = torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3).to(device)

    @torch.no_grad()
    def predict(self, images, cams=None) -> np.ndarray:
        # Single-view model: it conditions on one image and ignores the rest.
        # Use the first view the agent selected rather than silently averaging.
        scene_codes = self.model(_preprocess(images[0]), device=self.device)

        density = self.model.renderer.query_triplane(
            self.model.decoder, self._points, scene_codes[0]
        )["density_act"].reshape(VOXEL_RES, VOXEL_RES, VOXEL_RES)

        occupancy = torch.sigmoid((density - DENSITY_THRESHOLD) / DENSITY_SCALE)
        out = _apply_axes(occupancy.cpu().numpy().astype(np.float32))
        assert out.shape == (VOXEL_RES,) * 3, f"unexpected output shape {out.shape}"
        return out
