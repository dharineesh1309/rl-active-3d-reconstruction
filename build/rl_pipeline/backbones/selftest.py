"""
Backbone self-test: does each backbone load and produce a sane volume?

    python -m backbones.selftest

This is the cheap check — it proves the weights load strictly and the forward
pass runs at the right shape. It does NOT prove the weights are the right ones
or the preprocessing matches; that is what backbones.bench does by reproducing
the published IoU against real data.

Run from the rl_pipeline directory.
"""

import sys

import numpy as np
from PIL import Image

from . import VOXEL_RES, load_backbones, voxel_iou


def _fake_views(n=3, size=137, rgba=True):
    """Stand-in for ShapeNet renderings: 137x137, RGBA with a transparent border."""
    mode = "RGBA" if rgba else "RGB"
    imgs = []
    for _ in range(n):
        arr = np.random.randint(0, 255, (size, size, 4 if rgba else 3), dtype=np.uint8)
        if rgba:
            arr[..., 3] = 255
            arr[:20, :, 3] = 0          # transparent band, exercises compositing
        imgs.append(Image.fromarray(arr, mode=mode))
    return imgs


def check_iou_helper():
    gt = (np.random.rand(VOXEL_RES, VOXEL_RES, VOXEL_RES) > 0.7).astype(np.float32)
    assert voxel_iou(gt, gt) == 1.0, "identical grids must score IoU 1.0"
    assert voxel_iou(np.zeros_like(gt), gt) == 0.0, "empty prediction must score 0.0"
    assert voxel_iou(np.zeros_like(gt), np.zeros_like(gt)) == 0.0, "empty/empty must not divide by zero"
    print("  voxel_iou: OK")


def main():
    from config import Config

    cfg = Config()
    check_iou_helper()

    backbones = load_backbones(cfg)
    print(f"\n{len(backbones)} backbone(s) loaded\n")

    for b in backbones:
        for n_views in (1, 3, 5):
            out = b.predict(_fake_views(n_views))
            assert out.shape == (VOXEL_RES,) * 3, \
                f"{b.name}: expected {(VOXEL_RES,)*3}, got {out.shape}"
            assert out.dtype == np.float32, f"{b.name}: expected float32, got {out.dtype}"
            assert np.isfinite(out).all(), f"{b.name}: produced NaN or inf"
            assert 0.0 <= out.min() and out.max() <= 1.0, \
                f"{b.name}: values outside [0,1] — is a sigmoid missing?"
            print(f"  {b.name:<12} views={n_views}  shape={out.shape}  "
                  f"occ@0.4={float((out > 0.4).mean()):.4f}")

        # RGB input (no alpha) must work too — the ModelNet renders have no alpha.
        out = b.predict(_fake_views(3, rgba=False))
        assert out.shape == (VOXEL_RES,) * 3
        print(f"  {b.name:<12} RGB input: OK")

    print("\nAll backbone self-tests passed.")


if __name__ == "__main__":
    sys.exit(main())
