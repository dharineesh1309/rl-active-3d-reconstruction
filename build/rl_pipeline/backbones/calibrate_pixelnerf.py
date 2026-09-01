"""
Measure pixelNeRF's two free parameters against real data.

    python -m backbones.calibrate_pixelnerf --models 5

Unlike the voxel backbones, pixelNeRF has no fixed output grid. Two things must
be pinned down before its predictions mean anything:

  WORLD_SCALE  how big the scoring cube is in world units. The renderer's mesh
               normalisation is not recorded in rendering_metadata.txt, so this
               cannot be derived -- only measured.
  AXES         the permutation/flip mapping the query grid onto binvox order.

Both fail silently: a wrong scale samples empty space or crops the object, and
a wrong axis order returns a correctly-shaped volume that scores near zero. So
this sweeps scale (re-querying the network, the expensive part) and then all 48
orientations per scale (free, it just re-indexes a cached volume), and reports
the best combination to write into backbones/pixelnerf.py.
"""

import argparse
import itertools
import sys

import numpy as np

from . import voxel_iou
from .pixelnerf import PixelNeRF

ORIENTATIONS = [(perm, flips)
                for perm in itertools.permutations(range(3))
                for flips in itertools.product([False, True], repeat=3)]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=int, default=5,
                   help="Models to average over (pixelNeRF is slow; keep this small)")
    p.add_argument("--views", type=int, default=3)
    p.add_argument("--scales", type=float, nargs="+",
                   default=[0.25, 0.35, 0.5, 0.7, 1.0, 1.4])
    p.add_argument("--split", default="test")
    p.add_argument("--categories", nargs="+", default=["chair"])
    args = p.parse_args(argv)

    from config import Config
    from dataloader_shapenet import build_shapenet

    cfg = Config()
    if not PixelNeRF.available(cfg):
        print(f"pixelnerf weights not found at {PixelNeRF._ckpt_path(cfg)}")
        return 1

    dataset = build_shapenet(split=args.split, categories=args.categories,
                             limit_per_category=args.models)
    backbone = PixelNeRF(cfg)
    view_idx = list(range(min(args.views, dataset.n_views)))

    print(f"\ncalibrating pixelnerf over {len(dataset)} models, "
          f"{len(view_idx)} views, {len(args.scales)} scales "
          f"x {len(ORIENTATIONS)} orientations\n")

    results = []
    for scale in args.scales:
        volumes = []
        for i in range(len(dataset)):
            item = dataset[i]
            pred = backbone.predict(
                [item["images"][v] for v in view_idx],
                [item["cams"][v] for v in view_idx],
                scale=scale,
            )
            volumes.append((pred, item["voxels"]))

        occupied = float(np.mean([(p > 0.5).mean() for p, _ in volumes]))
        best = max(
            (np.mean([voxel_iou(_orient(p, perm, flips), gt) for p, gt in volumes]),
             perm, flips)
            for perm, flips in ORIENTATIONS
        )
        results.append((best[0], scale, best[1], best[2]))
        print(f"  scale {scale:<5} best IoU {best[0]:.4f}  "
              f"perm={best[1]} flip={best[2]}   (mean predicted occupancy {occupied:.3f})")

    results.sort(reverse=True)
    iou, scale, perm, flips = results[0]
    print(f"\nbest: WORLD_SCALE = {scale}, AXES = "
          f"{{'perm': {perm}, 'flip': {flips}}}  ->  IoU {iou:.4f}")
    if iou < 0.05:
        print("\nIoU is near zero at every scale and orientation. That points at the\n"
              "camera convention rather than these two constants -- the poses\n"
              "reconstructed from rendering_metadata.txt do not agree with what the\n"
              "sn64 checkpoint expects. Predicted occupancy above tells you whether\n"
              "the field is empty (nothing sampled) or full (everything dense).")
    return 0


def _orient(volume, perm, flips):
    out = np.transpose(volume, perm)
    for axis, f in enumerate(flips):
        if f:
            out = np.flip(out, axis=axis)
    return np.ascontiguousarray(out)


if __name__ == "__main__":
    sys.exit(main())
