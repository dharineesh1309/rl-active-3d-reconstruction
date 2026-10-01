"""
Tier 1, step 1 (GPU, once): image features for every object the router sees.

    python extract_router_feats.py --cache cache/utility_cache.json \
                                   --out artifacts/router/feats.npz

Every object in the utility cache, all 24 views, through the same frozen
ResNet-50 and transform the view policy uses (RGBA -> RGB the same way), stored
float16 with each view's 5-D pose descriptor. After this the router trains on
CPU.

Model ids listed under two categories in datasets/ShapeNet.json are skipped.
There are 263 such ids in the official split, some in train under one category
and test under another, and the cache key carries no synset, so for those ids
both the render set and the cached utilities are ambiguous.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = Path(__file__).resolve().parent


def taxonomy_index():
    """model_id -> list of (synset, category, split)."""
    where = defaultdict(list)
    for e in json.load(open(HERE / "datasets" / "ShapeNet.json")):
        for s in ("train", "val", "test"):
            for m in e.get(s, []):
                where[m].append((e["taxonomy_id"], e["taxonomy_name"], s))
    return where


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache/utility_cache.json")
    ap.add_argument("--out", default="artifacts/router/feats.npz")
    ap.add_argument("--n-views", type=int, default=24)
    args = ap.parse_args()

    from dataloader_shapenet import _read_metadata
    from env.rgb_view_env import ResNetFeatures
    from policy.pose_policy import load_pose_norm, pose_descriptor

    root = (os.environ.get("SHAPENET_RENDERING_ROOT")
            or os.path.join(os.environ["SHAPENET_ROOT"], "ShapeNetRendering"))
    where = taxonomy_index()
    ids = sorted({k.split("|")[0] for k in json.load(open(args.cache))})
    keep = [m for m in ids if len(where.get(m, [])) == 1]
    print(f"{len(ids)} objects in cache, {len(ids) - len(keep)} cross-listed "
          f"skipped, {len(keep)} to extract", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rn = ResNetFeatures(device=device)
    d_mean, d_std = load_pose_norm()

    V = args.n_views
    feats = np.zeros((len(keep), V, 2048), np.float16)
    poses = np.zeros((len(keep), V, 5), np.float32)
    t0 = time.time()
    for i, m in enumerate(keep):
        synset = where[m][0][0]
        rd = Path(root) / synset / m / "rendering"
        cams = _read_metadata(rd / "rendering_metadata.txt", V)
        x = torch.stack([rn.tf(Image.open(rd / f"{v:02d}.png").convert("RGB"))
                         for v in range(V)]).to(device)
        with torch.no_grad():
            feats[i] = rn.model(x).reshape(V, -1).cpu().numpy()
        poses[i] = pose_descriptor(cams, d_mean, d_std)
        if (i + 1) % 200 == 0 or i + 1 == len(keep):
            print(f"  {i+1:>5}/{len(keep)}  {time.time()-t0:6.0f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, model_id=np.array(keep),
             category=np.array([where[m][0][1] for m in keep]),
             split=np.array([where[m][0][2] for m in keep]),
             feats=feats, poses=poses)
    print(f"wrote {out} ({out.stat().st_size/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
