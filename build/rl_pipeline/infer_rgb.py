"""
RGB inference: images and camera poses in, a reconstruction out.
No ground truth is read unless --gt is given, and then only to score the
saved result afterwards.

    python infer_rgb.py --object-dir <ShapeNetRendering/<synset>/<model_id>> \
                        --policy artifacts/tier1/set_pose_s180000.pt \
                        --router artifacts/router/router.npz --out recon.npy
    ... --gt <ShapeNetVox32/<synset>/<model_id>/model.binvox>

1. Start from one view (--start), as in training.
2. The view policy picks its trained budget of further views, greedily, from
   images and poses. The budget comes from the checkpoint, not the command
   line: the policy and the router were both trained on that view count.
3. The router picks a backbone from the acquired views' image features.
4. Only that backbone is loaded and run.

Timing reports model loading separately, then one warm pass per stage (feature
cache cleared), so selection overhead and reconstruction are like for like.
"""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_object(obj_dir, n_views):
    from dataloader_shapenet import _read_metadata

    rd = Path(obj_dir) / "rendering"
    images = []
    for v in range(n_views):
        img = Image.open(rd / f"{v:02d}.png")
        img.load()
        images.append(img)
    return {"images": images, "model_id": Path(obj_dir).name,
            "cams": _read_metadata(rd / "rendering_metadata.txt", n_views)}


def pick_backbone(router, feats):
    """The two-stage router (train_router.py) on one view set's features."""
    from train_router import softmax

    s = ((feats.mean(0) - router["mu"]) / router["sd"]) @ router["W"]
    table = router["table"]
    if str(router["variant"]) == "expected":
        m = int((softmax(s[None], float(router["T"])) @ table).argmax())
    else:
        m = int(table[s.argmax()].argmax())
    return str(router["backbones"][m])


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object-dir", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--router", default="artifacts/router/router.npz")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n-views", type=int, default=24)
    ap.add_argument("--out", default="recon.npy")
    ap.add_argument("--gt", default=None, help="model.binvox, to score afterwards")
    args = ap.parse_args()

    from backbones import _candidates
    from config import Config
    from env.rgb_view_env import ResNetFeatures
    from policy.pose_policy import RGBPosePolicy
    from run_0b import ARMS, greedy_views

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t = time.time()
    ck = torch.load(args.policy, map_location=device, weights_only=False)
    budget = ck.get("budget", 4)           # Tier 0B checkpoints predate the key
    router = np.load(args.router)
    set_size = int(router["set_size"]) if "set_size" in router else budget + 1
    if set_size != budget + 1:
        raise SystemExit(f"router trained on {set_size}-view sets, policy acquires "
                         f"{budget + 1}")
    if not 0 <= args.start < args.n_views or args.n_views < budget + 1:
        raise SystemExit(f"need 0 <= start < n_views and n_views >= {budget + 1}")

    item = load_object(args.object_dir, args.n_views)
    policy = RGBPosePolicy(n_candidates=args.n_views, **ARMS[ck["arm"]]).to(device)
    policy.load_state_dict(ck["policy"])
    feats = ResNetFeatures(device=device)
    sync()
    t_load_ctrl = time.time() - t

    no_reward = lambda item, views: 0.0          # nothing here may see GT
    timings = []
    for _ in range(2):                           # cold, then warm
        feats._cache.clear()
        t0 = time.time()
        views = greedy_views(policy, no_reward, feats, budget, device, item, args.start)
        sync()
        t1 = time.time()
        F = np.stack([feats(item["images"][v], key=(item["model_id"], v)) for v in views])
        name = pick_backbone(router, F)
        timings.append((t1 - t0, time.time() - t1))

    t = time.time()
    backbone = {c.name: c for c in _candidates()}[name](Config(), device=device)
    sync()
    t_load_bb = time.time() - t
    imgs, cams = [item["images"][v] for v in views], [item["cams"][v] for v in views]
    recon = []
    for _ in range(2):
        t = time.time()
        pred = backbone.predict(imgs, cams)
        sync()
        recon.append(time.time() - t)
    np.save(args.out, pred)

    (sel, route), rec = timings[1], recon[1]
    print(f"views {views} -> {name}; saved {args.out} {pred.shape}")
    print(f"loading: policy + ResNet {t_load_ctrl:.2f}s, {name} {t_load_bb:.2f}s")
    print(f"warm, per object ({device}): view selection {sel:.3f}s "
          f"(policy + ResNet), router {route*1e3:.1f}ms, {name} {rec:.3f}s, "
          f"total {sel + route + rec:.3f}s")

    if args.gt:
        from backbones import voxel_iou
        from training.utility_envelope import DEFAULT_THRESHOLDS
        from utils import binvox_rw

        with open(args.gt, "rb") as f:
            gt = binvox_rw.read_as_3d_array(f).data.astype(np.float32)
        iou = voxel_iou(pred, gt, pred_thresh=DEFAULT_THRESHOLDS[name])
        lam = Config().cost_lambda
        print(f"scored: IoU {iou:.4f}, utility {iou - lam * backbone.cost:.4f}")


if __name__ == "__main__":
    main()
