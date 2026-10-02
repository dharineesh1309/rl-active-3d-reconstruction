"""
Per-component latency, warm, on this machine's device.

    python bench_latency.py --policy artifacts/tier1/set_pose_s164864.pt \
                            --router artifacts/router/router.npz \
                            --out artifacts/phase3/latency_cpu.json

Each component is timed on its own, after one untimed warm-up object, with the
device synchronised around every call; medians over objects:

    feature    ResNet-50 on one view: transform + forward + copy back
    policy     one view decision, its features already computed
    heuristic  one farthest-angle choice of 4 views from the start (poses only)
    router     one routing decision, the 5 views' features already computed
    <backbone> one 5-view prediction

plus each model's load time, reported separately. phase3_eval.py composes
per-strategy totals from these, counting a view's features once however many
components use them.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed(fn):
    sync()
    t = time.perf_counter()
    out = fn()
    sync()
    return time.perf_counter() - t, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--router", default="artifacts/router/router.npz")
    ap.add_argument("--objects", type=int, default=20)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from backbones import _candidates
    from config import Config
    from dataloader_shapenet import build_shapenet
    from env.rgb_view_env import RGBViewEnv, ResNetFeatures, stack_obs, to_tensor
    from infer_rgb import pick_backbone
    from phase3_views import farthest_angle
    from policy.pose_policy import RGBPosePolicy, load_pose_norm, pose_descriptor
    from run_0b import ARMS, _One

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_shapenet(split="test", limit_per_category=args.objects + 1)
    items = [ds[i] for i in range(min(args.objects + 1, len(ds)))]   # +1: warm-up
    nv = len(items[0]["images"])
    d_mean, d_std = load_pose_norm()

    load = {}
    t, ck = timed(lambda: torch.load(args.policy, map_location=device, weights_only=False))
    t2, policy = timed(lambda: RGBPosePolicy(n_candidates=nv, **ARMS[ck["arm"]]).to(device))
    policy.load_state_dict(ck["policy"])
    policy.eval()
    load["policy"] = t + t2
    load["resnet"], feats = timed(lambda: ResNetFeatures(device=device))
    router = np.load(args.router)
    bbs = {}
    for c in _candidates():
        load[c.name], bbs[c.name] = timed(lambda c=c: c(Config(), device=device))

    rec = {k: [] for k in ("feature", "policy", "heuristic", "router", *bbs)}
    for j, item in enumerate(items):
        warm = j == 0
        feats._cache.clear()
        for v in range(5):
            t, _ = timed(lambda v=v: feats(item["images"][v]))
            warm or rec["feature"].append(t)

        env = RGBViewEnv(_One(item), lambda *_: 0.0, view_budget=4, features=feats,
                         device=device, seed_initial_view=False)
        env._acquire(0)
        env.step_count = 0
        obs = env._obs()
        while not env.done:
            with torch.no_grad():
                t, (a, _, _) = timed(lambda: policy.act(to_tensor(stack_obs([obs]), device),
                                                        greedy=True))
            warm or rec["policy"].append(t)
            obs, *_ = env.step(int(a))

        poses = pose_descriptor(item["cams"], d_mean, d_std)
        t, _ = timed(lambda: farthest_angle(poses, 0))
        warm or rec["heuristic"].append(t)
        F = np.stack([feats(item["images"][v], key=(item["model_id"], v))
                      for v in env.selected])
        t, _ = timed(lambda: pick_backbone(router, F))
        warm or rec["router"].append(t)
        imgs = [item["images"][v] for v in env.selected]
        cams = [item["cams"][v] for v in env.selected]
        for name, b in bbs.items():
            t, _ = timed(lambda b=b: b.predict(imgs, cams))
            warm or rec[name].append(t)

    out = {"device": device,
           "device_name": (torch.cuda.get_device_name(0) if device == "cuda"
                           else platform.processor() or platform.machine()),
           "torch": torch.__version__, "objects": len(items) - 1, "views_per_object": nv,
           "median_s": {k: float(np.median(v)) for k, v in rec.items()},
           "mean_s": {k: float(np.mean(v)) for k, v in rec.items()},
           "load_s": load}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{device} ({out['device_name']}), {out['objects']} objects, medians:")
    for k, v in out["median_s"].items():
        print(f"  {k:<15} {v*1e3:9.2f} ms")
    print("load: " + ", ".join(f"{k} {v:.2f}s" for k, v in load.items()))


if __name__ == "__main__":
    main()
