"""
Latency on this machine's device: components, and the frozen evaluation
episodes end to end.

    python bench_latency.py --policy artifacts/tier1/set_pose_s164864.pt \
                            --router artifacts/router/router.npz \
                            --views artifacts/phase3/views_dev.json \
                            --out artifacts/phase3/latency_cpu.json

Cohort: `bench_cohort()` -- the first 2 dev objects of every category,
interleaved across categories, all 24 views each, so the policy sees the
experiment's 24 candidates. On a machine without the full renderings, point
SHAPENET_ROOT at the cohort the Phase 3 Kaggle job exports.

Episodes are the FROZEN evaluation ones (views_dev.json): per object, the first
--starts policy and heuristic episodes from their recorded start views, and the
same number of its recorded random view sets. Each measurement records the
start, the views and the backbone actually run, and whether the policy and the
heuristic reproduced their frozen views on this device.

Pipelines, 15 deployable ones: {random, heuristic, policy} x {pix2vox_f,
umiformer, umiformer_plus, router_plain, router_corrected}. Each is split into
controller seconds (view choice, ResNet features, routing; a view's features
paid once) and backbone seconds. Components are timed too:

    feature    ResNet-50 on one view: transform + forward + copy back
    policy     one view decision, its features already computed
    heuristic  farthest-angle choice of 4 views from a start (poses only)
    router_*   one routing decision, the 5 views' features already computed
    <backbone> one 5-view prediction

Every raw measurement is kept, with the thread settings; the device is
synchronised around each timed call; one untimed warm-up object runs first;
model loading is timed and reported on its own. Summaries give means and
medians -- the evaluation's cost term uses means, with object weights.
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


def bench_cohort(per_category=2):
    """First `per_category` dev objects of each category, interleaved."""
    from splits import load as load_splits, taxonomy

    dev = load_splits()["dev"]
    tax, _ = taxonomy()
    lists = [[m for m in e["test"] if m in dev][:per_category] for e in tax]
    return [l[i] for i in range(per_category) for l in lists if i < len(l)]


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
    ap.add_argument("--views", default="artifacts/phase3/views_dev.json")
    ap.add_argument("--per-category", type=int, default=2)
    ap.add_argument("--starts", type=int, default=4, help="frozen episodes per object")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from backbones import _candidates
    from config import Config
    from dataloader_shapenet import build_shapenet
    from env.rgb_view_env import RGBViewEnv, ResNetFeatures, stack_obs, to_tensor
    from infer_rgb import pick_backbone
    from phase3_views import farthest_angle
    from policy.pose_policy import RGBPosePolicy, load_pose_norm, pose_descriptor
    from run_0b import ARMS, _One, greedy_views

    device = "cuda" if torch.cuda.is_available() else "cpu"
    frozen = json.load(open(args.views))["views"]
    cohort = bench_cohort(args.per_category)
    ds = build_shapenet(split="test", only_ids=set(cohort))
    order = {m: i for i, (_, m) in enumerate(ds.samples)}
    if set(order) != set(cohort):
        raise SystemExit(f"{len(set(cohort) - set(order))} cohort objects missing on disk")
    items = [ds[order[m]] for m in cohort]
    if len(items[0]["images"]) != 24:
        raise SystemExit("cohort objects must have all 24 views")
    d_mean, d_std = load_pose_norm()

    load = {}
    t, ck = timed(lambda: torch.load(args.policy, map_location=device, weights_only=False))
    t2, policy = timed(lambda: RGBPosePolicy(n_candidates=24, **ARMS[ck["arm"]]).to(device))
    policy.load_state_dict(ck["policy"])
    policy.eval()
    load["policy"] = t + t2
    load["resnet"], feats = timed(lambda: ResNetFeatures(device=device))
    r = dict(np.load(args.router))
    routers = {"router_plain": {**r, "variant": np.array("expected")},
               "router_corrected": {**r, "variant": np.array("corrected")}}
    bbs = {}
    for c in _candidates():
        load[c.name], bbs[c.name] = timed(lambda c=c: c(Config(), device=device))
    no_reward = lambda *_: 0.0

    comp = {k: [] for k in ("feature", "policy", "heuristic", *routers, *bbs)}
    paths = []
    for j, item in enumerate([items[0]] + items):           # first: untimed warm-up
        warm, m = j == 0, item["model_id"]
        fz = frozen[m]
        poses = pose_descriptor(item["cams"], d_mean, d_std)
        imgs = lambda vs: [item["images"][v] for v in vs]
        cams = lambda vs: [item["cams"][v] for v in vs]
        rec = lambda kind, t: warm or comp[kind].append({"object": m, "s": t})

        # ── components, on the first frozen policy episode ──
        start0 = fz["starts"][0]
        feats._cache.clear()
        for v in fz["policy"][0]:
            rec("feature", timed(lambda v=v: feats(item["images"][v]))[0])
        env = RGBViewEnv(_One(item), no_reward, view_budget=4, features=feats,
                         device=device, seed_initial_view=False)
        env._acquire(start0)
        env.step_count = 0
        obs = env._obs()
        while not env.done:
            with torch.no_grad():
                t, (a, _, _) = timed(lambda: policy.act(to_tensor(stack_obs([obs]), device),
                                                        greedy=True))
            rec("policy", t)
            obs, *_ = env.step(int(a))
        rec("heuristic", timed(lambda: farthest_angle(poses, start0))[0])
        F = np.stack([feats(item["images"][v], key=(m, v)) for v in env.selected])
        for name, rt in routers.items():
            rec(name, timed(lambda rt=rt: pick_backbone(rt, F))[0])
        for name, b in bbs.items():
            rec(name, timed(lambda b=b: b.predict(imgs(env.selected), cams(env.selected)))[0])

        # ── the frozen evaluation episodes, end to end ──
        for i in range(1 if warm else args.starts):
            for strategy in ("random", "heuristic", "policy"):
                for choice in (*bbs, *routers):
                    feats._cache.clear()
                    sync()
                    t0 = time.perf_counter()
                    if strategy == "random":
                        views = fz["random"][i]
                    elif strategy == "heuristic":
                        views = farthest_angle(poses, fz["starts"][i])
                    else:
                        views = greedy_views(policy, no_reward, feats, 4, device, item,
                                             fz["starts"][i])
                    name = choice
                    if choice in routers:
                        F = np.stack([feats(item["images"][v], key=(m, v)) for v in views])
                        name = pick_backbone(routers[choice], F)
                    sync()
                    t1 = time.perf_counter()
                    bbs[name].predict(imgs(views), cams(views))
                    sync()
                    t2 = time.perf_counter()
                    if not warm:
                        ref = fz[strategy][i] if strategy != "random" else views
                        paths.append({"object": m, "episode": i, "strategy": strategy,
                                      "start": None if strategy == "random" else fz["starts"][i],
                                      "views": sorted(views), "choice": choice,
                                      "backbone": name,
                                      "matches_frozen": sorted(views) == sorted(ref),
                                      "controller_s": t1 - t0, "backbone_s": t2 - t1,
                                      "total_s": t2 - t0})

    stat = lambda xs: {"mean": float(np.mean(xs)), "median": float(np.median(xs)),
                       "n": len(xs)}
    by = {}
    for p in paths:
        by.setdefault(f"{p['strategy']}+{p['choice']}", []).append(p)
    out = {"device": device,
           "device_name": (torch.cuda.get_device_name(0) if device == "cuda"
                           else platform.processor() or platform.machine()),
           "torch": torch.__version__,
           "threads": {"torch": torch.get_num_threads(),
                       "interop": torch.get_num_interop_threads(),
                       "cpu_count": os.cpu_count(),
                       "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
           "cohort": cohort, "episodes_per_object": args.starts, "candidates": 24,
           "views_file": Path(args.views).name,
           "frozen_mismatches": {s: sum(not p["matches_frozen"] for p in paths
                                        if p["strategy"] == s)
                                 for s in ("policy", "heuristic")},
           "load_s": load,
           "components": {k: stat([x["s"] for x in v]) for k, v in comp.items()},
           "pipelines": {k: {f: stat([p[f] for p in v])
                             for f in ("controller_s", "backbone_s", "total_s")}
                         for k, v in by.items()},
           "raw": {"components": comp, "paths": paths}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{device} ({out['device_name']}), threads {out['threads']}, "
          f"{len(cohort)} objects x {args.starts} frozen episodes; "
          f"frozen-view mismatches {out['frozen_mismatches']}")
    print("components, mean ms: " + ", ".join(
        f"{k} {v['mean']*1e3:.1f}" for k, v in out["components"].items()))
    print(f"  {'pipeline':<28} {'controller':>11} {'backbone':>10} {'total':>9}  (mean s)")
    for k, v in out["pipelines"].items():
        print(f"  {k:<28} {v['controller_s']['mean']:>11.3f} "
              f"{v['backbone_s']['mean']:>10.3f} {v['total_s']['mean']:>9.3f}")
    print("load (s): " + ", ".join(f"{k} {v:.2f}" for k, v in load.items()))


if __name__ == "__main__":
    main()
