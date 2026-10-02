"""
Latency on this machine's device: components, and complete pipelines end to end.

    python bench_latency.py --policy artifacts/tier1/set_pose_s164864.pt \
                            --router artifacts/router/router.npz \
                            --out artifacts/phase3/latency_cpu.json

Cohort: `bench_cohort()` -- the first 2 dev objects of every category,
interleaved across categories, all 24 views each, so the policy sees the
experiment's 24 candidates. On a machine without the full renderings, point
SHAPENET_ROOT at the cohort the Phase 3 Kaggle job exports.

Every measurement is kept (object, repeat, seconds), with the thread settings.
The device is synchronised around each timed call; one untimed warm-up object
runs first; model loading is timed and reported on its own.

Components (per call):
    feature    ResNet-50 on one view: transform + forward + copy back
    policy     one view decision, its features already computed
    heuristic  farthest-angle choice of 4 views from the start (poses only)
    router     one routing decision, the 5 views' features already computed
    <backbone> one 5-view prediction

Paths (end to end, --repeats times per object, features recomputed each time):
    {random, heuristic, policy} x {each fixed backbone, router}
each split into controller seconds (view choice, features, routing) and
backbone seconds, so a view's features are paid once however many stages use
them.
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
    ap.add_argument("--per-category", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=2)
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
    router = np.load(args.router)
    bbs = {}
    for c in _candidates():
        load[c.name], bbs[c.name] = timed(lambda c=c: c(Config(), device=device))
    no_reward = lambda *_: 0.0

    comp = {k: [] for k in ("feature", "policy", "heuristic", "router", *bbs)}
    paths = []
    for j, item in enumerate([items[0]] + items):           # first: untimed warm-up
        warm, m = j == 0, item["model_id"]
        poses = pose_descriptor(item["cams"], d_mean, d_std)
        rnd = sorted(np.random.default_rng(j).choice(24, 5, replace=False).tolist())
        imgs = lambda vs: [item["images"][v] for v in vs]
        cams = lambda vs: [item["cams"][v] for v in vs]
        rec = lambda kind, t: warm or comp[kind].append({"object": m, "s": t})

        # ── components ──
        feats._cache.clear()
        for v in range(5):
            rec("feature", timed(lambda v=v: feats(item["images"][v]))[0])
        env = RGBViewEnv(_One(item), no_reward, view_budget=4, features=feats,
                         device=device, seed_initial_view=False)
        env._acquire(0)
        env.step_count = 0
        obs = env._obs()
        while not env.done:
            with torch.no_grad():
                t, (a, _, _) = timed(lambda: policy.act(to_tensor(stack_obs([obs]), device),
                                                        greedy=True))
            rec("policy", t)
            obs, *_ = env.step(int(a))
        rec("heuristic", timed(lambda: farthest_angle(poses, 0))[0])
        F = np.stack([feats(item["images"][v], key=(m, v)) for v in env.selected])
        rec("router", timed(lambda: pick_backbone(router, F))[0])
        for name, b in bbs.items():
            rec(name, timed(lambda b=b: b.predict(imgs(env.selected), cams(env.selected)))[0])

        # ── complete paths ──
        for r in range(1 if warm else args.repeats):
            for strategy in ("random", "heuristic", "policy"):
                for choice in (*bbs, "router"):
                    feats._cache.clear()
                    sync()
                    t0 = time.perf_counter()
                    if strategy == "random":
                        views = rnd
                    elif strategy == "heuristic":
                        views = farthest_angle(poses, 0)
                    else:
                        views = greedy_views(policy, no_reward, feats, 4, device, item, 0)
                    name = choice
                    if choice == "router":
                        F = np.stack([feats(item["images"][v], key=(m, v)) for v in views])
                        name = pick_backbone(router, F)
                    sync()
                    t1 = time.perf_counter()
                    bbs[name].predict(imgs(views), cams(views))
                    sync()
                    t2 = time.perf_counter()
                    if not warm:
                        paths.append({"object": m, "repeat": r, "strategy": strategy,
                                      "choice": choice, "backbone": name,
                                      "controller_s": t1 - t0, "backbone_s": t2 - t1,
                                      "total_s": t2 - t0})

    med = lambda xs: float(np.median([x["s"] for x in xs]))
    summary = {}
    for p in paths:
        summary.setdefault(f"{p['strategy']}+{p['choice']}", []).append(p)
    out = {"device": device,
           "device_name": (torch.cuda.get_device_name(0) if device == "cuda"
                           else platform.processor() or platform.machine()),
           "torch": torch.__version__,
           "threads": {"torch": torch.get_num_threads(),
                       "interop": torch.get_num_interop_threads(),
                       "cpu_count": os.cpu_count(),
                       "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
           "cohort": cohort, "repeats": args.repeats, "candidates": 24,
           "load_s": load,
           "component_median_s": {k: med(v) for k, v in comp.items()},
           "path_median_s": {k: {f: float(np.median([p[f] for p in v]))
                                 for f in ("controller_s", "backbone_s", "total_s")}
                             for k, v in summary.items()},
           "components": comp, "paths": paths}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(f"{device} ({out['device_name']}), threads {out['threads']}, "
          f"{len(cohort)} objects x {args.repeats} repeats")
    print("components (median ms): " + ", ".join(
        f"{k} {v*1e3:.1f}" for k, v in out["component_median_s"].items()))
    print(f"  {'path':<26} {'controller':>11} {'backbone':>10} {'total':>10}  (median s)")
    for k, v in out["path_median_s"].items():
        print(f"  {k:<26} {v['controller_s']:>11.3f} {v['backbone_s']:>10.3f} "
              f"{v['total_s']:>10.3f}")
    print("load (s): " + ", ".join(f"{k} {v:.2f}" for k, v in load.items()))


if __name__ == "__main__":
    main()
