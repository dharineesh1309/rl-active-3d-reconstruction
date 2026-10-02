"""
Final test, data collection only. It computes and prints NO outcome metric.

    python final_collect.py --policy artifacts/tier1/set_pose_s164864.pt \
                            --cache cache/utility_cache.json \
                            --out-views artifacts/final/views_final.json \
                            --out-feats artifacts/final/feats_final.npz

For every final_test object (all 312 must load, 24 views each), as
pre-registered in artifacts/final/preregistration.json:

    starts     4 start views: the first from default_rng(SEED), 3 more from
               default_rng(SEED + 1) -- the procedure run_0b.evaluate uses
    random     24 random five-view sets from the first stream, as evaluate draws
    policy     the frozen policy's greedy episode from each start
    heuristic  farthest-angle from each start

Every view set is reconstructed into the cache, tagged eval_final_<strategy>.
The final objects' 24-view ResNet features and poses go to a SEPARATE file,
never into router or policy training inputs. The views file carries its hash.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SEED = 2026      # pre-registered; differs from the dev evaluations' 1234


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--cache", default="cache/utility_cache.json")
    ap.add_argument("--out-views", default="artifacts/final/views_final.json")
    ap.add_argument("--out-feats", default="artifacts/final/feats_final.npz")
    args = ap.parse_args()

    from backbones import load_backbones
    from config import Config
    from dataloader_shapenet import build_shapenet
    from env.rgb_view_env import ResNetFeatures
    from extract_router_feats import object_features
    from phase3_views import farthest_angle
    from policy.pose_policy import RGBPosePolicy, load_pose_norm
    from run_0b import ARMS, greedy_views
    from splits import load as load_splits
    from training.utility_envelope import UtilityEnvelope, checkpoint_ids

    for p in (args.out_views, args.out_feats):
        if Path(p).exists():
            raise SystemExit(f"{p} exists: the final collection runs once")
    base = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    final = load_splits()["final_test"]
    ds = build_shapenet(split="test", only_ids=final)
    if {m for _, m in ds.samples} != final or ds.n_views != 24:
        raise SystemExit("final_test incomplete on disk, or not 24 views per object")

    ck = torch.load(args.policy, map_location=device, weights_only=False)
    policy = RGBPosePolicy(n_candidates=24, **ARMS[ck["arm"]]).to(device)
    policy.load_state_dict(ck["policy"])
    policy.eval()
    policy_sha = hashlib.sha256(Path(args.policy).read_bytes()).hexdigest()
    bbs = load_backbones(base, device=device, require_all=True)
    env = UtilityEnvelope(bbs, cost_lambda=base.cost_lambda, cache_path=args.cache,
                          checkpoints=checkpoint_ids(bbs, base), verbose=True,
                          run={"script": "final_collect", "split": "final_test",
                               "seed": SEED, "policy_sha256": policy_sha})
    feats = ResNetFeatures(device=device)
    d_mean, d_std = load_pose_norm()
    root = Path(ds.rendering_root)

    rng, extra = np.random.default_rng(SEED), np.random.default_rng(SEED + 1)
    views, ids = {}, []
    F = np.zeros((len(ds), 24, 2048), np.float16)
    Pz = np.zeros((len(ds), 24, 5), np.float32)
    t0 = time.time()
    for i in range(len(ds)):
        syn, m = ds.samples[i]
        item = ds[i]
        first = int(rng.integers(24))
        starts = [first] + extra.choice([v for v in range(24) if v != first], 3,
                                        replace=False).tolist()
        rnd = [sorted(rng.choice(24, size=5, replace=False).tolist()) for _ in range(24)]
        F[i], Pz[i] = object_features(feats, root / syn / m / "rendering", 24,
                                      d_mean, d_std, device)

        env.tag = {"kind": "eval_final_policy", "ref": Path(args.policy).name}
        pol = [sorted(greedy_views(policy, env, feats, 4, device, item, s)) for s in starts]
        heu = [sorted(farthest_angle(Pz[i], s)) for s in starts]
        env.tag = {"kind": "eval_final_heuristic", "ref": "farthest_angle"}
        for v in heu:
            env.ious(item, v)
        env.tag = {"kind": "eval_final_random", "ref": SEED}
        for v in rnd:
            env.ious(item, v)
        views[m] = {"starts": starts, "policy": pol, "heuristic": heu, "random": rnd}
        ids.append(m)
        if (i + 1) % 25 == 0 or i + 1 == len(ds):
            env.flush()
            print(f"  {i+1}/{len(ds)} objects  {env.stats()}  {time.time()-t0:5.0f}s",
                  flush=True)

    body = json.dumps(views, sort_keys=True)
    Path(args.out_views).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_views).write_text(json.dumps(
        {"split": "final_test", "seed": SEED, "policy": Path(args.policy).name,
         "policy_sha256": policy_sha,
         "sha256": hashlib.sha256(body.encode()).hexdigest(), "views": views}, indent=1))
    Path(args.out_feats).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_feats, model_id=np.array(ids),
             category=np.array([ds.category_of[s] for s, _ in ds.samples]),
             split=np.array(["final_test"] * len(ids)), feats=F, poses=Pz)
    print(f"collected {len(views)} final objects; views -> {args.out_views}, "
          f"features -> {args.out_feats}. No scores computed.")


if __name__ == "__main__":
    main()
