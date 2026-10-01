"""
Paired re-evaluation of the Tier 0B checkpoints.

    python eval_0b.py --ckpt-dir artifacts/tier0b --starts 4

The training-time evals give one greedy rollout per held-out object, and their
standard error (~0.0018 IoU) is about as large as the gaps between arms. Here
every arm is rolled out from the SAME start views on the SAME objects, so arms
are compared per object, and each object is averaged over several start views.

Reuse the training run's cache: the random and oracle baselines are seed-for-
seed what training drew, so with the cache attached they cost nothing.

Note the checkpoints were not cut at equal step counts; the steps are printed
beside each arm so a gap can be read against the training it had.
"""

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_0b import ARMS, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="artifacts/tier0b")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--starts", type=int, default=4, help="start views per object")
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--eval-objects", type=int, default=24)
    ap.add_argument("--eval-subsets", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cache", default="cache/utility_cache.json")
    ap.add_argument("--out", default=None, help="default: <ckpt-dir>/eval_0b.json")
    args = ap.parse_args()

    from backbones import load_backbones
    from config import Config
    from dataloader_shapenet import build_shapenet
    from env.rgb_view_env import ResNetFeatures
    from policy.pose_policy import RGBPosePolicy
    from training.utility_envelope import UtilityEnvelope

    base = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")
    test_ds = build_shapenet(split="test", limit_per_category=args.eval_objects)
    envelope = UtilityEnvelope(load_backbones(base, device=device),
                               cost_lambda=base.cost_lambda,
                               cache_path=args.cache, verbose=True)
    feats = ResNetFeatures(device=device)

    res = {}
    for arm in args.arms:
        ck = torch.load(Path(args.ckpt_dir) / f"{arm}.pt", map_location=device)
        policy = RGBPosePolicy(n_candidates=test_ds.n_views, **ARMS[arm]).to(device)
        policy.load_state_dict(ck["policy"])
        ev = evaluate(policy, test_ds, envelope, feats, args.budget,
                      args.eval_subsets, device, seed=args.seed,
                      n_starts=args.starts)
        envelope.flush()
        res[arm] = {"steps": ck["steps"], **ev}
        print(f"  {arm:<10} {ck['steps']:>7,} steps | policy-random "
              f"{ev['policy_minus_random']:+.4f} (se {ev['policy_minus_random_se']:.4f})"
              f" | captured {ev['fraction_captured']*100:5.1f}% of "
              f"{ev['headroom']:.4f}", flush=True)

    print("\npaired, per object (a - b):")
    pol = {a: np.array(res[a]["per_object"]["policy"]) for a in res}
    for a, b in itertools.combinations(res, 2):
        d = pol[a] - pol[b]
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"  {a:<10} - {b:<10} {d.mean():+.4f}  se {se:.4f}  "
              f"z {d.mean()/se:+.2f}  wins {np.mean(d > 0)*100:4.1f}%  "
              f"ties {np.mean(d == 0)*100:4.1f}%")

    print("\nper category, policy - random:")
    po = next(iter(res.values()))["per_object"]
    rnd = np.array(po["random"])
    by_cat = defaultdict(list)
    for j, c in enumerate(po["category"]):
        by_cat[c].append(j)
    print("  " + " " * 11 + "".join(f"{a:>11}" for a in res))
    for c in sorted(by_cat):
        j = by_cat[c]
        print(f"  {c:<11}" + "".join(f"{(pol[a][j] - rnd[j]).mean():>+11.4f}"
                                     for a in res))

    out = Path(args.out or Path(args.ckpt_dir) / "eval_0b.json")
    out.write_text(json.dumps({"args": vars(args), "arms": res}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
