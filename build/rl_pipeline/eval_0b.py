"""
Paired evaluation of view-policy checkpoints.

    python eval_0b.py artifacts/tier1/set_pose_s*.pt artifacts/tier0b/set_pose.pt
    python eval_0b.py <frozen choice> --split final_test      # once, at the end

The training-time evals give one greedy rollout per object, and their
standard error (~0.0018 IoU) is about as large as the gaps between arms. Here
every checkpoint is rolled out from the SAME start views on the SAME objects,
so they are compared per object, each averaged over several start views. Each
checkpoint's arm is read from the file.

`dev` (default) reuses the training run's cache: its random and oracle draws
are seed-for-seed what training drew. `final_test` is touched once, after every
choice is frozen.

Checkpoints are rarely cut at equal step counts; the steps are printed beside
each so a gap can be read against the training it had.
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

from run_0b import ARMS, DEV_PREFIX, evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+", help="policy checkpoints; arm read from each")
    ap.add_argument("--split", choices=["dev", "final_test"], default="dev")
    ap.add_argument("--starts", type=int, default=4, help="start views per object")
    ap.add_argument("--budget", type=int, default=4)
    ap.add_argument("--eval-subsets", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cache", default="cache/utility_cache.json")
    ap.add_argument("--out", default=None, help="default: eval_<split>.json beside "
                                                "the first checkpoint")
    args = ap.parse_args()

    from backbones import load_backbones
    from config import Config
    from dataloader_shapenet import build_shapenet
    from env.rgb_view_env import ResNetFeatures
    from policy.pose_policy import RGBPosePolicy
    from splits import load as load_splits
    from training.utility_envelope import UtilityEnvelope

    base = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")
    S = load_splits()
    if args.split == "dev":
        test_ds, keep = build_shapenet(split="test", limit_per_category=DEV_PREFIX), S["dev"]
    else:
        print("FINAL TEST: run once, after every choice is frozen.")
        test_ds, keep = build_shapenet(split="test", only_ids=S["final_test"]), None
    envelope = UtilityEnvelope(load_backbones(base, device=device),
                               cost_lambda=base.cost_lambda, cache_path=args.cache,
                               cfg=base, verbose=True,
                               run={"script": "eval_0b", "split": args.split,
                                    "ckpts": [Path(p).name for p in args.ckpts]})
    feats = ResNetFeatures(device=device)

    res = {}
    for path in args.ckpts:
        ck = torch.load(path, map_location=device, weights_only=False)
        arm, label = ck["arm"], Path(path).stem
        policy = RGBPosePolicy(n_candidates=test_ds.n_views, **ARMS[arm]).to(device)
        policy.load_state_dict(ck["policy"])
        ev = evaluate(policy, test_ds, envelope, feats, args.budget,
                      args.eval_subsets, device, seed=args.seed,
                      n_starts=args.starts, keep=keep)
        envelope.flush()
        res[label] = {"arm": arm, "steps": ck["steps"], "path": str(path), **ev}
        print(f"  {label:<18} {ck['steps']:>7,} steps | policy-random "
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

    out = Path(args.out or Path(args.ckpts[0]).parent / f"eval_{args.split}.json")
    out.write_text(json.dumps({"args": vars(args), "arms": res}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
