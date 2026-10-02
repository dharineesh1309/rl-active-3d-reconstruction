"""
Reconstruct and score a frozen list of (object, view set) pairs into the cache.

    python score_view_sets.py --views artifacts/phase3/views_dev_to_score.json \
                              --kind eval_heuristic --ref farthest_angle

Each entry is recorded with `kind` and `ref`, so later analysis can tell these
labels from training or policy ones. Dev objects only, unless --split says
otherwise; the full list must load.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", required=True, help="JSON list of [model_id, [views]]")
    ap.add_argument("--kind", required=True)
    ap.add_argument("--ref", default=None)
    ap.add_argument("--split", default="dev", choices=["dev", "final_test"])
    ap.add_argument("--cache", default="cache/utility_cache.json")
    args = ap.parse_args()

    from backbones import load_backbones
    from config import Config
    from dataloader_shapenet import build_shapenet
    from splits import load as load_splits
    from training.utility_envelope import UtilityEnvelope, checkpoint_ids

    raw = Path(args.views).read_text()
    todo = defaultdict(list)
    for m, v in json.loads(raw):
        todo[m].append(v)
    allowed = load_splits()[args.split]
    stray = set(todo) - allowed
    if stray:
        raise SystemExit(f"{len(stray)} objects outside {args.split}")

    base = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = build_shapenet(split="test", only_ids=set(todo))
    if {m for _, m in ds.samples} != set(todo):
        raise SystemExit("some listed objects are missing on disk")
    bbs = load_backbones(base, device=device, require_all=True)
    env = UtilityEnvelope(bbs, cost_lambda=base.cost_lambda, cache_path=args.cache,
                          checkpoints=checkpoint_ids(bbs, base), verbose=True,
                          run={"script": "score_view_sets", "kind": args.kind,
                               "ref": args.ref, "split": args.split,
                               "list": Path(args.views).name,
                               "list_sha256": hashlib.sha256(raw.encode()).hexdigest()})
    env.tag = {"kind": args.kind, "ref": args.ref}
    n = sum(map(len, todo.values()))
    for i in range(len(ds)):
        item = ds[i]
        for v in todo[item["model_id"]]:
            env.ious(item, v)
        if (i + 1) % 25 == 0 or i + 1 == len(ds):
            env.flush()
            print(f"  {i+1}/{len(ds)} objects  {env.stats()}", flush=True)
    print(f"scored {n} view sets ({env.misses} reconstructed, {env.hits} already cached)")


if __name__ == "__main__":
    main()
