"""
Phase 3, step 1: freeze the dev view sets every strategy is scored on.

    python phase3_views.py --out artifacts/phase3/views_dev.json

Per dev object, from the same 4 start views the policy evaluation used:

    policy     the selected checkpoint's views (from eval_0b's output)
    heuristic  farthest-angle: from the start, repeatedly add the candidate
               whose viewing direction is farthest (max-min angle) from every
               view already held; ties to the lowest index. Poses only --
               the same information the policy's angular features carry.
    random     the 24 random five-view sets the evals drew (replayed from the
               seed), independent of the start.

Writes the lists and the view sets not yet in the cache (for score_view_sets.py).
Nothing here reconstructs anything.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DEV_PREFIX, SEED, N_RANDOM, K = 24, 1234, 24, 5


def directions(poses):
    """(V, 5) pose descriptors -> (V, 3) unit viewing directions."""
    sa, ca, se, ce = poses[:, 0], poses[:, 1], poses[:, 2], poses[:, 3]
    return np.stack([ce * ca, ce * sa, se], 1)


def farthest_angle(poses, start, k=K):
    u = directions(poses)
    sel = [start]
    while len(sel) < k:
        ang = np.arccos(np.clip(u @ u[sel].T, -1, 1)).min(1)
        ang[sel] = -1
        sel.append(int(np.argmax(ang)))          # argmax: lowest index on ties
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="artifacts/tier1/eval_dev.json")
    ap.add_argument("--selection", default="artifacts/tier1/selection.json")
    ap.add_argument("--feats", default="artifacts/router/feats.npz")
    ap.add_argument("--cache", default="cache/utility_cache.json")
    ap.add_argument("--out", default="artifacts/phase3/views_dev.json")
    args = ap.parse_args()

    from splits import load as load_splits
    from training.utility_envelope import UtilityEnvelope, load_cache

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: the view sets are frozen. Use a new name.")
    chosen = json.load(open(args.selection))["selected"]
    po = json.load(open(args.eval))["arms"][chosen]["per_object"]
    pol = dict(zip(po["model_id"], po["views"]))
    d = np.load(args.feats)
    poses = d["poses"]
    idx = {m: i for i, m in enumerate(d["model_id"])}
    dev = load_splits()["dev"]
    tax = json.load(open(Path(__file__).resolve().parent / "datasets" / "ShapeNet.json"))

    rng = np.random.default_rng(SEED)
    views = {}
    for e in tax:
        for m in e["test"][:DEV_PREFIX]:             # replay every draw, kept or not
            rng.integers(24)
            rnd = [sorted(rng.choice(24, size=K, replace=False).tolist())
                   for _ in range(N_RANDOM)]
            if m not in dev:
                continue
            starts = [v[0] for v in pol[m]]
            views[m] = {"starts": starts,
                        "policy": [sorted(v) for v in pol[m]],
                        "heuristic": [sorted(farthest_angle(poses[idx[m]], s))
                                      for s in starts],
                        "random": rnd}
    assert set(views) == dev, f"{len(views)} of {len(dev)} dev objects"

    cache = load_cache(args.cache)["iou"]
    key = UtilityEnvelope.key
    missing = sorted({(m, tuple(v)) for m, vs in views.items()
                      for strat in ("policy", "heuristic", "random")
                      for v in vs[strat] if key(m, v) not in cache})
    body = json.dumps(views, sort_keys=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"policy_checkpoint": chosen, "seed": SEED,
                               "sha256": hashlib.sha256(body.encode()).hexdigest(),
                               "views": views}, indent=1))
    todo = out.with_name(out.stem + "_to_score.json")
    todo.write_text(json.dumps([[m, list(v)] for m, v in missing]))
    by = {s: sum(len(vs[s]) for vs in views.values()) for s in ("policy", "heuristic", "random")}
    print(f"{len(views)} dev objects; view sets {by}; {len(missing)} not yet scored "
          f"-> {todo}")


if __name__ == "__main__":
    main()
