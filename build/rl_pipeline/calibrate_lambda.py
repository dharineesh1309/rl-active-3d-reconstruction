"""
calibrate_lambda.py — find the cost_lambda that makes backbone selection a real decision.

    python calibrate_lambda.py --limit 60 --repeats 3

The problem this solves
───────────────────────
The model head only learns something if different backbones win the reward at
different view budgets. If one backbone wins everywhere, the head correctly
learns a constant and the whole backbone-selection contribution is vacuous.

Which backbone wins depends on lambda:

    reward_i(B) = IoU_i(B) - lambda * (B + cost_i)

so backbone i beats j at budget B exactly when

    IoU_i(B) - IoU_j(B) > lambda * (cost_i - cost_j)

The crossover lambda for that pair at that budget is the ratio of those two
differences. If the ratio varies across budgets, there is a lambda window in
between where the cheap model wins at some budgets and the dear one at others.
That window is what we want.

Why measure on the TRAIN split
──────────────────────────────
This is the mistake that collapsed the first run. Lambda was set from
test-split quality gaps, but the agent is trained on the train split, where
these backbones were pretrained and therefore memorise. The train-split gaps
are much wider, so the dear backbone wins by far more than the test numbers
suggest and the head learns to always pick it. Calibrate on the distribution
the agent actually sees.

Views are sampled randomly and averaged over several draws, because that is
what an untrained policy does and what the reward statistics reflect.
"""

import argparse
import itertools
import sys

import numpy as np

from backbones import load_backbones, voxel_iou
from config import Config


def measure(dataset, backbones, budgets, repeats, seed=0):
    """mean IoU per (backbone, budget), over random view draws."""
    rng = np.random.default_rng(seed)
    out = {b.name: {} for b in backbones}
    for B in budgets:
        acc = {b.name: [] for b in backbones}
        for i in range(len(dataset)):
            item = dataset[i]
            n = len(item["images"])
            for _ in range(repeats):
                views = list(rng.choice(n, size=min(B, n), replace=False))
                imgs = [item["images"][v] for v in views]
                cams = [item["cams"][v] for v in views]
                for b in backbones:
                    acc[b.name].append(voxel_iou(b.predict(imgs, cams), item["voxels"]))
        for name in acc:
            out[name][B] = float(np.mean(acc[name]))
        print(f"  B={B}: " + "  ".join(f"{n}={out[n][B]:.4f}" for n in out), flush=True)
    return out


def analyse(iou, cost, budgets, label):
    print("\n" + "=" * 72)
    print(f"  LAMBDA ANALYSIS - {label}")
    print("=" * 72)

    names = list(iou)
    print("\npairwise crossover lambda (below it the dearer model wins):")
    windows = []
    for a, b in itertools.combinations(names, 2):
        dcost = cost[a] - cost[b]
        if abs(dcost) < 1e-9:
            continue
        dear, cheap = (a, b) if dcost > 0 else (b, a)
        flips = {B: (iou[dear][B] - iou[cheap][B]) / abs(dcost) for B in budgets}
        lo, hi = min(flips.values()), max(flips.values())
        print(f"  {dear} vs {cheap}: " +
              "  ".join(f"B={B}:{f:.3f}" for B, f in flips.items()))
        if hi - lo > 0.01 and lo > 0:
            windows.append((lo, hi, dear, cheap))
            print(f"     -> live window lambda in ({lo:.3f}, {hi:.3f}): "
                  f"{cheap} wins where the gap is smallest")
        elif lo <= 0:
            print(f"     -> {cheap} is never worth it at any positive lambda")
        else:
            print(f"     -> gap is flat across budgets; no budget-dependent switch")

    # Scan lambda and count how many distinct backbones win somewhere.
    print("\nthree-way winner by lambda:")
    best = None
    for lam in np.arange(0.0, 0.85, 0.01):
        winners = {}
        for B in budgets:
            r = {n: iou[n][B] - lam * (B + cost[n]) for n in names}
            winners[B] = max(r, key=r.get)
        distinct = set(winners.values())
        if len(distinct) > 1:
            if best is None:
                best = [lam, lam, dict(winners)]
            else:
                best[1] = lam
    if best is None:
        print("  NO lambda produces a split decision: one backbone wins at every")
        print("  budget for every lambda tested. Backbone selection cannot be made")
        print("  meaningful by tuning lambda alone with this set.")
        return None

    lo, hi, pattern = best
    mid = round((lo + hi) / 2, 3)
    print(f"  split decision for lambda in ({lo:.2f}, {hi:.2f}), width {hi-lo:.2f}")
    print(f"  midpoint lambda = {mid}")
    r = {B: max({n: iou[n][B] - mid * (B + cost[n]) for n in names}.items(),
                key=lambda kv: kv[1]) for B in budgets}
    for B, (n, v) in r.items():
        print(f"     B={B}: {n}  (reward {v:+.3f})")
    return mid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", type=int, nargs="+", default=[3, 5, 8])
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--categories", nargs="+", default=["chair"])
    p.add_argument("--splits", nargs="+", default=["train", "test"])
    args = p.parse_args()

    from dataloader_shapenet import build_shapenet
    cfg = Config()
    backbones = load_backbones(cfg)
    cost = {b.name: b.cost for b in backbones}
    print(f"\nbackbone costs (measured wall-clock, normalised): {cost}")

    recommended = {}
    for split in args.splits:
        print(f"\n--- measuring IoU on the {split} split ---", flush=True)
        ds = build_shapenet(split=split, categories=args.categories,
                            limit_per_category=args.limit)
        iou = measure(ds, backbones, args.budgets, args.repeats)
        recommended[split] = analyse(iou, cost, args.budgets, f"{split} split")

    print("\n" + "=" * 72)
    print("  RECOMMENDATION")
    print("=" * 72)
    train = recommended.get("train")
    if train:
        print(f"  Set Config.cost_lambda = {train}")
        print("  This is the TRAIN-split figure, which is the distribution the agent")
        print("  is optimised against. Using the test-split number is what made the")
        print("  first run collapse to always-one-backbone.")
    else:
        print("  No lambda fixes this backbone set. Changing the set is the real fix.")
    if recommended.get("test") and train:
        print(f"\n  (test-split value would be {recommended['test']} - shown only to")
        print("   illustrate the gap; do not train with it.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
