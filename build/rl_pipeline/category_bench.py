"""
category_bench.py — per-category backbone comparison across view budgets.

    python category_bench.py --limit 10 --budgets 1 3 5 8

Measures what single-category benchmarking cannot: whether a backbone's
advantage depends on the OBJECT as well as the view budget. That is the part a
learned policy can exploit and a budget-only lookup table cannot, so it is the
measurement the model head's existence rests on.

Run on airplane, car and chair (EXPERIMENTS.md §10) it produced the registry
and `cost_lambda` now in use:

  * The UMIFormer / UMIFormer+ crossover holds in **all three** categories,
    between 3 and 5 views. Same architecture, same cost, so that decision is
    independent of `cost_lambda` and survives even at lambda = 0.
  * Pix2Vox-F wins **car at one view** (0.8848 vs UMIFormer 0.8762) and nothing
    else -- the entire measured case for its slot, on a 0.0086 margin.
  * OccNet and TripoSR won **zero of twelve** category-budget cells and were
    dropped; their code has since been removed.

Prints a per-category table per budget, then who wins where, then the crossover
budget per category.
"""

import argparse
import sys
from collections import defaultdict

import numpy as np

from backbones import load_backbones, voxel_iou
from config import Config

THRESHOLDS = (0.2, 0.3, 0.4, 0.5)


def even_views(n_take, n_total):
    if n_take >= n_total:
        return list(range(n_total))
    return [int(round(i * n_total / n_take)) % n_total for i in range(n_take)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", type=int, nargs="+", default=[1, 3, 5, 8])
    p.add_argument("--limit", type=int, default=10,
                   help="test models per category (keep small; this is O(cats x budgets x backbones))")
    p.add_argument("--categories", nargs="+", default=None, help="default: all 13")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()

    from dataloader_shapenet import build_shapenet

    cfg = Config()
    dataset = build_shapenet(split="test", categories=args.categories,
                             limit_per_category=args.limit)
    backbones = load_backbones(cfg, device=args.device)
    names = [b.name for b in backbones]

    # Group sample indices by category so each is scored independently.
    by_cat = defaultdict(list)
    for i in range(len(dataset)):
        by_cat[dataset.category_of[dataset.samples[i][0]]].append(i)
    cats = sorted(by_cat)
    print(f"\n{len(cats)} categories, {len(dataset)} models, "
          f"{len(backbones)} backbones, budgets {args.budgets}\n")

    # results[budget][category][backbone] = IoU
    results = {B: {} for B in args.budgets}
    for B in args.budgets:
        idxs = even_views(B, dataset.n_views)
        print(f"-- budget {B} " + "-" * 54)
        print(f"{'category':<14}" + "".join(f"{n:>16}" for n in names) + f"{'winner':>16}")
        for cat in cats:
            # Score every threshold separately and pick the one with the best
            # MEAN, rather than the best threshold per model. Per-model choice is
            # an oracle the policy would not have at run time, and it inflates
            # results by roughly the size of the margins we care about -- chair
            # read 0.654 that way against 0.638 here, while the UMIFormer /
            # UMIFormer+ gap at 3 views is 0.002. It also matches backbones.bench
            # and the published protocol, so the numbers stay comparable.
            per = {n: {t: [] for t in THRESHOLDS} for n in names}
            for i in by_cat[cat]:
                item = dataset[i]
                imgs = [item["images"][v] for v in idxs]
                cams = [item["cams"][v] for v in idxs]
                for b in backbones:
                    pred = b.predict(imgs, cams)
                    for t in THRESHOLDS:
                        per[b.name][t].append(voxel_iou(pred, item["voxels"], pred_thresh=t))
            row = {n: max(float(np.mean(v)) for v in per[n].values()) for n in names}
            results[B][cat] = row
            win = max(row, key=row.get)
            print(f"{cat:<14}" + "".join(f"{row[n]:>16.4f}" for n in names) + f"{win:>16}")
        mean = {n: float(np.mean([results[B][c][n] for c in cats])) for n in names}
        print(f"{'MEAN':<14}" + "".join(f"{mean[n]:>16.4f}" for n in names)
              + f"{max(mean, key=mean.get):>16}")
        print()

    # -- Summary --------------------------------------------------------------
    print("=" * 72)
    print("  WHO WINS WHERE")
    print("=" * 72)
    wins = defaultdict(int)
    for B in args.budgets:
        for cat in cats:
            wins[max(results[B][cat], key=results[B][cat].get)] += 1
    total = len(args.budgets) * len(cats)
    for n in names:
        print(f"  {n:<16} wins {wins[n]:>3}/{total} category-budget cells "
              f"({wins[n]/total*100:.0f}%)")

    dead = [n for n in names if wins[n] == 0]
    if dead:
        print(f"\n  DEAD ACTIONS (never win anywhere): {dead}")
        print("  These cannot be selected by a reward-maximising policy. Drop them")
        print("  from the registry or accept them as permanently unused.")

    print("\n" + "=" * 72)
    print("  CROSSOVER BY CATEGORY (which budget flips the winner)")
    print("=" * 72)
    for cat in cats:
        seq = [max(results[B][cat], key=results[B][cat].get) for B in args.budgets]
        flips = [f"B={B}:{w}" for B, w in zip(args.budgets, seq)]
        tag = "  <- crossover" if len(set(seq)) > 1 else "  (no crossover)"
        print(f"  {cat:<14} " + "  ".join(flips) + tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
