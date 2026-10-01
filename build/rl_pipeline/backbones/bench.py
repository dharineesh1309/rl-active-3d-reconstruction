"""
Backbone benchmark — the integration correctness gate.

    python -m backbones.bench --limit 20
    python -m backbones.bench --views 1 2 3 5 --limit 50 --categories chair car

Runs each available backbone over the official ShapeNet test split and reports
mean IoU per view count, next to the published number. This is what proves a
backbone wrapper is actually correct: selftest.py only shows that tensors flow,
whereas a wrong preprocessing chain, a mis-ordered weight mapping or a wrong
normalisation still produces perfectly-shaped garbage. Those failures show up
here as an IoU near zero instead of near the reference.

Reference figures (Pix2Vox paper, Table 1, ShapeNet test split):

    Pix2Vox-F   1 view 0.634   2 views 0.653   3 views 0.661
                4 views 0.666  5 views 0.668   8 views 0.672

IoU is reported as the maximum over thresholds {.2, .3, .4, .5}, matching
Pix2Vox's own TEST.VOXEL_THRESH sweep, so the numbers are like-for-like.

Views are sampled evenly around the 24-pose orbit rather than taken as the
first N, so a 3-view score reflects spread coverage instead of three nearly
identical angles.
"""

import argparse
import itertools
import sys

import numpy as np

from . import load_backbones, voxel_iou

THRESHOLDS = (0.2, 0.3, 0.4, 0.5)

PUBLISHED = {
    "pix2vox_f": {1: 0.634, 2: 0.653, 3: 0.661, 4: 0.666, 5: 0.668, 8: 0.672},
}


def even_views(n_take: int, n_total: int = 24) -> list:
    """Evenly spaced view indices around the orbit."""
    if n_take >= n_total:
        return list(range(n_total))
    return [int(round(i * n_total / n_take)) % n_total for i in range(n_take)]


def calibrate_axes(backbones, dataset, n_views=5, top=5):
    """
    Find the axis convention that lines a backbone's output up with the
    ground-truth grid.

    Different codebases disagree about axis order and direction, and a wrong
    convention is not a crash: it yields a correctly-shaped volume that scores
    near zero. Rather than guess, predict once per model and score every one of
    the 6 x 8 = 48 orientations against the ground truth, then report the best.

    The winning (perm, flip) belongs in that backbone's AXES constant.
    """
    idxs = even_views(n_views, dataset.n_views)
    orientations = [(perm, flips)
                    for perm in itertools.permutations(range(3))
                    for flips in itertools.product([False, True], repeat=3)]

    for b in backbones:
        print(f"\n== calibrating {b.name} over {len(dataset)} models, "
              f"{n_views} views, {len(orientations)} orientations ==")

        # Predict once per model; re-orienting afterwards is nearly free
        # compared with re-running the network 48 times.
        pairs = []
        for i in range(len(dataset)):
            item = dataset[i]
            pred = b.predict([item["images"][v] for v in idxs])
            pairs.append((pred, item["voxels"]))

        scores = []
        for perm, flips in orientations:
            total = 0.0
            for pred, gt in pairs:
                out = np.transpose(pred, perm)
                for axis, f in enumerate(flips):
                    if f:
                        out = np.flip(out, axis=axis)
                total += voxel_iou(np.ascontiguousarray(out), gt)
            scores.append((total / len(pairs), perm, flips))

        scores.sort(reverse=True)
        for iou, perm, flips in scores[:top]:
            marker = "  <- identity" if perm == (0, 1, 2) and not any(flips) else ""
            print(f"  IoU {iou:.4f}  perm={perm}  flip={flips}{marker}")

        best_iou, best_perm, best_flips = scores[0]
        ident = next(s for s in scores if s[1] == (0, 1, 2) and not any(s[2]))
        print(f"  best: perm={best_perm} flip={best_flips} -> IoU {best_iou:.4f}")
        print(f"  identity scores {ident[0]:.4f}"
              + ("  (already correct)" if best_iou - ident[0] < 1e-6 else
                 f"  -- set AXES in backbones/{b.name}.py to the best above"))


def main(argv=None):
    p = argparse.ArgumentParser(description="Benchmark backbones on ShapeNet test split")
    p.add_argument("--views", type=int, nargs="+", default=[1, 3, 5],
                   help="View counts to evaluate (default: 1 3 5)")
    p.add_argument("--limit", type=int, default=20,
                   help="Models per category (default: 20)")
    p.add_argument("--categories", nargs="+", default=None,
                   help="Taxonomy names or synset ids (default: all 13)")
    p.add_argument("--split", default="test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--calibrate-axes", action="store_true",
                   help="Search axis permutations/flips for the orientation that "
                        "maximises IoU, instead of benchmarking.")
    args = p.parse_args(argv)

    from config import Config
    from dataloader_shapenet import build_shapenet

    cfg = Config()
    dataset = build_shapenet(
        split=args.split,
        categories=args.categories,
        limit_per_category=args.limit,
    )
    backbones = load_backbones(cfg, device=args.device)

    if args.calibrate_axes:
        return calibrate_axes(backbones, dataset, n_views=max(args.views))

    # The PUBLISHED figures are averages over all 13 categories. Comparing a
    # single-category run against them is misleading -- chair sits well below
    # the mean (3D-R2N2 scores .550 on chair vs .634 overall), so a correct
    # wrapper looks "LOW". When one category is selected, show that category's
    # own 3D-R2N2 baseline from the taxonomy instead.
    single_category = len({dataset.category_of[s] for s, _ in dataset.samples}) == 1
    category_baseline = {}
    if single_category:
        import json
        name = dataset.category_of[dataset.samples[0][0]]
        for entry in json.load(open("datasets/ShapeNet.json")):
            if entry["taxonomy_name"] == name:
                category_baseline = {int(k.split("-")[0]): v
                                     for k, v in entry.get("baseline", {}).items()}
                break
        if category_baseline:
            print(f"\nSingle category '{name}': showing its own 3D-R2N2 baseline "
                  f"rather than the 13-category average.")

    print(f"\nEvaluating {len(backbones)} backbone(s) over {len(dataset)} models, "
          f"view counts {args.views}\n")

    for b in backbones:
        reference = PUBLISHED.get(b.name, {})
        print(f"-- {b.name} " + "-" * (56 - len(b.name)))

        for n_views in args.views:
            idxs = even_views(n_views, dataset.n_views)
            per_threshold = {t: [] for t in THRESHOLDS}
            per_category = {}

            for i in range(len(dataset)):
                item = dataset[i]
                images = [item["images"][v] for v in idxs]
                pred = b.predict(images)
                gt = item["voxels"]

                best = max(voxel_iou(pred, gt, pred_thresh=t) for t in THRESHOLDS)
                for t in THRESHOLDS:
                    per_threshold[t].append(voxel_iou(pred, gt, pred_thresh=t))
                per_category.setdefault(item["category"], []).append(best)

            best_t = max(THRESHOLDS, key=lambda t: float(np.mean(per_threshold[t])))
            mean_iou = float(np.mean(per_threshold[best_t]))

            ref = reference.get(n_views)
            verdict = ""
            if ref is not None and not single_category:
                delta = mean_iou - ref
                # Small subsets scatter, so this band is a sanity check, not a
                # reproduction test. A broken wrapper lands near 0.0, not near -0.05.
                status = "OK" if abs(delta) <= 0.05 else ("LOW" if delta < 0 else "HIGH")
                verdict = f"  published {ref:.3f}  delta {delta:+.3f}  [{status}]"

            if single_category and n_views in category_baseline:
                base = category_baseline[n_views]
                verdict = (f"  3D-R2N2 {name} baseline {base:.3f}  "
                           f"delta {mean_iou - base:+.3f}")
            print(f"  {n_views:>2} view(s): IoU {mean_iou:.4f} @ thresh {best_t}{verdict}")

            if len(per_category) > 1:
                ranked = sorted(per_category.items(), key=lambda kv: -float(np.mean(kv[1])))
                summary = "  ".join(f"{c}:{np.mean(v):.3f}" for c, v in ranked[:6])
                print(f"              {summary}")
        print()

    print("Note: a correct wrapper lands within a few points of the published "
          "figure.\nA near-zero IoU means the weights, preprocessing or channel "
          "order is wrong.")


if __name__ == "__main__":
    sys.exit(main())
