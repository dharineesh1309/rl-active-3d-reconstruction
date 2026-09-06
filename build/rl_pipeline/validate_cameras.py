"""
validate_cameras.py -- does view index i mean the same camera on every object?

Two jobs in one script.

**Engineering validation.** The pose-conditioned policy reads camera parameters
through `dataloader_shapenet._read_metadata`, and that function falls back to a
zeroed default when `rendering_metadata.txt` is missing or malformed -- silently.
Under the old index-based policy that only degraded the coverage grid; under a
pose-conditioned policy it would make every candidate view look identical, which
is a far worse failure and an invisible one. This script exercises the real
parser and counts fallbacks.

**Paper evidence.** Choy et al.'s ShapeNetRendering samples camera poses per
object rather than placing every model on one shared lattice. If that is true
here, a policy whose action is an integer index cannot generalise: action 7 is a
different physical direction for every object. The figures make that immediate.

    python validate_cameras.py                      # local subset
    python validate_cameras.py --limit 1000         # bigger scan
    python validate_cameras.py --out artifacts/camera_metadata

Emits, under --out:
    camera_pose_stats.json      global summary, incl. which fields are constant
    per_index_pose_stats.csv    per view index: mean/std/min/max per field
    pose_variance_by_view.png   figure 1: azimuth spread per index
    pose_scatter_by_view.png    figure 2: where each index actually lands
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataloader_shapenet import _read_metadata

FIELDS = ("azimuth", "elevation", "in_plane", "distance", "fov")
# What _read_metadata returns when it cannot read the file. Matching this exactly
# is how we detect a silent fallback rather than a real camera.
DEFAULT_CAM = {"azimuth": 0.0, "elevation": 0.0, "in_plane": 0.0,
               "distance": 1.0, "fov": 25.0}


def find_rendering_root() -> Path:
    for env in ("SHAPENET_RENDERING_ROOT",):
        if os.environ.get(env):
            return Path(os.environ[env])
    root = os.environ.get("SHAPENET_ROOT")
    if root:
        return Path(root) / "ShapeNetRendering"
    raise SystemExit("Set SHAPENET_ROOT or SHAPENET_RENDERING_ROOT.")


def collect(root: Path, limit: int, allowed=None):
    """Walk synset/model/rendering/rendering_metadata.txt, bounded."""
    records, fallbacks, n_views_seen, n_rows = [], [], [], []
    synsets = sorted(p for p in root.iterdir() if p.is_dir() and p.name.isdigit())
    if not synsets:
        raise SystemExit(f"No synset directories under {root}")
    per_syn = max(1, limit // len(synsets))
    for syn in synsets:
        taken = 0
        for model in sorted(syn.iterdir()):
            if taken >= per_syn:
                break
            if allowed is not None and model.name not in allowed:
                continue
            rend = model / "rendering"
            meta = rend / "rendering_metadata.txt"
            if not meta.is_file():
                continue
            n_views = len(sorted(rend.glob("*.png")))
            if n_views == 0:
                continue
            # Strict: _read_metadata now raises rather than returning zeroed
            # cameras. The validator is the one caller that wants to survive a
            # bad object and count it, so it catches instead of aborting.
            try:
                cams = _read_metadata(meta, n_views, allow_fallback=False)
            except RuntimeError as exc:
                fallbacks.append(f"{syn.name}/{model.name}: {str(exc).splitlines()[0]}")
                continue
            n_rows.append(len(cams))
            records.append((syn.name, model.name, cams))
            n_views_seen.append(n_views)
            taken += 1
    return records, fallbacks, n_views_seen, n_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400, help="objects to scan")
    ap.add_argument("--out", default="artifacts/camera_metadata")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--split", default="all", choices=["train", "val", "test", "all"],
                    help="Restrict the scan. Use 'train' when freezing pose "
                         "normalisation: validation and test must never "
                         "contribute to those statistics.")
    ap.add_argument("--taxonomy", default="datasets/ShapeNet.json")
    ap.add_argument("--freeze-pose-norm",
                    help="Write the distance mean/std to this config path. "
                         "Only meaningful with --split train.")
    args = ap.parse_args()

    root = find_rendering_root()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"scanning {root}")

    allowed = None
    if args.split != "all":
        tax = json.load(open(args.taxonomy))
        allowed = {m for c in tax for m in c.get(args.split, [])}
        print(f"  restricted to '{args.split}' split: {len(allowed)} model ids")
    records, fallbacks, n_views_seen, n_rows = collect(root, args.limit, allowed)
    if not records:
        raise SystemExit("No objects with readable camera metadata found.")
    n_views = int(np.bincount(n_views_seen).argmax())
    records = [r for r in records if len(r[2]) == n_views]
    print(f"  {len(records)} objects with {n_views} views each")
    if fallbacks:
        print(f"  WARNING: {len(fallbacks)} objects fell back to DEFAULT cameras "
              f"(unreadable metadata). Under a pose-conditioned policy those "
              f"objects would present identical candidates.")
        for f in fallbacks[:5]:
            print(f"    {f}")

    # arr[field][object, view]
    arr = {f: np.array([[c[f] for c in cams] for _, _, cams in records])
           for f in FIELDS}

    # ── per-index statistics ─────────────────────────────────────────────────
    csv_path = out / "per_index_pose_stats.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["view_idx", "n_objects",
                    "azimuth_mean", "azimuth_std", "azimuth_min", "azimuth_max",
                    "elevation_mean", "elevation_std",
                    "distance_mean", "distance_std"])
        for j in range(n_views):
            w.writerow([j, len(records),
                        f"{arr['azimuth'][:, j].mean():.4f}",
                        f"{arr['azimuth'][:, j].std():.4f}",
                        f"{arr['azimuth'][:, j].min():.4f}",
                        f"{arr['azimuth'][:, j].max():.4f}",
                        f"{arr['elevation'][:, j].mean():.4f}",
                        f"{arr['elevation'][:, j].std():.4f}",
                        f"{arr['distance'][:, j].mean():.4f}",
                        f"{arr['distance'][:, j].std():.4f}"])

    # ── pose-set duplication ─────────────────────────────────────────────────
    # Some objects share an identical camera set: the renderer reused random
    # seeds. Worth quantifying, because on a duplicated subset an index-based
    # policy WOULD see consistent viewpoints -- just not in general. It also
    # forces figure 2 to pick objects with DISTINCT pose sets, or duplicate
    # panels make the figure appear to argue the opposite of the finding.
    sig = [tuple(np.round([c["azimuth"] for c in cams], 4)) for _, _, cams in records]
    uniq = {}
    for i, k in enumerate(sig):
        uniq.setdefault(k, []).append(i)
    n_unique = len(uniq)
    dup_rate = 1.0 - n_unique / len(records)
    distinct_idx = [v[0] for v in uniq.values()]

    # ── global summary ───────────────────────────────────────────────────────
    per_index_az_std = float(arr["azimuth"].std(axis=0).mean())
    constant = {f: bool(arr[f].std() < 1e-9) for f in FIELDS}
    stats = {
        "rendering_root": str(root),
        "num_objects": len(records),
        "num_views_per_object": n_views,
        "num_metadata_fallbacks": len(fallbacks),
        "num_unique_pose_sets": n_unique,
        "pose_set_duplication_rate": round(dup_rate, 4),
        "mean_per_index_azimuth_std_deg": round(per_index_az_std, 4),
        "fields": {f: {"min": round(float(arr[f].min()), 6),
                       "max": round(float(arr[f].max()), 6),
                       "std": round(float(arr[f].std()), 6),
                       "constant": constant[f]} for f in FIELDS},
        "in_plane_unique": sorted({round(float(v), 6) for v in arr["in_plane"].ravel()})[:8],
        "fov_unique": sorted({round(float(v), 6) for v in arr["fov"].ravel()})[:8],
        "pose_descriptor": {
            "encoded": ["sin_azimuth", "cos_azimuth", "sin_elevation",
                        "cos_elevation", "normalised_distance"],
            "omitted": [f for f in FIELDS if constant[f]],
            "reason": ("constant across every view of every object, so it cannot "
                       "distinguish candidates"),
        },
        "verdict": ("index is NOT a stable physical viewpoint"
                    if per_index_az_std > 5.0 else
                    "index appears to be a stable physical viewpoint"),
    }
    # Distance normalisation, computed on whatever split was scanned. These two
    # numbers get frozen: validation and test must be normalised with the
    # TRAINING statistics, never recompute their own, or the policy sees a
    # different input distribution at evaluation than it trained on.
    d_mu, d_sd = float(arr["distance"].mean()), float(arr["distance"].std())
    stats["distance_norm"] = {"split": args.split, "mean": round(d_mu, 6),
                              "std": round(d_sd, 6), "n_objects": len(records)}
    with open(out / "camera_pose_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    if args.freeze_pose_norm:
        if args.split != "train":
            raise SystemExit("--freeze-pose-norm requires --split train")
        cfg_path = Path(args.freeze_pose_norm)
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg = {
            "schema": "pose_norm_v1",
            "descriptor": ["sin_azimuth", "cos_azimuth", "sin_elevation",
                           "cos_elevation", "normalised_distance"],
            "distance_mean": round(d_mu, 6),
            "distance_std": round(d_sd, 6),
            "split": "train",
            "n_objects": len(records),
            "n_views_per_object": n_views,
            "rendering_root": str(root),
            "omitted_fields": [f for f in FIELDS if constant[f]],
        }
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        print()
        print(f"  froze pose normalisation -> {cfg_path}")
        print(f"    distance mean {d_mu:.6f}  std {d_sd:.6f}  "
              f"(train split, {len(records)} objects)")

    # ── console summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    for f in FIELDS:
        tag = "CONSTANT -> omit from descriptor" if constant[f] else "varies -> encode"
        print(f"  {f:<10} min {arr[f].min():9.3f}  max {arr[f].max():9.3f}  "
              f"std {arr[f].std():8.3f}   {tag}")
    print(f"\n  mean per-index azimuth std: {per_index_az_std:.1f} deg")
    print(f"  a shared 24-pose lattice would give 0.0 deg")
    print(f"\n  unique camera sets: {n_unique}/{len(records)} objects "
          f"({dup_rate*100:.0f}% share a pose set with another object)")
    # Schema checks. These are the ones that must hold on Kaggle before the
    # pose-conditioned encoder is written against this interface.
    rows_ok = len(set(n_rows)) <= 1
    print(f"\n  metadata rows per object: {sorted(set(n_rows))}"
          f"   {'consistent' if rows_ok else 'INCONSISTENT'}")
    print(f"  parse failures: {len(fallbacks)}   "
          f"{'OK' if not fallbacks else 'INVESTIGATE (listed above)'}")
    print(f"\n  VERDICT: {stats['verdict']}")
    print("=" * 70)

    if not args.no_figures:
        make_figures(arr, records, n_views, out, per_index_az_std, distinct_idx)
    print(f"\nwrote {out}/camera_pose_stats.json")
    print(f"      {csv_path}")


def make_figures(arr, records, n_views, out, per_index_az_std, distinct_idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 150})
    INK, ACC = "#1f2933", "#2f6f9f"

    # ── Figure 1: azimuth spread per nominal index ───────────────────────────
    az = arr["azimuth"]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    rng = np.random.default_rng(0)
    for j in range(n_views):
        x = j + rng.uniform(-0.28, 0.28, az.shape[0])
        ax.scatter(x, az[:, j], s=3, alpha=0.28, color=ACC, linewidths=0)
    # What a shared lattice would look like, for contrast.
    for j in range(n_views):
        ax.hlines(j * (360.0 / n_views), j - 0.4, j + 0.4,
                  color="#c2410c", lw=1.6, zorder=3)
    ax.set_xlabel("nominal view index")
    ax.set_ylabel("actual azimuth (deg)")
    ax.set_ylim(-10, 370)
    ax.set_xticks(range(0, n_views, max(1, n_views // 12)))
    ax.set_title(f"Camera azimuth per view index across {az.shape[0]} objects\n"
                 f"mean per-index std {per_index_az_std:.0f} deg "
                 f"(a shared lattice would be 0)", color=INK, loc="left")
    ax.plot([], [], color="#c2410c", lw=1.6,
            label="where a shared lattice would place each index")
    ax.scatter([], [], s=12, color=ACC, label="observed azimuths")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out / "pose_variance_by_view.png", bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: where the same indices land, per object ────────────────────
    # Objects with DISTINCT pose sets. Chosen blindly, duplicates produce
    # identical panels and the figure appears to show a shared lattice.
    picks = distinct_idx[:3]
    fig, axes = plt.subplots(1, len(picks), figsize=(3.0 * len(picks), 3.1), sharey=True)
    axes = np.atleast_1d(axes)
    show = min(8, n_views)
    for ax, p in zip(axes, picks):
        syn, mid, cams = records[p]
        a = np.array([c["azimuth"] for c in cams])
        e = np.array([c["elevation"] for c in cams])
        ax.scatter(a, e, s=14, color=ACC, zorder=3, linewidths=0)
        for j in range(show):
            ax.annotate(str(j), (a[j], e[j]), fontsize=7, color=INK,
                        xytext=(3, 3), textcoords="offset points")
        ax.set_xlim(-10, 370)
        ax.set_xlabel("azimuth (deg)")
        ax.set_title(f"{syn}/{mid[:8]}", fontsize=8, color=INK)
    axes[0].set_ylabel("elevation (deg)")
    fig.suptitle("The same nominal view indices land in unrelated directions "
                 "on different objects", fontsize=9.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "pose_scatter_by_view.png", bbox_inches="tight")
    plt.close(fig)
    print(f"      {out}/pose_variance_by_view.png")
    print(f"      {out}/pose_scatter_by_view.png")


if __name__ == "__main__":
    main()
