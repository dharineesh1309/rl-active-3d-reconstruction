"""
The README figure, from the committed post hoc benchmark.

    python plot_benchmark.py --out ../../docs/benchmark.png

Rows: the full system minus each alternative on the final test (312 unseen
objects), 95% paired bootstrap intervals. Three panels, three metrics, each on
its own axis. Filled marker = interval excludes 0; hollow = not detectable
(also labelled, so significance is never carried by colour alone).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#b5b4ad", "#2a78d6"
ROWS = [("policy+router_corrected|random+pix2vox_f", "vs Pix2Vox-F (random views)"),
        ("policy+router_corrected|random+umiformer", "vs UMIFormer (random views)"),
        ("policy+router_corrected|random+umiformer_plus", "vs UMIFormer+ (random views)"),
        ("policy+router_corrected|heuristic+umiformer_plus",
         "vs farthest-angle views + UMIFormer+")]
PANELS = [("iou", "IoU"), ("v1", "Cost-aware utility\n(declared cost; project metric)"),
          ("cpu_est", "Utility with measured CPU time\n(estimated, dev timing profile)")]


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--bench", default=str(here.parents[1] / "artifacts/final/standard_benchmark.json"))
    ap.add_argument("--out", default=str(here.parents[1] / "docs/benchmark.png"))
    args = ap.parse_args()
    post = json.load(open(args.bench))["posthoc"]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True, facecolor=SURFACE)
    ys = list(range(len(ROWS)))[::-1]
    for ax, (metric, title) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        ax.axvline(0, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
        lo_all, hi_all = [], []
        for y, (key, _) in zip(ys, ROWS):
            m, lo, hi = post[key][metric]
            sig = lo > 0 or hi < 0
            ax.plot([lo, hi], [y, y], color=SERIES, lw=2, solid_capstyle="round", zorder=2)
            ax.plot(m, y, "o", ms=8, color=SERIES, mfc=SERIES if sig else SURFACE,
                    mew=2, zorder=3)
            ax.annotate(f"{m:+.4f}" + ("" if sig else "  (n.d.)"), (hi, y),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=9, color=INK)
            lo_all.append(lo); hi_all.append(hi)
        span = max(hi_all) - min(lo_all)
        ax.set_xlim(min(min(lo_all), 0) - 0.05 * span, max(max(hi_all), 0) + 0.45 * span)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.grid(axis="x", color="#e8e7e3", lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([label for _, label in ROWS], color=INK)
    fig.suptitle("Full system (learned views + router) minus each alternative - final test, "
                 "312 unseen objects, 95% paired bootstrap intervals",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.01, "Right of the dashed line = full system better. Filled = interval "
             "excludes 0; hollow / n.d. = no detectable difference. Only the UMIFormer+ "
             "cost-aware row was pre-registered; the rest are post hoc.",
             fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
