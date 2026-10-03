"""
kaggle_fetch_data.py — get the ShapeNet data into Kaggle without uploading it.

Run this in its own Kaggle notebook (Settings > Internet: ON), commit it, then
attach that notebook's output as a dataset to the training notebook.

    !python /kaggle/working/rl_pipeline/kaggle_fetch_data.py --categories chair car table

Why bother
──────────
The renderings are 12.3 GB. Downloading them on a slow home connection and then
uploading 12.3 GB to Kaggle is hours of work for no reason: Kaggle's own link to
Stanford is fast, so fetching them *there* takes minutes.

Try a public dataset first, though. An attached dataset lives in /kaggle/input
and does not consume the 20 GB /kaggle/working quota at all, so if
the public "ShapeNet-Pix2Vox" Kaggle dataset (12.9 GB) contains the
ShapeNetRendering tree, just attach it and skip this script entirely.

The disk constraint
───────────────────
/kaggle/working is ~20 GB. The archive is 12.3 GB and expands to roughly 20-25 GB,
so saving the archive *and* extracting it does not fit. This script therefore
streams the download straight into tar, so only the extracted files ever touch
disk, and can restrict extraction to chosen categories to stay well inside the
quota. All 13 categories will not fit; pick a subset unless you are writing to a
larger volume.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

RENDER_URL = "http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz"
VOXEL_URL = "http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz"
OUT = Path("/kaggle/working")

# synset ids, from the Pix2Vox taxonomy
SYNSETS = {
    "aeroplane": "02691156", "bench": "02828884", "cabinet": "02933112",
    "car": "02958343", "chair": "03001627", "display": "03211117",
    "lamp": "03636649", "speaker": "03691459", "rifle": "04090263",
    "sofa": "04256520", "table": "04379243", "telephone": "04401088",
    "watercraft": "04530566",
}


def run(cmd, **kw):
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, **kw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--categories", nargs="+", default=["chair", "car", "table"],
                   help="Category names or synset ids. 'all' fetches everything "
                        "(will not fit in /kaggle/working).")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if "all" in args.categories:
        wanted = None
        print("WARNING: fetching all 13 categories needs ~25 GB; /kaggle/working "
              "is ~20 GB. This will likely fail on disk space.")
    else:
        wanted = []
        for c in args.categories:
            syn = SYNSETS.get(c, c)
            if syn not in SYNSETS.values():
                raise SystemExit(f"Unknown category {c!r}. Known: {sorted(SYNSETS)}")
            wanted.append(syn)
        names = [n for n, s in SYNSETS.items() if s in wanted]
        print(f"categories: {names}  ->  {wanted}")

    # ── Voxels: small, always take everything ────────────────────────────────
    print("\n[1/2] ShapeNetVox32 (22 MB, all categories)")
    if (out / "ShapeNetVox32").is_dir():
        print("  already present, skipping")
    else:
        run(f"curl -sL --retry 5 --retry-all-errors {VOXEL_URL} | tar -xz -C {out}")

    # ── Renderings: stream, extracting only what we want ─────────────────────
    print("\n[2/2] ShapeNetRendering (12.3 GB stream, extracting selected categories)")
    print("  Streaming rather than saving the archive: the .tgz plus its expansion")
    print("  would not fit in /kaggle/working. This reads the whole stream either")
    print("  way, so it takes a few minutes even for one category.")

    if wanted is None:
        members = "ShapeNetRendering"
    else:
        members = " ".join(f"ShapeNetRendering/{s}" for s in wanted)

    rc = run(f"curl -sL --retry 5 --retry-all-errors {RENDER_URL} "
             f"| tar -xz -C {out} {members}").returncode
    if rc != 0:
        print(f"  tar exited {rc} (often harmless: it can exit non-zero once the "
              f"requested members have been read)")

    # ── Report ───────────────────────────────────────────────────────────────
    render_root = out / "ShapeNetRendering"
    vox_root = out / "ShapeNetVox32"
    print("\n" + "=" * 60)
    if render_root.is_dir():
        total = 0
        for syn_dir in sorted(render_root.iterdir()):
            if syn_dir.is_dir():
                n = len(list(syn_dir.iterdir()))
                total += n
                name = next((k for k, v in SYNSETS.items() if v == syn_dir.name),
                            syn_dir.name)
                print(f"  {name:<12} {syn_dir.name}  {n} models")
        print(f"  total: {total} models with renderings")
    else:
        print("  NO RENDERINGS EXTRACTED - check the output above")
    if vox_root.is_dir():
        print(f"  voxels: {sum(1 for _ in vox_root.rglob('model.binvox'))} binvox files")

    print("\n  Commit this notebook, then attach its output to the training\n"
          "  notebook as a dataset. kaggle_train.py finds both trees by name.")


if __name__ == "__main__":
    sys.exit(main())
