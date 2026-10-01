"""
package_for_kaggle.py — build the two zips to upload as Kaggle datasets.

    python package_for_kaggle.py

Produces, next to this script:

    kaggle_code.zip      the rl_pipeline source (small, upload as a dataset)
    kaggle_weights.zip   the backbone checkpoints (large, upload once)

Data is deliberately NOT packaged. ShapeNetRendering is ~12 GB and is better
attached from an existing public Kaggle dataset, or uploaded separately once;
re-zipping it here would just duplicate it on disk.

The trained RL checkpoint is excluded too — it is the artifact of a *previous*
run against a 24-view action space, and including it invites an accidental
resume from an incompatible policy. Copy it in deliberately if you want it.
"""

import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "rl_pipeline"

# Everything the pipeline needs to run, and nothing that bloats the upload.
CODE_INCLUDE = ("*.py", "*.txt", "*.md", "*.json")
CODE_SKIP_DIRS = {"__pycache__", "checkpoints", "logs", ".ipynb_checkpoints"}

# Exactly the three REGISTERED backbones (backbones/__init__.py). This must
# cover every registered one or training silently runs with a smaller action
# space than intended -- load_backbones() skips a missing checkpoint with a
# notice rather than failing.
WEIGHTS = [
    "Pix2Vox-F-ShapeNet.pth",
    "UMIFormer-ShapeNet.pth",
    "UMIFormerPlus-ShapeNet.pth",
]


def zip_code(out: Path):
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path in PIPELINE.rglob("*"):
            if not path.is_file():
                continue
            if any(part in CODE_SKIP_DIRS for part in path.parts):
                continue
            if not any(path.match(pat) for pat in CODE_INCLUDE):
                continue
            z.write(path, Path("rl_pipeline") / path.relative_to(PIPELINE))
            n += 1
    print(f"  {out.name}: {n} files, {out.stat().st_size / 1e6:.1f} MB")


def zip_weights(out: Path):
    present = [HERE / w for w in WEIGHTS if (HERE / w).is_file()]
    missing = [w for w in WEIGHTS if not (HERE / w).is_file()]
    if missing:
        print(f"  WARNING: not found, will be skipped: {missing}")
    if not present:
        print("  no weights found; skipping weights zip")
        return
    # Stored, not deflated: these are already-compressed tensors, so deflating
    # them costs minutes and saves almost nothing.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for path in present:
            print(f"    adding {path.name} ({path.stat().st_size / 1e6:.0f} MB) ...")
            z.write(path, path.name)
    print(f"  {out.name}: {len(present)} files, {out.stat().st_size / 1e6:.0f} MB")


def zip_voxels(out: Path):
    """
    ShapeNetVox32 ground truth: ~22 MB of binvox, trivial to upload.

    Shipped separately rather than relying on whatever rendering dataset is
    attached: the reward is computed against these, so a missing or
    differently-shaped voxel tree means no training signal at all.
    """
    root = HERE / "ShapeNetVox32"
    if not root.is_dir():
        print("  ShapeNetVox32/ not found; skipping voxels zip")
        return
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix != ".tgz":
                z.write(path, Path("ShapeNetVox32") / path.relative_to(root))
                n += 1
    print(f"  {out.name}: {n} files, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    print("packaging for Kaggle\n")
    zip_code(HERE / "kaggle_code.zip")
    zip_voxels(HERE / "kaggle_voxels.zip")
    zip_weights(HERE / "kaggle_weights.zip")
    print(
        "\nNext:\n"
        "  1. Kaggle > Datasets > New Dataset, upload kaggle_code.zip\n"
        "  2. Same again for kaggle_weights.zip\n"
        "  3. Attach both, plus ShapeNetRendering and ShapeNetVox32, to a notebook\n"
        "  4. In the notebook:\n"
        "       !cp -r /kaggle/input/<code-dataset>/rl_pipeline /kaggle/working/\n"
        "       !python /kaggle/working/rl_pipeline/kaggle_train.py\n"
    )
