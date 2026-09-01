"""
kaggle_bench.py — run the per-category backbone comparison on Kaggle.

    !python /kaggle/working/rl_pipeline/kaggle_bench.py --limit 10

Exists because the interesting measurement cannot be made locally. No seekable
mirror of ShapeNetRendering carries all 13 categories, so every number produced
on this project's development machine has come from chair alone (and, latterly,
airplane and car). Kaggle already has the canonical 13-category renderings
attached, where the sweep costs nothing but GPU minutes.

It reuses `kaggle_train.resolve_inputs()` rather than repeating the discovery,
so both entry points agree about which file is UMIFormer and which is
UMIFormer+ -- a distinction that fails silently if the two ever disagree.
"""

import subprocess
import sys
from pathlib import Path

from kaggle_train import PIPELINE, resolve_inputs


def main():
    env = resolve_inputs()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\ntorch {torch.__version__}  device={device}")
    if device == "cpu":
        print("  WARNING: no GPU. UMIFormer is a ViT and is very slow on CPU;\n"
              "  enable one under Settings > Accelerator or use a small --limit.")

    cmd = [sys.executable, str(Path(PIPELINE) / "category_bench.py"),
           "--device", device] + sys.argv[1:]
    print("\n" + " ".join(cmd) + "\n" + "=" * 66)
    return subprocess.run(cmd, cwd=str(PIPELINE), env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
