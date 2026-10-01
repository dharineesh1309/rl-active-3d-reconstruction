"""
kaggle_train.py — one-cell driver for training this project on Kaggle.

Paste into a Kaggle notebook cell and run:

    !python /kaggle/working/rl_pipeline/kaggle_train.py

Or import and call `main()`.

Why this exists rather than calling train.py directly
─────────────────────────────────────────────────────
Kaggle imposes two things the plain entry point does not handle by itself:

  * **Sessions are capped** (12 hours, often 9). A run that is still going when
    the cap hits is killed, losing everything since the last checkpoint. This
    script passes `--max-hours` with a safety margin so training stops itself
    and writes a resumable checkpoint first.

  * **`/kaggle/working` is wiped between sessions.** To continue, the previous
    notebook's *output* has to be attached to the next session as a dataset.
    This script looks through `/kaggle/input/**` for a `ckpt_resume.pt` (or any
    checkpoint) left by an earlier session and copies it in before resuming, so
    the multi-session loop needs no manual file shuffling.

Expected Kaggle inputs (attach these as datasets, names are matched loosely):

    /kaggle/input/<any>/ShapeNetRendering/      24-view renderings
    /kaggle/input/<any>/ShapeNetVox32/          32**3 ground-truth voxels
    /kaggle/input/<any>/*.pth, *.pt             backbone weights
    /kaggle/input/<any>/checkpoints/            previous session's output (optional)

Set `USE_GPU_BACKBONES = True` (default) to run the backbones and the ResNet-50
state encoder on the GPU. That forces DummyVecEnv — CUDA tensors cannot be
pickled into SubprocVecEnv workers — but on a T4 it is far faster than CPU
workers, because UMIFormer is a ViT and is very slow on CPU.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── Tunables ──────────────────────────────────────────────────────────────────

MAX_HOURS = 11.0          # Kaggle caps at 12h; leave room to save and commit
USE_GPU_BACKBONES = True  # GPU backbones (implies DummyVecEnv) vs CPU workers
N_ENVS_CPU = 4            # only used when USE_GPU_BACKBONES is False
CATEGORIES = None         # None = all 13; or e.g. ["chair", "car"]

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
PIPELINE = Path(__file__).resolve().parent


# ── Discovery ─────────────────────────────────────────────────────────────────

# Depth-limited walk. A blind INPUT.rglob() is unusable here: a full
# ShapeNetRendering tree is ~1M files, so recursing it to find a directory takes
# minutes. Everything we need sits within a few levels of /kaggle/input.
MAX_DEPTH = 5


def _walk(root: Path, depth: int = 0):
    if depth > MAX_DEPTH:
        return
    try:
        entries = list(root.iterdir())
    except OSError:
        return
    for p in entries:
        yield p, depth
        # Never into a synset (all-digit) directory: below it lie ~43k model
        # folders and a million renderings, and nothing this walk looks for.
        if p.is_dir() and not p.name.isdigit():
            yield from _walk(p, depth + 1)


def find_dir(name: str):
    """
    Shallowest directory called `name` under /kaggle/input.

    Shallowest, not first: some published datasets nest the tree inside a
    directory of the same name (ShapeNetRendering/ShapeNetRendering/<synset>).
    Both match, and the outer one is the wrong answer -- its children are a
    single directory rather than the 13 synsets. So prefer the shallowest match
    but reject one whose children do not look like synset ids.
    """
    if not INPUT.is_dir():
        return None
    hits = [p for p, _ in _walk(INPUT) if p.is_dir() and p.name == name]
    hits.sort(key=lambda p: len(p.parts))
    for h in hits:
        try:
            kids = [k for k in h.iterdir() if k.is_dir()]
        except OSError:
            continue
        # A real ShapeNet root holds numeric synset ids, not one nested dir.
        if kids and all(k.name.isdigit() for k in kids):
            return h
    return hits[0] if hits else None


def find_file(*patterns):
    """Largest file under /kaggle/input matching any glob, depth-limited."""
    if not INPUT.is_dir():
        return None
    hits = [p for p, _ in _walk(INPUT)
            if p.is_file() and any(p.match(pat) for pat in patterns)]
    return max(hits, key=lambda p: p.stat().st_size) if hits else None


def resolve_inputs() -> dict:
    """
    Find the data and every backbone checkpoint; return them as an env dict.

    Shared by training and by the per-category benchmark so the two cannot drift
    apart -- in particular over the UMIFormer disambiguation below, which is
    silent when it goes wrong.
    """
    rendering = (Path(os.environ["SHAPENET_RENDERING_ROOT"])
                 if os.environ.get("SHAPENET_RENDERING_ROOT")
                 else find_dir("ShapeNetRendering"))
    voxels = (Path(os.environ["SHAPENET_VOXEL_ROOT"])
              if os.environ.get("SHAPENET_VOXEL_ROOT")
              else find_dir("ShapeNetVox32"))
    print(f"\nrenderings : {rendering}")
    print(f"voxels     : {voxels}")
    if not rendering or not voxels:
        raise SystemExit(
            "Could not find ShapeNetRendering/ and ShapeNetVox32/ under /kaggle/input.\n"
            "Attach them via 'Add Data' in the notebook sidebar."
        )

    # UMIFormer and UMIFormer+ ship files of byte-identical size, so a shared
    # glob ("UMIFormer*.pth") matches both and find_file's largest-file
    # tie-break resolves arbitrarily. That would quietly load one checkpoint
    # into both slots, making the pair -- the one lambda-independent decision
    # the model head has to learn -- two copies of the same network. Match each
    # by exact filename instead.
    weights = {
        "PIX2VOX_F_CKPT": find_file("Pix2Vox-F-ShapeNet.pth"),
        "UMIFORMER_CKPT": find_file("UMIFormer-ShapeNet.pth"),
        "UMIFORMER_PLUS_CKPT": find_file("UMIFormerPlus-ShapeNet.pth",
                                         "UMIFormer+-ShapeNet.pth",
                                         "UMIFormer-Plus-ShapeNet.pth"),
    }
    if (weights["UMIFORMER_CKPT"] and weights["UMIFORMER_PLUS_CKPT"]
            and weights["UMIFORMER_CKPT"] == weights["UMIFORMER_PLUS_CKPT"]):
        raise SystemExit("UMIFormer and UMIFormer+ resolved to the same file; "
                         "rename one so the two are distinguishable.")

    print("\nregistered backbones (all three are required):")
    for var, path in weights.items():
        print(f"  {var:<22} {path if path else '*** NOT FOUND ***'}")

    # Every registered backbone must be present. A missing one is skipped with
    # only a notice by load_backbones(), so without this check training would
    # run happily with a two-action model head and the run would look fine --
    # the failure would only show up as a policy that cannot reproduce the
    # measured crossover.
    absent = [v for v, p in weights.items() if not p]
    if absent:
        raise SystemExit(
            f"Missing registered backbone weights: {absent}.\n"
            "Training would silently run with a smaller action space. Check that\n"
            "the weights dataset attached and holds all three .pth files."
        )

    env = os.environ.copy()
    env["SHAPENET_RENDERING_ROOT"] = str(rendering)
    env["SHAPENET_VOXEL_ROOT"] = str(voxels)
    for var, path in weights.items():
        if path:
            env[var] = str(path)
    return env


def stage_previous_checkpoint(ckpt_dir: Path) -> bool:
    """
    Copy a checkpoint left by an earlier session into the working checkpoint dir.

    Prefers ckpt_resume.pt, which is what the time-budget stop writes. Returns
    True if anything was staged, i.e. whether this run should resume.
    """
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if (ckpt_dir / "ckpt_resume.pt").is_file():
        return True

    for name in ("ckpt_resume.pt", "ckpt_final.pt"):
        found = find_file(name)
        if found:
            shutil.copy2(found, ckpt_dir / name)
            print(f"  staged {name} from {found.parent}")
            return True

    newest = None
    if INPUT.is_dir():
        candidates = [p for p, _ in _walk(INPUT)
                      if p.is_file() and p.match("ckpt_ep*.pt")]
        if candidates:
            newest = max(candidates, key=lambda p: p.stat().st_mtime)
    if newest:
        shutil.copy2(newest, ckpt_dir / newest.name)
        print(f"  staged {newest.name} from {newest.parent}")
        return True
    return False


def ensure_dependencies():
    """Install what Kaggle does not ship by default."""
    for module, package in (("gymnasium", "gymnasium"), ("einops", "einops"),
                            ("timm", "timm"), ("cv2", "opencv-python")):
        try:
            __import__(module)
        except ImportError:
            print(f"  installing {package} ...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", package],
                           check=True)


def preflight(require_cuda: bool = True):
    """Dependencies, the GPU, and the ResNet-50 weights the policy reads
    images with -- all checked before anything long starts."""
    ensure_dependencies()
    import torch
    if require_cuda and not torch.cuda.is_available():
        raise SystemExit("No GPU. Enable one under Settings > Accelerator.")
    print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}"
          + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    from env.rgb_view_env import ResNetFeatures
    ResNetFeatures(device="cuda" if torch.cuda.is_available() else "cpu")
    print("ResNet-50 weights: loaded")


def main():
    print("=" * 66)
    print("  Kaggle training driver")
    print("=" * 66)

    ensure_dependencies()
    import torch
    print(f"\ntorch {torch.__version__}  cuda={torch.cuda.is_available()}"
          + (f"  ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    if not torch.cuda.is_available():
        print("  WARNING: no GPU. Enable one under Settings > Accelerator, or this\n"
              "  will be far too slow to finish a curriculum.")

    env = resolve_inputs()

    # ── Environment ──────────────────────────────────────────────────────────
    ckpt_dir = WORKING / "checkpoints"
    resuming = stage_previous_checkpoint(ckpt_dir)
    print(f"\nresuming from a previous session: {resuming}")

    # Keep checkpoints and logs in /kaggle/working so they survive as notebook
    # output — that output is what the next session attaches to resume from.
    env["CHECKPOINT_DIR"] = str(ckpt_dir)
    env["LOG_DIR"] = str(WORKING / "logs")

    cmd = [sys.executable, str(PIPELINE / "train.py"),
           "--dataset", "shapenet", "--max-hours", str(MAX_HOURS)]
    if USE_GPU_BACKBONES and torch.cuda.is_available():
        cmd += ["--dummy-vec", "--backbone-device", "cuda"]
    else:
        cmd += ["--n-envs", str(N_ENVS_CPU)]
    if resuming:
        cmd.append("--resume")

    print("\n" + " ".join(cmd) + "\n" + "=" * 66)
    result = subprocess.run(cmd, cwd=str(PIPELINE), env=env)

    print("\n" + "=" * 66)
    print(f"  training exited with {result.returncode}")
    print(f"  checkpoints in {ckpt_dir}:")
    for f in sorted(ckpt_dir.glob("*.pt")):
        print(f"    {f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    print("\n  To continue in a new session: commit this notebook, then attach its\n"
          "  output as a dataset to the next run. ckpt_resume.pt is picked up\n"
          "  automatically.")
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
