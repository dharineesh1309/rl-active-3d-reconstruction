"""
kaggle_inspect.py — report whether an attached Kaggle dataset can drive this pipeline.

    !python /kaggle/working/rl_pipeline/kaggle_inspect.py

Self-contained (stdlib + PIL only) so it can also be pasted straight into a
notebook cell before any code has been uploaded.

It checks the specific things dataloader_shapenet.py depends on, rather than
just listing files:

  * a ShapeNetRendering-style tree: <root>/<synset>/<model>/rendering/NN.png
  * how many views per model (the policy's action space assumes 24)
  * rendering_metadata.txt, which supplies the per-view cameras
  * RGBA renderings — alpha is what the silhouettes and the backbones' background
    compositing rely on
  * ShapeNetVox32 binvox ground truth at 32**3
"""

import sys
from collections import Counter
from pathlib import Path

INPUT = Path("/kaggle/input")


def find_trees(root: Path, names=("ShapeNetRendering", "ShapeNetVox32"),
               max_depth=6):
    """
    Locate the data roots without walking into them.

    `root.rglob(name)` is the obvious version and it is unusable here: once the
    renderings are attached, /kaggle/input holds roughly a million files, and
    rglob descends into every synset and every model directory before it can
    report anything. Measured, that ran for 1117 seconds and printed nothing.

    Two bounds fix it. Depth is capped, because the trees sit within a few
    levels of /kaggle/input however the dataset is nested. And directories whose
    names are all digits are ShapeNet synsets, so they are never descended --
    that single prune is what removes the million files.
    """
    found, stack = {}, [(root, 0)]
    while stack and len(found) < len(names):
        d, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for p in entries:
            if not p.is_dir():
                continue
            if p.name in names and _is_real_root(p):
                found.setdefault(p.name, p)
            elif not p.name.isdigit():
                stack.append((p, depth + 1))
            elif p.name in names:
                stack.append((p, depth + 1))
    return found


def _is_real_root(p: Path) -> bool:
    """
    True when `p` holds synsets rather than another directory of the same name.

    Some published datasets nest the tree inside a directory with the same name
    (ShapeNetRendering/ShapeNetRendering/<synset>). Both match by name, and the
    outer one is the wrong answer -- its only child is a directory, so it
    reports one 'synset' with no renderings and no binvox, which reads as a
    broken dataset when nothing is wrong. A real root's children are numeric
    synset ids.
    """
    try:
        kids = [k for k in p.iterdir() if k.is_dir()]
    except OSError:
        return False
    return bool(kids) and all(k.name.isdigit() for k in kids)


def main():
    if not INPUT.is_dir():
        print("No /kaggle/input - run this inside a Kaggle notebook.")
        return 1

    print("=" * 68)
    print("  ATTACHED DATASETS")
    print("=" * 68)
    for d in sorted(INPUT.iterdir()):
        if d.is_dir():
            try:
                kids = sorted(p.name for p in d.iterdir())[:6]
            except OSError:
                kids = ["<unreadable>"]
            print(f"  {d.name}/")
            print(f"      {kids}")

    trees = find_trees(INPUT)
    print("\n" + "=" * 68)
    print("  TREES FOUND")
    print("=" * 68)
    for name in ("ShapeNetRendering", "ShapeNetVox32"):
        print(f"  {name:<20} {trees.get(name, 'NOT FOUND')}")

    # ── Renderings ───────────────────────────────────────────────────────────
    render = trees.get("ShapeNetRendering")
    if render:
        synsets = sorted(p for p in render.iterdir() if p.is_dir())
        print(f"\n  synsets: {len(synsets)}")
        total = 0
        for s in synsets[:15]:
            n = sum(1 for _ in s.iterdir())
            total += n
            print(f"    {s.name}  {n} models")
        print(f"  total models (first 15 synsets): {total}")

        # Sample a few models and check the per-model layout.
        print("\n  --- per-model layout (3 samples) ---")
        view_counts, modes, sizes, has_meta = Counter(), Counter(), Counter(), 0
        sampled = 0
        for s in synsets:
            for model in s.iterdir():
                if sampled >= 3:
                    break
                rend = model / "rendering"
                if not rend.is_dir():
                    print(f"    {model.name}: NO rendering/ subdir -> "
                          f"files are {sorted(p.name for p in model.iterdir())[:6]}")
                    sampled += 1
                    continue
                pngs = sorted(rend.glob("*.png"))
                view_counts[len(pngs)] += 1
                meta = (rend / "rendering_metadata.txt")
                has_meta += meta.is_file()
                print(f"    {s.name}/{model.name}: {len(pngs)} png, "
                      f"metadata={'yes' if meta.is_file() else 'NO'}, "
                      f"files={sorted(p.name for p in rend.iterdir())[:4]}")
                if meta.is_file():
                    first = meta.read_text().strip().splitlines()[:1]
                    print(f"        metadata line 1: {first}")
                if pngs:
                    try:
                        from PIL import Image
                        im = Image.open(pngs[0])
                        modes[im.mode] += 1
                        sizes[im.size] += 1
                        print(f"        image: {im.size} {im.mode}")
                    except Exception as e:
                        print(f"        image open failed: {e}")
                sampled += 1
            if sampled >= 3:
                break

        print(f"\n  views/model seen : {dict(view_counts)}")
        print(f"  image modes      : {dict(modes)}   (need RGBA for silhouettes)")
        print(f"  image sizes      : {dict(sizes)}   (Choy renders are 137x137)")

    # ── Voxels ───────────────────────────────────────────────────────────────
    vox = trees.get("ShapeNetVox32")
    if vox:
        # Sample rather than enumerate: rglob over 43k binvox files is slow and
        # the question here is only "is there ground truth in the expected
        # shape", which one file per synset answers.
        binvox = []
        for syn in sorted(p for p in vox.iterdir() if p.is_dir()):
            for model in syn.iterdir():
                if (model / "model.binvox").is_file():
                    binvox.append(model / "model.binvox")
                    break
        print(f"\n  synsets with binvox: {len(binvox)}")
        if binvox:
            with open(binvox[0], "rb") as f:
                header = [f.readline().decode(errors="replace").strip() for _ in range(4)]
            print(f"  sample header: {header}")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  VERDICT")
    print("=" * 68)
    ok = True
    if not render:
        print("  X no ShapeNetRendering tree -> cannot train"); ok = False
    if not vox:
        print("  X no ShapeNetVox32 tree -> no ground truth, no reward"); ok = False
    if render and vox:
        if 24 in view_counts:
            print("  + 24 views/model: matches the policy's action space")
        elif view_counts:
            k = max(view_counts, key=view_counts.get)
            print(f"  ! {k} views/model, not 24. Training still runs (train.py adapts "
                  f"n_views) but the existing 30,595-episode checkpoint cannot be reused.")
        if has_meta:
            print("  + rendering_metadata.txt present: real cameras for the coverage grid")
        else:
            print("  ! no rendering_metadata.txt: depth back-projection degrades to zeros")
        if "RGBA" in modes:
            print("  + RGBA: silhouettes and background compositing will work")
        elif modes:
            print(f"  ! mode {list(modes)} not RGBA: silhouettes fall back to a "
                  f"white-pixel threshold")
    print("\n  usable" if ok else "\n  NOT usable as-is")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
