# Kaggle training — step by step

Everything you upload is already built and sitting in `D:\rl_project\build\`.

---

## Step 1 — Upload three datasets

For each file below: **kaggle.com → Datasets → New Dataset → drag the file in →
give it the title → Create**. Kaggle unzips `.zip` uploads automatically.

| File in `D:\rl_project\build\` | Size | Title to give it |
|---|---|---|
| `kaggle_code.zip` | 1.0 MB | `rl-recon-code` |
| `kaggle_voxels.zip` | 21.8 MB | `shapenet-vox32` |
| `kaggle_weights.zip` | **1464 MB** | `rl-recon-weights` |

The weights upload is the only slow one. Start it first and do Step 2 while it runs.
It holds exactly the three registered backbones — `Pix2Vox-F-ShapeNet.pth`,
`UMIFormer-ShapeNet.pth`, `UMIFormerPlus-ShapeNet.pth`. All three are required:
`kaggle_train.py` now stops with an error rather than training on a smaller
action space if one is missing.

**Do not upload the renderings.** They are 12.9 GB and Step 3 attaches them instead.

---

## Step 2 — Create the notebook

1. **kaggle.com → Code → New Notebook**
2. Right sidebar → **Settings**:
   - **Accelerator → GPU T4 x2** (or P100). Without this UMIFormer's ViT is
     unusably slow.
   - **Internet → On**
3. **Persistence → Files only** (or Variables and Files). This is what lets
   `/kaggle/working` survive into the next session.

---

## Step 3 — Attach four datasets

Right sidebar → **Add Input** (or "Add Data"), and add all four:

1. `rl-recon-code` — yours
2. `shapenet-vox32` — yours
3. `rl-recon-weights` — yours
4. **Search "ShapeNet-Pix2Vox"** → `ShapeNet-Pix2Vox` (12.9 GB)

Number 4 is the renderings. It is the one thing not verified yet — Step 4 checks it.

Attached datasets live in `/kaggle/input` and do **not** count against the 20 GB
`/kaggle/working` quota, which is why the 12.9 GB one is free to use.

---

## Step 4 — Check the data before training

First cell. **Do not hardcode the input path** — Kaggle derives the dataset
slug from the title you typed, and expanding a zip can add a nesting level, so
`/kaggle/input/rl-recon-code/rl_pipeline` is a guess that usually fails. This
finds it instead:

```python
import shutil
from pathlib import Path

INPUT, WORKING = Path("/kaggle/input"), Path("/kaggle/working")

print("=== attached datasets ===")
for d in sorted(INPUT.iterdir()):
    try:
        kids = sorted(p.name for p in d.iterdir())
    except OSError:
        kids = ["<unreadable>"]
    print(f"  {d.name}/  ->  {kids[:8]}{' ...' if len(kids) > 8 else ''}")

# Depth-limited on purpose. A blind rglob over /kaggle/input is unusable once
# the renderings are attached -- that tree is ~1M files and walking it took
# 1117 seconds. All-digit directory names are ShapeNet synsets, so pruning
# them keeps this instant.
def find(name, root=INPUT, max_depth=5):
    hits, stack = [], [(root, 0)]
    while stack:
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
            if p.name == name:
                hits.append(p)
            elif not p.name.isdigit():
                stack.append((p, depth + 1))
    return sorted(hits, key=lambda p: len(p.parts))

found = find("rl_pipeline")
print()
print(f"=== rl_pipeline candidates: {len(found)} ===")
for p in found:
    print("  ", p)

if not found:
    raise SystemExit(
        "rl_pipeline not found under /kaggle/input. "
        "Check the listing above: attach the dataset made from kaggle_code.zip."
    )

src = found[0]
dst = WORKING / "rl_pipeline"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print()
print(f"copied {src}  ->  {dst}")
print(f"  {sum(1 for _ in dst.rglob('*.py'))} .py files")

for f in ("kaggle_train.py", "kaggle_bench.py", "category_bench.py",
          "backbones/__init__.py"):
    print(f"  {'OK  ' if (dst / f).is_file() else 'MISS'} {f}")
```

Then, in a second cell:

```python
!python /kaggle/working/rl_pipeline/kaggle_inspect.py
```

Read the **VERDICT** block at the bottom. It must say `usable`.

What matters in it:

| Line | Meaning |
|---|---|
| `24 views/model` | Your existing 30,595-episode checkpoint can be reused (see Step 6) |
| fewer than 24 | Fine, training adapts — but the old checkpoint cannot be carried over |
| `rendering_metadata.txt present` | Real cameras. Without it the coverage grid degrades badly |
| `RGBA` | Silhouettes and background compositing work correctly |
| `binvox files: 43783` | Ground truth found |

**If it says NOT usable**, the renderings dataset is wrong. Fall back to fetching
them inside Kaggle instead — replace the search in Step 3 with:

```python
!python /kaggle/working/rl_pipeline/kaggle_fetch_data.py --categories chair car table
```

then commit and attach that notebook's output. It streams from Stanford at
Kaggle's bandwidth, so it is minutes rather than hours, and needs no upload.

---

## Step 5 — Train

```python
!python /kaggle/working/rl_pipeline/kaggle_train.py
```

It finds the data and weights by name, installs the few packages Kaggle lacks,
puts the backbones on the GPU, and runs with an 11-hour budget so it stops and
saves before Kaggle's 12-hour cap kills the session.

Expect it to print the backbones it found, then per-episode lines like:

```
ep 1234 | reward +0.3421 | IoU 0.7203 | views 5 | cat chair | bb umiformer | ...
```

If a backbone prints `weights not found, skipping`, that one is missing — check
the weights dataset attached properly. Training still runs with the rest.

---

## Step 5b — Measure all 13 categories (optional)

```python
!python /kaggle/working/rl_pipeline/kaggle_bench.py --limit 10
```

Every backbone number in `EXPERIMENTS.md` comes from three categories
(airplane, car, chair) — no seekable mirror of the renderings carries more, and
the full archive is 12.3 GB. Kaggle has all 13 attached, so this is the one
place the rest can be measured.

It prints a per-category IoU table per budget, then which backbone wins where,
then the crossover budget per category. What to look for:

* **Does the UMIFormer to UMIFormer+ crossover hold in all 13?** It lands
  between 3 and 5 views in all three measured so far. If the flip budget varies
  by category, the model head has something real to condition on beyond view
  count, which is a stronger result than three categories support.
* **Does Pix2Vox-F win outside car?** It wins car at one view (0.8848 vs
  UMIFormer 0.8762) and nothing else so far. That single cell is the entire
  measured case for its slot, and the margin is 0.0086.

This is optional because the registry is already settled at three backbones.
Run it if you want the paper to rest on 13 categories rather than 3.

**Note:** OccNet and TripoSR are no longer shipped in the weights zip, so
`--all` (which re-tests the decision to drop them) cannot run. Add
`onet_img2mesh_3-f786b04a.pt` to `WEIGHTS` in `package_for_kaggle.py` and
re-upload if you want to reopen that.

---

## Step 6 — (Optional) reuse your trained checkpoint

**Only if Step 4 reported 24 views.** Your `ckpt_final.pt` holds 30,595 episodes
of view-selection training. To carry that forward instead of starting cold:

1. Upload `D:\rl_project\build\rl_pipeline\checkpoints\ckpt_final.pt` as a
   dataset, attach it
2. Edit the last line of `kaggle_train.py` to add `--load-view-head-only`, or run
   `train.py` directly with it

That loads every tensor that still matches and leaves only the new backbone-
selection head untrained. With fewer than 24 views the shapes differ and it will
correctly refuse.

---

## Step 7 — Continue past 12 hours

Training stops itself at 11 hours and writes `ckpt_resume.pt`. To continue:

1. **Save Version → Save & Run All (Commit)** — this persists `/kaggle/working`
   as the notebook's output
2. Open a **new notebook** (or new version), attach everything from Step 3 **plus
   the previous run's output** (Add Input → Your Work → the committed notebook)
3. Run the same cells

`kaggle_train.py` finds `ckpt_resume.pt` in the attached output, copies it in, and
passes `--resume` automatically. Repeat until the curriculum finishes.

---

## Afterwards

When it completes, `ckpt_final.pt` appears in `/kaggle/working/checkpoints/`.
Download it from the notebook's Output tab. Run it locally with:

```bash
python infer.py --object-dir <a model dir> --policy-ckpt <downloaded ckpt>
```

---

## Quick troubleshooting

| Symptom | Cause |
|---|---|
| `Could not find ShapeNetRendering/ and ShapeNetVox32/` | A dataset did not attach, or has a different internal layout — rerun Step 4 |
| `No backbone weights found` | `rl-recon-weights` not attached |
| `CUDA out of memory` | Set `USE_GPU_BACKBONES = False` near the top of `kaggle_train.py` |
| Very slow, no GPU line at startup | Accelerator not enabled (Step 2) |
| Session died with no checkpoint | `MAX_HOURS` too high for your quota — lower it in `kaggle_train.py` |
