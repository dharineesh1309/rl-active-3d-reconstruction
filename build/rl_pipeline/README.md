# Joint RL for Next-Best-View and Reconstructor Selection

A PPO agent that, for one object, picks **which viewpoints to look from** and then
**which reconstruction backbone should build the model** — the joint MDP the project
report describes but the original code never implemented.

The episode is the report's finite horizon `H = B + 1`:

```
steps 1..B    choose a viewpoint (no revisits)
step  B+1     choose a reconstruction backbone
reward        IoU(pred, gt) - lambda * (B + 1 + backbone_cost)  [terminal]
              + shaping_coef * (gamma*Phi(s') - Phi(s))          [every step]

Phi(s) = fraction of the 32**3 coverage grid filled. The shaping terms are
potential-based (Ng, Harada & Russell 1999), so they telescope to -Phi(s_0) and
the set of optimal policies is provably unchanged -- only the credit assignment
gets denser. Needed because a terminal-only reward gives all B view steps one
undifferentiated signal; measured, the view policy then sits at 90% of uniform
entropy after 30k episodes while a perfect planner would gain +0.04 IoU.
GenNBV (CVPR 2024) trains its NBV policy on the same signal.

The `B + 1` counts the free initial view each episode is seeded with, so that
the first decision is conditioned on something rather than on an all-zero state.
It is a constant within an episode and does not affect the backbone choice.

Set `Config.shaping_coef = 0.0` and `seed_initial_view = False` to recover the
original sparse, cold-start formulation for an ablation.
```

Two actor heads share one trunk and one action space; the `is_model_step` observation
flag routes each step to the right head.

---

## Quick start

```bash
pip install -r requirements.txt

export SHAPENET_RENDERING_ROOT=../ShapeNetRendering
export SHAPENET_VOXEL_ROOT=../ShapeNetVox32

python -m backbones.selftest                  # weights load, shapes correct
python -m backbones.bench --split test --categories chair --views 1 3 5
python env/tests/test_env.py                  # environment contract
python training/tests/test_ppo_loop.py        # full PPO loop, no data needed
python train.py --smoke-test --dummy-vec --n-envs 2
```

---

## Backbones

Three backbones, each of which **measurably wins somewhere**. That is the selection
criterion: an action a reward-maximising policy can never choose is not a weak
action, it is an unreachable one, and it still costs an index in the model head.
The registry order in `backbones/__init__.py` **is** the action index — append,
never insert, or a trained head silently points at the wrong model.

| Backbone | Paradigm | Weights | Wins |
|---|---|---|---|
| `pix2vox_f` | CNN + attention fusion -> voxel | `Pix2Vox-F-ShapeNet.pth` (30 MB) | car at 1 view; cheap, so more of the low-budget regime as lambda rises |
| `umiformer` | transformer over multi-view tokens | `UMIFormer-ShapeNet.pth` (717 MB) | 1-3 views |
| `umiformer_plus` | same architecture, many-view training | `UMIFormerPlus-ShapeNet.pth` (717 MB) | 5+ views |

Measured on the ShapeNet **test** split, 20 models per category, best single
threshold per cell over {.2,.3,.4,.5} (see EXPERIMENTS.md §10):

| budget | category | `pix2vox_f` | `umiformer` | `umiformer_plus` |
|---|---|---|---|---|
| 1 | aeroplane | 0.5807 | **0.6547** | 0.5588 |
| 1 | car | **0.8848** | 0.8762 | 0.7984 |
| 1 | chair | 0.5948 | **0.6383** | 0.5581 |
| 3 | aeroplane | 0.6208 | **0.7146** | 0.6862 |
| 3 | car | 0.9007 | **0.9131** | 0.9090 |
| 3 | chair | 0.6321 | **0.7320** | 0.7299 |
| 5 | aeroplane | 0.6338 | 0.7181 | **0.7369** |
| 5 | car | 0.9057 | 0.9144 | **0.9217** |
| 5 | chair | 0.6567 | 0.7480 | **0.7560** |
| 8 | aeroplane | 0.6346 | 0.7275 | **0.7489** |
| 8 | car | 0.9064 | 0.9187 | **0.9226** |
| 8 | chair | 0.6605 | 0.7536 | **0.7702** |

Compute at 5 views on CPU: `pix2vox_f` 0.51 s (cost 0.09), `umiformer` and
`umiformer_plus` 1.28 s (cost 0.938). UMIFormer's chair numbers land within 0.005
of its own checkpoint's claimed `best_iou 0.7529` — about as strong a correctness
signal as this gate gives.

**The UMIFormer / UMIFormer+ pair is the important part.** Identical architecture
means identical cost, so the choice between them is *pure quality and independent
of `cost_lambda`* — it stays live even at lambda = 0. That answers the fair
objection to a cost-driven split: the decision exists in the models, not in a
reward parameter tuned until it looked interesting. The crossover sits between 3
and 5 views in **all three** categories measured.

**Why Pix2Vox-F and not -A.** Pix2Vox-A was measured too (0.638 / 0.665 / 0.667
at 1.34 s) and UMIFormer *strictly dominates* it: equal at one view (0.6383 vs
0.6381), better at every larger budget, and slightly cheaper. There is no
positive lambda at which -A wins, so it would be a dead option.

**Dropped after measurement.** `occnet` and `triposr` won **zero of twelve**
category-budget cells (means 0.271 and 0.153 against UMIFormer's 0.800). OccNet
was kept on the theory that it would do better on complex shapes and worse on
simple ones; it is instead uniformly second-to-last. TripoSR is single-view by
construction, so it is flat across budgets and cannot benefit from view planning
at all. Both modules remain on disk with their calibrations —
`load_backbones(..., include_unregistered=True)`, or `category_bench --all` —
so the decision can be re-tested on all 13 categories. Training never sees them.

This costs the original "three families" framing (voxel / NeRF / implicit): the
NeRF and implicit members were the two that could not compete on this data.

A backbone missing its weights drops out of the action space at startup rather than
crashing, so the pipeline runs with whatever is present.

### Getting the weights

```
pix2vox_f       https://gateway.infinitescript.com/?fileName=Pix2Vox-F-ShapeNet.pth   (captcha - browser only)
umiformer       https://drive.usercontent.google.com/download?id=1kgqhxsm-H3MCCjYz5Ur1Hlmt6onYCn_g&export=download&confirm=t
umiformer_plus  https://github.com/GaryZhu1996/UMIFormer  (the "+" checkpoint)

# no longer registered, only needed to re-test the decision to drop them:
occnet          https://s3.eu-central-1.amazonaws.com/avg-projects/occupancy_networks/models/onet_img2mesh_3-f786b04a.pt
triposr         https://huggingface.co/stabilityai/TripoSR
```

Paths come from `config.py` or the matching env vars: `PIX2VOX_F_CKPT`,
`UMIFORMER_CKPT`, `UMIFORMER_PLUS_CKPT` (and `OCCNET_CKPT`, `TRIPOSR_DIR`).

### Not registered, and why

* **`r2n2`** — weights fine (215,798,999 bytes, 63 arrays), mapping provably correct
  (exact 35,968,706 parameter match), but the forward pass scores **IoU 0.031** vs a
  published 0.466. A port bug, not a weights bug. Left out so it cannot poison the
  reward. See the module docstring.
* **`pixelnerf`** — strict-loads all 278 tensors, but the cameras reconstructed from
  Choy's `rendering_metadata.txt` do not agree with the renderings: projecting known
  ground-truth voxels through them puts only **0.55** of the projection inside the
  silhouette, where a correct convention gives ~0.95, and that was the best of 1,536
  conventions swept. Fix is to use the DVR/NMR release, which ships exact
  `cameras.npz`. See the module docstring.
* **`pix2vox_a`** — works (0.638/0.665/0.667) but is strictly dominated by
  UMIFormer at equal-or-lower quality and higher cost, so it can never win.

---

## Data

The canonical Choy et al. 2016 ShapeNet release. It fits this project exactly: **24
views** per object (matching `n_views`), **native 32³ voxels** (matching the scoring
resolution, so nothing is ever resampled), and real per-view camera parameters.

```
ShapeNetVox32.tgz      22 MB    http://cvgl.stanford.edu/data2/ShapeNetVox32.tgz
ShapeNetRendering.tgz  12.3 GB  http://cvgl.stanford.edu/data2/ShapeNetRendering.tgz
```

`fetch_eval_subset.py` pulls a small chair subset for benchmarking without the 12.3 GB
download, by range-requesting a ZIP mirror's central directory (~25 MB instead of ~4 GB).
**That mirror has 10 views per model, not 24** — fine for benchmarking, not sufficient
for training, whose action space assumes 24.

---

## Training on a GPU box

Training does not run usefully on CPU. Everything is env-var driven:

```bash
export SHAPENET_ROOT=/data/ShapeNet          # holds ShapeNetRendering/ and ShapeNetVox32/
export PIX2VOX_F_CKPT=/weights/Pix2Vox-F-ShapeNet.pth
export OCCNET_CKPT=/weights/onet_img2mesh_3-f786b04a.pt
export UMIFORMER_CKPT=/weights/UMIFormer-ShapeNet.pth

python train.py --n-envs 4
```

Backbones load on **CPU inside each env worker** — keep it that way, it is why
`SubprocVecEnv` works without CUDA pickling errors. Three backbones × N workers is
several GB of RAM; drop `--n-envs` if memory is tight.

To carry a checkpoint trained before the model head existed:

```bash
python train.py --resume --load-view-head-only
```

That loads every tensor whose name and shape still match and leaves the model head
fresh, preserving the expensive view-selection training.

---

## Training on Kaggle

Kaggle imposes two constraints the plain entry point does not handle:
**sessions are capped** (12 hours, sometimes 9) and **`/kaggle/working` is wiped
between them**. `kaggle_train.py` handles both.

### Getting the data onto Kaggle

The renderings are 12.3 GB, so do **not** download them at home and upload them.
In order of preference:

1. **Attach a public dataset.** An attached dataset lives in `/kaggle/input` and
   does not count against the 20 GB `/kaggle/working` quota, so this costs
   nothing. the public "ShapeNet-Pix2Vox" dataset (12.9 GB) looks
   like the right tree — check it contains `ShapeNetRendering/<synset>/<model>/rendering/`.
2. **Fetch inside Kaggle.** Run `kaggle_fetch_data.py` in its own notebook with
   Internet enabled, commit it, and attach that output to the training notebook:

   ```bash
   !python /kaggle/working/rl_pipeline/kaggle_fetch_data.py --categories chair car table
   ```

   It streams the archive straight into `tar` rather than saving it, because the
   12.3 GB `.tgz` plus its ~25 GB expansion does not fit in `/kaggle/working`.
   For the same reason it extracts selected categories only — all 13 will not fit.

### One-time setup

```bash
python ../package_for_kaggle.py     # builds kaggle_code.zip (1 MB) + kaggle_weights.zip (908 MB)
```

Upload each as a Kaggle Dataset, then attach to a notebook along with
ShapeNetRendering and ShapeNetVox32 (either your own upload, or an existing
public ShapeNet dataset). Enable a GPU under *Settings > Accelerator*.

### Each session

```python
!cp -r /kaggle/input/<your-code-dataset>/rl_pipeline /kaggle/working/
!python /kaggle/working/rl_pipeline/kaggle_train.py
```

The driver locates the data and weights under `/kaggle/input` by name, installs
the few packages Kaggle lacks, and runs training with `--max-hours 11` so it
stops itself and writes `ckpt_resume.pt` before the session is killed.

### Continuing across sessions

1. **Commit** the notebook — that saves `/kaggle/working` as its output.
2. In the next session, **attach the previous notebook's output** as a dataset.
3. Run the same cell. It finds `ckpt_resume.pt` under `/kaggle/input`, stages it,
   and passes `--resume` automatically.

`CheckpointManager.load_latest()` prefers `ckpt_resume.pt` over any other
checkpoint for exactly this reason — ranking by episode count alone would let an
unrelated `ckpt_final.pt` with more episodes shadow the run you are continuing,
and every resume would then fail on a shape mismatch.

### GPU backbones

`USE_GPU_BACKBONES = True` (the default) runs the backbones and the ResNet-50
state encoder on the GPU, which implies `--dummy-vec`: CUDA tensors cannot be
pickled into `SubprocVecEnv` workers. On a T4 this is much faster than CPU
workers regardless, because UMIFormer is a ViT and is very slow on CPU. Set it
to `False` to fall back to parallel CPU workers.

---

## Tuning lambda

`Config.cost_lambda` prices the backbone choice. Because the view budget `B` is
*sampled by the environment* rather than chosen by the agent, the `-lambda*B` term is a
constant offset within an episode — lambda only ever affects the **backbone** decision.

`cost_lambda` is **0.07**, derived from the 3-category table above. Per-cell
crossover values span (0.0146, 0.1348), but width is the wrong criterion. What
matters is the sub-range where the best backbone depends on the **object** and
not only the budget, because that is exactly what a learned policy can exploit
and a lookup table cannot. At 0.07:

| budget | aeroplane | car | chair |
|---|---|---|---|
| B=1 | `umiformer` | `pix2vox_f` | `pix2vox_f` |
| B=3 | `umiformer` | `pix2vox_f` | `umiformer` |
| B=5 | `umiformer_plus` | `pix2vox_f` | `umiformer_plus` |
| B=8 | `umiformer_plus` | `pix2vox_f` | `umiformer_plus` |

All three backbones win somewhere, and the one-view choice splits by category.

0.07 is also the most robust point in that region: it sits 0.017 from the nearest
crossover, where 0.05 is 0.0013 away and 0.09 is 0.0027 away — both inside
measurement noise, so either would have the policy's target flipping between
runs. Re-derive with `category_bench.py` whenever the backbone set or the
hardware changes — the costs are wall-clock dependent.

Note the UMIFormer / UMIFormer+ half of the decision does **not** depend on this
value at all: same architecture, same cost, decided purely on quality.

---

## Known limitations

* **Evaluation covers 3 of 13 categories** (aeroplane, car, chair), 20 test models
  each. No seekable mirror of ShapeNetRendering carries more than three, so the
  full sweep has to run where the canonical data already sits — `kaggle_bench.py`.
  Conclusions that rest on a single cell are the fragile ones: Pix2Vox-F's slot
  rests entirely on car at one view, on a 0.0086 margin.
* **These mirrors ship 10 views per model, not Choy's 24**, so B=8 uses 8 of 10.
  Budget-ordering conclusions hold; absolute values at high budget are slightly
  pessimistic against published 24-view figures. Training assumes 24 and needs
  the canonical renderings.
* **Score at one threshold per cell, never per model.** Taking each model's best
  threshold is an oracle the policy does not have at inference and inflates IoU by
  ~0.015 — larger than several margins the conclusions rest on (UMIFormer vs
  UMIFormer+ at 3 views is 0.0021).
* Benchmark on the **test** split. Train-split models inflate badly — Pix2Vox-A scored
  0.84 there versus 0.67 on test.
* **`occnet`, if you re-enable it**, is scored against binvox ground truth while
  trained against watertight-mesh occupancy, and its released model is
  single-image conditioned with mean-pool multi-view fusion added here. Its
  absolute number is therefore not comparable to its paper.

---

## Layout

```
backbones/          one module per backbone + shared Backbone protocol and voxel_iou
  selftest.py       weights load, shapes sane
  bench.py          IoU vs published figures — the integration correctness gate
  calibrate_pixelnerf.py
  _vendor/          upstream network code, imports rewritten to be local
env/                ViewReconEnv, state builders, tests
policy/             ViewPolicy — two actor heads, shared trunk
training/           PPO trainer, rollout buffer, vec envs, loop test
dataloader.py       ModelNet loader (original)
dataloader_shapenet.py  Choy ShapeNet loader
fetch_eval_subset.py    range-request fetcher for a small eval subset
infer.py            run a trained policy on one object
```
