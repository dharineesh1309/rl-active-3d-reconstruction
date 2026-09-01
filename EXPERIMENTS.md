# Experiments log

A record of what was built, what was measured, and what failed, for the joint
next-best-view and reconstructor-selection agent in `build/rl_pipeline`.

Every number here is measured in this project unless marked *published*. Failed
attempts are included with the same detail as successes — several of them are
the most informative results.

Unless stated otherwise: **ShapeNet test split, chair category, IoU at 32³
against ShapeNetVox32 binvox, reported as the best mean over thresholds
{0.2, 0.3, 0.4, 0.5}** (matching Pix2Vox's own evaluation sweep).

---

## 1. Starting point

The project report claims a PPO agent doing next-best-view planning **and**
reconstruction-backbone selection. The shipped code did only the first:
`policy/view_policy.py` stated *"The old model-selection head has been fully
removed. There is exactly one reconstructor."* The report itself lists
multi-backbone selection as future work.

Three defects were found in the original code before any experiment could run:

| Defect | Effect |
|---|---|
| `dataloader.py` appended all-zero depth maps; `_depth_from_voxels` was written but never called | The 32³ coverage grid collapsed to one z-slice, so the policy's 3D CNN branch saw almost no geometry |
| `train.py` deleted `ckpt_*.pt` on a fresh run | Destroyed `ckpt_final.pt`, the only artifact of 30,595 training episodes |
| `utils/logger.py` read `Config` *class* attributes | `--smoke-test` overrides were silently ignored |

A fourth surfaced during the first Kaggle run: non-ASCII characters (`π`, `×`)
in the training progress output raise `UnicodeEncodeError` on a cp1252 console,
killing a run mid-training.

---

## 2. Dataset

Switched from ModelNet to the **Choy et al. 2016 ShapeNet** release, which fits
the existing design exactly on three axes:

| | Choy ShapeNet | Project |
|---|---|---|
| Views per object | 24 | `n_views = 24` |
| Voxel ground truth | native 32³ | scoring resolution 32³ |
| Camera protocol | 24-pose orbit + `rendering_metadata.txt` | *"matching the 3D-R2N2 rendering protocol"* |

This removed all resampling from the reward path and made every backbone
in-distribution, since they are all pretrained on this data.

**Note on the report's description.** Choy's viewpoints are *randomly sampled
per object* (azimuths scatter: 209°, 41°, 182°…, distance varies 0.66–0.94),
not the "24 uniformly-distributed poses with 15° azimuth separation" the report
describes.

### Fetching without the 12.3 GB archive

The renderings ship as a single `.tgz`. Gzip cannot be seeked, so reaching one
synset means streaming ~4 GB — at the 130 KB/s measured from Stanford, about
five hours. A Hugging Face mirror publishes the same chair data as a **ZIP**,
whose index sits at the end of the file, so `zipfile` over HTTP range requests
gives true random access: **25 MB of index plus a few MB of images** instead of
4 GB. Implemented in `fetch_eval_subset.py`.

That mirror has 10 views per model rather than 24 — sufficient for
benchmarking, not for training, whose action space assumes 24.

A second mirror (`r2n2_shapenet_dataset_full.zip`, 13.69 GB, 71 MB index)
carries **three** synsets — airplane `02691156`, car `02958343`, chair
`03001627` — and is the widest random-access source found. No seekable mirror
of all 13 categories exists, so the full sweep has to run where the canonical
data already sits. The three it does have are usefully unalike: thin open
structure, solid convex mass, thin cluttered structure.

Ground-truth voxels are not a constraint either way: `ShapeNetVox32.tgz` is
only 22 MB, so **all 13 categories of ground truth are already local**. Only
renderings are ever the expensive half.

---

## 3. Backbone integration

Seven models attempted. The gate in every case was `backbones/bench.py`:
reproduce the published figure on held-out data. A wrong preprocessing chain or
weight mapping does not crash — it produces correctly-shaped garbage — so
loading successfully proves nothing.

### Working

| Backbone | 1 view | 3 views | 5 views | Notes |
|---|---|---|---|---|
| **UMIFormer** | 0.638 | 0.732 | **0.748** | 5-view result lands within 0.005 of its own checkpoint's claimed `best_iou 0.7529` |
| **Pix2Vox-F** | 0.595 | 0.632 | 0.657 | Beats the 3D-R2N2 chair baseline (0.466/0.533/0.550) at every budget |
| **Pix2Vox-A** | 0.638 | 0.665 | 0.667 | Works, but dominated — see §6 |
| **OccNet** | 0.345 | 0.374 | 0.385 | After the box-size fix in §4 |
| **TripoSR** | 0.247 | flat | flat | Loads and runs; not competitive — see below |

### Failed

**3D-R2N2 — port does not reproduce.** The released weights are a Theano
parameter dump (215,798,999 bytes, 63 arrays) with no PyTorch model, so the
network was reimplemented. The mapping is provably correct: all 63 arrays
consumed, every shape matched, parameter count equal to the checkpoint's
35,968,706 exactly. It still scores **IoU 0.031 against a published 0.466**. The
bug is in the forward pass — pooling semantics, residual wiring, or the LeakyReLU
slope. Left unregistered so it cannot corrupt the reward.

**pixelNeRF — camera convention unresolved.** Strict-loads all 278 tensors. But
it consumes DVR-preprocessed `cameras.npz`, which the Choy renderings do not
ship, so cameras were reconstructed from `rendering_metadata.txt`. That
reconstruction was tested *independently of the network* by projecting known
ground-truth voxels through the candidate cameras and measuring what fraction
lands inside the rendered silhouette — a correct convention scores ~0.95.

| Sweep | Combinations | Best agreement |
|---|---|---|
| Axis order, flips, grid extent | 288 | 0.52 |
| + azimuth sign/offset, elevation sign, up-axis | 1,536 | **0.55** |
| Using binvox's own `translate`/`scale` header | 768 | 0.36 (worse) |

Equivalent rotations tied at that ceiling rather than one convention standing
out, which says the renderer applies a per-model normalisation the metadata does
not record. The fix is the DVR/NMR release (~36 GB), which ships exact cameras —
but NMR is 64×64 from a different renderer, so adopting it would degrade the
voxel backbones trained on Choy's 137×137.

---

## 4. Calibrations

Three constants could not be derived and had to be measured. Each fails
silently if wrong — producing a correctly-shaped volume that scores badly — so
each was swept rather than assumed.

**Axis convention (all backbones).** All 48 permutation/flip combinations
scored against real ground truth. Every voxel backbone came back at identity
with a sharp margin (Pix2Vox-A: 0.836 at identity vs 0.314 for the next
family), confirming the convention rather than assuming it. TripoSR needed
`perm=(0,2,1), flip=(F,F,T)`, stable across every extent swept.

**OccNet query-grid extent: 1.1 → 0.96.** im2mesh's mesh generation uses
`box_size = 1 + padding = 1.1`, where the padding gives marching cubes room
around the surface. Copying that for direct grid scoring shrinks the object
inside a tight 32³ grid. Sweeping shows a clean peak at 0.96. Worth **+0.02 to
+0.03 IoU** — a real fix, though an earlier claim in this project that it
"doubled" OccNet's score was wrong, arising from comparing two different
measurement methods.

**TripoSR grid half-extent: 0.87 → 0.35.** `renderer.cfg.radius = 0.87` bounds
ray marching, not the object. Using it put the chair in the middle ~8 voxels of
the grid: **IoU 0.062, with 2% of the grid occupied against 28% ground truth.**
Sweeping gives a clean peak at 0.35 (IoU 0.376 on 6 models). Worth **+0.31 IoU**
— the largest single correctness fix in the project.

**Compute cost (measured wall-clock, CPU, 5 views).** The initially guessed
values were wrong in ordering, not just magnitude:

| Backbone | Measured | Normalised |
|---|---|---|
| OccNet | 0.35 s | 0.00 |
| Pix2Vox-F | 0.51 s | 0.17 |
| UMIFormer | 1.28 s | 0.94 |
| Pix2Vox-A | 1.34 s | 1.00 |

OccNet had been assumed mid-cost; it is in fact the cheapest.

---

## 5. Training run

Restored the model-selection head: episode horizon `H = B + 1`, two actor heads
sharing one trunk and one action space, routed by an `is_model_step` observation
flag, with the rollout buffer tracking which head produced each action.

Trained on Kaggle (T4, GPU backbones, `--dummy-vec`), **30,274 episodes in ~5
hours** at 0.58 s/episode.

| episodes | reward | std | entropy | phase |
|---|---|---|---|---|
| 1,704 | 0.240 | 0.219 | 2.640 | 1 |
| 5,800 | 0.312 | 0.115 | 2.470 | 1 |
| 7,504 | 0.312 | 0.115 | **2.386** | 1 |
| 10,907 | 0.311 | 0.137 | 2.472 | 2 |
| 23,501 | 0.281 | 0.208 | 2.456 | 2 |

Phase 1 learned: reward rose 0.240 → 0.312, entropy fell 2.64 → 2.39, both
flattening by ~episode 6,000. The phase-2 drop is the budget mix, not
regression — variable budgets {3,5,8} arrive at episode 10,000 and B=8 carries a
larger `−λ·B` penalty, so mean reward falls and variance nearly doubles.

**Entropy plateaued at ~2.46**, i.e. `exp(2.46) ≈ 12` effective choices out of
24. The report claims decay to 1.45; this run levelled off well above that.

---

## 6. Evaluation (`evaluate.py`)

### View selection: real but marginal

Policy-chosen views vs random views, **matched budget and matched backbone**, so
the only difference is which views were picked:

| budget | policy | random | delta |
|---|---|---|---|
| 3 | 0.7767 | 0.7711 | **+0.0056** |
| 5 | 0.7903 | 0.7783 | **+0.0120** |
| 8 | 0.7913 | 0.7930 | −0.0017 |

Small positive gain at 3 and 5 views, none at 8. Consistent with the entropy
plateau: once coverage saturates, *which* views were chosen stops mattering.
The NBV component is worth about one IoU point, not the headline the report
implies.

### Backbone selection: collapsed

**100% UMIFormer at every budget.** Policy IoU equals UMIFormer-alone IoU to
four decimals (0.7767 / 0.7903 / 0.7913). As configured, the
backbone-selection contribution is vacuous.

---

## 7. Why it collapsed — λ was calibrated on the wrong distribution

`cost_lambda = 0.09` was set from **test-split** quality gaps. Training happens
on the **train split**, where these backbones were pretrained and therefore
memorise, so the gaps are far wider and the dear backbone wins every episode.

Measured train-split IoU (25 chair models — the distribution the agent actually
optimises against):

| budget | Pix2Vox-F | OccNet | UMIFormer | gap (UMI − F) | crossover λ |
|---|---|---|---|---|---|
| 1 | 0.574 | 0.221 | 0.641 | **0.067** | **0.079** |
| 3 | 0.620 | 0.217 | 0.773 | 0.153 | 0.181 |
| 5 | 0.638 | 0.234 | 0.792 | 0.154 | 0.182 |

Two things follow:

1. **The gap is flat across {3,5,8}** (0.181 vs 0.182). No λ can split a
   decision inside the budget range that was trained on — raising λ flips every
   budget at once.
2. **At B=1 the gap is less than half.** Pix2Vox-F is relatively much stronger
   when starved of views. Including small budgets makes λ ∈ (0.079, 0.181) a
   live window.

**OccNet is dead at any sensible λ**: it needs λ > 0.45 to beat UMIFormer and
λ > 3.9 to beat Pix2Vox-F, at which point the cost term (−0.45 × 8 = −3.6 at
B=8) swamps IoU entirely.

---

## 8. Methodological findings

These generalise beyond this project.

**Benchmark on the test split.** The first evaluation looked excellent —
Pix2Vox-A at 0.836 IoU. Checking split membership showed **16 of 17 models were
from the training split**. The backbones were pretrained on exactly those, so
the number measured memorisation. Test-split value: 0.67. The larger model
inflated far more than the smaller one (0.836 vs 0.524), which is the signature.

**Voxel-family models will always win a voxel-IoU benchmark.** Every other
representation pays a conversion penalty:

- **Implicit** (OccNet, DVR) — trained against watertight-mesh occupancy, scored
  against binvox solid voxelisation. This is most of the gap between OccNet's
  published 0.571 and the 0.385 measured here.
- **Neural rendering** (pixelNeRF, TripoSR) — density thresholded into
  occupancy, with no principled setting for the threshold.
- **Voxel** (Pix2Vox, UMIFormer) — trained on exactly ShapeNetVox32 from exactly
  these renderings. No penalty at all.

Tested with two independent non-voxel models, both landing 0.36–0.50 IoU
behind. A three-family comparison scored purely by voxel IoU is structurally
biased, and a reward-maximising agent will never select the handicapped
families.

**A dominated backbone is a dead action.** Before adding a backbone, check
whether it wins the reward *somewhere*. Pairing matters: 3D-R2N2 is dominated
by Pix2Vox-F at equal cost, but becomes competitive against Pix2Vox-A alone.

---

## 9. The fix: a λ-independent crossover

The published UMIFormer / UMIFormer+ numbers cross over at four views:

| views | 1 | 2 | 3 | **4** | 5 | 8 | 12 | 20 |
|---|---|---|---|---|---|---|---|---|
| UMIFormer | **0.680** | **0.738** | **0.752** | 0.757 | 0.761 | 0.766 | 0.768 | 0.770 |
| UMIFormer+ | 0.567 | 0.712 | 0.745 | **0.759** | **0.768** | **0.779** | **0.784** | **0.789** |

*(published, ShapeNet mean IoU)*

The `+` variant is trained under a many-view regime, buying large-budget
accuracy at the cost of small-budget accuracy. Because the two share an
architecture they also share a compute cost, so **the choice between them is
pure quality and independent of λ** — it stays live even at λ = 0, and the
crossover sits inside the {3,5,8} training range. GARNet/GARNet+ show the same
pattern, confirming it is systematic rather than an artifact.

This matters because it answers the fair criticism of a cost-driven split: the
decision exists in the models, not in a reward parameter tuned until the answer
looks interesting.

### Confirmed on our data

Measured here, chair, test split, 20 models, best threshold over {0.2,0.3,0.4,0.5}:

| views | 1 | 2 | 3 | 5 | 8 |
|---|---|---|---|---|---|
| UMIFormer | **0.6383** | **0.6971** | **0.7320** | 0.7480 | 0.7536 |
| UMIFormer+ | 0.5581 | 0.6932 | 0.7299 | **0.7560** | **0.7702** |
| margin | 0.0802 | 0.0039 | 0.0021 | 0.0080 | 0.0166 |

**The crossover is real**, and it lands between 3 and 5 views rather than the
published 4 — consistent, since our chair-only test split is not the 13-category
mean the published table reports.

Two caveats that matter for training:

* The margin at 2 and 3 views (0.0039, 0.0021) is **inside noise** for 20
  models. Only B=1 (0.080) and B=8 (0.017) are decisive. So the learnable
  signal is at the ends of the budget range, not across it, which is the
  argument for including **B=1** in the training budgets — it is where the
  choice is least ambiguous.
* Both sit far above Pix2Vox-F (0.595 / 0.632 / 0.657) and OccNet (0.345 /
  0.374 / 0.385) at every budget, so on chair those two are dominated whatever
  λ is. Whether they win on other categories is exactly what §10's open item 1
  asks, and is the sole remaining case for keeping them registered.

---

## 10. Multi-category measurement

Everything above was chair. The 3-class mirror (§2) supplies airplane and car,
so 60 test models over three categories were scored with `category_bench.py` at
budgets {1,3,5,8}. **The corrected chair column reproduces `backbones.bench`
to four decimals**, which is what says the two scoring paths agree.

Mean IoU, best single threshold per cell over {0.2,0.3,0.4,0.5}, 20 test models
per category:

| budget | category | pix2vox_f | occnet | umiformer | triposr | umiformer+ |
|---|---|---|---|---|---|---|
| 1 | aeroplane | 0.5807 | 0.1264 | **0.6547** | 0.0586 | 0.5588 |
| 1 | car | **0.8848** | 0.3048 | 0.8762 | 0.1538 | 0.7984 |
| 1 | chair | 0.5948 | 0.3449 | **0.6383** | 0.2471 | 0.5581 |
| 3 | aeroplane | 0.6208 | 0.1590 | **0.7146** | 0.0586 | 0.6862 |
| 3 | car | 0.9007 | 0.2636 | **0.9131** | 0.1538 | 0.9090 |
| 3 | chair | 0.6321 | 0.3737 | **0.7320** | 0.2471 | 0.7299 |
| 5 | aeroplane | 0.6338 | 0.1616 | 0.7181 | 0.0586 | **0.7369** |
| 5 | car | 0.9057 | 0.2722 | 0.9144 | 0.1538 | **0.9217** |
| 5 | chair | 0.6567 | 0.3852 | 0.7480 | 0.2471 | **0.7560** |
| 8 | aeroplane | 0.6346 | 0.1663 | 0.7275 | 0.0586 | **0.7489** |
| 8 | car | 0.9064 | 0.2708 | 0.9187 | 0.1538 | **0.9226** |
| 8 | chair | 0.6605 | 0.3746 | 0.7536 | 0.2471 | **0.7702** |

### Three findings

**1. The crossover is universal, not a chair artifact.** All three categories
flip UMIFormer to UMIFormer+ between 3 and 5 views, at the same place. This was
the open question from §9 and it resolves in favour of the pair.

**2. Car at one view flips to Pix2Vox-F** (0.8848 vs 0.8762). It is the only
category-dependent choice found, and the only thing keeping Pix2Vox-F from
being a dead action -- on chair alone it is dominated at every budget. The
margin is 0.0086, thin enough to want the 13-category run before leaning on it.

**3. OccNet and TripoSR win nothing, in any category, at any budget.** Means of
0.271 and 0.153 against UMIFormer's 0.800. This is the direct opposite of the
argument that kept OccNet: it was retained on the theory that it would do
better on complex shapes and worse on simple ones, and it is instead uniformly
last but one. TripoSR is worse still and, being single-view, is flat across
budgets by construction.

A reward-maximising policy can never select either. They are not weak actions,
they are **unreachable** ones, and they cost two indices in the model head's
output distribution.

### Methodological note: the oracle-threshold trap

The first run of this sweep scored each model at its own best threshold, which
inflated every number by roughly 0.015 -- chair read 0.6539 instead of 0.6383.
That is small in absolute terms and larger than several margins the conclusions
rest on (UMIFormer vs UMIFormer+ at 3 views is 0.0021). A per-model threshold
is also an oracle the policy would not have at inference. Fixed to one
threshold per cell, matching `backbones.bench` and the published protocol.

The general lesson, consistent with §8: **at these margins the measurement
protocol is a bigger effect than the thing being measured.** Two numbers are
only comparable if they were produced by the same protocol on the same split.

---

## 11. Registry and lambda, settled

**Registry trimmed to three**, confirmed by the user:
`[Pix2VoxF, UMIFormer, UMIFormerPlus]`. OccNet and TripoSR are dropped from the
action space on the §10 evidence -- zero wins in twelve category-budget cells.
Their modules stay on disk with their calibrations, reachable via
`load_backbones(..., include_unregistered=True)` and `category_bench --all`, so
the 13-category sweep can still re-test the decision. Training never sees them.

This costs the "three families" framing: the surviving set is two voxel CNNs and
a transformer pair, with no NeRF or implicit member. That is the honest outcome
of measuring rather than assuming -- the NeRF (TripoSR) and implicit (OccNet)
members were the two that could not compete on this data.

**`cost_lambda = 0.07`** (was 0.09), derived from the §10 table rather than from
chair alone. Per-cell crossover values:

| budget | aeroplane | car | chair |
|---|---|---|---|
| 1 | 0.0873 | *(Pix2Vox-F wins outright)* | 0.0513 |
| 3 | 0.1106 | 0.0146 | 0.1178 |
| 5 | 0.1216 | 0.0189 | 0.1171 |
| 8 | 0.1348 | 0.0191 | 0.1294 |

The live window is (0.0146, 0.1348), but width is the wrong criterion. What
matters is the sub-range where the best backbone depends on the **object** and
not only the budget, since that is exactly what a learned policy can exploit and
a lookup table cannot. At 0.07 the one-view decision splits by category:

        B=1   aeroplane umiformer   car pix2vox_f   chair pix2vox_f
        B=3   aeroplane umiformer   car pix2vox_f   chair umiformer
        B=5   aeroplane umi+        car pix2vox_f   chair umi+
        B=8   aeroplane umi+        car pix2vox_f   chair umi+

0.07 is also the most robust point in that region, sitting 0.017 from the
nearest crossover. 0.05 is 0.0013 away from one and 0.09 is 0.0027 away -- both
inside the noise established in §10, so either would have made the policy's
target flip between measurements.

Verified after the change: all 10 `env/tests/test_env.py` tests pass, and
`training/tests/test_ppo_loop.py` still shows **both** actor heads receiving
gradient (view_head 1.02e-03, model_head 4.10e-04).

---

## 12. The first full run, and why it did not learn

The 3-backbone policy trained 30,274 episodes on Kaggle. It did not learn.

| episodes | mean R | var(R) | V-loss | explained variance |
|---|---|---|---|---|
| 1,704 | 0.4286 | 0.0173 | 0.0231 | -0.34 |
| 5,800 | 0.4477 | 0.0110 | 0.0035 | **0.68** |
| 7,504 | 0.4468 | 0.0119 | 0.0030 | **0.74** |
| 10,907 | 0.4434 | 0.0144 | 0.0291 | -1.02 |
| 12,201 | 0.4175 | 0.0263 | 0.0248 | **0.06** |
| 20,600 | 0.4195 | 0.0261 | 0.0250 | **0.04** |
| 23,501 | 0.4186 | 0.0261 | 0.0235 | **0.10** |

Explained variance is `1 - V_loss/Var(R)`: the share of the return the critic
can predict. Mean reward never improved, and final entropy was 2.525 against a
ceiling of 2.850 for a uniform policy over 24 views plus 3 backbones -- **89% of
maximum after 30k episodes**. The policy stayed very close to random.

### Cause: the budget was not observable

`phase1_episodes = 10_000`. Explained variance is 0.74 at episode 7,504 and has
collapsed by 10,907 -- exactly the phase boundary, where the budget stops being
fixed at 5 and starts being sampled from {3,5,8}.

The observation was `coverage_grid`, `image_features`, `view_mask`,
`is_model_step`. None of them carries the budget, and `view_mask` is all zeros
at step 0 whatever the budget, so **a B=3 episode and a B=8 episode were
literally the same observation**. The terminal reward contains `-lambda*B`,
which is -0.21, -0.35 or -0.56 at lambda = 0.07.

The arithmetic confirms it. Taking train-split IoU of roughly 0.90/0.93/0.95 at
B = 3/5/8, the rewards are +0.69/+0.58/+0.39 and their variance is **0.01536**.
The observed variance jump across the phase boundary was 0.02610 - 0.01185 =
**0.01425**. The budget term accounts for essentially the entire increase.

So in phase 2 roughly 55% of reward variance was unpredictable-in-principle.
The advantage `A = R - V(s)` was therefore dominated by *which budget was drawn*
rather than *whether the action was good*, and the policy gradient was mostly
noise. That is the whole explanation for the flat reward and the high entropy;
no amount of further training would have fixed it.

### Fix

The observation gains `budget`: `[B / n_views, (B - views_spent) / n_views]`.
Two scalars, concatenated straight into the trunk, which widens from 576 to 578.
They go in raw rather than through an encoder because the critic needs them
linearly -- it has to subtract `lambda*B`.

This also fixes a second, quieter problem: the **view** head could not plan.
Knowing whether three or eight views remain should change which view is worth
taking next, and that information was simply absent. (The *model* head could
in principle have counted the ones in `view_mask`, since all B views are spent
by the time it acts, but it had to learn to count rather than being told.)

`env/tests/test_env.py::test_budget_observable` asserts that B=3 and B=8 give
different observations at step 0, and that the remaining count decreases. It is
there because this failure is silent: the run completed cleanly, wrote
checkpoints, and reported plausible-looking rewards throughout.

**Consequence:** the observation shape changed, so `ckpt_kaggle.pt`'s 30,274
episodes cannot be resumed. It is kept as the record of this diagnosis.

---

## 13. Correctness audit before retraining

Three defects, each of which alone would prevent the view head from learning.

### 13.1 GAE read the wrong terminal flag (the worst one)

`compute_gae` used `dones[t+1]` as the "did the episode end here" flag for step
`t`. But `dones[t]` is what `vec_env.step()` returned for action `t` -- "the
episode ended after this action" -- so the correct flag is `dones[t]`. The
`t == n_steps-1` branch already used `dones[t]`, which is what the general case
should have done.

On a `B+1` horizon the off-by-one puts the episode boundary one step early:

* Step `B`, the **last view step**, was treated as terminal, so the terminal
  reward never propagated back to *any* view step.
* Step `B+1`, the **model step** and the real terminal, was treated as
  non-terminal, so it bootstrapped `V` from the first state of the *next*
  episode.

A worked example with `V = 0.5`, terminal reward 1.0, `gamma = 0.99`:

| | t=0 | t=1 | t=2 | t=3 | t=4 | t=5 (terminal) |
|---|---|---|---|---|---|---|
| before | -0.410 | -0.430 | -0.452 | -0.475 | -0.500 | +0.990 |
| after | +0.346 | +0.373 | +0.402 | +0.433 | +0.465 | +0.500 |

The terminal return was 1.4903 against a true 1.0, and **every view step had a
negative advantage in an episode that earned +1.0**. The view head was training
on a constant negative signal carrying no information about which view it chose.

This is the primary explanation for the flat reward and 89% entropy in §12 --
larger than the budget defect, which only degraded the critic.

### 13.2 The coverage grid was not a projection

The report says depth and silhouette maps are *"projected onto this grid"*.
`CoverageGrid.update()` did not take a camera pose at all: it wrote
`grid[x, y, z]` in raw image-plane coordinates, so two of the three axes rotated
with the camera and the grid accumulated incomparable frames.

Fixing it needed two things together, and either alone is useless:

1. **Metric depth.** `_depth_from_voxels` normalised each view against its own
   min/max, so the same world distance meant something different per view. Now
   scaled to the fixed `[-1, 1]` world range the projection already assumes for
   x and y.
2. **The exact inverse rotation.** A hand-rolled camera basis is not enough --
   the depth map was made with `cam = (Rx @ Ry) @ world`, azimuth in the XZ
   plane, and `depth_map[py, px]`, so the inverse must use the same rotations in
   the same order and undo the `(row, col) = (cam_y, cam_x)` transpose.

Measured on 5 chair models, precision of back-projected points against
1-voxel-dilated ground truth:

| | image-plane | camera-aware | chance |
|---|---|---|---|
| mean precision | 0.4803 | **0.8261** | 0.3731 |

A first attempt using a hand-built basis scored 0.4842 -- indistinguishable from
the broken version. That near-miss is the point: a wrong projection still fills
the grid and still grows with each view, so nothing looks wrong.

### 13.3 The budget was invisible

See §12.

### Deviations from the report that remain

* **No Chamfer term.** The report's reward is
  `alpha*IoU + (1-alpha)*exp(-Chamfer) - lambda*cost` with `alpha = 0.5`; the
  implementation uses `IoU - lambda*cost`. Adding it needs a distance transform
  per episode on the reward path.
* **`cost_lambda` is 0.07, not the report's 0.05** -- derived by measurement in
  §11 rather than assumed.

### Regression tests added

`test_budget_observable` (B=3 and B=8 must differ at step 0) and
`test_coverage_is_geometric` (camera-aware projection must beat the image-plane
stencil by 1.5x). Both failures were silent: training completed, wrote
checkpoints, and logged plausible rewards throughout.

---

## 14. Should the Chamfer term be added? (measured)

The report specifies `r = a*IoU + (1-a)*exp(-Chamfer) - lambda*cost`, `a = 0.5`;
the implementation uses `IoU - lambda*cost`. Measured on 288 predictions (3
categories x 8 test models x 4 budgets x 3 backbones), Chamfer computed as the
symmetric mean nearest-neighbour distance via a Euclidean distance transform.

### The report does not state units, and it changes everything

| | min | median | max | spread |
|---|---|---|---|---|
| exp(-CD), voxel units | 0.2173 | 0.9066 | 0.9910 | **0.7738** |
| exp(-CD), world units | 0.9090 | 0.9939 | 0.9994 | **0.0904** |
| IoU, for comparison | - | - | - | 0.7424 |

One voxel is 2/32 = 0.0625 world units, so the choice moves the term between
"comparable in scale to IoU" and "almost constant". The report is unfalsifiable
as written; both were tested.

### It is 98.7% redundant with IoU

* `corr(IoU, exp(-CD))` = **+0.967** (voxel units), +0.914 (world units)
* Per-sample Spearman rank correlation = **+0.9936**, so shared rank variance is
  **98.7%** and independent information is **1.3%**.

Ranking backbones by `-Chamfer` alone rather than IoU alone disagrees in only
**2 of 12** category-budget cells -- and both are B=3 car and B=3 chair, exactly
the cells where s10 already measured the IoU margin as 0.0041 and 0.0021, inside
noise. The disagreement is noise, not signal.

### Effect on the actual decision

Optimal backbone per cell at lambda = 0.07:

| reward | decisions changed vs IoU-only |
|---|---|
| `0.5*IoU + 0.5*exp(-CD)`, voxel units | **0 / 12** |
| `0.5*IoU + 0.5*exp(-CD)`, world units | 4 / 12 |

The world-unit changes are **not** new information. Halving the weight on IoU
doubles lambda's effective strength, so the live window shrinks from
(0.0146, 0.1348) to (0.0080, 0.0735) and every change is toward the cheap
backbone. lambda = 0.07 then sits at 95% of the way to the window's top edge --
one measurement away from the collapse-to-cheapest failure of s12.

Measured separation ratios confirm the mechanism: `dE/dIoU` has median **1.021**
in voxel units (the term tracks IoU almost exactly) and **0.090** in world units
(it barely separates anything).

### Cost is not the argument

6.8 ms per episode, 7.8x the IoU computation -- but negligible beside the ~1.28 s
backbone forward pass that dominates the reward path. Roughly 0.5% overhead.

### Conclusion

**Do not add it for accuracy reasons.** It carries 1.3% independent rank
information, changes 0 of 12 decisions in the units where it is well-scaled, and
in the other units its only effect is a rescaling that pushes lambda toward
collapse.

**If report fidelity is wanted, voxel units are free**: 0/12 decisions change,
the live window is essentially unchanged at (0.0148, 0.1362) against
(0.0146, 0.1348), and lambda = 0.07 stays valid. That buys literal conformance
to the stated reward at ~0.5% compute and no behavioural change. World units
must not be used without re-deriving lambda to about 0.035.

**When this would change:** Tatarchenko et al. (CVPR 2019) show IoU and Chamfer
rank reconstruction methods differently, which is the real argument for the term.
That result concerns meshes and point clouds at much higher resolution. At 32**3
the grid is too coarse for surface distance to express anything IoU does not --
hence rho = 0.9936 here. Revisit if the output moves to 128**3 or to meshes.

---

## 15. Second full run: critic fixed, policy still at chance

Retrained on the corrected code (`shared.0.weight` is (512, 578), confirming the
budget-aware observation). 30,274 episodes.

### The critic fix worked exactly as predicted

Explained variance, `1 - V_loss/Var(R)`:

| episodes | run 1 | run 2 |
|---|---|---|
| 5,800 (phase 1) | 0.68 | 0.50 |
| 7,504 (phase 1) | 0.74 | 0.25 |
| 12,201 (phase 2) | **0.06** | **0.76** |
| 20,600 (phase 2) | **0.04** | **0.71** |
| 23,501 (phase 2) | **0.10** | **0.76** |

Phase 2 no longer collapses. The budget term is now predictable, which is what
s12 said it would take.

### The policy is still not learning

| | run 1 | run 2 | paper claim |
|---|---|---|---|
| final entropy | 2.525 | 2.448 | 1.45 |
| % of uniform ceiling | 93% | **90%** | 53% |
| mean reward, first -> last | 0.448 -> 0.419 | 0.441 -> 0.416 | rises |

**Correction to an earlier figure in this log:** the uniform-entropy ceiling is
**2.728**, not 2.850. The action mask shrinks the legal set as views are taken,
so step `t` has `ln(24 - t)` available, not `ln(24)`. Averaged over budgets
{3,5,8} with one 3-way model step per episode that gives 2.728. The policy is
therefore even closer to random than first stated.

### There IS a view-selection signal, so this is not an inherent limit

40 random 3-view subsets scored per chair, 6 test models, UMIFormer:

| quantity | IoU |
|---|---|
| within-object std (what view choice controls) | 0.0266 |
| between-object std (object difficulty, uncontrollable) | 0.0621 |
| ratio within/between | **0.429** |
| perfect planner over average | **+0.0402** |
| worst subset under average | -0.0726 |

Best-to-worst span is **0.113 IoU**, fourteen times the UMIFormer/UMIFormer+ gap
(0.0080) that the model head is expected to exploit, and signal-to-noise of 0.43
is workable. So a near-uniform view policy is **not** close to optimal, and high
entropy is not the correct answer -- the policy is leaving roughly 0.04 IoU on
the table per episode.

### Leading hypothesis: the first state is uninformative

At the first view step the state is all zeros -- empty coverage grid, zero image
features, zero view mask -- with only the budget set. So `V(s_0)` is identical
for every object, and the advantage at step 0 carries the entire
between-object variance (0.0621) on top of a signal of 0.0266.

This is not a coding defect; it is inherent to planning from zero observations.
The usual remedy in the NBV literature is to seed each episode with one free
initial view so every decision is conditioned on something. **Untested.**

Other candidates not yet ruled out:

* Advantages are normalised **globally** across view steps and the model step
  together, though the measured magnitudes after the GAE fix are comparable
  (0.35-0.47 vs 0.50), so this looks unlikely to dominate.
* `entropy_coef_view = 0.01` is applied to the mean entropy over both heads,
  mixing a 24-way and a 3-way distribution under one coefficient. Matches the
  report's stated value, so it is not a deviation, but the mixing is crude.

### Honest status

Defensible: the backbone-selection contribution, the 3-category measurement, the
lambda derivation, and the integration correctness gate. **Not demonstrated: the
next-best-view half.** The view policy performs at chance, and the measurement
above shows that is a failure to learn rather than an absence of anything to
learn.

---

## 16. Fixing view selection: dense credit from the literature

### What the literature does differently

[GenNBV (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Chen_GenNBV_Generalizable_Next-Best-View_Policy_for_Active_3D_Reconstruction_CVPR_2024_paper.pdf)
trains an NBV policy with PPO -- the same algorithm used here -- but on a **dense
per-step reward**, the change in coverage ratio: `r_{t+1} = CR_{t+1} - CR_t`. Not
terminal-only. This is the norm across the NBV literature, not an outlier.

Our reward was terminal-only, so all B view steps shared one undifferentiated
number. Even with correct GAE, PPO gets the same advantage for the view that
revealed a new surface and the view that duplicated an existing one.

### Why densifying is safe here

Ng, Harada & Russell (1999) proved that a shaping term of the form
`F(s, a, s') = gamma*Phi(s') - Phi(s)`, with `Phi(terminal) = 0`, leaves the set
of optimal policies **unchanged for any potential function Phi**. So this is not
a reward hack that risks optimising a proxy: the objective is provably identical,
only the credit assignment changes.

Implemented with `Phi(s)` = fraction of the 32**3 coverage grid occupied. Free to
compute. It is only meaningful because s13.2 made the coverage grid a real
back-projection (precision 0.826 against 0.373 chance); against the old
image-plane stencil this potential would have been noise.

### The terminal term is not optional

The guarantee needs the shaping terms to sum to something the agent cannot
influence. Under discounting they telescope to exactly `-Phi(s_0)`:

    sum_t gamma^t (gamma*Phi(s_{t+1}) - Phi(s_t)) = gamma^T*Phi(s_T) - Phi(s_0)

Drop the terminal `-Phi(s_B)` and they sum to `Phi(s_B) - Phi(s_0)` instead,
which the agent **can** influence -- silently changing the objective to "maximise
final coverage", a correlate of IoU rather than IoU. Measured, that mutation
introduces an action-dependent bias with spread 0.0028, comparable to the margins
the backbone decision turns on.

`env/tests/test_shaping.py::test_shaping_terms_telescope` measures
shaped-minus-sparse discounted return across four different action sequences and
requires a spread below 1e-9. With the terminal term the spread is **exactly
0.0**; mutation-tested, removing it produces 0.0028 and the test fails with the
right diagnostic.

**Note on the arithmetic:** the telescoping is exact only for the *discounted*
return. An undiscounted sum leaves a `(gamma-1)*sum(Phi)` residual, about 1e-3 at
gamma = 0.99. The first version of this test summed undiscounted and failed for
that reason -- the arithmetic, not the shaping.

### Second change: seed one initial view

At the first view step the state was all zeros, so `V(s_0)` was identical for
every object and that advantage carried the whole between-object spread (0.0621)
on top of a view-choice signal of 0.0266. Each episode now begins with one random
view already folded in, so every decision the policy makes is conditioned on
something. Standard practice in the NBV literature.

The seeded view does not consume budget (`step_count` is reset after it), so
B=3 still means three *chosen* views;
`test_seeded_view_does_not_consume_budget` asserts the episode is B+1 steps and
reconstructs B+1 views.

**Side effect worth knowing:** the seed view IS charged in the compute cost, so
the term is now `lambda*(B + 1 + backbone_cost)`. Mean reward therefore drops by
about 0.07 against earlier runs. It does **not** affect the backbone decision or
the lambda calibration, because the `+1` is constant within an episode and
cancels when comparing backbones at fixed B.

### Both are switchable

`Config.shaping_coef = 0.0` restores the sparse reward exactly (asserted by
`test_sparse_still_available`), and `seed_initial_view = False` restores the
cold start. So the previous behaviour remains reachable for an ablation, which is
what a paper would want to report.

### Answering the coupling question

All three heads currently share one trunk (`Linear(578, 512)`) and the encoders,
so a view-policy change invalidates the backbone head too -- gradients from
`view_head` reshape the representation `model_head` reads.

**Measured, that coupling is unnecessary.** Across 12 different view subsets per
object at B=3, the best backbone did not flip once (0/4 models), while it did
vary *between* objects (Pix2Vox-F for two, UMIFormer for the other two). So the
backbone decision is a function of (object, budget) and is stable against which
views were chosen. Mean reward spread across backbones was 0.051-0.076, the same
order as the view-choice signal, so both heads matter.

That means the model head could be given its own small encoder, or trained
supervised from the s10 IoU table, and then a view-policy change would **not**
require retraining it. Not yet implemented.

---

## 17. Full audit after run 3: what is sound, what is broken, what is conceptual

### Bugs found and fixed

| # | Defect | Consequence |
|---|---|---|
| 1 | `evaluate.py` sized the random control at `B` while the policy returned `B+1` views (seed + choices) | reported a spurious **+0.0225 "NBV helps"**; scored like-for-like the same policy is **-0.0007**, i.e. no better than random |
| 2 | One entropy coefficient over a pooled mean of a 24-way and a 3-way head | view head at 90% of uniform, backbone head at **15%** of its maximum -- opposite failures from one knob |
| 3 | Only the pooled entropy was logged | the backbone collapse was invisible for **three full runs** |
| 4 | No learning-rate annealing | constant 3e-4 for 30k episodes; standard in CleanRL and SB3 |
| 5 | `seed_initial_view` shifted budgets 3/5/8 -> 4/6/9 views | the UMIFormer crossover is between 3 and 4 views, so one backbone became correct nearly everywhere |
| 6 | CSV header changed without rollover | a resumed run would append 9-field rows under a 6-field header |

### What is sound (measured, not assumed)

* **GAE credit assignment.** View steps carry mean |r| of 0.0045 yet mean |A| of
  0.324 -- the terminal reward propagates back correctly.
* **Advantage balance.** Normalised |A| ratio model/view = **1.21**. Joint
  normalisation is not starving either head, so per-step-type normalisation is
  unnecessary.
* **Critic.** Explained variance reaches **0.88**.
* **Architecture.** One trunk, two heads, H = B+1, masking, and a reward that
  changes with the backbone choice (spread 0.079 on a fixed view set) -- the
  joint MDP the report specifies is genuinely implemented.

### Conceptual gap 1: coverage is not reconstruction quality

Potential-based shaping was added on GenNBV's precedent (per-step change in
coverage ratio). Measured on chair, 8 objects, every candidate next view:

| potential | mean rho with delta IoU | picks the best view |
|---|---|---|
| `filled` (fraction of grid occupied, shipping) | +0.305 | 1/8 |
| `recall` = coverage ∩ GT / GT | +0.207 | 0/8 |
| `iou_cov` = coverage vs GT IoU | +0.193 | 0/8 |

Chance is **1/8**. All three are at chance, and the GT-recall variant -- the
"obvious" improvement -- is **worse** than what shipped.

**Why.** GenNBV's coverage reward works because its reconstruction *is*
volumetric integration: coverage gain literally is reconstruction gain. Our
backbones are learned shape priors. UMIFormer infers unseen geometry from
training data, so revealing more surface does not monotonically raise IoU; what
matters is which views disambiguate the shape *for the network*. **A reward
design was imported from a setting whose reconstruction mechanism is different
in kind, without checking that the assumption transferred.**

Shaping is therefore policy-safe (PBRS guarantees that) but **inert**.

### Conceptual gap 2: view preferences are backbone-specific

If coverage will not serve as a potential, the aligned choice is reconstruction
quality itself, computed each step with the cheap backbone. That only works if a
cheap model's view preferences transfer to the expensive one:

    mean rho(delta IoU pix2vox_f, delta IoU umiformer_plus) = +0.171
    cheap model's best view IS dear model's best : 2/8   (chance 1.0/8)
    cheap model's best view in dear model's top 2: 2/8   (chance 2.0/8)

**They do not transfer.** (n = 8, so 2/8 against a 1/8 chance rate is not
significant; the honest reading is "no usable transfer".)

Two consequences:

1. Dense per-step credit cannot be bought cheaply. It needs the *same* backbone
   the reward uses, at roughly B expensive forwards per episode.
2. **It exposes an ordering assumption in the MDP.** The report's H = B+1 picks
   all B views first and the backbone last. If the best views depend on which
   backbone will score them, the view policy must choose while hedging over a
   decision it has not made yet. Reversing the order -- backbone first, then
   views conditioned on it -- would make view selection a well-posed problem,
   at the cost of departing from the report's stated MDP. Not attempted.

### Conceptual gap 3: the coverage grid is derived from ground truth

`_depth_from_voxels` synthesises each view's depth by projecting the GT
occupancy. This is defensible -- NBV systems assume a depth sensor and the
report specifies depth in the state -- but it is a *noiseless, idealised* sensor,
and the agent therefore observes geometry derived from the answer before
reconstructing it. It should be stated explicitly rather than left implicit.

### Conceptual gap 4: the signal is small next to the noise

Within-object spread from view choice is **0.0266 IoU**; between-object spread
is **0.0621**. The critic removes object difficulty only once it has seen a view,
and a perfect planner gains **+0.0402** per episode. That is a real but small
target under a terminal-only reward.

---

## 18. State at time of writing

**Registered and verified:** Pix2Vox-F, OccNet, UMIFormer, TripoSR.
**Added, verification pending:** UMIFormer+.
**Excluded, documented:** 3D-R2N2 (port fails), pixelNeRF (cameras), Pix2Vox-A
(dominated).

Trained policy: `ckpt_final.pt`, 30,274 episodes, both curriculum phases.

### Open

1. **Extend the sweep to all 13 categories.** Three is enough to kill the
   chair-only monoculture, not enough to settle whether OccNet wins anywhere.
   No seekable mirror carries more than three, so this runs on Kaggle where the
   canonical renderings are already attached — `kaggle_bench.py`, Step 5b.
2. **Decide the registry.** The measurement says three backbones each win
   somewhere — Pix2Vox-F (car at one view), UMIFormer (low budget), UMIFormer+
   (high budget) — and that OccNet and TripoSR win nowhere. Trimming to the
   three costs the "three families" framing (there would be no NeRF or implicit
   member) and buys a model head whose every action is reachable. This is a
   judgement call about what the paper claims, not a measurement, so it is
   flagged rather than taken.
3. **Retrain** with whatever registry survives (2), budgets including B=1,
   λ≈0.10. Note the current `ckpt_final.pt` model head is indexed against the
   five-backbone registry and will not transfer if the registry changes.
4. R2N2's forward pass and pixelNeRF's cameras remain unresolved.

### Measured but not yet acted on

The 10-view mirrors mean these numbers use 8 of 10 views at B=8, where training
assumes Choy's 24. Budget-ordering conclusions hold; absolute values at high
budget are slightly pessimistic against published 24-view figures.
