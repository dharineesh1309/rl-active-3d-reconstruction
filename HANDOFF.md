# Handoff — where this build stands

Written for whoever picks this up next, including a fresh assistant session with no
memory of the work. Everything below is verified unless marked otherwise.

Working tree: **`D:\rl_project\build\rl_pipeline`** (edited in place).
Read `build/rl_pipeline/README.md` first — it covers setup, the backbone table and
how to run everything. This file covers only *state and next actions*.

---

## What the goal was

The project report claims a PPO agent doing joint next-best-view planning **and**
reconstruction-backbone selection. The shipped code only did view selection —
`policy/view_policy.py` said outright that the model-selection head "has been fully
removed", and the report lists multi-backbone selection as future work. The task was
to build that missing contribution with real pretrained reconstructors.

## Status: the pipeline works end to end

`python train.py --smoke-test --dummy-vec --n-envs 2` runs both curriculum phases on
real ShapeNet chairs, with the agent choosing views **and** backbones, scored by real
IoU. `infer.py` runs a trained policy on a single object. All tests pass:

* `env/tests/test_env.py` — 10 tests incl. B+1 horizon, backbone masking, cost penalty
* `training/tests/test_ppo_loop.py` — full PPO loop, asserts **both** actor heads
  receive gradient (a head-routing bug would not crash, it would train the wrong head)
* `backbones/selftest.py`, `backbones/bench.py`

---

## IMMEDIATE NEXT STEP

**None blocking — the three-backbone build is complete and verified.**

UMIFormer finished downloading (717,174,122 bytes, exact), strict-loads
(219+185+30 tensors) and **passed the gate**: 0.638 / 0.732 / 0.748 IoU at
1/3/5 views on the chair test split, within 0.005 of its own checkpoint's
claimed `best_iou 0.7529`. One fix was needed — upstream hard-codes `.cuda()`
in the encoder's `grouping()`; it now builds those masks on the activations'
device.

Final set is `pix2vox_f` + `umiformer` + `umiformer_plus`,
`cost_lambda = 0.07` (see EXPERIMENTS.md §11 for the derivation).
All tests pass and `train.py --smoke-test` runs both phases selecting all three.

## READ FIRST (2026-09-25, third run pending)

Five defects found and fixed across two failed 30k-episode runs. Full working in
EXPERIMENTS.md sections 12-16.

| # | Defect | Effect if unfixed |
|---|---|---|
| 1 | GAE read `dones[t+1]` not `dones[t]` | terminal reward never reached ANY view step; every view step had negative advantage in a winning episode |
| 2 | view budget absent from the observation | critic could not predict `-lambda*B`; phase-2 explained variance 0.74 -> 0.06 |
| 3 | coverage grid was an image-plane stencil, not a projection | geometrically meaningless state; precision 0.48 vs 0.37 chance |
| 4 | sparse terminal reward only | all B view steps shared one undifferentiated signal |
| 5 | `evaluate.py`/`infer.py` did not mirror training | a working policy would have measured as broken |

Confirmed working after fix 2: phase-2 explained variance is now **0.71-0.76**.

Fixes 4 and 5 are new and UNVALIDATED at scale. Fix 4 is potential-based reward
shaping (Ng, Harada & Russell 1999) with `Phi` = coverage-grid occupancy, plus a
free seeded initial view. Policy invariance is verified numerically to exactly
0.0 spread, and mutation-tested.

### What to watch in the next run

1. **Entropy must fall.** The uniform ceiling is **2.728** (the action mask
   shrinks the legal set as views are taken, so step t has ln(24 - t)). Runs 1
   and 2 ended at 2.525 and 2.448, i.e. 93% and 90% of random. The report claims
   1.45. If it is still above ~2.3 by episode 15,000, view selection is still
   not learning.
2. **Explained variance** `1 - V_loss/Var(R)` must stay above 0 through the
   phase-2 boundary at episode 10,000. This is the number that exposed defect 2.
3. **Mean reward must rise.** It fell in both previous runs. Note it now sits
   about 0.07 lower in absolute terms because the seeded view is charged in the
   compute cost (`lambda*(B + 1 + backbone_cost)`); that offset is constant and
   does not affect the backbone decision.

### Known-good reference numbers

* A perfect view planner gains **+0.0402 IoU** over an average one (40 subsets,
  6 chairs, B=3). Within-object std 0.0266, between-object 0.0621. So there IS
  signal; a flat policy is a failure to learn, not an absence of anything.
* Best backbone does NOT depend on which views were chosen (0/4 models flipped
  across 12 subsets), but DOES vary by object. So the backbone head could be
  decoupled from the view policy -- not yet done.

### Ablation switches, for the paper

`Config.shaping_coef = 0.0` restores the sparse reward exactly;
`seed_initial_view = False` restores the cold start. Both previous behaviours
remain reachable.

## CURRENT STATUS (2026-09-25): backbone selection works, view selection does not

Second full run, 30,274 episodes on the corrected code.

**Fixed and confirmed:** the critic. Phase-2 explained variance went from
0.04-0.10 to **0.71-0.76**. The budget observation did what section 12 predicted.

**Still broken:** the view policy. Final entropy 2.448 against a uniform ceiling
of **2.728** -- 90% of random -- and mean reward fell (0.441 -> 0.416). Two runs,
same result.

**It is not a lack of signal.** Measured over 40 three-view subsets per chair:
within-object std 0.0266, between-object std 0.0621, and a perfect planner beats
an average one by **+0.0402 IoU** (best-to-worst span 0.113, i.e. 14x the
UMIFormer/UMIFormer+ gap). The policy is leaving real reward unclaimed.

**Leading hypothesis, UNTESTED:** at the first view step the state is all zeros
(empty coverage, zero features, zero mask), so `V(s_0)` is identical for every
object and that advantage carries the full between-object variance. The standard
NBV remedy is to seed each episode with one free initial view. Try this next.

**What is defensible for the paper today:** backbone selection, the 3-category
measurement, the lambda derivation, the integration gate. **The next-best-view
half is not demonstrated.**

## Correctness audit before the second run (2026-09-24)

Three defects found and fixed, each sufficient on its own to stop the view head
learning. Full working in EXPERIMENTS.md sections 12-13.

1. **GAE read `dones[t+1]` instead of `dones[t]`.** The worst one. On a B+1
   horizon this put the episode boundary a step early: the terminal reward never
   reached any view step (all five had NEGATIVE advantage in an episode earning
   +1.0), and the model step bootstrapped V from the next episode (terminal
   return 1.4903 against a true 1.0). The view head was training on a constant
   negative signal.
2. **The coverage grid was not a projection.** `update()` took no camera pose.
   Fixing it required BOTH metric depth (`_depth_from_voxels` normalised
   per-view, so the same distance meant different things in different views) and
   the exact inverse of `Rx @ Ry` including the (row,col)=(cam_y,cam_x)
   transpose. Precision against ground truth went 0.4803 -> 0.8261 (chance
   0.3731). A hand-rolled camera basis scored 0.4842, i.e. no better than broken.
3. **The budget was not observable.** See below.

**Chamfer term: deliberately NOT added.** The report specifies
`0.5*IoU + 0.5*exp(-Chamfer)`. Measured over 288 predictions it is 98.7%
rank-redundant with IoU (Spearman 0.9936) and changes 0 of 12 backbone decisions
in voxel units. In world units its only effect is to halve the IoU weight, which
doubles lambda's effective strength and pushes 0.07 to the edge of collapse. If
literal report conformance is ever needed, voxel units are free and lambda
stays 0.07; world units would need lambda ~= 0.035. EXPERIMENTS.md section 14.

## IMPORTANT: the first full run did not learn, and why

30,274 episodes trained on Kaggle with the 3-backbone registry. Mean reward
never improved and final entropy was 89% of maximum -- the policy stayed near
random.

**Cause: the view budget was not in the observation.** `view_mask` is all zeros
at step 0 whether B=3 or B=8, so those episodes were indistinguishable, while
the terminal reward carries `-lambda*B` (-0.21 / -0.35 / -0.56). The critic
could not predict it. Explained variance was 0.74 in phase 1 (fixed B=5) and
collapsed to 0.06 the moment phase 2 began sampling budgets at episode 10,000.
The budget term's variance (0.01536) accounts for the entire observed jump
(0.01425). See EXPERIMENTS.md section 12 for the full working.

**Fixed.** The observation now carries `budget` = `[B/n_views,
remaining/n_views]`; the policy trunk widens 576 -> 578.
`env/tests/test_env.py::test_budget_observable` guards it.

**This invalidates `ckpt_kaggle.pt`** (kept in `checkpoints/` as the record).
The next Kaggle run must start fresh -- re-upload `kaggle_code.zip`.

**Lesson worth keeping:** the run completed cleanly, wrote checkpoints, and
logged plausible rewards the whole way. Nothing crashed. Check *explained
variance* (`1 - V_loss/Var(R)`) and *entropy against its ceiling*, not just
whether training finished.

## Verified on Kaggle (2026-09-23)

A GPU smoke test ran end to end on the real data:

* 30,642 models, **24 views**, all 13 categories, `cuda`
* `Backbones: ['pix2vox_f', 'umiformer', 'umiformer_plus']`
* `cost_lambda = 0.07` confirmed from the rewards themselves, not the banner:
  `0.8355 - lambda*(5 + 0.09) = 0.4792` gives lambda = 0.0700.
* Both curriculum phases completed, checkpoint written.

**Gotcha this exposed:** an older dataset (`rl-project-dataset1`) was still
attached and the copy cell picked its `rl_pipeline` by path depth, so the first
smoke test silently ran the OLD registry and lambda 0.09 -- and completed
cleanly, which is what made it dangerous. The guard that would have caught it
was in the code that did not get copied. **Select the source tree by content,
not by path**: check for `backbones/umiformer_plus.py` and
`registered = [... UMIFormerPlus]` before copying. KAGGLE_STEPS Step 4 has the
cell that does this.

## Kaggle training is set up

`build/rl_pipeline/kaggle_train.py` drives training on Kaggle end to end, and
`build/package_for_kaggle.py` builds the two upload zips (`kaggle_code.zip`
~1 MB, `kaggle_weights.zip` ~908 MB). See README "Training on Kaggle".

What it handles, both verified locally:

* **Session cap** — `train.py --max-hours N` stops cleanly at the budget and
  writes `ckpt_resume.pt` instead of being killed mid-run.
* **Resume across sessions** — `CheckpointManager.load_latest()` now prefers
  `ckpt_resume.pt` over any other checkpoint. Without that it ranked purely by
  episode count, so the stale 30,595-episode `ckpt_final.pt` shadowed the run
  being continued and every resume died on a shape mismatch.
* **GPU backbones** — `--backbone-device cuda` (implies `--dummy-vec`, since
  CUDA tensors cannot be pickled into SubprocVecEnv workers). The ResNet-50
  state encoder moves with them. Guarded with a clear error if combined with
  SubprocVecEnv.

Still needed from you: upload the two zips plus ShapeNetRendering /
ShapeNetVox32 as Kaggle datasets, and enable a GPU accelerator.

## Next, in priority order

1. **Run the 13-category sweep on Kaggle** — `kaggle_bench.py`, KAGGLE_STEPS
   Step 5b. Three categories are now measured locally (airplane, car, chair;
   see EXPERIMENTS.md §10) and they say OccNet and TripoSR **win nothing, at
   any budget, in any category**. Thirteen would settle it. No seekable mirror
   carries more than three, so this cannot be done locally.
2. ~~Decide the registry~~ **DONE.** Trimmed to
   `[Pix2VoxF, UMIFormer, UMIFormerPlus]` and `cost_lambda = 0.07`. OccNet and
   TripoSR are off the action space but still benchmarkable via
   `category_bench --all`, so (1) can still overturn it. All tests pass.
   **This invalidates `ckpt_final.pt`'s model head** (indexed against five
   backbones); view-head weights can still be carried over with
   `--load-view-head-only`.
3. **Ship to the GPU box and run the real curriculum.** Nothing trains usefully
   on CPU here (`torch 2.12.1+cpu`). See README "Training on a GPU box".
4. **Re-derive `cost_lambda` on the target hardware** — the live window
   (0.053, 0.121) comes from CPU wall-clock, and relative costs will shift on a
   GPU. `backbones/bench.py` plus the cost measurement gives it.
5. Optional: debug the R2N2 port, or unblock pixelNeRF with DVR/NMR cameras.

## Decisions made, so they are not relitigated

* **Three backbones, each of which measurably wins somewhere.** Final set:
  **`pix2vox_f` + `umiformer` + `umiformer_plus`**, confirmed by the user.
  The original goal was three *families* (voxel / NeRF / implicit), but the NeRF
  member (TripoSR) and the implicit member (OccNet) won zero of twelve
  category-budget cells, so they were dropped in favour of a set where every
  action is reachable. The user had rejected "pix2vox a, b or something like
  that" — note UMIFormer/UMIFormer+ are a same-architecture pair, kept
  deliberately because identical cost makes their crossover independent of
  `cost_lambda`, which is the one decision that cannot be dismissed as a tuned
  reward parameter.
* **Pix2Vox-F, not -A.** Both were measured. UMIFormer strictly dominates -A
  (equal at 1 view, better at 3/5/8, and cheaper), so -A could never win and
  would be a dead action. -F is genuinely cheap (0.51s vs UMIFormer's 1.28s) and
  therefore wins at low budgets, which is what makes the head's choice real.
* **No ports, no retraining.** UMIFormer was chosen over 3D-R2N2 specifically because
  it is PyTorch-native, so the checkpoint loads directly.
* **ShapeNet, not ModelNet.** Every backbone is pretrained on ShapeNet; the Choy
  release also matches the design exactly — 24 views, native 32³ voxels, real cameras.
* **Registry order is the action index.** `backbones/__init__.py` — append, never
  insert, or a trained model head silently points at the wrong backbone.

---

## Gotchas that cost time — do not rediscover these

* **Benchmark on the `test` split.** Train-split models inflate results badly
  (Pix2Vox-A: 0.84 train vs 0.67 test). The fetcher defaults to `test` for this reason.
* **`ckpt_final.pt` holds 30,595 real training episodes.** A completed short run used
  to overwrite it; `CheckpointManager.save` now refuses to replace a `final` holding
  more episodes and writes `ckpt_final_ep<N>.pt` instead. An identical backup lives at
  `rl_proj_final - Copy (1)/rl_proj_final - Copy/checkpoints/ckpt_final.pt`.
* **Score at one threshold per cell, never per model.** Taking each model's
  best threshold is an oracle the policy does not have at inference, and it
  inflates IoU by ~0.015 — larger than several margins the conclusions rest on
  (UMIFormer vs UMIFormer+ at 3 views is 0.0021). `category_bench.py` shipped
  with this bug and was corrected; the fixed version reproduces
  `backbones.bench` to four decimals, which is the check that it agrees.
* **UMIFormer and UMIFormer+ checkpoints are byte-identical in size**
  (717,174,122 each). Any glob like `UMIFormer*.pth` matches both, and
  `kaggle_train.find_file` picks the largest — so it was resolving between them
  arbitrarily and could have loaded one checkpoint into both registry slots,
  silently making the pair two copies of one network. Matched by exact filename
  now, with a hard failure if they collide. Discovery lives in
  `kaggle_train.resolve_inputs()`, shared with `kaggle_bench.py` so it cannot
  drift.
* **Console output must stay ASCII.** This machine's console is cp1252; `π`, `×`, `±`
  and box-drawing characters raise `UnicodeEncodeError` and kill a run mid-training.
* **Heredocs mangle backslashes.** Writing Python containing `\n` inside a bash
  heredoc corrupts it. Use the file-writing tools for that, not `cat <<EOF`.
* **Stanford's server runs at ~130 KB/s here.** The 12.3 GB rendering archive is a
  13-hour download; that is why `fetch_eval_subset.py` exists.
* **Pix2Vox needs `TCONV_USE_BIAS=False`.** With `True` the decoder gains five bias
  tensors the checkpoint lacks and strict loading fails — this is what broke the
  original `build/pix2vox_f.py`.
* **The dataloader keeps RGBA.** Backbones composite the alpha themselves; the ResNet-50
  state encoder converts to RGB internally. Do not add a global `.convert("RGB")`.
* **`timm` 1.x incompatibilities** in vendored UMIFormer code are already shimmed
  (`HybridEmbed` removed, `default_cfgs` no longer subscriptable, DDP rank guards).

---

## Fixed along the way

Real bugs found in the original code, all repaired:

* `dataloader.py` fed **all-zero depth maps** — `_depth_from_voxels` was written and
  never called, so the 32³ coverage grid collapsed to one z-slice and the policy's 3D
  CNN branch saw no geometry.
* `python train.py` **deleted `ckpt_final.pt`**; and resume happened on *fresh* runs,
  which only looked correct because fresh runs deleted the checkpoints first.
* `utils/logger.py` read `Config` class attributes, so `--smoke-test` overrides were
  silently ignored.
* Non-ASCII characters in training output crashed on Windows.
* `infer.py` carried its own stale copies of `ViewPolicy`, the state builders and an
  obsolete reconstructor; it now imports the real modules.
