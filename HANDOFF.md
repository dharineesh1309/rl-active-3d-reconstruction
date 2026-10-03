# Handoff

Written for whoever picks this up next, with no memory of the work.
Everything below is measured unless marked otherwise.

Repo root: this directory (git).
Working branch: **`tier0-rgb-pose-policy`**. `master` holds the pre-rebuild
baseline.
Code: `build/rl_pipeline/`.

---

## What the project is

One agent that decides **which views to acquire** of an object and **which
pretrained reconstructor** should turn them into a 3D shape, under a budget that
prices both. The original proposal claimed this; the shipped code did only view
selection, and listed multi-backbone selection as future work.

---

## Current plan (agreed 2026-10-01) -- read this first

Fixed five-view system: policy picks 4 views after 1 random start, a two-stage
router picks the backbone. STOP is an optional extension, gated on a pilot.

| order | work | status |
|---|---|---|
| 0 | freeze splits, cache v2, cost definition, router code, resume, RGB inference | **substantially done**: CUDA resume and legacy-label verification run as gates at the start of Phase 1; `router.npz` is a Phase 2 output |
| 1 | retrain `set_pose` on `controller_train` (Kaggle), select checkpoints on dev | **done**: selected `set_pose_s164864` (`artifacts/tier1/selection.json`) |
| 2 | router from that run's cache; top up thin categories | **done**: `artifacts/router/router.npz`; no category thin (min 92) |
| 3 | dev evaluation matrix + latency + lambda sensitivity (spec below) | **done**: `artifacts/phase3/eval_dev.json` (results below) |
| 4 | STOP pilot (optional) | **skipped** for scope: variable-budget acquisition remains untested |
| 5 | freeze, run `final_test` once, rewrite report | **final test done** (results below; `artifacts/final/eval_final.json`, evaluated once under the enforced registration). Report rewrite next |

## FINAL TEST RESULTS (312 objects, evaluated once, 2026-10-03)

Pre-registered (`artifacts/final/preregistration.json` + amendment 1); the
evaluator verified every registered hash and setting before printing.

**Primary**: policy + corrected router vs random + always-UMIFormer+,
backbone-only utility (cost v1): **+0.0148 [+0.0093, +0.0200]. Supported.**

**Secondary** (intervals reported, not confirmatory):

| comparison | cost-v1 utility | CPU-estimated utility |
|---|---|---|
| full system - heuristic + umiformer_plus | +0.0017 [-0.0041, +0.0069] (no detectable difference) | -0.0232 [-0.0294, -0.0171] |
| full system - heuristic + pix2vox_f | +0.0499 [+0.0378, +0.0624] | -0.0272 [-0.0424, -0.0125] |
| corrected - plain router, random views | +0.0004 [-0.0028, +0.0033] | +0.0057 [+0.0022, +0.0093] |
| corrected - plain router, heuristic views | +0.0007 [-0.0029, +0.0041] | +0.0051 [+0.0013, +0.0095] |
| corrected - plain router, policy views | -0.0015 [-0.0062, +0.0022] | +0.0034 [-0.0012, +0.0077] |

**Exploratory**:
* **Policy vs farthest-angle heuristic: no detectable difference** for any
  backbone choice (corrected router -0.0018 [-0.0052, +0.0014]; fixed
  backbones -0.0014 to -0.0002). The dev advantage (+0.0041) did not replicate.
* Heuristic vs random: positive for every non-Pix2Vox choice (UMIFormer+
  +0.0130 [+0.0101, +0.0161]; routers +0.012). Spreading the views is where
  the view-selection gain comes from.
* The correction's **cost-v1 utility improvement** did not replicate. (It never
  improved quality: on policy views it lowered mean IoU vs the plain router on
  both dev, 0.7699 vs 0.7732, and final, 0.7769 vs 0.7840 -- its gain was in
  picking the cheaper backbone.) Its final CPU-utility gain on policy views is
  also uncertain: +0.0034 [-0.0012, +0.0077], despite lower estimated runtime.
* Highest cost-v1 means among deployable pipelines: heuristic + corrected
  router +0.0165, policy + plain +0.0163, heuristic + plain +0.0158, policy +
  corrected +0.0148 -- all within each other's noise.
* CPU-estimated (dev timing profile): the full system is -0.0102 [-0.0162,
  -0.0042] vs baseline; heuristic + umiformer_plus beats it under every timing
  variant checked (pipeline totals +0.0240 [+0.0032, +0.0447]; k=0.75 +0.0201;
  k=1.25 +0.0263). Pix2Vox-F pipelines lead the estimate but flip sign with
  machine speed (heuristic + pix2vox_f at k=0.75: -0.0109 [-0.0240, +0.0024]).
* Lambda (adaptive, cost-v1 prices): at lambda 0-0.04 routing adds ~0.001
  over policy + UMIFormer+; its value is the cost trade-off.

**Benchmark against the three reconstructors used normally** (post hoc,
requested after the final evaluation; `artifacts/final/standard_benchmark.json`;
same 312 objects, bootstrap and pairing; only the UMIFormer+ cost-v1 row is
pre-registered). "Normally" = 5 random views, one fixed reconstructor, the
Pix2Vox / UMIFormer evaluation protocol; 24 random view sets per object.

| full system minus | IoU | cost-v1 utility | CPU-estimated utility |
|---|---|---|---|
| Pix2Vox-F | +0.0991 [+0.0865, +0.1118] | +0.0524 [+0.0407, +0.0642] | -0.0248 [-0.0395, -0.0101] |
| UMIFormer | +0.0105 [+0.0024, +0.0177] | +0.0231 [+0.0155, +0.0300] | -0.0026 [-0.0103, +0.0050] |
| UMIFormer+ | +0.0021 [-0.0040, +0.0074] | **+0.0148 [+0.0093, +0.0200]** (pre-registered) | -0.0102 [-0.0162, -0.0042] |

Supported: better cost-aware utility than each standard reconstructor; better
IoU than Pix2Vox-F and UMIFormer; IoU vs UMIFormer+ not detectably different.
Not supported: any advantage once controller CPU time is charged (dev timing
profile). The plain-router variant would show IoU +0.0091 [+0.0049, +0.0133]
over UMIFormer+, but it is not the selected system; choosing it now would be
selecting on the final test, so it is reported as exploratory only.

**Report conclusion** (wording agreed in review): the pre-registered system
improved backbone-only utility over random views with fixed UMIFormer+. Its
learned view-selection advantage over farthest-angle selection was not
reproduced on the final test. The farthest-angle + UMIFormer+ baseline showed
no statistically detectable difference in backbone-only (cost-v1) utility --
which is not equivalence; the interval, full system minus baseline +0.0017
[-0.0041, +0.0069], admits meaningful differences either way -- and higher
estimated CPU-adjusted utility across all examined timing variants. These CPU
conclusions remain conditional on the measured profiles.

Utility is not reconstruction quality: mean IoU is 0.7878 for farthest-angle
+ UMIFormer+ against 0.7769 for the full system, which gives up some IoU by
routing to the cheaper backbone.

**Protocol** (`build/rl_pipeline/configs/splits_v1.json`, built by `splits.py`,
refuses to overwrite). The official split only says what the reconstructors
trained on. Controller splits are carved from the official TEST split:

| split | objects | use |
|---|---|---|
| `dev` | 309 | development and model selection -- every result so far |
| `final_test` | 312 (24/cat) | once, at the end. Excludes the first 60 test ids per category (all calibration/bench scripts drew from that prefix) and the 248 local benchmark objects |
| `controller_train` | 8,052 (>=163/cat) | policy and router training |
| `classifier_train` | official train | image -> category only |

263 cross-listed ids are excluded everywhere.

**Cost v1** (`config.py`): `backbone.cost` = CPU seconds per 5-view prediction
(Pix2Vox-F 0.51, UMIFormer/UMIFormer+ 1.28); `cost_lambda` = 0.0771 IoU/s,
chosen to keep the earlier operating point (penalty gap 0.0594, to 1e-5). All
56,592 cached routing decisions are unchanged; absolute utilities shift by an
approximately uniform -0.033 (-0.033021 Pix2Vox-F, -0.033028 UMIFormers), so
older tables differ in level, not in differences. Controller overhead is not
priced and must be reported. One `infer_rgb.py` run on CPU, warm, loading
excluded: view selection (policy + ResNet) 0.43 s, router 1.5 ms, Pix2Vox-F
0.53 s. Loading was 1.5 s (policy + ResNet) and 1.8 s (Pix2Vox-F).

**Reporting rule that follows.** At 0.0771 per CPU-second, that 0.43 s of
selection is worth 0.033 utility -- more than the whole projected
view + routing gain (~0.013). One CPU example proves nothing about the system,
but it means the primary metric must be named **backbone-only utility**, and
Phase 3 must also report total latency and utility *including* controller
overhead, on the declared CPU and on GPU.

**Cache v2** (`training/utility_envelope.py`): raw IoU per backbone, so lambda
and costs apply at read time; records the scoring version, thresholds, and per
run the SHA-256 of each checkpoint (file name and size are not enough: the two
UMIFormer files have identical sizes); each entry records its run, kind
(train / eval_policy / eval_random) and policy step or checkpoint; refuses
mismatched labels; backbone calls run in a forked, reseeded RNG.
`cache/utility_cache.json` is the converted Tier 0B cache: run `legacy`, IoU
derived as utility + 0.07 * old cost. Its producer stays "unknown" for good.
Before reuse, run_0b calls `verify_legacy()`: per unknown-producer run, 26
entries spread across categories (the first 24 in file order were all
aeroplanes) are recomputed with the live checkpoints; a match is recorded as
`compatible_with` (hashes, keys, tolerance, max difference), a mismatch or an
uncheckable run stops the job. Original kept as `cache/utility_cache_legacy.json`.

**Router** (`train_router.py`): two-stage, reproduced from committed code by
5-fold CV on dev (`artifacts/router/router_report_cv5.json`): category
accuracy 88.3%; argmax 18.0% of the routing gap; **expected (category
probabilities) 19.8%, +0.0066 vs best single backbone, 95% CI [+0.0037,
+0.0095]**; true-label reference 22.3%. This is a development CV result. The
interval supports beating the single backbone; it does not show that
probability weighting beats argmax (18.0%).

**Phase 1 run** (Kaggle T4, build 9809170, `artifacts/tier1/set_pose.jsonl`):
200,704 steps, 50,176 episodes, 9.19 h, one session. All gates passed on CUDA;
the legacy run verified compatible (26 entries, 13 categories, max |dIoU|
2.9e-4 -- GPU sessions are not bit-identical, a voxel or two). Dev
policy - random (one start, se ~0.0016) climbs to +0.0081 at 62k steps and
stays at +0.0068 to +0.0081 after, 17-19.5% of the 0.0417 headroom; Tier 0B's
policy was +0.0069 at its last eval. Kept: s62464, s123904, s164864 (+0.0081,
+0.0080, +0.0080) and the final s200704. Cache now 107,955 view sets (none of
the old changed), with labels on **4,327 controller_train objects** (telephone
92 ... table 851) and none on final_test.

**Phase 1b** (`artifacts/tier1/eval_dev.json`, 309 dev objects x 4 starts):
policy - random is +0.0065 to +0.0071 for each of the four clean checkpoints
and the old Tier 0B policy; every pairwise |diff| <= 0.0006 (z <= 0.64).
**This run showed no statistically detectable dev improvement from clean
retraining or from training beyond ~62k steps.** That is not equivalence: the
selected checkpoint vs Tier 0B under the router has a paired 95% interval of
about +-0.0032, which still admits meaningful differences, and a claim about
training in general would need independent seeds. Features extracted for
6,594 objects.

**Phase 2 router** (`train_router.py`, `artifacts/router/router_report_ctrl.json`):
table from 4,327 controller_train objects, classifier on all non-dev objects
(89.8% category accuracy). On dev:

| router | captured | vs best single [95% CI] |
|---|---|---|
| expected (two-stage) | 10.3% | +0.0034 [+0.0016, +0.0051] |
| true-category lookup (reference) | 12.2% | +0.0040 [+0.0017, +0.0062] |
| **corrected (shipped)** | **19.8%** | **+0.0065 [+0.0039, +0.0090]** |

**These are mixed-cache dev results**: scored on all 18,448 cached dev view sets
(random subsets plus several policies' views), frozen in
`artifacts/router/dev_cohort_v1.json` so later cache additions cannot change
them. On that mixture corrected - expected is +0.0031 [+0.0010, +0.0052]; on
the **selected policy's own views** (4 starts) it is +0.0028 with a 95% interval
touching zero ([+0.0001, +0.0054] or [-0.0001, +0.0054] depending on the
bootstrap draw) -- promising, not established. Phase 3 therefore scores both
the plain and the corrected router under every view strategy.

The correction's alpha/beta come from 5-fold CV over controller_train objects,
with the classifier, its temperature, the table and the ridge all refit inside
each fold (an earlier version calibrated the temperature outside the folds; 627
objects overlapped; fixing it left the selection and the router unchanged). A
single 15% hold-out was a lottery (one draw in five picked an over-regularised
alpha).

The earlier dev-CV 19.8% was genuinely out of fold; it was built from a
dev-population table. Fit on controller_train instead, the same two-stage
router scores 10.3% on dev; dev and controller_train disagree on two
categories (chair: umiformer vs umiformer_plus; telephone: pix2vox_f vs
umiformer). Held-out controller_train objects route better than dev (24% vs
10%), but that comparison does not isolate object difficulty: training CV
weights an imbalanced sample (car, chair and table are 54.1% of the 4,327
objects, 23.1% of the balanced final_test) and its view sets come from training
trajectories. **Final-test performance is unknown until it is measured.**

**Backbone-only headline so far** (selected policy, 4 starts, dev): policy +
corrected router vs random views + always-UMIFormer+ is **+0.0178 [+0.0141,
+0.0216]**. At 0.0771/CPU-second that covers ~0.23 s of controller overhead per
object; one CPU measurement of view selection was 0.43 s. Total latency decides
whether the system pays for itself.

**Phase 3 spec** (dev only, objects as the resampling unit, starts averaged
within objects, every view-set list frozen before scoring):
* views {random, farthest-angle heuristic, selected policy} x backbone
  choice {each fixed backbone, plain router, corrected router, true-category
  lookup (reference, not deployable), oracle (reference)}; "best of 24 sampled
  five-view sets" named as such.
* report IoU, **backbone-only utility**, total latency, and utility including
  controller overhead; shared ResNet features counted once; CPU and GPU
  measured separately; model loading stated explicitly (excluded from the
  per-object figure, reported on its own).
* two separate utility accountings, never mixed: **frozen cost v1** (declared
  backbone prices, backbone-only) and **fully measured** (IoU - lambda x the
  measured backbone + controller seconds of the same benchmark run).
  `bench_latency.py` times the 15 deployable pipelines ({random, heuristic,
  policy} x {3 fixed backbones, plain router, corrected router}) on the FROZEN
  evaluation episodes of a fixed 26-object cohort (2 dev objects per category,
  24 candidates), keeping every raw measurement with its start, views and
  routed backbone, and checking that the policy and heuristic reproduce their
  frozen views. Heuristic + router still pays ResNet (~0.32 s CPU); only
  heuristic + a fixed backbone is nearly free. The first CPU run used 10
  candidates (local mirror) and was discarded.
* the measured cost term uses MEAN runtimes with object weights (medians only
  describe latency). Evaluation decisions outside the benchmark cohort are
  priced from component means -- controller by strategy, backbone by the one
  actually chosen -- and labelled *estimated*; the cohort's end-to-end totals
  check that estimate. Timing uncertainty is resampled together with objects.
* caches are combined only with `merge_caches()` (union by key, refuses on
  conflicting labels or provenance), never replaced by a bigger copy.
* lambda sensitivity, both kinds, from raw IoU (no new reconstructions):
  *frozen-system* -- the decisions fixed, rescored at each lambda; *adaptive* --
  the table, ridge and their CV rebuilt from training labels at each lambda.
  The view policy stays fixed in both.

**Phase 3 results** (dev, 309 objects; `phase3_eval.py`; 2,000-rep stratified
joint bootstrap; baseline random views + always UMIFormer+):

| pipeline | IoU | v1 utility vs base [95% CI] | CPU-measured util (est.) vs base | GPU s |
|---|---|---|---|---|
| random + pix2vox_f | 0.672 | -0.0340 [-0.0458, -0.0228] | +0.0182 [+0.0045, +0.0324] | 0.044 |
| heuristic + pix2vox_f | 0.675 | -0.0303 | **+0.0219 [+0.0080, +0.0364]** | 0.044 |
| heuristic + umiformer_plus | 0.773 | +0.0082 [+0.0053, +0.0114] | +0.0082 | 0.193 |
| policy + umiformer_plus | 0.776 | +0.0109 [+0.0081, +0.0136] | -0.0184 | 0.241 |
| heuristic + router_corrected | 0.766 | +0.0137 [+0.0100, +0.0177] | -0.0085 | 0.204 |
| **policy + router_corrected** | 0.770 | **+0.0178 [+0.0141, +0.0216]** | -0.0067 [-0.0120, -0.0015] | 0.216 |
| policy + oracle (reference) | 0.784 | +0.0414 | -- | -- |

Findings:
* **Backbone-only, the full system has the highest dev mean** of the
  deployable pipelines (superiority over each alternative is only what the
  paired contrasts below establish).
* **The farthest-angle heuristic carries most of the view gain.** With
  UMIFormer+: random 0, heuristic +0.0082, policy +0.0109. Policy - heuristic
  is significant only with a router (+0.0041 [+0.0008, +0.0071] corrected;
  +0.0040 plain); with fixed backbones +0.0007 to +0.0027, intervals span 0.
* **Corrected - plain router**: +0.0035 [+0.0015, +0.0057] on random views;
  +0.0027/+0.0028 with lower bound 0.0000 on heuristic/policy views.
* **CPU-measured: under this CPU timing profile, the frozen system has
  negative estimated utility relative to the baseline** (-0.0067 [-0.0120,
  -0.0015]). ResNet features cost ~0.4 s per object (5 x 95 ms on this run),
  ~0.03 utility. This is a statement about the frozen, cost-v1-trained system
  on this profile; it does not show that a router optimised for measured CPU
  costs would fail (the adaptive lambda study still uses cost-v1 prices).
  Measured CPU backbone times (Pix2Vox-F 0.68 s, UMIFormer+ 2.12 s) also make
  UMIFormer+ far dearer than cost v1 declares (gap 1.45 s vs 0.77 s).
* **That conclusion is conditional on the timing model and machine state**
  (`cpu_timing_sensitivity` in eval_dev.json):

  | CPU contrast | components, k=1 | measured pipeline totals | k=0.75 | k=1.25 |
  |---|---|---|---|---|
  | full system - baseline | -0.0067 [-0.0120, -0.0015] | -0.0019 [-0.0204, +0.0187] | -0.0039 [-0.0082, +0.0004] | -0.0096 [-0.0157, -0.0035] |
  | heuristic + pix2vox_f - baseline | +0.0219 | +0.0270 [+0.0123, +0.0426] | -0.0060 [-0.0187, +0.0073] | +0.0497 |
  | heuristic + umiformer_plus - full system | +0.0149 | +0.0166 [-0.0044, +0.0361] | +0.0121 [+0.0075, +0.0167] | +0.0178 |

  k scales every CPU timing (earlier sessions on this machine ran ~0.77x this
  one's). The component estimate is within 3.3% of the cohort's measured
  totals per pipeline, but that error matters here: for the full system vs
  baseline it is 0.048 s, i.e. 0.0037 utility, more than the interval's
  distance from zero. Measured totals are themselves noisy (two identical-
  compute UMIFormer+ pipelines differ by 0.08 s). Only "a controller-free
  UMIFormer+ pipeline beats the full system on CPU" holds at both machine
  speeds, and not significantly under measured totals.
* Policy and heuristic reproduced every frozen episode on CPU and GPU.
* **Lambda**: with decisions frozen, the routers' selections do not change;
  their cost savings simply weigh more as lambda grows. Rebuilt per lambda
  (adaptive, cost-v1 prices), the corrected router shows little observed
  additional quality benefit at lambda 0 (+0.0003 IoU over policy +
  UMIFormer+); its value is the cost trade-off. The training-best single
  backbone flips to Pix2Vox-F at lambda >= 0.15.
* **STOP is skipped for scope, not because it cannot help**: at 95 ms per
  ResNet call, three acquired views instead of five would save ~0.19 s
  (~0.015 utility) before any quality loss -- enough to matter against the CPU
  result. Variable-budget acquisition remains untested and outside this
  fixed-budget study.

**Checkpoint selection** (`score_routed.py`, refinement 5): with the router,
every checkpoint's dev view sets score within noise (routed 0.6832-0.6843);
clean_s164864 ties the old Tier 0B policy at the top and is selected by the rule
"highest routed dev utility, ties to a controller_train policy".

**Scripts**: `run_0b.py` (trains on controller_train, evaluates on dev, keeps
the 3 best checkpoints + full-state `<arm>_last.pt`, `--resume`), `eval_0b.py`
(paired dev comparison of any checkpoints; `--split final_test` once),
`infer_rgb.py` (images + poses -> views -> router -> one backbone; GT only via
optional `--gt` scoring), `extract_router_feats.py`, `train_router.py`.

---

## Status: mid-rebuild

Four training runs were completed on the original architecture. None learned view
selection. The reason was found late and is structural:

> **View index `i` denotes a different camera direction on every object.**
> Measured across 988 Choy train objects: mean per-index azimuth std **102.5
> degrees**, where a shared 24-pose lattice would give 0.0. Elevation and
> distance also vary per object.

More precisely: the renderer drew from a pool of roughly **110 distinct
24-camera configurations**, each reused by about 280 train objects. Random
sampling of 988 objects found 109 distinct sets and saturated the pool -- at 500
configurations we would have expected 431. So index `i` is not random per
object, it is one of ~110 directions, which is still unlearnable from an index
alone because the policy has no way to tell which configuration it is in without
the pose data.

A policy whose action is an integer index therefore cannot generalise across
objects, and no amount of credit-assignment work downstream repairs that. The
rebuild replaces the index action space with pose-conditioned scoring.

Evidence: `artifacts/camera_metadata/` (figures, CSV, JSON), generated by
`build/rl_pipeline/validate_cameras.py`.

---

## Decisions already taken -- do not relitigate

**RGB only.** The state is images plus their known camera poses. Nothing derived
from ground-truth geometry enters it. The old depth-derived coverage grid is
gone from the RGB path.
*Not* because coverage was proven useless -- coverage was measured to be a poor
*reward* (at chance for picking the best next view, 1/8), which says nothing
about its value as an *observation*. It is excluded to keep an RGB sensor
contract, since the reconstructors are RGB-only. The RGB-D variant is a planned
later ablation and is preserved on `master`.

**Backbone routing leaves the RL loop.** All three reconstructors can be
evaluated offline on the same view set, so routing is full-information
supervised learning, not a bandit. During RL the terminal reward is
`max_m [IoU_m(S) - lambda*cost_m]`, so a good view set is never punished for a
routing mistake a learning head happens to make.

**Three backbones**, each of which measurably wins somewhere: `pix2vox_f`,
`umiformer`, `umiformer_plus`. `occnet` and `triposr` won zero of twelve
category-budget cells; they, `pix2vox_a`, `r2n2` and `pixelnerf` have been
removed (code and weights). Why each was dropped: README "Backbones".

**lambda = 0.07**, derived by measurement, not chosen.

---

## Key measurements -- reuse these, they cost hours

| quantity | value |
|---|---|
| mean per-index azimuth std | 102.5 deg (988 objects, 24 views) |
| distinct camera configurations in the dataset | ~110, each shared by ~280 train objects |
| `in_plane`, `fov` | constant (0.0, 25.0) -> pose descriptor is 5-D |
| distance train stats | mean 0.796219, std 0.085869 |
| view-choice signal (within object) | 0.0266 IoU |
| object-difficulty noise (between objects) | 0.0621 IoU |
| perfect view planner gains | +0.0402 IoU |
| total backbone-router headroom | +0.0299 |
| value of knowing category | +0.0146 |
| value of knowing the object | +0.0131 |
| **value of knowing the view set** | **+0.0022** (median 0.0000, P95 0.0117) |
| coverage gain vs best next view | at chance, 1/8 |
| cheap->dear backbone view-preference transfer | rho +0.171, no usable transfer |

**Read the +0.0022 carefully.** Given the object and the budget, which views were
taken barely affects which reconstructor is best. The decisions are close to
decoupled; the interaction is concentrated at low budgets on ambiguous objects
(chair B=3: 0.0039; car B=8: 0.0006). The honest framing is "sparse and
heterogeneous interaction", not "joint optimisation".

---

## Stage 1 -- DONE (all CPU)

`policy/pose_policy.py`, `env/rgb_view_env.py`, `training/rgb_ppo_trainer.py`,
`training/rgb_rollout_buffer.py`.

* 5-D pose `[sin az, cos az, sin el, cos el, normalised distance]`, distance
  normalised with **train-split** statistics frozen in
  `rl_pipeline/configs/pose_norm_v1.json`
* ResNet 2048 -> LayerNorm -> Linear -> GELU -> 256, replacing a fold that
  averaged four arbitrary channel blocks
* Pose-bound view tokens, 2-layer set transformer with a learned `[STATE]`
  token, no positional embeddings
* One shared candidate scorer `f(state, pose, angular relation)`
* **9 invariance tests** (`policy/tests/test_pose_policy_invariance.py`),
  including a control asserting the index head *fails* them -- without it the
  other eight could be vacuous
* **2 synthetic learning tests** (`policy/tests/test_synthetic_learning.py`):
  pose target 0.280 -> 0.938, delayed credit 0.145 -> 0.902, chance 0.125

All four Tier 0B ablation arms come from one class via `set_encoder` and
`pose_head` flags. The two arms that isolate the action abstraction differ by
2.6% in parameter count, so a difference between them is not capacity.

---

## Stage 2 -- NEXT, needs Kaggle GPU

1. **Terminal reward** `max_m U_m(S)`: run all three backbones at episode end,
   batched per backbone, memoised by `(object_id, sorted(view_ids))`. That cache
   doubles as the supervised router's training set, on states the policy
   actually visits.
2. **0B-1 rebuild gate**: old index policy vs new pose policy at fixed B. Same
   PPO, GAE, group baselines, LR schedule, seeds.
3. **0B-2 causal ablation**: new encoder + index head vs new encoder + pose
   head. This is the comparison that isolates the action abstraction; the
   rebuild gate alone cannot attribute the improvement.

Gate before Tier 1: no ground-truth data in the policy, all invariance tests
pass, synthetic tests pass, new architecture at least competitive with old,
held-out-object view regret improves or gives a clear diagnosis.

### Tier 0B, first run (Kaggle, 3 arms, B=4 plus one seeded view)

Held-out eval: 312 test objects (24 per category x 13), one greedy rollout each.
Random 0.7348, oracle (best of 24 random 5-view sets) 0.7763, headroom 0.0415.

| arm | steps reached | policy - random @21.5k | @42.0k | @62.5k |
|---|---|---|---|---|
| set_pose | 41,984 | +0.0053 | **+0.0069** (16.6%) | -- |
| set_index | 65,536 | +0.0031 | +0.0013 (3.2%) | +0.0030 (7.3%) |
| mean_pose | 63,488 | +0.0018 | +0.0062 (14.9%) | +0.0071 (17.0%) |

**SE of one eval is ~0.0018** (within-object sd of a 5-view set is 0.025,
recovered by replaying the eval RNG against the cache -- the replay reproduces
the logged random/oracle exactly). So: pose-conditioned arms lead the index arm
at matched steps (set_pose - set_index ~2.8 SE at 42k, unpaired, one of several
looks); set encoder vs mean pooling is indistinguishable; all arms still
learning (entropy 2.4-2.8 against 3.07 uniform).

**Settled by the paired re-eval** (`eval_0b.py`, same 312 objects, 4 start
views each; `artifacts/tier0b/eval_0b.json`). Its one-start gate reproduced the
training eval exactly (+0.0069, 16.6%), so checkpoint and cache load as trained.

| a - b, per object | mean | se | z | 95% CI (bootstrap) |
|---|---|---|---|---|
| set_pose - set_index | **+0.0053** | 0.0017 | +3.1 | [+0.0020, +0.0086] |
| set_pose - mean_pose | +0.0009 | 0.0014 | +0.6 | [-0.0019, +0.0035] |
| set_index - mean_pose | -0.0044 | 0.0014 | -3.2 | [-0.0071, -0.0018] |

Against random: set_pose +0.0067 (z 5.1, 16.1% of headroom), mean_pose +0.0058
(14.0%), set_index +0.0014 (z 1.2, 3.4%). set_pose is positive in 12 of 13
categories (cabinet ~0; largest on display, +0.019).

* **The action abstraction was the problem.** The pose head beats the index
  head with the encoder held fixed, although the index arm trained on 56% more
  steps.
* **The set transformer's contribution is not detected.** Mean pooling is
  within noise. set_pose stays the proposed model (higher estimate, 34% fewer
  steps), but report the encoder ablation as null.
* In absolute terms view selection is worth ~0.007 IoU here, ~1/6 of the
  best-of-24 headroom. Consistent with the earlier +0.0402 ceiling for a
  perfect planner: this task has a small view-selection signal.

**Tier 0B gate: passed.** No GT in the policy, invariance and synthetic tests
pass, pose arm beats the index arm, held-out regret improves.

Things this run taught:
* **Nothing reached 200k steps.** Throughput is set by cache misses, ~0.6 s
  each (three backbones); every arm did 17-18.5k misses in its 3.2 h.
* **The first arm pays for the eval baseline**: 7,488 random subsets, ~1.3 h of
  set_pose's budget. Later arms got them from the cache, which is why set_pose
  reached the fewest steps. Attach the cache to any later run.
* Same seeds make the arms' early trajectories near-identical (cache hit rate
  0.75 by 9k steps for arms 2 and 3) -- common random numbers, which helps the
  comparison.
* LR "annealing" was annealing over 1e9 episodes, i.e. constant 3e-4. Now
  constant on purpose.
* **Router dataset: 53,470 rows, all 5-view sets, and 10,833 (20%) are
  test-split objects** written by the evals. Tier 1 must split by object via
  `datasets/ShapeNet.json`, never by row. It also covers one view count only.

### Backbone determinism (checked 2026-10-01)

UMIFormer's token clustering adds `torch.rand(...) * 1e-6` to break density
ties at inference. Measured on 6 objects, 5 views, CPU: across 4 RNG seeds and
3 view orders per object, IoU at 0.4 did not change for any backbone; max
probability change 2.3e-5. The cache's "deterministic, order-free" assumption
holds in practice. Side effect still to fix: that call consumes the global
torch RNG, so cache misses shift the policy's sampling stream.

### Tier 1 -- router, in progress

`extract_router_feats.py` (Kaggle GPU, once) writes the policy's own ResNet
features and pose descriptors for every cached object; `train_router.py` then
trains on CPU. Split by object: test = the 309 Tier 0B eval objects, val/fit =
the rest. Self-check `training/tests/test_router.py` (mutation-tested: shuffled
features fail it).

Baselines on the held-out objects, measured from the cache alone:

| method | utility | regret vs per-set oracle |
|---|---|---|
| best fixed backbone (umiformer_plus) | 0.7023 | 0.0334 |
| best backbone per category (uses label) | 0.7065 | 0.0292 (12.4% captured) |
| oracle per view set | 0.7357 | 0 |

Routing headroom (0.0334) is five times what view selection captured (0.0067).

**The backbones memorised the train split, unevenly.** All three were trained
on the official train split, which is where every RL training episode -- and
so every router label from training -- comes from:

| cached rows | pix2vox_f IoU | umiformer IoU | umiformer_plus IoU |
|---|---|---|---|
| train-split objects (42,154) | 0.709 | 0.845 | 0.865 |
| test-split objects (13,842) | 0.673 | 0.762 | 0.768 |

UMIFormer/UMIFormer+ lose 0.08-0.10 on unseen objects, Pix2Vox-F 0.04. Who wins
shifts with it: umiformer_plus wins 60-84% of rows on seen objects in most
categories, 36-53% on unseen ones, and on chairs plain umiformer overtakes it.
Router labels from train objects are biased toward umiformer_plus; the first
router, trained on them, captured 4.7% of the gap on held-out objects (below
the category lookup's 12.4%) and peaked at epoch 0. Training RL there also
inflates the reward (Tier 0B train ~0.80 vs held-out random 0.735).
Clean objects: the official test split minus the 312 eval objects (~8,450,
>=187 per category). The val split was used for the backbones' checkpoint
selection, so it is not fully clean.

**What clean labels buy** (5-fold CV by object over the 309 unseen eval
objects, ~247 training objects per fold; share of the 0.0334 gap captured):

| router | captured |
|---|---|
| category lookup, true label, table from clean objects | 22.3% |
| **two-stage: category predicted from the 5 views, then that table** | **18.0%** |
| category lookup, table from seen objects | 12.4% |
| neural router (`train_router.py`), clean labels | ~0% |
| neural router, seen-object labels | 4.7% |

The category classifier is a ridge on 5-view mean ResNet features trained on
train-split objects -- category labels are not biased by memorisation, so the
big split is usable for it -- and gets 88.3% on unseen objects. The end-to-end
neural router does not learn from ~250 clean objects; it needs far more, or
the two-stage structure.

**Cross-listed ids.** 263 model ids appear under two categories in
`datasets/ShapeNet.json`, some in train under one and test under another (e.g.
`4bb41171...` aeroplane-train / watercraft-test). 23 of the 2,290 cached objects
are among them; 7 sit in both train and test. The utility cache key has no
synset, so for these ids renders and utilities are ambiguous. The router drops
them. Left as-is in RL: ~1% of objects, and changing the key would invalidate
the cache.

Then Tier 1 (supervised router, utility regression with gap-weighted ranking --
not classification, since 47% of argmax flips cost nearly nothing) and Tier 2
(STOP as a 25th candidate, justified by acquisition cost; the count/routing
coupling hypothesis was tested and came back at +0.0010).

---

## Gotchas that cost real time

* **`.gitignore`: never leave `env/` unanchored.** It matched
  `build/rl_pipeline/env/` and silently excluded the whole environment package.
  The `baseline` commit claimed to preserve the RGB-D work and did not; those
  files entered history at `dc4758b`, not `a08378f`.
* **Kaggle nests datasets**: `ShapeNetRendering/ShapeNetRendering/<synset>`.
  `validate_cameras._resolve_root` handles it; a helper that stops at the first
  name match finds only the shell.
* **Never `rglob` over `/kaggle/input`** -- about a million files once the
  renderings are attached; one scan ran 1117s and printed nothing. Use a
  depth-limited walk that prunes all-digit (synset) directories.
* **Bash heredocs mangle backslashes.** Writing Python containing `\n` through
  `<<'EOF'` corrupts it. Use the Write/Edit tools instead.
* **Console output must stay ASCII** -- the local console is cp1252 and
  non-ASCII in a `print()` kills a run.
* **Check group-sync per step, not per rollout.** A 24-step rollout of 3-step
  episodes legitimately visits eight objects.
* `ckpt_final.pt` holds 30,595 real episodes; `CheckpointManager.save` refuses
  to overwrite a `final` holding more episodes.

---

## Open

1. ~~`--sample random` duplication check~~ **DONE.** Random sampling gives 109
   distinct sets in 988 objects (89% duplication) against 134 (86%) for
   contiguous sampling -- more duplication, not less, so it was never a sampling
   artifact. Pool size ~110 configurations.
2. ~~R2N2's forward pass and pixelNeRF's cameras~~ moot: both removed.
3. `EXPERIMENTS.md` sections 1-17 describe the RGB-D era. They remain accurate
   as history but do not describe the current architecture.
