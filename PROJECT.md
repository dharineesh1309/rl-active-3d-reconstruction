# Learning where to look and which reconstructor to run

**Active view selection and reconstructor routing for multi-view 3D
reconstruction under a compute budget.**

## Abstract

Multi-view 3D reconstruction pipelines usually fix two things in advance: the
views they look from and the single model that turns those views into a shape.
This project replaces both fixed choices with learned ones. For each object, a
reinforcement-learning policy chooses which camera views to acquire, and a
learned router chooses which of three pretrained reconstructors -- Pix2Vox-F,
UMIFormer, UMIFormer+ -- to run on them, under an objective that charges for
compute. The policy conditions on camera *poses* rather than view indices,
after we found that view indices denote different directions on every object
of the dataset. Because all three reconstructors can be scored offline on any
view set, routing is learned as a full-information supervised problem rather
than inside the reinforcement-learning loop. All three reconstructors were
trained on ShapeNet's official train split and are measurably better on it,
so the controller is trained, selected and tested only on objects none of them
has seen. On 312 such objects, evaluated once under a pre-registered and
hash-enforced protocol, the system improves cost-aware utility over each
reconstructor used in the standard way (five random views): +0.052 over
Pix2Vox-F, +0.023 over UMIFormer, and +0.015 over UMIFormer+ (pre-registered,
95% interval [+0.009, +0.020]). A controller-free baseline -- spreading the
views as far apart as possible and always running UMIFormer+ -- shows no
statistically detectable difference in that utility, and once the controller's
own CPU time is charged the system no longer beats the standard
reconstructors. We report both, and what each implies.

## 1. Motivation

Reconstructing a 3D shape from a few images forces two decisions that are
usually made once, offline, and for every object alike:

* **Where to look.** A handful of views from a fixed or random set of camera
  positions. Some objects are captured well by any five views; others need the
  one view that shows a thin leg or a hidden cavity.
* **Which model to run.** Reconstruction networks differ in accuracy and cost,
  and the most accurate one is not always worth its compute -- a cheap model
  can be nearly as good on simple shapes.

Both choices could depend on the object. A system that makes them per object
should beat the fixed defaults on what actually matters: reconstruction quality
for the compute spent.

## 2. Problem formulation

An object is rendered from 24 candidate cameras. The system receives one
start view, then chooses 4 more (a budget of `B = 4`), then chooses a
reconstructor `m` from {Pix2Vox-F, UMIFormer, UMIFormer+}. The outcome is
scored by **cost-aware utility**

```
U(S, m) = IoU_m(S) - lambda * c_m
```

where `S` is the set of five acquired views, `IoU_m(S)` is the voxel IoU of
reconstructor `m` on `S` against the 32^3 ground truth (each reconstructor's
occupancy threshold frozen at 0.4), `c_m` is its compute cost in CPU seconds
per five-view prediction (declared before the study: Pix2Vox-F 0.51 s,
UMIFormer and UMIFormer+ 1.28 s), and `lambda = 0.0771` utility per CPU
second.

**Decomposition.** View acquisition is sequential and its value is only
revealed at the end, so it is a finite-horizon Markov decision process: the
state is what has been seen, an action is the next camera, and the terminal
reward is the utility of the resulting view set. Reconstructor choice is
different in kind: given a view set, all three reconstructors can be run
offline and their utilities compared exactly, so routing is a
full-information supervised problem, not a bandit. The view policy is
therefore trained against the **envelope reward**

```
R(S) = max_m U(S, m)
```

-- the best achievable utility over reconstructors -- so a good view set is
never penalised for a routing mistake a learning router happens to make, and
the router is trained separately on cached utilities of every reconstructor
on the view sets the policy visits.

## 3. Why the first design failed

An earlier version of the system used a policy that picked views by index
(0-23), with a depth-derived coverage grid in its state. Four training runs
learned no useful view selection. The cause was structural: across 988
training objects the azimuth of camera index `i` has a standard deviation of
102.5 degrees (a shared camera lattice would give 0). The renderings use
roughly 110 different 24-camera configurations, so "view 7" points in
unrelated directions on different objects, and no amount of training can
generalise an index-valued action. The rebuild therefore:

* scores candidate cameras by their **pose** (`sin`/`cos` of azimuth and
  elevation, normalised distance), so the action means the same thing on every
  object;
* uses an **RGB-only** state (image features and camera poses), with nothing
  derived from ground-truth geometry;
* takes routing **out of the reinforcement-learning loop** (section 2).

## 4. System

**Observation.** For every acquired view: a frozen ResNet-50 feature vector
(2048-d) and its 5-d pose descriptor. For every candidate: its pose and its
angular relation to the views already acquired (nearest and mean angular
separation).

**View policy.** An actor-critic network: image features are projected to 256
dimensions and bound to their pose embeddings as tokens; a two-layer set
transformer with a learned `[STATE]` token (no positional embeddings -- the
acquired views are a set) summarises them; one shared scorer rates each
candidate from the state, its pose and its angular relation; already-acquired
cameras are masked. It is trained with PPO and generalised advantage
estimation, with parallel environments synchronised on the same object so a
leave-one-out group baseline cancels object difficulty.

**Utility cache.** Every reconstruction is cached by object and sorted view
set, storing the raw IoU of all three reconstructors together with the run,
kind (training, policy evaluation, random, heuristic) and checkpoint hashes
that produced it. Utilities are computed from IoU at read time, so `lambda`
and costs can change without re-running anything; caches from different
sessions are merged by key and refused if they disagree.

**Router.** Two stages. A ridge classifier predicts the object's category
from the mean feature of the acquired views (89.8% accurate on development
objects); a temperature-scaled softmax turns its scores into category
probabilities, which weight a per-category table of each reconstructor's
average utility. A regularised ridge correction on the same features adjusts
that estimate per object. Its two hyper-parameters are chosen by five-fold
cross-validation over training objects, with the classifier and its
calibration refit inside each fold.

**Inference.** Images and camera poses in: the policy picks four views, the
router picks one reconstructor, and only that reconstructor runs. Ground truth
is never read.

## 5. Data and protocol

**Data.** ShapeNet renderings and 32^3 voxels as released with 3D-R2N2
[1, 2]: 13 categories, 24 views per object. Objects that the official
taxonomy lists under two categories (263) are excluded everywhere.

**The reconstructors memorised their training split.** All three were trained
on ShapeNet's official train split, and they behave differently there:

| cached view sets | Pix2Vox-F IoU | UMIFormer IoU | UMIFormer+ IoU |
|---|---|---|---|
| official-train objects (seen in training) | 0.709 | 0.845 | 0.865 |
| official-test objects (unseen) | 0.673 | 0.762 | 0.768 |

The gap is uneven across reconstructors, so routing labels from seen objects
are biased towards UMIFormer+, and a policy trained there optimises an
inflated reward. The controller is therefore trained and evaluated only on
objects from the official **test** split, divided into `controller_train`
(8,052 objects), `dev` (309) and `final_test` (312, 24 per category, drawn at
random after excluding every object any calibration script had touched). The
category classifier alone uses official-train objects, since memorisation
does not bias a category label.

**Pre-registration.** Before any final-test data existed, a registration fixed
the primary comparison, the secondary comparisons, the metrics to report
whatever the outcome, the sampling seed, the statistics, and the SHA-256 of
every artifact the evaluation depends on. The final evaluator refuses to print
anything unless all of them match. The final test was collected without
computing any aggregate result, and evaluated once.

**Statistics.** Object-level paired bootstrap (2,000 replicates): every system
is compared on the same objects with the same resampling weights; episodes
are averaged within objects.

## 6. Experiments and results

**6.1 Does pose conditioning fix view selection?** With the set encoder held
fixed, a pose-conditioned scorer beat an index head by +0.0053 [+0.0020,
+0.0086] in utility on development objects, although the index arm trained on
56% more steps. Replacing the set transformer by mean pooling showed no
detectable difference (+0.0009 [-0.0019, +0.0035]). The pose-conditioned
policy beat random views by +0.0067 (z = 5.1), about 16% of the gain a perfect
choice among 24 sampled view sets would give.

**6.2 Retraining on unseen objects.** Retrained for 200,704 steps on
`controller_train`, the policy plateaued by about 62k steps, and no
checkpoint was detectably better than the policy trained on seen objects.
This shows no detectable improvement from clean retraining in this run, not
that the two are equivalent.

**6.3 Routing.** On development view sets, the best fixed reconstructor
(UMIFormer+) leaves a regret of 0.033 utility against an oracle that picks the
best reconstructor per view set. The two-stage router recovers 10.3% of it;
with the per-object correction, 19.8% (+0.0065 [+0.0039, +0.0090] over the
best fixed reconstructor), measured on a mixture of cached view sets.

**6.4 Development matrix.** Crossing view strategies (random, farthest-angle,
policy) with reconstructor choices showed that a simple farthest-angle
heuristic -- add the candidate camera farthest from every view already held --
captures about three quarters of the policy's view-selection gain, and that
on a CPU the controller's ResNet-50 features (~95 ms per view) cost more
utility than routing recovers.

**6.5 Final test (312 unseen objects, evaluated once).**

| full system minus | IoU | cost-aware utility | CPU-adjusted utility (est.) |
|---|---|---|---|
| Pix2Vox-F, 5 random views | +0.0991 [+0.0865, +0.1118] | +0.0524 [+0.0407, +0.0642] | -0.0248 [-0.0395, -0.0101] |
| UMIFormer, 5 random views | +0.0105 [+0.0024, +0.0177] | +0.0231 [+0.0155, +0.0300] | -0.0026 [-0.0103, +0.0050] |
| UMIFormer+, 5 random views | +0.0021 [-0.0040, +0.0074] | **+0.0148 [+0.0093, +0.0200]** | -0.0102 [-0.0162, -0.0042] |
| farthest-angle views + UMIFormer+ | -0.0109 [-0.0171, -0.0058] | +0.0017 [-0.0041, +0.0069] | -0.0232 [-0.0294, -0.0171] |

The bold entry is the pre-registered primary comparison; the farthest-angle
utility comparison was a registered secondary; the rest were computed after
the final evaluation and are post hoc. CPU-adjusted utility charges the
measured CPU time of the controller and the chosen reconstructor, from one
machine's timing profile.

## 7. Discussion

**What the system achieves.** Against every reconstructor used in the standard
way it delivers more reconstruction quality per unit of reconstructor
compute, and against Pix2Vox-F and UMIFormer it is also more accurate.
Against UMIFormer+ the gain is compute rather than accuracy: the router sends
about a fifth of reconstructions to the much cheaper Pix2Vox-F where little
is lost.

**What it does not.** The learned view policy's advantage over a geometric
heuristic did not survive the move to unseen objects: on this data, most of
the value of choosing views comes from spreading them out, which a
one-line rule does as well. The router's value is the cost trade-off; with
cost removed (lambda = 0) it adds almost nothing over always running
UMIFormer+. And the controller is not free: ResNet-50 features for five views
take about 0.4 s on a CPU, roughly 0.03 utility -- twice what the system gains
over standard UMIFormer+ -- so the benefit depends on charging only
reconstructor compute or on running the controller on a GPU (about 0.06 s per
object on a T4).

**Why.** The view-selection signal on this data is small: a perfect choice
among 24 sampled five-view sets is worth about 0.04 IoU over random views, and
the views that matter for one reconstructor barely predict those for another
(rank correlation +0.17).
Routing has more headroom (0.033 utility), but most of it is per object and
hard to predict from five images.

## 8. Limitations and future work

* One training run of the policy; claims about training need independent seeds.
* A fixed budget of five views. Stopping early would save controller and
  acquisition cost -- three views instead of five would save roughly 0.19 s of
  CPU feature extraction -- and is untested.
* CPU-adjusted conclusions depend on one machine's timing profile.
* Synthetic renderings of 13 ShapeNet categories at 32^3 resolution.
* A cheaper controller (reusing a reconstructor's own image encoder instead of
  a separate ResNet-50), a router trained on measured rather than declared
  costs, and real captured images are the natural next steps.

## Reproducibility

The benchmark, its proof and a script that re-derives it are in
[BENCHMARK.md](BENCHMARK.md) and `verify_benchmark.py`; the complete record of
decisions and intermediate results is [HANDOFF.md](HANDOFF.md).

## References

1. C. B. Choy, D. Xu, J. Gwak, K. Chen, S. Savarese. 3D-R2N2: A unified
   approach for single and multi-view 3D object reconstruction. ECCV 2016.
2. A. X. Chang et al. ShapeNet: An information-rich 3D model repository.
   arXiv:1512.03012, 2015.
3. H. Xie, H. Yao, X. Sun, S. Zhou, S. Zhang. Pix2Vox: Context-aware 3D
   reconstruction from single and multi-view images. ICCV 2019.
4. Z. Zhu, L. Yang, N. Li, C. Jiang, Y. Liang. UMIFormer: Mining the
   correlations between similar tokens for multi-view 3D reconstruction.
   ICCV 2023.
5. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. Proximal policy
   optimization algorithms. arXiv:1707.06347, 2017.
6. J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel. High-dimensional
   continuous control using generalized advantage estimation. ICLR 2016.
7. K. He, X. Zhang, S. Ren, J. Sun. Deep residual learning for image
   recognition. CVPR 2016.
8. R. Zeng, Y. Wen, W. Zhao, Y.-J. Liu. View planning in robot active vision:
   A survey of systems, algorithms, and applications. Computational Visual
   Media 6(3), 2020.
9. D. Jayaraman, K. Grauman. Look-ahead before you leap: End-to-end active
   recognition by forecasting the effect of motion. ECCV 2016.
10. S. Huang, S. Ontanon. A closer look at invalid action masking in policy
    gradient algorithms. FLAIRS 2022.
11. J. R. Rice. The algorithm selection problem. Advances in Computers 15,
    1976.
12. W. Kool, H. van Hoof, M. Welling. Buy 4 REINFORCE samples, get a baseline
    for free! ICLR Workshop on Deep RL Meets Structured Prediction, 2019.
