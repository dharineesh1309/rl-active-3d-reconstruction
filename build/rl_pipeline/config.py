import os

_HERE = os.path.dirname(os.path.abspath(__file__))


class Config:
    # ── Dataset ───────────────────────────────────────────────────────────────
    # Path to the dataset_preprocessed/ folder.
    # Override via --dataset-root CLI arg or $DATASET_ROOT env var in train.py.
    dataset_root = r"C:\Users\<user>\Desktop\ML\RL\ModelNet_out_2"
    categories   = ["bed", "chair", "desk", "sofa", "table"]

    # ── Reconstructor ─────────────────────────────────────────────────────────
    # Path to the pretrained EncoderDecoder checkpoint (.pth file).
    # This model is loaded inside each environment worker (CPU) to score
    # reconstructions and compute the IoU reward.
    reconstructor_ckpt = r"C:\Users\<user>\Desktop\ML\3D reconstruction\ep78_encoder_decoder_64_frz(iou0.4893_loss0.2232).pth"
    feature_dim        = 512   # encoder output / decoder input dimension
    voxel_size         = 64    # reconstructor output grid: (64, 64, 64)
    recon_img_size     = 256   # input image size expected by the reconstructor

    # ── Backbones the RL agent selects between ────────────────────────────────
    # Each may be overridden by the matching environment variable, which is how
    # the GPU box points at its own copies without editing this file.
    # A backbone whose file is missing drops out of the action space at startup.
    pix2vox_f_ckpt = os.path.join(_HERE, "..", "Pix2Vox-F-ShapeNet.pth")   # $PIX2VOX_F_CKPT
    umiformer_ckpt = os.path.join(_HERE, "..", "UMIFormer-ShapeNet.pth")  # $UMIFORMER_CKPT
    umiformer_plus_ckpt = os.path.join(_HERE, "..", "UMIFormerPlus-ShapeNet.pth")  # $UMIFORMER_PLUS_CKPT

    # ── Environment ───────────────────────────────────────────────────────────
    n_views      = 24          # 3D-R2N2 renders 24 views per object
    # Prices the backbone choice: utility = IoU - cost_lambda * backbone.cost.
    #
    # COST DEFINITION v1 (frozen 2026-10-01). backbone.cost is measured CPU
    # wall-clock seconds for one 5-view prediction on the development machine:
    # Pix2Vox-F 0.51 s, UMIFormer 1.28 s, UMIFormer+ 1.28 s (same network).
    # It does not include image acquisition, ResNet features, policy or router
    # inference; those are reported separately, not priced.
    #
    # cost_lambda is IoU per second, chosen to keep the operating point every
    # earlier result used. Those used undocumented units (Pix2Vox-F 0.09,
    # UMIFormers 0.938) with lambda 0.07: a penalty gap of 0.07 * 0.848 =
    # 0.0594 IoU between Pix2Vox-F and the UMIFormers. 0.0771 * 0.77 s gives
    # the same gap, so every routing decision and every reported difference is
    # unchanged; absolute utilities are 0.033 lower than in the old units.
    #
    # The operating point is declared, not a unique optimum. It was chosen from
    # the 3-category measurement in EXPERIMENTS.md §10 (airplane,
    # car, chair x budgets 1/3/5/8). Per-cell crossovers run from 0.0146 to
    # 0.1348, so the live window is wide, but most of it is uninteresting: what
    # matters is where the optimal choice depends on the OBJECT and not just the
    # budget, because that is the part a budget-only lookup table cannot do and
    # a learned policy can.
    #
    # Numbers below are in the old units (lambda 0.07).
    #
    # At 0.07 the one-view choice splits by category -- aeroplane wants
    # UMIFormer, car and chair want Pix2Vox-F -- and all three backbones win
    # somewhere:
    #
    #        B=1   aeroplane umiformer   car pix2vox_f   chair pix2vox_f
    #        B=3   aeroplane umiformer   car pix2vox_f   chair umiformer
    #        B=5   aeroplane umi+        car pix2vox_f   chair umi+
    #        B=8   aeroplane umi+        car pix2vox_f   chair umi+
    #
    # 0.07 is also the safest point in that region: it sits 0.017 from the
    # nearest crossover, where 0.05 is 0.0013 from one and 0.09 is 0.0027 from
    # one -- both close enough that measurement noise flips the answer.
    #
    # Note the UMIFormer / UMIFormer+ half of the decision is independent of
    # this value: same architecture, same cost, so it is decided purely on
    # quality and survives even at lambda = 0.
    #
    # The old RGB-D environment also charges cost_lambda per acquired view, so
    # its per-view price moves from 0.07 to 0.0771; that pipeline is historical.
    cost_lambda  = 0.0771

    # ── Parallelism ───────────────────────────────────────────────────────────
    n_envs = 8                 # number of parallel environment workers
                               # drop to 4 if OOM, or use --dummy-vec to go sequential

    # ── Rollout buffer ────────────────────────────────────────────────────────
    n_steps_per_env = 256      # total buffer size = n_envs × n_steps_per_env = 2 048

    # ── PPO ───────────────────────────────────────────────────────────────────
    learning_rate   = 3e-4
    gamma           = 0.99
    gae_lambda      = 0.95
    clip_ratio      = 0.2
    n_epochs        = 4
    minibatch_size  = 64
    value_loss_coef = 0.5
    max_grad_norm   = 0.5

    # ── Entropy ───────────────────────────────────────────────────────────────
    # Applied PER HEAD. A single coefficient over a pooled mean cannot regulate
    # a 24-way view distribution and a 3-way backbone distribution at once: the
    # view head is B of every B+1 steps, so it dominates the average and the
    # backbone head drifts wherever its own gradient takes it.
    #
    # Measured after 30,274 episodes with one pooled coefficient: view head at
    # 90% of uniform entropy (barely learning), backbone head at 15% of its
    # maximum (collapsed to one action in 96.4% of episodes). Opposite failures
    # from the same knob.
    entropy_coef_view  = 0.01  # unchanged; matches the report
    entropy_coef_model = 0.05  # STARTING value only; adapted below

    # Auto-tune the model head's coefficient to hold its entropy at a target,
    # rather than guessing a constant (SAC's temperature trick):
    #
    #     alpha <- clip(alpha + lr * (H_target - H_observed), 0, alpha_max)
    #
    # A fixed 0.05 was measured to be far too weak. The model head settled at
    # H = 0.026 of a 1.099 maximum, i.e. roughly [0.995, 0.003, 0.002], so
    # Pix2Vox-F was sampled about once per 400 episodes. Cars are ~24% of the
    # data, so across 30,000 episodes the head saw Pix2Vox-F-on-a-car about
    # NINE times -- and Pix2Vox-F is the optimal choice for every car cell. No
    # conditional rule is learnable from nine samples.
    #
    # H_target = 0.6 is ~55% of ln(3), roughly [0.75, 0.15, 0.10]: about 10% of
    # episodes explore, giving ~720 car-with-Pix2Vox-F samples instead of 9.
    # Evaluation takes argmax, so the residual stochasticity costs nothing at
    # inference.
    entropy_target_model = 0.6
    entropy_alpha_lr     = 0.01
    entropy_alpha_max    = 1.0

    # Linearly anneal the learning rate to zero over the run.
    #
    # Standard PPO practice (CleanRL, Stable-Baselines3 both default to it) and
    # simply missing here -- the optimiser was built once at 3e-4 and never
    # scheduled. With the critic eventually explaining 88% of the return, the
    # residual advantage is mostly noise, and advantage normalisation rescales
    # that noise to unit variance regardless; a constant step size then keeps
    # taking full-size noise-driven steps. Entropy bottomed at 2.326 around
    # episode 12,800 and drifted back to 2.445 by 24,400.
    anneal_lr = True

    # ── Credit assignment for view selection ─────────────────────────────────
    # Potential-based reward shaping (Ng, Harada & Russell 1999). The episode
    # reward is terminal-only, so all B view steps share one undifferentiated
    # signal and PPO cannot tell which view was the good one. Measured: after
    # 30,274 episodes the view policy sat at 90% of uniform entropy, while a
    # perfect planner would gain +0.0402 IoU per episode -- the signal exists
    # and was not being found.
    #
    # Shaping adds  gamma*Phi(s') - Phi(s)  to each step. Those terms telescope,
    # so the return is unchanged up to a constant and **the set of optimal
    # policies is provably identical** -- densifying credit here cannot make the
    # agent optimise something else. GenNBV (CVPR 2024) trains its NBV policy
    # with exactly this signal, the per-step change in coverage ratio.
    #
    # Phi = fraction of the 32**3 coverage grid occupied. Free to compute, and
    # meaningful only because the grid became a real back-projection (precision
    # 0.826 vs 0.373 chance) -- with the old image-plane stencil this potential
    # would have been noise.
    shaping_coef = 1.0        # 0.0 disables shaping and restores sparse reward

    # Run every parallel env on the SAME object and budget, then centre each
    # advantage against the other envs in the group (leave-one-out).
    #
    # This is aimed at the measured bottleneck. Within-object spread from view
    # choice is 0.0266 IoU; between-object spread is 0.0621. Even with the
    # critic at 0.88 explained variance the residual noise outweighs the signal
    # about 4.4 to 1, so the view head has been learning from mostly noise.
    # A shared object makes the difficulty term cancel exactly instead of
    # approximately -- the same trick GRPO uses, and it costs nothing at
    # runtime: the same n_envs episodes, just coordinated.
    #
    # Requires n_envs > 1. The trainer raises if the envs ever desync.
    group_baseline = True

    # Give the agent one random view before its first decision.
    #
    # Without it the first state is all zeros -- empty coverage, zero image
    # features, zero view mask -- so V(s_0) is identical for every object and
    # that advantage carries the whole between-object spread (0.0621 IoU) on top
    # of a view-choice signal of 0.0266. Seeding is standard in the NBV
    # literature and costs one view from the budget.
    seed_initial_view = True

    # ── Training phases ───────────────────────────────────────────────────────
    phase1_episodes    = 10_000
    phase1_view_budget = 4   # 4 + seeded view = 5 views used

    phase2_episodes     = 20_000
    # {2,4,7} CHOICES, which with seed_initial_view become 3/5/8 views used.
    #
    # The seeded view is real and counts toward reconstruction, so budgets of
    # {3,5,8} produced 4/6/9 views -- and the UMIFormer -> UMIFormer+ crossover
    # sits between 3 and 4 views. That pushed every budget past it and made one
    # backbone correct almost everywhere, flattening exactly the decision this
    # project exists to study. Set seed_initial_view = False and these should go
    # back to [3, 5, 8].
    phase2_view_budgets = [2, 4, 7]

    # ── Checkpointing ─────────────────────────────────────────────────────────
    # Overridable so a capped environment (Kaggle) can point these at a
    # directory that survives as notebook output.
    checkpoint_dir   = os.environ.get("CHECKPOINT_DIR", "checkpoints")
    checkpoint_every = 500     # save every N episodes

    # ── Logging ───────────────────────────────────────────────────────────────
    log_dir   = os.environ.get("LOG_DIR", "logs")
    log_every = 100            # print + CSV flush every N episodes

    # ── Device (RL policy only) ───────────────────────────────────────────────
    # The reconstructor always runs on CPU inside environment workers.
    device = "cuda"
