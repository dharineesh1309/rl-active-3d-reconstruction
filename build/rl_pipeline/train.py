"""
train.py
────────
Entry point for RL training of the joint view/backbone selection policy.

Usage
─────
  python train.py                          # fresh run
  python train.py --resume                 # auto-load latest checkpoint
  python train.py --dummy-vec              # sequential envs (no multiprocessing)
  python train.py --smoke-test             # 3 episodes per phase, quick sanity check
  python train.py --n-envs 4               # override Config.n_envs
  python train.py --dataset shapenet       # Choy ShapeNet (default)
  python train.py --dataset modelnet       # the original ModelNet set

  # carry a pre-model-head checkpoint forward into the widened policy
  python train.py --resume --load-view-head-only

Environment variable overrides (useful on Kaggle / Colab):
  SHAPENET_ROOT=/path/to/data python train.py

Use --dummy-vec if multiprocessing misbehaves in a notebook kernel. Backbones
always run on CPU inside the environment workers, so CUDA objects never cross a
multiprocessing Pipe.
"""

import argparse
import glob
import os

import torch

# ── CLI arguments ─────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--resume", action="store_true",
                    help="Continue from the latest RL checkpoint.")
parser.add_argument("--dummy-vec", action="store_true",
                    help="Use DummyVecEnv (sequential) instead of SubprocVecEnv.")
parser.add_argument("--smoke-test", action="store_true",
                    help="Run 3 episodes per phase for a quick sanity check.")
parser.add_argument("--n-envs", type=int, default=None,
                    help="Override Config.n_envs.")
parser.add_argument("--dataset", choices=["shapenet", "modelnet"], default="shapenet",
                    help="Which dataset to train on (default: shapenet).")
parser.add_argument("--dataset-root", type=str, default=None,
                    help="Override the dataset root.")
parser.add_argument("--limit-per-category", type=int, default=None,
                    help="Cap models per category. Useful for short runs.")
parser.add_argument("--backbone-device", default=None,
                    help="Device for the reconstruction backbones and the ResNet-50 "
                         "state encoder. Defaults to cpu with SubprocVecEnv (CUDA "
                         "objects cannot cross a multiprocessing Pipe) and to the "
                         "training device with --dummy-vec.")
parser.add_argument("--max-hours", type=float, default=None,
                    help="Stop cleanly after this many hours, saving a resumable "
                         "checkpoint. For capped environments such as Kaggle's "
                         "12-hour sessions.")
parser.add_argument("--load-view-head-only", action="store_true",
                    help="Load only the tensors that match, leaving the model head "
                         "freshly initialised. Use this to carry a checkpoint trained "
                         "before backbone selection existed into the widened policy.")
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────

from config import Config

cfg = Config()

if args.smoke_test:
    cfg.phase1_episodes = 3
    cfg.phase2_episodes = 3
    cfg.n_steps_per_env = 16
    cfg.checkpoint_every = 2
    cfg.log_every = 1
    print("[smoke-test] Overriding episode counts to 3.")

if args.n_envs:
    cfg.n_envs = args.n_envs

if args.dataset_root:
    cfg.dataset_root = args.dataset_root

if os.environ.get("DATASET_ROOT"):
    cfg.dataset_root = os.environ["DATASET_ROOT"]

cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device  : {cfg.device}")
print(f"Dataset : {args.dataset}")

# ── Check at least one backbone is available before doing anything slow ───────

from backbones import _candidates, available_backbone_names

backbone_names = available_backbone_names(cfg)
expected = [c.name for c in _candidates()]
if not backbone_names:
    raise FileNotFoundError(
        "No reconstruction backbone weights were found, so no reward can be "
        f"computed.\nThe registered backbones are {expected}; set the matching\n"
        "*_CKPT environment variable for each, or edit config.py."
    )
print(f"Backbones: {backbone_names}")

# A missing checkpoint is skipped with a notice, not an error, so a partial set
# trains happily with a smaller action space than intended and the run looks
# normal. That failure only shows up much later as a policy that cannot
# reproduce the measured backbone crossover, so say it loudly here.
missing = [n for n in expected if n not in backbone_names]
if missing:
    print(f"  WARNING: registered but not found: {missing}\n"
          f"  The model head will have {len(backbone_names)} actions instead of "
          f"{len(expected)}. A checkpoint trained now will NOT be compatible with\n"
          "  one trained against the full set, because the action indices differ.")
if len(backbone_names) == 1:
    print("  NOTE: only one backbone is available, so the model head has a single\n"
          "  legal action and backbone selection cannot actually be learned. Add a\n"
          "  second backbone for the selection head to mean anything.")

# ── Imports ───────────────────────────────────────────────────────────────────

from policy.view_policy import ViewPolicy
from training.parallel_envs import SubprocVecEnv, DummyVecEnv
from training.ppo_trainer import PPOTrainer
from env.view_recon_env import ViewReconEnv

# ── Load dataset ──────────────────────────────────────────────────────────────

print("Loading dataset ...")
if args.dataset == "shapenet":
    from dataloader_shapenet import build_shapenet

    dataset = build_shapenet(
        split="train",
        limit_per_category=args.limit_per_category,
    )
else:
    from dataloader import build_dataset

    dataset = build_dataset(
        dataset_root=cfg.dataset_root,
        categories=cfg.categories,
        image_size=(cfg.recon_img_size, cfg.recon_img_size),
        n_views=cfg.n_views,
        preload=False,
    )
print(f"Dataset size: {len(dataset)} objects")

# The action space must match the data. A subset or mirror with fewer views
# than Config.n_views would otherwise index renderings that do not exist, and
# only fail once an episode happened to select a high view index.
dataset_views = getattr(dataset, "n_views", None)
if dataset_views and dataset_views != cfg.n_views:
    print(f"  NOTE: dataset has {dataset_views} views/object but Config.n_views "
          f"is {cfg.n_views}; using {dataset_views}.")
    if dataset_views < max(cfg.phase2_view_budgets):
        raise ValueError(
            f"View budgets {cfg.phase2_view_budgets} exceed the {dataset_views} "
            f"views available per object. Use the full 24-view ShapeNet release "
            f"for training, or lower the budgets."
        )
    cfg.n_views = dataset_views

# ── Environment factory ───────────────────────────────────────────────────────

# Backbones cannot live on the GPU inside SubprocVecEnv workers: CUDA tensors
# cannot be pickled across a multiprocessing Pipe. With --dummy-vec the envs run
# in this process, so the GPU is available and is much faster -- UMIFormer is a
# ViT and is painfully slow on CPU.
if args.backbone_device:
    backbone_device = args.backbone_device
elif args.dummy_vec:
    backbone_device = cfg.device
else:
    backbone_device = "cpu"

if backbone_device != "cpu" and not args.dummy_vec:
    raise ValueError(
        f"--backbone-device {backbone_device} needs --dummy-vec: CUDA objects "
        "cannot be pickled into SubprocVecEnv workers. Either add --dummy-vec "
        "(one process, GPU backbones - usually faster when a GPU is present) or "
        "drop --backbone-device to keep the workers on CPU."
    )
print(f"Backbone device: {backbone_device}")


def make_env_fn(dataset, cfg, view_budget, seed):
    """
    Build one ViewReconEnv inside the worker process.

    The backbones are constructed here rather than passed in, because torch
    models cannot be pickled across a multiprocessing Pipe. Each worker
    therefore holds its own CPU copy.
    """
    def _make():
        import random

        import numpy as np

        from backbones import load_backbones

        random.seed(seed)
        np.random.seed(seed)
        return ViewReconEnv(
            dataset=dataset,
            backbones=load_backbones(cfg, device=backbone_device),
            view_budget=view_budget,
            lambda_cost=cfg.cost_lambda,
            n_views=cfg.n_views,
            device=backbone_device,
            shaping_coef=cfg.shaping_coef,
            seed_initial_view=cfg.seed_initial_view,
            gamma=cfg.gamma,
            phase2_budgets=cfg.phase2_view_budgets,
            # One shared seed across ALL envs, so every env draws the same
            # object and budget each episode while its ACTIONS still differ.
            # That is what makes the group-mean baseline cancel object
            # difficulty exactly. Deliberately not seed+i.
            group_sync_seed=(1234 if cfg.group_baseline else None),
        )

    return _make


env_fns = [
    make_env_fn(dataset, cfg, cfg.phase1_view_budget, seed=i)
    for i in range(cfg.n_envs)
]

VecEnvClass = DummyVecEnv if args.dummy_vec else SubprocVecEnv
print(f"VecEnv : {VecEnvClass.__name__} x {cfg.n_envs} workers")

# group_baseline needs a GROUP. With one env there is nothing to centre against,
# so center_by_group() returns early and the mechanism is a silent no-op -- the
# run would look entirely normal while the variance reduction it depends on was
# absent. Fail instead.
if cfg.group_baseline and cfg.n_envs < 2:
    raise SystemExit(
        f"Config.group_baseline is True but n_envs = {cfg.n_envs}. Group-relative "
        "baselines centre each episode against the OTHER envs running the same "
        "object, so they need at least 2 (8 is the default). Either raise "
        "--n-envs or set group_baseline = False."
    )
if cfg.group_baseline:
    print(f"  group baselines ON: all {cfg.n_envs} envs share each episode's "
          f"object and budget; advantages centred leave-one-out")
vec_env = VecEnvClass(env_fns)

# ── Build RL policy ───────────────────────────────────────────────────────────

policy = ViewPolicy(
    n_views=cfg.n_views,
    n_backbones=len(backbone_names),
).to(cfg.device)
print(f"Policy parameters: {sum(p.numel() for p in policy.parameters()):,}")

# ── Optionally carry an older, narrower checkpoint forward ────────────────────

if args.load_view_head_only:
    from utils.checkpoint import CheckpointManager

    state = CheckpointManager(cfg.checkpoint_dir).load_latest()
    if state is None:
        print("  --load-view-head-only: no checkpoint found, starting fresh.")
    else:
        report = policy.load_compatible(state["policy"])
        print(f"  Carried {len(report['loaded'])} tensors forward from episode "
              f"{state.get('total_episodes', '?')}; "
              f"{len(report['left_fresh'])} left freshly initialised.")
        if report["skipped"]:
            print(f"  Skipped (shape or name changed): {report['skipped']}")

# ── Clear stale RL checkpoints on a fresh run ─────────────────────────────────

if not args.resume and not args.load_view_head_only:
    # Never delete a 'final' checkpoint. It is usually the only artifact of a
    # completed run, and a bare `python train.py` used to wipe it silently.
    all_ckpts = glob.glob(os.path.join(cfg.checkpoint_dir, "ckpt_*.pt"))
    stale = [f for f in all_ckpts
             if not os.path.basename(f).startswith("ckpt_final")]
    preserved = [f for f in all_ckpts
                 if os.path.basename(f).startswith("ckpt_final")]
    for f in stale:
        os.remove(f)
    if stale:
        print(f"Removed {len(stale)} stale RL checkpoint(s).")
    for f in preserved:
        print(f"Kept {os.path.basename(f)} (final checkpoints are never deleted).")
    print("Starting fresh training run.")

# ── Train ─────────────────────────────────────────────────────────────────────

trainer = PPOTrainer(policy, vec_env, cfg, resume=args.resume,
                     max_hours=args.max_hours)
try:
    trainer.train()
finally:
    vec_env.close()
    trainer.logger.close()
    print("Done.")
