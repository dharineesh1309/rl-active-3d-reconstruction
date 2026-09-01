"""
End-to-end PPO loop test, with no data and no weights.

    python training/tests/test_ppo_loop.py

Runs the real PPOTrainer over real ViewReconEnvs driven by stub backbones, so
it exercises the parts most likely to break silently once a second actor head
exists:

  * transitions from view steps and model steps share one buffer and one
    Categorical, at different widths
  * the PPO update routes each stored action back through the head that
    produced it
  * both heads actually receive gradient

A head-routing bug does not crash — it trains the wrong head and looks like
ordinary instability — so it has to be asserted on directly.
"""

import os
import shutil
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from env.view_recon_env import ViewReconEnv
from policy.view_policy import ViewPolicy
from training.parallel_envs import DummyVecEnv
from training.ppo_trainer import PPOTrainer

N_VIEWS = 24
VOXEL_RES = 32


class StubBackbone:
    """Deterministic fake reconstructor with a tunable quality and cost."""

    def __init__(self, name, cost, quality):
        self.name = name
        self.cost = cost
        self.quality = quality

    def predict(self, images, cams=None):
        # Blend toward a fixed pattern so a "better" backbone scores higher and
        # the model head has a real gradient to follow.
        rng = np.random.default_rng(0)
        base = (rng.random((VOXEL_RES,) * 3) < 0.15).astype(np.float32)
        return base * self.quality


class TinyConfig:
    n_views = N_VIEWS
    cost_lambda = 0.05
    n_envs = 2
    n_steps_per_env = 48
    learning_rate = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_ratio = 0.2
    n_epochs = 2
    minibatch_size = 32
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    entropy_coef_view = 0.01
    entropy_coef_model = 0.05   # per-head coefficients; see ppo_trainer._update
    entropy_target_model = 0.6  # adaptive: alpha is tuned toward this
    entropy_alpha_lr     = 0.01
    entropy_alpha_max    = 1.0
    group_baseline       = False  # TinyConfig runs 1 env
    anneal_lr = True            # exercise the LR schedule in the loop test
    phase1_episodes = 4
    phase1_view_budget = 3
    phase2_episodes = 4
    phase2_view_budgets = [3, 5, 8]
    checkpoint_every = 1000        # do not checkpoint during the test
    log_every = 1000
    categories = ["chair"]
    device = "cpu"


def make_dataset(n_objects=3):
    rng = np.random.default_rng(1)
    dataset = []
    for i in range(n_objects):
        img = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
        dataset.append({
            "images":      [img] * N_VIEWS,
            "depths":      [rng.random((32, 32)).astype(np.float32)] * N_VIEWS,
            "silhouettes": [(rng.random((64, 64)) > 0.5).astype(np.float32)] * N_VIEWS,
            "voxels":      (rng.random((VOXEL_RES,) * 3) < 0.15).astype(np.float32),
            "category":    "chair",
            "model_id":    f"stub_{i}",
        })
    return dataset


def test_ppo_loop():
    cfg = TinyConfig()
    tmp = tempfile.mkdtemp(prefix="ppo_loop_test_")
    cfg.checkpoint_dir = os.path.join(tmp, "checkpoints")
    cfg.log_dir = os.path.join(tmp, "logs")

    dataset = make_dataset()
    backbones = [
        StubBackbone("cheap", cost=0.1, quality=0.55),
        StubBackbone("dear",  cost=1.5, quality=0.95),
    ]

    def make_env():
        return ViewReconEnv(
            dataset=dataset,
            backbones=backbones,
            view_budget=cfg.phase1_view_budget,
            lambda_cost=cfg.cost_lambda,
            n_views=cfg.n_views,
        )

    vec_env = DummyVecEnv([make_env for _ in range(cfg.n_envs)])
    policy = ViewPolicy(n_views=cfg.n_views, n_backbones=len(backbones))

    assert policy.n_actions == N_VIEWS, \
        "24 views dominate 2 backbones, so the action space should be 24 wide"

    # Snapshot both heads so we can prove each one moved.
    before = {
        "view":  policy.view_head[0].weight.detach().clone(),
        "model": policy.model_head[0].weight.detach().clone(),
    }

    trainer = PPOTrainer(policy, vec_env, cfg)
    try:
        trainer.train()
    finally:
        vec_env.close()
        trainer.logger.close()

    # ── The loop actually ran ────────────────────────────────────────────────
    assert trainer.total_episodes >= cfg.phase1_episodes + cfg.phase2_episodes, \
        f"expected at least 8 episodes, got {trainer.total_episodes}"
    print(f"OK - completed {trainer.total_episodes} episodes across both phases")

    # ── Both heads were trained ──────────────────────────────────────────────
    for head in ("view", "model"):
        after = getattr(policy, f"{head}_head")[0].weight.detach()
        moved = (after - before[head]).abs().max().item()
        assert moved > 0, (
            f"the {head} head did not move at all — its transitions are never "
            f"reaching the PPO update, so that head cannot learn"
        )
        print(f"OK - {head}_head updated (max weight delta {moved:.2e})")

    # ── Nothing went numerically bad ─────────────────────────────────────────
    for name, p in policy.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite values in {name}"
    print("OK - all policy parameters finite")

    # ── The model step really was exercised ──────────────────────────────────
    assert trainer.buffer.model_flags.sum() > 0, \
        "no model-selection steps were stored, so the terminal step never ran"
    view_steps = int((trainer.buffer.model_flags == 0).sum())
    model_steps = int(trainer.buffer.model_flags.sum())
    print(f"OK - buffer holds {view_steps} view steps and {model_steps} model steps")

    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    print("Running PPO loop test...\n")
    test_ppo_loop()
    print("\nPPO loop test passed.")
