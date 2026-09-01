"""
Group-relative baselines must cancel object difficulty, not the signal.

The measured bottleneck is signal-to-noise: within-object spread from view
choice is 0.0266 IoU, between-object spread is 0.0621, and even with the critic
at 0.88 explained variance the residual noise outweighs the signal about 4.4 to
1. Running every parallel env on the SAME object and centring each advantage
against the others makes the difficulty term cancel exactly.

Two things have to hold, and both fail silently if they break:

  * the envs must actually draw the same object and budget each episode, or the
    baseline subtracts across DIFFERENT objects and removes the signal instead
    of the noise;
  * centring must not touch `returns`, which are the critic's regression target
    and must stay unbiased estimates of V.

    python training/tests/test_group_baseline.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from training.rollout_buffer import RolloutBuffer

N_VIEWS, N_ENVS, N_STEPS = 24, 4, 6


def _buf(rewards, values):
    """rewards/values: (N_STEPS, N_ENVS)."""
    b = RolloutBuffer(n_steps=N_STEPS, n_envs=N_ENVS, n_views=N_VIEWS,
                      n_actions=N_VIEWS, device="cpu")
    for t in range(N_STEPS):
        b.add(
            obs={"coverage_grid": np.zeros((N_ENVS, 32, 32, 32), np.float32),
                 "image_features": np.zeros((N_ENVS, 512), np.float32),
                 "view_mask": np.zeros((N_ENVS, N_VIEWS), np.float32),
                 "is_model_step": np.zeros((N_ENVS, 1), np.float32),
                 "budget": np.zeros((N_ENVS, 2), np.float32)},
            actions=np.zeros(N_ENVS, np.int64),
            log_probs=np.zeros(N_ENVS, np.float32),
            values=values[t].astype(np.float32),
            rewards=rewards[t].astype(np.float32),
            dones=np.array([t == N_STEPS - 1] * N_ENVS),
            action_masks=np.ones((N_ENVS, N_VIEWS), np.float32),
        )
    b.compute_gae(last_values=np.zeros(N_ENVS, np.float32),
                  last_dones=np.ones(N_ENVS, np.float32),
                  gamma=0.99, gae_lambda=0.95)
    return b


def test_shared_difficulty_cancels():
    """A per-object offset common to the whole group must vanish."""
    rng = np.random.default_rng(0)
    per_env = rng.normal(0, 0.03, N_ENVS)        # the view-choice signal
    rewards = np.zeros((N_STEPS, N_ENVS)); rewards[-1] = per_env
    values = np.zeros((N_STEPS, N_ENVS))

    base = _buf(rewards, values); base.center_by_group()
    a0 = base.advantages.copy()

    # Now add a large shared difficulty offset -- the SAME for every env,
    # because they all face the same object.
    OFF = 0.62                                    # 10x the measured spread
    rewards2 = rewards.copy(); rewards2[-1] += OFF
    shifted = _buf(rewards2, values); shifted.center_by_group()

    delta = np.abs(shifted.advantages - a0).max()
    assert delta < 1e-5, (
        f"a shared offset of {OFF} changed centred advantages by {delta:.6f}; "
        "object difficulty is NOT cancelling")
    print(f"OK - shared difficulty cancels (offset {OFF} -> max delta {delta:.2e})")


def test_signal_survives():
    """Centring must remove the common term, not the per-env differences."""
    rng = np.random.default_rng(1)
    per_env = rng.normal(0, 0.03, N_ENVS)
    rewards = np.zeros((N_STEPS, N_ENVS)); rewards[-1] = per_env + 0.62
    values = np.zeros((N_STEPS, N_ENVS))
    b = _buf(rewards, values)
    before = b.advantages[-1].copy()
    b.center_by_group()
    after = b.advantages[-1]
    # Ordering across envs must be preserved: that ordering IS the signal.
    assert np.array_equal(np.argsort(before), np.argsort(after)), (
        "centring changed the ranking of envs; it is destroying the signal")
    assert after.std() > 0.01, f"signal flattened to std {after.std():.4f}"
    print(f"OK - per-env signal survives (std {before.std():.4f} -> {after.std():.4f}, "
          f"ranking preserved)")


def test_returns_untouched():
    """`returns` train the critic and must stay unbiased."""
    rng = np.random.default_rng(2)
    rewards = np.zeros((N_STEPS, N_ENVS)); rewards[-1] = rng.normal(0.5, 0.1, N_ENVS)
    values = np.full((N_STEPS, N_ENVS), 0.3)
    b = _buf(rewards, values)
    ret_before = b.returns.copy()
    b.center_by_group()
    assert np.array_equal(ret_before, b.returns), (
        "center_by_group modified returns; the critic would regress to a "
        "group-centred target, which is not V")
    print("OK - returns left untouched (critic target stays unbiased)")


def test_leave_one_out_is_unbiased():
    """Each env's baseline must exclude its own advantage."""
    rewards = np.zeros((N_STEPS, N_ENVS)); rewards[-1] = np.array([1.0, 0.0, 0.0, 0.0])
    b = _buf(rewards, np.zeros((N_STEPS, N_ENVS)))
    raw = b.advantages[-1].copy()
    b.center_by_group()
    got = b.advantages[-1]
    n = N_ENVS
    want = raw - (raw.sum() - raw) / (n - 1)
    assert np.allclose(got, want, atol=1e-6), f"expected {want}, got {got}"
    # A full-group mean would give a different, slightly biased answer.
    full = raw - raw.mean()
    assert not np.allclose(got, full, atol=1e-6), "using the full group mean, not leave-one-out"
    print("OK - leave-one-out baseline (excludes self, unlike a plain group mean)")


if __name__ == "__main__":
    print("Running group-baseline tests...\n")
    test_shared_difficulty_cancels()
    test_signal_survives()
    test_returns_untouched()
    test_leave_one_out_is_unbiased()
    print("\nGroup-baseline tests passed.")
