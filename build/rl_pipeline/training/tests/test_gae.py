"""
GAE must respect episode boundaries.

This guards the defect that wasted a 30,274-episode run. `compute_gae` read
`dones[t+1]` where it needed `dones[t]` -- `dones[t]` is what vec_env.step()
returned for action t, "the episode ended after this action".

On the project's B+1 horizon the off-by-one moved the boundary one step early:

  * step B, the last VIEW step, was treated as terminal, so the terminal reward
    never propagated back to any view step. Every view step carried a negative
    advantage in episodes that earned a large positive reward, and the view head
    trained on a signal containing no information about which view it picked.
  * step B+1, the model step and the real terminal, was treated as
    non-terminal, so it bootstrapped V from the first state of the NEXT episode.

Nothing crashed. Training completed, wrote checkpoints and logged plausible
rewards for 30k episodes.

    python training/tests/test_gae.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from training.rollout_buffer import RolloutBuffer

N_VIEWS = 24
V = 0.5          # constant value estimate, so expectations stay hand-checkable
GAMMA = 0.99
LAM = 0.95


def _buffer_with_one_episode(n_steps=7, terminal_at=5, reward=1.0):
    """B=5 episode: 5 view steps + 1 model step, then the next episode starts."""
    buf = RolloutBuffer(n_steps=n_steps, n_envs=1, n_views=N_VIEWS,
                        n_actions=N_VIEWS, device="cpu")
    for t in range(n_steps):
        done = (t == terminal_at)
        buf.add(
            obs={"coverage_grid": np.zeros((1, 32, 32, 32), np.float32),
                 "image_features": np.zeros((1, 512), np.float32),
                 "view_mask": np.zeros((1, N_VIEWS), np.float32),
                 "is_model_step": np.zeros((1, 1), np.float32),
                 "budget": np.zeros((1, 2), np.float32)},
            actions=np.zeros(1, np.int64),
            log_probs=np.zeros(1, np.float32),
            values=np.full(1, V, np.float32),
            rewards=np.array([reward if done else 0.0], np.float32),
            dones=np.array([done]),
            action_masks=np.ones((1, N_VIEWS), np.float32),
        )
    buf.compute_gae(last_values=np.array([V], np.float32),
                    last_dones=np.array([0.0], np.float32),
                    gamma=GAMMA, gae_lambda=LAM)
    return buf


def test_terminal_does_not_bootstrap():
    buf = _buffer_with_one_episode()
    ret = float(buf.returns[5, 0])
    assert abs(ret - 1.0) < 1e-5, (
        f"terminal return {ret:.4f}, expected 1.0. A value larger by about "
        f"gamma*V ({GAMMA * V:.4f}) means the terminal step bootstrapped into "
        "the next episode.")
    print("OK - terminal step does not bootstrap across the episode boundary")


def test_reward_reaches_every_view_step():
    buf = _buffer_with_one_episode()
    adv = buf.advantages[:6, 0]
    assert all(a > 0 for a in adv), (
        f"advantages {[round(float(a), 4) for a in adv]}: the terminal reward "
        "is not reaching the view steps, so the view head cannot learn.")
    # Discounting means credit decays backwards from the terminal step.
    assert adv[0] < adv[4] < adv[5], "advantages should grow towards the terminal step"
    print(f"OK - reward reaches all view steps "
          f"({adv[0]:.3f} .. {adv[5]:.3f}, increasing)")


def test_next_episode_is_isolated():
    """Credit must not leak backwards across a boundary into the prior episode."""
    a = _buffer_with_one_episode(reward=1.0)
    b = _buffer_with_one_episode(reward=50.0)
    # Only the terminal reward differs; steps AFTER the boundary must match.
    assert abs(float(a.advantages[6, 0]) - float(b.advantages[6, 0])) < 1e-5, (
        "a change to this episode's terminal reward altered the next episode's "
        "advantage -- the boundary is not isolating them")
    print("OK - episodes are isolated from each other")


if __name__ == "__main__":
    print("Running GAE boundary tests...\n")
    test_terminal_does_not_bootstrap()
    test_reward_reaches_every_view_step()
    test_next_episode_is_isolated()
    print("\nGAE tests passed.")
