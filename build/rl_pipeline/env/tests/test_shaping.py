"""
Potential-based reward shaping must not change what the agent is optimising.

The episode reward is terminal-only, so all B view steps share one
undifferentiated number and PPO cannot tell which view was the good one.
Measured: after 30,274 episodes the view policy sat at 90% of uniform entropy
while a perfect planner would gain +0.0402 IoU per episode.

Shaping adds `gamma*Phi(s') - Phi(s)` per step, with `Phi(terminal) = 0`. Ng,
Harada & Russell (1999) proved that this exact form leaves the set of optimal
policies unchanged, whatever Phi is. GenNBV (CVPR 2024) trains its NBV policy on
the same signal -- the per-step change in coverage ratio.

The guarantee depends on the shaping terms summing to something the agent cannot
influence. They telescope to `-Phi(s_0)`, which is fixed once the episode's
object and seed view are drawn. **Drop the terminal `-Phi(s_B)` term and they
sum to `Phi(s_B) - Phi(s_0)` instead, which the agent CAN influence** -- that
would silently change the objective to "maximise final coverage", a correlate of
IoU rather than IoU itself. This test is what catches that.

    python env/tests/test_shaping.py
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))

from env.view_recon_env import ViewReconEnv

N_VIEWS = 8
BUDGET = 3
GAMMA = 0.99


class _Backbone:
    """Fixed-output stand-in: keeps the terminal IoU identical across runs."""
    name = "mock"
    cost = 0.0

    def predict(self, images, cams=None):
        g = np.zeros((32, 32, 32), dtype=np.float32)
        g[8:24, 8:24, 8:24] = 1.0
        return g


class _Dataset:
    """One object, deterministic, with camera metadata so coverage is geometric."""
    n_views = N_VIEWS

    def __init__(self):
        vox = np.zeros((32, 32, 32), np.float32)
        vox[10:22, 10:22, 10:22] = 1.0
        from dataloader import _depth_from_voxels
        from PIL import Image
        cams, depths, sils, imgs = [], [], [], []
        for i in range(N_VIEWS):
            az, el = i * (360.0 / N_VIEWS), 15.0
            d = _depth_from_voxels(vox, az, el)
            cams.append({"azimuth": az, "elevation": el})
            depths.append(d)
            sils.append((d < 1.0).astype(np.float32))
            imgs.append(Image.new("RGB", (137, 137)))
        self.item = {"images": imgs, "depths": depths, "silhouettes": sils,
                     "voxels": vox, "cams": cams, "category": "test"}

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return self.item


def _make(shaping):
    return ViewReconEnv(
        dataset=_Dataset(), backbones=[_Backbone()], view_budget=BUDGET,
        lambda_cost=0.07, n_views=N_VIEWS, device="cpu",
        shaping_coef=shaping, seed_initial_view=False, gamma=GAMMA,
    )


def _run(env, actions):
    """Return the DISCOUNTED return, which is what the theorem is about.

    The shaping terms telescope under discounting:

        sum_t gamma^t (gamma*Phi(s_{t+1}) - Phi(s_t)) = gamma^T*Phi(s_T) - Phi(s_0)

    and Phi(terminal) = 0, so the whole sum is exactly -Phi(s_0) for any gamma.
    An UNDISCOUNTED sum leaves a (gamma-1)*sum(Phi) residual -- about 1e-3 here
    -- which is a property of the arithmetic, not a defect in the shaping.
    """
    env.reset()
    total = 0.0
    for t, a in enumerate(actions):
        _, r, done, _, _ = env.step(a)
        total += (GAMMA ** t) * r
    assert done, "episode should have ended on the model step"
    return total


def test_shaping_terms_telescope():
    """Total shaped reward must equal sparse reward minus Phi(s_0), for ANY path."""
    paths = [[0, 1, 2, 0], [0, 4, 2, 0], [3, 5, 7, 0], [7, 6, 5, 0]]
    diffs = []
    for p in paths:
        sparse = _run(_make(0.0), p)
        shaped = _run(_make(1.0), p)
        diffs.append(shaped - sparse)
    spread = max(diffs) - min(diffs)
    assert spread < 1e-9, (
        f"shaped-minus-sparse varies by {spread:.6f} across action sequences "
        f"({[round(d, 6) for d in diffs]}). The shaping terms are not "
        "telescoping to an action-independent constant, so the optimal policy "
        "is NOT preserved -- most likely the terminal -Phi(s_B) term is missing.")
    print(f"OK - shaping is action-independent (offset {diffs[0]:+.6f}, "
          f"spread {spread:.2e})")


def test_shaping_is_dense():
    """Intermediate steps must carry non-zero reward, or nothing was gained."""
    env = _make(1.0)
    env.reset()
    rewards = [env.step(a)[1] for a in (0, 2, 4)]
    assert any(abs(r) > 1e-6 for r in rewards), (
        f"all view-step rewards are zero ({rewards}); shaping is not active and "
        "credit assignment is unchanged")
    print(f"OK - view steps now carry reward ({[round(r, 4) for r in rewards]})")


def test_sparse_still_available():
    """shaping_coef = 0 must reproduce the original sparse reward exactly."""
    env = _make(0.0)
    env.reset()
    rewards = [env.step(a)[1] for a in (0, 2, 4)]
    assert all(r == 0.0 for r in rewards), f"expected zeros, got {rewards}"
    print("OK - shaping_coef=0 restores the sparse reward")


def test_seeded_view_does_not_consume_budget():
    env = ViewReconEnv(
        dataset=_Dataset(), backbones=[_Backbone()], view_budget=BUDGET,
        lambda_cost=0.07, n_views=N_VIEWS, device="cpu",
        shaping_coef=1.0, seed_initial_view=True, gamma=GAMMA)
    env.reset()
    assert len(env.selected_view_indices) == 1, "seed view was not applied"
    steps = 0
    done = False
    while not done:
        legal = [i for i in range(N_VIEWS) if env.get_action_mask()[i] > 0]
        _, _, done, _, info = env.step(legal[0])
        steps += 1
    assert steps == BUDGET + 1, (
        f"episode took {steps} steps, expected {BUDGET + 1}; the seeded view "
        "is consuming budget instead of being free")
    assert info["n_views"] == BUDGET + 1, (
        f"reconstruction used {info['n_views']} views, expected {BUDGET + 1} "
        "(the seed plus the chosen ones)")
    print(f"OK - seeded view is free (episode {steps} steps, "
          f"{info['n_views']} views reconstructed)")


if __name__ == "__main__":
    print("Running reward-shaping tests...\n")
    test_shaping_terms_telescope()
    test_shaping_is_dense()
    test_sparse_still_available()
    test_seeded_view_does_not_consume_budget()
    print("\nShaping tests passed.")
