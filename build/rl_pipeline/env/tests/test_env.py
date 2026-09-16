"""
Unit tests for ViewReconEnv.

Run with `python env/tests/test_env.py` from the rl_pipeline directory.

Every test runs in mock-reward mode (backbones=None), so no weights are needed
and the suite stays fast. Mock mode returns a fixed IoU of 0.5, which is enough
to exercise the episode structure, masking and bookkeeping.
"""

import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from env.view_recon_env import ViewReconEnv


N_VIEWS = 24
VOXEL_RES = 32


class FakeBackbone:
    """Minimal stand-in so backbone selection can be tested without weights."""

    def __init__(self, name, cost, fill=0.5):
        self.name = name
        self.cost = cost
        self.fill = fill
        self.calls = 0

    def predict(self, images, cams=None):
        self.calls += 1
        return np.full((VOXEL_RES,) * 3, self.fill, dtype=np.float32)


def make_fake_dataset():
    fake_image      = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
    fake_depth      = np.random.rand(32, 32).astype(np.float32)
    fake_silhouette = (np.random.rand(64, 64) > 0.5).astype(np.float32)
    fake_voxels     = (np.random.rand(VOXEL_RES, VOXEL_RES, VOXEL_RES) > 0.5).astype(np.float32)

    return [{
        "images":      [fake_image]      * N_VIEWS,
        "depths":      [fake_depth]      * N_VIEWS,
        "silhouettes": [fake_silhouette] * N_VIEWS,
        "voxels":      fake_voxels,
        "category":    "chair",
        "model_id":    "fake_model_0",
    }]


def _make_env(view_budget=5, backbones=None, dataset=None):
    return ViewReconEnv(
        dataset     = make_fake_dataset() if dataset is None else dataset,
        backbones   = backbones,
        view_budget = view_budget,
    )


def _run_views(env, budget):
    """Spend the whole view budget, return the final step's output."""
    out = None
    for i in range(budget):
        out = env.step(i)
    return out


# ── Test 1: reset() returns correct shapes ───────────────────────────────────

def test_reset():
    env = _make_env()
    obs, _ = env.reset()

    assert obs["coverage_grid"].shape  == (32, 32, 32), "coverage grid shape wrong"
    assert obs["image_features"].shape == (512,),       "image features shape wrong"
    assert obs["view_mask"].shape      == (N_VIEWS,),   "view mask shape wrong"
    assert obs["is_model_step"].shape  == (1,),         "is_model_step shape wrong"
    assert obs["is_model_step"][0] == 0.0, "should not start on the model step"
    assert obs["budget"].shape == (2,), "budget shape wrong"
    print("OK 1 - reset() shapes are correct")


# ── Test 1b: the budget is observable ────────────────────────────────────────

def test_budget_observable():
    """
    Two episodes with different budgets must look different at step 0.

    This is the defect that wasted a 30,274-episode run: view_mask is all zeros
    at the first step whatever the budget, so the critic could not predict the
    -lambda*B term in the terminal reward. Explained variance fell from 0.74 to
    0.06 the moment phase 2 began sampling budgets, and the policy never left
    89% of maximum entropy. If this assert ever fails again, the same silent
    failure is back.
    """
    a = _make_env(); a.set_view_budget(3); obs_a, _ = a.reset()
    b = _make_env(); b.set_view_budget(8); obs_b, _ = b.reset()

    import numpy as np
    assert not np.allclose(obs_a["budget"], obs_b["budget"]), (
        "B=3 and B=8 produce identical observations at step 0; the critic "
        "cannot learn the budget-dependent part of the reward")
    # Remaining views must count down as views are spent.
    start = float(obs_a["budget"][1])
    a.step(0)
    assert float(a._get_obs()["budget"][1]) < start, "remaining budget did not decrease"
    print("OK 1b - budget is observable and counts down")


# ── Test 1c: the coverage grid is a real back-projection ─────────────────────

def test_coverage_is_geometric():
    """
    Back-projected points must land on the object far better than the old
    image-plane stencil did.

    The coverage grid accumulates several views into one geometry, which only
    works if every view maps into a shared world frame. Two things must be
    exactly right: depth must be metric (scaled to a fixed world range, not each
    view's own min/max), and the inverse rotation must be the same Rx@Ry that
    `_depth_from_voxels` applied, including the (row, col) = (cam_y, cam_x)
    transpose. Get either wrong and the grid still looks plausible -- populated,
    growing with each view -- while being geometrically meaningless.

    Asserting camera-aware beats image-plane is self-calibrating: it does not
    depend on the object's occupancy fraction, which sets the chance rate and
    varies a lot between a solid cube and a chair.
    """
    import numpy as np
    from env.state_builder import CoverageGrid
    from dataloader import _depth_from_voxels

    vox = np.zeros((32, 32, 32), np.float32)
    vox[10:22, 10:22, 10:22] = 1.0
    truth = vox > 0.5

    cams = [{"azimuth": a, "elevation": e}
            for a, e in ((0, 0), (90, 15), (180, -10), (270, 25))]

    def precision(use_cam):
        g = CoverageGrid()
        for c in cams:
            d = _depth_from_voxels(vox, c["azimuth"], c["elevation"])
            g.update(d, (d < 1.0).astype(np.float32), cam=c if use_cam else None)
        cov = g.get() > 0.5
        return (cov & truth).sum() / max(cov.sum(), 1)

    flat, aware = precision(False), precision(True)
    assert aware > flat * 1.5, (
        f"camera-aware precision {aware:.3f} is not clearly better than the "
        f"image-plane stencil {flat:.3f}; the back-projection is wrong")
    print(f"OK 1c - coverage is geometric (camera-aware {aware:.2f} vs flat {flat:.2f})")


# ── Test 1d: the budget field stays in bounds and matches evaluate.py ────────

def test_budget_field_bounds():
    """
    The budget observation must stay inside its Box(0, 1) and count DECISIONS.

    An earlier version computed remaining as `B - len(selected_view_indices)`.
    With seed_initial_view that list already holds the free seed, so it read B-1
    at the first decision and went NEGATIVE at the model step -- out of bounds,
    and disagreeing with evaluate.py, which counts chosen views. A policy
    evaluated against a different budget encoding than it trained on looks
    broken for no visible reason.
    """
    import numpy as np
    for seed in (False, True):
        env = _make_env()
        env.seed_initial_view = seed
        for B in (3, 5, 8):
            env.set_view_budget(B)
            obs, _ = env.reset()
            seen = [obs["budget"].copy()]
            done = False
            while not done:
                legal = [i for i in range(N_VIEWS) if env.get_action_mask()[i] > 0]
                obs, _, done, _, _ = env.step(legal[0])
                seen.append(obs["budget"].copy())
            arr = np.array(seen)
            assert arr.min() >= 0.0 and arr.max() <= 1.0, (
                f"budget field out of Box(0,1) for seed={seed} B={B}: "
                f"min {arr.min()}, max {arr.max()}")
            # First decision must show the full budget remaining, seeded or not.
            assert abs(float(seen[0][1]) - B / env.n_views) < 1e-6, (
                f"seed={seed} B={B}: first decision shows "
                f"{seen[0][1]:.4f} remaining, expected {B / env.n_views:.4f}")
    print("OK 1d - budget field in bounds and independent of seeding")


# ── Test 2: step() updates state correctly ───────────────────────────────────

def test_step():
    env = _make_env()
    env.reset()

    obs, reward, done, _, info = env.step(0)

    assert obs["view_mask"][0] == 1.0, "view 0 should be marked visited"
    assert reward == 0.0,              "intermediate step should return reward 0"
    assert done is False,              "should not be done after 1 of 5 view steps"
    print("OK 2 - step() updates state correctly")


# ── Test 3: episode runs B views then one model step ─────────────────────────

def test_episode_has_model_step():
    env = _make_env(view_budget=5)
    env.reset()

    for i in range(5):
        obs, reward, done, _, _ = env.step(i)
        assert done is False, f"should not be done at view step {i+1}"
        assert reward == 0.0, f"intermediate reward should be 0 at view step {i+1}"

    assert obs["is_model_step"][0] == 1.0, "budget spent — should be the model step"

    obs, reward, done, _, info = env.step(0)
    assert done is True,       "episode should end after the model step"
    assert reward != 0.0,      "terminal step should return a non-zero reward"
    assert "backbone" in info, "terminal info should name the chosen backbone"
    assert info["n_views"] == 5
    print(f"OK 3 - B+1 horizon, terminal reward {reward:+.4f}")


# ── Test 4: repeated view gives penalty ──────────────────────────────────────

def test_repeated_view():
    env = _make_env()
    env.reset()

    env.step(0)
    _, reward, done, _, _ = env.step(0)

    assert reward == -0.1, "repeated view should give -0.1 penalty"
    assert done is False,  "repeated view should not end episode"
    print("OK 4 - repeated view gives penalty correctly")


# ── Test 5: action mask blocks visited views ─────────────────────────────────

def test_action_mask():
    env = _make_env()
    env.reset()

    env.step(7)
    env.step(15)
    mask = env.get_action_mask()

    assert mask[7]  == False, "view 7 should be blocked"
    assert mask[15] == False, "view 15 should be blocked"
    assert mask[0]  == True,  "view 0 should still be available"
    assert mask.any(), "at least one action must always be legal"
    print("OK 5 - action mask blocks visited views correctly")


# ── Test 6: selected_view_indices tracks picked views ────────────────────────

def test_selected_view_indices():
    env = _make_env()
    env.reset()

    env.step(3)
    env.step(7)
    env.step(3)   # repeated — should NOT be appended

    assert env.selected_view_indices == [3, 7], \
        f"selected_view_indices should be [3, 7], got {env.selected_view_indices}"
    print("OK 6 - selected_view_indices tracked correctly")


# ── Test 7: on the model step, only backbones are legal ──────────────────────

def test_model_step_mask():
    backbones = [FakeBackbone("cheap", 0.1), FakeBackbone("dear", 1.0)]
    env = _make_env(view_budget=3, backbones=backbones)
    env.reset()
    _run_views(env, 3)

    mask = env.get_action_mask()
    assert mask[:2].all(), "both backbones should be legal on the model step"
    assert not mask[2:].any(), \
        "no action beyond the backbone count may be legal on the model step"
    print("OK 7 - model-step mask exposes only backbones")


# ── Test 8: the chosen backbone is the one actually run ──────────────────────

def test_chosen_backbone_is_used():
    cheap = FakeBackbone("cheap", 0.1)
    dear  = FakeBackbone("dear",  1.0)
    env = _make_env(view_budget=3, backbones=[cheap, dear])
    env.reset()
    _run_views(env, 3)

    _, _, _, _, info = env.step(1)
    assert info["backbone"] == "dear", f"expected 'dear', got {info['backbone']}"
    assert dear.calls == 1 and cheap.calls == 0, \
        "only the selected backbone should be invoked"
    print("OK 8 - the selected backbone is the one that runs")


# ── Test 9: an expensive backbone is charged more ────────────────────────────

def test_backbone_cost_affects_reward():
    """Same prediction, different cost: the dearer choice must pay for it."""
    # One shared dataset, so both runs score against identical ground truth and
    # the only difference left in the reward is the backbone's cost.
    dataset = make_fake_dataset()
    rewards = {}
    for idx in (0, 1):
        backbones = [FakeBackbone("cheap", 0.1), FakeBackbone("dear", 2.0)]
        env = _make_env(view_budget=3, backbones=backbones, dataset=dataset)
        env.reset()
        _run_views(env, 3)
        _, reward, _, _, info = env.step(idx)
        rewards[info["backbone"]] = reward

    assert rewards["cheap"] > rewards["dear"], (
        f"cheaper backbone should earn more for an identical reconstruction: "
        f"{rewards}"
    )
    # lambda_cost 0.05 x cost difference (2.0 - 0.1) = 0.095
    gap = rewards["cheap"] - rewards["dear"]
    assert abs(gap - 0.095) < 1e-6, f"expected a 0.095 cost gap, got {gap:.6f}"
    print(f"OK 9 - compute penalty applied, cheap-minus-dear = {gap:.4f}")


# ── Test 10: variable budgets still terminate correctly ──────────────────────

def test_variable_budgets():
    for budget in (3, 5, 8):
        backbones = [FakeBackbone("only", 0.1)]
        env = _make_env(view_budget=budget, backbones=backbones)
        env.reset()
        _run_views(env, budget)
        _, _, done, _, info = env.step(0)
        assert done is True, f"budget {budget} should terminate after the model step"
        assert info["n_views"] == budget
    print("OK 10 - budgets 3, 5 and 8 all terminate at B+1")


if __name__ == "__main__":
    print("Running env tests...\n")
    test_reset()
    test_budget_observable()
    test_coverage_is_geometric()
    test_budget_field_bounds()
    test_step()
    test_episode_has_model_step()
    test_repeated_view()
    test_action_mask()
    test_selected_view_indices()
    test_model_step_mask()
    test_chosen_backbone_is_used()
    test_backbone_cost_affects_reward()
    test_variable_budgets()
    print("\nAll tests passed.")
