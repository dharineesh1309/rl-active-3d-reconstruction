"""
Structural guarantees of the pose-conditioned policy.

The whole point of scoring candidates from geometry rather than from an array
slot is that the policy stops depending on how cameras happen to be numbered.
That is a property of the architecture, so it can be asserted exactly rather
than hoped for -- and every one of these failures would be silent in training.

    selected views   permutation INVARIANT   -- observations are a set
    candidates       permutation EQUIVARIANT -- renaming cameras must permute
                                                the logits, nothing more
    candidate IDs    invisible               -- only poses reach the network
    candidate count  flexible                -- M = 10 and M = 24 share weights
    padding          inert                   -- a short set batched with long
                                                ones scores the same alone

    python policy/tests/test_pose_policy_invariance.py
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from policy.pose_policy import (POSE_DIM, RGBPosePolicy, pose_descriptor,
                                relative_features)

TOL = 1e-5


def _obs(B=2, K=3, M=24, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    sel_mask = torch.ones(B, K)
    return {
        "selected_feats":  torch.randn(B, K, 2048, generator=g),
        "selected_poses":  torch.randn(B, K, POSE_DIM, generator=g),
        "selected_mask":   sel_mask,
        "candidate_poses": torch.randn(B, M, POSE_DIM, generator=g),
        "candidate_mask":  torch.ones(B, M),
    }


def _policy(**kw):
    p = RGBPosePolicy(n_candidates=24, **kw)
    p.eval()                      # dropout off; these are exact comparisons
    return p


def test_selected_views_permutation_invariant():
    """Reordering the acquired views must not change the state at all."""
    p, obs = _policy(), _obs()
    perm = torch.randperm(obs["selected_feats"].shape[1])
    shuf = dict(obs)
    shuf["selected_feats"] = obs["selected_feats"][:, perm]
    shuf["selected_poses"] = obs["selected_poses"][:, perm]
    shuf["selected_mask"] = obs["selected_mask"][:, perm]
    with torch.no_grad():
        a, b = p.encode(obs), p.encode(shuf)
    d = (a - b).abs().max().item()
    assert d < TOL, (f"state changed by {d:.2e} when the acquired views were "
                     "reordered; the encoder is treating a set as a sequence "
                     "(positional embeddings leaking in?)")
    print(f"OK - selected views permutation INVARIANT (max delta {d:.1e})")


def test_candidates_permutation_equivariant():
    """Permuting candidates must permute the logits and do nothing else."""
    p, obs = _policy(), _obs()
    M = obs["candidate_poses"].shape[1]
    perm = torch.randperm(M)
    shuf = dict(obs)
    shuf["candidate_poses"] = obs["candidate_poses"][:, perm]
    shuf["candidate_mask"] = obs["candidate_mask"][:, perm]
    with torch.no_grad():
        l0, _ = p(obs)
        l1, _ = p(shuf)
    d = (l1 - l0[:, perm]).abs().max().item()
    assert d < TOL, (f"logits differ by {d:.2e} beyond the permutation; a "
                     "candidate's score still depends on its array position")
    # The induced distribution must permute too.
    pr0 = torch.softmax(l0, -1)[:, perm]
    pr1 = torch.softmax(l1, -1)
    assert (pr0 - pr1).abs().max().item() < TOL
    print(f"OK - candidates permutation EQUIVARIANT (max delta {d:.1e})")


def test_mask_equivariance():
    """Permuting poses and mask together must permute the masked logits."""
    p, obs = _policy(), _obs()
    M = obs["candidate_poses"].shape[1]
    obs["candidate_mask"][:, [2, 5, 9]] = 0.0          # some already visited
    perm = torch.randperm(M)
    shuf = dict(obs)
    shuf["candidate_poses"] = obs["candidate_poses"][:, perm]
    shuf["candidate_mask"] = obs["candidate_mask"][:, perm]
    with torch.no_grad():
        l0, _ = p(obs)
        l1, _ = p(shuf)
    ref = l0[:, perm]
    finite = torch.isfinite(ref) & torch.isfinite(l1)
    assert torch.equal(torch.isfinite(ref), torch.isfinite(l1)), \
        "masked-out positions did not follow the permutation"
    d = (l1[finite] - ref[finite]).abs().max().item()
    assert d < TOL, f"masked logits differ by {d:.2e} beyond the permutation"
    print(f"OK - mask equivariance (max delta {d:.1e})")


def test_candidate_ids_invisible():
    """
    Renaming cameras must change nothing.

    Reversing the candidate order is a rename with no geometric content. The old
    index head would score differently; this one must not, once the logits are
    put back in their original order.
    """
    p, obs = _policy(), _obs()
    M = obs["candidate_poses"].shape[1]
    rev = torch.arange(M - 1, -1, -1)
    shuf = dict(obs)
    shuf["candidate_poses"] = obs["candidate_poses"][:, rev]
    shuf["candidate_mask"] = obs["candidate_mask"][:, rev]
    with torch.no_grad():
        l0, _ = p(obs)
        l1, _ = p(shuf)
    d = (l1[:, rev] - l0).abs().max().item()
    assert d < TOL, f"renaming candidates changed the policy by {d:.2e}"
    print(f"OK - candidate IDs invisible to the network (max delta {d:.1e})")


def test_variable_candidate_count():
    """M = 10 and M = 24 run through the same weights without reshaping."""
    p = _policy()
    for M in (10, 24, 40):
        obs = _obs(M=M)
        with torch.no_grad():
            logits, value = p(obs)
        assert logits.shape == (2, M), f"expected (2,{M}), got {tuple(logits.shape)}"
        assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    print("OK - candidate-count agnostic (M = 10, 24, 40 share one network)")


def test_padding_inert():
    """A 2-view state must score the same alone as padded inside a 5-view batch."""
    p = _policy()
    g = torch.Generator().manual_seed(3)
    K_small, K_pad, M = 2, 5, 24
    feats = torch.randn(1, K_pad, 2048, generator=g)
    poses = torch.randn(1, K_pad, POSE_DIM, generator=g)
    cand = torch.randn(1, M, POSE_DIM, generator=g)

    short = {"selected_feats": feats[:, :K_small], "selected_poses": poses[:, :K_small],
             "selected_mask": torch.ones(1, K_small),
             "candidate_poses": cand, "candidate_mask": torch.ones(1, M)}
    padded = {"selected_feats": feats, "selected_poses": poses,
              "selected_mask": torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]]),
              "candidate_poses": cand, "candidate_mask": torch.ones(1, M)}
    with torch.no_grad():
        l0, v0 = p(short)
        l1, v1 = p(padded)
    d = max((l1 - l0).abs().max().item(), (v1 - v0).abs().max().item())
    assert d < TOL, (f"padding changed the output by {d:.2e}; the attention "
                     "padding mask is not being honoured")
    print(f"OK - padding inert (max delta {d:.1e})")


def test_empty_selected_set():
    """The first decision, with nothing acquired yet, must be finite."""
    p = _policy()
    obs = _obs(K=1)
    obs["selected_mask"] = torch.zeros_like(obs["selected_mask"])
    with torch.no_grad():
        logits, value = p(obs)
    assert torch.isfinite(logits).all(), "empty acquired set produced non-finite logits"
    assert torch.isfinite(value).all()
    rel = relative_features(obs["candidate_poses"], obs["selected_poses"],
                            obs["selected_mask"], 24)
    assert torch.allclose(rel[..., 2], torch.zeros_like(rel[..., 2])), \
        "has_selected should be 0 with nothing acquired"
    assert torch.allclose(rel[..., :2], torch.zeros_like(rel[..., :2])), \
        "angular features should be 0 when has_selected is 0"
    print("OK - empty acquired set is finite, has_selected flag correct")


def test_pose_descriptor_wraps():
    """Azimuth 359 and 1 must be near neighbours, not opposite extremes."""
    cams = [{"azimuth": a, "elevation": 27.0, "distance": 0.8}
            for a in (359.0, 1.0, 180.0)]
    d = pose_descriptor(cams, 0.8, 0.09)
    near = np.linalg.norm(d[0, :2] - d[1, :2])
    far = np.linalg.norm(d[0, :2] - d[2, :2])
    assert near < 0.05, f"359 and 1 deg are {near:.3f} apart in descriptor space"
    assert far > near * 10, "180 deg apart should be far larger than 2 deg apart"
    print(f"OK - azimuth wraps correctly (359 vs 1: {near:.4f}, "
          f"359 vs 180: {far:.4f})")


def test_index_head_is_not_invariant():
    """
    The control: the index head SHOULD fail candidate-rename invariance.

    If it passed, the equivariance tests above would be vacuous -- they would be
    measuring something every architecture satisfies rather than the property
    the pose head was built for.
    """
    p, obs = _policy(pose_head=False), _obs()
    M = obs["candidate_poses"].shape[1]
    rev = torch.arange(M - 1, -1, -1)
    shuf = dict(obs)
    shuf["candidate_poses"] = obs["candidate_poses"][:, rev]
    shuf["candidate_mask"] = obs["candidate_mask"][:, rev]
    with torch.no_grad():
        l0, _ = p(obs)
        l1, _ = p(shuf)
    d = (l1[:, rev] - l0).abs().max().item()
    assert d > 1e-3, ("the index head appears invariant to candidate renaming, "
                      "which would make the equivariance tests vacuous")
    print(f"OK - control: index head IS sensitive to renaming (delta {d:.2e})")


if __name__ == "__main__":
    print("Running pose-policy invariance suite...\n")
    test_selected_views_permutation_invariant()
    test_candidates_permutation_equivariant()
    test_mask_equivariance()
    test_candidate_ids_invisible()
    test_variable_candidate_count()
    test_padding_inert()
    test_empty_selected_set()
    test_pose_descriptor_wraps()
    test_index_head_is_not_invariant()
    print("\nInvariance suite passed.")
