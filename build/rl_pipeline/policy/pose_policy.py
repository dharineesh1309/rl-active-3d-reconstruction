"""
RGB-only, pose-conditioned view-selection policy.

Replaces the index-based actor. The old head was `Linear(512 -> 256 -> 24)`,
which learns 24 independent action templates -- action 7 acquires an identity of
its own. That only works if index 7 denotes the same physical viewpoint on every
object, and it does not: measured across 988 Choy train objects, the mean
per-index azimuth standard deviation is 102.5 degrees, where a shared 24-pose
lattice would give 0.0. A policy over indices therefore cannot generalise, and no
amount of credit-assignment work downstream can fix that.

Here a candidate viewpoint is scored from its *geometry*:

    logit_j = f(h_t, pose_j, angular relation of pose_j to what is already seen)

with one shared f for every candidate. Two structural properties follow, and both
are asserted in tests/test_pose_policy_invariance.py:

    selected views   permutation INVARIANT   -- observations are a set
    candidates       permutation EQUIVARIANT -- renaming cameras cannot change
                                                the physical policy

Sensor contract: RGB only. The state is images and their known camera poses.
Nothing derived from ground-truth geometry enters it -- the depth-derived
coverage grid of the previous design is gone, and lives on the
`master` branch for a later RGB-D ablation.

The four ablation arms of Tier 0B are all constructed from this one class:

    set_encoder=False, pose_head=False   A/B-style index baseline
    set_encoder=True,  pose_head=False   new encoder, old action abstraction
    set_encoder=True,  pose_head=True    the proposed policy
    set_encoder=False, pose_head=True    pose head without the set encoder

so the comparison isolates the action abstraction from the encoder.
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

IMAGE_DIM = 2048        # frozen ResNet-50 global-average-pooled output
POSE_DIM = 5            # [sin az, cos az, sin el, cos el, normalised distance]
POSE_EMB = 128
IMG_EMB = 256
D_MODEL = 256
N_HEADS = 4
N_LAYERS = 2
REL_DIM = 4             # [delta_min, delta_mean, has_selected, n/N_max]

_DEFAULT_NORM = Path(__file__).resolve().parents[2] / "configs" / "pose_norm_v1.json"


# ── Pose descriptor ──────────────────────────────────────────────────────────

def load_pose_norm(path=None) -> tuple:
    """
    Distance mean/std, frozen from the TRAINING split.

    Validation and test must be normalised with the training statistics rather
    than recomputing their own, or the policy meets a different input
    distribution at evaluation than it trained on.
    """
    path = Path(path or os.environ.get("POSE_NORM", _DEFAULT_NORM))
    if not path.is_file():
        raise FileNotFoundError(
            f"Pose normalisation config not found: {path}\n"
            "Generate it with:  python validate_cameras.py --split train "
            "--freeze-pose-norm configs/pose_norm_v1.json"
        )
    cfg = json.load(open(path))
    return float(cfg["distance_mean"]), float(cfg["distance_std"])


def pose_descriptor(cams, d_mean: float, d_std: float) -> np.ndarray:
    """
    Cameras -> (N, 5).

    Angles go in as sin/cos pairs, never as raw degrees: 359 and 1 are adjacent
    directions but numerically far apart, and an MLP fed the raw value would have
    to learn that the space wraps.

    in_plane and fov are deliberately absent. Both are constant across every view
    of every object in this dataset (0.0 and 25.0), so they cannot distinguish
    candidates and would be dead inputs.
    """
    out = np.zeros((len(cams), POSE_DIM), dtype=np.float32)
    for i, c in enumerate(cams):
        a, e = math.radians(c["azimuth"]), math.radians(c["elevation"])
        out[i] = (math.sin(a), math.cos(a), math.sin(e), math.cos(e),
                  (float(c["distance"]) - d_mean) / max(d_std, 1e-8))
    return out


def viewing_directions(poses: torch.Tensor) -> torch.Tensor:
    """(..., 5) -> (..., 3) unit vectors, from the sin/cos entries directly."""
    sin_a, cos_a, sin_e, cos_e = (poses[..., 0], poses[..., 1],
                                  poses[..., 2], poses[..., 3])
    return torch.stack([cos_e * cos_a, cos_e * sin_a, sin_e], dim=-1)


def relative_features(cand_poses, sel_poses, sel_mask, n_max: int) -> torch.Tensor:
    """
    Per candidate: [delta_min, delta_mean, has_selected, n_selected / n_max].

    Angular separation from the already-acquired views, normalised to [0, 1] by
    pi. This is the inductive bias that replaces the coverage grid: it tells the
    policy how a candidate relates *geometrically* to what it already holds,
    without telling it anything about the object's shape.

    With nothing selected yet the two angles are 0 and `has_selected` is 0. The
    flag is what disambiguates -- a bare 0 would otherwise read as "coincides
    with an existing view", the opposite of the truth.
    """
    B, M, _ = cand_poses.shape
    u_c = viewing_directions(cand_poses)                       # (B, M, 3)
    u_s = viewing_directions(sel_poses)                        # (B, K, 3)
    cos = torch.clamp(torch.einsum("bmd,bkd->bmk", u_c, u_s), -1.0, 1.0)
    theta = torch.arccos(cos) / math.pi                        # (B, M, K)

    m = sel_mask.unsqueeze(1)                                  # (B, 1, K)
    n_sel = sel_mask.sum(dim=1)                                # (B,)
    has = (n_sel > 0).float()

    # Masked min and mean. Padding gets +inf for the min and 0 weight for the
    # mean, so absent views cannot masquerade as coincident ones.
    theta_min = torch.where(m > 0, theta, torch.full_like(theta, float("inf")))
    d_min = theta_min.min(dim=2).values
    d_min = torch.where(torch.isinf(d_min), torch.zeros_like(d_min), d_min)
    d_mean = (theta * m).sum(dim=2) / n_sel.clamp(min=1).unsqueeze(1)

    d_min = d_min * has.unsqueeze(1)
    d_mean = d_mean * has.unsqueeze(1)
    frac = (n_sel.float() / max(n_max, 1)).unsqueeze(1).expand(B, M)
    return torch.stack([d_min, d_mean, has.unsqueeze(1).expand(B, M), frac], dim=-1)


# ── Modules ──────────────────────────────────────────────────────────────────

class ImageProjection(nn.Module):
    """
    2048 -> 256, learned.

    The previous encoder did `feat.reshape(4, 512).mean(axis=0)`: it folded
    ResNet-50's 2048-D output to 512 by averaging four arbitrary 512-channel
    blocks. Those blocks are not interchangeable observations and not
    semantically homogeneous, so the operation mixed unrelated channels with no
    learned reason to. Treated here as a bug, not a style choice.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(IMAGE_DIM),
                                 nn.Linear(IMAGE_DIM, IMG_EMB), nn.GELU())

    def forward(self, x):
        return self.net(x)


class PoseEncoder(nn.Module):
    """5 -> 64 -> 128, shared by acquired views and candidate cameras so both
    land in one geometric latent space."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(POSE_DIM, 64), nn.GELU(),
                                 nn.Linear(64, POSE_EMB), nn.GELU())

    def forward(self, p):
        return self.net(p)


class ViewSetEncoder(nn.Module):
    """
    Pose-bound view tokens -> one state vector.

    Tokens are `[image ‖ pose]`, so each carries *what* was seen and *where from*.
    Mean pooling the image features alone, as the previous design did, keeps the
    appearance and discards the binding.

    No positional embeddings: the acquired views are a set, not a sequence, and
    the pose already says where each one came from. A learned [STATE] token reads
    out the summary, avoiding another arbitrary pooling choice.
    """

    def __init__(self, use_transformer: bool = True):
        super().__init__()
        self.use_transformer = use_transformer
        self.token = nn.Sequential(nn.Linear(IMG_EMB + POSE_EMB, D_MODEL), nn.GELU())
        self.state_token = nn.Parameter(torch.zeros(1, 1, D_MODEL))
        nn.init.normal_(self.state_token, std=0.02)
        if use_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=4 * D_MODEL,
                dropout=0.0, batch_first=True, norm_first=True,
                activation="gelu")
            # enable_nested_tensor is incompatible with norm_first and would
            # only warn; disabling it also keeps the padded and unpadded paths
            # numerically identical, which test_padding_inert checks.
            self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS,
                                                 enable_nested_tensor=False)

    def forward(self, img_emb, pose_emb, mask):
        B = img_emb.shape[0]
        x = self.token(torch.cat([img_emb, pose_emb], dim=-1))     # (B, K, D)
        if not self.use_transformer:
            w = mask.unsqueeze(-1)
            return (x * w).sum(1) / w.sum(1).clamp(min=1.0)

        st = self.state_token.expand(B, 1, D_MODEL)
        x = torch.cat([st, x], dim=1)                              # (B, 1+K, D)
        # src_key_padding_mask: True marks positions to IGNORE. The [STATE]
        # column is never masked, so an empty acquired set still attends to
        # itself instead of producing NaN.
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device),
                         mask < 0.5], dim=1)
        return self.encoder(x, src_key_padding_mask=pad)[:, 0]


class CandidateScorer(nn.Module):
    """One shared scorer over every candidate: f(state, pose, relation) -> logit."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_MODEL + POSE_EMB + REL_DIM, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 1))

    def forward(self, h, cand_pose_emb, rel):
        B, M, _ = cand_pose_emb.shape
        z = torch.cat([h.unsqueeze(1).expand(B, M, D_MODEL), cand_pose_emb, rel], -1)
        return self.net(z).squeeze(-1)


class RGBPosePolicy(nn.Module):
    """
    Actor-critic over candidate viewpoints.

    No backbone-selection head: routing became a full-information supervised
    problem once the counterfactual utilities turned out to be computable
    offline, and it is trained separately. During RL the terminal reward is
    max_m U_m(S), so the router cannot punish a good view set for a routing
    mistake it is still learning to avoid.
    """

    def __init__(self, n_candidates: int = 24, set_encoder: bool = True,
                 pose_head: bool = True):
        super().__init__()
        self.n_candidates = n_candidates
        self.pose_head = pose_head
        self.img_proj = ImageProjection()
        self.pose_enc = PoseEncoder()
        self.set_enc = ViewSetEncoder(use_transformer=set_encoder)
        if pose_head:
            self.scorer = CandidateScorer()
        else:
            # Index head, for the controlled ablation only. Learns one template
            # per slot, which is exactly the abstraction the measurement says is
            # invalid on this data.
            self.index_head = nn.Sequential(
                nn.Linear(D_MODEL, 256), nn.GELU(), nn.Linear(256, n_candidates))
        self.value_head = nn.Sequential(
            nn.Linear(D_MODEL, 128), nn.GELU(), nn.Linear(128, 1))
        self.apply(self._init)
        last = self.scorer.net[-1] if pose_head else self.index_head[-1]
        nn.init.orthogonal_(last.weight, gain=0.01)
        nn.init.zeros_(last.bias)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.zeros_(m.bias)

    def encode(self, obs):
        img = self.img_proj(obs["selected_feats"])
        pos = self.pose_enc(obs["selected_poses"])
        return self.set_enc(img, pos, obs["selected_mask"])

    def forward(self, obs):
        h = self.encode(obs)
        if self.pose_head:
            cp = self.pose_enc(obs["candidate_poses"])
            rel = relative_features(obs["candidate_poses"], obs["selected_poses"],
                                    obs["selected_mask"], self.n_candidates)
            logits = self.scorer(h, cp, rel)
        else:
            logits = self.index_head(h)[:, :obs["candidate_poses"].shape[1]]
        logits = logits.masked_fill(obs["candidate_mask"] < 0.5, float("-inf"))
        return logits, self.value_head(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, greedy: bool = False):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(-1) if greedy else dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(self, obs, actions):
        logits, value = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), value, dist.entropy()
