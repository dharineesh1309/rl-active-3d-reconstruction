"""
policy/view_policy.py
─────────────────────
Two-head actor-critic policy for joint view and backbone selection.

Architecture
────────────
  Encoder branches
    coverage_encoder : CoverageGridEncoder  (32³ voxel grid)  → 256
    feature_encoder  : MLP                  (512 image feats)  → 256
    history_encoder  : MLP                  (24 view mask)     →  64
                                             ─────────────────────
    concat + shared trunk                   576  →  512

  Heads
    view_head  : Linear  512 → 256 → n_views      (actor — picks next view)
    model_head : Linear  512 → 256 → n_backbones  (actor — picks backbone)
    value_head : Linear  512 → 128 → 1            (critic — estimates V(s))

Both actor heads write into one Discrete(max(n_views, n_backbones)) action
space. Which head produces the logits is decided per sample by the
`is_model_step` observation flag the environment sets on the terminal step,
and the environment's action mask zeroes whatever is illegal. Sharing the
trunk is the point: the representation built while choosing views is exactly
the evidence needed to judge which backbone can exploit them.
"""

import torch
import torch.nn as nn


# ── Sub-module: 3-D CNN for coverage grid ────────────────────────────────────

class CoverageGridEncoder(nn.Module):
    """
    Input : (B, 32, 32, 32) voxel occupancy grid
    Output: (B, 256)

    Stride-2 convolutions halve every spatial dimension:
      32³ → 16³ → 8³ → 4³  → Flatten(64×4³=4096) → Linear → 256
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4 * 4, 256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 32, 32, 32)  →  add channel dim  →  (B, 1, 32, 32, 32)
        return self.net(x.unsqueeze(1))


# ── Main policy ───────────────────────────────────────────────────────────────

class ViewPolicy(nn.Module):
    """
    Actor-critic policy that selects the next viewpoint.

    Parameters
    ----------
    n_views : int
        Number of candidate views (default 24, matching 3D-R2N2 rendering).
    n_backbones : int
        Number of reconstruction backbones selectable on the terminal step.
        Must match len(load_backbones(...)) at both train and inference time,
        since it sets the width of the model head.
    """

    def __init__(self, n_views: int = 24, n_backbones: int = 1):
        super().__init__()
        self.n_views = n_views
        self.n_backbones = n_backbones
        self.n_actions = max(n_views, n_backbones)

        # ── Encoder branches ─────────────────────────────────────────────────
        self.coverage_encoder = CoverageGridEncoder()           # → 256
        self.feature_encoder  = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
        )                                                        # → 256
        self.history_encoder  = nn.Sequential(
            nn.Linear(n_views, 64), nn.ReLU(inplace=True),
        )                                                        # →  64

        # ── Shared trunk  (256 + 256 + 64 = 576 → 512) ───────────────────────
        self.shared = nn.Sequential(
            # 256 coverage + 256 image features + 64 history + 2 budget.
            nn.Linear(578, 512), nn.ReLU(inplace=True),
        )

        # ── Actor head: picks next view ───────────────────────────────────────
        self.view_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, n_views),
        )

        # ── Actor head: picks the reconstruction backbone ────────────────────
        self.model_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(inplace=True),
            nn.Linear(256, n_backbones),
        )

        # ── Critic head: estimates state value ───────────────────────────────
        self.value_head = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 1),
        )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv3d)):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Shared encoding step ──────────────────────────────────────────────────

    def _encode(self, obs: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : dict with keys
            'coverage_grid'  : (B, 32, 32, 32) float32
            'image_features' : (B, 512)         float32
            'view_mask'      : (B, n_views)      float32
            'budget'         : (B, 2)            float32

        Returns
        -------
        latent : (B, 512)
        """
        cov  = self.coverage_encoder(obs['coverage_grid'])   # (B, 256)
        feat = self.feature_encoder(obs['image_features'])   # (B, 256)
        hist = self.history_encoder(obs['view_mask'])        # (B,  64)
        # Budget goes in raw: it is already two scalars in [0,1], and the
        # critic needs it linearly to subtract lambda*B from its estimate.
        return self.shared(
            torch.cat([cov, feat, hist, obs['budget']], dim=-1))  # (B, 512)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self,
                obs: dict,
                action_mask: torch.Tensor = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Routes each sample to the head its `is_model_step` flag selects, then
        writes both heads' outputs into one shared-width logit tensor. Entries
        no head wrote stay at -inf and are therefore never sampled — that is
        how a 24-wide view step and an n_backbones-wide model step coexist in
        a single Categorical.

        Parameters
        ----------
        obs         : dict of (B, *) tensors, including 'is_model_step' (B,1)
        action_mask : (B, n_actions) float32 — 1 = legal, 0 = illegal

        Returns
        -------
        logits : (B, n_actions)
        values : (B,)
        """
        latent = self._encode(obs)
        values = self.value_head(latent).squeeze(-1)

        batch = latent.shape[0]
        is_model = obs['is_model_step'].reshape(batch) > 0.5

        logits = latent.new_full((batch, self.n_actions), float('-inf'))
        if (~is_model).any():
            logits[~is_model, :self.n_views] = self.view_head(latent[~is_model])
        if is_model.any():
            logits[is_model, :self.n_backbones] = self.model_head(latent[is_model])

        if action_mask is not None:
            logits = logits.masked_fill(action_mask == 0, float('-inf'))

        return logits, values

    # ── Inference (no gradient) ───────────────────────────────────────────────

    @torch.no_grad()
    def act(self,
            obs: dict,
            action_mask: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the current policy.

        Parameters
        ----------
        obs         : dict of (n_envs, *) tensors
        action_mask : (n_envs, n_views) float32

        Returns
        -------
        actions   : (n_envs,) int64
        log_probs : (n_envs,) float32
        values    : (n_envs,) float32
        """
        logits, values = self.forward(obs, action_mask)
        dist      = torch.distributions.Categorical(logits=logits)
        actions   = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values

    # ── Re-evaluation with gradient (used inside PPO update) ─────────────────

    def evaluate_actions(self,
                         obs: dict,
                         actions: torch.Tensor,
                         action_masks: torch.Tensor
                         ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Re-compute log-probs, values, and entropy for a batch of stored actions.

        Returns
        -------
        log_probs : (B,)
        values    : (B,)
        entropy   : (B,)
        """
        logits, values = self.forward(obs, action_masks)
        dist      = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy   = dist.entropy()
        return log_probs, values, entropy

    # ── Partial loading (carry an old view-only checkpoint forward) ──────────

    def load_compatible(self, state_dict: dict) -> dict:
        """
        Load every tensor whose name and shape match this policy, skip the rest.

        Checkpoints trained before the model head existed (and before the
        action space widened) cannot satisfy load_state_dict(strict=True). This
        keeps their view-selection competence — the expensive part, tens of
        thousands of episodes — and leaves the new head freshly initialised.

        Returns a summary dict of what was loaded and what was skipped.
        """
        own = self.state_dict()
        loaded, skipped = [], []

        for name, tensor in state_dict.items():
            if name in own and own[name].shape == tensor.shape:
                own[name] = tensor
                loaded.append(name)
            else:
                skipped.append(name)

        self.load_state_dict(own)
        missing = [n for n in own if n not in set(loaded)]
        return {"loaded": loaded, "skipped": skipped, "left_fresh": missing}
