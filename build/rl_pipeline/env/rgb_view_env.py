"""
RGB-only active view acquisition.

Differs from the previous ViewReconEnv in three ways that matter.

**Sensor contract.** The state is images and their known camera poses. Nothing
derived from ground-truth geometry enters it. The depth-derived coverage grid is
gone; it survives on `master` for a later RGB-D ablation, which is a genuinely
open question rather than a settled one -- coverage was measured to be a poor
*reward*, which says nothing about its value as an *observation*.

**No backbone action.** The episode is B view steps and then it ends. Routing
became a full-information supervised problem once the counterfactual utilities
turned out to be computable offline, and the terminal reward is

    R = max_m [ IoU_m(S) - lambda * cost_m ]

so a good view set is never punished for a routing mistake a learning head
happens to make. `utility_fn` supplies that envelope, which keeps the
environment independent of how many reconstructors exist and lets tests inject a
known-answer function.

**Candidate identity is bookkeeping.** Integer view indices still exist here --
they address images and mark visits -- but they never reach the policy, which
sees only pose descriptors. Renaming or reordering cameras cannot change
behaviour, which `test_pose_policy_invariance.py` asserts directly.

Observation (the frozen interface):

    selected_feats   (K, 2048)  frozen ResNet-50 features of acquired views
    selected_poses   (K, 5)     matching pose descriptors
    selected_mask    (K,)       1 real, 0 padding
    candidate_poses  (M, 5)     every candidate camera for THIS object
    candidate_mask   (M,)       1 legal, 0 already visited
    budget           (2,)       [B / M, views remaining / M]
"""

import numpy as np
import torch

from policy.pose_policy import POSE_DIM, load_pose_norm, pose_descriptor


class ResNetFeatures:
    """
    Frozen ResNet-50 global-average-pooled features, 2048-D.

    Returns the full 2048 rather than the folded 512 the old extractor produced:
    that fold averaged four arbitrary channel blocks together. The learned
    projection to 256 lives in the policy, so the environment stays a pure
    observation source.

    Features are deterministic per (object, view), and with object-conditioned
    baselines every parallel environment replays the same object, so caching
    them removes most of the ResNet cost.
    """

    def __init__(self, device: str = "cpu", cache_size: int = 20000):
        import torchvision.models as models
        import torchvision.transforms as T

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.tf = T.Compose([
            T.Resize((224, 224)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self._cache = {}
        self._cache_size = cache_size

    @torch.no_grad()
    def __call__(self, image, key=None):
        if key is not None and key in self._cache:
            return self._cache[key]
        if image.mode != "RGB":
            image = image.convert("RGB")
        x = self.tf(image).unsqueeze(0).to(self.device)
        f = self.model(x).reshape(-1).cpu().numpy().astype(np.float32)
        if key is not None and len(self._cache) < self._cache_size:
            self._cache[key] = f
        return f


class RGBViewEnv:
    """
    Sequential RGB view acquisition. Gymnasium-like but deliberately minimal.

    Parameters
    ----------
    dataset      : ShapeNetChoyDataset (or anything with the same item contract)
    utility_fn   : (item, view_indices) -> float, the terminal reward. Should
                   return max_m [IoU_m - lambda*cost_m]; tests inject a
                   known-answer stand-in.
    view_budget  : number of views the agent CHOOSES. With seed_initial_view the
                   episode reconstructs from B+1 views.
    """

    def __init__(self, dataset, utility_fn, view_budget: int = 4,
                 features=None, device: str = "cpu",
                 seed_initial_view: bool = True,
                 group_sync_seed: int = None,
                 phase2_budgets=(2, 4, 7),
                 pose_norm_path=None):
        self.dataset = dataset
        self.utility_fn = utility_fn
        self.B = view_budget
        self.n_views = getattr(dataset, "n_views", 24)
        self.features = features or ResNetFeatures(device=device)
        self.seed_initial_view = seed_initial_view
        self._phase2_budgets = list(phase2_budgets)
        self._phase = 1
        self.d_mean, self.d_std = load_pose_norm(pose_norm_path)
        # Same RNG across a group so every parallel env replays one object, which
        # is what makes the leave-one-out baseline cancel object difficulty
        # exactly rather than approximately.
        self._sync = (np.random.RandomState(group_sync_seed)
                      if group_sync_seed is not None else None)
        self.reset()

    # ── phase / budget ───────────────────────────────────────────────────────

    def set_phase(self, phase: int):
        self._phase = phase

    def set_view_budget(self, b: int):
        self.B = b

    # ── episode ──────────────────────────────────────────────────────────────

    def reset(self):
        rng = self._sync if self._sync is not None else np.random
        if self._phase == 2:
            self.B = int(rng.choice(self._phase2_budgets))
        idx = int(rng.randint(len(self.dataset)))
        self.item = self.dataset[idx]
        self.model_id = self.item.get("model_id", str(idx))

        self.cand_poses = pose_descriptor(self.item["cams"], self.d_mean, self.d_std)
        self.selected = []
        self.step_count = 0
        self.done = False

        # One free view before the first decision. Without it the acquired set is
        # empty, so the [STATE] token is identical for every object and the first
        # choice cannot be informed by anything.
        if self.seed_initial_view:
            self._acquire(int(rng.randint(self.n_views)))
            self.step_count = 0        # the gift does not consume budget
        return self._obs()

    def _acquire(self, v: int):
        self.selected.append(v)
        self.step_count += 1

    def _obs(self):
        k = max(len(self.selected), 1)
        feats = np.zeros((k, 2048), dtype=np.float32)
        poses = np.zeros((k, POSE_DIM), dtype=np.float32)
        mask = np.zeros((k,), dtype=np.float32)
        for i, v in enumerate(self.selected):
            feats[i] = self.features(self.item["images"][v], key=(self.model_id, v))
            poses[i] = self.cand_poses[v]
            mask[i] = 1.0

        cmask = np.ones((self.n_views,), dtype=np.float32)
        for v in self.selected:
            cmask[v] = 0.0
        remaining = max(self.B - self.step_count, 0)
        return {
            "selected_feats": feats,
            "selected_poses": poses,
            "selected_mask": mask,
            "candidate_poses": self.cand_poses.copy(),
            "candidate_mask": cmask,
            "budget": np.array([self.B / self.n_views,
                                remaining / self.n_views], dtype=np.float32),
        }

    def action_mask(self):
        return self._obs()["candidate_mask"]

    def step(self, action: int):
        assert not self.done, "step() after the episode ended"
        assert 0 <= action < self.n_views, f"invalid view {action}"
        if action in self.selected:
            # Unreachable while the caller honours the mask; degrade rather than
            # corrupt state.
            return self._obs(), -0.1, False, {"warning": "repeated view"}

        self._acquire(action)
        if self.step_count < self.B:
            return self._obs(), 0.0, False, {}

        self.done = True
        reward = float(self.utility_fn(self.item, list(self.selected)))
        info = {"n_views": len(self.selected), "views": list(self.selected),
                "category": self.item.get("category", "?"),
                "group_key": (self.model_id, self.B)}
        return self._obs(), reward, True, info


def stack_obs(obs_list):
    """
    Batch observations of differing acquired-set sizes, right-padding to the
    longest. `selected_mask` marks the padding, and the policy's attention mask
    makes it inert -- asserted by test_padding_inert.
    """
    K = max(o["selected_feats"].shape[0] for o in obs_list)
    B = len(obs_list)
    out = {
        "selected_feats": np.zeros((B, K, 2048), np.float32),
        "selected_poses": np.zeros((B, K, POSE_DIM), np.float32),
        "selected_mask": np.zeros((B, K), np.float32),
        "candidate_poses": np.stack([o["candidate_poses"] for o in obs_list]),
        "candidate_mask": np.stack([o["candidate_mask"] for o in obs_list]),
        "budget": np.stack([o["budget"] for o in obs_list]),
    }
    for i, o in enumerate(obs_list):
        k = o["selected_feats"].shape[0]
        out["selected_feats"][i, :k] = o["selected_feats"]
        out["selected_poses"][i, :k] = o["selected_poses"]
        out["selected_mask"][i, :k] = o["selected_mask"]
    return out


def to_tensor(batch, device="cpu"):
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}
