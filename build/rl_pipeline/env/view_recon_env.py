import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env.state_builder import CoverageGrid, ImageFeatureExtractor, ViewHistoryMask


class ViewReconEnv(gym.Env):
    """
    Joint view-selection and backbone-selection environment.

    The episode is the finite-horizon MDP the report describes, H = B + 1:

        steps 1..B   pick a viewpoint from the n_views candidates (no revisits)
        step  B+1    pick which reconstruction backbone scores the result

    Reward is sparse. Every view step returns 0; the single terminal reward is

        reward = IoU(pred, gt) - lambda_cost * (B + backbone.cost)

    so the agent is paid for reconstruction quality and charged for both the
    views it spent and the expense of the backbone it chose. That cost term is
    the whole reason backbone selection is a decision rather than "always pick
    the biggest model".

    Observations
    ────────────
    coverage_grid  (32,32,32) accumulated occupancy from the chosen views
    image_features (512,)     mean-pooled frozen ResNet-50 features
    view_mask      (n_views,) 1.0 where a view has been taken
    is_model_step  (1,)       1.0 on the terminal backbone-selection step

    `is_model_step` is what lets one policy drive two heads: the network reads
    it to decide whether an action indexes a viewpoint or a backbone.

    Actions
    ───────
    A single Discrete(max(n_views, n_backbones)) space. Which entries are legal
    changes per step and is published through get_action_mask(); the alternative
    (two separate action spaces) would need a branch in the rollout buffer and
    in every vector-env wrapper for no gain.

    Parameters
    ----------
    dataset : sequence of dict
        Items as produced by dataloader.py or dataloader_shapenet.py.
    backbones : list[Backbone] | None
        Reconstruction models the agent chooses between, in a fixed order that
        defines the action indices. Must be built inside the worker process —
        torch models cannot be pickled across a multiprocessing Pipe.
        None puts the env in mock-reward mode for unit tests.
    view_budget : int
        B, the number of views selectable per episode.
    lambda_cost : float
        Coefficient on the compute penalty.
    n_views : int
        Number of candidate viewpoints (24, the 3D-R2N2 protocol).
    """

    def __init__(
        self,
        dataset,
        backbones=None,
        view_budget: int = 5,
        lambda_cost: float = 0.05,
        shaping_coef: float = 0.0,
        seed_initial_view: bool = False,
        gamma: float = 0.99,
        group_sync_seed: int = None,
        phase2_budgets = (3, 5, 8),
        n_views: int = 24,
        device: str = "cpu",
    ):
        super().__init__()

        self.dataset = dataset
        self.B = view_budget
        self.lambda_cost = lambda_cost
        self.shaping_coef = shaping_coef
        self.seed_initial_view = seed_initial_view
        self.gamma = gamma
        # Group-relative baselines need every parallel env on the SAME object
        # and budget, so the object-difficulty term cancels exactly when the
        # group mean is subtracted. Rather than plumbing reset options through
        # the vec-env's auto-reset, each env draws its object and budget from a
        # dedicated RNG seeded identically across the group. Actions still
        # diverge -- the policy samples stochastically -- and episodes are always
        # exactly B+1 steps, so the envs stay in lockstep.
        #
        # The trainer asserts the group really is on one object. A silent
        # desync would make the baseline subtract across DIFFERENT objects,
        # which removes the signal instead of the noise.
        # Was hard-coded to [3, 5, 8] inside reset(), so Config.phase2_view_budgets
        # was silently ignored -- the corrected [2, 4, 7] would never have taken
        # effect. Passed in now.
        self._phase2_budgets = list(phase2_budgets)
        self._sync = (np.random.RandomState(group_sync_seed)
                      if group_sync_seed is not None else None)
        self.n_views = n_views
        self.backbones = backbones or []
        self.n_backbones = max(len(self.backbones), 1)   # mock mode still needs one slot

        # ── RL state components ───────────────────────────────────────────────
        self.coverage_grid = CoverageGrid()
        self.feature_extractor = ImageFeatureExtractor(device=device)
        self.view_mask = ViewHistoryMask(self.n_views)

        # ── Observation / action spaces ───────────────────────────────────────
        self.observation_space = spaces.Dict({
            "coverage_grid":  spaces.Box(0, 1, shape=(32, 32, 32), dtype=np.float32),
            "image_features": spaces.Box(-np.inf, np.inf, shape=(512,), dtype=np.float32),
            "view_mask":      spaces.Box(0, 1, shape=(n_views,), dtype=np.float32),
            "is_model_step":  spaces.Box(0, 1, shape=(1,), dtype=np.float32),
            # [total budget, views still unspent], both scaled by n_views.
            #
            # Without this the agent cannot tell a B=3 episode from a B=8 one:
            # view_mask is all zeros at the first step either way. The terminal
            # reward carries -lambda*B, which differs by 0.35 between those two
            # budgets, so the critic had to average over budgets it could not
            # observe. Measured over a 30,274-episode run, explained variance
            # fell from 0.74 in phase 1 (fixed B) to 0.06 as soon as phase 2
            # started sampling budgets -- the advantages became mostly "which
            # budget did I draw" rather than "was that a good view", and the
            # policy stayed at 89% of maximum entropy.
            "budget":         spaces.Box(0, 1, shape=(2,), dtype=np.float32),
        })
        self.n_actions = max(self.n_views, self.n_backbones)
        self.action_space = spaces.Discrete(self.n_actions)

        # ── Episode state ─────────────────────────────────────────────────────
        self.current_object = None
        self.step_count = 0
        self.done = False
        self.selected_view_indices = []
        self.selected_backbone = None

    # ── Phase / budget control ────────────────────────────────────────────────

    def set_phase(self, phase: int):
        """Phase 1 -> fixed budget. Phase 2 -> budget resampled each reset."""
        self._phase = phase

    def set_view_budget(self, budget: int):
        self.B = budget

    # ── reset ─────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        rng = self._sync if self._sync is not None else np.random
        if getattr(self, "_phase", 1) == 2:
            self.B = int(rng.choice(self._phase2_budgets))

        idx = int(rng.randint(len(self.dataset)))
        self.current_object = self.dataset[idx]

        self.coverage_grid.reset()
        self.feature_extractor.reset()
        self.view_mask.reset()
        self.step_count = 0
        self.done = False
        self.selected_view_indices = []
        self.selected_backbone = None

        # One free view before the first decision. Without it the initial state
        # is identical for every object, so V(s_0) cannot express object
        # difficulty and that advantage carries the full between-object spread.
        if self.seed_initial_view:
            self._apply_view(int(np.random.randint(self.n_views)))
            # The seeded view is a gift, not a decision: it must not consume
            # budget, or B=3 would really be 2 chosen views.
            self.step_count = 0

        self._prev_potential = self._potential()
        return self._get_obs(), {}

    def _potential(self) -> float:
        """
        Phi(s) for potential-based shaping: fraction of the coverage grid filled.

        A proxy for "how much of the object have I seen". Only meaningful because
        the coverage grid is a real back-projection; against the old image-plane
        stencil this number was close to noise.
        """
        return float((self.coverage_grid.get() > 0.5).mean())

    def _apply_view(self, action):
        """Fold one view into the state. Shared by reset() and _view_step()."""
        cams = self.current_object.get("cams")
        self.coverage_grid.update(
            self.current_object["depths"][action],
            self.current_object["silhouettes"][action],
            cam=cams[action] if cams else None,
        )
        self.feature_extractor.add_view(self.current_object["images"][action])
        self.view_mask.mark(action)
        self.selected_view_indices.append(action)
        self.step_count += 1

    # ── step ──────────────────────────────────────────────────────────────────

    def step(self, action):
        assert not self.done, "Episode done — call reset() first."
        action = int(action)

        if self._is_model_step():
            return self._model_step(action)
        return self._view_step(action)

    def _view_step(self, action):
        assert 0 <= action < self.n_views, f"Invalid view action {action}."

        # Unreachable while the caller honours the action mask; kept so an
        # unmasked caller degrades to a penalty instead of corrupting state.
        if action in self.view_mask.visited():
            return self._get_obs(), -0.1, False, False, {"warning": "repeated view"}

        self._apply_view(action)

        # Potential-based shaping: gamma*Phi(s') - Phi(s). Summed over an
        # episode these telescope to gamma^T*Phi(s_T) - Phi(s_0), so the return
        # is unchanged up to a constant and the optimal policy is provably the
        # same (Ng, Harada & Russell 1999). What changes is that each view step
        # now gets credit for the coverage IT added, instead of B steps sharing
        # one terminal number.
        shaped = 0.0
        if self.shaping_coef:
            phi = self._potential()
            shaped = self.shaping_coef * (self.gamma * phi - self._prev_potential)
            self._prev_potential = phi

        return self._get_obs(), float(shaped), False, False, {}

    def _model_step(self, action):
        assert 0 <= action < self.n_backbones, f"Invalid backbone action {action}."

        self.selected_backbone = action
        self.done = True

        iou = self._reconstruct_iou(action)
        cost = self._compute_cost(action)
        name = self.backbones[action].name if self.backbones else "mock"

        info = {
            "category": self.current_object.get("category", "unknown"),
            "iou": round(iou, 5),
            "n_views": len(self.selected_view_indices),
            "backbone": name,
            "backbone_idx": action,
        }
        info["group_key"] = (self.current_object.get("model_id", "?"), self.B)
        # The terminal shaping term. By convention Phi(terminal) = 0, so this is
        # gamma*0 - Phi(s_B) = -Phi(s_B).
        #
        # It is NOT optional. With it, the episode's shaping terms sum to
        # -Phi(s_0), which does not depend on any action the agent took, so the
        # optimal policy is unchanged. Omit it and they sum to Phi(s_B) - Phi(s_0),
        # which DOES depend on the agent's choices -- that would quietly turn the
        # objective into "maximise final coverage", a correlate of IoU rather
        # than IoU itself.
        shaped = 0.0
        if self.shaping_coef:
            shaped = -self.shaping_coef * self._prev_potential
            self._prev_potential = 0.0
        info["shaping"] = round(shaped, 5)
        return self._get_obs(), float(iou - cost + shaped), True, False, info

    # ── helpers ───────────────────────────────────────────────────────────────

    def _is_model_step(self) -> bool:
        """True once every view in the budget has been spent."""
        return self.step_count >= self.B

    def _get_obs(self):
        return {
            "coverage_grid":  self.coverage_grid.get(),
            "image_features": self.feature_extractor.get(),
            "view_mask":      self.view_mask.get(),
            "is_model_step":  np.array([float(self._is_model_step())], dtype=np.float32),
            # Remaining is counted in DECISIONS (step_count), not in views held.
            # With seed_initial_view the two differ: selected_view_indices
            # already contains the free seed, so `B - len(selected)` would read
            # B-1 at the first decision and go NEGATIVE at the model step,
            # breaking this field's Box(0, 1) bounds and disagreeing with
            # evaluate.py. step_count is the number of choices the agent has
            # made, so this runs B -> 0 regardless of seeding.
            "budget":         np.array(
                [self.B / self.n_views,
                 max(self.B - self.step_count, 0) / self.n_views],
                dtype=np.float32),
        }

    def _compute_cost(self, backbone_idx: int) -> float:
        """Views spent plus the chosen backbone's relative expense."""
        backbone_cost = self.backbones[backbone_idx].cost if self.backbones else 0.0
        return self.lambda_cost * (len(self.selected_view_indices) + backbone_cost)

    def _reconstruct_iou(self, backbone_idx: int) -> float:
        """
        Run the chosen backbone on the selected views and score it against the
        ground-truth occupancy grid. Returns 0.5 in mock mode (no backbones
        loaded), which is what the unit tests run against.
        """
        if not self.backbones:
            return 0.5

        from backbones import voxel_iou

        images = [self.current_object["images"][i] for i in self.selected_view_indices]

        # Pose-dependent backbones (pixelNeRF) need the cameras for exactly the
        # views the agent chose; voxel backbones ignore the argument. Datasets
        # without camera records simply pass None.
        all_cams = self.current_object.get("cams")
        cams = ([all_cams[i] for i in self.selected_view_indices]
                if all_cams is not None else None)

        pred = self.backbones[backbone_idx].predict(images, cams)
        gt = self.current_object["voxels"]

        if pred.shape != gt.shape:
            raise ValueError(
                f"Backbone '{self.backbones[backbone_idx].name}' returned {pred.shape} "
                f"but the ground truth is {gt.shape}. Every backbone must emit "
                f"32**3 and the dataset must supply 32**3 ground truth "
                f"(ShapeNetVox32); a 64**3 dataset cannot be scored here."
            )
        return voxel_iou(pred, gt)

    def get_action_mask(self) -> np.ndarray:
        """
        Boolean array over the action space: True = legal this step.

        View steps expose the unvisited viewpoints; the terminal step exposes
        the available backbones. Always has at least one True entry, so the
        masked logits can never be uniformly -inf.
        """
        mask = np.zeros(self.n_actions, dtype=bool)
        if self._is_model_step():
            mask[:self.n_backbones] = True
        else:
            mask[:self.n_views] = True
            for idx in self.view_mask.visited():
                mask[idx] = False
        return mask
