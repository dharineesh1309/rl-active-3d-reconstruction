import numpy as np
import torch


class RolloutBuffer:
    """
    Stores n_steps × n_envs transitions, then computes GAE.

    Tracks `is_model_step` alongside each transition. Without it the PPO update
    could not tell which actor head produced a stored action, and would score
    backbone choices through the view head — silently wrong, and it would look
    like a training-instability problem rather than a bookkeeping bug.
    """

    def __init__(self, n_steps: int, n_envs: int, n_views: int,
                 n_actions: int, device: str):
        self.n_steps   = n_steps
        self.n_envs    = n_envs
        self.n_views   = n_views     # 24
        self.n_actions = n_actions   # max(n_views, n_backbones)
        self.device    = device
        self._alloc()

    def _alloc(self):
        N, E, V, A = self.n_steps, self.n_envs, self.n_views, self.n_actions

        # Observations (numpy until minibatch is sent to GPU)
        self.cov_grids   = np.zeros((N, E, 32, 32, 32), dtype=np.float32)
        self.img_feats   = np.zeros((N, E, 512),        dtype=np.float32)
        self.hist_masks  = np.zeros((N, E, V),          dtype=np.float32)
        self.model_flags = np.zeros((N, E, 1),          dtype=np.float32)
        self.budgets     = np.zeros((N, E, 2),          dtype=np.float32)

        # Actions and masks
        self.actions      = np.zeros((N, E),    dtype=np.int64)
        self.action_masks = np.zeros((N, E, A), dtype=np.float32)

        # PPO scalars
        self.log_probs = np.zeros((N, E), dtype=np.float32)
        self.values    = np.zeros((N, E), dtype=np.float32)
        self.rewards   = np.zeros((N, E), dtype=np.float32)
        self.dones     = np.zeros((N, E), dtype=np.float32)

        # Filled by compute_gae()
        self.advantages = np.zeros((N, E), dtype=np.float32)
        self.returns    = np.zeros((N, E), dtype=np.float32)

        self.ptr  = 0
        self.full = False

    def reset(self):
        self._alloc()

    def add(self, obs, actions, log_probs, values,
            rewards, dones, action_masks):
        t = self.ptr
        self.cov_grids[t]   = obs['coverage_grid']
        self.img_feats[t]   = obs['image_features']
        self.hist_masks[t]  = obs['view_mask']
        self.model_flags[t] = obs['is_model_step']
        self.budgets[t]     = obs['budget']
        self.actions[t]     = actions
        self.log_probs[t]   = log_probs
        self.values[t]      = values
        self.rewards[t]     = rewards
        self.dones[t]       = dones.astype(np.float32)
        self.action_masks[t] = action_masks

        self.ptr += 1
        if self.ptr == self.n_steps:
            self.full = True

    def compute_gae(self, last_values: np.ndarray, last_dones: np.ndarray,
                    gamma: float, gae_lambda: float):
        gae = np.zeros(self.n_envs, dtype=np.float32)

        for t in reversed(range(self.n_steps)):
            # `dones[t]` is what vec_env.step() returned for action t: "the
            # episode ended after this action". So the flag that decides whether
            # step t bootstraps is dones[t], NOT dones[t+1].
            #
            # Reading dones[t+1] here put the episode boundary one step early,
            # with two compounding effects on a B+1 horizon:
            #   * step B (the last VIEW step) was treated as terminal, so the
            #     terminal reward never propagated back to any view step -- the
            #     view head trained on a constant negative advantage and could
            #     not learn view selection at all.
            #   * step B+1 (the model step, the real terminal) was treated as
            #     non-terminal, so it bootstrapped V from the first state of the
            #     NEXT episode, inflating its return by gamma*V (a 49% error in
            #     a worked example: 1.4903 against a true 1.0).
            # Note the t == n_steps-1 branch below always used dones[t], which
            # is what the general case should have done too.
            next_non_terminal = 1.0 - self.dones[t]
            if t == self.n_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            delta = (self.rewards[t]
                     + gamma * next_values * next_non_terminal
                     - self.values[t])
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae

        self.returns = self.advantages + self.values

    def center_by_group(self):
        """
        Leave-one-out centring of advantages across the parallel envs.

        Every env in the group runs the SAME object with the SAME budget, so at
        any timestep the only thing separating their advantages is which views
        and backbone each one chose. Subtracting the mean of the *other* envs
        therefore removes the object-difficulty term exactly, where a learned
        critic only removes it approximately.

        This is the measured bottleneck: within-object spread from view choice
        is 0.027 IoU while between-object spread is 0.062, and even with the
        critic at 0.88 explained variance the residual noise still outweighs the
        signal 4.4 to 1.

        Leave-one-out rather than the plain group mean because env i's own
        advantage depends on the action it took; including it in its own
        baseline biases the gradient by 1/n. GRPO uses the full mean and
        tolerates that, but excluding self is free here.

        `returns` are deliberately NOT recentred -- they are the critic's
        regression target and must stay an unbiased estimate of V.
        """
        n = self.n_envs
        if n < 2:
            return
        total = self.advantages.sum(axis=1, keepdims=True)
        loo = (total - self.advantages) / (n - 1)
        self.advantages = self.advantages - loo

    def get_batches(self, minibatch_size: int):
        assert self.full, "Buffer not full — call add() n_steps times first."
        N  = self.n_steps * self.n_envs
        V  = self.n_views
        A  = self.n_actions

        # Flatten
        cov   = self.cov_grids.reshape(N, 32, 32, 32)
        feat  = self.img_feats.reshape(N, 512)
        hist  = self.hist_masks.reshape(N, V)
        mflag = self.model_flags.reshape(N, 1)
        budg  = self.budgets.reshape(N, 2)
        acts  = self.actions.reshape(N)
        lps   = self.log_probs.reshape(N)
        advs  = self.advantages.reshape(N)
        rets  = self.returns.reshape(N)
        amask = self.action_masks.reshape(N, A)

        # Normalise advantages
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        indices = np.random.permutation(N)
        for start in range(0, N, minibatch_size):
            idx = indices[start: start + minibatch_size]
            obs_batch = {
                'coverage_grid':  torch.FloatTensor(cov[idx]).to(self.device),
                'image_features': torch.FloatTensor(feat[idx]).to(self.device),
                'view_mask':      torch.FloatTensor(hist[idx]).to(self.device),
                'is_model_step':  torch.FloatTensor(mflag[idx]).to(self.device),
                'budget':         torch.FloatTensor(budg[idx]).to(self.device),
            }
            yield (
                obs_batch,
                torch.LongTensor(acts[idx]).to(self.device),
                torch.FloatTensor(lps[idx]).to(self.device),
                torch.FloatTensor(advs[idx]).to(self.device),
                torch.FloatTensor(rets[idx]).to(self.device),
                torch.FloatTensor(amask[idx]).to(self.device),
            )