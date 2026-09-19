"""
Rollout storage for the RGB pose policy.

The previous buffer allocated fixed-shape arrays for a fixed-shape observation.
This one holds observations whose acquired-set length grows within an episode
(1 up to B+1) and whose budget varies between episodes, so it stores per-step
dicts and pads only when a minibatch is assembled.

Carries forward two things that were fixed the hard way in the old buffer:

  * GAE uses `dones[t]`, the flag vec_env returned for action t. Reading
    `dones[t+1]` put the episode boundary one step early, which on a B+1 horizon
    meant the terminal reward never reached any view step at all.
  * Advantages are centred leave-one-out across the parallel environments, which
    all replay the same object. Object difficulty then cancels exactly rather
    than approximately. `returns` are deliberately left uncentred -- they are the
    critic's regression target and must stay unbiased estimates of V.
"""

import numpy as np
import torch

from env.rgb_view_env import stack_obs


class RGBRolloutBuffer:
    def __init__(self, n_steps: int, n_envs: int, device: str = "cpu"):
        self.n_steps, self.n_envs, self.device = n_steps, n_envs, device
        self.reset()

    def reset(self):
        self.obs = [[None] * self.n_envs for _ in range(self.n_steps)]
        z = lambda: np.zeros((self.n_steps, self.n_envs), dtype=np.float32)
        self.actions = np.zeros((self.n_steps, self.n_envs), dtype=np.int64)
        self.log_probs, self.values = z(), z()
        self.rewards, self.dones = z(), z()
        self.advantages, self.returns = z(), z()
        self.ptr = 0

    def add(self, obs_list, actions, log_probs, values, rewards, dones):
        t = self.ptr
        self.obs[t] = list(obs_list)
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.values[t] = values
        self.rewards[t] = rewards
        self.dones[t] = np.asarray(dones, dtype=np.float32)
        self.ptr += 1

    def compute_gae(self, last_values, last_dones, gamma: float, lam: float):
        gae = np.zeros(self.n_envs, dtype=np.float32)
        for t in reversed(range(self.ptr)):
            # dones[t] is "the episode ended after action t". Using dones[t+1]
            # here is the off-by-one that cost three training runs.
            non_terminal = 1.0 - self.dones[t]
            next_values = last_values if t == self.ptr - 1 else self.values[t + 1]
            delta = self.rewards[t] + gamma * next_values * non_terminal - self.values[t]
            gae = delta + gamma * lam * non_terminal * gae
            self.advantages[t] = gae
        self.returns = self.advantages + self.values

    def center_by_group(self):
        """
        Leave-one-out centring across the parallel envs, which share an object.

        Excludes each env's own advantage from its baseline: including it biases
        the gradient by 1/n. Cheap to do and strictly better than the plain group
        mean.
        """
        n = self.n_envs
        if n < 2:
            return
        a = self.advantages[:self.ptr]
        loo = (a.sum(axis=1, keepdims=True) - a) / (n - 1)
        self.advantages[:self.ptr] = a - loo

    def batches(self, minibatch_size: int):
        N = self.ptr * self.n_envs
        flat_obs = [self.obs[t][e] for t in range(self.ptr) for e in range(self.n_envs)]
        acts = self.actions[:self.ptr].reshape(N)
        lps = self.log_probs[:self.ptr].reshape(N)
        advs = self.advantages[:self.ptr].reshape(N)
        rets = self.returns[:self.ptr].reshape(N)

        advs = (advs - advs.mean()) / (advs.std() + 1e-8)
        order = np.random.permutation(N)      # one shuffle per pass, not per slice
        for start in range(0, N, minibatch_size):
            idx = order[start:start + minibatch_size]
            batch = stack_obs([flat_obs[i] for i in idx])
            yield (
                {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()},
                torch.as_tensor(acts[idx], device=self.device),
                torch.as_tensor(lps[idx], device=self.device),
                torch.as_tensor(advs[idx], device=self.device),
                torch.as_tensor(rets[idx], device=self.device),
            )
