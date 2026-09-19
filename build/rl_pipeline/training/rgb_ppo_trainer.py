"""
PPO for the RGB pose policy.

Same algorithm as the previous trainer -- clipped surrogate, GAE, entropy bonus,
linear LR anneal -- against the new observation interface. The differences are
all consequences of decisions already measured:

  * One actor head. Backbone routing left the RL loop once the counterfactual
    utilities turned out to be computable offline, so the terminal reward is the
    envelope max_m U_m(S) and a good view set is never punished for a routing
    mistake a learning head happens to make.
  * Environments are stepped in-process. The reconstruction backbones are
    invoked once per episode by the reward function rather than living inside
    each worker, so there is nothing to pickle across a process boundary.
  * Advantages are centred leave-one-out across envs that share an object.
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

from env.rgb_view_env import stack_obs, to_tensor
from training.rgb_rollout_buffer import RGBRolloutBuffer


class RGBPPOTrainer:
    def __init__(self, policy, envs, cfg, device: str = "cpu",
                 on_episode=None):
        self.policy = policy.to(device)
        self.envs = envs
        self.cfg = cfg
        self.device = device
        self.on_episode = on_episode          # callback for logging/eval hooks
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
        self.buffer = RGBRolloutBuffer(cfg.n_steps_per_env, len(envs), device)
        self.total_episodes = 0
        self.total_steps = 0
        self.total_target = cfg.phase1_episodes + cfg.phase2_episodes
        self.obs = [e.reset() for e in envs]

    # ── rollout ──────────────────────────────────────────────────────────────

    def collect(self):
        self.buffer.reset()
        self.policy.eval()
        ep_rewards, ep_infos = [], []

        for _ in range(self.cfg.n_steps_per_env):
            batch = to_tensor(stack_obs(self.obs), self.device)
            with torch.no_grad():
                actions, log_probs, values = self.policy.act(batch)
            a = actions.cpu().numpy()

            next_obs, rewards, dones = [], [], []
            step_keys = []
            for i, env in enumerate(self.envs):
                o, r, d, info = env.step(int(a[i]))
                if d:
                    ep_rewards.append(r)
                    ep_infos.append(info)
                    step_keys.append(info["group_key"])
                    o = env.reset()
                next_obs.append(o)
                rewards.append(r)
                dones.append(d)

            # Compare envs finishing on the SAME step. Checking every key in the
            # rollout instead would flag a healthy run: a 24-step rollout of
            # 3-step episodes legitimately visits eight different objects.
            if getattr(self.cfg, "group_baseline", False) and len(step_keys) > 1:
                if len(set(step_keys)) > 1:
                    raise RuntimeError(
                        f"group_baseline is on but envs finished the same "
                        f"episode on different objects/budgets: "
                        f"{sorted(set(step_keys))[:4]}. They must share a "
                        f"group_sync_seed and stay in lockstep.")

            self.buffer.add(self.obs, a, log_probs.cpu().numpy(),
                            values.cpu().numpy(), np.array(rewards, np.float32),
                            dones)
            self.obs = next_obs
            self.total_steps += len(self.envs)

        batch = to_tensor(stack_obs(self.obs), self.device)
        with torch.no_grad():
            _, last_values = self.policy(batch)
        self.buffer.compute_gae(last_values.cpu().numpy(),
                                np.array([0.0] * len(self.envs), np.float32),
                                self.cfg.gamma, self.cfg.gae_lambda)
        if getattr(self.cfg, "group_baseline", False):
            self.buffer.center_by_group()

        self.total_episodes += len(ep_rewards)
        return ep_rewards, ep_infos

    # ── update ───────────────────────────────────────────────────────────────

    def update(self):
        if getattr(self.cfg, "anneal_lr", False) and self.total_target > 0:
            frac = 1.0 - min(self.total_episodes / self.total_target, 1.0)
            for g in self.opt.param_groups:
                g["lr"] = self.cfg.learning_rate * frac

        self.policy.train()
        pl, vl, ent = [], [], []
        for _ in range(self.cfg.n_epochs):
            for obs, acts, old_lp, adv, ret in self.buffer.batches(
                    self.cfg.minibatch_size):
                lp, value, entropy = self.policy.evaluate_actions(obs, acts)
                ratio = torch.exp(lp - old_lp)
                l1 = -adv * ratio
                l2 = -adv * torch.clamp(ratio, 1 - self.cfg.clip_ratio,
                                        1 + self.cfg.clip_ratio)
                policy_loss = torch.max(l1, l2).mean()
                value_loss = F.mse_loss(value, ret)
                loss = (policy_loss
                        + self.cfg.value_loss_coef * value_loss
                        - self.cfg.entropy_coef_view * entropy.mean())
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(),
                                               self.cfg.max_grad_norm)
                self.opt.step()
                pl.append(policy_loss.item())
                vl.append(value_loss.item())
                ent.append(entropy.mean().item())

        return {"policy_loss": float(np.mean(pl)),
                "value_loss": float(np.mean(vl)),
                "entropy": float(np.mean(ent)),
                "lr": float(self.opt.param_groups[0]["lr"])}

    # ── driver ───────────────────────────────────────────────────────────────

    def train(self, target_episodes: int, phase: int = 1, max_hours=None):
        for e in self.envs:
            e.set_phase(phase)
        started = time.time()
        while self.total_episodes < target_episodes:
            rewards, infos = self.collect()
            metrics = self.update()
            if self.on_episode:
                self.on_episode(self.total_episodes, rewards, infos, metrics)
            if max_hours and (time.time() - started) / 3600 > max_hours:
                print(f"  [time budget] stopping at episode {self.total_episodes}")
                return False
        return True
