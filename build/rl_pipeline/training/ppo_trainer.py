import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from training.rollout_buffer import RolloutBuffer
from utils.checkpoint import CheckpointManager
from utils.logger import Logger


def _obs_to_tensor(obs: dict, device: str) -> dict:
    return {k: torch.FloatTensor(v).to(device) for k, v in obs.items()}


class PPOTrainer:
    def __init__(self, policy, vec_env, config, resume: bool = False,
                 max_hours: float = None):
        self.policy  = policy
        self.vec_env = vec_env
        self.cfg     = config
        self.device  = config.device
        # Resuming is opt-in. It used to happen whenever a checkpoint file
        # existed, which only looked correct because a fresh run deleted them
        # first; once 'final' checkpoints are preserved, that silently resumed
        # a fresh run from stale weights.
        self.resume  = resume
        # Wall-clock budget for capped environments (Kaggle sessions end at 12h
        # and kill the process). When it runs out we save a resumable checkpoint
        # and stop cleanly, rather than losing the run to a hard kill.
        self.max_hours = max_hours
        self.group_baseline = getattr(config, "group_baseline", False)
        # Starting value; adapted toward config.entropy_target_model in _update().
        self.alpha_model = float(config.entropy_coef_model)
        self.started_at = None

        # Total episodes across both phases, used for the LR schedule.
        self.total_target = config.phase1_episodes + config.phase2_episodes
        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=config.learning_rate
        )
        self.buffer = RolloutBuffer(
            n_steps   = config.n_steps_per_env,
            n_envs    = config.n_envs,
            n_views   = config.n_views,
            n_actions = policy.n_actions,
            device    = config.device,
        )
        self.ckpt   = CheckpointManager(config.checkpoint_dir)
        self.logger = Logger(config)

        self.total_episodes = 0
        self.total_steps    = 0
        self.phase          = 1
        self.current_obs    = None
        self.current_masks  = None   # (n_envs, 24) action masks between rollouts

    # ── Entry point ───────────────────────────────────────────────────────

    def train(self):
        self.started_at = time.time()
        if self.resume:
            self._maybe_resume()

        # Phase 1 may already be complete when resuming mid-run.
        if self.phase == 1:
            print("=== Phase 1: fixed view budget B=5 =====================")
            self.vec_env.set_phase(1)
            self.vec_env.set_budgets([self.cfg.phase1_view_budget] * self.cfg.n_envs)
            self.current_obs, self.current_masks = self.vec_env.reset()
            if self._run_phase(target_episodes=self.cfg.phase1_episodes,
                               phase_start=0):
                return
            self.phase = 2

        print("=== Phase 2: curriculum {3, 5, 8} ======================")
        self.vec_env.set_phase(2)
        self.current_obs, self.current_masks = self.vec_env.reset()
        if self._run_phase(target_episodes=self.cfg.phase2_episodes,
                           phase_start=self.cfg.phase1_episodes):
            return

        print("=== Training complete. ==================================")
        self.ckpt.save(self._state_dict(), tag="final")

    # ── Phase loop ────────────────────────────────────────────────────────

    def _out_of_time(self) -> bool:
        if not self.max_hours or self.started_at is None:
            return False
        return (time.time() - self.started_at) >= self.max_hours * 3600

    def _run_phase(self, target_episodes: int, phase_start: int = 0):
        """Run until the phase target. Returns True if it stopped on the clock.

        `phase_start` is how many episodes precede this phase overall, so a
        resumed run picks up where it left off instead of restarting the phase.
        """
        episodes_at_start = phase_start

        while (self.total_episodes - episodes_at_start) < target_episodes:
            ep_rewards, ep_infos = self._collect_rollout()
            metrics = self._update()

            # ── Per-episode terminal output ───────────────────────────────
            for idx, (r, info) in enumerate(zip(ep_rewards, ep_infos)):
                ep_num = self.total_episodes + idx + 1
                iou    = info.get('iou',      float('nan'))
                views  = info.get('n_views',  '?')
                cat    = info.get('category', '?')
                bb     = info.get('backbone', '?')
                print(
                    f"  ep {ep_num:>6} | "
                    f"reward {r:+.4f} | "
                    f"IoU {iou:.4f} | "
                    f"views {views} | "
                    f"cat {cat} | "
                    f"bb {bb} | "
                    f"pi {metrics['policy_loss']:.4f}  "
                    f"V {metrics['value_loss']:.4f}  "
                    f"H {metrics['entropy']:.3f}"
                )

            self.total_episodes += len(ep_rewards)
            self.logger.record(ep_rewards, ep_infos, metrics, self.total_episodes)

            if self.total_episodes % self.cfg.checkpoint_every < self.cfg.n_envs:
                self.ckpt.save(self._state_dict(), tag=f"ep{self.total_episodes}")
                print(f"  [checkpoint] episode {self.total_episodes}")

            if self._out_of_time():
                self.ckpt.save(self._state_dict(), tag="resume")
                hrs = (time.time() - self.started_at) / 3600
                print(f"\n  [time budget] {hrs:.2f}h of {self.max_hours}h used at "
                      f"episode {self.total_episodes}. Saved ckpt_resume.pt.\n"
                      f"  Continue with:  python train.py --resume\n")
                return True

    # ── Rollout collection ────────────────────────────────────────────────

    def _collect_rollout(self):
        self.buffer.reset()
        self.policy.eval()

        completed_rewards = []
        completed_infos   = []
        obs   = self.current_obs
        masks = self.current_masks   # (n_envs, 24) float32

        print(f"  [rollout] collecting {self.cfg.n_steps_per_env} steps x {self.cfg.n_envs} envs "
              f"(total ep so far: {self.total_episodes}) ...", flush=True)

        for _ in range(self.cfg.n_steps_per_env):
            obs_t  = _obs_to_tensor(obs, self.device)
            mask_t = torch.FloatTensor(masks).to(self.device)

            with torch.no_grad():
                actions, log_probs, values = self.policy.act(obs_t, mask_t)

            next_obs, rewards, dones, next_masks, infos = self.vec_env.step(
                actions.cpu().numpy()
            )

            finished = [infos[i] for i, d in enumerate(dones) if d]
            for i, done in enumerate(dones):
                if done:
                    completed_rewards.append(rewards[i])
                    completed_infos.append(infos[i])

            # Group baselines are only valid if every env really is on the same
            # object and budget. A desync would make the baseline subtract
            # across DIFFERENT objects, removing the signal instead of the
            # noise -- and it would do so silently, which is how three runs were
            # already lost. Check it where it is cheap: at episode end.
            if self.group_baseline and len(finished) > 1:
                keys = {inf.get("group_key") for inf in finished}
                if len(keys) > 1:
                    raise RuntimeError(
                        f"group baseline is on but the parallel envs finished on "
                        f"different objects/budgets: {keys}. They must share a "
                        f"group_sync_seed and stay in lockstep."
                    )

            self.buffer.add(
                obs          = obs,
                actions      = actions.cpu().numpy(),
                log_probs    = log_probs.cpu().numpy(),
                values       = values.cpu().numpy(),
                rewards      = rewards,
                dones        = dones,
                action_masks = masks,
            )

            obs   = next_obs
            masks = next_masks
            self.total_steps += self.cfg.n_envs

        # Bootstrap value after the final stored step
        with torch.no_grad():
            _, last_values = self.policy.forward(
                _obs_to_tensor(obs, self.device)
            )

        self.buffer.compute_gae(
            last_values = last_values.cpu().numpy(),
            last_dones  = dones.astype(np.float32),
            gamma       = self.cfg.gamma,
            gae_lambda  = self.cfg.gae_lambda,
        )
        if self.group_baseline:
            self.buffer.center_by_group()

        self.current_obs   = obs
        self.current_masks = masks
        return completed_rewards, completed_infos

    # ── PPO update ────────────────────────────────────────────────────────

    def _update(self) -> dict:
        self.policy.train()
        # Linear LR anneal to zero across the whole run. Standard PPO practice
        # (CleanRL, SB3) and previously absent: the optimiser ran at a constant
        # 3e-4 for 30k episodes. Once the critic explains most of the return the
        # remaining advantage is largely noise -- normalised to unit variance
        # regardless -- so a constant step size keeps moving the policy on noise.
        if getattr(self.cfg, "anneal_lr", False) and self.total_target > 0:
            frac = 1.0 - min(self.total_episodes / self.total_target, 1.0)
            for g in self.optimizer.param_groups:
                g["lr"] = self.cfg.learning_rate * frac

        policy_losses, value_losses, entropies = [], [], []
        # Tracked separately because the pooled mean hid a collapsed backbone
        # head for three full runs: the view head dominates the average (it is
        # B of every B+1 steps), so a model head at 15% of its maximum entropy
        # barely moved the number that was being logged.
        ent_view, ent_model = [], []

        for _ in range(self.cfg.n_epochs):
            for batch in self.buffer.get_batches(self.cfg.minibatch_size):
                obs_b, actions_b, old_lp_b, adv_b, ret_b, amask_b = batch

                log_probs, values, entropy = self.policy.evaluate_actions(
                    obs_b, actions_b, amask_b
                )

                # Clipped surrogate loss
                ratio    = torch.exp(log_probs - old_lp_b)
                pg_loss1 = -adv_b * ratio
                pg_loss2 = -adv_b * torch.clamp(
                    ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio
                )
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                value_loss = F.mse_loss(values, ret_b)

                # Entropy bonus (maximise entropy → subtract from loss).
                #
                # PER HEAD, not pooled. One coefficient over a mean that mixes a
                # 24-way view distribution with a 3-way backbone distribution
                # cannot regulate both: measured after 30,274 episodes the view
                # head sat at 90% of uniform entropy while the backbone head had
                # collapsed to 0.170 of its 1.099 maximum -- 15% -- picking one
                # backbone in 96.4% of episodes when the optimum is a 7/4/1
                # split. It never explored enough to find that Pix2Vox-F wins
                # every car cell by +0.027 to +0.037, and car is ~24% of the
                # dataset.
                is_model = obs_b['is_model_step'].reshape(-1) > 0.5
                coef = torch.where(
                    is_model,
                    torch.full_like(entropy, self.alpha_model),
                    torch.full_like(entropy, self.cfg.entropy_coef_view),
                )
                entropy_loss = -(coef * entropy).mean()

                loss = policy_loss + self.cfg.value_loss_coef * value_loss + entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.cfg.max_grad_norm
                )
                self.optimizer.step()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
                if (~is_model).any():
                    ent_view.append(entropy[~is_model].mean().item())
                if is_model.any():
                    ent_model.append(entropy[is_model].mean().item())

        # Drive the model head's coefficient toward the target entropy. Raising
        # it whenever the head is too deterministic is what keeps the
        # rarely-optimal backbone sampled often enough to be learnable: a fixed
        # 0.05 left Pix2Vox-F at ~0.3% probability, so across 30,000 episodes it
        # was tried on a car about nine times -- and cars are the one category
        # where it wins.
        h_model = float(np.mean(ent_model)) if ent_model else float('nan')
        if not np.isnan(h_model) and getattr(self.cfg, "entropy_target_model", None):
            err = self.cfg.entropy_target_model - h_model
            self.alpha_model = float(np.clip(
                self.alpha_model + self.cfg.entropy_alpha_lr * err,
                0.0, self.cfg.entropy_alpha_max))
            # A pinned coefficient that still cannot reach the target is a
            # ceiling, and a ceiling nobody notices is how runs get wasted here.
            # alpha_model is in the CSV, but say it out loud once as well.
            if (self.alpha_model >= self.cfg.entropy_alpha_max - 1e-9
                    and h_model < 0.5 * self.cfg.entropy_target_model
                    and not getattr(self, "_warned_alpha_max", False)):
                self._warned_alpha_max = True
                print(f"\n  WARNING: entropy_alpha_max ({self.cfg.entropy_alpha_max}) "
                      f"is saturated and the model head is still at H={h_model:.3f} "
                      f"vs target {self.cfg.entropy_target_model}.\n"
                      f"  The backbone head is not exploring enough to learn a "
                      f"per-category rule. Raise entropy_alpha_max.\n")

        return {
            'policy_loss': float(np.mean(policy_losses)),
            'value_loss':  float(np.mean(value_losses)),
            'lr':          float(self.optimizer.param_groups[0]["lr"]),
            'alpha_model': self.alpha_model,
            'entropy':     float(np.mean(entropies)),
            'entropy_view':  float(np.mean(ent_view)) if ent_view else float('nan'),
            'entropy_model': float(np.mean(ent_model)) if ent_model else float('nan'),
        }

    # ── Helpers ───────────────────────────────────────────────────────────

    def _state_dict(self) -> dict:
        return {
            'policy':         self.policy.state_dict(),
            'optimizer':      self.optimizer.state_dict(),
            'total_episodes': self.total_episodes,
            'total_steps':    self.total_steps,
            'phase':          self.phase,
            # Without this a resumed session restarts exploration from the
            # initial coefficient, undoing however far the adaptation had got.
            'alpha_model':    self.alpha_model,
        }

    def _maybe_resume(self):
        state = self.ckpt.load_latest()
        if state is None:
            self.current_obs, self.current_masks = self.vec_env.reset()
            return
        try:
            self.policy.load_state_dict(state['policy'])
        except RuntimeError as exc:
            raise RuntimeError(
                f"Checkpoint does not fit the current policy:\n{exc}\n\n"
                "This happens when the action space or the head layout changed "
                "since it was saved -- a different view count, or a checkpoint "
                "from before backbone selection existed. Re-run with "
                "--load-view-head-only to carry over every tensor that still "
                "matches and start the model head fresh."
            ) from None
        self.optimizer.load_state_dict(state['optimizer'])
        self.total_episodes = state['total_episodes']
        self.total_steps    = state['total_steps']
        self.phase          = state['phase']
        # Without this a resumed session restarts the entropy adaptation from
        # its initial value, discarding however far it had climbed. Older
        # checkpoints have no such key, so fall back to the configured start.
        self.alpha_model    = state.get('alpha_model', self.alpha_model)
        self.current_obs, self.current_masks = self.vec_env.reset()
        print(f"  Resumed from episode {self.total_episodes} (phase {self.phase})")