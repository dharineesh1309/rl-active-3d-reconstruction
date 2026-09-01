import os
import csv
import numpy as np


class Logger:
    """
    Writes two CSVs:
      logs/episode_metrics.csv  — reward + PPO losses per log interval
      logs/view_entropy.csv     — entropy over time (proxy for exploration)
    """

    def __init__(self, cfg):
        # Takes the whole config object, not loose values: the previous version
        # re-imported the Config *class* inside record(), so instance overrides
        # such as --smoke-test's log_every were silently ignored.
        self.cfg        = cfg
        self.log_dir    = cfg.log_dir
        self.categories = cfg.categories
        os.makedirs(self.log_dir, exist_ok=True)

        # A resumed run appends to an existing file. If that file was written
        # by an older build its header has fewer columns, so the new rows would
        # silently misalign -- the kind of quietly corrupt artifact that is only
        # noticed when a plot looks wrong. Roll the old file aside instead.
        csv_path = os.path.join(self.log_dir, "episode_metrics.csv")
        if os.path.isfile(csv_path) and os.path.getsize(csv_path) > 0:
            with open(csv_path, newline='') as _f:
                first = _f.readline().strip().split(',')
            if first != ['total_episodes', 'mean_reward', 'std_reward',
                         'policy_loss', 'value_loss', 'entropy',
                         'entropy_view', 'entropy_model', 'lr', 'alpha_model']:
                bak = csv_path + ".old"
                os.replace(csv_path, bak)
                print(f"  [logger] metrics header changed; previous CSV moved to {bak}")

        self._ep_file  = open(csv_path, 'a', newline='')
        self._ep_writer = csv.DictWriter(
            self._ep_file,
            # entropy_view / entropy_model are separate on purpose. The pooled
            # 'entropy' hid a collapsed backbone head for three full runs: the
            # view head is B of every B+1 steps so it dominates the mean, and a
            # model head at 15% of its maximum barely moved the logged number.
            fieldnames=['total_episodes', 'mean_reward', 'std_reward',
                        'policy_loss', 'value_loss', 'entropy',
                        'entropy_view', 'entropy_model', 'lr', 'alpha_model']
        )
        if self._ep_file.tell() == 0:
            self._ep_writer.writeheader()

        # Rolling buffers
        self._rewards: list = []
        self._last_logged = 0

    def record(self, ep_rewards: list, ep_infos: list,
               update_metrics: dict, total_episodes: int):
        self._rewards.extend(ep_rewards)

        # Interval measured against the LAST logged episode, not modulo.
        #
        # `record()` is called once per rollout, and one rollout of
        # n_steps_per_env * n_envs steps holds a few hundred episodes -- so
        # total_episodes jumps by ~300 at a time. The old test,
        # `total_episodes % log_every < n_envs`, assumed increments of about
        # n_envs and therefore landed inside its own window only ~8% of the
        # time: a 30,288-episode run wrote TWO rows instead of ~300, and every
        # diagnosis in this project was made from 2-11 data points.
        if (total_episodes - self._last_logged) >= self.cfg.log_every and self._rewards:
            self._flush(update_metrics, total_episodes)
            self._last_logged = total_episodes

    def _flush(self, metrics: dict, total_episodes: int):
        rewards = np.array(self._rewards)
        mean_r  = float(rewards.mean()) if len(rewards) else 0.0
        std_r   = float(rewards.std())  if len(rewards) else 0.0

        print(
            f"  ep {total_episodes:>7} | "
            f"reward {mean_r:+.4f} +/- {std_r:.4f} | "
            f"pi-loss {metrics['policy_loss']:.4f} | "
            f"V-loss {metrics['value_loss']:.4f} | "
            f"H {metrics['entropy']:.3f} "
            f"(view {metrics.get('entropy_view', float('nan')):.3f} / "
            f"model {metrics.get('entropy_model', float('nan')):.3f})"
        )

        self._ep_writer.writerow({
            'total_episodes': total_episodes,
            'mean_reward':    round(mean_r, 5),
            'std_reward':     round(std_r,  5),
            'policy_loss':    round(metrics['policy_loss'], 5),
            'value_loss':     round(metrics['value_loss'],  5),
            'entropy':        round(metrics['entropy'],     5),
            'entropy_view':   round(metrics.get('entropy_view',  float('nan')), 5),
            'entropy_model':  round(metrics.get('entropy_model', float('nan')), 5),
            'lr':             round(metrics.get('lr', float('nan')), 8),
            'alpha_model':    round(metrics.get('alpha_model', float('nan')), 5),
        })
        self._ep_file.flush()
        self._rewards = []

    def close(self):
        self._ep_file.close()