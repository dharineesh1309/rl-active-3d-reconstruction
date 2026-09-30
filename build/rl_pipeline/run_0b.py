"""
Tier 0B: does pose conditioning actually fix view selection?

    python run_0b.py --arm set_pose  --steps 200000
    python run_0b.py --arm set_index --steps 200000
    python run_0b.py --arm mean_pose --steps 200000

Three arms, identical in every respect except the two flags that define them:

    set_pose    set encoder + pose-conditioned scorer   <- proposed
    set_index   set encoder + index head                <- same encoder, old
                                                           action abstraction
    mean_pose   mean pooling + pose scorer              <- isolates the encoder

`set_index` vs `set_pose` is the experiment. Comparing the old RGB-D policy
against the new one would confound the action abstraction with the sensor
contract, the image projection and the encoder all at once; holding the encoder
fixed and changing only the head answers "was it the indices?" directly. The
arms differ by 2.6% in parameter count, so a gap between them is not capacity.

Budgeted in ENVIRONMENT STEPS, not episodes. The arms have slightly different
per-episode costs, and equal episode counts would hand one of them more
experience.

Reward is the envelope max_m [IoU_m(S) - lambda*cost_m], so no arm is ever
penalised for a routing decision: routing is not in this loop.

Held-out evaluation, every --eval-every steps, on objects never trained on:

    policy      the arm's own views
    random      the same number of views drawn at random
    oracle      best of --eval-subsets random subsets of that size

reporting (policy - random) and the fraction of (oracle - random) captured. That
fraction is the real measure: entropy falling only says the policy became less
uniform, not that it became better.
"""

import argparse
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ARMS = {
    "set_pose":  dict(set_encoder=True,  pose_head=True),
    "set_index": dict(set_encoder=True,  pose_head=False),
    "mean_pose": dict(set_encoder=False, pose_head=True),
}


class Cfg:
    learning_rate = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_ratio = 0.2
    n_epochs = 4
    minibatch_size = 64
    value_loss_coef = 0.5
    max_grad_norm = 0.5
    entropy_coef_view = 0.01
    n_steps_per_env = 128
    n_envs = 8
    group_baseline = True
    anneal_lr = True
    phase1_episodes = 0        # set from the step budget below
    phase2_episodes = 0


def build(args):
    from backbones import load_backbones
    from config import Config
    from dataloader_shapenet import build_shapenet
    from env.rgb_view_env import RGBViewEnv, ResNetFeatures
    from policy.pose_policy import RGBPosePolicy
    from training.utility_envelope import UtilityEnvelope

    base = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")

    train_ds = build_shapenet(split="train", categories=args.categories,
                              limit_per_category=args.limit)
    test_ds = build_shapenet(split="test", categories=args.categories,
                             limit_per_category=args.eval_objects)
    bbs = load_backbones(base, device=device)
    envelope = UtilityEnvelope(bbs, cost_lambda=base.cost_lambda,
                               cache_path=args.cache, verbose=True)

    # One feature extractor shared by every environment: ResNet-50 is frozen, so
    # a copy per environment would be pure duplication, and the cache is shared.
    feats = ResNetFeatures(device=device)
    envs = [RGBViewEnv(train_ds, envelope, view_budget=args.budget,
                       features=feats, device=device, seed_initial_view=True,
                       group_sync_seed=args.seed)
            for _ in range(Cfg.n_envs)]
    policy = RGBPosePolicy(n_candidates=train_ds.n_views, **ARMS[args.arm])
    return policy, envs, test_ds, envelope, bbs, feats, device


def evaluate(policy, test_ds, envelope, feats, budget, n_subsets, device, seed=0):
    """Policy views vs random vs best-of-N-subsets, on held-out objects."""
    from env.rgb_view_env import RGBViewEnv, stack_obs, to_tensor

    rng = np.random.default_rng(seed)
    pol_u, rnd_u, orc_u = [], [], []
    policy.eval()
    for i in range(len(test_ds)):
        item = test_ds[i]
        nv = len(item["images"])
        env = RGBViewEnv(test_ds, envelope, view_budget=budget, features=feats,
                         device=device, seed_initial_view=True)
        env.item, env.model_id = item, item.get("model_id", str(i))
        from policy.pose_policy import pose_descriptor
        env.cand_poses = pose_descriptor(item["cams"], env.d_mean, env.d_std)
        env.selected, env.step_count, env.done = [], 0, False
        seeded = int(rng.integers(nv))
        env._acquire(seeded)
        env.step_count = 0

        obs = env._obs()
        while not env.done:
            with torch.no_grad():
                a, _, _ = policy.act(to_tensor(stack_obs([obs]), device), greedy=True)
            obs, r, done, info = env.step(int(a))
        views = list(env.selected)
        pol_u.append(envelope(item, views))

        k = len(views)
        subsets = [rng.choice(nv, size=k, replace=False).tolist()
                   for _ in range(n_subsets)]
        vals = [envelope(item, s) for s in subsets]
        rnd_u.append(float(np.mean(vals)))
        orc_u.append(float(np.max(vals)))

    P, R, O = map(float, (np.mean(pol_u), np.mean(rnd_u), np.mean(orc_u)))
    frac = (P - R) / (O - R) if O - R > 1e-9 else float("nan")
    return {"policy": P, "random": R, "oracle": O,
            "policy_minus_random": P - R, "headroom": O - R,
            "fraction_captured": frac}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--steps", type=int, default=200000, help="environment steps")
    ap.add_argument("--budget", type=int, default=4, help="views the agent CHOOSES")
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="train objects/category")
    ap.add_argument("--eval-objects", type=int, default=12)
    ap.add_argument("--eval-subsets", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cache", default="cache/utility_cache.json")
    ap.add_argument("--out", default="artifacts/tier0b")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--n-envs", type=int, default=Cfg.n_envs)
    ap.add_argument("--n-steps-per-env", type=int, default=Cfg.n_steps_per_env,
                    help="Rollout length. One rollout ends roughly "
                         "n_envs*n_steps/(budget+1) episodes, and each episode "
                         "end runs all three backbones, so this sets how long "
                         "the script goes before its first output.")
    args = ap.parse_args()

    Cfg.n_envs = args.n_envs
    Cfg.n_steps_per_env = args.n_steps_per_env

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    from training.rgb_ppo_trainer import RGBPPOTrainer

    policy, envs, test_ds, envelope, bbs, feats, device = build(args)
    cfg = Cfg()
    cfg.phase1_episodes = 10 ** 9          # driven by the step budget instead
    trainer = RGBPPOTrainer(policy, envs, cfg, device=device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"{args.arm}.jsonl"
    log = open(log_path, "a")
    print(f"\narm '{args.arm}' -> {ARMS[args.arm]}  "
          f"({sum(p.numel() for p in policy.parameters()):,} params)")
    print(f"budget {args.steps:,} env steps, logging to {log_path}\n")

    started = time.time()
    next_eval = 0
    while trainer.total_steps < args.steps:
        rewards, infos = trainer.collect()
        metrics = trainer.update()
        row = {"steps": trainer.total_steps, "episodes": trainer.total_episodes,
               "reward": float(np.mean(rewards)) if rewards else None,
               **metrics, "cache": envelope.stats()}

        if trainer.total_steps >= next_eval:
            next_eval = trainer.total_steps + args.eval_every
            ev = evaluate(policy, test_ds, envelope, feats, args.budget,
                          args.eval_subsets, device, seed=args.seed)
            row["eval"] = ev
            print(f"  {trainer.total_steps:>8,} steps | reward "
                  f"{row['reward']:+.4f} | H {metrics['entropy']:.3f} | "
                  f"policy-random {ev['policy_minus_random']:+.4f} | "
                  f"captured {ev['fraction_captured']*100:5.1f}% of "
                  f"{ev['headroom']:.4f}")
            envelope.flush()
        log.write(json.dumps(row) + "\n")
        log.flush()

        if args.max_hours and (time.time() - started) / 3600 > args.max_hours:
            print("  [time budget] stopping")
            break

    envelope.flush()
    n = envelope.export_router_dataset(out / "router_dataset.json")
    torch.save({"policy": policy.state_dict(), "arm": args.arm,
                "steps": trainer.total_steps,
                "episodes": trainer.total_episodes},
               out / f"{args.arm}.pt")
    print(f"\ndone: {trainer.total_steps:,} steps, {trainer.total_episodes:,} "
          f"episodes, {n:,} router rows cached")


if __name__ == "__main__":
    main()
