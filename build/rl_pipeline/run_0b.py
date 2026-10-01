"""
View-policy training on the controller split.

    python run_0b.py --arm set_pose --steps 200000
    python run_0b.py --arm set_pose --steps 200000 --resume artifacts/tier1/set_pose_last.pt

Arms, identical except for the two flags that define them:

    set_pose    set encoder + pose-conditioned scorer   <- proposed
    set_index   set encoder + index head                <- same encoder, old
                                                           action abstraction
    mean_pose   mean pooling + pose scorer              <- isolates the encoder

Tier 0B (artifacts/tier0b) ran these on the official TRAIN split. The backbones
were trained there too and score 0.08-0.10 IoU higher on it, unevenly, so the
policy now trains on `controller_train` from configs/splits_v1.json -- objects
no backbone has seen -- and is evaluated on `dev`. `final_test` is never
touched here.

Budgeted in ENVIRONMENT STEPS, not episodes. Reward is the envelope
max_m [IoU_m(S) - lambda*cost_m], so routing is not in this loop. Every
reconstruction lands in the utility cache, which is also the router's training
set.

Dev evaluation, every --eval-every steps:

    policy      the arm's own views
    random      the same number of views drawn at random
    oracle      best of --eval-subsets sampled view sets of that size

The --keep best checkpoints by dev (policy - random) are kept as
<arm>_s<steps>.pt, plus <arm>_last.pt with the full training state.

Resume restores policy, optimizer, counters, every RNG (torch, CUDA, numpy,
python, and each environment's object stream) and the best-checkpoint list,
and refuses unless the resolved experiment is identical: arguments, PPO
settings, lambda, costs, thresholds, checkpoint hashes, scoring version, the
manifest's hash and the hashes of the object lists actually loaded. Only
--steps, --max-hours, --out, --cache and --resume may change. This is
continuation, not exact reproduction: episodes in flight at the cut are
dropped and the environments start fresh ones from their restored streams.

The full dev set and (unless --categories/--limit ask for a subset) the full
controller_train set must load, all three backbones must be present, and
converted legacy labels are checked by recomputation before training starts.
"""

import argparse
import hashlib
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
SPLITS = "splits_v1"
# Dev is built as the first 24 official test ids per category -- the exact
# objects and RNG draws of every Tier 0B eval, so their random/oracle
# reconstructions are already cached. The 3 cross-listed ids in that prefix are
# drawn for but not scored.
DEV_PREFIX = 24


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
    # Constant. Arms get cut by the time budget at different step counts, and
    # an annealed LR would leave them at different points on the schedule.
    anneal_lr = False
    phase1_episodes = 0        # set from the step budget below
    phase2_episodes = 0


def _sha(text) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def code_build() -> str:
    """The commit package_for_kaggle stamped into BUILD.txt, else git, else unknown."""
    here = Path(__file__).resolve().parent
    if (here / "BUILD.txt").is_file():
        return (here / "BUILD.txt").read_text().strip()
    try:
        import subprocess
        return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=here,
                              capture_output=True, text=True).stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def build(args):
    from backbones import load_backbones
    from config import Config
    from dataloader_shapenet import build_shapenet
    from env.rgb_view_env import RGBViewEnv, ResNetFeatures
    from policy.pose_policy import RGBPosePolicy
    from splits import DEFAULT as SPLITS_PATH, load as load_splits, taxonomy
    from training.utility_envelope import SCORING, UtilityEnvelope, checkpoint_ids

    base = Config()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {device}")

    S = load_splits()
    _, where = taxonomy()
    in_cats = lambda m: args.categories is None or any(
        c in args.categories for c in (where[m][0][0], where[m][0][1]))
    train_ds = build_shapenet(split="test", categories=args.categories,
                              only_ids=S["controller_train"],
                              limit_per_category=args.limit)
    train_ids = sorted(m for _, m in train_ds.samples)
    want = {m for m in S["controller_train"] if in_cats(m)}
    if args.limit is None and set(train_ids) != want:
        raise SystemExit(f"controller_train incomplete on disk: {len(train_ids)} of "
                         f"{len(want)}. Pass --limit/--categories for a subset.")
    dev_ds = build_shapenet(split="test", categories=args.categories,
                            limit_per_category=DEV_PREFIX)
    dev_ids = {m for m in S["dev"] if in_cats(m)}
    missing = dev_ids - {m for _, m in dev_ds.samples}
    if missing:
        raise SystemExit(f"{len(missing)} dev objects missing on disk; the dev set "
                         "must be complete")

    bbs = load_backbones(base, device=device, require_all=True)
    ckpts = checkpoint_ids(bbs, base)
    envelope = UtilityEnvelope(
        bbs, cost_lambda=base.cost_lambda, cache_path=args.cache, checkpoints=ckpts,
        run={"script": "run_0b", "arm": args.arm, "seed": args.seed,
             "budget": args.budget, "train": "controller_train", "splits": SPLITS,
             "code": code_build()},
        verbose=True)
    dev_index = {m: i for i, (_, m) in enumerate(dev_ds.samples)}
    checked = envelope.verify_legacy(lambda m: dev_ds[dev_index[m]],
                                     {m: where[m][0][1] for m in dev_ids})
    for rid, diff in checked.items():
        if diff is None:
            raise SystemExit(f"cache run {rid} has unknown checkpoints and no dev "
                             "entries to check them by; start from a fresh cache")
        print(f"  [utility] run {rid}: labels reproduce with these checkpoints "
              f"(max |dIoU| {diff:.1e}), reused as compatible")
    envelope.flush()

    config = {
        "args": {k: v for k, v in vars(args).items()
                 if k not in ("steps", "max_hours", "out", "cache", "resume")},
        "ppo": {k: v for k, v in vars(Cfg).items() if not k.startswith("_")},
        "cost_lambda": envelope.lam, "costs": envelope.costs,
        "thresholds": envelope.thr, "checkpoints": ckpts, "scoring": SCORING,
        "splits_sha256": _sha(Path(SPLITS_PATH).read_text()),
        "train_ids": {"n": len(train_ids), "sha256": _sha("\n".join(train_ids))},
        "dev_ids": {"n": len(dev_ids), "sha256": _sha("\n".join(sorted(dev_ids)))},
    }

    # One feature extractor shared by every environment: ResNet-50 is frozen, so
    # a copy per environment would be pure duplication, and the cache is shared.
    feats = ResNetFeatures(device=device)
    envs = [RGBViewEnv(train_ds, envelope, view_budget=args.budget,
                       features=feats, device=device, seed_initial_view=True,
                       group_sync_seed=args.seed)
            for _ in range(Cfg.n_envs)]
    policy = RGBPosePolicy(n_candidates=train_ds.n_views, **ARMS[args.arm])
    return policy, envs, dev_ds, dev_ids, envelope, feats, device, config


class _One:
    """A one-object dataset, so an episode env loads nothing else."""

    def __init__(self, item):
        self.item, self.n_views = item, len(item["images"])

    def __len__(self):
        return 1

    def __getitem__(self, i):
        return self.item


def greedy_views(policy, utility_fn, feats, budget, device, item, start):
    """One greedy episode on `item`, starting from view `start`."""
    from env.rgb_view_env import RGBViewEnv, stack_obs, to_tensor

    # No random seed view: the start is given, so nothing is acquired or
    # featurised only to be thrown away.
    env = RGBViewEnv(_One(item), utility_fn, view_budget=budget, features=feats,
                     device=device, seed_initial_view=False)
    env._acquire(start)
    env.step_count = 0                       # the free start view, as in training

    obs = env._obs()
    while not env.done:
        with torch.no_grad():
            a, _, _ = policy.act(to_tensor(stack_obs([obs]), device), greedy=True)
        obs, r, done, info = env.step(int(a))
    return list(env.selected)


def evaluate(policy, test_ds, envelope, feats, budget, n_subsets, device, seed=0,
             n_starts=1, keep=None, policy_ref=None):
    """
    Policy views vs random vs best-of-N-subsets, on held-out objects.

    With n_starts > 1 the policy is also rolled out from extra start views per
    object and its utility averaged. Those starts come from a separate stream,
    so the first start and the random subsets are exactly what n_starts=1
    draws -- and so already in the cache. Random and oracle do not depend on
    the start view: they are budget+1 views drawn at random.

    `keep` restricts scoring to those model ids, and every one of them must be
    scored. Objects outside it still get their random draws, so the stream --
    and the cache -- stays aligned. `policy_ref` (a step or checkpoint name) is
    recorded with the view sets the policy generates.

    Per-object values are returned too, so two arms evaluated with the same
    seed can be compared object by object rather than through their means.
    """
    rng = np.random.default_rng(seed)
    extra = np.random.default_rng(seed + 1)
    k = budget + 1                                   # seeded view + budget
    pol_u, rnd_u, orc_u, ids, cats = [], [], [], [], []
    policy.eval()
    for i in range(len(test_ds)):
        mid = test_ds.samples[i][1]
        nv = test_ds.n_views
        first = int(rng.integers(nv))
        starts = [first] + extra.choice(
            [v for v in range(nv) if v != first], n_starts - 1, replace=False).tolist()
        subsets = [rng.choice(nv, size=k, replace=False).tolist()
                   for _ in range(n_subsets)]
        if keep is not None and mid not in keep:
            continue

        item = test_ds[i]
        u = []
        envelope.tag = {"kind": "eval_policy", "ref": policy_ref}
        for s in starts:
            views = greedy_views(policy, envelope, feats, budget, device, item, s)
            assert len(views) == k
            u.append(envelope(item, views))
        pol_u.append(float(np.mean(u)))
        envelope.tag = {"kind": "eval_random", "ref": seed}
        vals = [envelope(item, s) for s in subsets]
        rnd_u.append(float(np.mean(vals)))
        orc_u.append(float(np.max(vals)))
        ids.append(mid)
        cats.append(item.get("category", "?"))

    if keep is not None and set(ids) != set(keep):
        raise RuntimeError(f"evaluate scored {len(ids)} of {len(keep)} required "
                           "objects; the eval dataset is incomplete")
    if len(ids) < 2:
        raise RuntimeError(f"evaluate scored {len(ids)} object(s)")
    P, R, O = map(float, (np.mean(pol_u), np.mean(rnd_u), np.mean(orc_u)))
    frac = (P - R) / (O - R) if O - R > 1e-9 else float("nan")
    d = np.array(pol_u) - np.array(rnd_u)
    return {"policy": P, "random": R, "oracle": O,
            "policy_minus_random": P - R,
            "policy_minus_random_se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "headroom": O - R, "fraction_captured": frac,
            "per_object": {"model_id": ids, "category": cats, "policy": pol_u,
                           "random": rnd_u, "oracle": orc_u}}


def save_state(path, trainer, envs, **extra):
    st = {"policy": trainer.policy.state_dict(), "opt": trainer.opt.state_dict(),
          "steps": trainer.total_steps, "episodes": trainer.total_episodes,
          "rng": {"torch": torch.get_rng_state(),
                  "cuda": (torch.cuda.get_rng_state_all()
                           if torch.cuda.is_available() else None),
                  "numpy": np.random.get_state(), "python": random.getstate()},
          "env_rng": [e._sync.get_state() for e in envs], **extra}
    tmp = Path(str(path) + ".tmp")
    torch.save(st, tmp)
    os.replace(tmp, path)                   # a killed session leaves the old one


def config_diff(a, b, prefix=""):
    """Paths at which two nested config dicts differ."""
    if not (isinstance(a, dict) and isinstance(b, dict)):
        return [] if a == b else [prefix or "<root>"]
    out = []
    for k in sorted(set(a) | set(b), key=str):
        p = f"{prefix}.{k}" if prefix else str(k)
        out += ([p] if (k in a) != (k in b) else config_diff(a[k], b[k], p))
    return out


def restore_state(path, trainer, envs, config, out_dir):
    # CPU: map_location=cuda would move the RNG states too, and
    # torch.set_rng_state only takes a CPU ByteTensor. The policy and the
    # optimizer copy their tensors onto the parameters' device on load.
    st = torch.load(path, map_location="cpu", weights_only=False)
    diff = config_diff(st.get("config", {}), config)
    if diff:
        raise SystemExit("cannot resume, the experiment differs at: " + ", ".join(diff))
    gone = [n for _, _, n in st["best"] if not (Path(out_dir) / n).is_file()]
    if gone:
        raise SystemExit(f"kept checkpoints missing from {out_dir}: {gone}")
    trainer.policy.load_state_dict(st["policy"])
    trainer.opt.load_state_dict(st["opt"])
    trainer.total_steps, trainer.total_episodes = st["steps"], st["episodes"]
    torch.set_rng_state(st["rng"]["torch"])
    if st["rng"]["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["rng"]["cuda"])
    np.random.set_state(st["rng"]["numpy"])
    random.setstate(st["rng"]["python"])
    for e, s in zip(envs, st["env_rng"]):
        e._sync.set_state(s)
    trainer.obs = [e.reset() for e in envs]    # in-flight episodes dropped
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--steps", type=int, default=200000, help="environment steps, total")
    ap.add_argument("--budget", type=int, default=4, help="views the agent CHOOSES")
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None, help="train objects/category")
    ap.add_argument("--eval-subsets", type=int, default=24)
    ap.add_argument("--eval-every", type=int, default=20000)
    ap.add_argument("--keep", type=int, default=3, help="best checkpoints kept")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cache", default="cache/utility_cache.json")
    ap.add_argument("--out", default="artifacts/tier1")
    ap.add_argument("--resume", default=None, help="<arm>_last.pt to continue from")
    ap.add_argument("--max-hours", type=float, default=None, help="this session")
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

    policy, envs, dev_ds, dev_ids, envelope, feats, device, config = build(args)
    cfg = Cfg()
    cfg.phase1_episodes = 10 ** 9          # driven by the step budget instead
    trainer = RGBPPOTrainer(policy, envs, cfg, device=device)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    last = out / f"{args.arm}_last.pt"
    ident = {"arm": args.arm, "budget": args.budget, "config": config,
             "code": code_build()}
    print(f"code {ident['code']}")
    best, next_eval, elapsed0 = [], 0, 0.0      # best: [score, steps, file]
    if args.resume:
        st = restore_state(args.resume, trainer, envs, config, out)
        if st.get("code") != ident["code"]:
            print(f"  NOTE: code changed since the checkpoint "
                  f"({st.get('code')} -> {ident['code']})")
        best, next_eval, elapsed0 = st["best"], st["next_eval"], st["elapsed_h"]
        print(f"resumed from {args.resume} at {trainer.total_steps:,} steps")

    log_path = out / f"{args.arm}.jsonl"
    log = open(log_path, "a")
    print(f"\narm '{args.arm}' -> {ARMS[args.arm]}  "
          f"({sum(p.numel() for p in policy.parameters()):,} params)")
    print(f"budget {args.steps:,} env steps, logging to {log_path}\n")

    started = time.time()
    elapsed = lambda: elapsed0 + (time.time() - started) / 3600
    while trainer.total_steps < args.steps:
        envelope.tag = {"kind": "train", "ref": trainer.total_steps}
        rewards, infos = trainer.collect()
        metrics = trainer.update()
        row = {"steps": trainer.total_steps, "episodes": trainer.total_episodes,
               "elapsed_h": elapsed(),
               "reward": float(np.mean(rewards)) if rewards else None,
               **metrics, "cache": envelope.stats()}

        if trainer.total_steps >= next_eval:
            next_eval = trainer.total_steps + args.eval_every
            ev = evaluate(policy, dev_ds, envelope, feats, args.budget,
                          args.eval_subsets, device, seed=args.seed, keep=dev_ids,
                          policy_ref=trainer.total_steps)
            row["eval"] = ev
            print(f"  {trainer.total_steps:>8,} steps | reward "
                  f"{row['reward']:+.4f} | H {metrics['entropy']:.3f} | "
                  f"dev policy-random {ev['policy_minus_random']:+.4f} "
                  f"(se {ev['policy_minus_random_se']:.4f}) | "
                  f"captured {ev['fraction_captured']*100:5.1f}% of "
                  f"{ev['headroom']:.4f}", flush=True)

            score = ev["policy_minus_random"]
            if len(best) < args.keep or score > min(b[0] for b in best):
                name = f"{args.arm}_s{trainer.total_steps}.pt"
                torch.save({"policy": policy.state_dict(), **ident,
                            "steps": trainer.total_steps,
                            "dev": {k: v for k, v in ev.items() if k != "per_object"}},
                           out / name)
                best = sorted(best + [[score, trainer.total_steps, name]], reverse=True)
                for _, _, dropped in best[args.keep:]:
                    (out / dropped).unlink(missing_ok=True)
                best = best[:args.keep]
            envelope.flush()
            save_state(last, trainer, envs, **ident, best=best,
                       next_eval=next_eval, elapsed_h=elapsed())
        log.write(json.dumps(row) + "\n")
        log.flush()

        if args.max_hours and (time.time() - started) / 3600 > args.max_hours:
            print("  [time budget] stopping")
            break

    envelope.flush()
    save_state(last, trainer, envs, **ident, best=best, next_eval=next_eval,
               elapsed_h=elapsed())
    torch.save({"policy": policy.state_dict(), **ident,
                "steps": trainer.total_steps, "episodes": trainer.total_episodes},
               out / f"{args.arm}.pt")
    print(f"\n{trainer.total_steps:,} steps, {trainer.total_episodes:,} episodes, "
          f"{elapsed():.2f} h total. Best on dev: "
          + ", ".join(f"{n} ({s:+.4f})" for s, _, n in best))


if __name__ == "__main__":
    main()
