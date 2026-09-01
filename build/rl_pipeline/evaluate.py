"""
evaluate.py — what did the trained policy actually learn?

    python evaluate.py --policy-ckpt checkpoints/ckpt_final.pt --limit 30

Training IoU is not the answer to that question. The backbones were pretrained
on the ShapeNet train split, so rewards seen during training run ~0.95 purely
from memorisation. This evaluates on the **test** split and asks the two things
the project actually claims:

  1. **Does view selection beat picking views at random?** That is the NBV
     claim. Compared at matched budgets and matched backbones, so the only
     difference is which views were chosen.

  2. **Does the model head discriminate, or did it collapse?** A head that
     always emits the same backbone has learned nothing, and the
     backbone-selection contribution would be vacuous. Reported as the
     selection distribution per view budget — if the paper's story holds, the
     cheap backbone should win more often at small budgets and the expensive
     one at large budgets.

Also reports each backbone used alone, so the policy's choices can be compared
against always-picking-one.
"""

import argparse
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

from backbones import load_backbones, voxel_iou
from config import Config
from env.state_builder import CoverageGrid, ImageFeatureExtractor, ViewHistoryMask


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policy-ckpt", default="checkpoints/ckpt_final.pt")
    p.add_argument("--budgets", type=int, nargs="+", default=[3, 5, 8])
    p.add_argument("--limit", type=int, default=30, help="test models per category")
    p.add_argument("--categories", nargs="+", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _fold_view(item, v, cov, feat, hist):
    cams = item.get("cams")
    cov.update(item["depths"][v], item["silhouettes"][v],
               cam=cams[v] if cams else None)
    feat.add_view(item["images"][v])
    hist.mark(v)


def run_policy(policy, backbones, item, budget, n_actions, device, greedy=True,
               seed_initial_view=None, rng=None):
    """One episode: policy picks views, then a backbone. Returns (views, idx)."""
    n_views = len(item["images"])
    cov, feat, hist = CoverageGrid(), ImageFeatureExtractor(device=device), ViewHistoryMask(n_views)
    chosen = []

    # Mirror ViewReconEnv.reset(). If training seeded a free initial view, the
    # policy has never made a decision from an all-zeros state, so evaluating
    # without the seed measures the policy on a distribution it never saw and
    # makes it look worse than it is.
    if seed_initial_view is None:
        seed_initial_view = getattr(Config, "seed_initial_view", False)
    seeded = None
    if seed_initial_view:
        rng = rng or np.random.default_rng(0)
        seeded = int(rng.integers(n_views))
        _fold_view(item, seeded, cov, feat, hist)

    for step in range(budget + 1):
        is_model_step = step == budget
        obs = {
            "coverage_grid": cov.get(),
            "image_features": feat.get(),
            "view_mask": hist.get(),
            "is_model_step": np.array([float(is_model_step)], dtype=np.float32),
            # Must match ViewReconEnv._get_obs exactly, or the policy is being
            # evaluated on a state distribution it never saw in training.
            # Scaled by the policy's own n_views, not this dataset's, since
            # that is what the network was trained against.
            "budget": np.array([budget / policy.n_views,
                                (budget - len(chosen)) / policy.n_views],
                               dtype=np.float32),
        }
        # The policy's view head is as wide as it was trained (24). If this
        # dataset has fewer views, mask the missing ones rather than reshaping
        # the network -- the head still works, it just has fewer legal actions.
        obs_t = {}
        for k, v in obs.items():
            arr = v
            if k == "view_mask" and len(v) < n_actions:
                arr = np.concatenate([v, np.zeros(n_actions - len(v), np.float32)])
            obs_t[k] = torch.FloatTensor(arr).unsqueeze(0).to(device)

        mask = np.zeros(n_actions, dtype=np.float32)
        if is_model_step:
            mask[:len(backbones)] = 1.0
        else:
            mask[:n_views] = 1.0
            for v in hist.visited():
                mask[v] = 0.0
        mask_t = torch.FloatTensor(mask).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = policy(obs_t, mask_t)
            action = (int(logits.argmax(-1)) if greedy
                      else int(torch.distributions.Categorical(logits=logits).sample()))

        if is_model_step:
            # The seeded view was really captured, so it must be part of the
            # reconstruction -- exactly as the environment does it.
            used = ([seeded] if seeded is not None else []) + chosen
            return used, action
        _fold_view(item, action, cov, feat, hist)
        chosen.append(action)

    raise RuntimeError("unreachable")


def even_views(n, total):
    return [int(round(i * total / n)) % total for i in range(n)] if n < total else list(range(total))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    cfg = Config()

    from dataloader_shapenet import build_shapenet
    dataset = build_shapenet(split="test", categories=args.categories,
                             limit_per_category=args.limit)
    backbones = load_backbones(cfg, device=args.device)
    names = [b.name for b in backbones]

    state = torch.load(args.policy_ckpt, map_location="cpu", weights_only=False)
    n_views_trained = state["policy"]["view_head.2.weight"].shape[0]
    n_backbones = state["policy"]["model_head.2.weight"].shape[0]
    assert n_backbones == len(backbones), (
        f"checkpoint has a {n_backbones}-way model head but {len(backbones)} "
        f"backbones are loaded: {names}. The registry must match training.")

    from policy.view_policy import ViewPolicy
    policy = ViewPolicy(n_views=n_views_trained, n_backbones=n_backbones).to(args.device)
    policy.load_state_dict(state["policy"])
    policy.eval()
    print(f"\npolicy: episode {state['total_episodes']}, "
          f"{n_views_trained} views, {n_backbones} backbones {names}\n")

    results = defaultdict(dict)
    for B in args.budgets:
        picks, policy_ious = Counter(), []
        random_ious = defaultdict(list)
        fixed_ious = defaultdict(list)

        for i in range(len(dataset)):
            item = dataset[i]
            gt = item["voxels"]
            n_avail = len(item["images"])

            # --- the trained policy: its views, its backbone ---
            views, b_idx = run_policy(policy, backbones, item, B,
                                      policy.n_actions, args.device)
            picks[names[b_idx]] += 1
            pred = backbones[b_idx].predict([item["images"][v] for v in views],
                                            [item["cams"][v] for v in views])
            policy_ious.append(voxel_iou(pred, gt))

            # --- control: random views, same backbone the policy chose ---
            #
            # size=len(views), NOT size=B. With seed_initial_view the policy
            # returns B+1 views (the free seed plus its B choices), so sizing
            # this control at B gave the policy an extra view and made random
            # selection look worse than it is. That bug reported a spurious
            # +0.0225 "NBV helps" at B=3; scored like-for-like the same policy
            # comes out at -0.0007, i.e. no better than random.
            rv = list(rng.choice(n_avail, size=min(len(views), n_avail), replace=False))
            pred = backbones[b_idx].predict([item["images"][v] for v in rv],
                                            [item["cams"][v] for v in rv])
            random_ious["random_views"].append(voxel_iou(pred, gt))

            # --- control: policy's views, each backbone alone ---
            for bi, b in enumerate(backbones):
                pred = b.predict([item["images"][v] for v in views],
                                 [item["cams"][v] for v in views])
                fixed_ious[b.name].append(voxel_iou(pred, gt))

        total = sum(picks.values())
        results[B] = {
            "policy": float(np.mean(policy_ious)),
            "random_views": float(np.mean(random_ious["random_views"])),
            "fixed": {k: float(np.mean(v)) for k, v in fixed_ious.items()},
            "picks": {k: v / total for k, v in picks.items()},
        }

    # ── Report ───────────────────────────────────────────────────────────────
    print("=" * 72)
    print("  1. VIEW SELECTION  (policy views vs random views, same backbone)")
    print("=" * 72)
    print(f"{'budget':>7} {'policy':>9} {'random':>9} {'delta':>9}")
    for B in args.budgets:
        r = results[B]
        d = r["policy"] - r["random_views"]
        print(f"{B:>7} {r['policy']:>9.4f} {r['random_views']:>9.4f} {d:>+9.4f}"
              + ("   <- NBV helps" if d > 0.005 else "   <- no better than random" if d < 0.005 else ""))

    print("\n" + "=" * 72)
    print("  2. BACKBONE SELECTION  (how often each was chosen, per budget)")
    print("=" * 72)
    hdr = f"{'budget':>7}" + "".join(f"{n:>12}" for n in names)
    print(hdr)
    for B in args.budgets:
        row = f"{B:>7}" + "".join(f"{results[B]['picks'].get(n,0)*100:>11.1f}%" for n in names)
        print(row)
    allpicks = Counter()
    for B in args.budgets:
        for n, f in results[B]["picks"].items():
            allpicks[n] += f
    dominant = max(allpicks.values()) / len(args.budgets)
    print(f"\n  most-chosen backbone takes {dominant*100:.1f}% of decisions overall")
    if dominant > 0.95:
        print("  -> COLLAPSED: the head effectively always picks one backbone.")
        print("     Backbone selection is not contributing; raise cost_lambda or")
        print("     use backbones whose quality/cost trade off more evenly.")
    elif len({max(results[B]['picks'], key=results[B]['picks'].get) for B in args.budgets}) > 1:
        print("  -> DISCRIMINATING: different budgets prefer different backbones,")
        print("     which is the behaviour the project set out to produce.")
    else:
        print("  -> partially discriminating: one backbone leads at every budget,")
        print("     but the split is not degenerate.")

    print("\n" + "=" * 72)
    print("  3. VS ALWAYS-ONE-BACKBONE  (policy's views throughout)")
    print("=" * 72)
    print(f"{'budget':>7} {'policy':>9}" + "".join(f"{n:>12}" for n in names))
    for B in args.budgets:
        r = results[B]
        print(f"{B:>7} {r['policy']:>9.4f}"
              + "".join(f"{r['fixed'].get(n,float('nan')):>12.4f}" for n in names))
    print("\n  Policy should land at or above the best single backbone. Below it means")
    print("  the head is picking worse models than always taking the strongest.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
