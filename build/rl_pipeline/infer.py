"""
infer.py — run a trained policy on one object.

    python infer.py --object-dir ../ShapeNetRendering/03001627/<model_id>
    python infer.py --object-dir <dir> --budget 8 --greedy
    python infer.py --object-dir <dir> --save-voxels out.npy

Loads a trained ViewPolicy, lets it pick views and then a reconstruction
backbone, runs that backbone, and reports the IoU against ground truth.

This imports the real modules rather than carrying its own copies. The previous
version inlined ViewPolicy, the state builders and a reconstructor, which meant
every fix to the training code silently failed to reach inference — including
the all-zero depth maps and the addition of the backbone-selection head.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

from backbones import load_backbones, voxel_iou
from config import Config
from env.view_recon_env import ViewReconEnv
from policy.view_policy import ViewPolicy
from utils.checkpoint import CheckpointManager


def parse_args():
    p = argparse.ArgumentParser(description="RL view + backbone selection inference")
    p.add_argument("--object-dir", required=True,
                   help="Object folder containing rendering/ (ShapeNet layout)")
    p.add_argument("--voxel-root", default=None,
                   help="ShapeNetVox32 root, for ground truth. Defaults from $SHAPENET_VOXEL_ROOT.")
    p.add_argument("--budget", type=int, default=5, help="Views the policy may select")
    p.add_argument("--policy-ckpt", default="checkpoints/ckpt_final.pt")
    p.add_argument("--save-voxels", default=None, help="Write predicted occupancy to this .npy")
    p.add_argument("--device", default="cpu", help="Device for the policy")
    p.add_argument("--greedy", action="store_true",
                   help="Take the argmax action instead of sampling")
    return p.parse_args()


def load_object(object_dir: str, voxel_root: str):
    """Build a one-object dataset in the format ViewReconEnv consumes."""
    from dataloader_shapenet import (_read_metadata, _silhouette_from_alpha,
                                     COVERAGE_RES)
    from dataloader import _depth_from_voxels
    from utils import binvox_rw
    from PIL import Image

    path = Path(object_dir)
    render_dir = path / "rendering"
    if not render_dir.is_dir():
        raise FileNotFoundError(f"No rendering/ subfolder in {path}")

    model_id, synset = path.name, path.parent.name
    vox_path = Path(voxel_root) / synset / model_id / "model.binvox"
    if not vox_path.is_file():
        raise FileNotFoundError(
            f"Ground-truth voxels not found: {vox_path}\n"
            "Pass --voxel-root or set SHAPENET_VOXEL_ROOT."
        )
    with open(vox_path, "rb") as f:
        voxels = binvox_rw.read_as_3d_array(f).data.astype(np.float32)

    pngs = sorted(render_dir.glob("*.png"))
    n_views = len(pngs)
    cams = _read_metadata(render_dir / "rendering_metadata.txt", n_views)

    images, depths, silhouettes = [], [], []
    for v in range(n_views):
        img = Image.open(render_dir / f"{v:02d}.png")
        img.load()
        images.append(img)
        silhouettes.append(_silhouette_from_alpha(img))
        depths.append(_depth_from_voxels(voxels, cams[v]["azimuth"],
                                         cams[v]["elevation"], grid_size=COVERAGE_RES))

    return [{
        "images": images, "depths": depths, "silhouettes": silhouettes,
        "cams": cams, "voxels": voxels,
        "category": synset, "model_id": model_id,
    }], n_views


def main():
    args = parse_args()
    cfg = Config()

    voxel_root = (args.voxel_root or os.environ.get("SHAPENET_VOXEL_ROOT")
                  or "../ShapeNetVox32")

    print("=" * 62)
    print("  RL view + backbone selection - inference")
    print("=" * 62)

    dataset, n_views = load_object(args.object_dir, voxel_root)
    obj = dataset[0]
    print(f"\nObject   : {obj['category']}/{obj['model_id']}  ({n_views} views available)")

    backbones = load_backbones(cfg, device=args.device)
    # Must mirror training: a policy trained with a seeded initial view has
    # never chosen from an all-zeros state, so running inference without the
    # seed evaluates it off-distribution.
    env = ViewReconEnv(dataset=dataset, backbones=backbones,
                       view_budget=min(args.budget, n_views),
                       lambda_cost=cfg.cost_lambda, n_views=n_views,
                       shaping_coef=cfg.shaping_coef,
                       seed_initial_view=cfg.seed_initial_view,
                       gamma=cfg.gamma)

    if not os.path.isfile(args.policy_ckpt):
        raise FileNotFoundError(f"Policy checkpoint not found: {args.policy_ckpt}")
    state = torch.load(args.policy_ckpt, map_location="cpu", weights_only=False)
    policy = ViewPolicy(n_views=n_views, n_backbones=len(backbones)).to(args.device)
    report = policy.load_compatible(state["policy"])
    policy.eval()
    print(f"Policy   : episode {state.get('total_episodes', '?')}, "
          f"{len(report['loaded'])} tensors loaded, {len(report['left_fresh'])} fresh")
    if report["left_fresh"]:
        print("           (fresh tensors mean this checkpoint predates part of the "
              "current policy; its choices there are untrained)")

    obs, _ = env.reset()
    env.current_object = obj          # pin the object rather than sampling
    selected, chosen_backbone = [], None

    while True:
        mask = env.get_action_mask()
        obs_t = {k: torch.FloatTensor(v).unsqueeze(0).to(args.device)
                 for k, v in obs.items()}
        mask_t = torch.FloatTensor(mask.astype(np.float32)).unsqueeze(0).to(args.device)

        with torch.no_grad():
            logits, value = policy(obs_t, mask_t)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            action = (int(logits.argmax(-1).item()) if args.greedy
                      else int(torch.distributions.Categorical(logits=logits).sample().item()))

        is_model_step = bool(obs["is_model_step"][0])
        obs, reward, done, _, info = env.step(action)

        if is_model_step:
            chosen_backbone = info["backbone"]
            print(f"\n  backbone step: chose {chosen_backbone}  "
                  f"(p={probs[action]:.3f})")
        else:
            selected.append(action)
            print(f"  view step {len(selected)}: view {action:2d}  "
                  f"p={probs[action]:.3f}  V={float(value):.4f}")

        if done:
            break

    pred = backbones[info["backbone_idx"]].predict(
        [obj["images"][i] for i in selected],
        [obj["cams"][i] for i in selected],
    )
    iou = voxel_iou(pred, obj["voxels"])

    print("\n" + "-" * 62)
    print(f"  Selected views : {selected}")
    print(f"  Backbone       : {chosen_backbone}")
    print(f"  IoU            : {iou:.4f}")
    print(f"  Reward         : {reward:+.4f}  (IoU minus compute penalty)")
    print("-" * 62)

    if args.save_voxels:
        np.save(args.save_voxels, pred)
        print(f"  saved predicted occupancy -> {args.save_voxels}")


if __name__ == "__main__":
    sys.exit(main())
