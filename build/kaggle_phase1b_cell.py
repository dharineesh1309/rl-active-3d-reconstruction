EXPECT_BUILD = "89fa12c"   # code version (rl_pipeline/BUILD.txt) -- not a dataset name

# Phase 1b: (1) paired dev comparison of the Phase 1 checkpoints against the old
# Tier 0B policy, 4 start views per object; (2) image features for every
# labelled object, so the router trains locally.

import json, shutil, subprocess, sys
from pathlib import Path
import torch

if not torch.cuda.is_available():
    raise SystemExit("No GPU: enable one under Settings > Accelerator.")
INPUT, WORKING = Path("/kaggle/input"), Path("/kaggle/working")

def find(name, root=INPUT, max_depth=7):
    """Bounded walk; the match branch also descends, since some mirrors nest
    a directory inside another of the same name. NEVER rglob here."""
    hits, stack = [], [(root, 0)]
    while stack:
        d, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for p in entries:
            if not p.is_dir():
                continue
            if p.name == name:
                hits.append(p); stack.append((p, depth + 1))
            elif not p.name.isdigit():
                stack.append((p, depth + 1))
    return sorted(hits, key=lambda p: len(p.parts))

def ckpt(p):
    return torch.load(p, map_location="cpu", weights_only=False)

# ── Code ─────────────────────────────────────────────────────────────────────
builds = {p: (p / "BUILD.txt").read_text().strip()
          for p in find("rl_pipeline") if (p / "BUILD.txt").is_file()}
match = [p for p, b in builds.items() if b == EXPECT_BUILD]
if not match:
    raise SystemExit(f"no attached code with build {EXPECT_BUILD}; "
                     f"found {sorted(set(builds.values())) or 'none'}")
dst = WORKING / "rl_pipeline"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(match[0], dst)
print(f"code {match[0]} (build {EXPECT_BUILD})")

# ── Phase 1 run: the attached tier1 output with the most steps, + its cache ──
runs = {p: ckpt(p / "set_pose_last.pt")["steps"]
        for p in find("tier1") if (p / "set_pose_last.pt").is_file()}
if not runs:
    raise SystemExit("attach the Phase 1 notebook's output (artifacts/tier1)")
run = max(runs, key=runs.get)
src_cache = run.parent.parent / "cache/utility_cache.json"
c = json.load(open(src_cache))
if c.get("format") != 2 or "src" not in c:
    raise SystemExit(f"{src_cache} is not a current-format cache")
cache = WORKING / "cache/utility_cache.json"
cache.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(src_cache, cache)
print(f"phase 1 run {run} ({runs[run]:,} steps); cache {len(c['iou']):,} view sets")

# ── Candidates, under unique names ───────────────────────────────────────────
cand = WORKING / "candidates"
cand.mkdir(exist_ok=True)
for p in sorted(run.glob("set_pose_s*.pt")) + [run / "set_pose.pt"]:
    shutil.copy(p, cand / f"clean_s{ckpt(p)['steps']}.pt")
old = [p / "set_pose.pt" for p in find("tier0b") if (p / "set_pose.pt").is_file()]
if not old:
    raise SystemExit("attach the tier1 inputs dataset (holds the Tier 0B set_pose.pt)")
shutil.copy(old[0], cand / "tier0b_set_pose.pt")
print("candidates:", sorted(p.name for p in cand.iterdir()))

sys.path.insert(0, str(dst))
from kaggle_train import preflight, resolve_inputs
preflight(require_cuda=True)
env = resolve_inputs()
if subprocess.run([sys.executable, "-u", str(dst / "training/tests/test_utility_envelope.py")],
                  cwd=str(dst)).returncode:
    raise SystemExit("cache test FAILED")

out = WORKING / "artifacts"
(out / "tier1").mkdir(parents=True, exist_ok=True)
print("\n--- paired dev comparison ---", flush=True)
subprocess.run([sys.executable, "-u", str(dst / "eval_0b.py"),
                *sorted(str(p) for p in cand.glob("*.pt")),
                "--starts", "4", "--cache", str(cache),
                "--out", str(out / "tier1/eval_dev.json")],
               cwd=str(dst), env=env, check=True)

print("\n--- router features ---", flush=True)
subprocess.run([sys.executable, "-u", str(dst / "extract_router_feats.py"),
                "--cache", str(cache), "--out", str(out / "router/feats.npz")],
               cwd=str(dst), env=env, check=True)
