EXPECT_BUILD = "11edd00"   # code version (rl_pipeline/BUILD.txt) -- not a dataset name

# Phase 3 GPU job: (1) reconstruct the frozen farthest-angle view sets on dev,
# (2) per-component latency on this GPU.

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

def one(name, member):
    hits = [p / member for p in find(name) if (p / member).is_file()]
    if not hits:
        raise SystemExit(f"attach the phase 3 inputs (no {name}/{member} found)")
    return hits[0]

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

# ── Inputs, by content ───────────────────────────────────────────────────────
caches = []
for d in find("cache"):
    f = d / "utility_cache.json"
    if f.is_file():
        c = json.load(open(f))
        if c.get("format") == 2 and "src" in c:
            caches.append((len(c["iou"]), f))
if not caches:
    raise SystemExit("no current-format cache attached")
n, src_cache = max(caches)
cache = WORKING / "cache/utility_cache.json"
cache.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(src_cache, cache)
views = one("phase3", "views_dev_to_score.json")
router = one("router", "router.npz")
selected = json.load(open(one("tier1", "selection.json")))["selected"]
policy = one("tier1", f"set_pose_s{selected.split('_s')[-1]}.pt")
print(f"cache {src_cache} ({n:,} view sets)\nviews {views}\nrouter {router}\n"
      f"policy {policy} ({selected})")

sys.path.insert(0, str(dst))
from kaggle_train import preflight, resolve_inputs
preflight(require_cuda=True)
env = resolve_inputs()

out = WORKING / "artifacts/phase3"
out.mkdir(parents=True, exist_ok=True)
print("\n--- farthest-angle view sets ---", flush=True)
subprocess.run([sys.executable, "-u", str(dst / "score_view_sets.py"),
                "--views", str(views), "--kind", "eval_heuristic",
                "--ref", "farthest_angle", "--cache", str(cache)],
               cwd=str(dst), env=env, check=True)

print("\n--- GPU latency ---", flush=True)
subprocess.run([sys.executable, "-u", str(dst / "bench_latency.py"),
                "--policy", str(policy), "--router", str(router),
                "--objects", "20", "--out", str(out / "latency_gpu.json")],
               cwd=str(dst), env=env, check=True)
