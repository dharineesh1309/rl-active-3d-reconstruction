EXPECT_BUILD = "3028ea3"   # code version (rl_pipeline/BUILD.txt) -- not a dataset name

# Phase 3 GPU job: (1) reconstruct the frozen farthest-angle view sets on dev,
# (2) latency on this GPU: components and the 15 pipelines on the frozen
# evaluation episodes, (3) export the benchmark cohort's 24-view renderings so
# the CPU benchmark can run locally on the same objects.

import hashlib, json, shutil, subprocess, sys
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

def sha(f):
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()

def one(name, member):
    """The attached `name/member`; several copies are fine only if identical."""
    hits = [p / member for p in find(name) if (p / member).is_file()]
    if not hits:
        raise SystemExit(f"attach the phase 3 inputs (no {name}/{member} found)")
    if len({sha(h) for h in hits}) > 1:
        raise SystemExit(f"conflicting copies of {name}/{member}: {hits}; attach one")
    return hits[0]

def entries(f):
    try:
        c = json.load(open(f))
    except (OSError, ValueError):
        return None
    return len(c["iou"]) if c.get("format") == 2 and "src" in c else None

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

# ── Cache: the union of this session's cache and every attached one ─────────
# merge_caches keeps every entry of every copy and refuses (leaving the files
# untouched) if they disagree on scoring, checkpoints, provenance or a label.
sys.path.insert(0, str(dst))
from training.utility_envelope import merge_caches
cache = WORKING / "cache/utility_cache.json"
sources = [d / "utility_cache.json" for d in find("cache")
           if entries(d / "utility_cache.json") is not None]
if cache.is_file():
    sources.append(cache)
if not sources:
    raise SystemExit("no current-format cache attached")
try:
    merged = merge_caches([json.load(open(f)) for f in sources])
except ValueError as e:
    raise SystemExit(f"caches cannot be merged ({e}); nothing was overwritten")
cache.parent.mkdir(parents=True, exist_ok=True)
tmp = cache.with_suffix(".tmp")
tmp.write_text(json.dumps(merged))
tmp.replace(cache)
print(f"cache: union of {len(sources)} copies -> {len(merged['iou']):,} view sets")

views = one("phase3", "views_dev_to_score.json")
frozen = one("phase3", "views_dev.json")
router = one("router", "router.npz")
selected = json.load(open(one("tier1", "selection.json")))["selected"]
policy = one("tier1", f"set_pose_s{selected.split('_s')[-1]}.pt")
print(f"views {views}\nrouter {router}\npolicy {policy} ({selected})")

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
                "--views", str(frozen), "--out", str(out / "latency_gpu.json")],
               cwd=str(dst), env=env, check=True)

print("\n--- export the benchmark cohort for the CPU run ---", flush=True)
from bench_latency import bench_cohort
from splits import taxonomy
_, where = taxonomy()
R, V = Path(env["SHAPENET_RENDERING_ROOT"]), Path(env["SHAPENET_VOXEL_ROOT"])
bench = WORKING / "bench_objects"
for m in bench_cohort():
    syn = where[m][0][0]
    shutil.copytree(R / syn / m, bench / "ShapeNetRendering" / syn / m, dirs_exist_ok=True)
    (bench / "ShapeNetVox32" / syn / m).mkdir(parents=True, exist_ok=True)
    shutil.copy(V / syn / m / "model.binvox", bench / "ShapeNetVox32" / syn / m / "model.binvox")
print(f"exported {len(bench_cohort())} objects to {bench}")
