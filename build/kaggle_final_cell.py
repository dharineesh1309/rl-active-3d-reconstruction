EXPECT_BUILD = "b9d9c66"   # the pre-registered code build -- not a dataset name
EXPECT_POLICY = "84d66f32d8f4878e3b198607773c087c26c7ec054c3a6b59eb66980e2b6937f2"   # pre-registered sha256

# Final test, collection only (pre-registered in artifacts/final/preregistration.json):
# draw the frozen starts and random sets for all 312 final objects, run the
# frozen policy and the farthest-angle heuristic, reconstruct every view set,
# and extract the final objects' features into a separate file.
# No score is computed or printed. Re-running after an interruption reuses
# every reconstruction already in the cache.

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
        raise SystemExit(f"attach the final inputs (no {name}/{member} found)")
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

selected = json.load(open(one("tier1", "selection.json")))["selected"]
policy = one("tier1", f"set_pose_s{selected.split('_s')[-1]}.pt")
if sha(policy) != EXPECT_POLICY:
    raise SystemExit("the attached policy is not the pre-registered one")
print(f"policy {policy} ({selected}, sha256 {sha(policy)[:12]})")

from kaggle_train import preflight, resolve_inputs
preflight(require_cuda=True)
env = resolve_inputs()

out = WORKING / "artifacts/final"
out.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, "-u", str(dst / "final_collect.py"),
                "--policy", str(policy), "--cache", str(cache),
                "--out-views", str(out / "views_final.json"),
                "--out-feats", str(out / "feats_final.npz")],
               cwd=str(dst), env=env, check=True)
