EXPECT_BUILD = "9809170"   # code version (rl_pipeline/BUILD.txt) -- not a dataset name

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

def current_cache(f):
    try:
        c = json.load(open(f))
    except (OSError, ValueError):
        return None
    return len(c["iou"]) if c.get("format") == 2 and "src" in c else None

# ── Code: whichever attached rl_pipeline carries the expected build ──────────
builds = {p: (p / "BUILD.txt").read_text().strip()
          for p in find("rl_pipeline") if (p / "BUILD.txt").is_file()}
match = [p for p, b in builds.items() if b == EXPECT_BUILD]
if not match:
    raise SystemExit(f"no attached code with build {EXPECT_BUILD}; "
                     f"found {sorted(set(builds.values())) or 'none'}")
src, dst = match[0], WORKING / "rl_pipeline"
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src, dst)
print(f"code {src} (build {EXPECT_BUILD})")

# ── State: this session's own progress first, never overwritten ──────────────
out, cache = WORKING / "artifacts/tier1", WORKING / "cache/utility_cache.json"
last = out / "set_pose_last.pt"
if last.is_file():
    if not cache.is_file():
        raise SystemExit("working checkpoint found without its cache")
    print("continuing from this session's own output")
else:
    prev = {p: torch.load(p / "set_pose_last.pt", map_location="cpu",
                          weights_only=False)["steps"]
            for p in find("tier1") if (p / "set_pose_last.pt").is_file()}
    if prev:
        top = max(prev.values())
        best = [p for p, s in prev.items() if s == top]
        if len(best) > 1:
            raise SystemExit(f"several previous runs at {top:,} steps: {best}; attach one")
        p = best[0]
        prev_cache = p.parent.parent / "cache/utility_cache.json"
        if current_cache(prev_cache) is None:
            raise SystemExit(f"{p} has no current-format cache beside it")
        shutil.copytree(p, out)
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(prev_cache, cache)
        print(f"resuming the run with the most steps ({top:,}): {p}\n"
              f"  other candidates: { {str(q): s for q, s in prev.items() if q != p} or 'none' }")
    else:
        found = [(n, d / "utility_cache.json") for d in find("cache")
                 if (n := current_cache(d / "utility_cache.json")) is not None]
        if not found:
            raise SystemExit("no current-format cache attached (the tier1 inputs zip)")
        n, f = max(found)
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(f, cache)
        print(f"fresh start; cache {f} ({n:,} view sets)"
              + (f"; ignored {[str(g) for _, g in found if g != f]}" if len(found) > 1 else ""))
resume = last.is_file()
if resume:
    print(f"  checkpoint at {torch.load(last, map_location='cpu', weights_only=False)['steps']:,} steps")

sys.path.insert(0, str(dst))
from kaggle_train import preflight, resolve_inputs
preflight(require_cuda=True)
env = resolve_inputs()

print("\n--- gates ---", flush=True)
for t in ("policy/tests/test_pose_policy_invariance.py",
          "policy/tests/test_synthetic_learning.py",
          "training/tests/test_utility_envelope.py",
          "training/tests/test_router.py",
          "training/tests/test_resume.py"):
    if subprocess.run([sys.executable, "-u", str(dst / t)], cwd=str(dst)).returncode:
        raise SystemExit(f"{t} FAILED -- not starting Phase 1.")

cmd = [sys.executable, "-u", str(dst / "run_0b.py"),
       "--arm", "set_pose", "--steps", "200000", "--budget", "4",
       "--n-envs", "8", "--n-steps-per-env", "128", "--eval-every", "20000",
       "--keep", "3", "--cache", str(cache), "--out", str(out), "--max-hours", "10.5"]
if resume:
    cmd += ["--resume", str(last)]
subprocess.run(cmd, cwd=str(dst), env=env, check=True)
