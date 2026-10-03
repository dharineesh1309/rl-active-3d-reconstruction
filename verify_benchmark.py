"""
Verify the final benchmark from this repository.

    python verify_benchmark.py                                  # checks 1-2
    python verify_benchmark.py --archive final_artifacts_archive.zip   # + check 3

1. Integrity: every hash the pre-registration (and its amendment) records for
   a git-tracked artifact matches the file in this checkout, and the final
   collection's completion marker matches the collected views.
2. Order: git history shows the pre-registration committed before its
   amendment, the amendment before the final collection, the collection
   before the results, and the post hoc benchmark after the results.
3. Recomputation (needs the archive of the artifacts too large for git --
   final-test features, utility cache, selected policy; see ARTIFACTS.md):
   checks the archive against its SHA256SUMS, re-runs the enforced
   pre-registered evaluation and the post hoc benchmark into a temporary
   directory, and compares every number with the committed results
   (artifacts/final/eval_final.json, artifacts/final/standard_benchmark.json).

Exit code 0 only if every check that ran passed.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FINAL = ROOT / "artifacts" / "final"
POSTHOC = ["policy+router_corrected|random+pix2vox_f",
           "policy+router_corrected|random+umiformer",
           "policy+router_corrected|random+umiformer_plus",
           "policy+router_plain|random+pix2vox_f",
           "policy+router_plain|random+umiformer",
           "policy+router_plain|random+umiformer_plus",
           "policy+router_corrected|heuristic+umiformer_plus"]

failures = []


def check(ok, what):
    print(f"  [{'PASS' if ok else 'FAIL'}] {what}")
    if not ok:
        failures.append(what)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def first_commit(path):
    """The commit that first added `path`."""
    return git("log", "--diff-filter=A", "--format=%h", "--", path).splitlines()[-1]


def close(a, b, tol=1e-9):
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k], tol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=tol, abs_tol=tol) or (math.isnan(a) and math.isnan(b))
    return a == b


def integrity():
    print("\n1. integrity of git-tracked artifacts")
    pre = json.load(open(FINAL / "preregistration.json"))
    A = pre["artifacts"]
    res = json.load(open(FINAL / "eval_final.json"))["registration"]
    mk = json.load(open(FINAL / "collection_complete.json"))
    for path, h in [("build/rl_pipeline/configs/splits_v1.json", pre["split"]["sha256"]),
                    (A["router"]["file"], A["router"]["sha256"]),
                    (A["adaptive_routers"]["file"], A["adaptive_routers"]["sha256"]),
                    (A["cpu_timing_profile"]["file"], A["cpu_timing_profile"]["sha256"]),
                    (A["gpu_timing_profile"]["file"], A["gpu_timing_profile"]["sha256"]),
                    ("artifacts/final/preregistration.json", res["sha256"]),
                    ("artifacts/final/" + res["amendments"][0]["file"],
                     res["amendments"][0]["sha256"]),
                    ("artifacts/final/views_final.json", mk["views_sha256"])]:
        check(sha(ROOT / path) == h, f"{path} matches its recorded sha256")


def order():
    print("\n2. order of events in git history")
    steps = [("pre-registration", "artifacts/final/preregistration.json"),
             ("amendment 1", "artifacts/final/preregistration_amendment_1.json"),
             ("final collection (views + marker)", "artifacts/final/views_final.json"),
             ("pre-registered results", "artifacts/final/eval_final.json"),
             ("post hoc benchmark", "artifacts/final/standard_benchmark.json")]
    commits = []
    for name, path in steps:
        c = first_commit(path)
        commits.append(c)
        print(f"     {name:<36} {c}  {git('log', '-1', '--format=%ad', '--date=iso', c)}")
    for (n1, _), (n2, _), c1, c2 in zip(steps, steps[1:], commits, commits[1:]):
        ok = subprocess.run(["git", "merge-base", "--is-ancestor", c1, c2], cwd=ROOT).returncode == 0
        check(ok and c1 != c2, f"{n1} committed before {n2}")


def recompute(archive):
    print("\n3. recomputation from the archive")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with zipfile.ZipFile(archive) as z:
            z.extractall(tmp)
        sums = {line.split()[1]: line.split()[0]
                for line in (tmp / "SHA256SUMS").read_text().splitlines() if line.strip()}
        for p, h in sums.items():
            check(sha(tmp / p) == h, f"archive {p} matches SHA256SUMS")
        mk = json.load(open(FINAL / "collection_complete.json"))
        check(sums.get("artifacts/final/feats_final.npz") == mk["feats_sha256"],
              "archived final features are the ones the completion marker binds")
        pre = json.load(open(FINAL / "preregistration.json"))
        check(sums.get("artifacts/tier1/set_pose_s164864.pt") == pre["artifacts"]["policy"]["sha256"],
              "archived policy is the pre-registered one")

        evaluator = ROOT / "build" / "rl_pipeline" / "phase3_eval.py"
        common = [sys.executable, str(evaluator), "--split", "final_test",
                  "--views", str(FINAL / "views_final.json"),
                  "--eval-feats", str(tmp / "artifacts/final/feats_final.npz"),
                  "--cache", str(tmp / "cache/utility_cache.json")]
        for name, extra, committed in (
                ("pre-registered evaluation", [], "eval_final.json"),
                ("post hoc benchmark", ["--posthoc-pairs", *POSTHOC], "standard_benchmark.json")):
            out = tmp / committed
            r = subprocess.run(common + ["--out", str(out)] + extra, cwd=ROOT,
                               capture_output=True, text=True)
            if r.returncode != 0:
                check(False, f"{name} re-ran (exit {r.returncode}): {r.stdout[-300:]}{r.stderr[-300:]}")
                continue
            check(close(json.load(open(out)), json.load(open(FINAL / committed))),
                  f"{name}: every recomputed number equals {committed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=None,
                    help="final_artifacts_archive.zip (see ARTIFACTS.md); enables check 3")
    args = ap.parse_args()
    integrity()
    order()
    if args.archive:
        recompute(args.archive)
    else:
        print("\n3. recomputation: skipped (pass --archive final_artifacts_archive.zip)")
    print(f"\n{'ALL CHECKS PASSED' if not failures else f'{len(failures)} CHECK(S) FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
