"""
Controller splits, frozen in configs/splits_v1.json.

    python splits.py --write configs/splits_v1.json     # once; then never again

The official split says which objects the RECONSTRUCTORS trained on, and that is
all it is used for here. All three backbones trained on the official train
split and score 0.08-0.10 IoU higher on it, unevenly across backbones, so
their behaviour there is not their behaviour on new objects. The controller
(view policy + router) therefore gets its own split, carved out of the
official test split:

    dev               the 309 objects every Tier 0B / router result so far was
                      measured on (first 24 test ids per category, minus
                      cross-listed). Development and model selection only.
    final_test        24 per category drawn at random from the rest, excluding
                      the first 60 test ids per category (every calibration and
                      benchmark script took its objects from that prefix) and
                      every object in the local benchmark subset. Touched once,
                      at the end.
    controller_train  every other eligible test-split object: policy and
                      router training.
    classifier_train  official train split minus cross-listed: the image ->
                      category classifier only, since memorisation cannot bias
                      a category label.

Cross-listed ids (listed under two categories, some in train under one and
test under another) are excluded everywhere.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT = HERE / "configs" / "splits_v1.json"
DEV_PREFIX, CALIB_PREFIX, FINAL_PER_CAT, SEED = 24, 60, 24, 20261001


def taxonomy():
    """(entries, model_id -> [(synset, category, official split)])."""
    tax = json.load(open(HERE / "datasets" / "ShapeNet.json"))
    where = defaultdict(list)
    for e in tax:
        for s in ("train", "val", "test"):
            for m in e.get(s, []):
                where[m].append((e["taxonomy_id"], e["taxonomy_name"], s))
    return tax, where


def build(local_root=None):
    tax, where = taxonomy()
    cross = sorted(m for m, w in where.items() if len(w) > 1)
    bad = set(cross)
    local = set()
    if local_root and Path(local_root).is_dir():
        local = {p.name for s in Path(local_root).iterdir() if s.is_dir()
                 for p in s.iterdir() if p.is_dir()}

    rng = np.random.default_rng(SEED)
    dev, final, train = [], [], []
    for e in tax:
        test = [m for m in e.get("test", []) if m not in bad]
        head = set(e.get("test", [])[:CALIB_PREFIX])
        d = [m for m in e.get("test", [])[:DEV_PREFIX] if m not in bad]
        pool = [m for m in test if m not in head and m not in local]
        f = sorted(rng.choice(pool, FINAL_PER_CAT, replace=False).tolist())
        dev += d
        final += f
        train += [m for m in test if m not in set(d) | set(f)]

    n_cls = sum(1 for e in tax for m in e.get("train", []) if m not in bad)
    return {
        "version": "splits_v1",
        "seed": SEED,
        "rules": {"dev": f"first {DEV_PREFIX} official test ids per category",
                  "final_test": f"{FINAL_PER_CAT}/category at random from official "
                                f"test, excluding the first {CALIB_PREFIX} per "
                                f"category and {len(local)} local benchmark objects",
                  "controller_train": "all other official test ids",
                  "classifier_train": "official train split",
                  "all": "cross_listed ids excluded everywhere"},
        "counts": {"dev": len(dev), "final_test": len(final),
                   "controller_train": len(train), "classifier_train": n_cls,
                   "cross_listed": len(cross)},
        "dev": dev, "final_test": final, "controller_train": train,
        "cross_listed": cross,
    }


def load(path=None) -> dict:
    """{name: set(model ids)} for dev, final_test, controller_train, plus
    'classifier_train' as the official train split minus cross-listed."""
    s = json.load(open(path or DEFAULT))
    _, where = taxonomy()
    bad = set(s["cross_listed"])
    out = {k: set(s[k]) for k in ("dev", "final_test", "controller_train")}
    out["classifier_train"] = {m for m, w in where.items()
                               if m not in bad and w[0][2] == "train"}
    return out


def _check(s):
    sets = {k: set(s[k]) for k in ("dev", "final_test", "controller_train")}
    for a in sets:
        for b in sets:
            assert a == b or not (sets[a] & sets[b]), f"{a} overlaps {b}"
        assert not (sets[a] & set(s["cross_listed"])), f"{a} holds cross-listed"
    _, where = taxonomy()
    for k, ids in sets.items():
        assert all(where[m][0][2] == "test" for m in ids), f"{k} not all official test"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", default=None, help="output path; refuses to overwrite")
    ap.add_argument("--local-root", default=str(HERE.parent / "ShapeNetRendering"))
    args = ap.parse_args()

    s = build(args.local_root)
    _check(s)
    print(json.dumps(s["counts"]))
    if args.write:
        out = Path(args.write)
        if out.exists():
            raise SystemExit(f"{out} exists; splits are frozen. Make a new version.")
        out.write_text(json.dumps(s, indent=1))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
