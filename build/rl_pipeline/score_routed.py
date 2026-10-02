"""
Score view-policy checkpoints by ROUTED utility -- the backbone the trained
router actually picks -- alongside the oracle backbone and the best single one.

    python score_routed.py --eval artifacts/tier1/eval_dev.json \
                           --router artifacts/router/router.npz \
                           --feats artifacts/router/feats.npz \
                           --out artifacts/tier1/selection.json

Reads each checkpoint's chosen view sets from an eval_0b.py output (per object,
per start), routes every one with the router, and looks the utility up in the
cache: nothing is reconstructed. Picks the checkpoint with the highest routed
dev utility; a tie within --tie goes to a policy trained on controller_train
("clean_" prefix), the protocol's training set.
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="artifacts/tier1/eval_dev.json")
    ap.add_argument("--router", default="artifacts/router/router.npz")
    ap.add_argument("--feats", default="artifacts/router/feats.npz")
    ap.add_argument("--cache", default="cache/utility_cache.json")
    ap.add_argument("--out", default="artifacts/tier1/selection.json")
    ap.add_argument("--tie", type=float, default=1e-4)
    args = ap.parse_args()

    from backbones import _candidates
    from config import Config
    from infer_rgb import pick_backbone
    from training.utility_envelope import load_cache, utilities

    ev = json.load(open(args.eval))["arms"]
    cache = load_cache(args.cache)["iou"]
    d = np.load(args.feats)
    idx = {m: i for i, m in enumerate(d["model_id"])}
    F = d["feats"]                    # read once: an npz re-reads on every access
    router = np.load(args.router)
    costs = {c.name: c.cost for c in _candidates()}
    lam = Config().cost_lambda
    names = [str(n) for n in router["backbones"]]
    single = names[int(np.argmax(router["table"].mean(0)))]

    res = {}
    for label, a in ev.items():
        po = a["per_object"]
        rows = {"oracle": [], "routed": [], "single": []}
        for m, starts in zip(po["model_id"], po["views"]):
            o, r, s = [], [], []
            for v in starts:
                u = utilities(cache[f"{m}|" + ",".join(map(str, sorted(v)))], lam, costs)
                pick = pick_backbone(router, F[idx[m], v].astype(np.float32))
                o.append(max(u.values())); r.append(u[pick]); s.append(u[single])
            for k, x in (("oracle", o), ("routed", r), ("single", s)):
                rows[k].append(float(np.mean(x)))
        res[label] = {k: np.array(v) for k, v in rows.items()}

    n = len(next(iter(res.values()))["routed"])
    print(f"{n} objects; single backbone = {single}")
    print(f"  {'checkpoint':<18} {'oracle':>8} {'routed':>8} {'single':>8}")
    for l, r in res.items():
        print(f"  {l:<18} {r['oracle'].mean():>8.4f} {r['routed'].mean():>8.4f} "
              f"{r['single'].mean():>8.4f}")
    paired = {}
    print("\npaired on routed utility (a - b):")
    for a, b in itertools.combinations(res, 2):
        x = res[a]["routed"] - res[b]["routed"]
        se = float(x.std(ddof=1) / np.sqrt(len(x)))
        paired[f"{a} - {b}"] = {"mean": float(x.mean()), "se": se}
        print(f"  {a:<16} - {b:<16} {x.mean():+.4f}  se {se:.4f}  z {x.mean()/se:+.2f}")

    top = max(r["routed"].mean() for r in res.values())
    tied = [l for l, r in res.items() if top - r["routed"].mean() <= args.tie]
    chosen = next((l for l in tied if l.startswith("clean_")), tied[0])
    print(f"\nselected: {chosen} (routed {res[chosen]['routed'].mean():.4f}; "
          f"tied within {args.tie:g}: {tied})")
    json.dump({"selected": chosen, "path": ev[chosen].get("path"),
               "rule": f"highest routed dev utility; ties within {args.tie:g} go to "
                       "a controller_train ('clean_') policy",
               "single_backbone": single, "objects": n,
               "means": {l: {k: float(v.mean()) for k, v in r.items()} for l, r in res.items()},
               "paired_routed": paired}, open(args.out, "w"), indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
