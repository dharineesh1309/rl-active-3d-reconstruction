"""
Tier 1: two-stage backbone router (CPU).

    python train_router.py --cv-dev 5     # before clean labels: table by CV on dev
    python train_router.py                # table from controller_train, scored on dev

Stage 1, images -> category. Ridge on the mean ResNet-50 feature of the
acquired views, trained on every labelled object outside dev and final_test.
Category labels are not biased by the backbones' memorisation, so official
train objects are fine here. A temperature-scaled softmax over its scores,
fitted on held-out training objects, gives category probabilities.

Stage 2, category -> backbone. A table of object-averaged utility per
(category, backbone), built ONLY from objects no backbone saw: controller_train,
or under --cv-dev the other dev folds.

Variants, chosen by utility on dev -- not by category accuracy:
    argmax    the predicted category's table row
    expected  sum_c p(c | views) * table[c], which keeps the uncertainty

References that are not deployable: the true-category lookup (uses the label)
and the per-view-set oracle.

Every number is object-averaged over dev, and paired comparisons bootstrap over
objects: a view set is not an independent sample, an object is. Splits come from
configs/splits_v1.json.

This replaces a neural utility router that captured 4.7% of the gap on
seen-object labels and ~0% on clean ones (HANDOFF, Tier 1).
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ALPHA = 1e3                      # ridge strength; fixed, never tuned on dev


def object_mean(x, obj):
    """Mean within each object, then across objects."""
    cnt = np.bincount(obj)
    has = cnt > 0
    return float((np.bincount(obj, weights=x)[has] / cnt[has]).mean())


def load(feats_path, cache_path):
    """Rows of every cached view set whose object has features."""
    from backbones import _candidates
    from config import Config
    from training.utility_envelope import load_cache, utilities

    d = np.load(feats_path)
    idx = {m: i for i, m in enumerate(d["model_id"])}
    costs = {c.name: c.cost for c in _candidates()}
    lam = Config().cost_lambda
    iou = load_cache(cache_path)["iou"]
    names = sorted(next(iter(iou.values())))
    obj, views, U = [], [], []
    for k, v in iou.items():
        m, s = k.split("|")
        if m in idx:
            u = utilities(v, lam, costs)
            obj.append(idx[m])
            views.append([int(x) for x in s.split(",")])
            U.append([u[n] for n in names])
    views = np.array(views)
    assert views.ndim == 2, "view sets of different sizes; route them per size"
    return d, names, np.array(obj), views, np.array(U)


def set_features(F, obj, views):
    return F[obj[:, None], views].astype(np.float32).mean(1)


def fit_classifier(X, y, obj, classes):
    """Object-weighted ridge, one-hot targets: returns (mu, sd, W)."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    Y = (y[:, None] == classes[None]).astype(np.float32)
    w = 1.0 / np.bincount(obj)[obj]
    W = np.linalg.solve(Z.T @ (Z * w[:, None]) + ALPHA * np.eye(Z.shape[1]),
                        (Z * w[:, None]).T @ Y)
    return mu, sd, W


def class_scores(clf, X):
    mu, sd, W = clf
    return ((X - mu) / sd) @ W


def softmax(s, T):
    z = s / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(scores, y, classes):
    t = np.searchsorted(classes, y)
    grid = np.logspace(-3, 1, 81)
    nll = [-np.log(softmax(scores, T)[np.arange(len(t)), t] + 1e-12).mean() for T in grid]
    return float(grid[int(np.argmin(nll))])


def fit_table(U, obj, cat_of_row, classes, fallback):
    """(C, K) object-averaged utility; a category with no rows gets `fallback`."""
    tab = np.tile(fallback, (len(classes), 1))
    for i, c in enumerate(classes):
        r = cat_of_row == c
        if r.any():
            tab[i] = [object_mean(U[r, k], obj[r]) for k in range(U.shape[1])]
    return tab


def route(scores, T, table, classes_idx=None):
    """{variant: picks}; with classes_idx also the true-label reference."""
    out = {"argmax": table[scores.argmax(1)].argmax(1),
           "expected": (softmax(scores, T) @ table).argmax(1)}
    if classes_idx is not None:
        out["category (true label)"] = table[classes_idx].argmax(1)
    return out


def summarise(U, obj, picks, names, seed=0):
    """Object-averaged utility/regret per method, and a bootstrap CI over
    objects for each method against the best single backbone."""
    oracle = U.max(1)
    per_obj = lambda x: np.array([x[obj == o].mean() for o in np.unique(obj)])
    o_oracle = per_obj(oracle)
    res = {}
    rng = np.random.default_rng(seed)
    boot = rng.integers(len(o_oracle), size=(2000, len(o_oracle)))
    base = per_obj(U[np.arange(len(U)), picks["single"]])
    for m, p in picks.items():
        u = per_obj(U[np.arange(len(U)), p])
        d = u - base
        lo, hi = np.percentile(d[boot].mean(1), [2.5, 97.5])
        res[m] = {"utility": float(u.mean()), "regret": float((o_oracle - u).mean()),
                  "vs_single": float(d.mean()), "vs_single_ci95": [float(lo), float(hi)],
                  "share": {n: float(np.mean(p == k)) for k, n in enumerate(names)}}
    gap = res["single"]["regret"]
    for v in res.values():
        v["captured"] = (gap - v["regret"]) / gap if gap > 0 else float("nan")
    res["oracle"] = {"utility": float(o_oracle.mean()), "regret": 0.0, "captured": 1.0}
    return res


def print_report(title, res, acc):
    print(f"\n{title}   (category accuracy from views {acc*100:.1f}%)")
    print(f"  {'':<22} {'utility':>8} {'regret':>8} {'captured':>9}   vs single [95% CI]")
    for m, v in res.items():
        ci = (f"{v['vs_single']:+.4f} [{v['vs_single_ci95'][0]:+.4f}, "
              f"{v['vs_single_ci95'][1]:+.4f}]") if "vs_single" in v else ""
        print(f"  {m:<22} {v['utility']:>8.4f} {v['regret']:>8.4f} "
              f"{v['captured']*100:>8.1f}%   {ci}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="artifacts/router/feats.npz")
    ap.add_argument("--cache", default="cache/utility_cache.json")
    ap.add_argument("--out", default="artifacts/router")
    ap.add_argument("--cv-dev", type=int, default=0,
                    help="K: build the table by K-fold CV over dev objects")
    ap.add_argument("--min-objects", type=int, default=60,
                    help="flag categories with fewer clean table objects")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from splits import load as load_splits

    S = load_splits()
    d, names, obj, views, U = load(args.feats, args.cache)
    mids, cats = d["model_id"], d["category"]
    classes = np.unique(cats)
    in_ = lambda key: np.array([m in S[key] for m in mids])
    is_dev, is_final, is_ctrl = in_("dev"), in_("final_test"), in_("controller_train")
    assert not is_final.any(), "final_test objects in the router data"
    X = set_features(d["feats"], obj, views)

    # Stage 1 on everything labelled outside dev, minus a slice for T.
    clf_obj = np.flatnonzero(~is_dev)
    rng = np.random.default_rng(args.seed)
    t_obj = set(rng.choice(clf_obj, len(clf_obj) // 7, replace=False).tolist())
    r_fit = np.flatnonzero(~is_dev[obj] & ~np.isin(obj, list(t_obj)))
    r_T = np.flatnonzero(np.isin(obj, list(t_obj)))
    clf = fit_classifier(X[r_fit], cats[obj[r_fit]], obj[r_fit], classes)
    T = fit_temperature(class_scores(clf, X[r_T]), cats[obj[r_T]], classes)
    clf = fit_classifier(X[~is_dev[obj]], cats[obj[~is_dev[obj]]], obj[~is_dev[obj]], classes)

    dev_rows = np.flatnonzero(is_dev[obj])
    sc = class_scores(clf, X[dev_rows])
    true_idx = np.searchsorted(classes, cats[obj[dev_rows]])
    acc = object_mean((sc.argmax(1) == true_idx).astype(float), obj[dev_rows])

    picks = {k: np.zeros(len(dev_rows), int) for k in
             ("single", "argmax", "expected", "category (true label)")}
    if args.cv_dev:
        dev_objs = np.sort(np.unique(obj[dev_rows]))
        folds = np.array_split(np.random.default_rng(args.seed).permutation(dev_objs),
                               args.cv_dev)
        for f in folds:
            te = np.isin(obj[dev_rows], f)
            tr = dev_rows[~te]
            single = np.array([object_mean(U[tr, k], obj[tr]) for k in range(len(names))])
            table = fit_table(U[tr], obj[tr], cats[obj[tr]], classes, single)
            picks["single"][te] = single.argmax()
            for k, p in route(sc[te], T, table, true_idx[te]).items():
                picks[k][te] = p
        table_src = f"{args.cv_dev}-fold CV over dev objects"
        fold_ids = [[str(mids[o]) for o in f] for f in folds]
    else:
        tr = np.flatnonzero(is_ctrl[obj])
        assert len(tr), "no controller_train rows in the cache yet; use --cv-dev"
        cover = {c: len(set(obj[tr][cats[obj[tr]] == c])) for c in classes}
        print("table objects per category: " + ", ".join(
            f"{c} {n}" + (" (THIN: top up)" if n < args.min_objects else "")
            for c, n in cover.items()))
        single = np.array([object_mean(U[tr, k], obj[tr]) for k in range(len(names))])
        table = fit_table(U[tr], obj[tr], cats[obj[tr]], classes, single)
        picks["single"][:] = single.argmax()
        picks.update(route(sc, T, table, true_idx))
        table_src = f"controller_train, {len(set(obj[tr]))} objects"
        fold_ids = None

    res = summarise(U[dev_rows], obj[dev_rows], picks, names)
    deployable = ("argmax", "expected")
    chosen = max(deployable, key=lambda m: res[m]["utility"])
    print_report(f"dev, {len(set(obj[dev_rows]))} objects; table: {table_src}; "
                 f"T={T:.3g}", res, acc)
    print(f"\nselected variant (dev utility): {chosen}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"cv{args.cv_dev}" if args.cv_dev else "ctrl"
    report = {"table_source": table_src, "variant": chosen, "temperature": T,
              "category_accuracy": acc, "backbones": names, "dev": res,
              "folds": fold_ids, "coverage": None if args.cv_dev else cover,
              "alpha": ALPHA, "seed": args.seed}
    (out / f"router_report_{tag}.json").write_text(json.dumps(report, indent=1))
    if not args.cv_dev:
        np.savez(out / "router.npz", classes=classes, backbones=np.array(names),
                 mu=clf[0], sd=clf[1], W=clf[2], T=T, table=table,
                 variant=np.array(chosen), set_size=views.shape[1])
    print(f"wrote {out / f'router_report_{tag}.json'}"
          + ("" if args.cv_dev else f" and {out / 'router.npz'}"))


if __name__ == "__main__":
    main()
