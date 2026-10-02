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
    argmax     the predicted category's table row
    expected   sum_c p(c | views) * table[c], which keeps the uncertainty
    corrected  expected, centred, + beta * a ridge regression of centred
               utilities on the set features: the per-object correction.
               alpha and beta are chosen by K-fold CV over controller_train
               objects, never on dev.

References that are not deployable: the true-category lookup (uses the label)
and the per-view-set oracle.

Every number is object-averaged over dev, and paired comparisons bootstrap over
objects: a view set is not an independent sample, an object is. Splits come from
configs/splits_v1.json.

This replaces a neural utility router that captured 4.7% of the gap on
seen-object labels and ~0% on clean ones (HANDOFF, Tier 1).
"""

import argparse
import hashlib
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


def load_iou(feats_path, cache_path):
    """Rows of every cached view set whose object has features: raw IoU."""
    from training.utility_envelope import load_cache

    d = np.load(feats_path)
    idx = {m: i for i, m in enumerate(d["model_id"])}
    iou = load_cache(cache_path)["iou"]
    names = sorted(next(iter(iou.values())))
    obj, views, I = [], [], []
    for k, v in iou.items():
        m, s = k.split("|")
        if m in idx:
            obj.append(idx[m])
            views.append([int(x) for x in s.split(",")])
            I.append([v[n] for n in names])
    views = np.array(views)
    assert views.ndim == 2, "view sets of different sizes; route them per size"
    return d, names, np.array(obj), views, np.array(I)


def costs_of(names):
    from backbones import _candidates
    c = {k.name: k.cost for k in _candidates()}
    return np.array([c[n] for n in names])


def load(feats_path, cache_path):
    """As load_iou, as utilities IoU - lambda * cost at the configured lambda."""
    from config import Config

    d, names, obj, views, I = load_iou(feats_path, cache_path)
    return d, names, obj, views, I - Config().cost_lambda * costs_of(names)


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


def fit_stage1(X, cats, obj, rows, classes, seed):
    """
    Classifier and temperature from `rows` only: T is fitted on a 1/7 object
    slice held out from a first classifier fit, then the classifier is refit on
    all of `rows`. Called once per CV fold, so calibration never sees the fold.
    """
    objs = np.unique(obj[rows])
    t_obj = np.random.default_rng(seed).choice(objs, len(objs) // 7, replace=False)
    in_t = np.isin(obj[rows], t_obj)
    a, b = rows[~in_t], rows[in_t]
    clf = fit_classifier(X[a], cats[obj[a]], obj[a], classes)
    T = fit_temperature(class_scores(clf, X[b]), cats[obj[b]], classes)
    return fit_classifier(X[rows], cats[obj[rows]], obj[rows], classes), T


def fit_table(U, obj, cat_of_row, classes, fallback):
    """(C, K) object-averaged utility; a category with no rows gets `fallback`."""
    tab = np.tile(fallback, (len(classes), 1))
    for i, c in enumerate(classes):
        r = cat_of_row == c
        if r.any():
            tab[i] = [object_mean(U[r, k], obj[r]) for k in range(U.shape[1])]
    return tab


RIDGE_ALPHAS = (1e2, 1e3, 1e4, 1e5, 1e6)
BETAS = (0.0, 0.25, 0.5, 1.0, 2.0)          # 0 = no correction


def fit_ridge(X, U, obj, alphas):
    """Per-object correction: centred utilities on standardised set features,
    each object weighted once. Returns (mu, sd, {alpha: W})."""
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = (X - mu) / sd
    w = 1.0 / np.bincount(obj)[obj]
    A = Z.T @ (Z * w[:, None])
    B = (Z * w[:, None]).T @ (U - U.mean(1, keepdims=True))
    return mu, sd, {a: np.linalg.solve(A + a * np.eye(len(A)), B) for a in alphas}


def corrected(scores, T, table, X, mu, sd, W, beta):
    """The expected-table utilities, centred, plus beta * the ridge correction."""
    P = softmax(scores, T) @ table
    return (P - P.mean(1, keepdims=True) + beta * (((X - mu) / sd) @ W)).argmax(1)


def cv_folds(obj, tr, n, seed):
    """K folds over the objects of rows `tr`."""
    objs = np.unique(obj[tr])
    return objs, np.array_split(np.random.default_rng(seed + 1).permutation(objs), n)


def fold_classifiers(X, cats, obj, allowed, folds, classes, seed):
    """Stage 1 refit without each fold, on `allowed` objects only (training
    splits -- never dev or final_test). Category labels do not depend on
    lambda, so these are computed once and reused for every lambda."""
    out = []
    for f in folds:
        in_f = np.isin(obj, f)
        r_cls = np.flatnonzero(allowed[obj] & ~in_f)
        out.append((f, in_f, *fit_stage1(X, cats, obj, r_cls, classes, seed)))
    return out


def fit_router(X, U, obj, cats, classes, tr, fold_clfs):
    """
    The utility-dependent half of the router, from rows `tr`: the per-category
    table and the ridge correction, with (alpha, beta) chosen by K-fold CV over
    `tr`'s objects -- table and ridge refit without each fold, alongside that
    fold's stage-1 classifier and temperature. Everything is then refit on all
    of `tr`.
    """
    K = U.shape[1]
    sums = {(a, b): 0.0 for a in RIDGE_ALPHAS for b in BETAS}
    n_objs = 0
    for f, in_f, clf_f, T_f in fold_clfs:
        r_fit, r_f = tr[~in_f[tr]], np.flatnonzero(in_f)
        sv = np.array([object_mean(U[r_fit, k], obj[r_fit]) for k in range(K)])
        tab_f = fit_table(U[r_fit], obj[r_fit], cats[obj[r_fit]], classes, sv)
        fmu, fsd, Wf = fit_ridge(X[r_fit], U[r_fit], obj[r_fit], RIDGE_ALPHAS)
        sc_f, Uf = class_scores(clf_f, X[r_f]), U[r_f]
        for a, b in sums:
            p = corrected(sc_f, T_f, tab_f, X[r_f], fmu, fsd, Wf[a], b)
            sums[a, b] += len(f) * object_mean(Uf.max(1) - Uf[np.arange(len(Uf)), p],
                                               obj[r_f])
        n_objs += len(f)
    cv = {k: v / n_objs for k, v in sums.items()}
    alpha, beta = min(cv, key=cv.get)
    single = np.array([object_mean(U[tr, k], obj[tr]) for k in range(K)])
    table = fit_table(U[tr], obj[tr], cats[obj[tr]], classes, single)
    rmu, rsd, Ws = fit_ridge(X[tr], U[tr], obj[tr], (alpha,))
    return {"single": single, "table": table, "alpha": alpha, "beta": beta,
            "ridge_mu": rmu, "ridge_sd": rsd, "ridge_W": Ws[alpha], "cv_regret": cv}


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
    ap.add_argument("--dev-cohort", default="artifacts/router/dev_cohort_v1.json",
                    help="frozen list of dev view sets to score; written on first run")
    ap.add_argument("--folds", type=int, default=5,
                    help="CV folds over controller_train for the correction")
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
    # Training objects, named explicitly rather than "everything outside dev".
    allowed = is_ctrl | in_("classifier_train")
    X = set_features(d["feats"], obj, views)

    # Stage 1 on every training object.
    clf, T = fit_stage1(X, cats, obj, np.flatnonzero(allowed[obj]), classes, args.seed)

    # The dev view sets scored are frozen on first use: later cache additions
    # (heuristic or policy evaluations) must not change the population.
    keys = np.array([f"{mids[o]}|" + ",".join(map(str, v)) for o, v in zip(obj, views)])
    dev_rows = np.flatnonzero(is_dev[obj])
    cohort = Path(args.dev_cohort)
    if cohort.is_file():
        want = json.load(open(cohort))["keys"]
        dev_rows = dev_rows[np.isin(keys[dev_rows], want)]
        if len(dev_rows) != len(want):
            raise SystemExit(f"{len(want) - len(dev_rows)} frozen dev view sets missing "
                             "from the cache")
    else:
        cohort.parent.mkdir(parents=True, exist_ok=True)
        cohort.write_text(json.dumps({
            "note": "dev view sets the router is scored on: every cached dev entry at "
                    "freezing -- a MIXTURE of random subsets and several policies' "
                    "views, not one deployment distribution",
            "keys": sorted(keys[dev_rows].tolist())}))
        print(f"froze {len(dev_rows):,} dev view sets in {cohort}")
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
        # Choose the correction's (alpha, beta) by K-fold CV over
        # controller_train objects -- a single held-out slice proved a lottery
        # (one draw in five picked an over-regularised alpha) -- with the
        # classifier, table and ridge refit without each fold. Then refit
        # everything on all of controller_train. Dev is only scored.
        hold, folds = cv_folds(obj, tr, args.folds, args.seed)
        fit = fit_router(X, U, obj, cats, classes, tr,
                         fold_classifiers(X, cats, obj, allowed, folds, classes, args.seed))
        alpha, beta, hold_regret = fit["alpha"], fit["beta"], fit["cv_regret"]
        table, rmu, rsd = fit["table"], fit["ridge_mu"], fit["ridge_sd"]
        Ws = {alpha: fit["ridge_W"]}
        print(f"correction chosen by {args.folds}-fold CV over {len(hold)} training "
              f"objects: alpha {alpha:g}, beta {beta:g} (regret "
              f"{hold_regret[(alpha, beta)]:.4f} vs {hold_regret[(alpha, 0.0)]:.4f} uncorrected)")

        picks["single"][:] = fit["single"].argmax()
        picks.update(route(sc, T, table, true_idx))
        picks["corrected"] = corrected(sc, T, table, X[dev_rows], rmu, rsd, Ws[alpha], beta)
        table_src = f"controller_train, {len(set(obj[tr]))} objects"
        fold_ids = None

    res = summarise(U[dev_rows], obj[dev_rows], picks, names)
    deployable = [m for m in ("argmax", "expected", "corrected") if m in picks]
    chosen = max(deployable, key=lambda m: res[m]["utility"])
    print_report(f"dev, {len(set(obj[dev_rows]))} objects; table: {table_src}; "
                 f"T={T:.3g}", res, acc)
    print(f"\nselected variant (dev utility): {chosen}")
    cve = None
    if "corrected" in picks:
        Ud, od = U[dev_rows], obj[dev_rows]
        per_obj = lambda p: np.array([(Ud.max(1) - Ud[np.arange(len(Ud)), p])[od == o].mean()
                                      for o in np.unique(od)])
        dlt = per_obj(picks["expected"]) - per_obj(picks["corrected"])
        bs = np.random.default_rng(args.seed).integers(len(dlt), size=(4000, len(dlt)))
        cve = [float(dlt.mean()), *map(float, np.percentile(dlt[bs].mean(1), [2.5, 97.5]))]
        print(f"corrected vs expected, dev regret reduction {cve[0]:+.4f} "
              f"[95% CI {cve[1]:+.4f}, {cve[2]:+.4f}]")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"cv{args.cv_dev}" if args.cv_dev else "ctrl"
    report = {"table_source": table_src, "variant": chosen, "temperature": T,
              "category_accuracy": acc, "backbones": names, "dev": res,
              "folds": fold_ids, "coverage": None if args.cv_dev else cover,
              "corrected_vs_expected": cve, "alpha": ALPHA, "seed": args.seed,
              "dev_cohort": {"path": str(cohort), "view_sets": int(len(dev_rows)),
                             "kind": "mixed cached dev view sets"},
              "train_view_sets": None if args.cv_dev else {
                  "n": int(len(tr)),
                  "sha256": hashlib.sha256("\n".join(sorted(keys[tr])).encode()).hexdigest()}}
    (out / f"router_report_{tag}.json").write_text(json.dumps(report, indent=1))
    if not args.cv_dev:
        report["correction"] = {"alpha": alpha, "beta": beta, "cv_folds": args.folds,
                                "cv_objects": len(hold),
                                "cv_regret": {f"{a:g},{b:g}": r for (a, b), r
                                              in hold_regret.items()}}
        (out / f"router_report_{tag}.json").write_text(json.dumps(report, indent=1))
        np.savez(out / "router.npz", classes=classes, backbones=np.array(names),
                 mu=clf[0], sd=clf[1], W=clf[2], T=T, table=table,
                 ridge_mu=rmu, ridge_sd=rsd, ridge_W=Ws[alpha], beta=beta,
                 variant=np.array(chosen), set_size=views.shape[1])
    print(f"wrote {out / f'router_report_{tag}.json'}"
          + ("" if args.cv_dev else f" and {out / 'router.npz'}"))


if __name__ == "__main__":
    main()
