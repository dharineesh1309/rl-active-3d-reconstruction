"""
Phase 3: the dev evaluation matrix.

    python phase3_eval.py --out artifacts/phase3/eval_dev.json

View strategy (frozen by phase3_views.py) x backbone choice, per dev object;
episodes averaged within an object, objects the unit of everything:

    views   random (24 sets) | heuristic (4) | policy (4)
    choice  pix2vox_f | umiformer | umiformer_plus | router_plain |
            router_corrected | category_true (reference: uses the label) |
            oracle (reference: the best backbone for each set)
plus "best of 24 sampled five-view sets": random views, oracle backbone, the
best of the object's 24 sets.

Three accountings, never mixed:
    IoU
    backbone-only utility -- frozen cost v1: IoU - lambda * declared cost
    CPU-measured utility, ESTIMATED -- IoU - lambda * (the pipeline's measured
        controller seconds + the measured seconds of each backbone it chose),
        from latency_cpu.json. Controller seconds: per object, then across
        objects. Backbone seconds: from the balanced fixed-backbone runs (12
        per object per backbone), per object, then across objects. The cohort's
        own end-to-end totals are compared with the same estimate as a check.
GPU latency is reported in seconds only: lambda is priced per CPU-second.

Uncertainty: one stratified bootstrap over dev objects -- the 26 timing-cohort
objects and the other 283 resampled separately -- whose weights drive quality,
routing choices AND timings together (the cohort objects are dev objects), and
are shared by every pipeline, so every contrast is paired.

Inputs are validated first: exactly the 309 frozen dev objects and the views'
recorded hash; a CPU and a GPU latency file on the same cohort, the expected
one, with all 15 pipelines complete and zero frozen-view mismatches.

Lambda sensitivity, from raw IoU, view policy fixed, every curve also against
the common baseline random + umiformer_plus and in absolute utility:
    frozen    today's decisions, rescored at each lambda
    adaptive  the router's table, ridge correction and their CV rebuilt from
              training labels (training splits only) at each lambda
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

STRATS = ("random", "heuristic", "policy")
FIXED = ("pix2vox_f", "umiformer", "umiformer_plus")
DEPLOY = (*FIXED, "router_plain", "router_corrected")
REFS = ("category_true", "oracle")
BASE = ("random", "umiformer_plus")
LAMBDAS = (0.0, 0.02, 0.04, 0.0771, 0.1, 0.15, 0.2, 0.3)


def per_object(values, obj_of_row, n_obj):
    """Mean over each object's rows."""
    return (np.bincount(obj_of_row, weights=values, minlength=n_obj)
            / np.bincount(obj_of_row, minlength=n_obj))


def wmean(W, x):
    """Weighted means of per-object x under each row of weights W."""
    return (W @ x) / W.sum(1)


def interval(reps, point):
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return [float(point), float(lo), float(hi)]


def validate(frozen_doc, dev, lat_cpu, lat_gpu, cohort):
    views = frozen_doc["views"]
    if set(views) != set(dev) or len(views) != 309:
        raise SystemExit("frozen views do not cover exactly the 309 dev objects")
    h = hashlib.sha256(json.dumps(views, sort_keys=True).encode()).hexdigest()
    if h != frozen_doc["sha256"]:
        raise SystemExit("frozen views do not match their recorded hash")
    for L, dev_kind in ((lat_cpu, "cpu"), (lat_gpu, "cuda")):
        if L["device"] != dev_kind:
            raise SystemExit(f"latency file for {dev_kind} reports device {L['device']}")
        if L["cohort"] != cohort:
            raise SystemExit(f"{dev_kind} latency cohort is not bench_cohort()")
        if any(L["frozen_mismatches"].values()):
            raise SystemExit(f"{dev_kind} latency run did not reproduce the frozen views")
        want = {f"{s}+{c}" for s in STRATS for c in DEPLOY}
        n = len(cohort) * L["episodes_per_object"]
        if set(L["pipelines"]) != want or any(
                v["total_s"]["n"] != n for v in L["pipelines"].values()):
            raise SystemExit(f"{dev_kind} latency run is missing pipelines or episodes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--views", default="artifacts/phase3/views_dev.json")
    ap.add_argument("--cache", default="cache/utility_cache.json")
    ap.add_argument("--feats", default="artifacts/router/feats.npz")
    ap.add_argument("--router", default="artifacts/router/router.npz")
    ap.add_argument("--latency-cpu", default="artifacts/phase3/latency_cpu.json")
    ap.add_argument("--latency-gpu", default="artifacts/phase3/latency_gpu.json")
    ap.add_argument("--out", default="artifacts/phase3/eval_dev.json")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--no-adaptive", action="store_true")
    args = ap.parse_args()

    from bench_latency import bench_cohort
    from config import Config
    from splits import load as load_splits
    from train_router import (class_scores, corrected, costs_of, cv_folds,
                              fold_classifiers, fit_router, load_iou, set_features,
                              softmax)
    from training.utility_envelope import UtilityEnvelope, load_cache

    S = load_splits()
    lam = Config().cost_lambda
    frozen_doc = json.load(open(args.views))
    Lc, Lg = json.load(open(args.latency_cpu)), json.load(open(args.latency_gpu))
    cohort = bench_cohort()
    validate(frozen_doc, S["dev"], Lc, Lg, cohort)
    frozen = frozen_doc["views"]
    cache = load_cache(args.cache)["iou"]
    names = sorted(next(iter(cache.values())))
    assert tuple(names) == FIXED
    cost = costs_of(names)
    R = np.load(args.router)
    d = np.load(args.feats)
    F, cats_all = d["feats"], d["category"]
    idx = {m: i for i, m in enumerate(d["model_id"])}
    classes = R["classes"]

    # ── every evaluated view set, flattened ──
    objs = sorted(frozen)
    o_of, s_of, rows_v, I = [], [], [], []
    for oi, m in enumerate(objs):
        for s in STRATS:
            for v in frozen[m][s]:
                o_of.append(oi); s_of.append(s); rows_v.append(v)
                I.append([cache[UtilityEnvelope.key(m, v)][n] for n in names])
    o_of, s_of, I = np.array(o_of), np.array(s_of), np.array(I)
    n_obj = len(objs)
    X = np.stack([F[idx[objs[o]], v].astype(np.float32).mean(0) for o, v in zip(o_of, rows_v)])
    sc = class_scores((R["mu"], R["sd"], R["W"]), X)
    T = float(R["T"])
    cat_idx = np.searchsorted(classes, cats_all[[idx[objs[o]] for o in o_of]])

    def decisions(table, rmu, rsd, rW, beta):
        dec = {"router_plain": (softmax(sc, T) @ table).argmax(1),
               "router_corrected": corrected(sc, T, table, X, rmu, rsd, rW, beta),
               "category_true": table[cat_idx].argmax(1)}
        for k, n in enumerate(names):
            dec[n] = np.full(len(I), k)
        return dec

    dec = decisions(R["table"], R["ridge_mu"], R["ridge_sd"], R["ridge_W"], float(R["beta"]))

    def per_pipeline(lmb, dec):
        """{(s, c): per-object IoU, v1 utility and backbone shares}."""
        U = I - lmb * cost
        picks = {**dec, "oracle": U.argmax(1)}
        out = {}
        for s in STRATS:
            r = np.flatnonzero(s_of == s)
            for c in (*DEPLOY, *REFS):
                ch = picks[c][r]
                out[s, c] = {"iou": per_object(I[r, ch], o_of[r], n_obj),
                             "util": per_object(U[r, ch], o_of[r], n_obj),
                             "share": np.stack([per_object((ch == k).astype(float), o_of[r], n_obj)
                                                for k in range(len(names))], 1)}
        r = np.flatnonzero(s_of == "random")
        best = np.full(n_obj, -np.inf)
        np.maximum.at(best, o_of[r], U[r].max(1))
        out["best_of_24"] = {"util": best}
        return out

    P = per_pipeline(lam, dec)

    # ── stratified bootstrap weights over dev objects, shared by everything ──
    rng = np.random.default_rng(0)
    pos = {m: i for i, m in enumerate(objs)}
    c_idx = np.array([pos[m] for m in cohort])
    rest = np.setdiff1d(np.arange(n_obj), c_idx)
    B = args.boot
    W = np.zeros((B, n_obj))
    for b in range(B):
        np.add.at(W[b], rng.choice(c_idx, len(c_idx)), 1)
        np.add.at(W[b], rng.choice(rest, len(rest)), 1)
    Wc = W[:, c_idx]                                    # the same draws, cohort part
    ones = np.ones((1, n_obj))

    # ── timings, per cohort object ──
    def timings(L):
        raw = L["raw"]["paths"]
        cpos = {m: i for i, m in enumerate(cohort)}
        def per_obj(ps, field):
            return per_object(np.array([p[field] for p in ps]),
                              np.array([cpos[p["object"]] for p in ps]), len(cohort))
        ctrl = {(s, c): per_obj([p for p in raw if p["strategy"] == s and p["choice"] == c],
                                "controller_s") for s in STRATS for c in DEPLOY}
        bb = np.stack([per_obj([p for p in raw if p["choice"] == n], "backbone_s")
                       for n in names], 1)              # balanced fixed-backbone runs
        return ctrl, bb

    ctrl_c, bb_c = timings(Lc)
    ctrl_g, bb_g = timings(Lg)

    def seconds(ctrl, bb, s, c, Wt):
        """(reps, n_obj): estimated seconds per object for pipeline (s, c)."""
        cm = wmean(Wt, ctrl[s, c])                      # (reps,)
        bm = (Wt @ bb) / Wt.sum(1)[:, None]             # (reps, 3)
        return cm[:, None] + bm @ P[s, c]["share"].T

    def cpu_util(s, c, Wd, Wt):
        """(reps,): weighted mean CPU-measured utility."""
        u = P[s, c]["iou"][None] - lam * seconds(ctrl_c, bb_c, s, c, Wt)
        return (u * Wd).sum(1) / Wd.sum(1)

    ones_c = np.ones((1, len(cohort)))

    def contrast_v1(a, b):
        x = P[a]["util"] - P[b]["util"]
        return interval(wmean(W, x), x.mean())

    def contrast_cpu(a, b):
        reps = cpu_util(*a, W, Wc) - cpu_util(*b, W, Wc)
        point = (cpu_util(*a, ones, ones_c) - cpu_util(*b, ones, ones_c))[0]
        return interval(reps, point)

    report = {"objects": n_obj, "lambda": lam, "backbones": names,
              "baseline": "+".join(BASE), "bootstrap": {"reps": B, "kind":
              "stratified: 26 timing-cohort objects and 283 others; shared weights"},
              "pipelines": {}}
    print(f"dev, {n_obj} objects; lambda {lam}; baseline {'+'.join(BASE)}; {B} joint "
          "bootstrap reps")
    print(f"  {'pipeline':<28}{'IoU':>7}{'util v1':>9}  {'v1 - base [95% CI]':<28}"
          f"{'CPU util*':>10}  {'CPU - base [95% CI]':<28}{'CPU s':>7}{'GPU s':>7}")
    for s in STRATS:
        for c in (*DEPLOY, *REFS):
            e = P[s, c]
            row = {"iou": float(e["iou"].mean()), "util_v1": float(e["util"].mean()),
                   "util_v1_vs_base": contrast_v1((s, c), BASE),
                   "backbone_share": dict(zip(names, map(float, e["share"].mean(0))))}
            line = (f"  {s + '+' + c:<28}{row['iou']:>7.4f}{row['util_v1']:>9.4f}  "
                    "{0:+.4f} [{1:+.4f}, {2:+.4f}]  ".format(*row["util_v1_vs_base"]))
            if c in DEPLOY:
                row["util_cpu_est"] = float(cpu_util(s, c, ones, ones_c)[0])
                row["util_cpu_est_vs_base"] = contrast_cpu((s, c), BASE)
                row["cpu_s_est"] = float(seconds(ctrl_c, bb_c, s, c, ones_c).mean())
                row["gpu_s_est"] = float(seconds(ctrl_g, bb_g, s, c, ones_c).mean())
                line += ("{:>9.4f}  ".format(row["util_cpu_est"])
                         + "{0:+.4f} [{1:+.4f}, {2:+.4f}]".format(*row["util_cpu_est_vs_base"])
                         + f"{row['cpu_s_est']:>7.3f}{row['gpu_s_est']:>7.3f}")
            report["pipelines"][f"{s}+{c}"] = row
            print(line)
    b24 = P["best_of_24"]["util"]
    report["best_of_24_sampled_five_view_sets"] = {
        "util_v1": float(b24.mean()),
        "vs_base": interval(wmean(W, b24 - P[BASE]["util"]), (b24 - P[BASE]["util"]).mean())}
    print(f"  best of 24 sampled five-view sets (oracle backbone): util v1 {b24.mean():.4f}")
    print("  * CPU util and seconds are ESTIMATED from mean measured component times")

    # ── direct paired contrasts, both accountings ──
    pairs = [((s, "router_corrected"), (s, "router_plain")) for s in STRATS]
    pairs += [(("policy", c), ("heuristic", c)) for c in DEPLOY]
    pairs += [(("heuristic", c), ("random", c)) for c in DEPLOY]
    pairs += [(("heuristic", "umiformer_plus"), ("policy", "router_corrected")),
              (("heuristic", "pix2vox_f"), ("policy", "router_corrected"))]
    con = {}
    print(f"\npaired contrasts  {'':<50}{'v1 utility':<30}CPU utility (est.)")
    for a, b in pairs:
        k = f"{'+'.join(a)} - {'+'.join(b)}"
        con[k] = {"v1": contrast_v1(a, b), "cpu_est": contrast_cpu(a, b)}
        print(f"  {k:<64}" + "{0:+.4f} [{1:+.4f}, {2:+.4f}]   ".format(*con[k]["v1"])
              + "{0:+.4f} [{1:+.4f}, {2:+.4f}]".format(*con[k]["cpu_est"]))
    report["contrasts"] = con

    # ── the estimate against the cohort's measured end-to-end totals ──
    chk = {}
    cpos = {m: i for i, m in enumerate(cohort)}
    for s in STRATS:
        for c in DEPLOY:
            ps = [p for p in Lc["raw"]["paths"] if p["strategy"] == s and p["choice"] == c]
            est = [ctrl_c[s, c].mean() + bb_c[:, names.index(p["backbone"])].mean() for p in ps]
            chk[f"{s}+{c}"] = {"measured_s": float(np.mean([p["total_s"] for p in ps])),
                               "estimated_s": float(np.mean(est))}
    report["cpu_estimate_check"] = chk
    worst = max(abs(v["estimated_s"] / v["measured_s"] - 1) for v in chk.values())
    print(f"\nCPU estimate vs the cohort's measured end-to-end means: worst relative "
          f"error {worst*100:.1f}% over 15 pipelines")
    report["latency_files"] = {k: {f: L[f] for f in ("device_name", "threads", "torch")}
                               for k, L in (("cpu", Lc), ("gpu", Lg))}

    # ── lambda sensitivity, common baseline and absolute ──
    keyp = [("policy", "router_corrected"), ("policy", "router_plain"),
            ("heuristic", "router_corrected"), ("policy", "umiformer_plus"),
            ("heuristic", "umiformer_plus"), ("heuristic", "pix2vox_f")]

    def curve(lmb, d_):
        t = per_pipeline(lmb, d_)
        base = t[BASE]["util"]
        return t, {f"{s}+{c}": {"util": float(t[s, c]["util"].mean()),
                                "vs_random+umiformer_plus":
                                    interval(wmean(W, t[s, c]["util"] - base),
                                             (t[s, c]["util"] - base).mean())}
                   for s, c in keyp + [BASE]}

    sens = {"frozen": {}, "adaptive": {}}
    for lmb in LAMBDAS:
        sens["frozen"][f"{lmb:g}"] = curve(lmb, dec)[1]
    if not args.no_adaptive:
        dd, nm, obj, views, It = load_iou(args.feats, args.cache)
        assert tuple(nm) == FIXED
        mids, cats = dd["model_id"], dd["category"]
        if any(m in S["final_test"] for m in mids):
            raise SystemExit("final_test objects in the feature file")
        is_ctrl = np.array([m in S["controller_train"] for m in mids])
        allowed = is_ctrl | np.array([m in S["classifier_train"] for m in mids])
        Xt = set_features(dd["feats"], obj, views)
        tr = np.flatnonzero(is_ctrl[obj])
        _, folds = cv_folds(obj, tr, 5, 0)
        fclf = fold_classifiers(Xt, cats, obj, allowed, folds, classes, 0)
        for lmb in LAMBDAS:
            fit = fit_router(Xt, It - lmb * cost, obj, cats, classes, tr, fclf)
            t, row = curve(lmb, decisions(fit["table"], fit["ridge_mu"], fit["ridge_sd"],
                                          fit["ridge_W"], fit["beta"]))
            single = names[int(fit["single"].argmax())]
            row["best_single_on_training"] = single
            row["alpha"], row["beta"] = fit["alpha"], fit["beta"]
            sens["adaptive"][f"{lmb:g}"] = row
            print(f"  adaptive lambda {lmb:g}: training-best single {single}, "
                  f"alpha {fit['alpha']:g}, beta {fit['beta']:g}", flush=True)
    report["lambda_sensitivity"] = sens
    for kind, rows in sens.items():
        if not rows:
            continue
        print(f"\nlambda sensitivity ({kind}), utility minus random+umiformer_plus:")
        for lmb, row in rows.items():
            print(f"  {lmb:>6}  " + "  ".join(f"{k} {v['vs_random+umiformer_plus'][0]:+.4f}"
                                              for k, v in row.items()
                                              if isinstance(v, dict) and k != "+".join(BASE)))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
