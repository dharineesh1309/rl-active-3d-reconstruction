"""
Tier 1: supervised backbone router (CPU).

    python train_router.py --feats artifacts/router/feats.npz \
                           --cache cache/utility_cache.json

Every cached view set carries the utility of all three backbones, so routing is
full-information supervised learning, not a bandit. The router sees what the
view policy sees -- image features and camera poses of the acquired views, no
category label -- predicts each backbone's utility, and routes to the argmax.

Loss: centred-utility regression plus gap-weighted pairwise ranking. Many argmax
flips between backbones cost almost nothing; weighting each ordering mistake by
|U_m - U_n| makes it cost what it actually costs. Only differences between
backbones are learned: the object's overall difficulty cannot change the route.

Split by OBJECT, never by row (each object has dozens of rows):
    test  cached objects in the official test split -- the Tier 0B eval objects
    val   a held-out slice of the remaining objects, for early stopping
    fit   the rest

Reported on test objects, each object weighted equally:
    single    best fixed backbone, chosen on fit
    category  best backbone per category, chosen on fit (uses the label)
    router    argmax of the router's predictions (does not)
    oracle    best backbone for each view set
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from policy.pose_policy import D_MODEL, ImageProjection, PoseEncoder, ViewSetEncoder


class Router(nn.Module):
    """The policy's own image/pose tokens, mean-pooled (Tier 0B found the set
    transformer added nothing measurable), then one utility per backbone."""

    def __init__(self, n_out: int):
        super().__init__()
        self.img_proj = ImageProjection()
        self.pose_enc = PoseEncoder()
        self.set_enc = ViewSetEncoder(use_transformer=False)
        self.head = nn.Sequential(nn.Linear(D_MODEL, 128), nn.GELU(),
                                  nn.Linear(128, n_out))

    def forward(self, feats, poses):
        mask = torch.ones(feats.shape[:2], device=feats.device)
        return self.head(self.set_enc(self.img_proj(feats), self.pose_enc(poses), mask))


def router_loss(s, U, sigma, rank_weight=1.0):
    """Centred MSE + gap-weighted pairwise logistic ranking, both in units of sigma."""
    y = (U - U.mean(1, keepdim=True)) / sigma
    mse = F.mse_loss(s - s.mean(1, keepdim=True), y)
    i, j = torch.triu_indices(U.shape[1], U.shape[1], 1)
    du = (U[:, i] - U[:, j]) / sigma
    rank = (du.abs() * F.softplus(-torch.sign(du) * (s[:, i] - s[:, j]))).mean()
    return mse + rank_weight * rank


def object_mean(x, obj):
    """Mean within each object, then across objects."""
    cnt = np.bincount(obj)
    has = cnt > 0
    return float((np.bincount(obj, weights=x)[has] / cnt[has]).mean())


def load(feats_path, cache_path):
    d = np.load(feats_path)
    idx = {m: i for i, m in enumerate(d["model_id"])}
    cache = json.load(open(cache_path))
    names = sorted(next(iter(cache.values())))
    obj, views, U = [], [], []
    for k, u in cache.items():
        m, v = k.split("|")
        if m in idx:
            obj.append(idx[m])
            views.append([int(x) for x in v.split(",")])
            U.append([u[n] for n in names])
    views = np.array(views)
    assert views.ndim == 2, "view sets of different sizes; batch them per size"
    return d, names, np.array(obj), views, np.array(U, np.float32)


@torch.no_grad()
def predict(model, F_all, P_all, obj, views, bs=2048):
    model.eval()
    out = []
    for a in range(0, len(obj), bs):
        o, v = obj[a:a + bs, None], views[a:a + bs]
        out.append(model(F_all[o, v].float(), P_all[o, v]).numpy())
    return np.concatenate(out)


def fit(F_all, P_all, obj, views, U, fit_rows, val_rows, epochs=40, patience=6,
        lr=1e-3, rank_weight=1.0, seed=0, verbose=True):
    """Train on fit_rows, keep the epoch with the lowest object-averaged val regret."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Router(U.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    Ut = torch.from_numpy(U)
    c = U[fit_rows] - U[fit_rows].mean(1, keepdims=True)
    sigma = float(c.std())
    best = (float("inf"), None, -1)
    bad = 0
    for ep in range(epochs):
        model.train()
        order = rng.permutation(fit_rows)
        for a in range(0, len(order), 512):
            r = order[a:a + 512]
            o, v = obj[r, None], views[r]
            s = model(F_all[o, v].float(), P_all[o, v])
            loss = router_loss(s, Ut[r], sigma, rank_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        s = predict(model, F_all, P_all, obj[val_rows], views[val_rows])
        Uv = U[val_rows]
        regret = Uv.max(1) - Uv[np.arange(len(Uv)), s.argmax(1)]
        reg = object_mean(regret, obj[val_rows])
        if verbose:
            print(f"  epoch {ep:>2}  loss {loss.item():.4f}  val regret {reg:.5f}",
                  flush=True)
        if reg < best[0] - 1e-6:
            best = (reg, {k: t.clone() for k, t in model.state_dict().items()}, ep)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best[1])
    return model, best[2]


def report(U, obj, rows, picks, names):
    """Object-averaged utility and regret per method on `rows`."""
    Ur, o = U[rows], obj[rows]
    oracle = object_mean(Ur.max(1), o)
    out = {}
    for name, p in picks.items():
        u = object_mean(Ur[np.arange(len(rows)), p], o)
        out[name] = {"utility": u, "regret": oracle - u,
                     "share": {n: float(np.mean(p == k)) for k, n in enumerate(names)}}
    out["oracle"] = {"utility": oracle, "regret": 0.0}
    gap = oracle - out["single"]["utility"]
    for v in out.values():
        v["captured"] = (v["utility"] - out["single"]["utility"]) / gap if gap > 0 else float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="artifacts/router/feats.npz")
    ap.add_argument("--cache", default="cache/utility_cache.json")
    ap.add_argument("--out", default="artifacts/router")
    ap.add_argument("--rank-weight", type=float, default=1.0)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d, names, obj, views, U = load(args.feats, args.cache)
    F_all = torch.from_numpy(d["feats"])
    P_all = torch.from_numpy(d["poses"])
    cats = d["category"]

    is_test = d["split"] == "test"
    rest = np.flatnonzero(~is_test)
    rng = np.random.default_rng(args.seed)
    val_obj = set(rng.choice(rest, int(len(rest) * args.val_frac), replace=False).tolist())
    test_rows = np.flatnonzero(is_test[obj])
    val_rows = np.flatnonzero([(not is_test[o]) and o in val_obj for o in obj])
    fit_rows = np.flatnonzero([(not is_test[o]) and o not in val_obj for o in obj])
    assert not (set(obj[fit_rows]) & set(obj[test_rows])), "object leak fit/test"
    assert not (set(obj[fit_rows]) & set(obj[val_rows])), "object leak fit/val"
    print(f"backbones {names}")
    for n, r in (("fit", fit_rows), ("val", val_rows), ("test", test_rows)):
        print(f"  {n:<5} {len(r):>6} rows  {len(set(obj[r])):>5} objects")

    model, ep = fit(F_all, P_all, obj, views, U, fit_rows, val_rows,
                    rank_weight=args.rank_weight, seed=args.seed)
    print(f"best epoch {ep}")

    # Baselines are chosen on fit rows only.
    Uf, of = U[fit_rows], obj[fit_rows]
    single = int(np.argmax([object_mean(Uf[:, k], of) for k in range(len(names))]))
    by_cat = {}
    for c in np.unique(cats):
        r = cats[of] == c
        by_cat[c] = (int(np.argmax([object_mean(Uf[r, k], of[r]) for k in range(len(names))]))
                     if r.any() else single)

    s = predict(model, F_all, P_all, obj[test_rows], views[test_rows])
    picks = {"single": np.full(len(test_rows), single),
             "category": np.array([by_cat.get(c, single) for c in cats[obj[test_rows]]]),
             "router": s.argmax(1)}
    res = report(U, obj, test_rows, picks, names)

    print(f"\ntest, {len(set(obj[test_rows]))} held-out objects, object-averaged:")
    print(f"  {'':<9} {'utility':>8} {'regret':>8} {'captured':>9}   picks")
    for k, v in res.items():
        share = "  ".join(f"{n} {p*100:4.1f}%" for n, p in v.get("share", {}).items())
        print(f"  {k:<9} {v['utility']:>8.4f} {v['regret']:>8.4f} "
              f"{v['captured']*100:>8.1f}%   {share}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"router": model.state_dict(), "backbones": names,
                "best_epoch": ep, "args": vars(args)}, out / "router.pt")
    (out / "router_report.json").write_text(json.dumps(
        {"backbones": names, "single": names[single],
         "category": {c: names[k] for c, k in by_cat.items()}, "test": res}, indent=1))
    print(f"\nwrote {out / 'router.pt'} and router_report.json")


if __name__ == "__main__":
    main()
