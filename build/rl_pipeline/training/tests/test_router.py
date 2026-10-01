"""
The router learns a routing signal that exists only in the images.

Synthetic: each object has a hidden type in {0, 1, 2} written into its image
features, and backbone k wins by 0.05 on type-k objects. On top sits a large
per-object difficulty offset shared by all backbones (sd 0.06, larger than the
signal) -- the router must learn the differences, not the level.

A fixed backbone has regret ~0.033 here. The router must cut that by 70%+.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from train_router import fit, object_mean, predict


def main():
    rng = np.random.default_rng(0)
    n_obj, V, per_obj = 240, 24, 20
    typ = rng.integers(3, size=n_obj)
    proto = rng.normal(size=(3, 2048)).astype(np.float32)
    feats = np.abs(rng.normal(size=(n_obj, V, 2048)) * 0.5 + proto[typ][:, None])
    F_all = torch.from_numpy(feats.astype(np.float16))
    P_all = torch.from_numpy(rng.normal(size=(n_obj, V, 5)).astype(np.float32))

    obj = np.repeat(np.arange(n_obj), per_obj)
    views = np.array([rng.choice(V, 5, replace=False) for _ in obj])
    level = rng.normal(0.75, 0.06, size=n_obj)
    U = (level[obj][:, None] + 0.05 * (np.arange(3)[None] == typ[obj][:, None])
         + rng.normal(0, 0.01, size=(len(obj), 3))).astype(np.float32)

    split = rng.permutation(n_obj)
    fit_o, val_o, test_o = set(split[:160]), set(split[160:200]), set(split[200:])
    rows = lambda S: np.flatnonzero([o in S for o in obj])
    fit_rows, val_rows, test_rows = rows(fit_o), rows(val_o), rows(test_o)

    model, _ = fit(F_all, P_all, obj, views, U, fit_rows, val_rows,
                   epochs=15, verbose=False)
    s = predict(model, F_all, P_all, obj[test_rows], views[test_rows])
    Ut, ot = U[test_rows], obj[test_rows]
    oracle = Ut.max(1)
    router = object_mean(oracle - Ut[np.arange(len(Ut)), s.argmax(1)], ot)
    single = min(object_mean(oracle - Ut[:, k], ot) for k in range(3))
    print(f"test regret: single {single:.4f}  router {router:.4f}")
    assert router < 0.3 * single, "router did not learn the image-borne signal"
    print("PASS")


if __name__ == "__main__":
    main()
