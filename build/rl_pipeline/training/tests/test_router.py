"""
The two-stage router recovers a routing signal that exists only in the images.

Synthetic: 6 categories, each with its own feature prototype, and category c's
objects are best served by backbone c % 3 (by 0.05 IoU). A large per-object
difficulty offset (sd 0.06) sits on top. Fixed-backbone regret is ~0.033; the
router, never shown a label at routing time, must cut it by 80%+.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from train_router import (class_scores, fit_classifier, fit_table, fit_temperature,
                          object_mean, route)


def main(shuffle=False):
    rng = np.random.default_rng(0)
    n_obj, per_obj, C = 300, 12, 6
    cat = rng.integers(C, size=n_obj)
    proto = rng.normal(size=(C, 64))
    X_obj = proto[cat] + rng.normal(scale=0.8, size=(n_obj, 64))
    if shuffle:
        X_obj = X_obj[rng.permutation(n_obj)]
    obj = np.repeat(np.arange(n_obj), per_obj)
    X = X_obj[obj] + rng.normal(scale=0.3, size=(len(obj), 64))
    level = rng.normal(0.75, 0.06, size=n_obj)
    U = (level[obj][:, None] + 0.05 * (np.arange(3)[None] == (cat[obj] % 3)[:, None])
         + rng.normal(0, 0.01, size=(len(obj), 3)))
    classes = np.arange(C)
    y = cat[obj]

    tr = obj < 200
    te = ~tr
    clf = fit_classifier(X[tr], y[tr], obj[tr], classes)
    T = fit_temperature(class_scores(clf, X[tr]), y[tr], classes)
    single = np.array([object_mean(U[tr, k], obj[tr]) for k in range(3)])
    table = fit_table(U[tr], obj[tr], y[tr], classes, single)
    picks = route(class_scores(clf, X[te]), T, table)

    Ut, ot = U[te], obj[te]
    regret = lambda p: object_mean(Ut.max(1) - Ut[np.arange(len(Ut)), p], ot)
    r_single = regret(np.full(len(Ut), single.argmax()))
    r = {k: regret(p) for k, p in picks.items()}
    print(f"test regret: single {r_single:.4f}  "
          + "  ".join(f"{k} {v:.4f}" for k, v in r.items()))
    return r_single, r


if __name__ == "__main__":
    r_single, r = main()
    assert all(v < 0.2 * r_single for v in r.values()), "router missed the image signal"
    r_single, r = main(shuffle=True)
    assert all(v > 0.5 * r_single for v in r.values()), "router 'learned' shuffled features"
    print("PASS")
