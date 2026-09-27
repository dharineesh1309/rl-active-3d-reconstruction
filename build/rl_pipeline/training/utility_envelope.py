"""
Terminal reward for RGB view acquisition: the best achievable reconstruction
utility over the available reconstructors.

    V(S) = max_m [ IoU_m(S) - lambda * cost_m ]

Training the view policy against the envelope rather than against whatever a
learning router happened to pick removes a moving target: previously a good view
set could be punished because the model head chose the wrong backbone, and the
view head was learning against a policy that was itself still changing.

Every evaluation is cached by `(object_id, sorted(view_ids))`. The backbones are
frozen and deterministic, so a repeat costs nothing -- and the cache is not only
an optimisation. It records the utility of EVERY backbone on each view set, so
it is exactly the full-information dataset the supervised router needs, gathered
on the states the policy actually visits rather than on a uniform sample.

Thresholds are frozen per backbone before any of this runs. Left free, a router
would be learning "which reconstructor plus threshold is best" rather than
"which reconstructor is best", and the two are not the same question.
"""

import json
import os
from pathlib import Path

import numpy as np

from backbones import voxel_iou

# One frozen threshold per backbone, verified rather than assumed: swept over
# {0.2, 0.3, 0.4, 0.5} on 576 held-out states, 0.4 maximised the global mean for
# all three (pix2vox_f 0.7622, umiformer 0.8242, umiformer_plus 0.8301). Pinned
# so the router's labels cannot shift underneath it.
#
# Worth knowing how fine the resulting margins are: at 0.4, umiformer leads
# pix2vox_f by 0.062 IoU while lambda*delta_cost is 0.0594, so the average net
# margin is 0.0026. The choice flips easily per object, which is why the router
# is trained on utility regression with gap-weighted ranking rather than on
# argmax classification -- most flips cost almost nothing and should not be
# weighted like the expensive ones.
DEFAULT_THRESHOLDS = {"pix2vox_f": 0.4, "umiformer": 0.4, "umiformer_plus": 0.4}


class UtilityEnvelope:
    """
    Callable for RGBViewEnv.utility_fn.

    Parameters
    ----------
    backbones   : list of loaded Backbone instances
    cost_lambda : price per unit of backbone compute
    cache_path  : JSON file; loaded at construction, written by flush()
    """

    def __init__(self, backbones, cost_lambda: float = 0.07,
                 thresholds: dict = None, cache_path=None, verbose: bool = False):
        self.backbones = backbones
        self.names = [b.name for b in backbones]
        self.lam = cost_lambda
        self.thr = dict(DEFAULT_THRESHOLDS)
        if thresholds:
            self.thr.update(thresholds)
        missing = [n for n in self.names if n not in self.thr]
        if missing:
            raise ValueError(
                f"No frozen threshold for {missing}. Add it to DEFAULT_THRESHOLDS "
                "or pass thresholds=; an unfrozen threshold makes the router's "
                "labels depend on a free parameter.")
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        self.hits = self.misses = 0
        if self.cache_path and self.cache_path.is_file():
            self.cache = json.load(open(self.cache_path))
            if verbose:
                print(f"  [utility] loaded {len(self.cache)} cached view sets")

    # ── cache key ────────────────────────────────────────────────────────────

    @staticmethod
    def key(model_id: str, views) -> str:
        # Sorted: the reconstructors are order-invariant over the view set, so
        # [3,1,7] and [1,3,7] must not occupy two entries.
        return f"{model_id}|" + ",".join(str(v) for v in sorted(views))

    # ── evaluation ───────────────────────────────────────────────────────────

    def utilities(self, item, views) -> dict:
        """{backbone name: IoU - lambda*cost}, cached."""
        k = self.key(item.get("model_id", "?"), views)
        if k in self.cache:
            self.hits += 1
            return self.cache[k]

        self.misses += 1
        imgs = [item["images"][v] for v in views]
        cams = [item["cams"][v] for v in views]
        gt = item["voxels"]
        out = {}
        for b in self.backbones:
            pred = b.predict(imgs, cams)
            iou = voxel_iou(pred, gt, pred_thresh=self.thr[b.name])
            out[b.name] = float(iou - self.lam * b.cost)
        self.cache[k] = out
        return out

    def __call__(self, item, views) -> float:
        u = self.utilities(item, views)
        return max(u.values())

    def best(self, item, views) -> str:
        u = self.utilities(item, views)
        return max(u, key=u.get)

    # ── persistence ──────────────────────────────────────────────────────────

    def flush(self):
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(".tmp")
        with open(tmp, "w") as fh:
            json.dump(self.cache, fh)
        os.replace(tmp, self.cache_path)    # atomic; a killed session cannot
                                            # leave a half-written cache behind

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {"entries": len(self.cache), "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total else 0.0}

    def export_router_dataset(self, path):
        """
        Dump the cache as the router's supervised training set.

        One row per (object, view set) with the utility of every backbone, which
        is the full-information signal that makes routing a regression problem
        rather than a bandit. This is what removes the exploration failure that
        left Pix2Vox-F sampled about nine times on cars across 30,000 episodes.
        """
        rows = []
        for k, u in self.cache.items():
            mid, views = k.split("|")
            rows.append({"model_id": mid,
                         "views": [int(v) for v in views.split(",") if v != ""],
                         "utilities": u,
                         "best": max(u, key=u.get)})
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"lambda": self.lam, "thresholds": self.thr,
                       "backbones": self.names, "rows": rows}, fh, indent=1)
        return len(rows)
