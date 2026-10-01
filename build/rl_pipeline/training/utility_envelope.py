"""
Terminal reward for RGB view acquisition: the best achievable reconstruction
utility over the available reconstructors.

    V(S) = max_m [ IoU_m(S) - lambda * cost_m ]

Training the view policy against the envelope rather than against whatever a
learning router happened to pick removes a moving target: previously a good view
set could be punished because the model head chose the wrong backbone, and the
view head was learning against a policy that was itself still changing.

Every evaluation is cached by `(object_id, sorted(view_ids))`. The backbones are
deterministic in practice (HANDOFF, "Backbone determinism") and order-free, so
a repeat costs nothing -- and the cache is not only an optimisation. It records
every backbone's IoU on each view set, which is the full-information dataset
the supervised router trains on.

Cache format 2 stores RAW IoU, not utility, so lambda and the costs apply at
read time and can change without re-running a single reconstruction. It also
records what produced the labels -- thresholds, checkpoints, the run -- and
refuses to mix labels from different thresholds or checkpoints.

    {"format": 2,
     "meta": {"thresholds": {...}, "checkpoints": {name: "file:bytes"},
              "runs": {run_id: {...provenance...}}},
     "iou":  {key: {name: iou}},
     "run":  {key: run_id}}

Thresholds are frozen per backbone before any of this runs. Left free, a router
would be learning "which reconstructor plus threshold is best" rather than
"which reconstructor is best", and the two are not the same question.
"""

import json
import os
import time
from pathlib import Path

import torch

from backbones import voxel_iou

# One frozen threshold per backbone, verified rather than assumed: swept over
# {0.2, 0.3, 0.4, 0.5} on 576 held-out states, 0.4 maximised the global mean for
# all three (pix2vox_f 0.7622, umiformer 0.8242, umiformer_plus 0.8301). Pinned
# so the router's labels cannot shift underneath it.
DEFAULT_THRESHOLDS = {"pix2vox_f": 0.4, "umiformer": 0.4, "umiformer_plus": 0.4}

FORMAT = 2


def checkpoint_ids(backbones, cfg) -> dict:
    """{name: 'file:bytes'} -- cheap identity for the weights that made a label."""
    out = {}
    for b in backbones:
        p = type(b)._ckpt_path(cfg)
        out[b.name] = f"{os.path.basename(p)}:{os.path.getsize(p)}"
    return out


def load_cache(path) -> dict:
    """A format-2 cache, or an empty one if `path` does not exist."""
    path = Path(path)
    if not path.is_file():
        return {"format": FORMAT, "meta": {"runs": {}}, "iou": {}, "run": {}}
    c = json.load(open(path))
    if c.get("format") != FORMAT:
        raise ValueError(f"{path} is a legacy cache (utilities, not IoU). "
                         "Convert it once with convert_legacy().")
    return c


def utilities(iou: dict, lam: float, costs: dict) -> dict:
    return {n: v - lam * costs[n] for n, v in iou.items()}


class UtilityEnvelope:
    """
    Callable for RGBViewEnv.utility_fn.

    backbones   : list of loaded Backbone instances
    cost_lambda : IoU per unit of backbone.cost (cost v1: CPU seconds)
    cache_path  : format-2 JSON; loaded at construction, written by flush()
    run         : provenance for entries this process creates (script, seed,
                  policy, split...). Stored once in meta["runs"].
    cfg         : Config, to identify the checkpoints. Omit only in tests.
    """

    def __init__(self, backbones, cost_lambda: float, thresholds: dict = None,
                 cache_path=None, run: dict = None, cfg=None, verbose: bool = False):
        self.backbones = backbones
        self.names = [b.name for b in backbones]
        self.lam = cost_lambda
        self.costs = {b.name: b.cost for b in backbones}
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
        self.cache = load_cache(self.cache_path) if self.cache_path else load_cache("")
        meta = self.cache["meta"]
        thr = {n: self.thr[n] for n in self.names}
        ckpts = checkpoint_ids(backbones, cfg) if cfg is not None else None
        for key, now in (("thresholds", thr), ("checkpoints", ckpts)):
            if now is None:
                continue
            if meta.get(key, now) != now:
                raise ValueError(f"cache {key} {meta[key]} != current {now}; "
                                 "these labels are not comparable. Use a new cache.")
            meta[key] = now

        self.run_id = f"r{len(meta['runs'])}_{time.strftime('%Y%m%d-%H%M%S')}"
        meta["runs"][self.run_id] = dict(run or {})
        self.hits = self.misses = 0
        if verbose:
            print(f"  [utility] {len(self.cache['iou'])} cached view sets, "
                  f"run {self.run_id}")

    # ── cache key ────────────────────────────────────────────────────────────

    @staticmethod
    def key(model_id: str, views) -> str:
        # Sorted: the reconstructors are order-invariant over the view set, so
        # [3,1,7] and [1,3,7] must not occupy two entries.
        return f"{model_id}|" + ",".join(str(v) for v in sorted(views))

    # ── evaluation ───────────────────────────────────────────────────────────

    def ious(self, item, views) -> dict:
        """{backbone name: IoU at its frozen threshold}, cached."""
        k = self.key(item.get("model_id", "?"), views)
        if k in self.cache["iou"]:
            self.hits += 1
            return self.cache["iou"][k]

        self.misses += 1
        imgs = [item["images"][v] for v in views]
        cams = [item["cams"][v] for v in views]
        out = {}
        # UMIFormer draws torch.rand at inference. Forked and reseeded, so the
        # backbones never advance the policy's RNG stream -- otherwise whether a
        # view set happened to be cached would change the policy's sampling.
        with torch.random.fork_rng():
            torch.manual_seed(0)
            for b in self.backbones:
                pred = b.predict(imgs, cams)
                out[b.name] = float(voxel_iou(pred, item["voxels"],
                                              pred_thresh=self.thr[b.name]))
        self.cache["iou"][k] = out
        self.cache["run"][k] = self.run_id
        return out

    def utilities(self, item, views) -> dict:
        """{backbone name: IoU - lambda*cost}."""
        return utilities(self.ious(item, views), self.lam, self.costs)

    def __call__(self, item, views) -> float:
        return max(self.utilities(item, views).values())

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
        return {"entries": len(self.cache["iou"]), "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / total if total else 0.0}


def convert_legacy(src, dst, lam: float, costs: dict, thresholds: dict, note: str):
    """
    Legacy cache {key: {name: utility}} -> format 2, as IoU = utility + lam*cost.

    Only valid when the lambda and costs that wrote `src` are known; they are
    recorded with the run so the recovered IoUs stay marked as derived.
    """
    old = json.load(open(src))
    if old.get("format") == FORMAT:
        raise ValueError(f"{src} is already format {FORMAT}")
    run_id = "legacy"
    new = {"format": FORMAT,
           "meta": {"thresholds": thresholds,
                    "runs": {run_id: {"legacy_derived": True, "lambda": lam,
                                      "costs": costs, "note": note}}},
           "iou": {k: {n: u + lam * costs[n] for n, u in v.items()}
                   for k, v in old.items()},
           "run": {k: run_id for k in old}}
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(new, fh)
    return len(new["iou"])
