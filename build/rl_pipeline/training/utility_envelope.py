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
records what produced each label, and refuses to mix incompatible ones:

    {"format": 2,
     "meta": {"scoring": SCORING, "thresholds": {...},
              "runs": {run_id: {...provenance..., "checkpoints":
                                {name: "sha256:..."} or "unknown"}}},
     "iou":  {key: {name: iou}},
     "src":  {key: [run_id, kind, ref]}}

`kind` is what generated the view set -- "train", "eval_policy", "eval_random",
or "unknown" for converted legacy entries -- and `ref` the policy step or
checkpoint. Converted legacy runs keep an "unknown" producer for good; before
their labels are reused, verify_legacy() recomputes a category-spread sample
with the live checkpoints and records the run as "compatible_with" them.

Thresholds are frozen per backbone before any of this runs. Left free, a router
would be learning "which reconstructor plus threshold is best" rather than
"which reconstructor is best", and the two are not the same question.
"""

import hashlib
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
# Bump whenever voxel_iou, the ground-truth threshold, the grid, or any
# backbone's preprocessing changes: labels across versions are not comparable.
SCORING = ("v1: voxel_iou(pred > frozen threshold, gt > 0.5) at 32^3; backbone "
           "preprocessing as of 2026-10-01")

_HASHES = {}


def sha256_file(path) -> str:
    path = str(path)
    if path not in _HASHES:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 24), b""):
                h.update(chunk)
        _HASHES[path] = "sha256:" + h.hexdigest()
    return _HASHES[path]


def checkpoint_ids(backbones, cfg) -> dict:
    """{name: content hash} of the weights each backbone loaded."""
    return {b.name: sha256_file(type(b)._ckpt_path(cfg)) for b in backbones}


def load_cache(path) -> dict:
    """A format-2 cache, or an empty one if `path` does not exist."""
    path = Path(path)
    if not path.is_file():
        return {"format": FORMAT, "meta": {"runs": {}}, "iou": {}, "src": {}}
    c = json.load(open(path))
    if c.get("format") != FORMAT or "src" not in c:
        raise ValueError(f"{path} is not a current format-{FORMAT} cache. Convert "
                         "the legacy file again with convert_legacy().")
    return c


def utilities(iou: dict, lam: float, costs: dict) -> dict:
    return {n: v - lam * costs[n] for n, v in iou.items()}


def _known(ckpts) -> bool:
    return isinstance(ckpts, dict)


class UtilityEnvelope:
    """
    Callable for RGBViewEnv.utility_fn.

    backbones   : list of loaded Backbone instances
    cost_lambda : IoU per unit of backbone.cost (cost v1: CPU seconds)
    cache_path  : format-2 JSON; loaded at construction, written by flush()
    run         : provenance for entries this process creates (script, seed,
                  policy, split...). Stored once in meta["runs"].
    checkpoints : checkpoint_ids(backbones, cfg). Omit only in tests; entries
                  are then recorded with "unknown" checkpoints.

    Set `.tag = {"kind": ..., "ref": ...}` before a batch of calls to record
    what generated the view sets that miss the cache.
    """

    def __init__(self, backbones, cost_lambda: float, thresholds: dict = None,
                 cache_path=None, run: dict = None, checkpoints: dict = None,
                 verbose: bool = False):
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
        for key, now in (("scoring", SCORING), ("thresholds", thr)):
            if meta.setdefault(key, now) != now:
                raise ValueError(f"cache {key} {meta[key]!r} != current {now!r}; "
                                 "these labels are not comparable. Use a new cache.")
        self.checkpoints = checkpoints
        if _known(checkpoints):
            for rid, r in meta["runs"].items():
                if _known(r.get("checkpoints")) and r["checkpoints"] != checkpoints:
                    raise ValueError(f"run {rid} labelled with other checkpoints "
                                     f"{r['checkpoints']} than these {checkpoints}")

        self.run_id = f"r{len(meta['runs'])}_{time.strftime('%Y%m%d-%H%M%S')}"
        meta["runs"][self.run_id] = {**(run or {}),
                                     "checkpoints": checkpoints or "unknown"}
        self.tag = {"kind": "unspecified", "ref": None}
        self.hits = self.misses = 0
        if verbose:
            print(f"  [utility] {len(self.cache['iou']):,} cached view sets, "
                  f"run {self.run_id}")

    # ── cache key ────────────────────────────────────────────────────────────

    @staticmethod
    def key(model_id: str, views) -> str:
        # Sorted: the reconstructors are order-invariant over the view set, so
        # [3,1,7] and [1,3,7] must not occupy two entries.
        return f"{model_id}|" + ",".join(str(v) for v in sorted(views))

    # ── evaluation ───────────────────────────────────────────────────────────

    def _predict_ious(self, item, views) -> dict:
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
        return out

    def ious(self, item, views) -> dict:
        """{backbone name: IoU at its frozen threshold}, cached."""
        k = self.key(item.get("model_id", "?"), views)
        if k in self.cache["iou"]:
            self.hits += 1
            return self.cache["iou"][k]
        self.misses += 1
        out = self._predict_ious(item, views)
        self.cache["iou"][k] = out
        self.cache["src"][k] = [self.run_id, self.tag.get("kind"), self.tag.get("ref")]
        return out

    def utilities(self, item, views) -> dict:
        """{backbone name: IoU - lambda*cost}."""
        return utilities(self.ious(item, views), self.lam, self.costs)

    def __call__(self, item, views) -> float:
        return max(self.utilities(item, views).values())

    def best(self, item, views) -> str:
        u = self.utilities(item, views)
        return max(u, key=u.get)

    # ── provenance ───────────────────────────────────────────────────────────

    def verify_legacy(self, get_item, category_of: dict, n: int = 26,
                      tol: float = 1e-3) -> dict:
        """
        Check that labels from runs whose checkpoints are unknown reproduce with
        the live ones, so they can be reused.

        Each such run is checked on its own: up to `n` of its entries, one per
        object, objects from `category_of` (model id -> category) taken
        round-robin across categories. A match is recorded as a
        "compatible_with" entry -- checkpoint hashes, keys, tolerance, max
        difference -- and the run's producer stays "unknown": a matching sample
        supports reusing the labels, it does not establish which weights made
        them. A mismatch raises.

        Returns {run_id: max |dIoU|, or None if the run had no checkable entry}
        for every run not already compatible with these checkpoints.
        """
        meta = self.cache["meta"]
        if not _known(self.checkpoints):
            return {}
        todo = {rid: {} for rid, r in meta["runs"].items()
                if rid != self.run_id and not _known(r.get("checkpoints"))
                and not any(c["checkpoints"] == self.checkpoints
                            for c in r.get("compatible_with", []))}
        seen = set()
        for k, (rid, _, _) in self.cache["src"].items():
            mid = k.split("|")[0]
            if rid in todo and mid in category_of and (rid, mid) not in seen:
                seen.add((rid, mid))
                todo[rid].setdefault(category_of[mid], []).append(k)

        out = {}
        for rid, groups in todo.items():
            lists = [groups[c] for c in sorted(groups)]
            keys = [l[i] for i in range(max(map(len, lists), default=0))
                    for l in lists if i < len(l)][:n]
            if not keys:
                out[rid] = None
                continue
            diff = 0.0
            for k in keys:
                mid, v = k.split("|")
                fresh = self._predict_ious(get_item(mid), [int(x) for x in v.split(",")])
                diff = max(diff, max(abs(fresh[b] - self.cache["iou"][k][b])
                                     for b in fresh))
            if diff > tol:
                raise ValueError(f"run {rid}: labels do not reproduce with these "
                                 f"checkpoints (max |dIoU| {diff:.2e}, {len(keys)} "
                                 "entries). Do not reuse them.")
            meta["runs"][rid].setdefault("compatible_with", []).append({
                "checkpoints": self.checkpoints, "checked_by": self.run_id,
                "keys": keys, "categories": sorted(category_of[k.split('|')[0]]
                                                   for k in keys),
                "tol": tol, "max_abs_diff": diff})
            out[rid] = diff
        return out

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
    recorded with the run, and its checkpoints stay "unknown" until
    UtilityEnvelope.verify_legacy() reproduces a sample of its labels.
    """
    old = json.load(open(src))
    if "format" in old:
        raise ValueError(f"{src} is not a legacy cache")
    new = {"format": FORMAT,
           "meta": {"scoring": SCORING, "thresholds": thresholds,
                    "runs": {"legacy": {"legacy_derived": True, "lambda": lam,
                                        "costs": costs, "note": note,
                                        "checkpoints": "unknown"}}},
           "iou": {k: {n: u + lam * costs[n] for n, u in v.items()}
                   for k, v in old.items()},
           "src": {k: ["legacy", "unknown", None] for k in old}}
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "w") as fh:
        json.dump(new, fh)
    return len(new["iou"])
