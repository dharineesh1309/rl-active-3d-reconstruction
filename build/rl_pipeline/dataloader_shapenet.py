"""
dataloader_shapenet.py
──────────────────────
Loader for the Choy et al. 2016 ShapeNet release — the data every backbone in
this project was pretrained on, and the reason cross-backbone IoU is comparable
at all.

It returns the same dict ShapeNetDataset in dataloader.py returns, so
ViewReconEnv needs no changes:

    {
        'images'      : list[PIL.Image]           # n_views, RGBA, native 137x137
        'depths'      : list[np.ndarray (32,32)]  # float32, normalised 0-1
        'silhouettes' : list[np.ndarray (H,W)]    # float32, {0.0, 1.0}
        'voxels'      : np.ndarray (32,32,32)     # float32 occupancy
        'category'    : str                       # e.g. 'chair'
        'model_id'    : str
    }

Two deliberate differences from the ModelNet loader:

  * **Images keep their alpha channel.** The renderings are RGBA with a
    transparent background, and Pix2Vox's preprocessing composites that alpha
    onto a specific near-white background. Calling .convert("RGB") here would
    discard it and silently change the input distribution the pretrained
    weights expect, so we hand PIL images through untouched.

  * **Voxels are 32**3, not 64**3.** That is the native ShapeNetVox32
    resolution and the scoring resolution for every backbone, so nothing is
    resampled anywhere in the reward path.

Silhouettes come straight from the alpha channel rather than the ModelNet
loader's white-pixel threshold, which is both exact and cheaper.

Layout expected
───────────────
    {rendering_root}/{synset}/{model_id}/rendering/00.png … 23.png
    {rendering_root}/{synset}/{model_id}/rendering/rendering_metadata.txt
    {voxel_root}/{synset}/{model_id}/model.binvox
"""

import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from dataloader import _depth_from_voxels   # reuse: same orthographic projection
from utils import binvox_rw

N_VIEWS = 24
COVERAGE_RES = 32       # matches CoverageGrid in env/state_builder.py
VIEW_PROBE_SAMPLES = 32  # models sampled to infer the view count


def _read_metadata(path: Path, n_views: int,
                   allow_fallback: bool = False) -> List[dict]:
    """
    Parse rendering_metadata.txt.

    Each line is: azimuth elevation in_plane_rotation distance field_of_view

    **Raises by default.** This used to return zeroed cameras for a missing or
    malformed file, on the reasoning that the coverage grid would merely degrade
    to a flat projection while training continued. That trade no longer holds:
    the pose-conditioned policy scores candidate views *from these numbers*, so a
    silent fallback would hand it 24 identical candidates and it would learn
    nothing, with no error anywhere. A crash is now far safer than a default.

    `allow_fallback=True` restores the old behaviour and exists for the
    validator, which needs to count unreadable objects rather than stop at the
    first one, and for the legacy RGB-D path.
    """
    default = {"azimuth": 0.0, "elevation": 0.0, "in_plane": 0.0,
               "distance": 1.0, "fov": 25.0}

    def _fail(msg):
        if allow_fallback:
            return None
        raise RuntimeError(
            f"Camera metadata unusable: {msg}\n  path: {path}\n"
            "The pose-conditioned policy reads candidate viewpoints from this "
            "file; a zeroed fallback would make every candidate identical. Pass "
            "allow_fallback=True only if you genuinely want that."
        )

    try:
        lines = path.read_text().strip().splitlines()
    except OSError as exc:
        if _fail(f"cannot read ({exc})") is None:
            return [dict(default) for _ in range(n_views)]

    if len(lines) < n_views:
        if _fail(f"{len(lines)} rows for {n_views} views") is None:
            lines = list(lines) + [""] * (n_views - len(lines))

    cams = []
    for i, line in enumerate(lines[:n_views]):
        parts = line.split()
        try:
            cams.append({
                "azimuth":   float(parts[0]),
                "elevation": float(parts[1]),
                "in_plane":  float(parts[2]) if len(parts) > 2 else 0.0,
                "distance":  float(parts[3]) if len(parts) > 3 else 1.0,
                "fov":       float(parts[4]) if len(parts) > 4 else 25.0,
            })
        except (IndexError, ValueError) as exc:
            if _fail(f"row {i} malformed ({exc}): {line!r}") is None:
                cams.append(dict(default))

    while len(cams) < n_views:
        cams.append(dict(default))
    return cams


def _silhouette_from_alpha(img: Image.Image) -> np.ndarray:
    """1.0 where the object is, 0.0 where the render is transparent."""
    arr = np.asarray(img)
    if arr.ndim == 3 and arr.shape[2] == 4:
        return (arr[:, :, 3] > 0).astype(np.float32)
    # No alpha channel: fall back to the white-background threshold.
    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    return (~np.all(rgb >= (250.0 / 255.0), axis=-1)).astype(np.float32)


class ShapeNetChoyDataset:
    """
    Choy 2016 ShapeNet, addressed by the official Pix2Vox taxonomy splits.

    Parameters
    ----------
    rendering_root : str   path to ShapeNetRendering/
    voxel_root     : str   path to ShapeNetVox32/
    taxonomy_path  : str   datasets/ShapeNet.json (carries train/test/val ids)
    split          : str   'train', 'test' or 'val'
    categories     : list[str] | None
        Taxonomy names ('chair') or synset ids ('03001627'). None = all 13.
    limit_per_category : int | None
        Cap models per category. Useful for a quick bench without the full split.
    preload : bool
        Cache every item up front. Fast, but the full train split will not fit
        in RAM — intended for small subsets only.
    """

    def __init__(
        self,
        rendering_root: str,
        voxel_root: str,
        taxonomy_path: str = "datasets/ShapeNet.json",
        split: str = "train",
        categories: Optional[List[str]] = None,
        limit_per_category: Optional[int] = None,
        n_views: Optional[int] = None,
        preload: bool = False,
        verify_views: bool = False,
    ):
        self.rendering_root = Path(rendering_root)
        self.voxel_root = Path(voxel_root)
        # None means detect from the data. The canonical Choy release has 24
        # views, but mirrors and subsets ship fewer, and silently indexing
        # views that are not there would fail deep inside a training run.
        self.n_views = n_views

        for label, root in (("rendering", self.rendering_root), ("voxel", self.voxel_root)):
            if not root.exists():
                raise FileNotFoundError(f"ShapeNet {label} root not found: {root}")

        with open(taxonomy_path) as f:
            taxonomy = json.load(f)

        wanted = set(categories) if categories else None
        self.samples: List[tuple] = []
        self.category_of: dict = {}

        for entry in taxonomy:
            synset, name = entry["taxonomy_id"], entry["taxonomy_name"]
            if wanted is not None and synset not in wanted and name not in wanted:
                continue
            self.category_of[synset] = name

            if split == "all":
                # Useful for a locally-fetched subset, where whichever models
                # you happen to hold are spread across the official splits.
                ids = entry.get("train", []) + entry.get("val", []) + entry.get("test", [])
            else:
                ids = entry.get(split, [])

            # Keep only what is actually on disk. Capping the id list first
            # would truncate to ids we may not hold and silently yield nothing,
            # so filter first -- but stop as soon as the cap is satisfied, since
            # each check is a filesystem stat and on a network-backed mount
            # (Kaggle) scanning all 43k models costs minutes per run.
            present = []
            for m in ids:
                if (self.voxel_root / synset / m / "model.binvox").is_file()                         and (self.rendering_root / synset / m / "rendering").is_dir():
                    present.append(m)
                    if limit_per_category is not None and len(present) >= limit_per_category:
                        break

            self.samples.extend((synset, m) for m in present)

        if not self.samples:
            raise RuntimeError(
                f"No ShapeNet samples found for split='{split}'.\n"
                f"  renderings: {self.rendering_root}\n"
                f"  voxels    : {self.voxel_root}\n"
                "Check both archives are extracted and that the category filter matches."
            )

        # Determine the view count from a sample rather than from every model.
        # Globbing all of them costs one directory listing per object, which on
        # the full 43k-model release is minutes of startup on every run. Taking
        # the most common count over a sample is just as reliable, because a
        # release either has a consistent view count or is broken.
        if self.n_views is None:
            sample = self.samples[::max(1, len(self.samples) // VIEW_PROBE_SAMPLES)]
            counts = []
            for synset, model_id in sample[:VIEW_PROBE_SAMPLES]:
                render_dir = self.rendering_root / synset / model_id / "rendering"
                counts.append(len(list(render_dir.glob("*.png"))))
            if not counts or max(counts) == 0:
                raise RuntimeError(
                    f"Models found but no rendering PNGs under {self.rendering_root}")
            self.n_views = max(set(counts), key=counts.count)
            if len(set(counts)) > 1:
                print(f"[ShapeNetChoy] view counts vary across the sample "
                      f"{sorted(set(counts))}; using {self.n_views}. Models with "
                      f"fewer views will error when first loaded.")

        if verify_views:
            # Opt-in exhaustive check, for a subset you suspect is incomplete.
            short = []
            for synset, model_id in self.samples:
                render_dir = self.rendering_root / synset / model_id / "rendering"
                if len(list(render_dir.glob("*.png"))) < self.n_views:
                    short.append((synset, model_id))
            if short:
                print(f"[ShapeNetChoy] skipping {len(short)} model(s) with fewer "
                      f"than {self.n_views} views")
                self.samples = [s for s in self.samples if s not in set(short)]

        cats = sorted({self.category_of[s] for s, _ in self.samples})
        print(f"[ShapeNetChoy] {len(self.samples)} models, split='{split}', "
              f"{self.n_views} views/model, {len(cats)} categories: {cats}")

        self._cache: dict = {}
        if preload:
            print("[ShapeNetChoy] Preloading ...", flush=True)
            for i in range(len(self.samples)):
                self._cache[i] = self._load(i)
            print("[ShapeNetChoy] Preload complete.")

    def _load(self, idx: int) -> dict:
        synset, model_id = self.samples[idx]
        render_dir = self.rendering_root / synset / model_id / "rendering"

        with open(self.voxel_root / synset / model_id / "model.binvox", "rb") as f:
            voxels = binvox_rw.read_as_3d_array(f).data.astype(np.float32)

        cams = _read_metadata(render_dir / "rendering_metadata.txt", self.n_views)

        images, depths, silhouettes = [], [], []
        for v in range(self.n_views):
            # Keep the alpha channel: the backbones composite it themselves.
            img = Image.open(render_dir / f"{v:02d}.png")
            img.load()
            images.append(img)
            silhouettes.append(_silhouette_from_alpha(img))
            depths.append(_depth_from_voxels(
                voxels, cams[v]["azimuth"], cams[v]["elevation"],
                grid_size=COVERAGE_RES,
            ))

        return {
            "images": images,
            "depths": depths,
            "silhouettes": silhouettes,
            "voxels": voxels,
            # Per-view camera parameters. The backbones ignore these; the pose
            # policy builds its view descriptors from them.
            "cams": cams,
            "category": self.category_of[synset],
            "model_id": model_id,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        if idx in self._cache:
            return self._cache[idx]
        return self._load(idx)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def build_shapenet(rendering_root=None, voxel_root=None, **kwargs) -> ShapeNetChoyDataset:
    """
    Build a ShapeNetChoyDataset, defaulting the roots from the environment.

    SHAPENET_ROOT may point at a parent holding both archives, in which case the
    two standard subdirectory names are used.
    """
    root = os.environ.get("SHAPENET_ROOT")
    if rendering_root is None:
        rendering_root = (os.environ.get("SHAPENET_RENDERING_ROOT")
                          or (os.path.join(root, "ShapeNetRendering") if root else None))
    if voxel_root is None:
        voxel_root = (os.environ.get("SHAPENET_VOXEL_ROOT")
                      or (os.path.join(root, "ShapeNetVox32") if root else None))

    if not rendering_root or not voxel_root:
        raise ValueError(
            "ShapeNet paths not set. Either pass rendering_root/voxel_root, or set "
            "SHAPENET_ROOT to the directory containing ShapeNetRendering/ and ShapeNetVox32/."
        )
    return ShapeNetChoyDataset(rendering_root, voxel_root, **kwargs)


if __name__ == "__main__":
    ds = build_shapenet(split="test", limit_per_category=2)
    item = ds[0]
    print(f"  category    : {item['category']}")
    print(f"  model_id    : {item['model_id']}")
    print(f"  images      : {len(item['images'])} x {item['images'][0].size} {item['images'][0].mode}")
    print(f"  silhouettes : {item['silhouettes'][0].shape}  coverage={item['silhouettes'][0].mean():.3f}")
    print(f"  depths      : {item['depths'][0].shape}  range=({item['depths'][0].min():.2f}, {item['depths'][0].max():.2f})")
    print(f"  voxels      : {item['voxels'].shape}  occupancy={item['voxels'].mean():.4f}")
    print("\nShapeNet dataloader OK.")
