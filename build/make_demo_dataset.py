# -*- coding: utf-8 -*-
"""
make_demo_dataset.py — CPU-only ModelNet demo dataset builder.

Produces the exact layout the RL pipeline's ShapeNetDataset + dataloader
consume (see dataloader.py contract):

    out/{category}/{model_id}/
        images/{000..0NN}.png     256x256 shaded RGB on white background
        cameras.json              per-view {azimuth} (+ elevation/distance)
        vox_64.npy                (64, 64, 64) float32 occupancy

Silhouette/depth are derived downstream by the dataloader (silhouette from
white-bg threshold; depth back-projected from voxels + cameras) so we do not
need to emit them here.

Rendering is a pure-numpy painter's-algorithm z-buffer (CPU, no GL/GPU):
each view uses an orbit camera; triangles are back-to-front shaded with a
lambertian-ish term against a fixed light direction.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import trimesh

IMG = 256
ELEV = 20.0
DIST = 1.75
LIGHT = np.array([0.35, 0.70, 0.62])
LIGHT = LIGHT / np.linalg.norm(LIGHT)


def orbit_camera(azimuth_deg, elevation_deg=ELEV, distance=DIST):
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    eye = np.array([
        distance * math.cos(el) * math.cos(az),
        distance * math.sin(el),
        distance * math.cos(el) * math.sin(az),
    ])
    fwd = -eye / np.linalg.norm(eye)
    right = np.cross(np.array([0.0, 1.0, 0.0]), fwd)
    right /= np.linalg.norm(right)
    up = np.cross(fwd, right)
    return eye, np.stack([right, up, fwd], axis=1)


def normalize_mesh(mesh):
    m = mesh.copy()
    m.vertices -= m.centroid
    size = (m.bounds[1] - m.bounds[0]).max()
    if size > 1e-9:
        m.vertices /= size
    return m


def raster_views(mesh, n_views):
    """CPU painter's-algorithm renderer -> list of (rgb, sil, depth)."""
    m = normalize_mesh(mesh)
    f = (IMG / 2.0) / math.tan(math.radians(30.0) / 2.0)
    verts = m.vertices
    faces = m.faces
    out = []
    for i in range(n_views):
        az = 360.0 * i / n_views
        eye, rot = orbit_camera(az, ELEV, DIST)
        cam = (verts - eye) @ rot
        z = cam[:, 2]
        fsc = f / np.maximum(z, 1e-6)
        px = fsc * cam[:, 0] + IMG / 2.0
        py = IMG / 2.0 - fsc * cam[:, 1]
        rgb = np.full((IMG, IMG, 3), 0.95, dtype=np.float32)
        dep = np.full((IMG, IMG), np.inf, dtype=np.float32)
        sil = np.zeros((IMG, IMG), dtype=np.float32)
        # painter's algorithm: sort faces back-to-front
        face_z = z[faces].mean(axis=1)
        order = np.argsort(-face_z)
        for fi in order:
            t = faces[fi]
            if (z[t] <= 1e-3).any():
                continue
            p = np.stack([px[t], py[t]], axis=1)
            x0 = max(0, int(math.floor(p[:, 0].min())))
            x1 = min(IMG, int(math.ceil(p[:, 0].max())) + 1)
            y0 = max(0, int(math.floor(p[:, 1].min())))
            y1 = min(IMG, int(math.ceil(p[:, 1].max())) + 1)
            if x0 >= x1 or y0 >= y1:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1, dtype=np.float32),
                                 np.arange(y0, y1, dtype=np.float32))
            a, b, c = p[0], p[1], p[2]
            denom = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if abs(denom) < 1e-12:
                continue
            w0 = ((b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0])) / denom
            w1 = ((c[0] - a[0]) * (ys - a[1]) - (c[1] - a[1]) * (xs - a[0])) / denom
            w2 = 1.0 - w0 - w1
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
            if not inside.any():
                continue
            zi = w0 * z[t[0]] + w1 * z[t[1]] + w2 * z[t[2]]
            pxu, pyu = xs.astype(int), ys.astype(int)
            upd = inside & (zi < dep[pyu, pxu])
            if not upd.any():
                continue
            n = np.cross(verts[t[1]] - verts[t[0]], verts[t[2]] - verts[t[0]])
            n /= (np.linalg.norm(n) + 1e-9)
            shade = 0.40 + 0.60 * max(0.0, float(n @ LIGHT))
            rgb[pyu[upd], pxu[upd], :] = shade
            dep[pyu[upd], pxu[upd]] = zi[upd]
            sil[pyu[upd], pxu[upd]] = 1.0
        out.append((rgb, sil, dep))
    return out


def voxelize64(mesh, res=64):
    """Robust voxelize: normalize -> fit within res^3 cells -> center-pad."""
    m = normalize_mesh(mesh)
    m = m.copy()
    m.vertices = (m.vertices - m.bounds.min(axis=0)) / (m.bounds.max(axis=0) - m.bounds.min(axis=0)).max()
    pitch = 1.0 / (res * 0.95)   # leaves margin so full extent fits inside grid
    try:
        vox = m.voxelized(pitch=pitch)
        mat = vox.matrix.astype(np.float32)
    except Exception:
        return np.zeros((res, res, res), dtype=np.float32)
    if mat.shape == (res, res, res):
        return mat
    # center-pad or crop to exactly (res, res, res)
    out = np.zeros((res, res, res), dtype=np.float32)
    if np.any(np.array(mat.shape) > res):
        # crop centered then it will still be <= res
        offs = [(0 if s <= res else (s - res) // 2) for s in mat.shape]
        mat = mat[offs[0]:offs[0] + res, offs[1]:offs[1] + res, offs[2]:offs[2] + res]
    offs = [(res - s) // 2 for s in mat.shape]
    out[offs[0]:offs[0] + mat.shape[0],
        offs[1]:offs[1] + mat.shape[1],
        offs[2]:offs[2] + mat.shape[2]] = mat
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh-root", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--views", type=int, default=24)
    args = ap.parse_args()

    from PIL import Image
    mr, out = Path(args.mesh_root), Path(args.out_root)
    meshes = sorted([f for f in mr.iterdir() if f.suffix.lower() in ('.off', '.obj')])
    assert meshes, f"no meshes under {mr}"
    for meshf in meshes:
        m0 = trimesh.load(str(meshf), force='mesh')
        if isinstance(m0, trimesh.Scene):
            m0 = trimesh.util.concatenate(list(m0.geometry.values()))
        stem = meshf.stem
        cat = stem.split('_')[0]
        odir = out / cat / stem
        img_dir = odir / 'images'
        img_dir.mkdir(parents=True, exist_ok=True)

        vox = voxelize64(m0)
        np.save(str(odir / 'vox_64.npy'), vox)

        cams = []
        for i, (rgb, sil, dep) in enumerate(raster_views(m0, args.views)):
            png = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            Image.fromarray(png).save(str(img_dir / f"{i:03d}.png"))
            cams.append({"azimuth": 360.0 * i / args.views,
                         "elevation": ELEV,
                         "distance": DIST})
        with open(str(odir / 'cameras.json'), 'w') as fh:
            json.dump({"views": cams}, fh, indent=2)
        print(f"{cat}/{stem}: views={args.views} "
              f"vox_occ={(np.load(str(odir / 'vox_64.npy')) > 0).mean():.3f}")


if __name__ == "__main__":
    main()
