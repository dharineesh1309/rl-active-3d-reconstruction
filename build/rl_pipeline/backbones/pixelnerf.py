"""
pixelNeRF backbone (sxyu/pixel-nerf, sn64 multi-category ShapeNet model).

This is the odd one out, and the reason it is worth having: unlike the voxel
backbones it does not predict occupancy at all. It predicts a radiance field —
colour and volume density at arbitrary 3D points — so occupancy has to be read
out of the density field. It also has a steep dependence on view count, which
is exactly the behaviour that makes the policy's budget/backbone trade-off
non-trivial.

How occupancy is obtained
─────────────────────────
Query the network on the 32**3 scoring grid, take the density sigma, and
convert with the standard volume-rendering opacity of a single cell:

    occupancy = 1 - exp(-sigma * cell_size)

That is the same alpha NeRF integrates along a ray, evaluated over one voxel,
so it is a principled reading of "how much matter is in this cell" rather than
an arbitrary threshold on an unbounded quantity.

Cameras
───────
pixelNeRF consumes DVR-preprocessed `cameras.npz` matrices, which the Choy
renderings do not ship. The poses are therefore reconstructed from Choy's
rendering_metadata.txt (azimuth, elevation, in-plane rotation, distance, fov)
using the placement formula from the 3D-R2N2 render script:

    x = d cos(az) cos(el),  y = d sin(az) cos(el),  z = d sin(el)

looking at the origin in a Z-up world. Poses are built directly in pixelNeRF's
camera convention (x right, y up, z backward — OpenGL), which is what
PixelNeRFNet.encode expects after its own coordinate transforms. Focal length
comes from the recorded field of view: focal_px = (W/2) / tan(fov/2).

STATUS: NOT REGISTERED AS AN ACTIVE BACKBONE
─────────────────────────────────────────────
This module loads, runs, and produces a volume, but the cameras derived from
Choy metadata do not agree with the renderings, so its output is not
trustworthy and it is deliberately left out of the registry in __init__.py.
An unvalidated backbone in the action space would quietly corrupt the reward
rather than fail loudly.

What was measured
─────────────────
The camera convention was tested independently of the network, by projecting
known ground-truth voxels through the reconstructed cameras and measuring what
fraction lands inside the rendered silhouette. A correct convention scores
~0.95. Sweeping azimuth sign and offset, elevation sign, world up-axis, and all
48 grid orientations — over 1500 combinations — the best reached only **0.55**,
with equivalent rotations tying at that ceiling rather than one convention
standing out. Using binvox's own translate/scale header made it *worse* (0.36),
which says the renderer re-normalises each model by something the metadata does
not record.

The pixelNeRF sweep agreed: IoU never plateaued across scales, and the
best-scoring orientation changed at every scale — the signature of fitting
noise, not of finding an alignment.

What would actually fix it
──────────────────────────
Use the data pixelNeRF consumes. The DVR/NMR release ships `cameras.npz` with
exact world_mat/camera_mat per view, which is what DVRDataset reads, removing
the reconstruction entirely. It is ~36 GB for all 13 categories, impractical on
a slow link but unremarkable on the GPU box. Once those cameras are in hand,
`cams_to_poses` is replaced by reading the matrices and this backbone can be
registered and benchmarked like the others.

A separate, smaller caveat remains even then: sn64 was trained on NMR
renderings at 64x64, not Choy's 137x137 — same shapes, different renderer and
lighting, so some domain shift is unavoidable.
"""

import math
import os

import numpy as np
import torch
from PIL import Image

from . import VOXEL_RES, Backbone
from ._vendor.pixelnerf.conf import SN64
from ._vendor.pixelnerf.models import PixelNeRFNet

IMG_SIZE = 137          # Choy renderings are 137x137
POINT_CHUNK = 4096      # query points per forward; 32**3 at once exhausts RAM

# Object extent visible at the nominal Choy camera: 2 * d * tan(fov/2) with
# d ~ 0.8 and fov 25 degrees. This is how large the object is in the renderer's
# own units, and so how wide the scoring cube must be before rescaling.
BASE_EXTENT = 0.35

# Single scale knob, applied to BOTH the camera distances and the grid.
#
# Scaling only the grid would change the geometry -- growing the object towards
# fixed cameras. Scaling both leaves every projection identical while moving the
# whole scene to the absolute scale the checkpoint was trained at, which matters
# because the MLP consumes camera-space xyz directly rather than just ray
# directions. sn64 was trained on NMR, where objects are ~1 unit and cameras
# ~2 units out, so the expected value is roughly 1 / BASE_EXTENT.
WORLD_SCALE = 2.85
AXES = {"perm": (0, 1, 2), "flip": (False, False, False)}


def _look_at_pose(eye: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """
    Camera-to-world matrix looking from `eye` at the origin, OpenGL convention.

    Columns are (right, up, back): +Z points from the target back towards the
    camera, so points in front of the camera have negative z in camera space,
    which is what pixelNeRF's uv = -xy/z assumes.
    """
    back = eye / (np.linalg.norm(eye) + 1e-12)
    up = np.asarray(up, dtype=np.float64)
    if abs(float(np.dot(up, back))) > 0.999:        # degenerate straight-down view
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, back)
    right /= (np.linalg.norm(right) + 1e-12)
    true_up = np.cross(back, right)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = back
    pose[:3, 3] = eye
    return pose


def cams_to_poses(cams, scale=None):
    """
    Choy camera records -> (NS, 4, 4) camera-to-world poses and a focal length.

    `cams` entries carry azimuth/elevation/distance in degrees and world units,
    as parsed by dataloader_shapenet._read_metadata.
    """
    scale = WORLD_SCALE if scale is None else scale
    poses, focal = [], None
    for cam in cams:
        az = math.radians(cam["azimuth"])
        el = math.radians(cam["elevation"])
        d = cam["distance"] * scale
        eye = np.array([
            d * math.cos(az) * math.cos(el),
            d * math.sin(az) * math.cos(el),
            d * math.sin(el),
        ], dtype=np.float64)
        poses.append(_look_at_pose(eye))

        fov = math.radians(cam.get("fov", 25.0))
        focal = (IMG_SIZE / 2.0) / math.tan(fov / 2.0)

    return torch.from_numpy(np.stack(poses)), float(focal)


def _grid_points(scale: float = None) -> torch.Tensor:
    """The 32**3 scoring grid, in world units, ordered so it reshapes to (X,Y,Z)."""
    scale = WORLD_SCALE if scale is None else scale
    lin = torch.linspace(-0.5, 0.5, VOXEL_RES) * (BASE_EXTENT * scale)
    xs = lin.view(-1, 1, 1).expand(VOXEL_RES, VOXEL_RES, VOXEL_RES)
    ys = lin.view(1, -1, 1).expand(VOXEL_RES, VOXEL_RES, VOXEL_RES)
    zs = lin.view(1, 1, -1).expand(VOXEL_RES, VOXEL_RES, VOXEL_RES)
    return torch.stack([xs, ys, zs], dim=-1).reshape(-1, 3)


def _apply_axes(volume: np.ndarray, axes=None) -> np.ndarray:
    axes = AXES if axes is None else axes
    out = np.transpose(volume, axes["perm"])
    for axis, flip in enumerate(axes["flip"]):
        if flip:
            out = np.flip(out, axis=axis)
    return np.ascontiguousarray(out)


def _preprocess(images) -> torch.Tensor:
    """PIL images -> (NS, 3, H, W) in [-1, 1], matching get_image_to_tensor_balanced."""
    tensors = []
    for img in images:
        img = img.convert("RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = (arr - 0.5) / 0.5
        tensors.append(torch.from_numpy(arr.transpose(2, 0, 1).copy()))
    return torch.stack(tensors)


class PixelNeRF(Backbone):
    """Pretrained pixelNeRF (sn64). Highest ceiling at many views, and the slowest."""

    name = "pixelnerf"
    cost = 1.0      # placeholder until measured; it is by far the dearest

    @staticmethod
    def _ckpt_path(cfg) -> str:
        return os.environ.get("PIXELNERF_CKPT", getattr(cfg, "pixelnerf_ckpt", ""))

    @classmethod
    def available(cls, cfg) -> bool:
        return os.path.isfile(cls._ckpt_path(cfg))

    def __init__(self, cfg, device: str = "cpu"):
        self.device = device
        self.net = PixelNeRFNet(SN64)
        state = torch.load(self._ckpt_path(cfg), map_location="cpu", weights_only=False)
        self.net.load_state_dict(state, strict=True)
        self.net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def predict(self, images, cams=None, scale=None) -> np.ndarray:
        if cams is None:
            raise ValueError(
                "pixelnerf needs per-view camera parameters; the environment must "
                "pass the object's `cams` for the selected views. Voxel backbones "
                "ignore this argument, which is why it is optional on the interface."
            )

        views = _preprocess(images).to(self.device)
        poses, focal = cams_to_poses(cams, scale)
        poses = poses.to(self.device)

        # c is left as None on purpose: pixelNeRF then uses the image centre,
        # which is what these renderings use. Passing a length-2 tensor here is
        # a trap -- encode() reads that as per-view cx for two views.
        self.net.encode(
            views.unsqueeze(0),
            poses.unsqueeze(0),
            torch.tensor(focal, dtype=torch.float32, device=self.device),
        )

        points = _grid_points(scale).to(self.device)
        cell = BASE_EXTENT * (WORLD_SCALE if scale is None else scale) / VOXEL_RES

        # Density is view-independent, but the MLP still takes a direction, so
        # feed a constant one rather than pretending each cell has a ray.
        sigmas = []
        for start in range(0, points.shape[0], POINT_CHUNK):
            chunk = points[start:start + POINT_CHUNK].unsqueeze(0)
            viewdirs = torch.zeros_like(chunk)
            viewdirs[..., 2] = -1.0
            out = self.net(chunk, coarse=True, viewdirs=viewdirs)
            sigmas.append(torch.relu(out[0, :, 3]))

        sigma = torch.cat(sigmas).reshape(VOXEL_RES, VOXEL_RES, VOXEL_RES)
        occupancy = 1.0 - torch.exp(-sigma * cell)

        out = _apply_axes(occupancy.cpu().numpy().astype(np.float32))
        assert out.shape == (VOXEL_RES,) * 3, f"unexpected output shape {out.shape}"
        return out
