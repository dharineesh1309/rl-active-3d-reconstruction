import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ── 1. Coverage Grid (32³ voxel occupancy) ──────────────────────────────────

class CoverageGrid:
    def __init__(self):
        self.grid = np.zeros((32, 32, 32), dtype=np.float32)

    def reset(self):
        self.grid = np.zeros((32, 32, 32), dtype=np.float32)

    def update(self, depth_map, silhouette, cam=None):
        """
        Back-project one view's depth/silhouette into the shared 32**3 grid.

        depth_map   : (H, W) float array, normalised 0-1
        silhouette  : (H, W) binary array, 1 = object present
        cam         : dict with 'azimuth'/'elevation' in degrees, or None

        **`cam` is what makes this a projection rather than a stencil.** Without
        it every view writes to grid[x, y, z] in its own image plane, so a view
        from the front and a view from directly behind land on the same voxels
        and the grid cannot express which surface each one actually revealed.
        That defeats the whole point of the coverage branch: the policy is meant
        to learn to prefer complementary viewpoints, and it can only do that if
        complementary viewpoints produce visibly different states.

        With `cam`, each pixel is lifted to a camera-frame point and rotated
        into a canonical world frame shared by all views, so coverage
        accumulates the way the report describes.

        `cam=None` falls back to the old image-plane behaviour, which keeps the
        ModelNet loader (no camera metadata) working; it is strictly worse and
        should not be used for ShapeNet.
        """
        depth_resized = np.array(
            Image.fromarray(depth_map).resize((32, 32), Image.BILINEAR)
        )
        sil_resized = np.array(
            Image.fromarray(silhouette.astype(np.float32)).resize((32, 32), Image.BILINEAR)
        )
        mask = sil_resized > 0.5
        if not mask.any():
            return
        z_indices = np.clip((depth_resized * 31).astype(int), 0, 31)
        xs, ys = np.where(mask)

        if cam is None:
            self.grid[xs, ys, z_indices[xs, ys]] = 1.0
            return

        # Exactly invert the transform that produced the depth map, which is
        # `_depth_from_voxels`: it computes cam = (Rx @ Ry) @ world, takes
        # px from cam_x, py from cam_y, depth from cam_z, and stores
        # depth_map[py, px]. So a hand-rolled camera basis will not do -- the
        # inverse has to use the same two rotations, in the same order, with
        # azimuth in the XZ plane. Getting that wrong scatters points off the
        # object and the grid is no better than noise.
        #
        # np.where gives (row, col) = (py, px), so rows carry cam_y and columns
        # carry cam_x. That transpose is easy to miss and silently mirrors the
        # reconstruction.
        cy = 2.0 * (xs / 31.0) - 1.0
        cx = 2.0 * (ys / 31.0) - 1.0
        cz = 2.0 * (z_indices[xs, ys] / 31.0) - 1.0

        az = np.deg2rad(cam["azimuth"])
        el = np.deg2rad(cam["elevation"])
        Ry = np.array([[np.cos(az), 0.0, np.sin(az)],
                       [0.0,        1.0, 0.0       ],
                       [-np.sin(az), 0.0, np.cos(az)]])
        Rx = np.array([[1.0, 0.0,         0.0        ],
                       [0.0, np.cos(el), -np.sin(el)],
                       [0.0, np.sin(el),  np.cos(el)]])
        world = np.stack([cx, cy, cz], axis=1) @ (Rx @ Ry)   # == (R.T @ v).T

        idx = np.clip(((world + 1.0) / 2.0 * 31).astype(int), 0, 31)
        # Voxels are indexed [z, y, x] to match `_depth_from_voxels`.
        self.grid[idx[:, 2], idx[:, 1], idx[:, 0]] = 1.0

    def get(self):
        return self.grid.copy()


# ── 2. Image Feature Extractor (frozen ResNet-50) ───────────────────────────

class ImageFeatureExtractor:
    def __init__(self, device: str = "cpu"):
        # Runs once per view step, so on a GPU box this is worth moving off the
        # CPU along with the backbones -- otherwise it becomes the bottleneck
        # when the reconstruction itself is fast.
        self.device = device
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.model = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        self.features = []

    def reset(self):
        self.features = []

    def add_view(self, image):
        """image: PIL Image or (H,W,3) numpy array"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image.astype(np.uint8))
        # ShapeNet renderings are RGBA; the dataloader keeps the alpha channel
        # because the reconstruction backbones composite it themselves. ResNet-50
        # and its ImageNet normalisation need exactly 3 channels, so drop it here
        # rather than forcing the dataloader to choose one consumer's format.
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor).squeeze().cpu().numpy()  # (2048,)
        feat_512 = feat.reshape(4, 512).mean(axis=0)
        self.features.append(feat_512)

    def get(self):
        if len(self.features) == 0:
            return np.zeros(512, dtype=np.float32)
        return np.mean(self.features, axis=0).astype(np.float32)


# ── 3. View History Mask (24-dim binary vector) ──────────────────────────────

class ViewHistoryMask:
    def __init__(self, n_views=24):          # ← 36 → 24
        self.n_views = n_views
        self.mask = np.zeros(n_views, dtype=np.float32)

    def reset(self):
        self.mask = np.zeros(self.n_views, dtype=np.float32)

    def mark(self, view_index):
        self.mask[view_index] = 1.0

    def get(self):
        return self.mask.copy()

    def visited(self):
        return set(np.where(self.mask == 1.0)[0].tolist())