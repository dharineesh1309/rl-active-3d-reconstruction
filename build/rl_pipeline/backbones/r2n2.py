"""
3D-R2N2 backbone (chrischoy/3D-R2N2, ResidualGRUNet).

STATUS: NOT REGISTERED — PORT DOES NOT REPRODUCE
─────────────────────────────────────────────────
The weight mapping is provably correct: all 63 arrays consumed, every shape
matched, and the module's parameter count equals the checkpoint's exactly
(35,968,706). But benchmarked on test-split chairs it scores **IoU 0.031**
against a published 0.466 at one view. Near-zero means the forward pass is
wrong, not the weights — most likely the pooling semantics, the residual
wiring, or the LeakyReLU slope.

It is therefore left out of the registry in __init__.py. A backbone scoring
0.03 inside the action space would not fail loudly; it would quietly teach the
policy that one of its options is worthless, and corrupt the reward while
looking like ordinary training noise.

Fixing it means bisecting the forward pass against the Theano reference, which
is open-ended work. UMIFormer was chosen instead precisely because it is
PyTorch-native and needs no reimplementation. Keep this module for reference,
or debug it layer by layer if a recurrent backbone is wanted later.

A PyTorch port of the original Theano/Lasagne model, loading the released
`ResidualGRUNet.npy` weights directly. The upstream repo is Theano on Python
3.6, which cannot coexist with a modern torch install, so the network is
reimplemented here and the parameter dump mapped onto it.

Why it earns a slot: this is the only *recurrent* reconstructor in the set. It
fuses views by running a 3D convolutional GRU over them one at a time, rather
than pooling or attending over per-view features in one shot. Different
inductive bias, different failure modes, and it is cheap.

Weight mapping
──────────────
`ResidualGRUNet.npy` is a flat list of 63 arrays in Theano's parameter-creation
order, which the upstream `network_definition` fixes exactly:

    [0:32]   2D encoder    conv1a,1b, 2a,2b,2c, 3a,3b,3c, 4a,4b, 5a,5b,5c, 6a,6b, fc7
    [32:41]  3D GRU        update, reset, candidate  (each: Wh conv, Wx fc, bias)
    [41:63]  3D decoder    conv7a..conv11

Layout differs too. Theano keeps 3D tensors as (B, D, C, H, W) and 3D conv
weights as (out, kD, in, kH, kW); torch wants (B, C, D, H, W) and
(out, in, kD, kH, kW), so every 3D kernel is transposed on load. Getting this
wrong does not crash — it produces a plausible-looking volume that scores near
zero, which is what backbones.bench is for.

Geometry
────────
Input is 127x127 (the renderings are 137x137, centre-cropped as upstream does).
Pooling is `pool_2d(ds=2, ignore_border=True, padding=1)`, i.e. pad one then
floor-pool, so 127 -> 64 -> 33 -> 17 -> 9 -> 5 -> 3, and the flattened encoder
output is 256*3*3 = 2304, matching the fc7 weight exactly. That chain is the
check that the input size is right.

Output is a 2-channel 32**3 logit volume; channel 1 after softmax is occupancy.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from . import VOXEL_RES, Backbone

IMG_SIZE = 127
LEAK = 0.01             # lib/layers.py LeakyReLU default
GRU_VOX = 4             # hidden state is 128 x 4 x 4 x 4


def _pool(x):
    """Theano's pool_2d(ds=2, ignore_border=True, padding=1)."""
    return F.max_pool2d(x, kernel_size=2, stride=2, padding=1)


def _unpool3d(x):
    """
    Zero-insertion upsampling by 2 along D, H, W.

    Upstream writes the input into every other cell of a zero tensor. This is
    NOT nearest-neighbour: the interleaved cells stay zero, and the following
    convolutions are what fill them in.
    """
    b, c, d, h, w = x.shape
    out = x.new_zeros(b, c, d * 2, h * 2, w * 2)
    out[:, :, ::2, ::2, ::2] = x
    return out


class ResidualGRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        nc = [96, 128, 256, 256, 256, 256]
        nd = [128, 128, 128, 64, 32, 2]

        # ── 2D encoder ────────────────────────────────────────────────────────
        self.conv1a = nn.Conv2d(3, nc[0], 7, padding=3)
        self.conv1b = nn.Conv2d(nc[0], nc[0], 3, padding=1)
        self.conv2a = nn.Conv2d(nc[0], nc[1], 3, padding=1)
        self.conv2b = nn.Conv2d(nc[1], nc[1], 3, padding=1)
        self.conv2c = nn.Conv2d(nc[0], nc[1], 1)
        self.conv3a = nn.Conv2d(nc[1], nc[2], 3, padding=1)
        self.conv3b = nn.Conv2d(nc[2], nc[2], 3, padding=1)
        self.conv3c = nn.Conv2d(nc[1], nc[2], 1)
        self.conv4a = nn.Conv2d(nc[2], nc[3], 3, padding=1)
        self.conv4b = nn.Conv2d(nc[3], nc[3], 3, padding=1)
        self.conv5a = nn.Conv2d(nc[3], nc[4], 3, padding=1)
        self.conv5b = nn.Conv2d(nc[4], nc[4], 3, padding=1)
        self.conv5c = nn.Conv2d(nc[3], nc[4], 1)
        self.conv6a = nn.Conv2d(nc[4], nc[5], 3, padding=1)
        self.conv6b = nn.Conv2d(nc[5], nc[5], 3, padding=1)
        self.fc7 = nn.Linear(2304, 1024)

        # ── 3D convolutional GRU: update, reset, candidate ───────────────────
        for gate in ("update", "reset", "cand"):
            setattr(self, f"{gate}_conv", nn.Conv3d(nd[0], nd[0], 3, padding=1, bias=False))
            setattr(self, f"{gate}_fc", nn.Linear(1024, nd[0] * GRU_VOX ** 3, bias=False))
            setattr(self, f"{gate}_bias", nn.Parameter(torch.zeros(nd[0])))

        # ── 3D decoder ───────────────────────────────────────────────────────
        self.conv7a = nn.Conv3d(nd[0], nd[1], 3, padding=1)
        self.conv7b = nn.Conv3d(nd[1], nd[1], 3, padding=1)
        self.conv8a = nn.Conv3d(nd[1], nd[2], 3, padding=1)
        self.conv8b = nn.Conv3d(nd[2], nd[2], 3, padding=1)
        self.conv9a = nn.Conv3d(nd[2], nd[3], 3, padding=1)
        self.conv9b = nn.Conv3d(nd[3], nd[3], 3, padding=1)
        self.conv9c = nn.Conv3d(nd[2], nd[3], 1)
        self.conv10a = nn.Conv3d(nd[3], nd[4], 3, padding=1)
        self.conv10b = nn.Conv3d(nd[4], nd[4], 3, padding=1)
        self.conv10c = nn.Conv3d(nd[4], nd[4], 3, padding=1)
        self.conv11 = nn.Conv3d(nd[4], nd[5], 3, padding=1)

    # ── per-view encoder ──────────────────────────────────────────────────────

    def encode(self, x):
        a = lambda t: F.leaky_relu(t, LEAK)

        x = _pool(a(self.conv1b(a(self.conv1a(x)))))

        res = self.conv2c(x)
        x = _pool(res + a(self.conv2b(a(self.conv2a(x)))))

        res = self.conv3c(x)
        x = _pool(a(self.conv3b(a(self.conv3a(x)))) + res)

        x = _pool(a(self.conv4b(a(self.conv4a(x)))))

        res = self.conv5c(x)
        x = _pool(res + a(self.conv5b(a(self.conv5a(x)))))

        x = _pool(x + a(self.conv6b(a(self.conv6a(x)))))

        return a(self.fc7(x.flatten(1)))

    # ── GRU over views ────────────────────────────────────────────────────────

    def _gate(self, name, hidden, feat):
        conv = getattr(self, f"{name}_conv")(hidden)
        fc = getattr(self, f"{name}_fc")(feat)
        fc = fc.view(-1, GRU_VOX, 128, GRU_VOX, GRU_VOX)      # Theano (D,C,H,W)
        fc = fc.permute(0, 2, 1, 3, 4)                        # -> torch (C,D,H,W)
        bias = getattr(self, f"{name}_bias").view(1, -1, 1, 1, 1)
        return conv + fc + bias

    def forward(self, views):
        """views: (V, 3, 127, 127) -> (2, 32, 32, 32) logits."""
        hidden = views.new_zeros(1, 128, GRU_VOX, GRU_VOX, GRU_VOX)

        for v in range(views.shape[0]):
            feat = self.encode(views[v:v + 1])
            update = torch.sigmoid(self._gate("update", hidden, feat))
            reset = torch.sigmoid(self._gate("reset", hidden, feat))
            cand = torch.tanh(self._gate("cand", reset * hidden, feat))
            hidden = update * hidden + (1.0 - update) * cand

        a = lambda t: F.leaky_relu(t, LEAK)

        x = _unpool3d(hidden)                                  # 4 -> 8
        x = x + a(self.conv7b(a(self.conv7a(x))))

        x = _unpool3d(x)                                       # 8 -> 16
        x = x + a(self.conv8b(a(self.conv8a(x))))

        x = _unpool3d(x)                                       # 16 -> 32
        skip = self.conv9c(x)
        x = skip + a(self.conv9b(a(self.conv9a(x))))

        mid = a(self.conv10a(x))
        x = self.conv10c(mid) + a(self.conv10b(mid))

        return self.conv11(x).squeeze(0)


# ── Weight loading ────────────────────────────────────────────────────────────

# Parameter order is fixed by the upstream network_definition; see module docstring.
_ENCODER_2D = ["conv1a", "conv1b", "conv2a", "conv2b", "conv2c",
               "conv3a", "conv3b", "conv3c", "conv4a", "conv4b",
               "conv5a", "conv5b", "conv5c", "conv6a", "conv6b"]
_DECODER_3D = ["conv7a", "conv7b", "conv8a", "conv8b",
               "conv9a", "conv9b", "conv9c",
               "conv10a", "conv10b", "conv10c", "conv11"]


def load_theano_weights(model: nn.Module, path: str):
    """Map the 63-array Theano dump onto the torch module, checking every shape."""
    arrays = [np.asarray(a) for a in np.load(path, allow_pickle=True)]
    if len(arrays) != 63:
        raise ValueError(f"expected 63 parameter arrays, got {len(arrays)}")

    def assign(param, value, what):
        t = torch.from_numpy(np.ascontiguousarray(value)).float()
        if tuple(param.shape) != tuple(t.shape):
            raise ValueError(f"{what}: module wants {tuple(param.shape)}, "
                             f"checkpoint has {tuple(t.shape)}")
        param.data.copy_(t)

    i = 0
    for name in _ENCODER_2D:                       # 2D convs already match layout
        layer = getattr(model, name)
        assign(layer.weight, arrays[i], f"{name}.weight"); i += 1
        assign(layer.bias, arrays[i], f"{name}.bias"); i += 1

    assign(model.fc7.weight, arrays[i].T, "fc7.weight"); i += 1   # Theano is (in, out)
    assign(model.fc7.bias, arrays[i], "fc7.bias"); i += 1

    for gate in ("update", "reset", "cand"):
        # Theano 3D kernel (out, kD, in, kH, kW) -> torch (out, in, kD, kH, kW)
        assign(getattr(model, f"{gate}_conv").weight,
               arrays[i].transpose(0, 2, 1, 3, 4), f"{gate}_conv"); i += 1
        assign(getattr(model, f"{gate}_fc").weight, arrays[i].T, f"{gate}_fc"); i += 1
        assign(getattr(model, f"{gate}_bias"), arrays[i], f"{gate}_bias"); i += 1

    for name in _DECODER_3D:
        layer = getattr(model, name)
        assign(layer.weight, arrays[i].transpose(0, 2, 1, 3, 4), f"{name}.weight"); i += 1
        assign(layer.bias, arrays[i], f"{name}.bias"); i += 1

    if i != 63:
        raise ValueError(f"consumed {i} of 63 arrays -- mapping is incomplete")


def _preprocess(images) -> torch.Tensor:
    """
    PIL images -> (V, 3, 127, 127) in [0, 1].

    Upstream composites the transparent background and crops 137 -> 127 rather
    than resizing, so the object keeps its rendered scale.
    """
    out = []
    for img in images:
        arr = np.asarray(img).astype(np.float32) / 255.0
        if arr.shape[2] == 4:                                   # composite onto white
            alpha = arr[:, :, 3:4]
            arr = arr[:, :, :3] * alpha + (1.0 - alpha)
        h, w = arr.shape[:2]
        if h >= IMG_SIZE and w >= IMG_SIZE:                      # centre crop
            top, left = (h - IMG_SIZE) // 2, (w - IMG_SIZE) // 2
            arr = arr[top:top + IMG_SIZE, left:left + IMG_SIZE]
        else:
            arr = np.asarray(Image.fromarray((arr * 255).astype(np.uint8))
                             .resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)) / 255.0
        out.append(torch.from_numpy(arr.transpose(2, 0, 1).copy()))
    return torch.stack(out)


class R2N2(Backbone):
    """Pretrained 3D-R2N2 ResidualGRUNet. Cheap, recurrent, flat across budgets."""

    name = "r2n2"
    cost = 0.09     # set from measurement once benchmarked; similar scale to pix2vox_f

    @staticmethod
    def _ckpt_path(cfg) -> str:
        return os.environ.get("R2N2_CKPT", getattr(cfg, "r2n2_ckpt", ""))

    @classmethod
    def available(cls, cfg) -> bool:
        return os.path.isfile(cls._ckpt_path(cfg))

    def __init__(self, cfg, device: str = "cpu"):
        self.device = device
        self.net = ResidualGRUNet()
        load_theano_weights(self.net, self._ckpt_path(cfg))
        self.net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def predict(self, images, cams=None) -> np.ndarray:
        views = _preprocess(images).to(self.device)
        logits = self.net(views)                        # (2, 32, 32, 32)
        probs = torch.softmax(logits, dim=0)[1]         # channel 1 is occupied
        out = probs.cpu().numpy().astype(np.float32)
        assert out.shape == (VOXEL_RES,) * 3, f"unexpected output shape {out.shape}"
        return out
