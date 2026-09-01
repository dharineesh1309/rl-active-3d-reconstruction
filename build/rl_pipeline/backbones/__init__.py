"""
Reconstruction backbones the RL agent chooses between.

Every backbone exposes the same small surface, so ViewReconEnv can treat them
alike and the policy's model head just picks an index into `load_backbones()`:

    .name                   str    identifier used in logs and checkpoints
    .cost                   float  relative compute, scaled by Config.cost_lambda
    .predict(images, cams)  ->     (32, 32, 32) float32 occupancy probabilities

`images` is the list of PIL Images the agent selected this episode. Each backbone
applies its own input transform, because they were trained with different input
sizes and normalisation — that conversion is the backbone's business, not the
environment's. `cams` carries per-view camera parameters for backbones that need
pose (pixelNeRF); the voxel backbones ignore it.

Everything is scored at 32**3, which is the native resolution of the Choy et al.
2016 ShapeNetVox32 ground truth. Backbones that predict at other resolutions
resample to 32**3 themselves, so cross-backbone IoU is a like-for-like number.
Without that the model head would learn output resolution rather than
reconstruction quality.
"""

import numpy as np

VOXEL_RES = 32


class Backbone:
    """Interface for a reconstruction model. See module docstring."""

    name = "unnamed"
    cost = 1.0          # relative compute; tune against measured wall-clock

    def predict(self, images, cams=None) -> np.ndarray:
        raise NotImplementedError

    @classmethod
    def available(cls, cfg) -> bool:
        """True when this backbone's weights are present and it can be built."""
        raise NotImplementedError


def voxel_iou(pred_probs: np.ndarray, gt: np.ndarray,
              pred_thresh: float = 0.4, gt_thresh: float = 0.5) -> float:
    """
    Volumetric IoU between predicted occupancy probabilities and ground truth.

    Shared by the environment's reward and by backbones.bench, so the training
    signal and the published-number comparison are computed the same way. Both
    grids must already be the same resolution.
    """
    if pred_probs.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred_probs.shape} vs gt {gt.shape}")

    pred = (pred_probs > pred_thresh)
    truth = (gt > gt_thresh)
    union = np.logical_or(pred, truth).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(pred, truth).sum() / union)


def _candidates(include_unregistered: bool = False):
    """
    Registered backbone classes, in the order that defines action indices.

    `include_unregistered` appends the ones dropped from the action space but
    kept on disk (OccNet, TripoSR). Benchmarking needs them -- the case for
    dropping them rests on a 3-category measurement, and re-testing that on all
    13 is pointless if they cannot be loaded -- but training must never see
    them, or the model head grows two actions it can never profitably take. So
    they are available to `backbones.bench` and `category_bench` and to nothing
    else, and they go last so registered indices are unaffected either way.
    """
    from .pix2vox_f import Pix2VoxF
    from .umiformer import UMIFormer
    from .umiformer_plus import UMIFormerPlus

    # Three backbones, each of which demonstrably WINS somewhere. That is the
    # selection criterion: an action a reward-maximising policy can never choose
    # is not a weak action, it is an unreachable one, and it still costs an index
    # in the model head's output distribution.
    #
    #   Pix2Vox-F   cheap CNN + attention fusion -> voxel.  Wins car at 1 view
    #               (0.8848 vs UMIFormer 0.8762), and is cheap enough (0.51s vs
    #               1.28s) to win more of the low-budget regime as lambda rises.
    #   UMIFormer   transformer over multi-view tokens.  Wins 1-3 views.
    #   UMIFormer+  same architecture, many-view training regime.  Wins 5+ views.
    #
    # The UMIFormer / UMIFormer+ split is the important one: identical
    # architecture means identical cost, so the choice between them is **pure
    # quality and independent of cost_lambda**. It stays live even at lambda = 0,
    # which answers the objection that a cost-driven split only exists because a
    # reward parameter was tuned until it did. Measured on airplane, car and
    # chair, the crossover lands between 3 and 5 views in all three.
    #
    # Dropped after measurement (backbones/category_bench.py, 12 category-budget
    # cells over 3 categories): **occnet** and **triposr** won zero cells.
    # OccNet means 0.271 and TripoSR 0.153 against UMIFormer's 0.800. OccNet was
    # retained on the theory that it would do better on complex shapes and worse
    # on simple ones; it is instead uniformly second-to-last. TripoSR is
    # single-view by construction, so it is flat across budgets and cannot
    # benefit from the view planning this whole project is about. Both modules
    # are kept on disk, with their calibrations, in case the 13-category sweep
    # overturns this -- they are simply not registered.
    #
    # Also excluded, for different reasons (see each module's docstring):
    # pix2vox_a (dominated by UMIFormer at every budget), r2n2 (port does not
    # reproduce, scores 0.03), pixelnerf (camera convention unresolved).
    #
    # Order defines the model head's action indices, so append, never insert.
    # This list changed on 2026-09-23 (five entries to three), which invalidates
    # any model head trained before that date -- including ckpt_final.pt.
    registered = [Pix2VoxF, UMIFormer, UMIFormerPlus]
    if not include_unregistered:
        return registered

    # Imported only on request. TripoSR pulls omegaconf and transformers when
    # built, and neither it nor OccNet is needed for training, so a deployment
    # that ships only the three registered checkpoints should never have to
    # satisfy their dependencies to start.
    from .occnet import OccNet
    from .triposr import TripoSR
    return registered + [OccNet, TripoSR]


def available_backbone_names(cfg) -> list:
    """
    Names of the backbones whose weights are present, without building them.

    The trainer needs the count up front to size the policy's model head, but
    instantiating every backbone in the parent process just to count them would
    load hundreds of megabytes that the workers then load again.
    """
    return [c.name for c in _candidates() if c.available(cfg)]


def load_backbones(cfg, device: str = "cpu", include_unregistered: bool = False) -> list:
    """
    Instantiate every backbone whose weights are on disk.

    Order is stable and defined here, because it *is* the action index the
    policy's model head learns. Adding a backbone in the middle of this list
    would silently invalidate a trained model head, so append, never insert.

    A backbone with missing weights is skipped with a notice rather than
    raising, so the pipeline runs with whatever is actually available.
    """
    loaded = []
    for cls in _candidates(include_unregistered):
        if not cls.available(cfg):
            print(f"[backbones] {cls.name}: weights not found, skipping")
            continue
        loaded.append(cls(cfg, device=device))
        print(f"[backbones] {cls.name}: ready (cost={cls.cost})")

    if not loaded:
        raise RuntimeError(
            "No reconstruction backbone is available — every weights file is "
            "missing, so no reward can be computed. Check the *_CKPT paths in "
            "config.py or the corresponding environment variables."
        )
    return loaded
