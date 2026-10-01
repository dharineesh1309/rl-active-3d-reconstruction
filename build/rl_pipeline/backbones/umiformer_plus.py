"""
UMIFormer+ backbone (GaryZhu1996/UMIFormer, the "+" checkpoint).

Same architecture as UMIFormer, trained under a many-view regime. That training
difference is the whole reason it is here: it buys accuracy at large view counts
by giving some up at small ones, so the two cross over.

Published ShapeNet mean IoU:

    views        1       2       3       4       5       8      12      20
    UMIFormer   .680    .738    .752    .757    .761    .766    .768    .770
    UMIFormer+  .567    .712    .745    .759    .768    .779    .784    .789
                 ^--- UMIFormer ahead ---^   ^--- UMIFormer+ ahead --------^

The crossover sits at four views, inside the {3,5,8} training budget range, and
because the two share an architecture they also share a compute cost. So the
choice between them is **pure quality and independent of cost_lambda** -- it
stays live even at lambda = 0. That matters: it is a decision the policy can
learn without the reward being tuned until an interesting answer appears, which
is the criticism that applies to a cost-driven split.

Implementation is a subclass rather than a copy: identical network, identical
preprocessing, only the weights and the cost differ.
"""

import os

from .umiformer import UMIFormer


class UMIFormerPlus(UMIFormer):
    """Pretrained UMIFormer+. Better with many views, worse with few."""

    name = "umiformer_plus"
    # Same architecture as UMIFormer, so the same compute: UMIFormer's measured
    # 1.28 s per 5-view prediction (config.py, cost v1).
    cost = 1.28

    @staticmethod
    def _ckpt_path(cfg) -> str:
        return os.environ.get("UMIFORMER_PLUS_CKPT",
                              getattr(cfg, "umiformer_plus_ckpt", ""))
