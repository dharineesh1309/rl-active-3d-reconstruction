"""
The utility cache must not corrupt labels or the policy's randomness.

  1. A cache miss does not advance the global torch RNG, even though a backbone
     (like UMIFormer) draws torch.rand inside predict().
  2. Utilities are IoU - lambda*cost computed at read time, and survive a
     flush/reload.
  3. A cache written at other thresholds is refused.
"""

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from training.utility_envelope import UtilityEnvelope


class Fake:
    def __init__(self, name, cost, fill):
        self.name, self.cost, self.fill = name, cost, fill

    def predict(self, images, cams=None):
        torch.rand(10)                       # like UMIFormer's tie-break noise
        g = np.zeros((32, 32, 32), np.float32)
        g[:self.fill] = 1.0
        return g


def main():
    bbs = [Fake("pix2vox_f", 0.51, 16), Fake("umiformer", 1.28, 8)]
    gt = np.zeros((32, 32, 32), np.float32)
    gt[:16] = 1.0
    item = {"model_id": "obj", "images": [None] * 24, "cams": [None] * 24,
            "voxels": gt}

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.json")
        env = UtilityEnvelope(bbs, cost_lambda=0.0771, cache_path=path,
                              run={"script": "test"})

        torch.manual_seed(123)
        expected = torch.rand(3)
        torch.manual_seed(123)
        env(item, [3, 1, 7])                 # miss: runs both fakes
        assert torch.equal(torch.rand(3), expected), "miss advanced the policy RNG"

        u = env.utilities(item, [1, 7, 3])   # hit, any order
        assert env.hits == 1 and env.misses == 1
        assert abs(u["pix2vox_f"] - (1.0 - 0.0771 * 0.51)) < 1e-9
        assert abs(u["umiformer"] - (0.5 - 0.0771 * 1.28)) < 1e-9
        env.flush()

        again = UtilityEnvelope(bbs, cost_lambda=0.05, cache_path=path)
        assert again.utilities(item, [7, 3, 1])["umiformer"] == 0.5 - 0.05 * 1.28
        assert again.misses == 0, "reload lost the cache"

        try:
            UtilityEnvelope(bbs, cost_lambda=0.0771, cache_path=path,
                            thresholds={"pix2vox_f": 0.3})
        except ValueError:
            pass
        else:
            raise AssertionError("cache at other thresholds was accepted")
    print("Utility envelope tests passed.")


if __name__ == "__main__":
    main()
