"""
The utility cache must not corrupt labels or the policy's randomness.

  1. A cache miss does not advance the global torch RNG, even though a backbone
     (like UMIFormer) draws torch.rand inside predict().
  2. Utilities are IoU - lambda*cost computed at read time, and survive a
     flush/reload; each new entry records what generated it.
  3. A cache written at other thresholds, or by other checkpoints, is refused.
  4. Converted legacy labels stay "unknown" until a recomputation reproduces
     them, and are refused if it does not.
"""

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from training.utility_envelope import UtilityEnvelope, convert_legacy

CK = {"pix2vox_f": "sha256:aa", "umiformer": "sha256:bb"}


class Fake:
    def __init__(self, name, cost, fill):
        self.name, self.cost, self.fill = name, cost, fill

    def predict(self, images, cams=None):
        torch.rand(10)                       # like UMIFormer's tie-break noise
        g = np.zeros((32, 32, 32), np.float32)
        g[:self.fill] = 1.0
        return g


def refused(**kw):
    try:
        UtilityEnvelope(**kw)
    except ValueError:
        return True
    return False


def main():
    bbs = [Fake("pix2vox_f", 0.51, 16), Fake("umiformer", 1.28, 8)]
    gt = np.zeros((32, 32, 32), np.float32)
    gt[:16] = 1.0
    item = {"model_id": "obj", "images": [None] * 24, "cams": [None] * 24,
            "voxels": gt}

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.json")
        env = UtilityEnvelope(bbs, cost_lambda=0.0771, cache_path=path,
                              run={"script": "test"}, checkpoints=CK)

        torch.manual_seed(123)
        expected = torch.rand(3)
        torch.manual_seed(123)
        env.tag = {"kind": "train", "ref": 512}
        env(item, [3, 1, 7])                 # miss: runs both fakes
        assert torch.equal(torch.rand(3), expected), "miss advanced the policy RNG"
        assert env.cache["src"]["obj|1,3,7"] == [env.run_id, "train", 512]

        u = env.utilities(item, [1, 7, 3])   # hit, any order
        assert env.hits == 1 and env.misses == 1
        assert abs(u["pix2vox_f"] - (1.0 - 0.0771 * 0.51)) < 1e-9
        assert abs(u["umiformer"] - (0.5 - 0.0771 * 1.28)) < 1e-9
        env.flush()

        again = UtilityEnvelope(bbs, cost_lambda=0.05, cache_path=path, checkpoints=CK)
        assert again.utilities(item, [7, 3, 1])["umiformer"] == 0.5 - 0.05 * 1.28
        assert again.misses == 0, "reload lost the cache"

        assert refused(backbones=bbs, cost_lambda=0.0771, cache_path=path,
                       thresholds={"pix2vox_f": 0.3}), "other thresholds accepted"
        assert refused(backbones=bbs, cost_lambda=0.0771, cache_path=path,
                       checkpoints={**CK, "umiformer": "sha256:cc"}), \
            "other checkpoints accepted"

        # Legacy: utilities written at lambda 0.07, old costs; true IoUs 1.0/0.5.
        old_cost = {"pix2vox_f": 0.09, "umiformer": 0.938}
        for corrupt, ok in ((0.0, True), (0.01, False)):
            leg = os.path.join(d, "legacy.json")
            json.dump({"obj|0,2,4": {"pix2vox_f": 1.0 - 0.07 * 0.09,
                                     "umiformer": 0.5 + corrupt - 0.07 * 0.938}},
                      open(leg, "w"))
            conv = os.path.join(d, f"conv{ok}.json")
            convert_legacy(leg, conv, 0.07, old_cost,
                           {"pix2vox_f": 0.4, "umiformer": 0.4}, "test")
            e = UtilityEnvelope(bbs, cost_lambda=0.0771, cache_path=conv, checkpoints=CK)
            assert e.cache["meta"]["runs"]["legacy"]["checkpoints"] == "unknown"
            try:
                e.verify_legacy(lambda mid: item, {"obj"})
                passed = True
            except ValueError:
                passed = False
            assert passed == ok, f"verify_legacy with corruption {corrupt}: {passed}"
            if ok:
                assert e.cache["meta"]["runs"]["legacy"]["checkpoints"] == CK
    print("Utility envelope tests passed.")


if __name__ == "__main__":
    main()
