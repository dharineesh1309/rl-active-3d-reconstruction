"""
Checkpoint resume, on CUDA when available -- the case a CPU run cannot catch.

  1. Every RNG stream (torch CPU, CUDA, numpy, python, each env) continues
     exactly where it was saved.
  2. Policy weights come back, and the optimizer state sits on the
     parameters' device.
  3. A different experiment config is refused, naming what changed.
  4. A kept checkpoint missing from the output directory is refused.
"""

import os
import random
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from policy.pose_policy import RGBPosePolicy
from run_0b import restore_state, save_state


class Env:
    def __init__(self):
        self._sync = np.random.RandomState(7)

    def reset(self):
        return None


def stub(device):
    policy = RGBPosePolicy().to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    sum(p.sum() for p in policy.parameters()).backward()
    opt.step()
    return SimpleNamespace(policy=policy, opt=opt, total_steps=4096,
                           total_episodes=1024, obs=None)


def draws(envs, device):
    out = [torch.rand(3).tolist(), np.random.rand(3).tolist(), random.random(),
           [e._sync.randint(1000) for e in envs]]
    if device == "cuda":
        out.append(torch.rand(3, device="cuda").tolist())
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {"args": {"arm": "set_pose", "budget": 4}, "costs": {"pix2vox_f": 0.51}}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "last.pt")
        torch.manual_seed(1)
        tr, envs = stub(device), [Env(), Env()]
        open(os.path.join(d, "best_s1.pt"), "w").close()
        save_state(path, tr, envs, arm="set_pose", budget=4, config=config,
                   best=[[0.01, 1, "best_s1.pt"]], next_eval=8192, elapsed_h=1.5)
        expected = draws(envs, device)
        weights = {k: v.clone() for k, v in tr.policy.state_dict().items()}

        torch.manual_seed(99)                      # a fresh session
        np.random.seed(99)
        random.seed(99)
        tr2, envs2 = stub(device), [Env(), Env()]
        st = restore_state(path, tr2, envs2, config, d)
        assert draws(envs2, device) == expected, "an RNG stream did not resume"
        assert tr2.total_steps == 4096 and st["next_eval"] == 8192
        for k, v in tr2.policy.state_dict().items():
            assert torch.equal(v, weights[k]), f"weights differ at {k}"
        for s in tr2.opt.state.values():
            for v in s.values():
                if torch.is_tensor(v) and v.dim() > 0:
                    assert v.device.type == device, f"optimizer state on {v.device}"

        for bad, why in (({**config, "costs": {"pix2vox_f": 0.09}}, "config"),
                         (config, "missing checkpoint")):
            if why == "missing checkpoint":
                os.remove(os.path.join(d, "best_s1.pt"))
            try:
                restore_state(path, stub(device), [Env(), Env()], bad, d)
            except SystemExit as e:
                assert why != "config" or "costs.pix2vox_f" in str(e), str(e)
            else:
                raise AssertionError(f"resume accepted a {why}")
    print(f"Resume tests passed on {device}.")


if __name__ == "__main__":
    main()
