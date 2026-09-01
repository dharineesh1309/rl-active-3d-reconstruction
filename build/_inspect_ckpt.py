# -*- coding: utf-8 -*-
"""inspect_ckpt.py — dump Pix2Vox-F checkpoint architecture (read-only)."""
import torch

P = r"D:\rl_project\build\Pix2Vox-F-ShapeNet.pth"


def heads(sd, name):
    print("### %s (%d tensors)" % (name, len(sd)))
    for k, v in sd.items():
        if k.endswith(".weight") and v.dim() >= 2:
            print("   %-40s %s" % (k, tuple(v.shape)))
        elif k.endswith(".weight") and v.dim() == 1:
            pass
    print()


def main():
    ck = torch.load(P, map_location="cpu", weights_only=False)
    print("meta:", ck.get("best_iou"), ck.get("best_epoch"), "epoch", ck.get("epoch_idx"))
    heads(ck["encoder_state_dict"], "ENCODER")
    heads(ck["decoder_state_dict"], "DECODER")
    heads(ck["merger_state_dict"], "MERGER")


if __name__ == "__main__":
    main()
