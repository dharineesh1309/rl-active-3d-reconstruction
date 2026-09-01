"""
UMIFormer's architecture config, as a plain attribute-access object.

The repo reads these from an easydict in config.py. Reproduced here so the
vendored modules can be constructed without pulling in the repo's config
machinery. Values are the released defaults, which is what the published
ShapeNet checkpoint was trained with.

PRETRAINED is forced False: the checkpoint supplies every encoder weight, and
leaving it True makes the ViT download ImageNet weights it then overwrites.
"""


class Cfg(dict):
    """Attribute access over nested dicts, standing in for easydict."""

    def __getattr__(self, key):
        try:
            value = self[key]
        except KeyError:
            raise AttributeError(key) from None
        return Cfg(value) if isinstance(value, dict) else value


UMIFORMER = Cfg({
    "NETWORK": {
        "ENCODER": {
            "VIT_IVDB": {
                "MODEL_NAME": "vit_deit_base_distilled_patch16_224",
                "PRETRAINED": False,
                "USE_CLS_TOKEN": False,
                # 0 = intra-image block, 1 = inter-image block; 16 blocks total
                "BLOCK_TYPES_LIST": [0, 0, 0, 1] * 4,
                "TYPE": 1,
                "K": 5,
            },
        },
        "DECODER": {
            "VOXEL_SIZE": 32,
            "RETR": {"DEPTH": 8, "HEADS": 12, "DIM": 768},
        },
        "MERGER": {
            "WITHOUT_PARAMETERS": False,
            "STM": {"DIM": 768, "OUT_TOKEN_LENS": [196, 196], "K": 15, "NUM_HEAD": 12},
        },
    },
    "CONST": {"IMG_W": 224, "IMG_H": 224, "N_VOX": 32},
})
