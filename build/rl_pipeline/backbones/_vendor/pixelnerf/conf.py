"""
A minimal stand-in for the pyhocon config objects pixelNeRF's `from_conf`
constructors expect.

pixelNeRF reads its architecture from .conf files via pyhocon. Rather than add
that dependency to load one fixed architecture, this supplies the same accessor
surface over a plain nested dict, and SN64 below spells out the settings the
released sn64 checkpoint was trained with.

SN64 is the resolved result of conf/exp/sn64.conf, which includes
conf/default_mv.conf, which includes conf/default.conf. The values that
actually differ from the single-view default and matter here:

  * encoder.use_first_pool = False   (sn64.conf: avoid over-reducing 64px input)
  * mlp_*.n_blocks         = 5       (default_mv: wider MLP for multiview)
  * mlp_*.combine_layer    = 3       (default_mv: average across views at layer 3)

That combine_layer is what makes this model genuinely multi-view: features from
every input view are averaged partway through the MLP, so more views sharpen
the prediction instead of being ignored.
"""

_MISSING = object()


class Conf(dict):
    """Dict with pyhocon's get_string/get_int/get_float/get_bool accessors."""

    def _lookup(self, key, default, cast):
        if key in self:
            value = self[key]
            return value if cast is None else cast(value)
        if default is _MISSING:
            raise KeyError(f"required config key missing: {key}")
        return default

    def get_string(self, key, default=_MISSING):
        return self._lookup(key, default, str)

    def get_int(self, key, default=_MISSING):
        return self._lookup(key, default, int)

    def get_float(self, key, default=_MISSING):
        return self._lookup(key, default, float)

    def get_bool(self, key, default=_MISSING):
        return self._lookup(key, default, bool)

    def get_list(self, key, default=_MISSING):
        return self._lookup(key, default, list)

    def __getitem__(self, key):
        value = dict.__getitem__(self, key)
        return Conf(value) if isinstance(value, dict) else value


SN64 = Conf({
    "use_encoder": True,
    "use_global_encoder": False,
    "use_xyz": True,
    "canon_xyz": False,
    "use_code": True,
    "code": {"num_freqs": 6, "freq_factor": 1.5, "include_input": True},
    "use_viewdirs": True,
    "use_code_viewdirs": False,
    "normalize_z": True,
    "encoder": {
        "type": "spatial",
        "backbone": "resnet34",
        "pretrained": False,     # every weight comes from the checkpoint
        "num_layers": 4,
        "use_first_pool": False,
        "index_interp": "bilinear",
        "index_padding": "border",
        "upsample_interp": "bilinear",
        "feature_scale": 1.0,
        "use_first_pool_": None,
    },
    "mlp_coarse": {
        "type": "resnet",
        "n_blocks": 5,
        "d_hidden": 512,
        "combine_layer": 3,
        "combine_type": "average",
    },
    "mlp_fine": {
        "type": "resnet",
        "n_blocks": 5,
        "d_hidden": 512,
        "combine_layer": 3,
        "combine_type": "average",
    },
})
