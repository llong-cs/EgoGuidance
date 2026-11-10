def get_vlm_model(config):
    """Build the VLM wrapper selected by ``config.framework.qwenvl.base_vlm``.

    Only Qwen3.5 checkpoints are supported; the backbone is recognised by the substring
    ``Qwen3.5`` in the checkpoint path / hub id (e.g. ``playground/Pretrained_models/Qwen3.5-2B``).
    """
    vlm_name = str(config.framework.qwenvl.base_vlm)

    if "Qwen3.5" in vlm_name:
        from .QWen3_5 import _QWen3_5_VL_Interface

        return _QWen3_5_VL_Interface(config)
    raise NotImplementedError(
        f"Unsupported framework.qwenvl.base_vlm={vlm_name!r}: only Qwen3.5 checkpoints are supported, and the "
        "path / hub id must contain 'Qwen3.5' (e.g. playground/Pretrained_models/Qwen3.5-2B). If your local "
        "copy is stored under a different directory name, rename it or symlink it to a name containing 'Qwen3.5'."
    )
