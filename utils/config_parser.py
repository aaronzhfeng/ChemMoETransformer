"""
utils/config_parser.py
----------------------

Tiny helper around pyyaml that converts a YAML experiment file
into a dot-accessible nested object.

Example
-------
>>> from utils.config_parser import load_config
>>> cfg = load_config("config/example_moe_config.yaml")
>>> print(cfg.model.moe_layers)
[2, 4, 6]
"""
from types import SimpleNamespace
from pathlib import Path
import yaml


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert dict -> SimpleNamespace so we can
    write cfg.model.d_model instead of cfg['model']['d_model']."""
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = _dict_to_namespace(v)
        else:
            out[k] = v
    return SimpleNamespace(**out)


def load_config(path: str) -> SimpleNamespace:
    """Load YAML file *path* and return a SimpleNamespace tree."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf‑8") as f:
        raw_cfg = yaml.safe_load(f)

    cfg = _dict_to_namespace(raw_cfg)
    cfg._cfg_path = str(path)  # handy for logging
    return cfg
