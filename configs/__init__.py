"""Shared configuration loader used by all scripts in this repository."""

import os
import re
from pathlib import Path

import yaml


def load_config(config_path: str | None = None) -> dict:
    """Load and resolve configs/paths.yaml.

    Args:
        config_path: Path to paths.yaml. Defaults to configs/paths.yaml
                     relative to this file's parent directory.

    Returns:
        Fully resolved configuration dictionary with all ${root} tokens
        replaced by the actual repository root path.
    """
    if config_path is None:
        config_path = Path(__file__).parent / "paths.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Determine repository root
    root = os.environ.get("PANCREAS_CYST_ROOT")
    if not root:
        # Default: parent of configs/
        root = str(Path(__file__).parent.parent.resolve())

    def _resolve(obj, root_val):
        if isinstance(obj, str):
            obj = obj.replace("${PANCREAS_CYST_ROOT}", root_val)
            obj = obj.replace("${root}", root_val)
            # Also handle any remaining env vars via os.path.expandvars
            obj = os.path.expandvars(obj)
            return obj
        if isinstance(obj, dict):
            return {k: _resolve(v, root_val) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(item, root_val) for item in obj]
        return obj

    cfg = _resolve(cfg, root)
    cfg["root"] = root
    return cfg
