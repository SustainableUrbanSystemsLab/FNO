"""
Shared config loader for all FNO training pipelines.

Loading order:
  1. config.toml        — shared training params, committed to repo
  2. config.local.toml  — user-specific overrides (paths, ICE credentials), gitignored

Local values win on conflict, so any key in config.local.toml
overwrites the same key in config.toml for that section.
"""
import os
import tomllib


def load_config(config_file="config.toml"):
    """Load config.toml and merge config.local.toml on top if it exists."""
    root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )

    # 1. Load base config
    base_path = os.path.join(root, "config.toml")
    config = {}
    if os.path.exists(base_path):
        with open(base_path, "rb") as f:
            config = tomllib.load(f)
    else:
        print("Warning: config.toml not found, using defaults")

    # 2. Merge config.local.toml on top (section by section); FNO_CONFIG_LOCAL
    #    points at a different overrides file so concurrent variant runs sharing
    #    one checkout do not have to overwrite each other's copy
    local_path = os.environ.get("FNO_CONFIG_LOCAL") or os.path.join(root, "config.local.toml")
    if os.path.exists(local_path):
        with open(local_path, "rb") as f:
            local = tomllib.load(f)
        for section, values in local.items():
            if section in config and isinstance(config[section], dict):
                config[section].update(values)
            else:
                config[section] = values
        print(f"Loaded local overrides from {local_path}")

    return config
