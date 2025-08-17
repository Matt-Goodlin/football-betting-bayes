import yaml
from pathlib import Path

def load_config(path: str = "conf/default.yaml") -> dict:
    """
    Load YAML config into a dict.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)
