from pathlib import Path

def ensure_dir(path: Path) -> None:
    """Create directory path if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)
