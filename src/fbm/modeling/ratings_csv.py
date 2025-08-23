from pathlib import Path
from typing import Dict

def load_ratings_csv(path: Path) -> Dict[str, float]:
    """
    CSV schema: team,rating
    Returns {} if file doesn't exist.
    """
    ratings: Dict[str, float] = {}
    if not path.exists():
        return ratings
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lower().startswith("team,"):
            continue
        team, val = line.split(",", 1)
        ratings[team.strip()] = float(val.strip())
    return ratings
