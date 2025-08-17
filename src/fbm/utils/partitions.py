from pathlib import Path

def part_path(root: str, layer: str, league: str, season: int, week: int | None = None) -> Path:
    """
    Build a data path like:
    data/bronze/league=NFL/season=2025/week=1
    """
    base = Path(root) / layer / f"league={league}" / f"season={season}"
    return base / f"week={week}" if week is not None else base
