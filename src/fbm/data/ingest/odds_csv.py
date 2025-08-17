import csv
from pathlib import Path
from typing import Dict, List

def load_odds_csv(path: Path) -> List[Dict[str, str]]:
    """Load a simple moneyline CSV:
    columns: game_id,home_team,away_team,home_ml,away_ml
    """
    rows: List[Dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows
