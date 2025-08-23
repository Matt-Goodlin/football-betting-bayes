import csv
from pathlib import Path
from typing import List, Dict

def load_odds_csv(path: Path) -> List[Dict[str, str]]:
    """
    Load odds CSV into list of dicts with normalized lowercase keys.
    """
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            # normalize column names: lower + underscores
            normed = {k.strip().lower().replace(" ", "_"): v.strip() for k, v in row.items()}
            rows.append(normed)
        return rows