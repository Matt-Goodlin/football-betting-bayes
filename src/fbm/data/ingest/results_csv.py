from pathlib import Path
from typing import List, Dict, Any
import csv

def load_results_csv(path: Path) -> List[Dict[str, Any]]:
    """
    Load historical game results.
    Expected columns (case/space-insensitive):
      date, home_team, away_team, home_pts, away_pts
    Returns a list of dicts with normalized keys and int scores.
    Non-parsable rows are skipped.
    """
    rows: List[Dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for rec in reader:
            # normalize keys: lower-case, underscores
            norm = {
                (k.strip().lower().replace(" ", "_") if k is not None else ""):
                (v.strip() if v is not None else "")
                for k, v in rec.items()
            }
            # require score fields
            if "home_pts" not in norm or "away_pts" not in norm:
                continue
            try:
                norm["home_pts"] = int(norm["home_pts"])
                norm["away_pts"] = int(norm["away_pts"])
            except ValueError:
                # skip rows with non-integer scores
                continue
            rows.append(norm)
    return rows
