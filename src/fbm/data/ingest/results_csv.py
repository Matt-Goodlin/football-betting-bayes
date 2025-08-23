from pathlib import Path
from typing import List, Dict, Any
import re

# --- Header normalization helpers ------------------------------------------------

_ALIAS = {
    # canonical -> accepted variants
    "date": {"date", "game_date"},
    "home_team": {"home_team", "home", "hometeam", "home team"},
    "away_team": {"away_team", "away", "awayteam", "away team"},
    "home_pts": {"home_pts", "home_points", "home score", "home_score", "homepts"},
    "away_pts": {"away_pts", "away_points", "away score", "away_score", "awaypts"},
}

_NUMERIC_INT_FIELDS = {"home_pts", "away_pts"}

def _snake(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("/", " ").replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s

def _canonical(h: str) -> str:
    """
    Map various header spellings to our canonical names.
    Returns the normalized key if matched, else the snake-cased header.
    """
    s = _snake(h)
    for canon, variants in _ALIAS.items():
        if s == canon or s in { _snake(v) for v in variants }:
            return canon
    return s

def _coerce_value(key: str, val: str) -> Any:
    """
    Coerce specific fields to ints when possible; otherwise return original string.
    """
    if key in _NUMERIC_INT_FIELDS:
        v = val.strip()
        if v == "":
            return v
        try:
            return int(v)
        except ValueError:
            return v
    return val

# --- Loaders --------------------------------------------------------------------

def load_results_csv(path: Path) -> List[Dict[str, Any]]:
    """
    Load a single results CSV with a flexible header.
    Canonical keys returned when present:
      - date (str), home_team (str), away_team (str), home_pts (int|str), away_pts (int|str)
    Any extra columns are normalized to snake_case as well.
    """
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []

    raw_header = [h.strip() for h in lines[0].split(",")]
    header = [_canonical(h) for h in raw_header]

    rows: List[Dict[str, Any]] = []
    for ln in lines[1:]:
        if not ln.strip():
            continue
        parts = [p.strip() for p in ln.split(",")]
        # pad/truncate to header length
        if len(parts) < len(header):
            parts = parts + [""] * (len(header) - len(parts))
        elif len(parts) > len(header):
            parts = parts[:len(header)]
        # coerce known numeric fields
        coerced = {k: _coerce_value(k, v) for k, v in zip(header, parts)}
        rows.append(coerced)
    return rows

def load_results_dir(dir_path: Path) -> List[Dict[str, Any]]:
    """
    Load all *.csv files in a directory, concatenated and sorted by date ascending.
    Accepts flexible headers per load_results_csv. Missing dir returns [].
    """
    if not dir_path.exists():
        return []
    all_rows: List[Dict[str, Any]] = []
    for p in sorted(dir_path.glob("*.csv")):
        all_rows.extend(load_results_csv(p))
    # Sort by date string ascending (YYYY-MM-DD sorts lexicographically)
    all_rows.sort(key=lambda r: r.get("date", ""))
    return all_rows